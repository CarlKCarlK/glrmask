//! Bounded flat-stack execution and recycled frontier storage.

use super::admission::ActionableTerminals;
use super::admission::runtime_future_contains_ignore;
use super::lexical::SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX;
use super::lexical::advance_terminals_disallowed_over_bytes;
use super::lexical::collect_unique_actionable_matches;
use super::tokenizer_scan;
use super::tokenizer_scan::execute_tokenizer_reusable;
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::TokenizerMatch;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::table::Action;
use crate::compiler::glr::table::AdmissionPolicy;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::INLINE_PARSER_STATE_CAPACITY;
use crate::runtime::state::LINEAR_STACK_RESERVE;
use crate::runtime::state::ParserStateMap;
use smallvec::SmallVec;

pub(super) const FLAT_FRONTIER_MAX_BRANCHES: usize = 128;

pub(super) const FLAT_CONTINUATION_CACHE_CAPACITY: usize = 128;

pub(super) const FLAT_ACTION_MAX_BRANCHES: usize = 16;

pub(super) const FLAT_ACTION_MAX_STEPS: usize = 256;

pub(super) const FLAT_FRONTIER_PREALLOCATED_GSS: usize = FLAT_FRONTIER_GSS_POOL_CAPACITY;

pub(super) type FlatInlineStack = SmallVec<[u32; LINEAR_STACK_RESERVE]>;

#[derive(Debug, Default)]
pub(super) struct FlatActionScratch {
    pub(super) pending: SmallVec<[FlatInlineStack; FLAT_ACTION_MAX_BRANCHES]>,
    pub(super) complete: SmallVec<[FlatInlineStack; FLAT_ACTION_MAX_BRANCHES]>,
}

impl FlatActionScratch {
    pub(super) fn clear(&mut self) {
        self.pending.clear();
        self.complete.clear();
    }

    pub(super) fn push_pending(&mut self, stack: FlatInlineStack) -> bool {
        if self.pending.iter().any(|existing| existing == &stack) {
            return true;
        }
        if self.pending.len() == self.pending.capacity() {
            return false;
        }
        self.pending.push(stack);
        true
    }

    pub(super) fn push_complete(&mut self, stack: FlatInlineStack) -> bool {
        if self.complete.iter().any(|existing| existing == &stack) {
            return true;
        }
        if self.complete.len() == self.complete.capacity() {
            return false;
        }
        self.complete.push(stack);
        true
    }
}

// Keep a bounded reserve large enough for repeated handoffs between persistent
// GSS states and the allocation-free flat frontier.  One object reserves a
// 64-entry linear stack, so 256 spares cost about 65â€“75 KiB per active
// constraint state while avoiding allocator cliffs on multi-thousand-token
// JSON examples. The bound remains fixed; unsupported larger frontiers still
// fall back to the general persistent-GSS path.
pub(super) const FLAT_FRONTIER_GSS_POOL_CAPACITY: usize = 256;

pub(super) const FLAT_FRONTIER_RETIRED_GSS_CAPACITY: usize = 256;

#[derive(Debug)]
pub(super) struct FlatBranchScratch {
    pub(super) offset: usize,
    pub(super) tokenizer_state: u32,
    pub(super) stack: Vec<u32>,
    pub(super) acc: TerminalsDisallowed,
    pub(super) processed: bool,
    pub(super) initial_pruned: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct FlatContinuationDecision {
    pub(super) offset: usize,
    pub(super) tokenizer_state: u32,
    pub(super) end_state: u32,
    pub(super) viable: bool,
}

impl Default for FlatBranchScratch {
    fn default() -> Self {
        Self::with_stack_capacity(crate::runtime::state::LINEAR_STACK_RESERVE)
    }
}

#[derive(Debug)]
pub(crate) struct FlatFrontierScratch {
    pub(super) branches: [FlatBranchScratch; FLAT_FRONTIER_MAX_BRANCHES],
    pub(super) len: usize,
    pub(super) action: FlatActionScratch,
    pub(super) continuation_cache: [FlatContinuationDecision; FLAT_CONTINUATION_CACHE_CAPACITY],
    pub(super) continuation_cache_len: usize,
    // Double-buffered single-path GSS objects. Runtime commits write a small
    // output frontier into these preallocated objects and then swap them with
    // the active state. This permits branch creation/collapse without allocator
    // activity. Old active entries are recycled here after the atomic swap.
    pub(super) gss_pool: SmallVec<[ParserGSS; FLAT_FRONTIER_GSS_POOL_CAPACITY]>,
    // Non-segment single-path GSSs displaced by a successful flat commit are
    // retained until the state is dropped. This prevents their destructors from
    // running in the token hot path without allowing unbounded retention.
    pub(super) retired_gss: SmallVec<[ParserGSS; FLAT_FRONTIER_RETIRED_GSS_CAPACITY]>,
}

impl Default for FlatFrontierScratch {
    fn default() -> Self {
        Self::with_preallocated_gss(FLAT_FRONTIER_PREALLOCATED_GSS)
    }
}

impl FlatFrontierScratch {

    pub(crate) fn with_preallocated_gss(preallocated_gss: usize) -> Self {
        Self::with_reservation_policy(preallocated_gss, crate::runtime::state::LINEAR_STACK_RESERVE)
    }

    pub(crate) fn clear(&mut self) {
        self.len = 0;
        self.action.clear();
        self.continuation_cache_len = 0;
    }

    pub(super) fn continuation_decision(
        &self,
        offset: usize,
        tokenizer_state: u32,
        end_state: u32,
    ) -> Option<bool> {
        self.continuation_cache[..self.continuation_cache_len]
            .iter()
            .find(|decision| {
                decision.offset == offset
                    && decision.tokenizer_state == tokenizer_state
                    && decision.end_state == end_state
            })
            .map(|decision| decision.viable)
    }

    pub(super) fn cache_continuation_decision(
        &mut self,
        offset: usize,
        tokenizer_state: u32,
        end_state: u32,
        viable: bool,
    ) {
        if self.continuation_cache_len == self.continuation_cache.len() {
            return;
        }
        self.continuation_cache[self.continuation_cache_len] =
            FlatContinuationDecision { offset, tokenizer_state, end_state, viable };
        self.continuation_cache_len += 1;
    }

    pub(super) fn enqueue(
        &mut self,
        offset: usize,
        tokenizer_state: u32,
        stack: &[u32],
        acc: TerminalsDisallowed,
    ) -> bool {
        if !acc.is_inline() {
            return false;
        }
        for branch in &mut self.branches[..self.len] {
            if branch.offset == offset
                && branch.tokenizer_state == tokenizer_state
                && branch.stack.as_slice() == stack
            {
                let Some(merged) = branch.acc.try_merge_inline(&acc) else {
                    return false;
                };
                branch.acc = merged;
                return true;
            }
        }
        if self.len == self.branches.len() {
            return false;
        }
        let branch = &mut self.branches[self.len];
        if stack.len() > branch.stack.capacity() {
            return false;
        }
        branch.offset = offset;
        branch.tokenizer_state = tokenizer_state;
        branch.stack.clear();
        branch.stack.extend_from_slice(stack);
        branch.acc = acc;
        branch.processed = false;
        branch.initial_pruned = false;
        self.len += 1;
        true
    }

    pub(super) fn can_recycle_old_state(
        &self,
        state: &ParserStateMap,
        selected_count: usize,
    ) -> bool {
        let mut recyclable = 0usize;
        let mut retired = 0usize;
        for (_, gss) in &state.entries {
            if gss.can_replace_single_path_state_in_place(&[0]) {
                recyclable += 1;
            } else {
                retired += 1;
            }
        }
        self.gss_pool.len().saturating_sub(selected_count) + recyclable
            <= FLAT_FRONTIER_GSS_POOL_CAPACITY
            && self.retired_gss.len() + retired <= FLAT_FRONTIER_RETIRED_GSS_CAPACITY
    }

    pub(super) fn reclaim_retired_gss(&mut self) {
        if self.retired_gss.is_empty() || self.gss_pool.len() == FLAT_FRONTIER_GSS_POOL_CAPACITY {
            return;
        }
        let mut still_retired = SmallVec::<[ParserGSS; FLAT_FRONTIER_RETIRED_GSS_CAPACITY]>::new();
        for gss in self.retired_gss.drain(..) {
            if self.gss_pool.len() < FLAT_FRONTIER_GSS_POOL_CAPACITY
                && gss.can_replace_single_path_state_in_place(&[0])
            {
                self.gss_pool.push(gss);
            } else {
                still_retired.push(gss);
            }
        }
        self.retired_gss = still_retired;
    }

    pub(super) fn recycle_old_entries(
        &mut self,
        old_entries: SmallVec<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>,
    ) {
        self.reclaim_retired_gss();
        for (_, gss) in old_entries {
            if gss.can_replace_single_path_state_in_place(&[0]) {
                debug_assert!(self.gss_pool.len() < FLAT_FRONTIER_GSS_POOL_CAPACITY);
                self.gss_pool.push(gss);
            } else {
                debug_assert!(self.retired_gss.len() < FLAT_FRONTIER_RETIRED_GSS_CAPACITY);
                self.retired_gss.push(gss);
            }
        }
    }

    pub(super) fn replace_state_with_shared_uniform_stack_keys(
        &mut self,
        state: &mut ParserStateMap,
        keys: &[u32],
        stack: &[u32],
        acc: &TerminalsDisallowed,
    ) -> bool {
        if keys.is_empty()
            || keys.windows(2).any(|pair| pair[0] >= pair[1])
            || !self.can_recycle_old_state(state, 1)
        {
            return false;
        }
        self.reclaim_retired_gss();
        let Some(pool_index) =
            self.gss_pool.iter().rposition(|gss| gss.can_replace_single_path_state_in_place(stack))
        else {
            return false;
        };
        let mut shared = self.gss_pool.swap_remove(pool_index);
        if !shared.try_replace_single_path_state_in_place(stack, acc.clone()) {
            self.gss_pool.push(shared);
            return false;
        }

        let mut new_entries = SmallVec::<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>::new();
        new_entries.reserve(keys.len());
        for &key in &keys[..keys.len() - 1] {
            new_entries.push((key, shared.clone()));
        }
        new_entries.push((*keys.last().unwrap(), shared));
        let old_entries = std::mem::replace(&mut state.entries, new_entries);
        self.recycle_old_entries(old_entries);
        true
    }

    pub(super) fn replace_state_with_uniform_stack_keys(
        &mut self,
        state: &mut ParserStateMap,
        keys: &[u32],
        stack: &[u32],
        acc: &TerminalsDisallowed,
    ) -> bool {
        if keys.is_empty()
            || keys.len() > INLINE_PARSER_STATE_CAPACITY
            || keys.windows(2).any(|pair| pair[0] >= pair[1])
            || self.gss_pool.len() < keys.len()
            || !self.can_recycle_old_state(state, keys.len())
        {
            return false;
        }

        let mut selected = SmallVec::<[ParserGSS; INLINE_PARSER_STATE_CAPACITY]>::new();
        for _ in keys {
            let Some(pool_index) = self
                .gss_pool
                .iter()
                .rposition(|gss| gss.can_replace_single_path_state_in_place(stack))
            else {
                for gss in selected.drain(..) {
                    self.gss_pool.push(gss);
                }
                return false;
            };
            selected.push(self.gss_pool.swap_remove(pool_index));
        }

        let mut new_entries = SmallVec::<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>::new();
        for (mut gss, &key) in selected.into_iter().zip(keys) {
            let replaced = gss.try_replace_single_path_state_in_place(stack, acc.clone());
            debug_assert!(replaced, "uniform flat-frontier spare eligibility changed");
            if !replaced {
                unreachable!("uniform spare was validated before mutation");
            }
            new_entries.push((key, gss));
        }

        let old_entries = std::mem::replace(&mut state.entries, new_entries);
        self.recycle_old_entries(old_entries);
        true
    }
}

pub(super) fn apply_flat_reduce(
    constraint: &Constraint,
    stack: &mut FlatInlineStack,
    nonterminal: u32,
    len: u32,
) -> Option<bool> {
    let pop = len as usize;
    if pop >= stack.len() {
        return Some(false);
    }
    stack.truncate(stack.len() - pop);
    let goto_from = *stack.last()?;
    let (target, replace) = constraint.table.goto_target(goto_from, nonterminal)?;
    if replace {
        *stack.last_mut()? = target;
    } else {
        if stack.len() == LINEAR_STACK_RESERVE {
            return None;
        }
        stack.push(target);
    }
    Some(true)
}

pub(super) fn apply_flat_shift(
    stack: &mut FlatInlineStack,
    target: u32,
    replace: bool,
) -> Option<bool> {
    if replace {
        *stack.last_mut()? = target;
    } else {
        if stack.len() == LINEAR_STACK_RESERVE {
            return None;
        }
        stack.push(target);
    }
    Some(true)
}

pub(super) fn apply_flat_stack_effect(
    stack: &mut FlatInlineStack,
    pop: u32,
    pushes: &[u32],
) -> Option<bool> {
    let pop = pop as usize;
    // Stack effects are atomic: popping the final visible state is valid when
    // the same effect pushes a replacement. Only a true underflow is invalid.
    if pop > stack.len() {
        return Some(false);
    }
    let new_len = stack.len() - pop + pushes.len();
    if new_len > LINEAR_STACK_RESERVE {
        return None;
    }
    stack.truncate(stack.len() - pop);
    stack.extend_from_slice(pushes);
    Some(true)
}

pub(super) fn flat_guard_matches(
    stack: &[u32],
    guard: &crate::compiler::glr::table::StackShiftGuard,
) -> bool {
    let pop = guard.pop as usize;
    pop < stack.len() && guard.states.iter().any(|state| *state == stack[stack.len() - 1 - pop])
}

/// Apply one terminal to one concrete LR stack, retaining a bounded set of
/// exact concrete outcomes. This is the small-ambiguity counterpart to the
/// persistent GSS interpreter: no heap allocation is permitted, and any
/// capacity overflow or accept-state edge declines to the authoritative path.
pub(super) fn apply_terminal_to_flat_stacks(
    constraint: &Constraint,
    terminal: u32,
    source: &[u32],
    scratch: &mut FlatActionScratch,
) -> Option<bool> {
    if source.is_empty() || source.len() > LINEAR_STACK_RESERVE {
        return None;
    }
    scratch.clear();
    let mut initial = FlatInlineStack::new();
    initial.extend_from_slice(source);
    if !scratch.push_pending(initial) {
        return None;
    }

    let mut steps = 0usize;
    while let Some(mut stack) = scratch.pending.pop() {
        steps += 1;
        if steps > FLAT_ACTION_MAX_STEPS {
            return None;
        }
        let top = *stack.last()?;
        match constraint.table.action(top, terminal)? {
            Action::Skip => {
                if !scratch.push_complete(stack) {
                    return None;
                }
            }
            Action::Shift(target, replace) => {
                if apply_flat_shift(&mut stack, *target, *replace)? && !scratch.push_complete(stack)
                {
                    return None;
                }
            }
            Action::ReplaceShifts(targets) => {
                for &target in targets.iter() {
                    let mut candidate = stack.clone();
                    if apply_flat_stack_effect(&mut candidate, 1, &[target])?
                        && !scratch.push_complete(candidate)
                    {
                        return None;
                    }
                }
            }
            Action::StackShifts(shifts) => {
                for shift in shifts {
                    let mut candidate = stack.clone();
                    if apply_flat_stack_effect(&mut candidate, shift.pop, &shift.pushes)?
                        && !scratch.push_complete(candidate)
                    {
                        return None;
                    }
                }
            }
            Action::GuardedStackShifts(shifts) => {
                for shift in shifts {
                    if !shift.guards.iter().all(|guard| flat_guard_matches(&stack, guard)) {
                        continue;
                    }
                    let mut candidate = stack.clone();
                    if apply_flat_stack_effect(&mut candidate, shift.pop, &shift.pushes)?
                        && !scratch.push_complete(candidate)
                    {
                        return None;
                    }
                }
            }
            Action::Reduce(nonterminal, len) => {
                if apply_flat_reduce(constraint, &mut stack, *nonterminal, *len)?
                    && !scratch.push_pending(stack)
                {
                    return None;
                }
            }
            Action::Split { shift, reduces, accept } => {
                if *accept {
                    return None;
                }
                if let Some((target, replace)) = shift {
                    let mut candidate = stack.clone();
                    if apply_flat_shift(&mut candidate, *target, *replace)?
                        && !scratch.push_complete(candidate)
                    {
                        return None;
                    }
                }
                for &(nonterminal, len) in reduces {
                    let mut candidate = stack.clone();
                    if apply_flat_reduce(constraint, &mut candidate, nonterminal, len)?
                        && !scratch.push_pending(candidate)
                    {
                        return None;
                    }
                }
            }
            Action::Accept => return None,
        }
    }
    Some(!scratch.complete.is_empty())
}

pub(super) fn flat_stack_may_advance_on_any(
    constraint: &Constraint,
    stack: &[u32],
    terminals: &crate::ds::bitset::BitSet,
    scratch: &mut FlatActionScratch,
) -> Option<bool> {
    let top = *stack.last()?;
    if runtime_future_contains_ignore(constraint, terminals) {
        return Some(true);
    }
    if constraint.table.admission_policy == AdmissionPolicy::RowPresenceExact {
        return Some(constraint.table.advance_row_intersects(top, terminals));
    }

    let mut unknown = false;
    for terminal in terminals.iter() {
        match apply_terminal_to_flat_stacks(constraint, terminal as u32, stack, scratch) {
            Some(true) => return Some(true),
            Some(false) => {}
            None => unknown = true,
        }
    }
    (!unknown).then_some(false)
}

pub(super) fn flat_frontier_group_may_advance_on_any(
    constraint: &Constraint,
    branches: &[FlatBranchScratch],
    offset: usize,
    tokenizer_state: u32,
    terminals: &crate::ds::bitset::BitSet,
    scratch: &mut FlatActionScratch,
) -> Option<bool> {
    let mut saw_branch = false;
    let mut unknown = false;
    for branch in branches {
        if branch.offset != offset || branch.tokenizer_state != tokenizer_state {
            continue;
        }
        saw_branch = true;
        match flat_stack_may_advance_on_any(constraint, &branch.stack, terminals, scratch) {
            Some(true) => return Some(true),
            Some(false) => {}
            None => unknown = true,
        }
    }

    if !saw_branch {
        Some(false)
    } else if unknown {
        None
    } else {
        Some(false)
    }
}

pub(super) fn prune_flat_branch_acc(
    constraint: &Constraint,
    acc: &TerminalsDisallowed,
    tokenizer_state: u32,
    end_states: &[u32],
    matches: &[TokenizerMatch],
    bytes: &[u8],
) -> Option<Option<TerminalsDisallowed>> {
    if !acc.is_inline() {
        return None;
    }
    match advance_terminals_disallowed_over_bytes(
        constraint,
        acc,
        bytes,
        Some((tokenizer_state, end_states, matches)),
    ) {
        None => Some(None),
        Some(remapped) if remapped.is_inline() => Some(Some(remapped)),
        Some(_) => None,
    }
}

pub(super) fn apply_flat_future_disallow(
    constraint: &Constraint,
    acc: &TerminalsDisallowed,
    end_states: &[u32],
    terminal: u32,
) -> Option<TerminalsDisallowed> {
    let mut updated = acc.clone();
    for &end_state in end_states {
        if constraint.tokenizer.possible_future_terminals(end_state).contains(terminal as usize) {
            updated = updated.try_with_insert_inline(end_state, terminal)?;
        }
    }
    Some(updated)
}

pub(super) fn try_commit_flat_frontier_in_place(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    original: &mut Vec<u32>,
    work: &mut Vec<u32>,
    tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
    frontier: &mut FlatFrontierScratch,
) -> Option<Result<(), String>> {
    frontier.reclaim_retired_gss();
    macro_rules! flat_decline {
        ($reason:literal) => {{
            if std::env::var_os("GLRMASK_DEBUG_COMMIT_PATH").is_some() {
                eprintln!("[glrmask/debug][commit_path] flat_frontier_decline reason={}", $reason);
            }
            return None;
        }};
    }
    if state.is_empty() || state.len() > FLAT_FRONTIER_MAX_BRANCHES || bytes.is_empty() {
        flat_decline!("input-shape");
    }
    frontier.clear();
    for (&tokenizer_state, gss) in state.iter() {
        if !gss.copy_single_path_stack_into(original) {
            flat_decline!("input-not-single-path");
        }
        let Some(acc) = gss.single_path_acc() else {
            flat_decline!("missing-single-acc");
        };
        if !frontier.enqueue(0, tokenizer_state, original, acc) {
            flat_decline!("initial-enqueue");
        }
    }

    let initial_tokenizer_state = constraint.runtime_commit_initial_state();
    for offset in 0..bytes.len() {
        let mut index = 0usize;
        while index < frontier.len {
            if frontier.branches[index].processed || frontier.branches[index].offset != offset {
                index += 1;
                continue;
            }

            let tokenizer_state = frontier.branches[index].tokenizer_state;
            if frontier.branches[index].stack.len() > original.capacity() {
                flat_decline!("copy-capacity");
            }
            original.clear();
            original.extend_from_slice(&frontier.branches[index].stack);
            let mut acc = frontier.branches[index].acc.clone();
            frontier.branches[index].processed = true;

            if !execute_tokenizer_reusable(
                constraint,
                &bytes[offset..],
                tokenizer_state,
                tokenizer_scratch,
            ) || tokenizer_scratch.matches.len() > SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX
            {
                flat_decline!("tokenizer-scratch-or-matches");
            }
            let top_state = *original.last()?;
            if offset == 0 {
                // Delayed exclusions are correlated with parser alternatives,
                // but tokenizer execution is shared by every alternative under
                // one lexer key. Prune the whole group once and persist the
                // exact remapped accumulators before deciding continuation
                // viability from the group. The former per-branch local copy
                // left stale nonempty accumulators in sibling branches and
                // forced a general-path fallback even when pruning discharged
                // every exclusion.
                if !frontier.branches[index].initial_pruned {
                    let group_len = frontier.len;
                    for prune_index in 0..group_len {
                        if frontier.branches[prune_index].offset != offset
                            || frontier.branches[prune_index].tokenizer_state != tokenizer_state
                            || frontier.branches[prune_index].initial_pruned
                        {
                            continue;
                        }
                        let result = prune_flat_branch_acc(
                            constraint,
                            &frontier.branches[prune_index].acc,
                            tokenizer_state,
                            &tokenizer_scratch.states,
                            &tokenizer_scratch.matches,
                            bytes,
                        );
                        match result {
                            None => flat_decline!("prune-acc-promotion"),
                            Some(Some(pruned)) => {
                                frontier.branches[prune_index].acc = pruned;
                                frontier.branches[prune_index].initial_pruned = true;
                            }
                            Some(None) => {
                                // Keep array indices stable while removing this
                                // alternative from all later offset/key groups.
                                frontier.branches[prune_index].offset = usize::MAX;
                                frontier.branches[prune_index].processed = true;
                                frontier.branches[prune_index].initial_pruned = true;
                            }
                        }
                    }
                }
                if frontier.branches[index].offset != offset {
                    index += 1;
                    continue;
                }
                acc = frontier.branches[index].acc.clone();
            }

            let actionable = ActionableTerminals::SingleState(top_state);
            let normalized_matches = collect_unique_actionable_matches(
                constraint,
                Some(&actionable),
                constraint.ignore_terminal,
                &tokenizer_scratch.matches,
                None,
            );
            for matched in normalized_matches {
                let new_offset = offset + matched.width;
                if matched.width == 0 || new_offset > bytes.len() {
                    flat_decline!("invalid-match-width");
                }
                if matched.ignored {
                    if !frontier.enqueue(new_offset, initial_tokenizer_state, original, acc.clone())
                    {
                        flat_decline!("ignored-enqueue");
                    }
                    continue;
                }

                let parser_action_result = apply_terminal_to_flat_stacks(
                    constraint,
                    matched.terminal_id,
                    original,
                    &mut frontier.action,
                );
                match parser_action_result {
                    Some(true) => {}
                    Some(false) => continue,
                    None => flat_decline!("parser-action-capacity"),
                }
                let Some(advanced_acc) = apply_flat_future_disallow(
                    constraint,
                    &acc,
                    &tokenizer_scratch.states,
                    matched.terminal_id,
                ) else {
                    return None;
                };
                for action_index in 0..frontier.action.complete.len() {
                    work.clear();
                    work.extend_from_slice(&frontier.action.complete[action_index]);
                    if !frontier.enqueue(
                        new_offset,
                        initial_tokenizer_state,
                        work,
                        advanced_acc.clone(),
                    ) {
                        flat_decline!("advanced-enqueue");
                    }
                }
            }

            for &end_state in &tokenizer_scratch.states {
                let viable = if end_state == initial_tokenizer_state {
                    true
                } else {
                    let branches = &frontier.branches[..frontier.len];
                    let mut group = branches.iter().filter(|branch| {
                        branch.offset == offset && branch.tokenizer_state == tokenizer_state
                    });
                    let Some(first) = group.next() else {
                        flat_decline!("missing-continuation-group");
                    };
                    let correlated = group.next().is_some();
                    let possible = constraint.tokenizer.possible_future_terminals(end_state);
                    if !correlated {
                        let Some(viable) = flat_stack_may_advance_on_any(
                            constraint,
                            &first.stack,
                            possible,
                            &mut frontier.action,
                        ) else {
                            flat_decline!("continuation-admission-ambiguous");
                        };
                        viable
                    } else if let Some(viable) =
                        frontier.continuation_decision(offset, tokenizer_state, end_state)
                    {
                        viable
                    } else {
                        let group_is_safe = offset > 0
                            || branches
                                .iter()
                                .filter(|branch| {
                                    branch.offset == offset
                                        && branch.tokenizer_state == tokenizer_state
                                })
                                .all(|branch| branch.acc.is_empty());
                        if !group_is_safe {
                            flat_decline!("initial-continuation-group-needs-pruning");
                        }
                        let Some(viable) = flat_frontier_group_may_advance_on_any(
                            constraint,
                            branches,
                            offset,
                            tokenizer_state,
                            possible,
                            &mut frontier.action,
                        ) else {
                            flat_decline!("continuation-admission-ambiguous");
                        };
                        frontier.cache_continuation_decision(
                            offset,
                            tokenizer_state,
                            end_state,
                            viable,
                        );
                        viable
                    }
                };
                if viable && !frontier.enqueue(bytes.len(), end_state, original, acc.clone()) {
                    flat_decline!("continuation-enqueue");
                }
            }
            index += 1;
        }
    }

    let mut outputs = SmallVec::<[usize; INLINE_PARSER_STATE_CAPACITY]>::new();
    for index in 0..frontier.len {
        if frontier.branches[index].offset == bytes.len() {
            // The inline capacity is a performance tier, not a semantic limit.
            // SmallVec spills once for 65-128 outputs while retaining the exact
            // flat algorithm instead of restarting through the general queue.
            outputs.push(index);
        }
    }
    if outputs.is_empty() {
        return Some(Err("commit rejected: no valid parser states remain".to_string()));
    }
    outputs.sort_unstable_by(|&left, &right| {
        frontier.branches[left]
            .tokenizer_state
            .cmp(&frontier.branches[right].tokenizer_state)
            .then_with(|| frontier.branches[left].stack.cmp(&frontier.branches[right].stack))
    });

    // Use both preallocated spares and uniquely-owned active GSS objects for
    // the output double buffer. The old implementation required the spare pool
    // alone to cover every output, even when all active entries were exactly
    // recyclable. A seven-way lexer frontier could therefore decline with two
    // spares plus seven recyclable active entries and fall into the allocating
    // general path despite having nine usable objects for seven outputs.
    let recyclable_old = state
        .entries
        .iter()
        .filter(|(_, gss)| gss.can_replace_single_path_state_in_place(&[0]))
        .count();
    let retired_old = state.len() - recyclable_old;
    let available_count = frontier.gss_pool.len() + recyclable_old;
    if available_count < outputs.len()
        || available_count - outputs.len() > FLAT_FRONTIER_GSS_POOL_CAPACITY
        || frontier.retired_gss.len() + retired_old > FLAT_FRONTIER_RETIRED_GSS_CAPACITY
    {
        flat_decline!("gss-pool-capacity");
    }

    // Verify a concrete one-to-one assignment before moving anything out of
    // the active state. Match the longest output stacks first so a larger spare
    // cannot be consumed by a short stack while a smaller spare remains.
    let assignments = {
        let mut candidates = SmallVec::<[&ParserGSS; 32]>::new();
        candidates.extend(frontier.gss_pool.iter());
        candidates.extend(
            state
                .entries
                .iter()
                .map(|(_, gss)| gss)
                .filter(|gss| gss.can_replace_single_path_state_in_place(&[0])),
        );
        debug_assert_eq!(candidates.len(), available_count);

        let mut output_order = SmallVec::<[usize; INLINE_PARSER_STATE_CAPACITY]>::new();
        output_order.extend(0..outputs.len());
        output_order.sort_unstable_by_key(|&output_index| {
            std::cmp::Reverse(frontier.branches[outputs[output_index]].stack.len())
        });

        let mut used = [false; FLAT_FRONTIER_GSS_POOL_CAPACITY + FLAT_FRONTIER_MAX_BRANCHES];
        let mut assignments = SmallVec::<[usize; INLINE_PARSER_STATE_CAPACITY]>::new();
        assignments.resize(outputs.len(), usize::MAX);
        for output_index in output_order {
            let stack = &frontier.branches[outputs[output_index]].stack;
            let Some(candidate_index) =
                candidates.iter().enumerate().position(|(candidate_index, gss)| {
                    !used[candidate_index] && gss.can_replace_single_path_state_in_place(stack)
                })
            else {
                flat_decline!("gss-pool-shared-or-shape");
            };
            used[candidate_index] = true;
            assignments[output_index] = candidate_index;
        }
        assignments
    };

    // Candidate order is stable: existing spares first, followed by recyclable
    // active entries. Non-recyclable active GSSs remain retained so dropping
    // shared persistent nodes never lands in the token hot path.
    let mut available = SmallVec::<[Option<ParserGSS>; 32]>::new();
    available.extend(frontier.gss_pool.drain(..).map(Some));
    let old_entries = std::mem::take(&mut state.entries);
    for (_, gss) in old_entries {
        if gss.can_replace_single_path_state_in_place(&[0]) {
            available.push(Some(gss));
        } else {
            frontier.retired_gss.push(gss);
        }
    }
    debug_assert_eq!(available.len(), available_count);

    let mut new_entries = SmallVec::<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>::new();
    for (output_index, &output) in outputs.iter().enumerate() {
        let mut gss = available[assignments[output_index]]
            .take()
            .expect("prevalidated flat-frontier GSS assignment must remain available");
        let branch = &frontier.branches[output];
        let replaced =
            gss.try_replace_single_path_state_in_place(&branch.stack, branch.acc.clone());
        debug_assert!(replaced, "flat-frontier spare eligibility changed");
        if !replaced {
            unreachable!("flat-frontier spare was validated before mutation");
        }
        new_entries.push((branch.tokenizer_state, gss));
    }
    for gss in available.into_iter().flatten() {
        frontier.gss_pool.push(gss);
    }
    debug_assert!(frontier.gss_pool.len() <= FLAT_FRONTIER_GSS_POOL_CAPACITY);
    state.entries = new_entries;
    Some(Ok(()))
}

#[cfg(test)]
#[test]
fn mask_only_flat_frontier_keeps_empty_semantics_without_branch_allocations() {
    let lean = FlatFrontierScratch::for_mask_only_shadow();
    let commit = FlatFrontierScratch::with_preallocated_gss(0);
    assert_eq!(lean.len, commit.len);
    assert_eq!(lean.continuation_cache_len, commit.continuation_cache_len);
    assert!(lean.gss_pool.is_empty() && commit.gss_pool.is_empty());
    assert!(lean.retired_gss.is_empty() && commit.retired_gss.is_empty());
    for (left, right) in lean.branches.iter().zip(&commit.branches) {
        assert!(left.stack.is_empty() && right.stack.is_empty());
        assert_eq!(left.stack.capacity(), 0);
        assert!(right.stack.capacity() >= crate::runtime::state::LINEAR_STACK_RESERVE);
        assert_eq!(left.offset, right.offset);
        assert_eq!(left.tokenizer_state, right.tokenizer_state);
        assert_eq!(left.processed, right.processed);
        assert_eq!(left.initial_pruned, right.initial_pruned);
        assert_eq!(left.acc, right.acc);
    }
}


impl FlatBranchScratch {
    fn with_stack_capacity(stack_capacity: usize) -> Self {
        Self {
            offset: 0,
            tokenizer_state: 0,
            stack: Vec::with_capacity(stack_capacity),
            acc: TerminalsDisallowed::new(),
            processed: false,
            initial_pruned: false,
        }
    }
}


impl FlatFrontierScratch {
    /// Read-only mask shadows never execute the flat commit frontier. Avoid
    /// reserving 128 unused branch stacks as well as the already-omitted GSS
    /// pool. Logical empty state is identical; ordinary commits keep their
    /// original branch-stack reservations unchanged.
    pub(crate) fn for_mask_only_shadow() -> Self {
        Self::with_reservation_policy(0, 0)
    }
}


impl FlatFrontierScratch {

    fn with_reservation_policy(preallocated_gss: usize, stack_capacity: usize) -> Self {
        let mut gss_pool = SmallVec::new();
        for _ in 0..preallocated_gss.min(FLAT_FRONTIER_GSS_POOL_CAPACITY) {
            let mut gss = ParserGSS::from_single_stack(
                vec![0],
                TerminalsDisallowed::new(),
            );
            let reserved = gss.reserve_single_segment_capacity(
                crate::runtime::state::LINEAR_STACK_RESERVE,
            );
            debug_assert!(reserved, "fresh flat-frontier GSS must be reservable");
            gss_pool.push(gss);
        }
        Self {
            branches: std::array::from_fn(|_| FlatBranchScratch::with_stack_capacity(stack_capacity)),
            len: 0,
            action: FlatActionScratch::default(),
            continuation_cache: [FlatContinuationDecision::default();
                FLAT_CONTINUATION_CACHE_CAPACITY],
            continuation_cache_len: 0,
            gss_pool,
            retired_gss: SmallVec::new(),
        }
    }
}
