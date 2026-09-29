//! Bounded static masking for explicit parser paths.
//!
//! Path storage belongs to the sequence scratch and retains spilled capacity.
//! Repeated-stack planning is isolated so ordinary masks do not reserve its
//! large inline instruction buffer. All eligibility and work limits remain exact.

use smallvec::SmallVec;

use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::labels::encode_positive_label;
use crate::runtime::constraint::RuntimeWeightRef;
use crate::runtime::state::ConstraintState;

use super::{
    Constraint, DenseMaskAcc, DenseTokenMaskCache, DELTA_SEED_MIN_SAVINGS,
    MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY,
    MASK_SINGLE_PATH_DIRECT_INLINE_STACK_DEPTH, MASK_SINGLE_PATH_DIRECT_MAX_DEPTH,
    MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS, MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS,
    MASK_SINGLE_PATH_DIRECT_MAX_PATHS_PER_GSS,
    MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES, MASK_SINGLE_PATH_DIRECT_MIN_PLAN_STACK_VALUES,
    MASK_SINGLE_PATH_DIRECT_TWO_PASS_MIN_STATE_COUNT, StackWalkEvent,
    intersect_static_dense_with_weight, mask_delta_profile_enabled, mask_inner_profile_enabled,
    mask_single_path_to_stacks_fallback_disabled, walk_single_stack,
};

// Speculative delta pricing must stay cheaper than simply rebuilding a mask.
// This budget limits the optimization, never the set of admitted tokens.
const MAX_MONOTONE_MASK_ADDITIONS: u32 = 64;

fn bounded_monotone_growth(previous: &[u64], current: &[u64]) -> bool {
    if previous.len() != current.len() {
        return false;
    }
    let mut additions = 0u32;
    for (&old, &new) in previous.iter().zip(current) {
        if old & !new != 0 {
            return false;
        }
        additions += (new & !old).count_ones();
        if additions > MAX_MONOTONE_MASK_ADDITIONS {
            return false;
        }
    }
    true
}

type SinglePathMaskPath = (
    u32,
    TerminalsDisallowed,
    SmallVec<[u32; MASK_SINGLE_PATH_DIRECT_INLINE_STACK_DEPTH]>,
);

// This is only inline storage, not an admission limit. Wider paths spill once
// and retain the allocation between masks; MAX_TOTAL_PATHS still admits 128.
pub(crate) type SinglePathMaskPaths = SmallVec<[SinglePathMaskPath; 4]>;

pub(super) fn single_path_direct_stack_work(
    stack_lengths: impl IntoIterator<Item = usize>,
) -> Option<usize> {
    let mut total = 0usize;
    for len in stack_lengths {
        total = total.saturating_add(len);
        if total > MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES {
            return None;
        }
    }
    Some(total)
}

#[inline]
pub(super) fn single_path_direct_plan_reuse_dominates(
    path_count: usize,
    total_stack_values: usize,
    repeated_stack_values: usize,
) -> bool {
    path_count >= 3
        && total_stack_values >= MASK_SINGLE_PATH_DIRECT_MIN_PLAN_STACK_VALUES
        && repeated_stack_values > total_stack_values.saturating_sub(repeated_stack_values)
}

#[derive(Clone, Copy)]
enum SinglePathDirectPlanOp<'a> {
    Merge(RuntimeWeightRef<'a>),
    Intersect(RuntimeWeightRef<'a>),
}

#[derive(Clone, Copy)]
struct SinglePathDirectStackPlan {
    representative_path: usize,
    stack_fingerprint: u64,
    ops_start: usize,
    ops_end: usize,
}

#[inline]
fn single_path_direct_stack_fingerprint(stack: &[u32]) -> u64 {
    // FNV-1a is sufficient here: equality is still checked before sharing a
    // plan, so the fingerprint only avoids repeatedly comparing long stacks.
    let mut fingerprint = 0xcbf29ce484222325u64;
    for &state in stack {
        fingerprint ^= u64::from(state);
        fingerprint = fingerprint.wrapping_mul(0x100000001b3);
    }
    fingerprint ^ (stack.len() as u64).wrapping_mul(0x9e3779b97f4a7c15)
}

fn materialize_single_path_seed_intersection(
    base: &[u64],
    dense: &mut Vec<u64>,
    internal_tsid: u32,
    weight: RuntimeWeightRef<'_>,
    constraint: &Constraint,
) -> bool {
    debug_assert!(!weight.is_full());
    let Some(token_set) = weight.token_set_for_tsid(internal_tsid) else {
        dense.clear();
        return false;
    };

    dense.clear();
    dense.resize(base.len(), 0);
    if let Some(mask) = constraint.runtime_token_set_dense_mask(token_set) {
        let mut any = false;
        for (idx, dense_word) in dense.iter_mut().enumerate() {
            *dense_word = base[idx] & mask.get(idx).copied().unwrap_or(0);
            any |= *dense_word != 0;
        }
        return any;
    }

    let mut any = false;
    DenseMaskAcc::for_each_runtime_token_range_word(
        token_set,
        base.len(),
        |word_idx, token_mask| {
            let word = base[word_idx] & token_mask;
            dense[word_idx] |= word;
            any |= word != 0;
        },
    );
    any
}

impl ConstraintState<'_> {
    pub(super) fn try_fill_mask_single_path_direct(&self, buf: &mut [u32]) -> bool {
        if mask_inner_profile_enabled() || mask_delta_profile_enabled() {
            return false;
        }

        if self.state.is_empty() || self.state.len() > MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS {
            return false;
        }

        let mut paths = {
            let mut scratch = self.mask_scratch.lock().unwrap();
            std::mem::take(&mut scratch.single_path_paths)
        };
        debug_assert!(paths.is_empty());
        let result = self.fill_mask_single_path_direct_with_paths(buf, &mut paths);
        paths.clear();
        self.mask_scratch.lock().unwrap().single_path_paths = paths;
        result
    }

    fn fill_mask_single_path_direct_with_paths(
        &self,
        buf: &mut [u32],
        paths: &mut SinglePathMaskPaths,
    ) -> bool {
        if !self.collect_single_path_mask_paths(paths) {
            return false;
        }
        let Some(total_stack_values) =
            single_path_direct_stack_work(paths.iter().map(|(_, _, stack)| stack.len()))
        else {
            return false;
        };
        if self.constraint.runtime_parser_dwa_state_count() == 0 {
            return false;
        }

        buf.fill(0);

        let precomputed = &self.constraint.weight_token_dense_masks;
        let dense_words = self.constraint.internal_token_dense_words;
        let (mut merged, mut output_scratch, mut single_path_aux, mut single_path_acc) = {
            let mut scratch = self.mask_scratch.lock().unwrap();
            (
                std::mem::take(&mut scratch.merged_dense),
                std::mem::take(&mut scratch.output_buf),
                std::mem::take(&mut scratch.single_path_aux_dense),
                std::mem::take(&mut scratch.single_path_acc_dense),
            )
        };
        merged.clear();
        merged.resize(dense_words, 0);
        let mut used_direct_final = false;
        let mut direct_buf_dirty = false;

        let restore_scratch = |merged: Vec<u64>,
                               output_scratch: Vec<u32>,
                               single_path_aux: Vec<u64>,
                               single_path_acc: Vec<u64>| {
            let mut scratch = self.mask_scratch.lock().unwrap();
            scratch.merged_dense = merged;
            scratch.chain_merged_dense.clear();
            scratch.output_buf = output_scratch;
            scratch.single_path_aux_dense = single_path_aux;
            scratch.single_path_acc_dense = single_path_acc;
        };

        let used_stack_plans = paths.len() >= 3
            && total_stack_values >= MASK_SINGLE_PATH_DIRECT_MIN_PLAN_STACK_VALUES
            && self.with_repeated_stack_plans(
                paths,
                total_stack_values,
                |plan_ops, stack_plans, path_plan_indices| {
                    for (path_index, (original_tokenizer_state, terminals_disallowed, _)) in
                        paths.iter().enumerate()
                    {
                        let internal_tsid = self
                            .constraint
                            .internal_tsid_for_state(*original_tokenizer_state);
                        let seed_base = &self.constraint.seed_universe_dense;
                        let mut dense_is_seed = terminals_disallowed.is_empty();
                        if dense_is_seed {
                            if seed_base.is_empty() {
                                continue;
                            }
                        } else if !self.fill_single_path_seed_dense(
                            terminals_disallowed,
                            &mut single_path_aux,
                            &mut single_path_acc,
                        ) {
                            continue;
                        }

                        let plan = &stack_plans[path_plan_indices[path_index] as usize];
                        for op in &plan_ops[plan.ops_start..plan.ops_end] {
                            match *op {
                                SinglePathDirectPlanOp::Merge(weight) => {
                                    used_direct_final = true;
                                    let dense = if dense_is_seed {
                                        seed_base.as_ref()
                                    } else {
                                        single_path_acc.as_slice()
                                    };
                                    self.merge_single_path_final_weight_to_internal(
                                        weight,
                                        internal_tsid,
                                        dense,
                                        precomputed,
                                        &mut merged,
                                        Some(&mut *buf),
                                        &mut direct_buf_dirty,
                                    );
                                    if weight.is_full() { break; }
                                }
                                SinglePathDirectPlanOp::Intersect(weight) => if dense_is_seed {
                                    if weight.is_full() {
                                        continue;
                                    }
                                    if !materialize_single_path_seed_intersection(
                                        seed_base,
                                        &mut single_path_acc,
                                        internal_tsid,
                                        weight,
                                        self.constraint,
                                    ) {
                                        break;
                                    }
                                    dense_is_seed = false;
                                } else if !Self::intersect_single_path_dense_with_weight_in_place(
                                    &mut single_path_acc,
                                    &mut single_path_aux,
                                    internal_tsid,
                                    weight,
                                    self.constraint,
                                ) {
                                    break;
                                },
                            }
                        }
                    }
                },
            );
        if !used_stack_plans {
            for (original_tokenizer_state, terminals_disallowed, stack) in paths.iter() {
                let internal_tsid = self
                    .constraint
                    .internal_tsid_for_state(*original_tokenizer_state);
                let seed_base = &self.constraint.seed_universe_dense;
                let mut dense_is_seed = terminals_disallowed.is_empty();
                if dense_is_seed {
                    if seed_base.is_empty() {
                        continue;
                    }
                } else if !self.fill_single_path_seed_dense(
                    terminals_disallowed,
                    &mut single_path_aux,
                    &mut single_path_acc,
                ) {
                    continue;
                }

                walk_single_stack::<true, _>(
                    self.constraint.runtime_parser_dwa_start_state(),
                    stack,
                    |state| self.constraint.runtime_parser_dwa_final_weight(state),
                    |state, parser| self.constraint.runtime_parser_dwa_transition(state, parser),
                    |event| {
                        match event {
                            StackWalkEvent::Final(final_weight) => {
                                used_direct_final = true;
                                let dense = if dense_is_seed {
                                    seed_base.as_ref()
                                } else {
                                    single_path_acc.as_slice()
                                };
                                self.merge_single_path_final_weight_to_internal(
                                    final_weight,
                                    internal_tsid,
                                    dense,
                                    precomputed,
                                    &mut merged,
                                    Some(&mut *buf),
                                    &mut direct_buf_dirty,
                                );
                                if final_weight.is_full() { return false; }
                            }
                            StackWalkEvent::Top(parser_state) => {
                                let positive_label = encode_positive_label(parser_state);
                                let dense = if dense_is_seed {
                                    seed_base.as_ref()
                                } else {
                                    single_path_acc.as_slice()
                                };
                                let mut used_equivalent_wide_summary = false;
                                if let Some(summary) = self
                                    .constraint
                                    .direct_regular_wide_acceptance_for_parser_state(parser_state)
                                    && let Some(accepted) = summary.dense_by_tsid.get(internal_tsid)
                                {
                                    let n = dense.len().min(accepted.len()).min(merged.len());
                                    for word in 0..n {
                                        merged[word] |= dense[word] & accepted[word];
                                    }
                                    used_direct_final = true;
                                    used_equivalent_wide_summary = true;
                                }

                                if !used_equivalent_wide_summary {
                                    if let Some(accept_weight) =
                                        self.constraint.runtime_parser_top_accept(positive_label)
                                    {
                                        used_direct_final = true;
                                        self.merge_single_path_final_weight_to_internal(
                                            accept_weight,
                                            internal_tsid,
                                            dense,
                                            precomputed,
                                            &mut merged,
                                            Some(&mut *buf),
                                            &mut direct_buf_dirty,
                                        );
                                    }
                                    let accept_parts = self
                                        .constraint
                                        .runtime_parser_top_accept_parts(positive_label);
                                    if !accept_parts.is_empty() {
                                        used_direct_final = true;
                                        for accept_weight in accept_parts {
                                            self.merge_single_path_final_weight_to_internal(
                                                accept_weight,
                                                internal_tsid,
                                                dense,
                                                precomputed,
                                                &mut merged,
                                                Some(&mut *buf),
                                                &mut direct_buf_dirty,
                                            );
                                        }
                                    }
                                    let used_l1 =
                                        self.constraint.for_each_direct_regular_l1_acceptance(
                                            parser_state,
                                            |accept_weight| {
                                                self.merge_single_path_final_weight_to_internal(
                                                    accept_weight,
                                                    internal_tsid,
                                                    dense,
                                                    precomputed,
                                                    &mut merged,
                                                    Some(&mut *buf),
                                                    &mut direct_buf_dirty,
                                                );
                                            },
                                        );
                                    used_direct_final |= used_l1;
                                }
                            }
                            StackWalkEvent::Intersect(weight) => {
                                if dense_is_seed {
                                    if !weight.is_full() {
                                        if !materialize_single_path_seed_intersection(
                                            seed_base,
                                            &mut single_path_acc,
                                            internal_tsid,
                                            weight,
                                            self.constraint,
                                        ) {
                                            return false;
                                        }
                                        dense_is_seed = false;
                                    }
                                } else if !Self::intersect_single_path_dense_with_weight_in_place(
                                    &mut single_path_acc,
                                    &mut single_path_aux,
                                    internal_tsid,
                                    weight,
                                    self.constraint,
                                ) {
                                    return false;
                                }
                            }
                        }
                        true
                    },
                );
            }
        }
        if !used_direct_final && !self.is_accepting() {
            restore_scratch(merged, output_scratch, single_path_aux, single_path_acc);
            return false;
        }

        let reused_dense = !direct_buf_dirty
            && self.try_replay_monotone_dense_cache(&merged, buf);
        if !reused_dense && merged.iter().any(|&word| word != 0) {
            let buf_zeroed = !direct_buf_dirty;
            self.constraint.or_internal_dense_to_buf_fast_with_scratch(
                &merged,
                buf,
                buf_zeroed,
                &mut output_scratch,
            );
        }
        if direct_buf_dirty {
            self.store_mask_cache_reuse_dense(buf);
        } else {
            self.store_mask_cache(buf, &merged);
        }
        restore_scratch(merged, output_scratch, single_path_aux, single_path_acc);
        true
    }

    /// Reuse only a cached pure internal-token projection. All additional
    /// output contributions clear the cached bitmap, and the caller declines
    /// this path when it has already written direct final-mask contributions.
    /// Equality or an addition-only bitmap proves that old output bits remain
    /// valid; any removal declines before the output is touched.
    fn try_replay_monotone_dense_cache(&self, merged: &[u64], buf: &mut [u32]) -> bool {
        if self.constraint.final_mask_mapping.internal_len() != 0 || merged.is_empty() {
            return false;
        }
        let count = self.constraint.internal_token_count();
        if count == 0 || merged.len() != count.div_ceil(64) {
            return false;
        }
        let cache = self.mask_cache.lock().unwrap();
        let Some(previous) = cache.as_ref().filter(|previous| {
            previous.mask.len() == buf.len() && previous.merged_dense.len() == merged.len()
        }) else {
            return false;
        };
        let tail_bits = count % 64;
        if tail_bits != 0
            && ((merged[merged.len() - 1] | previous.merged_dense[merged.len() - 1])
                >> tail_bits) != 0
        {
            return false;
        }
        if previous.merged_dense == merged {
            buf.copy_from_slice(&previous.mask);
            return true;
        }

        // Prove the complete subset relation and bound new work before
        // pricing any aliases. Broad growth uses the ordinary exact rebuild.
        if !bounded_monotone_growth(&previous.merged_dense, merged) {
            return false;
        }
        let mut added_cost = 0u64;
        for (wi, (&current, &old)) in merged.iter().zip(&previous.merged_dense).enumerate() {
            let added = current & !old;
            if added == 0 {
                continue;
            }
            let remaining = count - wi * 64;
            let valid = if remaining >= 64 { u64::MAX } else { (1u64 << remaining) - 1 };
            added_cost = added_cost.saturating_add(
                self.constraint.internal_bits_grouped_buf_op_cost(wi, added, valid, buf.len())
                    as u64,
            );
        }
        // Keep the established conservative delta-versus-rebuild margin.
        // These are work estimates; no mask admission depends on their value.
        let delta_cost = (buf.len() as u64).saturating_add(added_cost);
        let rebuild_cost = self.constraint.estimate_internal_dense_to_buf_cost(merged);
        if rebuild_cost.saturating_sub(delta_cost) <= DELTA_SEED_MIN_SAVINGS
            || delta_cost.saturating_mul(2) >= rebuild_cost
        {
            return false;
        }
        buf.copy_from_slice(&previous.mask);
        self.constraint.apply_internal_dense_delta_to_buf(&previous.merged_dense, merged, buf);
        true
    }

    fn collect_single_path_mask_paths(&self, paths: &mut SinglePathMaskPaths) -> bool {
        if self.state.len() < MASK_SINGLE_PATH_DIRECT_TWO_PASS_MIN_STATE_COUNT {
            // Below half the path budget, accepted multipath states are common
            // and a separate counting traversal costs more than it saves. Keep
            // the original one-pass admission/materialization algorithm.
            for (&original_tokenizer_state, gss) in &self.state {
                if gss.max_depth() > MASK_SINGLE_PATH_DIRECT_MAX_DEPTH {
                    return false;
                }

                let mut stack =
                    SmallVec::<[u32; MASK_SINGLE_PATH_DIRECT_INLINE_STACK_DEPTH]>::new();
                if let Some(terminals_disallowed) = gss.single_path_top_first_and_acc(&mut stack) {
                    paths.push((original_tokenizer_state, terminals_disallowed, stack));
                    continue;
                }

                if mask_single_path_to_stacks_fallback_disabled() {
                    return false;
                }
                let remaining = MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS.saturating_sub(paths.len())
                    .min(MASK_SINGLE_PATH_DIRECT_MAX_PATHS_PER_GSS);
                let complete = gss.for_each_stack_top_first_bounded(
                    remaining,
                    |stack_top_first, terminals_disallowed| {
                        let mut path_stack =
                            SmallVec::<[u32; MASK_SINGLE_PATH_DIRECT_INLINE_STACK_DEPTH]>::new();
                        path_stack.extend(stack_top_first.iter().copied());
                        paths.push((
                            original_tokenizer_state,
                            terminals_disallowed.clone(),
                            path_stack,
                        ));
                    },
                );
                if !complete {
                    return false;
                }
            }
        } else {
            // Once active tokenizer states consume at least half the path
            // budget, a small amount of branching is likely to reject the
            // specialized kernel. Count without cloning stack values first;
            // only accepted states pay to materialize concrete stacks.
            let mut all_single_path = true;
            for (&original_tokenizer_state, gss) in &self.state {
                if gss.max_depth() > MASK_SINGLE_PATH_DIRECT_MAX_DEPTH {
                    return false;
                }

                let mut stack =
                    SmallVec::<[u32; MASK_SINGLE_PATH_DIRECT_INLINE_STACK_DEPTH]>::new();
                let Some(terminals_disallowed) = gss.single_path_top_first_and_acc(&mut stack)
                else {
                    all_single_path = false;
                    break;
                };
                paths.push((original_tokenizer_state, terminals_disallowed, stack));
            }

            if !all_single_path {
                if mask_single_path_to_stacks_fallback_disabled() {
                    return false;
                }

                let mut total_paths = 0usize;
                let mut total_stack_values = 0usize;
                for gss in self.state.values() {
                    if gss.max_depth() > MASK_SINGLE_PATH_DIRECT_MAX_DEPTH {
                        return false;
                    }
                    let remaining =
                        MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS.saturating_sub(total_paths)
                            .min(MASK_SINGLE_PATH_DIRECT_MAX_PATHS_PER_GSS);
                    let complete = gss.for_each_stack_len_bounded(remaining, |stack_len, _| {
                        total_paths += 1;
                        total_stack_values = total_stack_values.saturating_add(stack_len);
                    });
                    if !complete
                        || total_stack_values > MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_STACK_VALUES
                    {
                        return false;
                    }
                }

                paths.clear();
                for (&original_tokenizer_state, gss) in &self.state {
                    let remaining =
                        MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS.saturating_sub(paths.len())
                    .min(MASK_SINGLE_PATH_DIRECT_MAX_PATHS_PER_GSS);
                    let complete = gss.for_each_stack_top_first_bounded(
                        remaining,
                        |stack_top_first, terminals_disallowed| {
                            let mut path_stack = SmallVec::<
                                [u32; MASK_SINGLE_PATH_DIRECT_INLINE_STACK_DEPTH],
                            >::new();
                            path_stack.extend(stack_top_first.iter().copied());
                            paths.push((
                                original_tokenizer_state,
                                terminals_disallowed.clone(),
                                path_stack,
                            ));
                        },
                    );
                    debug_assert!(
                        complete,
                        "admitted GSS must materialize within the path budget"
                    );
                    if !complete {
                        return false;
                    }
                }
            }
        }
        if paths.iter().any(|(tokenizer_state, _, _)| {
            self.constraint
                .internal_tsids_for_state(*tokenizer_state)
                .len()
                != 1
        }) {
            return false;
        }
        true
    }

    /// Build a program only when repeated stack traversal outweighs unique work.
    /// A declined plan never writes output; the ordinary stack walk remains authoritative.
    #[inline(never)]
    fn with_repeated_stack_plans<'c>(
        &'c self,
        paths: &[SinglePathMaskPath],
        total_stack_values: usize,
        apply: impl FnOnce(&[SinglePathDirectPlanOp<'c>], &[SinglePathDirectStackPlan], &[u8]),
    ) -> bool {
        let mut plan_ops =
            SmallVec::<[SinglePathDirectPlanOp<'_>; MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS]>::new();
        let mut stack_plans = SmallVec::<
            [SinglePathDirectStackPlan; MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY],
        >::new();
        let mut path_plan_indices =
            SmallVec::<[u8; MASK_SINGLE_PATH_DIRECT_INLINE_PATH_CAPACITY]>::new();
        let mut repeated_stack_values = 0usize;

        // Planning pays only when repeated parser-stack traversal dominates the
        // unique work.  This replaces the old 32-path switch with a direct cost
        // comparison: three copies of one deep stack can qualify, while 64
        // unrelated stacks do not build programs merely because the frontier is
        // wide.  First group stacks without touching the parser DWA; compile
        // programs only after the reuse test succeeds.
        if paths.len() >= 3 {
            for (path_index, (_, _, stack)) in paths.iter().enumerate() {
                let stack_fingerprint = single_path_direct_stack_fingerprint(stack);
                let existing = stack_plans.iter().position(|plan| {
                    plan.stack_fingerprint == stack_fingerprint
                        && paths[plan.representative_path].2.as_slice() == stack.as_slice()
                });
                let plan_index = if let Some(existing) = existing {
                    repeated_stack_values = repeated_stack_values.saturating_add(stack.len());
                    existing
                } else {
                    let plan_index = stack_plans.len();
                    stack_plans.push(SinglePathDirectStackPlan {
                        representative_path: path_index,
                        stack_fingerprint,
                        ops_start: 0,
                        ops_end: 0,
                    });
                    plan_index
                };
                path_plan_indices.push(plan_index as u8);
            }
        }

        let should_build_stack_plans = path_plan_indices.len() == paths.len()
            && single_path_direct_plan_reuse_dominates(
                paths.len(),
                total_stack_values,
                repeated_stack_values,
            );
        let mut plans_complete = should_build_stack_plans;
        if should_build_stack_plans {
            'build_plans: for plan_index in 0..stack_plans.len() {
                let representative_path = stack_plans[plan_index].representative_path;
                let stack = &paths[representative_path].2;
                let ops_start = plan_ops.len();
                walk_single_stack::<true, _>(
                    self.constraint.runtime_parser_dwa_start_state(),
                    stack,
                    |state| self.constraint.runtime_parser_dwa_final_weight(state),
                    |state, parser| self.constraint.runtime_parser_dwa_transition(state, parser),
                    |event| {
                        match event {
                            StackWalkEvent::Final(final_weight) => {
                                if plan_ops.len() == MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS {
                                    plans_complete = false;
                                    return false;
                                }
                                plan_ops.push(SinglePathDirectPlanOp::Merge(final_weight));
                                // Later intersections cannot add tokens to this path.
                                if final_weight.is_full() { return false; }
                            }
                            StackWalkEvent::Top(parser_state) => {
                                let positive_label = encode_positive_label(parser_state);
                                let has_direct_regular_acceptance = self
                                    .constraint
                                    .direct_regular_wide_acceptance_for_parser_state(parser_state)
                                    .is_some()
                                    || self.constraint.for_each_direct_regular_l1_acceptance(
                                        parser_state,
                                        |_| {},
                                    );
                                if has_direct_regular_acceptance {
                                    plans_complete = false;
                                    return false;
                                }
                                if let Some(accept_weight) =
                                    self.constraint.runtime_parser_top_accept(positive_label)
                                {
                                    if plan_ops.len() == MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS {
                                        plans_complete = false;
                                        return false;
                                    }
                                    plan_ops.push(SinglePathDirectPlanOp::Merge(accept_weight));
                                }
                                let accept_parts = self
                                    .constraint
                                    .runtime_parser_top_accept_parts(positive_label);
                                if !accept_parts.is_empty() {
                                    for accept_weight in accept_parts {
                                        if plan_ops.len() == MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS {
                                            plans_complete = false;
                                            return false;
                                        }
                                        plan_ops.push(SinglePathDirectPlanOp::Merge(accept_weight));
                                    }
                                }
                            }
                            StackWalkEvent::Intersect(weight) => {
                                if plan_ops.len() == MASK_SINGLE_PATH_DIRECT_MAX_PLAN_OPS {
                                    plans_complete = false;
                                    return false;
                                }
                                plan_ops.push(SinglePathDirectPlanOp::Intersect(weight));
                            }
                        }
                        true
                    },
                );
                if !plans_complete {
                    break 'build_plans;
                }

                stack_plans[plan_index].ops_start = ops_start;
                stack_plans[plan_index].ops_end = plan_ops.len();
            }
        }
        if !plans_complete || !should_build_stack_plans {
            return false;
        }
        apply(&plan_ops, &stack_plans, &path_plan_indices);
        true
    }
    fn fill_single_path_seed_dense(
        &self,
        terminals_disallowed: &TerminalsDisallowed,
        aux: &mut Vec<u64>,
        dense: &mut Vec<u64>,
    ) -> bool {
        let base = &self.constraint.seed_universe_dense;
        if base.is_empty() {
            dense.clear();
            return false;
        }

        dense.clear();
        dense.extend_from_slice(base);

        if terminals_disallowed.is_empty() {
            return true;
        }

        self.fill_blocked_seed_dense(terminals_disallowed, aux);

        if aux.iter().all(|&word| word == 0) {
            return true;
        }

        let mut any = false;
        for (allowed_word, blocked_word) in dense.iter_mut().zip(aux.iter().copied()) {
            *allowed_word &= !blocked_word;
            any |= *allowed_word != 0;
        }
        any
    }

    fn intersect_single_path_dense_with_weight_in_place(
        dense: &mut Vec<u64>,
        aux: &mut Vec<u64>,
        internal_tsid: u32,
        weight: RuntimeWeightRef<'_>,
        constraint: &Constraint,
    ) -> bool {
        intersect_static_dense_with_weight(dense, aux, internal_tsid, weight, |tokens| {
            constraint.runtime_token_set_dense_mask(tokens)
        })
    }

    fn merge_single_path_final_weight_to_internal(
        &self,
        final_weight: RuntimeWeightRef<'_>,
        internal_tsid: u32,
        dense: &[u64],
        precomputed: &DenseTokenMaskCache,
        merged: &mut [u64],
        direct_buf: Option<&mut [u32]>,
        direct_buf_dirty: &mut bool,
    ) -> bool {
        if final_weight.is_full() {
            for (output, &word) in merged.iter_mut().zip(dense) {
                *output |= word;
            }
            return false;
        }
        let Some(tokens) = final_weight.token_set_for_tsid(internal_tsid) else {
            return true;
        };
        self.merge_final_token_set(
            tokens,
            dense,
            precomputed,
            merged,
            direct_buf,
            direct_buf_dirty,
        )
    }
}

#[cfg(test)]
mod tests;
