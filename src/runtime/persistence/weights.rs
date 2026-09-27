//! Packed weight-pool construction and non-DWA weight attachment.

use super::{BTreeMap, Constraint, FxHashMap, FxHashSet, Weight};

pub(super) fn constraint_serialized_weight_pool_with_ids(
    constraint: &Constraint,
) -> (Vec<Weight>, Vec<u32>, usize) {
    let mut by_ptr = FxHashMap::<usize, u32>::default();
    let mut weights = Vec::new();
    let mut ids = Vec::new();
    let mut total_ranges = 0usize;
    let mut push = |weight: &Weight| {
        let key = weight.ptr_key();
        let id = if let Some(&id) = by_ptr.get(&key) {
            id
        } else {
            let id = weights.len() as u32;
            by_ptr.insert(key, id);
            if !weight.is_full() {
                total_ranges = total_ranges.saturating_add(weight.num_ranges());
            }
            weights.push(weight.clone());
            id
        };
        ids.push(id);
    };

    for weight in constraint.parser_top_accept.values() {
        push(weight);
    }
    for parts in constraint.parser_top_accept_parts.values() {
        for weight in parts {
            push(weight);
        }
    }
    for weight in constraint.direct_regular_l1_complete_by_terminal.values() {
        push(weight);
    }
    for weight in constraint.possible_matches.values() {
        push(weight);
    }
    (weights, ids, total_ranges)
}

pub(super) fn constraint_serialized_weight_pool(constraint: &Constraint) -> Vec<Weight> {
    constraint_serialized_weight_pool_with_ids(constraint).0
}

/// IDs in the same field/map order used by ConstraintSerde. Structural Weight
/// values are placeholders when a packed non-DWA pool is present; interning
/// those empty values would collapse distinct references to the same pool ID.
pub(super) fn packed_constraint_serialized_weight_ids(constraint: &Constraint) -> Option<Vec<u32>> {
    let packed = constraint.packed_non_dwa_weights.as_ref()?;
    let mut ids = Vec::new();
    for key in constraint.parser_top_accept.keys() {
        ids.push(packed.parser_top_accept[key]);
    }
    for (key, parts) in &constraint.parser_top_accept_parts {
        let packed_parts = &packed.parser_top_accept_parts[key];
        assert_eq!(parts.len(), packed_parts.len(), "packed acceptance-part count mismatch");
        ids.extend_from_slice(packed_parts);
    }
    for key in constraint.direct_regular_l1_complete_by_terminal.keys() {
        ids.push(packed.direct_regular_l1_complete_by_terminal[key]);
    }
    for key in constraint.possible_matches.keys() {
        ids.push(packed.possible_matches[key]);
    }
    Some(ids)
}

pub(super) fn constraint_serialized_weight_ranges_at_least(
    constraint: &Constraint,
    min_weight_ranges: usize,
) -> bool {
    if min_weight_ranges == 0 {
        return true;
    }
    let mut seen = FxHashSet::<usize>::default();
    let mut total_ranges = 0usize;
    let mut crosses_threshold = |weight: &Weight| {
        if !seen.insert(weight.ptr_key()) || weight.is_full() {
            return false;
        }
        total_ranges = total_ranges.saturating_add(weight.num_ranges());
        total_ranges >= min_weight_ranges
    };

    for weight in constraint.parser_top_accept.values() {
        if crosses_threshold(weight) {
            return true;
        }
    }
    for parts in constraint.parser_top_accept_parts.values() {
        for weight in parts {
            if crosses_threshold(weight) {
                return true;
            }
        }
    }
    for weight in constraint.direct_regular_l1_complete_by_terminal.values() {
        if crosses_threshold(weight) {
            return true;
        }
    }
    for weight in constraint.possible_matches.values() {
        if crosses_threshold(weight) {
            return true;
        }
    }
    false
}

pub(super) fn compact_non_dwa_weight_runtime_if_at_least(
    constraint: &mut Constraint,
    min_weight_ranges: usize,
) -> bool {
    if constraint.packed_non_dwa_weights.is_some() {
        return false;
    }
    if !constraint_serialized_weight_ranges_at_least(constraint, min_weight_ranges) {
        return false;
    }
    let (weights, ids, _) = constraint_serialized_weight_pool_with_ids(constraint);
    let wire = crate::ds::weight::pack_pooled_weights(&weights);
    let pool = crate::ds::weight::PackedRuntimeWeightPool::from_packed_bytes(&wire)
        .expect("fresh packed non-DWA Weight runtime should decode");
    attach_packed_non_dwa_weights(constraint, std::sync::Arc::new(pool), ids)
        .expect("fresh packed non-DWA Weight ids should match runtime maps");
    true
}

pub(crate) fn compact_large_non_dwa_weight_runtime(constraint: &mut Constraint) -> bool {
    const MIN_WEIGHT_RANGES: usize = 100_000;
    compact_non_dwa_weight_runtime_if_at_least(constraint, MIN_WEIGHT_RANGES)
}

pub(super) fn attach_packed_non_dwa_weights(
    constraint: &mut Constraint,
    pool: std::sync::Arc<crate::ds::weight::PackedRuntimeWeightPool>,
    ids: Vec<u32>,
) -> Result<(), String> {
    let top_len = constraint.parser_top_accept.len();
    let parts_len = constraint
        .parser_top_accept_parts
        .values()
        .map(Vec::len)
        .sum::<usize>();
    let direct_len = constraint.direct_regular_l1_complete_by_terminal.len();
    let possible_len = constraint.possible_matches.len();
    let expected = top_len
        .checked_add(parts_len)
        .and_then(|value| value.checked_add(direct_len))
        .and_then(|value| value.checked_add(possible_len))
        .ok_or_else(|| "packed Weight id count overflow".to_owned())?;
    if ids.len() != expected {
        return Err(format!(
            "packed Weight id count mismatch: expected {expected}, found {}",
            ids.len(),
        ));
    }

    let (top_ids, rest) = ids.split_at(top_len);
    let (part_ids, rest) = rest.split_at(parts_len);
    let (direct_ids, possible_ids) = rest.split_at(direct_len);

    let build_small = || {
        let parser_top_accept = constraint
            .parser_top_accept
            .keys()
            .copied()
            .zip(top_ids.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let mut part_pos = 0usize;
        let parser_top_accept_parts = constraint
            .parser_top_accept_parts
            .iter()
            .map(|(&label, parts)| {
                let end = part_pos + parts.len();
                let ids = part_ids[part_pos..end].to_vec();
                part_pos = end;
                (label, ids)
            })
            .collect::<BTreeMap<_, _>>();
        debug_assert_eq!(part_pos, part_ids.len());
        (parser_top_accept, parser_top_accept_parts)
    };
    let build_direct = || {
        constraint
            .direct_regular_l1_complete_by_terminal
            .keys()
            .copied()
            .zip(direct_ids.iter().copied())
            .collect::<BTreeMap<_, _>>()
    };
    let build_possible = || {
        constraint
            .possible_matches
            .keys()
            .copied()
            .zip(possible_ids.iter().copied())
            .collect::<BTreeMap<_, _>>()
    };
    let ((parser_top_accept, parser_top_accept_parts), (direct_regular_l1_complete_by_terminal, possible_matches)) =
        if expected >= 1_024 && rayon::current_num_threads() > 1 {
            rayon::join(build_small, || rayon::join(build_direct, build_possible))
        } else {
            (build_small(), (build_direct(), build_possible()))
        };

    constraint.packed_non_dwa_weights = Some(std::sync::Arc::new(
        crate::runtime::artifact::PackedNonDwaWeights {
            pool,
            parser_top_accept,
            parser_top_accept_parts,
            direct_regular_l1_complete_by_terminal,
            possible_matches,
        },
    ));
    Ok(())
}
