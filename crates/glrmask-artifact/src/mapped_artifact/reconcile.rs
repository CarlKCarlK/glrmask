use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::{CheckSortedDisjointMap, RangeMapBlaze, RangeSetBlaze};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::weight::{SharedTokenSet, Weight, finalize_weight_map, shared_rangeset};

pub(super) fn reconcile_weight_id_maps(
    left_weights: &mut [&mut Weight],
    left_id_map: &mut InternalIdMap,
    right_weights: &mut [&mut Weight],
    right_id_map: &mut InternalIdMap,
) {
    if try_reconcile_refinement_fast_path(left_weights, left_id_map, right_weights, right_id_map) {
        return;
    }

    let common_id_map = build_common_internal_id_map(&[left_id_map, right_id_map]);

    let left_tsid_map = build_local_to_common_tsid_map(left_id_map, &common_id_map);
    let left_token_map = build_local_to_common_token_map_from_common_classes(left_id_map, &common_id_map);
    let right_tsid_map = build_local_to_common_tsid_map(right_id_map, &common_id_map);
    let right_token_map = build_local_to_common_token_map_from_common_classes(right_id_map, &common_id_map);

    remap_weights_with_maps(
        left_weights,
        &left_tsid_map,
        &left_token_map,
        common_id_map.num_tsids() as usize,
    );
    remap_weights_with_maps(
        right_weights,
        &right_tsid_map,
        &right_token_map,
        common_id_map.num_tsids() as usize,
    );

    *left_id_map = common_id_map.clone();
    *right_id_map = common_id_map;
}

pub(super) fn reconcile_weight_id_maps_into_common(
    left_weights: &mut [&mut Weight],
    left_id_map: &InternalIdMap,
    right_weights: &mut [&mut Weight],
    right_id_map: &InternalIdMap,
) -> InternalIdMap {
    if right_weights.is_empty() {
        return left_id_map.clone();
    }
    if left_weights.is_empty() {
        return right_id_map.clone();
    }
    if internal_id_map_refines(left_id_map, right_id_map) {
        remap_weights_into_existing_common(right_weights, right_id_map, left_id_map);
        return left_id_map.clone();
    }
    if internal_id_map_refines(right_id_map, left_id_map) {
        remap_weights_into_existing_common(left_weights, left_id_map, right_id_map);
        return right_id_map.clone();
    }

    reconcile_weight_id_maps_into_forced_common(
        left_weights,
        left_id_map,
        right_weights,
        right_id_map,
    )
}

pub(super) fn reconcile_weight_id_maps_into_forced_common(
    left_weights: &mut [&mut Weight],
    left_id_map: &InternalIdMap,
    right_weights: &mut [&mut Weight],
    right_id_map: &InternalIdMap,
) -> InternalIdMap {
    let profiling = std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some();
    let total_started_at = profiling.then(Instant::now);
    let build_started_at = profiling.then(Instant::now);
    let common_id_map = build_common_internal_id_map(&[left_id_map, right_id_map]);
    let build_ms = build_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let maps_started_at = profiling.then(Instant::now);
    let left_tsid_map = build_local_to_common_tsid_map(left_id_map, &common_id_map);
    let left_token_map = build_local_to_common_token_map_from_common_classes(left_id_map, &common_id_map);
    let right_tsid_map = build_local_to_common_tsid_map(right_id_map, &common_id_map);
    let right_token_map = build_local_to_common_token_map_from_common_classes(right_id_map, &common_id_map);
    let maps_ms = maps_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let common_tsid_count = common_id_map.num_tsids() as usize;
    let parallel_remap = std::env::var("GLRMASK_PARALLEL_FORCED_COMMON_REMAP")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    let ((left_ms, ()), (right_ms, ())) = if parallel_remap {
        rayon::join(
            || {
                let started = profiling.then(Instant::now);
                remap_weights_with_maps(
                    left_weights,
                    &left_tsid_map,
                    &left_token_map,
                    common_tsid_count,
                );
                (
                    started
                        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0),
                    (),
                )
            },
            || {
                let started = profiling.then(Instant::now);
                remap_weights_with_maps(
                    right_weights,
                    &right_tsid_map,
                    &right_token_map,
                    common_tsid_count,
                );
                (
                    started
                        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0),
                    (),
                )
            },
        )
    } else {
        let left_started = profiling.then(Instant::now);
        remap_weights_with_maps(
            left_weights,
            &left_tsid_map,
            &left_token_map,
            common_tsid_count,
        );
        let left_ms = left_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let right_started = profiling.then(Instant::now);
        remap_weights_with_maps(
            right_weights,
            &right_tsid_map,
            &right_token_map,
            common_tsid_count,
        );
        let right_ms = right_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        ((left_ms, ()), (right_ms, ()))
    };

    if let Some(total_started_at) = total_started_at {
        let fanout = |map: &[Vec<u32>]| {
            let total = map.iter().map(Vec::len).sum::<usize>();
            let multi = map.iter().filter(|targets| targets.len() > 1).count();
            let maximum = map.iter().map(Vec::len).max().unwrap_or(0);
            (total, multi, maximum)
        };
        let (left_tsid_total, left_tsid_multi, left_tsid_max) = fanout(&left_tsid_map);
        let (right_tsid_total, right_tsid_multi, right_tsid_max) = fanout(&right_tsid_map);
        let (left_token_total, left_token_multi, left_token_max) = fanout(&left_token_map);
        let (right_token_total, right_token_multi, right_token_max) = fanout(&right_token_map);
        eprintln!(
            "[glrmask/profile][forced_common_reconcile] left_weights={} right_weights={} left_tsids={} right_tsids={} common_tsids={} left_tokens={} right_tokens={} common_tokens={} build_ms={:.3} maps_ms={:.3} left_remap_ms={:.3} right_remap_ms={:.3} total_ms={:.3} left_tsid_fanout={}/{}/{} right_tsid_fanout={}/{}/{} left_token_fanout={}/{}/{} right_token_fanout={}/{}/{}",
            left_weights.len(),
            right_weights.len(),
            left_id_map.num_tsids(),
            right_id_map.num_tsids(),
            common_id_map.num_tsids(),
            left_id_map.num_internal_tokens(),
            right_id_map.num_internal_tokens(),
            common_id_map.num_internal_tokens(),
            build_ms,
            maps_ms,
            left_ms,
            right_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
            left_tsid_total,
            left_tsid_multi,
            left_tsid_max,
            right_tsid_total,
            right_tsid_multi,
            right_tsid_max,
            left_token_total,
            left_token_multi,
            left_token_max,
            right_token_total,
            right_token_multi,
            right_token_max,
        );
    }

    common_id_map
}

fn try_reconcile_refinement_fast_path(
    left_weights: &mut [&mut Weight],
    left_id_map: &mut InternalIdMap,
    right_weights: &mut [&mut Weight],
    right_id_map: &mut InternalIdMap,
) -> bool {
    if right_weights.is_empty() {
        *right_id_map = left_id_map.clone();
        return true;
    }
    if left_weights.is_empty() {
        *left_id_map = right_id_map.clone();
        return true;
    }
    if internal_id_map_refines(left_id_map, right_id_map) {
        remap_weights_into_existing_common(right_weights, right_id_map, left_id_map);
        *right_id_map = left_id_map.clone();
        return true;
    }
    if internal_id_map_refines(right_id_map, left_id_map) {
        remap_weights_into_existing_common(left_weights, left_id_map, right_id_map);
        *left_id_map = right_id_map.clone();
        return true;
    }
    false
}

pub(super) fn remap_weights_into_existing_common(
    weights: &mut [&mut Weight],
    local_id_map: &InternalIdMap,
    common_id_map: &InternalIdMap,
) {
    if weights.is_empty() {
        return;
    }
    let tsid_map = build_local_to_common_tsid_map(local_id_map, common_id_map);
    let token_map = build_local_to_common_token_map_from_common_classes(local_id_map, common_id_map);
    remap_weights_with_maps(
        weights,
        &tsid_map,
        &token_map,
        common_id_map.num_tsids() as usize,
    );
}

pub(super) fn internal_id_map_refines(finer: &InternalIdMap, coarser: &InternalIdMap) -> bool {
    many_to_one_id_map_refines(&finer.tokenizer_states, &coarser.tokenizer_states, false)
        && many_to_one_id_map_refines(&finer.vocab_tokens, &coarser.vocab_tokens, true)
}

fn many_to_one_id_map_refines(
    finer: &ManyToOneIdMap,
    coarser: &ManyToOneIdMap,
    allow_unmapped: bool,
) -> bool {
    if finer.original_to_internal.len() < coarser.original_to_internal.len() {
        return false;
    }

    let mut coarser_by_finer = vec![None; finer.internal_to_originals.len()];
    for original in 0..finer.original_to_internal.len() {
        let finer_internal = finer.original_to_internal[original];
        let coarser_internal = coarser
            .original_to_internal
            .get(original)
            .copied()
            .unwrap_or(u32::MAX);
        if allow_unmapped && finer_internal == u32::MAX && coarser_internal == u32::MAX {
            continue;
        }
        let Some(slot) = coarser_by_finer.get_mut(finer_internal as usize) else {
            return false;
        };
        match slot {
            Some(previous) if *previous != coarser_internal => return false,
            Some(_) => {}
            None => *slot = Some(coarser_internal),
        }
    }

    true
}

pub(super) fn build_common_internal_id_map(inputs: &[&InternalIdMap]) -> InternalIdMap {
    build_common_internal_id_map_impl(inputs, false)
}

pub(super) fn build_common_internal_id_map_preserving_unmapped_tokenizer(
    inputs: &[&InternalIdMap],
) -> InternalIdMap {
    build_common_internal_id_map_impl(inputs, true)
}

fn build_common_internal_id_map_impl(
    inputs: &[&InternalIdMap],
    allow_unmapped_tokenizer: bool,
) -> InternalIdMap {
    let num_tokenizer_states = inputs
        .iter()
        .map(|input| input.tokenizer_states.original_to_internal.len())
        .max()
        .unwrap_or(0);
    let num_original_tokens = inputs
        .iter()
        .map(|input| input.vocab_tokens.original_to_internal.len())
        .max()
        .unwrap_or(0);

    let parallel_dimensions = std::env::var("GLRMASK_PARALLEL_COMMON_ID_DIMENSIONS")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    let (tokenizer_states, vocab_tokens) = if parallel_dimensions {
        rayon::join(
            || {
                build_common_many_to_one_id_map(
                    inputs,
                    num_tokenizer_states,
                    |input| &input.tokenizer_states,
                    allow_unmapped_tokenizer,
                )
            },
            || {
                build_common_many_to_one_id_map(
                    inputs,
                    num_original_tokens,
                    |input| &input.vocab_tokens,
                    true,
                )
            },
        )
    } else {
        (
            build_common_many_to_one_id_map(
                inputs,
                num_tokenizer_states,
                |input| &input.tokenizer_states,
                allow_unmapped_tokenizer,
            ),
            build_common_many_to_one_id_map(
                inputs,
                num_original_tokens,
                |input| &input.vocab_tokens,
                true,
            ),
        )
    };

    InternalIdMap {
        tokenizer_states,
        vocab_tokens,
        deferred_vocab_singleton_original_ids: None,
    }
}

fn build_common_many_to_one_id_map(
    inputs: &[&InternalIdMap],
    num_originals: usize,
    project: impl Fn(&InternalIdMap) -> &ManyToOneIdMap,
    allow_unmapped: bool,
) -> ManyToOneIdMap {
    if let [left, right] = inputs {
        return build_common_many_to_one_id_map_pair(
            project(left),
            project(right),
            num_originals,
            allow_unmapped,
        );
    }

    let mut composite_to_class: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut original_to_internal = vec![u32::MAX; num_originals];
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut representatives: Vec<u32> = Vec::new();

    for original in 0..num_originals {
        let composite: Vec<u32> = inputs
            .iter()
            .map(|input| {
                project(input)
                    .original_to_internal
                    .get(original)
                    .copied()
                    .unwrap_or(u32::MAX)
            })
            .collect();
        if allow_unmapped && composite.iter().all(|&value| value == u32::MAX) {
            continue;
        }

        let next_id = internal_to_originals.len() as u32;
        let class_id = *composite_to_class.entry(composite).or_insert_with(|| {
            internal_to_originals.push(Vec::new());
            representatives.push(original as u32);
            next_id
        });
        original_to_internal[original] = class_id;
        internal_to_originals[class_id as usize].push(original as u32);
    }

    reorder_common_classes(
        composite_to_class,
        &mut original_to_internal,
        &mut internal_to_originals,
        &mut representatives,
        allow_unmapped,
    );

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids: representatives,
    }
}

fn build_common_many_to_one_id_map_pair(
    left: &ManyToOneIdMap,
    right: &ManyToOneIdMap,
    num_originals: usize,
    allow_unmapped: bool,
) -> ManyToOneIdMap {
    let use_sparse_unmapped_pair = std::env::var("GLRMASK_SPARSE_UNMAPPED_COMMON_PAIR_MAP")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);
    if allow_unmapped && use_sparse_unmapped_pair {
        let direct_sparse = std::env::var("GLRMASK_DIRECT_SPARSE_COMMON_PAIR_MAP")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
        if direct_sparse {
            let direct = build_common_many_to_one_id_map_pair_direct_sparse(
                left,
                right,
                num_originals,
            );
            if std::env::var_os("GLRMASK_VALIDATE_DIRECT_SPARSE_COMMON_PAIR_MAP").is_some() {
                let reference = build_common_many_to_one_id_map_pair_sparse_unmapped(
                    left,
                    right,
                    num_originals,
                );
                assert_eq!(
                    direct.original_to_internal,
                    reference.original_to_internal,
                    "direct sparse common pair original map differs from reference",
                );
                assert_eq!(
                    direct.internal_to_originals,
                    reference.internal_to_originals,
                    "direct sparse common pair classes differ from reference",
                );
                assert_eq!(
                    direct.representative_original_ids,
                    reference.representative_original_ids,
                    "direct sparse common pair representatives differ from reference",
                );
            }
            return direct;
        }
        return build_common_many_to_one_id_map_pair_sparse_unmapped(
            left,
            right,
            num_originals,
        );
    }
    build_common_many_to_one_id_map_pair_hashed(left, right, num_originals, allow_unmapped)
}

fn build_common_many_to_one_id_map_pair_direct_sparse(
    left: &ManyToOneIdMap,
    right: &ManyToOneIdMap,
    num_originals: usize,
) -> ManyToOneIdMap {
    let right_classes = right.internal_to_originals.len();
    let mut original_to_internal = vec![u32::MAX; num_originals];
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representatives = Vec::<u32>::new();

    // Composite pair order is lexicographic `(left, right)`, with the unmapped
    // sentinel sorting after every real class. Emit exactly that order directly.
    let mut by_right = (0..right_classes).map(|_| Vec::<u32>::new()).collect::<Vec<_>>();
    let mut touched_right = Vec::<usize>::new();
    let mut left_only = Vec::<u32>::new();

    let mut emit = |mut originals: Vec<u32>| {
        if originals.is_empty() {
            return;
        }
        // Most maps are already canonical and this branch is just one linear
        // check. Some independently/parallel-built artifacts can arrive with
        // an equivalent class whose inverse list is not in strict ID order.
        // The common-map relation is set-valued, so canonicalize that rare
        // input locally instead of making the direct sparse fast path depend
        // on incidental inverse-list construction order.
        if !originals.windows(2).all(|pair| pair[0] < pair[1]) {
            originals.sort_unstable();
            originals.dedup();
        }
        let class = internal_to_originals.len() as u32;
        for &original in &originals {
            original_to_internal[original as usize] = class;
        }
        representatives.push(originals[0]);
        internal_to_originals.push(originals);
    };

    for left_originals in &left.internal_to_originals {
        touched_right.clear();
        left_only.clear();
        for &original in left_originals {
            let right_internal = right
                .original_to_internal
                .get(original as usize)
                .copied()
                .unwrap_or(u32::MAX);
            if right_internal == u32::MAX {
                left_only.push(original);
            } else {
                let slot = &mut by_right[right_internal as usize];
                if slot.is_empty() {
                    touched_right.push(right_internal as usize);
                }
                slot.push(original);
            }
        }
        touched_right.sort_unstable();
        for right_class in touched_right.drain(..) {
            emit(std::mem::take(&mut by_right[right_class]));
        }
        emit(std::mem::take(&mut left_only));
    }

    // `(MAX, right)` classes follow every `(left, *)` class.
    for right_originals in &right.internal_to_originals {
        let mut right_only = Vec::new();
        for &original in right_originals {
            let left_internal = left
                .original_to_internal
                .get(original as usize)
                .copied()
                .unwrap_or(u32::MAX);
            if left_internal == u32::MAX {
                right_only.push(original);
            }
        }
        emit(right_only);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids: representatives,
    }
}

fn build_common_many_to_one_id_map_pair_sparse_unmapped(
    left: &ManyToOneIdMap,
    right: &ManyToOneIdMap,
    num_originals: usize,
) -> ManyToOneIdMap {
    let left_classes = left.internal_to_originals.len();
    let right_classes = right.internal_to_originals.len();
    let mut left_only_counts = vec![0usize; left_classes];
    let mut right_only_counts = vec![0usize; right_classes];
    let mut overlap_counts = FxHashMap::<(u32, u32), usize>::default();

    let pair = |original: usize| {
        let left_internal = left
            .original_to_internal
            .get(original)
            .copied()
            .unwrap_or(u32::MAX);
        let right_internal = right
            .original_to_internal
            .get(original)
            .copied()
            .unwrap_or(u32::MAX);
        (left_internal, right_internal)
    };

    for original in 0..num_originals {
        let (left_internal, right_internal) = pair(original);
        match (left_internal, right_internal) {
            (u32::MAX, u32::MAX) => {}
            (left, u32::MAX) => left_only_counts[left as usize] += 1,
            (u32::MAX, right) => right_only_counts[right as usize] += 1,
            (left, right) => *overlap_counts.entry((left, right)).or_insert(0) += 1,
        }
    }

    let mut overlap_keys = overlap_counts.keys().copied().collect::<Vec<_>>();
    overlap_keys.sort_unstable();

    let mut left_only_class = vec![u32::MAX; left_classes];
    let mut right_only_class = vec![u32::MAX; right_classes];
    let mut overlap_class = FxHashMap::<(u32, u32), u32>::default();
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representatives = Vec::<u32>::new();

    let mut overlap_cursor = 0usize;
    for left_class in 0..left_classes {
        while overlap_cursor < overlap_keys.len()
            && overlap_keys[overlap_cursor].0 as usize == left_class
        {
            let key = overlap_keys[overlap_cursor];
            let class = internal_to_originals.len() as u32;
            overlap_class.insert(key, class);
            internal_to_originals.push(Vec::with_capacity(overlap_counts[&key]));
            representatives.push(u32::MAX);
            overlap_cursor += 1;
        }
        let count = left_only_counts[left_class];
        if count != 0 {
            left_only_class[left_class] = internal_to_originals.len() as u32;
            internal_to_originals.push(Vec::with_capacity(count));
            representatives.push(u32::MAX);
        }
    }
    debug_assert_eq!(overlap_cursor, overlap_keys.len());
    for right_class in 0..right_classes {
        let count = right_only_counts[right_class];
        if count != 0 {
            right_only_class[right_class] = internal_to_originals.len() as u32;
            internal_to_originals.push(Vec::with_capacity(count));
            representatives.push(u32::MAX);
        }
    }

    let mut original_to_internal = vec![u32::MAX; num_originals];
    for (original, slot) in original_to_internal.iter_mut().enumerate() {
        let (left_internal, right_internal) = pair(original);
        let class = match (left_internal, right_internal) {
            (u32::MAX, u32::MAX) => continue,
            (left, u32::MAX) => left_only_class[left as usize],
            (u32::MAX, right) => right_only_class[right as usize],
            (left, right) => overlap_class[&(left, right)],
        };
        debug_assert_ne!(class, u32::MAX);
        *slot = class;
        let originals = &mut internal_to_originals[class as usize];
        if originals.is_empty() {
            representatives[class as usize] = original as u32;
        }
        originals.push(original as u32);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids: representatives,
    }
}

fn build_common_many_to_one_id_map_pair_hashed(
    left: &ManyToOneIdMap,
    right: &ManyToOneIdMap,
    num_originals: usize,
    allow_unmapped: bool,
) -> ManyToOneIdMap {
    let mut composite_to_class: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut original_to_internal = vec![u32::MAX; num_originals];
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut representatives: Vec<u32> = Vec::new();

    for (original, slot) in original_to_internal.iter_mut().enumerate() {
        let left_internal = left
            .original_to_internal
            .get(original)
            .copied()
            .unwrap_or(u32::MAX);
        let right_internal = right
            .original_to_internal
            .get(original)
            .copied()
            .unwrap_or(u32::MAX);
        if allow_unmapped && left_internal == u32::MAX && right_internal == u32::MAX {
            continue;
        }

        let composite = (left_internal, right_internal);
        let next_id = internal_to_originals.len() as u32;
        let class_id = *composite_to_class.entry(composite).or_insert_with(|| {
            internal_to_originals.push(Vec::new());
            representatives.push(original as u32);
            next_id
        });
        *slot = class_id;
        internal_to_originals[class_id as usize].push(original as u32);
    }

    reorder_common_pair_classes(
        composite_to_class,
        &mut original_to_internal,
        &mut internal_to_originals,
        &mut representatives,
        allow_unmapped,
    );

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids: representatives,
    }
}

fn reorder_common_pair_classes(
    composite_to_class: FxHashMap<(u32, u32), u32>,
    original_to_internal: &mut [u32],
    internal_to_originals: &mut Vec<Vec<u32>>,
    representatives: &mut Vec<u32>,
    allow_unmapped: bool,
) {
    let num_classes = internal_to_originals.len();
    if num_classes <= 1 {
        return;
    }

    let mut sorted: Vec<((u32, u32), u32)> = composite_to_class.into_iter().collect();
    sorted.sort_unstable_by_key(|(composite, _)| *composite);

    let mut old_to_new = vec![0u32; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        old_to_new[*old_id as usize] = new_id as u32;
    }

    for value in original_to_internal.iter_mut() {
        if *value == u32::MAX && allow_unmapped {
            continue;
        }
        *value = old_to_new[*value as usize];
    }

    let mut new_internal_to_originals = vec![Vec::new(); num_classes];
    let mut new_representatives = vec![u32::MAX; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        new_internal_to_originals[new_id] = std::mem::take(&mut internal_to_originals[*old_id as usize]);
        new_representatives[new_id] = representatives[*old_id as usize];
    }
    *internal_to_originals = new_internal_to_originals;
    *representatives = new_representatives;
}

fn reorder_common_classes(
    composite_to_class: HashMap<Vec<u32>, u32>,
    original_to_internal: &mut [u32],
    internal_to_originals: &mut Vec<Vec<u32>>,
    representatives: &mut Vec<u32>,
    allow_unmapped: bool,
) {
    let num_classes = internal_to_originals.len();
    if num_classes <= 1 {
        return;
    }

    let mut sorted: Vec<(Vec<u32>, u32)> = composite_to_class.into_iter().collect();
    sorted.sort_by(|left, right| left.0.cmp(&right.0));

    let mut old_to_new = vec![0u32; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        old_to_new[*old_id as usize] = new_id as u32;
    }

    for value in original_to_internal.iter_mut() {
        if *value == u32::MAX && allow_unmapped {
            continue;
        }
        *value = old_to_new[*value as usize];
    }

    let mut new_internal_to_originals = vec![Vec::new(); num_classes];
    let mut new_representatives = vec![u32::MAX; num_classes];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        new_internal_to_originals[new_id] = std::mem::take(&mut internal_to_originals[*old_id as usize]);
        new_representatives[new_id] = representatives[*old_id as usize];
    }
    *internal_to_originals = new_internal_to_originals;
    *representatives = new_representatives;
}

fn build_local_to_common_tsid_map(
    local_id_map: &InternalIdMap,
    common_id_map: &InternalIdMap,
) -> Vec<Vec<u32>> {
    let num_local = local_id_map.num_tsids() as usize;
    let mut local_to_common = vec![Vec::new(); num_local];

    for (state, &local_tsid) in local_id_map
        .tokenizer_states
        .original_to_internal
        .iter()
        .enumerate()
    {
        if local_tsid == u32::MAX {
            continue;
        }
        let common_tsid = common_id_map
            .tokenizer_states
            .original_to_internal
            .get(state)
            .copied()
            .unwrap_or(u32::MAX);
        if common_tsid == u32::MAX {
            continue;
        }
        local_to_common[local_tsid as usize].push(common_tsid);
    }

    sort_dedup_local_to_common(local_to_common)
}

fn build_local_to_common_token_map_from_common_classes(
    local_id_map: &InternalIdMap,
    common_id_map: &InternalIdMap,
) -> Vec<Vec<u32>> {
    let num_local = local_id_map.num_internal_tokens() as usize;
    let mut local_to_common = vec![Vec::new(); num_local];

    for (common_token, originals) in common_id_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .enumerate()
    {
        let Some(&representative) = originals.first() else {
            continue;
        };
        let local_token = local_id_map
            .vocab_tokens
            .original_to_internal
            .get(representative as usize)
            .copied()
            .unwrap_or(u32::MAX);
        if local_token == u32::MAX {
            continue;
        }
        if let Some(common_tokens) = local_to_common.get_mut(local_token as usize) {
            common_tokens.push(common_token as u32);
        }
    }

    sort_dedup_local_to_common(local_to_common)
}

fn sort_dedup_local_to_common(mut local_to_common: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    for ids in &mut local_to_common {
        if ids.len() <= 1 {
            continue;
        }
        ids.sort_unstable();
        ids.dedup();
    }
    local_to_common
}

#[derive(Debug)]
struct InjectiveLocalMap {
    destination_by_local: Vec<u32>,
    destination_order_is_monotone: bool,
}

#[derive(Debug)]
struct DisjointRunLocalMap {
    runs_by_local: Vec<Vec<(u32, u32)>>,
    destination_order_is_monotone: bool,
    /// When all non-empty local classes occupy adjacent destination spans in
    /// local-ID order, any inclusive local interval maps to at most one
    /// destination interval.  These prefix/suffix lookups make that mapping
    /// constant-time.
    contiguous_interval_lookup: Option<ContiguousIntervalLookup>,
}

#[derive(Debug)]
struct ContiguousIntervalLookup {
    first_destination_at_or_after: Vec<u32>,
    last_destination_at_or_before: Vec<u32>,
}

impl ContiguousIntervalLookup {
    #[inline]
    fn remap(&self, local_start: u32, local_end: u32) -> Option<(u32, u32)> {
        let start = *self
            .first_destination_at_or_after
            .get(local_start as usize)?;
        let end = *self
            .last_destination_at_or_before
            .get(local_end as usize)?;
        (start != u32::MAX && end != u32::MAX && start <= end).then_some((start, end))
    }
}

impl DisjointRunLocalMap {
    fn from_local_to_common(local_to_common: &[Vec<u32>], common_count: usize) -> Option<Self> {
        let mut seen = vec![false; common_count];
        let mut runs_by_local = Vec::with_capacity(local_to_common.len());
        let mut previous_destination = None;
        let mut destination_order_is_monotone = true;

        for destinations in local_to_common {
            let mut runs = Vec::new();
            let mut run_start = None;
            let mut run_end = 0u32;
            for &destination in destinations {
                let index = destination as usize;
                if index >= common_count || std::mem::replace(&mut seen[index], true) {
                    return None;
                }
                if let Some(previous) = previous_destination
                    && destination <= previous
                {
                    destination_order_is_monotone = false;
                }
                previous_destination = Some(destination);

                match run_start {
                    Some(_) if run_end.checked_add(1) == Some(destination) => {
                        run_end = destination;
                    }
                    Some(start) => {
                        runs.push((start, run_end));
                        run_start = Some(destination);
                        run_end = destination;
                    }
                    None => {
                        run_start = Some(destination);
                        run_end = destination;
                    }
                }
            }
            if let Some(start) = run_start {
                runs.push((start, run_end));
            }
            runs_by_local.push(runs);
        }

        let contiguous_interval_lookup = if destination_order_is_monotone
            && runs_by_local.iter().all(|runs| runs.len() <= 1)
        {
            let mut expected_next = None;
            let mut adjacent = true;
            for runs in &runs_by_local {
                let Some(&(start, end)) = runs.first() else {
                    continue;
                };
                if let Some(expected) = expected_next
                    && start != expected
                {
                    adjacent = false;
                    break;
                }
                expected_next = end.checked_add(1);
            }
            adjacent.then(|| {
                let mut first_destination_at_or_after = vec![u32::MAX; runs_by_local.len()];
                let mut next = u32::MAX;
                for (local, runs) in runs_by_local.iter().enumerate().rev() {
                    if let Some(&(start, _)) = runs.first() {
                        next = start;
                    }
                    first_destination_at_or_after[local] = next;
                }
                let mut last_destination_at_or_before = vec![u32::MAX; runs_by_local.len()];
                let mut previous = u32::MAX;
                for (local, runs) in runs_by_local.iter().enumerate() {
                    if let Some(&(_, end)) = runs.first() {
                        previous = end;
                    }
                    last_destination_at_or_before[local] = previous;
                }
                ContiguousIntervalLookup {
                    first_destination_at_or_after,
                    last_destination_at_or_before,
                }
            })
        } else {
            None
        };

        Some(Self {
            runs_by_local,
            destination_order_is_monotone,
            contiguous_interval_lookup,
        })
    }
}

impl InjectiveLocalMap {
    fn from_local_to_common(local_to_common: &[Vec<u32>], common_count: usize) -> Option<Self> {
        let mut destination_by_local = vec![u32::MAX; local_to_common.len()];
        let mut seen_destinations = vec![false; common_count];
        let mut previous_destination = None;
        let mut destination_order_is_monotone = true;

        for (local, destinations) in local_to_common.iter().enumerate() {
            let &destination = match destinations.as_slice() {
                [] => continue,
                [destination] => destination,
                _ => return None,
            };
            let destination_index = destination as usize;
            if destination_index >= common_count || std::mem::replace(&mut seen_destinations[destination_index], true) {
                return None;
            }
            if let Some(previous) = previous_destination
                && destination <= previous
            {
                destination_order_is_monotone = false;
            }
            previous_destination = Some(destination);
            destination_by_local[local] = destination;
        }

        Some(Self {
            destination_by_local,
            destination_order_is_monotone,
        })
    }

    #[inline]
    fn destination(&self, local: u32) -> Option<u32> {
        self.destination_by_local
            .get(local as usize)
            .copied()
            .filter(|&destination| destination != u32::MAX)
    }
}

fn remap_token_set_with_injective_map(
    tokens: &SharedTokenSet,
    token_map: &InjectiveLocalMap,
    cache: &mut HashMap<usize, SharedTokenSet>,
) -> SharedTokenSet {
    let key = Arc::as_ptr(tokens) as usize;
    if let Some(cached) = cache.get(&key) {
        return Arc::clone(cached);
    }

    let mut mapped = RangeSetBlaze::new();
    for local_token in tokens.iter() {
        if let Some(common_token) = token_map.destination(local_token) {
            mapped.insert(common_token);
        }
    }
    // The injective path is already cheap, so retain its local cache and avoid
    // paying for a global prepass on small one-to-one maps.
    let mapped = Arc::new(mapped);
    cache.insert(key, Arc::clone(&mapped));
    mapped
}

fn remap_token_set_with_general_map(
    tokens: &SharedTokenSet,
    local_to_common_tokens: &[Vec<u32>],
    cache: &mut HashMap<usize, SharedTokenSet>,
) -> SharedTokenSet {
    let key = Arc::as_ptr(tokens) as usize;
    if let Some(mapped) = cache.get(&key) {
        return Arc::clone(mapped);
    }
    let mut mapped = RangeSetBlaze::new();
    for local_token in tokens.iter() {
        if let Some(common_tokens) = local_to_common_tokens.get(local_token as usize) {
            for &common_token in common_tokens {
                mapped.insert(common_token);
            }
        }
    }
    let mapped = Arc::new(mapped);
    cache.insert(key, Arc::clone(&mapped));
    mapped
}

fn remap_token_set_with_disjoint_runs(
    tokens: &SharedTokenSet,
    token_map: &DisjointRunLocalMap,
    cache: &mut HashMap<usize, SharedTokenSet>,
) -> SharedTokenSet {
    let key = Arc::as_ptr(tokens) as usize;
    if let Some(mapped) = cache.get(&key) {
        return Arc::clone(mapped);
    }

    let mapped = remap_token_set_with_disjoint_runs_uncached(tokens, token_map);
    cache.insert(key, Arc::clone(&mapped));
    mapped
}

fn remap_token_set_with_disjoint_runs_uncached(
    tokens: &SharedTokenSet,
    token_map: &DisjointRunLocalMap,
) -> SharedTokenSet {
    // Parent-major common coordinates make the dominant map interval-preserving:
    // a whole local range becomes one global range.  Fall back to exact local
    // class expansion for child/non-monotone maps.
    if let Some(lookup) = token_map.contiguous_interval_lookup.as_ref() {
        // `tokens.ranges()` is sorted/disjoint, and an interval-preserving
        // monotone map keeps the images sorted/disjoint.  Let RangeSetBlaze
        // consume that ordered stream directly instead of paying one balanced
        // tree insertion per source range (millions of ranges in large parser
        // artifacts).
        return Arc::new(RangeSetBlaze::from_iter(tokens.ranges().filter_map(
            |local_range| {
                lookup
                    .remap(*local_range.start(), *local_range.end())
                    .map(|(start, end)| start..=end)
            },
        )));
    } else {
        let mut mapped = RangeSetBlaze::new();
        for local_token in tokens.iter() {
            if let Some(runs) = token_map.runs_by_local.get(local_token as usize) {
                for &(start, end) in runs {
                    mapped.ranges_insert(start..=end);
                }
            }
        }
        Arc::new(mapped)
    }
}

fn remap_local_token_range_with_disjoint_runs(
    local_start: u32,
    local_end: u32,
    token_map: &DisjointRunLocalMap,
) -> SharedTokenSet {
    let mut mapped = RangeSetBlaze::new();
    if let Some(lookup) = token_map.contiguous_interval_lookup.as_ref() {
        if let Some((start, end)) = lookup.remap(local_start, local_end) {
            mapped.ranges_insert(start..=end);
        }
    } else {
        for local_token in local_start..=local_end {
            if let Some(runs) = token_map.runs_by_local.get(local_token as usize) {
                for &(start, end) in runs {
                    mapped.ranges_insert(start..=end);
                }
            }
        }
    }
    Arc::new(mapped)
}

fn remap_token_set_with_precomputed_local_ranges(
    tokens: &SharedTokenSet,
    remapped_ranges: &FxHashMap<(u32, u32), SharedTokenSet>,
) -> SharedTokenSet {
    let mut mapped = RangeSetBlaze::new();
    for local_range in tokens.ranges() {
        let key = (*local_range.start(), *local_range.end());
        let remapped = remapped_ranges
            .get(&key)
            .expect("precomputed token-range remap must cover every source range");
        for destination in remapped.ranges() {
            mapped.ranges_insert(*destination.start()..=*destination.end());
        }
    }
    Arc::new(mapped)
}

fn remap_weight_with_injective_maps(
    weight: &Weight,
    tsid_map: &InjectiveLocalMap,
    token_map: &InjectiveLocalMap,
    precomputed_token_sets: Option<&FxHashMap<usize, SharedTokenSet>>,
    token_cache: &mut HashMap<usize, SharedTokenSet>,
) -> Weight {
    // Preserve the generic path for the special universal representation. It
    // is rare and has different "all mapped IDs" semantics.
    if weight.is_empty() || weight.is_full() {
        return weight.clone();
    }

    let mut entries = Vec::<(u32, SharedTokenSet)>::new();
    for (local_range, tokens) in weight.raw_range_values() {
        let mapped_tokens = if let Some(precomputed) = precomputed_token_sets {
            precomputed
                .get(&(Arc::as_ptr(tokens) as usize))
                .cloned()
                .expect("precomputed token remap must cover every source token set")
        } else {
            remap_token_set_with_injective_map(tokens, token_map, token_cache)
        };
        if mapped_tokens.is_empty() {
            continue;
        }
        for local_tsid in *local_range.start()..=*local_range.end() {
            if let Some(common_tsid) = tsid_map.destination(local_tsid) {
                entries.push((common_tsid, Arc::clone(&mapped_tokens)));
            }
        }
    }

    if entries.is_empty() {
        return Weight::empty();
    }
    if !tsid_map.destination_order_is_monotone {
        entries.sort_unstable_by_key(|(common_tsid, _)| *common_tsid);
    }

    weight_from_sorted_disjoint_runs(
        entries.into_iter().map(|(tsid, tokens)| (tsid, tsid, tokens)),
    )
}

/// The maps used by the fast remappers are injective or disjoint expansions,
/// so sorted destination runs cannot overlap. Coalesce the same adjacent runs
/// as CompactRangeBuilder, then build the tree from the ordered stream once.
/// The final interning boundary and identifier numbering remain unchanged.
fn weight_from_sorted_disjoint_runs(
    runs: impl IntoIterator<Item = (u32, u32, SharedTokenSet)>,
) -> Weight {
    let runs = runs.into_iter();
    // The ordinary builder is cheaper than an extra buffer for tiny weights.
    if runs.size_hint().1.is_some_and(|maximum| maximum <= 16) {
        return Weight::from_tsid_runs_shared(runs);
    }
    let mut ranges: Vec<(std::ops::RangeInclusive<u32>, SharedTokenSet)> = Vec::new();
    for (start, end, tokens) in runs {
        if tokens.is_empty() {
            continue;
        }
        if let Some((previous, previous_tokens)) = ranges.last_mut()
            && previous.end().checked_add(1) == Some(start)
            && (Arc::ptr_eq(previous_tokens, &tokens)
                || previous_tokens.as_ref() == tokens.as_ref())
        {
            let first = *previous.start();
            *previous = first..=end;
        } else {
            ranges.push((start..=end, tokens));
        }
    }
    let map = RangeMapBlaze::from_sorted_disjoint_map(CheckSortedDisjointMap::new(
        ranges.iter().map(|(range, tokens)| (range.clone(), tokens)),
    ));
    finalize_weight_map(map)
}

fn remap_weight_with_disjoint_tsid_runs(
    weight: &Weight,
    tsid_map: &DisjointRunLocalMap,
    token_map: Option<&InjectiveLocalMap>,
    token_run_map: Option<&DisjointRunLocalMap>,
    precomputed_token_sets: Option<&FxHashMap<usize, SharedTokenSet>>,
    local_to_common_tokens: &[Vec<u32>],
    token_cache: &mut HashMap<usize, SharedTokenSet>,
) -> Option<Weight> {
    if weight.is_empty() {
        return Some(weight.clone());
    }
    if weight.is_full() {
        return None;
    }

    let mut entries = Vec::new();
    for (local_start, local_end, tokens) in weight.range_entries() {
        let mapped_tokens = if let Some(precomputed_token_sets) = precomputed_token_sets {
            let key = Arc::as_ptr(tokens) as usize;
            precomputed_token_sets
                .get(&key)
                .cloned()
                .expect("precomputed token remap must cover every source token set")
        } else if let Some(token_map) = token_map {
            remap_token_set_with_injective_map(tokens, token_map, token_cache)
        } else if let Some(token_run_map) = token_run_map {
            remap_token_set_with_disjoint_runs(tokens, token_run_map, token_cache)
        } else {
            remap_token_set_with_general_map(tokens, local_to_common_tokens, token_cache)
        };
        if mapped_tokens.is_empty() {
            continue;
        }
        if let Some(lookup) = tsid_map.contiguous_interval_lookup.as_ref() {
            if let Some((start, end)) = lookup.remap(local_start, local_end) {
                entries.push((start, end, Arc::clone(&mapped_tokens)));
            }
            continue;
        }
        for local_tsid in local_start..=local_end {
            for &(start, end) in tsid_map.runs_by_local.get(local_tsid as usize)? {
                entries.push((start, end, Arc::clone(&mapped_tokens)));
            }
        }
    }
    if !tsid_map.destination_order_is_monotone {
        entries.sort_unstable_by_key(|(start, _, _)| *start);
    }
    Some(weight_from_sorted_disjoint_runs(entries))
}

fn local_to_common_is_identity(map: &[Vec<u32>], common_count: usize) -> bool {
    map.len() == common_count
        && map
            .iter()
            .enumerate()
            .all(|(local, targets)| targets.as_slice() == [local as u32])
}

fn remap_weight_with_identity_tsids(
    weight: &Weight,
    token_map: Option<&InjectiveLocalMap>,
    token_run_map: Option<&DisjointRunLocalMap>,
    precomputed_token_sets: Option<&FxHashMap<usize, SharedTokenSet>>,
    local_to_common_tokens: &[Vec<u32>],
) -> Weight {
    if weight.is_empty() || weight.is_full() {
        return weight.clone();
    }
    let mut token_cache = HashMap::<usize, SharedTokenSet>::new();
    weight.remap_token_sets_preserving_tsid_ranges(|tokens| {
        if let Some(precomputed_token_sets) = precomputed_token_sets {
            precomputed_token_sets
                .get(&(Arc::as_ptr(tokens) as usize))
                .cloned()
                .expect("precomputed token remap must cover every source token set")
        } else if let Some(token_map) = token_map {
            remap_token_set_with_injective_map(tokens, token_map, &mut token_cache)
        } else if let Some(token_run_map) = token_run_map {
            remap_token_set_with_disjoint_runs(tokens, token_run_map, &mut token_cache)
        } else {
            remap_token_set_with_general_map(tokens, local_to_common_tokens, &mut token_cache)
        }
    })
}

pub fn remap_weights_with_maps(
    weights: &mut [&mut Weight],
    local_to_common_tsids: &[Vec<u32>],
    local_to_common_tokens: &[Vec<u32>],
    common_tsid_count: usize,
) {
    let profiling = std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some();
    let total_started_at = profiling.then(Instant::now);
    let tsid_map = InjectiveLocalMap::from_local_to_common(local_to_common_tsids, common_tsid_count);
    let tsid_run_map =
        DisjointRunLocalMap::from_local_to_common(local_to_common_tsids, common_tsid_count);
    let common_token_count = local_to_common_tokens
        .iter()
        .flatten()
        .copied()
        .max()
        .map_or(0, |maximum| maximum as usize + 1);
    let token_map = InjectiveLocalMap::from_local_to_common(
        local_to_common_tokens,
        common_token_count,
    );
    let token_run_map =
        DisjointRunLocalMap::from_local_to_common(local_to_common_tokens, common_token_count);
    let identity_tsids = local_to_common_is_identity(local_to_common_tsids, common_tsid_count);
    let identity_tokens = local_to_common_is_identity(local_to_common_tokens, common_token_count);
    // Reconciliation is literally the identity when both coordinates are already
    // published in the common numbering. Avoid walking/interning every weight and
    // token set only to reconstruct the same artifact.
    if identity_tsids && identity_tokens {
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][token_run_reconcile] weights={} identity_tsids=true identity_tokens=true action=skip total_ms={:.3}",
                weights.len(),
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return;
    }
    let mut unique_by_ptr = FxHashMap::<usize, usize>::default();
    let mut unique_weights = Vec::<Weight>::new();
    let mut weight_to_unique = Vec::with_capacity(weights.len());
    for weight in weights.iter() {
        let ptr = weight.ptr_key();
        let unique = *unique_by_ptr.entry(ptr).or_insert_with(|| {
            let index = unique_weights.len();
            unique_weights.push((**weight).clone());
            index
        });
        weight_to_unique.push(unique);
    }

    let precompute_started_at = profiling.then(Instant::now);
    let precomputed_token_sets = if tsid_run_map.is_some()
        && let Some(token_run_map) = token_run_map.as_ref()
    {
        let mut source_by_ptr = FxHashMap::<usize, SharedTokenSet>::default();
        for weight in &unique_weights {
            for (_, _, tokens) in weight.range_entries() {
                source_by_ptr
                    .entry(Arc::as_ptr(tokens) as usize)
                    .or_insert_with(|| Arc::clone(tokens));
            }
        }
        let sources = source_by_ptr.into_iter().collect::<Vec<_>>();
        let total_source_tokens = sources
            .iter()
            .map(|(_, tokens)| tokens.len() as usize)
            .sum::<usize>();
        let total_source_ranges = sources
            .iter()
            .map(|(_, tokens)| tokens.ranges().count())
            .sum::<usize>();
        // Building a global range cache pays a hash/interning cost before any
        // source set is remapped. With only a few dozen distinct source token
        // sets, direct disjoint-run remapping is cheaper even for fairly dense
        // sets; reserve the shared range cache for workloads with enough
        // independent sets to amortize that setup cost.
        let use_range_cache = sources.len() >= PARALLEL_UNIQUE_WEIGHT_THRESHOLD
            && total_source_tokens >= 8_192
            && total_source_ranges.saturating_mul(4) < total_source_tokens;
        let remapped = if use_range_cache {
            let mut unique_ranges = FxHashSet::<(u32, u32)>::default();
            for (_, tokens) in &sources {
                unique_ranges.extend(
                    tokens
                        .ranges()
                        .map(|range| (*range.start(), *range.end())),
                );
            }
            let unique_ranges = unique_ranges.into_iter().collect::<Vec<_>>();
            let remapped_ranges = if unique_ranges.len() >= PARALLEL_UNIQUE_WEIGHT_THRESHOLD
                && rayon::current_num_threads() > 1
            {
                unique_ranges
                    .par_iter()
                    .map(|&(start, end)| {
                        (
                            (start, end),
                            remap_local_token_range_with_disjoint_runs(
                                start,
                                end,
                                token_run_map,
                            ),
                        )
                    })
                    .collect::<FxHashMap<_, _>>()
            } else {
                unique_ranges
                    .iter()
                    .map(|&(start, end)| {
                        (
                            (start, end),
                            remap_local_token_range_with_disjoint_runs(
                                start,
                                end,
                                token_run_map,
                            ),
                        )
                    })
                    .collect::<FxHashMap<_, _>>()
            };
            if sources.len() >= PARALLEL_UNIQUE_WEIGHT_THRESHOLD
                && rayon::current_num_threads() > 1
            {
                sources
                    .par_iter()
                    .map(|(key, tokens)| {
                        (
                            *key,
                            remap_token_set_with_precomputed_local_ranges(
                                tokens,
                                &remapped_ranges,
                            ),
                        )
                    })
                    .collect::<Vec<_>>()
            } else {
                sources
                    .iter()
                    .map(|(key, tokens)| {
                        (
                            *key,
                            remap_token_set_with_precomputed_local_ranges(
                                tokens,
                                &remapped_ranges,
                            ),
                        )
                    })
                    .collect::<Vec<_>>()
            }
        } else if sources.len() >= PARALLEL_UNIQUE_WEIGHT_THRESHOLD
            && rayon::current_num_threads() > 1
        {
            sources
                .par_iter()
                .map(|(key, tokens)| {
                    (
                        *key,
                        remap_token_set_with_disjoint_runs_uncached(tokens, token_run_map),
                    )
                })
                .collect::<Vec<_>>()
        } else {
            sources
                .iter()
                .map(|(key, tokens)| {
                    (
                        *key,
                        remap_token_set_with_disjoint_runs_uncached(tokens, token_run_map),
                    )
                })
                .collect::<Vec<_>>()
        };
        Some(remapped.into_iter().collect::<FxHashMap<_, _>>())
    } else {
        None
    };
    let precompute_ms = precompute_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let token_set_profile = profiling.then(|| {
        let mut seen = BTreeSet::new();
        let mut total_refs = 0usize;
        let mut unique_local_tokens = 0usize;
        let mut unique_local_ranges = 0usize;
        for weight in &unique_weights {
            for (_, _, tokens) in weight.range_entries() {
                total_refs += 1;
                let key = Arc::as_ptr(tokens) as usize;
                if seen.insert(key) {
                    unique_local_tokens += tokens.len() as usize;
                    unique_local_ranges += tokens.ranges().count();
                }
            }
        }
        (
            total_refs,
            seen.len(),
            unique_local_tokens,
            unique_local_ranges,
        )
    });

    let remap_one =
        |weight: &Weight,
         precomputed_token_sets: Option<&FxHashMap<usize, SharedTokenSet>>| {
            if identity_tsids {
                return remap_weight_with_identity_tsids(
                    weight,
                    token_map.as_ref(),
                    token_run_map.as_ref(),
                    precomputed_token_sets,
                    local_to_common_tokens,
                );
            }
            match (&tsid_map, &token_map) {
        (Some(tsid_map), Some(token_map)) if !weight.is_full() => {
            let mut token_cache = HashMap::<usize, SharedTokenSet>::new();
            remap_weight_with_injective_maps(
                weight, tsid_map, token_map, precomputed_token_sets, &mut token_cache,
            )
        }
        _ if !weight.is_full() && tsid_run_map.is_some() => {
            let mut token_cache = HashMap::<usize, SharedTokenSet>::new();
            tsid_run_map
                .as_ref()
                .and_then(|tsid_run_map| {
                    remap_weight_with_disjoint_tsid_runs(
                        weight,
                        tsid_run_map,
                        token_map.as_ref(),
                        token_run_map.as_ref(),
                        precomputed_token_sets,
                        local_to_common_tokens,
                        &mut token_cache,
                    )
                })
                .unwrap_or_else(|| {
                    remap_weight_general(
                        weight,
                        local_to_common_tsids,
                        local_to_common_tokens,
                        common_tsid_count,
                    )
                })
        }
        _ => remap_weight_general(
            weight,
            local_to_common_tsids,
            local_to_common_tokens,
            common_tsid_count,
        ),
            }
        };

    // Parser-family reconciliation often has 100-200 unique weights whose remaps
    // are individually substantial; 256 leaves that expensive middle regime serial.
    const PARALLEL_UNIQUE_WEIGHT_THRESHOLD: usize = 128;
    let unique_remap_started_at = profiling.then(Instant::now);
    let remapped_unique: Vec<Weight> = if unique_weights.len() >= PARALLEL_UNIQUE_WEIGHT_THRESHOLD
        && rayon::current_num_threads() > 1
    {
        unique_weights
            .par_iter()
            .map(|weight| remap_one(weight, precomputed_token_sets.as_ref()))
            .collect()
    } else {
        unique_weights
            .iter()
            .map(|weight| remap_one(weight, precomputed_token_sets.as_ref()))
            .collect()
    };
    let unique_remap_ms = unique_remap_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    if std::env::var_os("GLRMASK_VALIDATE_GLOBAL_TOKEN_REMAP_CACHE").is_some()
        && precomputed_token_sets.is_some()
    {
        let reference: Vec<Weight> = if unique_weights.len() >= PARALLEL_UNIQUE_WEIGHT_THRESHOLD
            && rayon::current_num_threads() > 1
        {
            unique_weights
                .par_iter()
                .map(|weight| remap_one(weight, None))
                .collect()
        } else {
            unique_weights
                .iter()
                .map(|weight| remap_one(weight, None))
                .collect()
        };
        for (index, (candidate, reference)) in
            remapped_unique.iter().zip(reference.iter()).enumerate()
        {
            assert_eq!(
                candidate, reference,
                "global token remap cache changed unique weight {index}"
            );
        }
        eprintln!(
            "[glrmask/validate][global_token_remap_cache] unique_weights={} token_sets={} exact=true",
            unique_weights.len(),
            precomputed_token_sets.as_ref().map_or(0, FxHashMap::len),
        );
    }

    for (weight, unique) in weights.iter_mut().zip(weight_to_unique) {
        **weight = remapped_unique[unique].clone();
    }
    if let Some(total_started_at) = total_started_at {
        let (token_set_refs, unique_token_sets, unique_local_tokens, unique_local_ranges) =
            token_set_profile.unwrap_or_default();
        eprintln!(
            "[glrmask/profile][token_run_reconcile] weights={} unique_weights={} token_set_refs={} unique_token_sets={} unique_local_tokens={} unique_local_ranges={} identity_tsids={} tsid_map={} tsid_run_map={} token_map={} token_run_map={} precomputed_token_sets={} precompute_ms={precompute_ms:.3} unique_remap_ms={unique_remap_ms:.3} total_ms={:.3}",
            weights.len(),
            unique_weights.len(),
            token_set_refs,
            unique_token_sets,
            unique_local_tokens,
            unique_local_ranges,
            identity_tsids,
            tsid_map.is_some(),
            tsid_run_map.is_some(),
            token_map.is_some(),
            token_run_map.is_some(),
            precomputed_token_sets.as_ref().map_or(0, FxHashMap::len),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
}

fn remap_weight_cached_general(
    weight: &Weight,
    local_to_common_tsids: &[Vec<u32>],
    local_to_common_tokens: &[Vec<u32>],
    common_tsid_count: usize,
    cache: &mut HashMap<usize, Weight>,
) -> Weight {
    let ptr = weight.ptr_key();
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }

    let remapped = remap_weight_general(
        weight,
        local_to_common_tsids,
        local_to_common_tokens,
        common_tsid_count,
    );
    cache.insert(ptr, remapped.clone());
    remapped
}

fn remap_weight_general(
    weight: &Weight,
    local_to_common_tsids: &[Vec<u32>],
    local_to_common_tokens: &[Vec<u32>],
    common_tsid_count: usize,
) -> Weight {
    if weight.is_empty() {
        return weight.clone();
    }

    if weight.is_full() {
        let mut all_common_tokens = RangeSetBlaze::new();
        for common_tokens in local_to_common_tokens {
            for &common_token in common_tokens {
                all_common_tokens.insert(common_token);
            }
        }
        if all_common_tokens.is_empty() {
            return Weight::empty();
        }

        let mut all_common_tsids = BTreeSet::new();
        for common_tsids in local_to_common_tsids {
            for &common_tsid in common_tsids {
                if (common_tsid as usize) < common_tsid_count {
                    all_common_tsids.insert(common_tsid);
                }
            }
        }
        if all_common_tsids.is_empty() {
            return Weight::empty();
        }

        return Weight::from_per_tsid_token_sets(
            all_common_tsids
                .into_iter()
                .map(|common_tsid| (common_tsid, all_common_tokens.clone())),
        );
    }

    let Some(entries) = weight.compact_entries() else {
        return weight.clone();
    };

    let mut token_cache = HashMap::<usize, Arc<RangeSetBlaze<u32>>>::new();
    let mut tokens_by_common_tsid: Vec<Option<Arc<RangeSetBlaze<u32>>>> = vec![None; common_tsid_count];
    let mut any_set = false;

    for (start, end, tokens) in entries {
        let token_key = Arc::as_ptr(&tokens) as usize;
        let mapped_tokens = token_cache
            .entry(token_key)
            .or_insert_with(|| {
                let mut result = RangeSetBlaze::new();
                for local_token in tokens.iter() {
                    if let Some(common_tokens) = local_to_common_tokens.get(local_token as usize) {
                        for &common_token in common_tokens {
                            result.insert(common_token);
                        }
                    }
                }
                Arc::new(result)
            })
            .clone();

        for local_tsid in start..=end {
            let Some(common_tsids) = local_to_common_tsids.get(local_tsid as usize) else {
                continue;
            };
            for &common_tsid in common_tsids {
                let index = common_tsid as usize;
                if index >= common_tsid_count {
                    continue;
                }
                match &mut tokens_by_common_tsid[index] {
                    Some(existing) => {
                        let merged = existing.as_ref() | mapped_tokens.as_ref();
                        *existing = shared_rangeset(merged);
                    }
                    slot @ None => {
                        *slot = Some(Arc::clone(&mapped_tokens));
                    }
                }
                any_set = true;
            }
        }
    }

    if !any_set {
        return Weight::empty();
    }

    let mut map = RangeMapBlaze::<u32, Arc<RangeSetBlaze<u32>>>::new();
    let mut run_start: Option<u32> = None;
    let mut run_end = 0u32;
    let mut run_tokens: Option<Arc<RangeSetBlaze<u32>>> = None;

    for (index, slot) in tokens_by_common_tsid.iter().enumerate() {
        let common_tsid = index as u32;
        if let Some(tokens) = slot {
            if let Some(ref current) = run_tokens {
                if Arc::ptr_eq(current, tokens) || current.as_ref() == tokens.as_ref() {
                    run_end = common_tsid;
                    continue;
                }
                map.extend_simple(std::iter::once((
                    run_start.unwrap()..=run_end,
                    Arc::clone(current),
                )));
            }
            run_start = Some(common_tsid);
            run_end = common_tsid;
            run_tokens = Some(Arc::clone(tokens));
        } else if let Some(ref current) = run_tokens {
            map.extend_simple(std::iter::once((
                run_start.unwrap()..=run_end,
                Arc::clone(current),
            )));
            run_start = None;
            run_tokens = None;
        }
    }
    if let Some(tokens) = run_tokens {
        map.extend_simple(std::iter::once((run_start.unwrap()..=run_end, tokens)));
    }

    finalize_weight_map(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(original_to_internal: Vec<u32>, num_internal: u32) -> ManyToOneIdMap {
        ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            original_to_internal,
            num_internal,
        )
    }

    fn id_map(
        states: Vec<u32>,
        num_states: u32,
        tokens: Vec<u32>,
        num_tokens: u32,
    ) -> InternalIdMap {
        InternalIdMap {
            tokenizer_states: map(states, num_states),
            vocab_tokens: map(tokens, num_tokens),
            deferred_vocab_singleton_original_ids: None,
        }
    }

    fn entries_key(weight: &Weight) -> Vec<(u32, u32, Vec<(u32, u32)>)> {
        weight
            .raw_range_values()
            .map(|(range, tokens)| {
                (
                    *range.start(),
                    *range.end(),
                    tokens
                        .ranges()
                        .map(|range| (*range.start(), *range.end()))
                        .collect(),
                )
            })
            .collect()
    }

    fn id_map_key(id_map: &InternalIdMap) -> (Vec<u32>, Vec<Vec<u32>>, Vec<u32>, Vec<Vec<u32>>) {
        (
            id_map.tokenizer_states.original_to_internal.clone(),
            id_map.tokenizer_states.internal_to_originals.clone(),
            id_map.vocab_tokens.original_to_internal.clone(),
            id_map.vocab_tokens.internal_to_originals.clone(),
        )
    }

    fn reconcile_generic_for_test(
        left_weight: &mut Weight,
        left_id_map: &mut InternalIdMap,
        right_weight: &mut Weight,
        right_id_map: &mut InternalIdMap,
    ) {
        let common_id_map = build_common_internal_id_map(&[left_id_map, right_id_map]);
        let left_tsid_map = build_local_to_common_tsid_map(left_id_map, &common_id_map);
        let left_token_map =
            build_local_to_common_token_map_from_common_classes(left_id_map, &common_id_map);
        let right_tsid_map = build_local_to_common_tsid_map(right_id_map, &common_id_map);
        let right_token_map =
            build_local_to_common_token_map_from_common_classes(right_id_map, &common_id_map);

        remap_weights_with_maps(
            &mut [left_weight],
            &left_tsid_map,
            &left_token_map,
            common_id_map.num_tsids() as usize,
        );
        remap_weights_with_maps(
            &mut [right_weight],
            &right_tsid_map,
            &right_token_map,
            common_id_map.num_tsids() as usize,
        );

        *left_id_map = common_id_map.clone();
        *right_id_map = common_id_map;
    }

    #[test]
    fn sparse_unmapped_pair_common_map_matches_hashed_pair_order() {
        for (left, right, num_originals, allow_unmapped) in [
            (
                map(vec![0, 0, 1, 1, 2, 2], 3),
                map(vec![0, 1, 0, 1, 2, 2], 3),
                6,
                false,
            ),
            (
                map(vec![0, u32::MAX, 1, 1, u32::MAX, 2], 3),
                map(vec![u32::MAX, 0, 0, 1, u32::MAX, 1], 2),
                6,
                true,
            ),
        ] {
            let hashed = build_common_many_to_one_id_map_pair_hashed(
                &left,
                &right,
                num_originals,
                allow_unmapped,
            );
            if allow_unmapped {
                let sparse = build_common_many_to_one_id_map_pair_sparse_unmapped(
                    &left,
                    &right,
                    num_originals,
                );
                assert_eq!(sparse.original_to_internal, hashed.original_to_internal);
                assert_eq!(sparse.internal_to_originals, hashed.internal_to_originals);
                assert_eq!(
                    sparse.representative_original_ids,
                    hashed.representative_original_ids,
                );
            }
        }
    }

    #[test]
    fn injective_reconcile_remap_matches_general_for_reordered_ids() {
        let weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=1])),
            (1, RangeSetBlaze::from_iter([3..=3])),
            (2, RangeSetBlaze::from_iter([0..=1])),
            (3, RangeSetBlaze::from_iter([3..=3])),
        ]);
        let tsid_map = vec![vec![2], vec![0], vec![3], vec![1]];
        let token_map = vec![vec![3], vec![1], vec![0], vec![2]];

        let general = remap_weight_general(&weight, &tsid_map, &token_map, 4);
        let fast = remap_weight_with_injective_maps(
            &weight,
            &InjectiveLocalMap::from_local_to_common(&tsid_map, 4).expect("injective tsids"),
            &InjectiveLocalMap::from_local_to_common(&token_map, 4).expect("injective tokens"),
            None,
            &mut HashMap::new(),
        );

        assert_eq!(entries_key(&fast), entries_key(&general));
    }

    #[test]
    fn bulk_sorted_weight_builder_matches_incremental_builder() {
        let mut seed = 0x5a17_6943u32;
        for count in [0, 1, 2, 15, 16, 17, 32, 128, 257] {
            for _ in 0..16 {
                let mut start = 0u32;
                let mut runs = Vec::new();
                for i in 0..count {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let end = start + (seed >> 8) % 4;
                    let tokens = if i % 11 == 0 {
                        RangeSetBlaze::new()
                    } else {
                        // Equal values in independent Arcs must coalesce too.
                        RangeSetBlaze::from_iter([((i / 3) % 7) as u32..=((i / 3) % 7 + 2) as u32])
                    };
                    runs.push((start, end, Arc::new(tokens)));
                    start = end + 1 + (seed >> 16) % 3;
                }
                let reference = Weight::from_tsid_ranges_shared(runs.clone());
                let candidate = weight_from_sorted_disjoint_runs(runs);
                assert_eq!(entries_key(&candidate), entries_key(&reference));
                assert_eq!(candidate, reference);
            }
        }
        let tokens = Arc::new(RangeSetBlaze::from_iter([1..=3]));
        let runs = (0..32).map(|i| {
            let point = u32::MAX - 64 + i * 2;
            (point, point, Arc::clone(&tokens))
        }).collect::<Vec<_>>();
        assert_eq!(weight_from_sorted_disjoint_runs(runs.clone()),
            Weight::from_tsid_ranges_shared(runs));
    }

    #[test]
    fn injective_token_remaps_use_global_precomputation() {
        let weight = Weight::from_tsid_ranges_shared([
            (0, 1, Arc::new(RangeSetBlaze::from_iter([0..=1]))),
            (2, 3, Arc::new(RangeSetBlaze::from_iter([1..=1, 3..=3]))),
        ]);
        let tokens = vec![vec![3], vec![1], vec![0], vec![2]];
        let token_map = InjectiveLocalMap::from_local_to_common(&tokens, 4).unwrap();
        let token_runs = DisjointRunLocalMap::from_local_to_common(&tokens, 4).unwrap();
        let precomputed = weight.range_entries().map(|(_, _, source)| {
            (Arc::as_ptr(source) as usize,
             remap_token_set_with_disjoint_runs_uncached(source, &token_runs))
        }).collect::<FxHashMap<_, _>>();

        let tsids = vec![vec![2], vec![0], vec![3], vec![1]];
        let tsid_map = InjectiveLocalMap::from_local_to_common(&tsids, 4).unwrap();
        let mut local_cache = HashMap::new();
        let mapped = remap_weight_with_injective_maps(
            &weight, &tsid_map, &token_map, Some(&precomputed), &mut local_cache,
        );
        assert_eq!(mapped, remap_weight_general(&weight, &tsids, &tokens, 4));
        assert!(local_cache.is_empty(), "do not recompute globally mapped token sets");

        let tsids = vec![vec![3, 4], vec![], vec![0, 1, 2], vec![5]];
        let tsid_runs = DisjointRunLocalMap::from_local_to_common(&tsids, 6).unwrap();
        let mapped = remap_weight_with_disjoint_tsid_runs(
            &weight, &tsid_runs, Some(&token_map), Some(&token_runs),
            Some(&precomputed), &tokens, &mut local_cache,
        ).unwrap();
        assert_eq!(mapped, remap_weight_general(&weight, &tsids, &tokens, 6));
        assert!(local_cache.is_empty(), "disjoint TSID expansions must also use the cache");
    }

    #[test]
    fn contiguous_disjoint_tsid_range_remap_matches_general() {
        let tokens_a = Arc::new(RangeSetBlaze::from_iter([0..=1]));
        let tokens_b = Arc::new(RangeSetBlaze::from_iter([2..=3]));
        let weight = Weight::from_tsid_ranges_shared([
            (0, 1, Arc::clone(&tokens_a)),
            (3, 3, Arc::clone(&tokens_b)),
        ]);
        let tsid_map = vec![vec![0, 1], vec![2, 3], vec![], vec![4, 5]];
        let token_map = (0..4).map(|token| vec![token]).collect::<Vec<_>>();
        let run_map =
            DisjointRunLocalMap::from_local_to_common(&tsid_map, 6).expect("disjoint TSID runs");
        assert!(run_map.contiguous_interval_lookup.is_some());

        let general = remap_weight_general(&weight, &tsid_map, &token_map, 6);
        let fast = remap_weight_with_disjoint_tsid_runs(
            &weight,
            &run_map,
            None,
            DisjointRunLocalMap::from_local_to_common(&token_map, 4).as_ref(),
            None,
            &token_map,
            &mut HashMap::new(),
        )
        .expect("disjoint TSID run remap");

        assert_eq!(entries_key(&fast), entries_key(&general));
        assert_eq!(
            entries_key(&fast)
                .into_iter()
                .map(|(start, end, _)| (start, end))
                .collect::<Vec<_>>(),
            vec![(0, 3), (4, 5)],
        );
    }

    #[test]
    fn identity_tsid_remap_matches_general_with_split_token_class() {
        let weight = Weight::from_tsid_ranges_shared([
            (
                0,
                2,
                Arc::new(RangeSetBlaze::from_iter([0..=1, 3..=3])),
            ),
            (5, 9, Arc::new(RangeSetBlaze::from_iter([2..=2]))),
        ]);
        let tsid_map = (0..10).map(|tsid| vec![tsid]).collect::<Vec<_>>();
        let token_map = vec![vec![0], vec![1, 2, 3], vec![4], vec![5]];
        assert!(local_to_common_is_identity(&tsid_map, 10));

        let general = remap_weight_general(&weight, &tsid_map, &token_map, 10);
        let token_run_map =
            DisjointRunLocalMap::from_local_to_common(&token_map, 6).expect("disjoint token runs");
        let fast = remap_weight_with_identity_tsids(
            &weight,
            None,
            Some(&token_run_map),
            None,
            &token_map,
        );

        assert_eq!(entries_key(&fast), entries_key(&general));
        assert_eq!(
            entries_key(&fast)
                .into_iter()
                .map(|(start, end, _)| (start, end))
                .collect::<Vec<_>>(),
            vec![(0, 2), (5, 9)],
            "identity TSID remapping must preserve source ranges",
        );
    }

    #[test]
    fn disjoint_run_token_remap_matches_general_for_split_reordered_ids() {
        let tokens = Arc::new(RangeSetBlaze::from_iter([0..=2, 4..=4]));
        let token_map = vec![vec![4, 5], vec![0, 1], vec![2, 3], vec![], vec![7, 8, 9]];
        let run_map =
            DisjointRunLocalMap::from_local_to_common(&token_map, 10).expect("disjoint runs");
        assert!(!run_map.destination_order_is_monotone);

        let fast = remap_token_set_with_disjoint_runs(&tokens, &run_map, &mut HashMap::new());
        let general = remap_token_set_with_general_map(&tokens, &token_map, &mut HashMap::new());

        assert_eq!(fast.as_ref(), general.as_ref());
        assert_eq!(
            fast.ranges()
                .map(|range| (*range.start(), *range.end()))
                .collect::<Vec<_>>(),
            vec![(0, 5), (7, 9)],
        );
    }

    #[test]
    fn refinement_reconcile_matches_generic_with_non_empty_coarser_weights() {
        let left_id_map = id_map(
            vec![0, 0, 1, 1, 2, 2],
            3,
            vec![0, 0, 1, 1, 2, 2],
            3,
        );
        let right_id_map = id_map(vec![0, 0, 0, 0, 1, 1], 2, vec![0, 0, 0, 0, 0, 0], 1);
        assert!(internal_id_map_refines(&left_id_map, &right_id_map));

        let left_weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=1])),
            (2, RangeSetBlaze::from_iter([2..=2])),
        ]);
        let right_weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=0])),
            (1, RangeSetBlaze::from_iter([0..=0])),
        ]);

        let mut fast_left_id_map = left_id_map.clone();
        let mut fast_right_id_map = right_id_map.clone();
        let mut fast_left_weight = left_weight.clone();
        let mut fast_right_weight = right_weight.clone();
        reconcile_weight_id_maps(
            &mut [&mut fast_left_weight],
            &mut fast_left_id_map,
            &mut [&mut fast_right_weight],
            &mut fast_right_id_map,
        );

        let mut generic_left_id_map = left_id_map;
        let mut generic_right_id_map = right_id_map;
        let mut generic_left_weight = left_weight;
        let mut generic_right_weight = right_weight;
        reconcile_generic_for_test(
            &mut generic_left_weight,
            &mut generic_left_id_map,
            &mut generic_right_weight,
            &mut generic_right_id_map,
        );

        assert_eq!(id_map_key(&fast_left_id_map), id_map_key(&generic_left_id_map));
        assert_eq!(id_map_key(&fast_right_id_map), id_map_key(&generic_right_id_map));
        assert_eq!(entries_key(&fast_left_weight), entries_key(&generic_left_weight));
        assert_eq!(entries_key(&fast_right_weight), entries_key(&generic_right_weight));
    }

    #[test]
    fn refinement_reconcile_preserves_finer_weights_when_other_side_is_empty() {
        let left_id_map = id_map(
            vec![0, 1, 2, 3],
            4,
            vec![0, 1, 2, 3, 4, 5],
            6,
        );
        let right_id_map = id_map(vec![0, 0, 0, 0], 1, vec![0, 0, 0, 0, 0, 0], 1);
        assert!(internal_id_map_refines(&left_id_map, &right_id_map));

        let left_weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=2])),
            (3, RangeSetBlaze::from_iter([4..=5])),
        ]);
        let mut fast_left_id_map = left_id_map.clone();
        let mut fast_right_id_map = right_id_map;
        let mut fast_left_weight = left_weight.clone();
        reconcile_weight_id_maps(
            &mut [&mut fast_left_weight],
            &mut fast_left_id_map,
            &mut [],
            &mut fast_right_id_map,
        );

        assert_eq!(id_map_key(&fast_left_id_map), id_map_key(&left_id_map));
        assert_eq!(id_map_key(&fast_right_id_map), id_map_key(&left_id_map));
        assert_eq!(entries_key(&fast_left_weight), entries_key(&left_weight));
    }

    #[test]
    fn empty_side_reconcile_adopts_non_empty_side_after_domain_compaction() {
        let left_id_map = id_map(
            vec![0, 0, 1, 1],
            2,
            vec![0, 0, 1, 1, 2, 2],
            3,
        );
        let compact_empty_id_map = id_map(vec![0], 1, vec![0], 1);
        assert!(!internal_id_map_refines(&left_id_map, &compact_empty_id_map));

        let left_weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=1])),
            (1, RangeSetBlaze::from_iter([2..=2])),
        ]);
        let mut fast_left_id_map = left_id_map.clone();
        let mut fast_right_id_map = compact_empty_id_map;
        let mut fast_left_weight = left_weight.clone();
        reconcile_weight_id_maps(
            &mut [&mut fast_left_weight],
            &mut fast_left_id_map,
            &mut [],
            &mut fast_right_id_map,
        );

        assert_eq!(id_map_key(&fast_left_id_map), id_map_key(&left_id_map));
        assert_eq!(id_map_key(&fast_right_id_map), id_map_key(&left_id_map));
        assert_eq!(entries_key(&fast_left_weight), entries_key(&left_weight));
    }
}
