use super::*;

fn weight_for_tsid(tsid: u32, ranges: &[(u32, u32)]) -> Weight {
    let token_set = rangeset_from_ranges(ranges.iter().map(|(start, end)| *start..=*end));
    Weight::from_token_set_for_tsid(tsid, token_set)
}

#[test]
fn public_intersection_memo_does_not_keep_weights_alive() {
    clear_weight_op_caches();
    let left = weight_for_tsid(1, &[(1, 10)]);
    let right = weight_for_tsid(1, &[(5, 15)]);
    let left_weak = Arc::downgrade(&left.0);
    let right_weak = Arc::downgrade(&right.0);
    let result = left.intersection(&right);
    let result_weak = Arc::downgrade(&result.0);
    assert!(with_public_weight_intersection_memo(|memo| !memo.results.is_empty()));

    drop(result);
    drop(right);
    drop(left);

    assert!(left_weak.upgrade().is_none());
    assert!(right_weak.upgrade().is_none());
    assert!(result_weak.upgrade().is_none());
}

#[test]
fn scoped_weight_op_cache_weakly_validates_pointer_key_operands() {
    let mut cache = ScopedWeightOpCache::default();
    let left = weight_for_tsid(1, &[(1, 3)]);
    let right = weight_for_tsid(1, &[(7, 9)]);
    let key = scoped_commutative_weight_pair_key(&left, &right);
    let left_weak = Arc::downgrade(&left.0);
    let right_weak = Arc::downgrade(&right.0);
    let result = cache.union(&left, &right);
    assert!(!Arc::ptr_eq(&result.0, &left.0));
    assert!(!Arc::ptr_eq(&result.0, &right.0));

    drop(result);
    drop(right);
    drop(left);

    // The scoped cache must not keep temporary operands alive, but it must
    // retain weak identity guards so a recycled pointer cannot become a
    // false cache hit (ABA).
    assert!(left_weak.upgrade().is_none());
    assert!(right_weak.upgrade().is_none());
    let entry = cache.union_entries.get(&key).unwrap();
    assert!(entry.left_operand.upgrade().is_none());
    assert!(entry.right_operand.upgrade().is_none());
}

#[test]
fn interner_cleanup_deferral_releases_once() {
    let initial = INTERNER_CLEANUP_DEFERRAL_DEPTH.load(Ordering::Acquire);
    let first = defer_weight_interner_cleanup();
    let second = defer_weight_interner_cleanup();
    assert_eq!(
        INTERNER_CLEANUP_DEFERRAL_DEPTH.load(Ordering::Acquire),
        initial + 2,
    );
    second.finish();
    assert_eq!(
        INTERNER_CLEANUP_DEFERRAL_DEPTH.load(Ordering::Acquire),
        initial + 1,
    );
    first.finish();
    assert_eq!(INTERNER_CLEANUP_DEFERRAL_DEPTH.load(Ordering::Acquire), initial);
}

#[test]
fn interner_cleanup_threshold_has_one_concurrent_claimant() {
    use std::sync::{Arc, Barrier};

    let counter = Arc::new(AtomicUsize::new(INTERNER_CLEANUP_INTERVAL - 1));
    let in_progress = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(16));
    let workers: Vec<_> = (0..16)
        .map(|_| {
            let counter = Arc::clone(&counter);
            let in_progress = Arc::clone(&in_progress);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                usize::from(claim_interner_cleanup(&counter, &in_progress))
            })
        })
        .collect();
    let claims: usize = workers.into_iter().map(|worker| worker.join().unwrap()).sum();

    assert_eq!(claims, 1);
    assert!(in_progress.load(Ordering::Acquire));

    // Crossing additional intervals while the first caller is still
    // sweeping cannot claim another concurrent retain.
    for _ in 0..(INTERNER_CLEANUP_INTERVAL * 2) {
        assert!(!claim_interner_cleanup(&counter, &in_progress));
    }
    in_progress.store(false, Ordering::Release);
    assert!(claim_interner_cleanup(&counter, &in_progress));
}

#[test]
fn scoped_weight_bulk_ops_union_all_identities() {
    let mut cache = ScopedWeightOpCache::default();

    assert_eq!(cache.union_all(std::iter::empty::<&Weight>()), Weight::empty());
    assert_eq!(cache.union_all([&Weight::empty()]), Weight::empty());
    assert_eq!(cache.union_all([&Weight::all()]), Weight::all());
}

#[test]
fn scoped_weight_bulk_ops_union_all_matches_sequential_union() {
    let left = weight_for_tsid(1, &[(1, 3), (7, 8)]);
    let middle = weight_for_tsid(1, &[(2, 6)]);
    let right = weight_for_tsid(2, &[(10, 12)]);
    let weights = [&left, &middle, &right, &Weight::empty()];

    let mut cache = ScopedWeightOpCache::default();
    let bulk = cache.union_all(weights);

    let sequential = left.union(&middle).union(&right).union(&Weight::empty());
    assert_eq!(bulk, sequential);
}

#[test]
fn scoped_bulk_union_cache_matches_plain_multiway_across_repeated_calls() {
    let weights = (0u32..8)
        .map(|index| {
            Weight::from_per_tsid_token_sets([
                (index % 3, RangeSetBlaze::from_iter([index..=index + 5])),
                ((index + 1) % 3, RangeSetBlaze::from_iter([20 + index..=24 + index])),
            ])
        })
        .collect::<Vec<_>>();
    let refs = weights.iter().collect::<Vec<_>>();
    let expected = Weight::union_all(refs.iter().copied());

    let mut cache = ScopedWeightOpCache::default();
    let first = cache.union_all(refs.iter().copied());
    let entries_after_first = cache.bulk_token_union_entry_count();
    let second = cache.union_all(refs.iter().copied());

    assert_eq!(first, expected);
    assert_eq!(second, expected);
    assert!(entries_after_first > 0);
    assert_eq!(cache.bulk_token_union_entry_count(), entries_after_first);
}

#[test]
fn sorted_point_entry_union_matches_sequential_weight_union() {
    let first = shared_rangeset(rangeset_from_ranges([1..=3]));
    let second = shared_rangeset(rangeset_from_ranges([3..=5]));
    let third = shared_rangeset(rangeset_from_ranges([7..=9]));
    let entries = vec![
        (2, Arc::clone(&first)),
        (2, Arc::clone(&second)),
        (4, Arc::clone(&third)),
    ];
    let direct = Weight::union_sorted_point_entries(entries.clone());
    let sequential = entries.into_iter().fold(Weight::empty(), |acc, (tsid, tokens)| {
        acc.union(&Weight::from_per_tsid_shared(std::iter::once((tsid, tokens))))
    });

    assert_eq!(direct, sequential);
    assert_eq!(direct.tokens_for_tsid(2), rangeset_from_ranges([1..=5]));
    assert_eq!(direct.tokens_for_tsid(4), rangeset_from_ranges([7..=9]));
}

#[test]
fn multiway_union_disjoint_ranges_matches_sequential_union() {
    let alpha = RangeSetBlaze::from_iter([1..=3]);
    let beta = RangeSetBlaze::from_iter([7..=9]);
    let gamma = RangeSetBlaze::from_iter([12..=15]);
    let weights = [
        Weight::from_uniform(20..=21, gamma.clone()),
        Weight::from_uniform(0..=1, alpha.clone()),
        Weight::from_uniform(2..=3, alpha),
        Weight::from_uniform(7..=8, beta.clone()),
        Weight::from_uniform(12..=14, beta),
    ];
    let bulk = Weight::union_all(weights.iter());
    let sequential = weights
        .iter()
        .fold(Weight::empty(), |acc, weight| acc.union(weight));

    assert_eq!(bulk, sequential);
    assert_eq!(bulk.range_entries().count(), 4);
}

#[test]
fn multiway_union_matches_sequential_union_for_overlapping_ranges() {
    let mut weights = Vec::new();
    for index in 0..80u32 {
        weights.push(Weight::from_uniform(
            index..=index + 20,
            RangeSetBlaze::from_iter([index % 11..=(index % 11) + 3]),
        ));
    }
    weights.push(Weight::from_uniform(
        (u32::MAX - 2)..=u32::MAX,
        RangeSetBlaze::from_iter([99..=101]),
    ));

    let bulk = Weight::union_all(weights.iter());
    let direct = Weight::union_all_direct(weights.iter());
    let sequential = weights
        .iter()
        .fold(Weight::empty(), |acc, weight| acc.union(weight));

    assert_eq!(bulk, sequential);
    assert_eq!(direct, sequential);
}

#[test]
fn repeated_token_body_range_coalescing_preserves_union() {
    let alpha = shared_rangeset(RangeSetBlaze::from_iter([1..=5]));
    let beta = shared_rangeset(RangeSetBlaze::from_iter([7..=11]));
    let mut entries = Vec::new();
    for index in 0..2_400u32 {
        let start = index % 100;
        entries.push(WeightRangeEntry {
            start,
            end: start + 3,
            tokens: if index % 2 == 0 {
                Arc::clone(&alpha)
            } else {
                Arc::clone(&beta)
            },
        });
    }

    let sequential_union = entries.iter().fold(Weight::empty(), |acc, entry| {
        acc.union(&Weight::from_uniform(
            entry.start..=entry.end,
            entry.tokens.as_ref().clone(),
        ))
    });
    let coalesced = coalesce_repeated_token_body_ranges(entries);
    assert!(coalesced.len() < 10);
    let coalesced_union = coalesced.iter().fold(Weight::empty(), |acc, entry| {
        acc.union(&Weight::from_uniform(
            entry.start..=entry.end,
            entry.tokens.as_ref().clone(),
        ))
    });

    assert_eq!(coalesced_union, sequential_union);
}

#[test]
fn repeated_token_body_range_coalescing_skips_impossible_compression() {
    let token_bodies: Vec<_> = (0..100u32)
        .map(|token| shared_rangeset(RangeSetBlaze::from_iter([token..=token])))
        .collect();
    let entries: Vec<_> = (0..2_400u32)
        .map(|index| WeightRangeEntry {
            start: index % 200,
            end: index % 200,
            tokens: Arc::clone(&token_bodies[index as usize % token_bodies.len()]),
        })
        .collect();

    let unchanged = coalesce_repeated_token_body_ranges(entries);
    assert_eq!(unchanged.len(), 2_400);
}

#[test]
fn indexed_intersection_matches_generic_intersection() {
    fn assert_matches(sparse: Weight, dense: Weight) {
        let index = dense.intersection_index();
        clear_weight_op_caches();
        let indexed = sparse.intersection_with_index(&index);
        clear_weight_op_caches();
        let generic = sparse.intersection_uncached(&dense);
        assert_eq!(indexed, generic);
    }

    let dense = Weight::from_per_tsid_token_sets((0..160u32).map(|tsid| {
        let tokens = match tsid % 5 {
            0 => RangeSetBlaze::from_iter([0..=7, 40..=47]),
            1 => RangeSetBlaze::from_iter([4..=13]),
            2 => RangeSetBlaze::from_iter([20..=29]),
            3 => RangeSetBlaze::from_iter([8..=11, 30..=36]),
            _ => RangeSetBlaze::from_iter([50..=65]),
        };
        (tsid * 3, tokens)
    }));
    let sparse = Weight::from_per_tsid_token_sets([
        (2, RangeSetBlaze::from_iter([3..=10])),
        (93, RangeSetBlaze::from_iter([0..=5, 44..=52])),
        (231, RangeSetBlaze::from_iter([7..=35])),
        (351, RangeSetBlaze::from_iter([30..=70])),
    ]);
    assert_matches(sparse, dense.clone());

    let index = dense.intersection_index();
    assert_eq!(Weight::all().intersection_with_index(&index), dense);
    assert_eq!(Weight::empty().intersection_with_index(&index), Weight::empty());

    for case in 0..64u32 {
        let dense = Weight::from_per_tsid_token_sets((0..192u32).map(|tsid| {
            let tokens = match (tsid / 3 + case) % 7 {
                0 => RangeSetBlaze::from_iter([0..=9, 42..=57]),
                1 => RangeSetBlaze::from_iter([5..=18]),
                2 => RangeSetBlaze::from_iter([20..=37]),
                3 => RangeSetBlaze::from_iter([11..=14, 30..=45]),
                4 => RangeSetBlaze::from_iter([48..=66]),
                5 => RangeSetBlaze::from_iter([3..=7, 70..=79]),
                _ => RangeSetBlaze::from_iter([25..=31, 60..=73]),
            };
            (tsid, tokens)
        }));
        let sparse = Weight::from_per_tsid_token_sets((0..192u32).filter_map(|tsid| {
            if (tsid * 17 + case * 11) % 5 == 0 {
                return None;
            }
            let tokens = match (tsid / 2 + case * 3) % 9 {
                0 => RangeSetBlaze::from_iter([0..=6, 40..=53]),
                1 => RangeSetBlaze::from_iter([4..=15]),
                2 => RangeSetBlaze::from_iter([16..=29]),
                3 => RangeSetBlaze::from_iter([8..=12, 28..=38]),
                4 => RangeSetBlaze::from_iter([32..=51]),
                5 => RangeSetBlaze::from_iter([50..=69]),
                6 => RangeSetBlaze::from_iter([65..=82]),
                7 => RangeSetBlaze::from_iter([2..=4, 75..=91]),
                _ => RangeSetBlaze::from_iter([24..=27, 56..=64]),
            };
            Some((tsid, tokens))
        }));
        assert_matches(sparse, dense);
    }
}

#[test]
fn ranged_shared_entries_match_point_shared_entries() {
    let alpha = shared_rangeset(RangeSetBlaze::from_iter([1..=3]));
    let beta = shared_rangeset(RangeSetBlaze::from_iter([7..=9]));
    let gamma = shared_rangeset(RangeSetBlaze::from_iter([2..=8]));
    let ranges = [
        (0, 2, Arc::clone(&alpha)),
        (3, 5, Arc::clone(&beta)),
        (6, 6, Arc::clone(&alpha)),
        (8, 10, Arc::clone(&gamma)),
    ];
    let ranged = Weight::from_tsid_ranges_shared(ranges.iter().cloned());
    let points = Weight::from_per_tsid_shared(ranges.iter().flat_map(|(start, end, tokens)| {
        (*start..=*end).map(move |tsid| (tsid, Arc::clone(tokens)))
    }));
    assert_eq!(ranged, points);
}

#[test]
fn sparse_tsid_overrides_intersection_matches_two_step_form() {
    let alpha = shared_rangeset(RangeSetBlaze::from_iter([1..=3]));
    let beta = shared_rangeset(RangeSetBlaze::from_iter([4..=7]));
    let gamma = shared_rangeset(RangeSetBlaze::from_iter([2..=6]));
    let delta = shared_rangeset(RangeSetBlaze::from_iter([8..=10]));
    let base = Weight::from_per_tsid_shared([
        (0, Arc::clone(&alpha)),
        (1, Arc::clone(&alpha)),
        (2, Arc::clone(&alpha)),
        (3, Arc::clone(&alpha)),
        (5, Arc::clone(&beta)),
        (6, Arc::clone(&beta)),
        (7, Arc::clone(&beta)),
    ]);
    let domain = Weight::from_per_tsid_shared([
        (1, Arc::clone(&gamma)),
        (2, Arc::clone(&gamma)),
        (4, Arc::clone(&delta)),
        (5, Arc::clone(&beta)),
        (6, Arc::clone(&gamma)),
        (8, Arc::clone(&delta)),
        (10, Arc::clone(&alpha)),
    ]);
    let overrides = [
        (1, Arc::clone(&delta)),
        (4, Arc::clone(&alpha)),
        (6, Arc::clone(&EMPTY_RANGESET)),
        (8, Arc::clone(&gamma)),
        (10, Arc::clone(&beta)),
    ];

    clear_weight_op_caches();
    let expected = base.with_sparse_tsid_overrides(&overrides).intersection(&domain);
    clear_weight_op_caches();
    let actual = base.with_sparse_tsid_overrides_intersection(&overrides, &domain);
    assert_eq!(actual, expected);
}

#[test]
fn sparse_tsid_range_overrides_intersection_matches_point_form() {
    let alpha = shared_rangeset(RangeSetBlaze::from_iter([1..=3]));
    let beta = shared_rangeset(RangeSetBlaze::from_iter([4..=7]));
    let gamma = shared_rangeset(RangeSetBlaze::from_iter([2..=6]));
    let delta = shared_rangeset(RangeSetBlaze::from_iter([8..=10]));
    let base = Weight::from_tsid_ranges_shared([
        (0, 3, Arc::clone(&alpha)),
        (5, 9, Arc::clone(&beta)),
        (12, 15, Arc::clone(&gamma)),
    ]);
    let domain = Weight::from_tsid_ranges_shared([
        (1, 6, Arc::clone(&gamma)),
        (8, 13, Arc::clone(&delta)),
        (15, 18, Arc::clone(&alpha)),
    ]);
    let range_overrides = [
        (0, 1, Arc::clone(&delta)),
        (3, 6, Arc::clone(&alpha)),
        (8, 10, Arc::clone(&EMPTY_RANGESET)),
        (13, 17, Arc::clone(&beta)),
    ];
    let point_overrides = range_overrides
        .iter()
        .flat_map(|(start, end, tokens)| {
            (*start..=*end).map(move |tsid| (tsid, Arc::clone(tokens)))
        })
        .collect::<Vec<_>>();

    clear_weight_op_caches();
    let expected = base
        .with_sparse_tsid_overrides(&point_overrides)
        .intersection(&domain);
    clear_weight_op_caches();
    let actual =
        base.with_sparse_tsid_range_overrides_intersection(&range_overrides, &domain);
    assert_eq!(actual, expected);
}

#[test]
fn streaming_difference_matches_boundary_reference_exhaustively() {
    let weights = (0u32..64)
        .map(|code| {
            Weight::from_per_tsid_token_sets((0u32..3).filter_map(|tsid| {
                let token_bits = (code >> (tsid * 2)) & 0b11;
                (token_bits != 0).then(|| {
                    let tokens = (0u32..2)
                        .filter(|token| token_bits & (1 << token) != 0)
                        .collect::<RangeSetBlaze<_>>();
                    (tsid, tokens)
                })
            }))
        })
        .collect::<Vec<_>>();

    for left in &weights {
        for right in &weights {
            let streaming = difference_weights(left, right);
            let reference = combine_compact_entries(left, right, difference_token_sets);
            assert_eq!(
                streaming, reference,
                "streaming difference differs for left={left} right={right}",
            );
        }
    }
}

#[test]
fn scoped_weight_bulk_ops_difference_many_identities() {
    let base = weight_for_tsid(3, &[(4, 9)]);
    let mut cache = ScopedWeightOpCache::default();

    assert_eq!(cache.difference_many(&base, std::iter::empty::<&Weight>()), base);
    assert_eq!(cache.difference_many(&base, [&base]), Weight::empty());
}

#[test]
fn scoped_weight_bulk_ops_difference_many_matches_sequential_difference() {
    let base = Weight::union_all([
        &weight_for_tsid(1, &[(1, 5), (8, 10)]),
        &weight_for_tsid(2, &[(20, 24)]),
    ]);
    let subtract_left = weight_for_tsid(1, &[(2, 3), (9, 12)]);
    let subtract_right = weight_for_tsid(2, &[(21, 22)]);

    let mut cache = ScopedWeightOpCache::default();
    let bulk = cache.difference_many(&base, [&subtract_left, &subtract_right]);

    let sequential = base.difference(&subtract_left).difference(&subtract_right);
    assert_eq!(bulk, sequential);
}
