use once_cell::sync::Lazy;
use range_set_blaze::{
    CheckSortedDisjoint, CheckSortedDisjointMap, RangeMapBlaze, RangeSetBlaze,
    SortedDisjointMap,
};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use dashmap::DashMap;

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

// STICKY NOTE: DO NOT REMOVE THIS COMMENT.
//
// This module uses RangeSetBlaze and RangeMapBlaze as its core data structures.
// Performance characteristics are counterintuitive and differ from naive bitmaps
// or hash maps. Read this note before writing hot-path code that creates,
// mutates, or queries Weight objects.
//
// 1. RangeSetBlaze / RangeMapBlaze complexity
//    Cost is proportional to the NUMBER OF RANGES, not the numeric span covered.
//    A set covering 0..=1_000_000 as a single range is much cheaper than 1_000_001
//    individual points stored as 1_000_001 singleton ranges.
//    RangeMapBlaze merges adjacent keys only when the VALUES are also equal.
//    A Weight with 10 key-ranges each mapping to a 3-range token set can be
//    more expensive than a Weight with 50 key-ranges each mapping to a 1-range
//    token set, depending on operation mix.
//
// 2. Why remapping / rearranging IDs matters
//    If many Weights will be stored or queried together, arrange numeric IDs
//    so that each Weight's inner sets/maps form FEWER TOTAL RANGES.
//    The target is NOT "small max ID"; it is fewer ranges across the relevant
//    weights. In DWA/id-map/possible-matches-style code this means:
//    - Group IDs that co-occur in the same token sets into contiguous ranges.
//    - Recompact parser-state and terminal IDs jointly with token vocab IDs
//      when the weights are built together.
//    - A renumbering that cuts total unique ranges by 2x often beats a
//      renumbering that merely lowers the max ID.
//
// 3. Two-level interning
//    Outer Weight maps (RangeMapBlaze<u32, SharedTokenSet>) are interned via
//    GLOBAL_WEIGHTS. Inner RangeSetBlaze values are interned via
//    GLOBAL_TOKEN_SETS. Both use Arc deduplication.
//    When measuring static complexity of a collection of weights, count UNIQUE
//    interned weights once, and count UNIQUE inner rangesets once, not once per
//    key-range occurrence. Example: if the same weight appears 100 times in a
//    collection of 101 weights, its static cost is roughly that weight plus the
//    1 other unique weight, plus their shared inner rangesets. Do not multiply by 100.
//
//    Static complexity model for DWA / possible-matches recompaction:
//    - For a unique Weight, count its unique outer key ranges, plus the ranges
//      in each unique interned inner RangeSetBlaze (counted once per unique
//      inner set, not once per reference).
//    - For a collection of weights, count each unique interned Weight once and
//      each unique inner set once.
//    - This is the mental model behind minimizing
//      total_outer_ranges + total_inner_ranges in DWA::stats()-style accounting.
//
// 4. Mutation pitfalls
//    Small repeated mutations (insert one token, union one range) can be
//    surprisingly expensive because each operation may trigger normalization
//    and intern-table lookup. In hot paths, avoid building a Weight one item
//    at a time. Prefer construction APIs that collect from sorted/ranged data
//    (e.g. CompactRangeBuilder, from_per_tsid_shared)
//    so normalization and interning happen once at the end.
//
// 5. Lookup / iteration implications
//    Lookups are O(log num_key_ranges). Iteration yields ranges, not individual
//    points. Union / intersection iterate over both operands' ranges in lockstep.
//    The cheap case is when both operands have very few ranges or are the same
//    interned Arc (fast path: Arc::ptr_eq). The expensive case is many
//    misaligned small ranges on both sides.
//    Cloning is cheap because it is just an Arc clone of the interned map, but
//    only until you mutate; then a fresh normalized map must be built.
//
// 6. Practical guidance
//    - Prefer dense contiguous IDs for things that co-occur in the same sets.
//    - Recompact / remap based on the WHOLE COLLECTION of weights that will be
//      queried or stored together, not per-weight in isolation.
//    - When designing optimizers, consider the interning boundary: reducing
//      total unique interned weights and unique inner rangesets is often more
//      valuable than shrinking any single weight's local range count.
//    - If you must measure, measure total unique interned structures and total
//      ranges across the representative workload, not max ID or per-weight size.
//
// DO NOT REMOVE THIS NOTE. Future maintainers will need it.

#[derive(Debug, Clone)]
pub struct Weight(Arc<WeightMap>);

#[derive(Clone)]
struct ScopedWeightOpEntry {
    // Pointer keys are only valid while they still identify the same operands.
    // Weak refs let us validate that identity without retaining every temporary
    // operand for the full determinization pass.
    left_operand: Weak<WeightMap>,
    right_operand: Weak<WeightMap>,
    result: Weight,
}

impl ScopedWeightOpEntry {
    fn matches(&self, left: &Weight, right: &Weight) -> bool {
        let Some(cached_left) = self.left_operand.upgrade() else {
            return false;
        };
        let Some(cached_right) = self.right_operand.upgrade() else {
            return false;
        };
        Arc::ptr_eq(&cached_left, &left.0) && Arc::ptr_eq(&cached_right, &right.0)
    }
}

#[derive(Default)]
pub struct ScopedWeightOpCache {
    union_entries: FxHashMap<(usize, usize), ScopedWeightOpEntry>,
    intersection_entries: FxHashMap<(usize, usize), ScopedWeightOpEntry>,
    difference_entries: FxHashMap<(usize, usize), ScopedWeightOpEntry>,
    bulk_token_union_entries: FxHashMap<Vec<usize>, SharedTokenSet>,
}

impl ScopedWeightOpCache {
    pub fn union(&mut self, left: &Weight, right: &Weight) -> Weight {
        if left.is_full() || right.is_full() {
            return Weight::all();
        }
        if left.is_empty() {
            return right.clone();
        }
        if right.is_empty() {
            return left.clone();
        }
        if Arc::ptr_eq(&left.0, &right.0) {
            return left.clone();
        }

        let (key, ordered_left, ordered_right) = ordered_commutative_weight_pair(left, right);
        if let Some(existing) = self.union_entries.get(&key) {
            if existing.matches(ordered_left, ordered_right) {
                return existing.result.clone();
            }
            self.union_entries.remove(&key);
        }

        let value = left.union_uncached(right);
        self.union_entries.insert(
            key,
            ScopedWeightOpEntry {
                left_operand: Arc::downgrade(&ordered_left.0),
                right_operand: Arc::downgrade(&ordered_right.0),
                result: value.clone(),
            },
        );
        value
    }

    pub fn intersection(&mut self, left: &Weight, right: &Weight) -> Weight {
        if left.is_empty() || right.is_empty() {
            return Weight::empty();
        }
        if Arc::ptr_eq(&left.0, &right.0) {
            return left.clone();
        }
        if left.is_full() {
            return right.clone();
        }
        if right.is_full() {
            return left.clone();
        }

        let (key, ordered_left, ordered_right) = ordered_commutative_weight_pair(left, right);
        if let Some(existing) = self.intersection_entries.get(&key) {
            if existing.matches(ordered_left, ordered_right) {
                return existing.result.clone();
            }
            self.intersection_entries.remove(&key);
        }

        let value = left.intersection_uncached_impl(right);
        self.intersection_entries.insert(
            key,
            ScopedWeightOpEntry {
                left_operand: Arc::downgrade(&ordered_left.0),
                right_operand: Arc::downgrade(&ordered_right.0),
                result: value.clone(),
            },
        );
        value
    }

    pub fn union_entry_count(&self) -> usize {
        self.union_entries.len()
    }

    pub fn intersection_entry_count(&self) -> usize {
        self.intersection_entries.len()
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn bulk_token_union_entry_count(&self) -> usize {
        self.bulk_token_union_entries.len()
    }

    pub fn difference(&mut self, left: &Weight, right: &Weight) -> Weight {
        if left.is_empty() || right.is_full() {
            return Weight::empty();
        }
        if right.is_empty() {
            return left.clone();
        }
        if Arc::ptr_eq(&left.0, &right.0) {
            return Weight::empty();
        }
        if left.is_full() {
            return Weight::all();
        }

        let key = (left.ptr_key(), right.ptr_key());
        if let Some(existing) = self.difference_entries.get(&key) {
            if existing.matches(left, right) {
                return existing.result.clone();
            }
            self.difference_entries.remove(&key);
        }

        let value = left.difference_uncached(right);
        self.difference_entries.insert(
            key,
            ScopedWeightOpEntry {
                left_operand: Arc::downgrade(&left.0),
                right_operand: Arc::downgrade(&right.0),
                result: value.clone(),
            },
        );
        value
    }

    /// Scoped bulk intersection with mathematical identity semantics.
    /// An empty input intersects to `Weight::all()`.
    pub fn intersection_all<'a>(&mut self, weights: impl IntoIterator<Item = &'a Weight>) -> Weight {
        let mut meaningful = SmallVec::<[&Weight; 8]>::new();
        for weight in weights {
            if weight.is_empty() {
                return Weight::empty();
            }
            if weight.is_full() {
                continue;
            }
            meaningful.push(weight);
        }

        match meaningful.len() {
            0 => Weight::all(),
            1 => meaningful[0].clone(),
            _ => {
                meaningful.sort_unstable_by_key(|weight| weight.ptr_key());
                meaningful.dedup_by_key(|weight| weight.ptr_key());
                let mut iter = meaningful.into_iter();
                let mut acc = iter.next().unwrap().clone();
                for weight in iter {
                    acc = self.intersection(&acc, weight);
                    if acc.is_empty() {
                        return acc;
                    }
                }
                acc
            }
        }
    }

    pub fn union_all<'a>(&mut self, weights: impl IntoIterator<Item = &'a Weight>) -> Weight {
        let mut meaningful = SmallVec::<[&Weight; 8]>::new();
        for weight in weights {
            if weight.is_full() {
                return Weight::all();
            }
            if weight.is_empty() {
                continue;
            }
            meaningful.push(weight);
        }

        match meaningful.len() {
            0 => Weight::empty(),
            1 => meaningful[0].clone(),
            _ if meaningful.len() > 4 => {
                meaningful.sort_unstable_by_key(|weight| weight.ptr_key());
                meaningful.dedup_by_key(|weight| weight.ptr_key());
                if meaningful.len() == 1 {
                    meaningful[0].clone()
                } else if let Some(result) = union_all_single_tsid_entries(&meaningful) {
                    result
                } else if meaningful.len() > 4 {
                    union_all_multiway_with_token_cache(
                        &meaningful,
                        &mut self.bulk_token_union_entries,
                    )
                } else {
                    let mut iter = meaningful.into_iter();
                    let mut acc = iter.next().unwrap().clone();
                    for weight in iter {
                        acc = self.union(&acc, weight);
                    }
                    acc
                }
            }
            _ => {
                if let Some(result) = union_all_single_tsid_entries(&meaningful) {
                    result
                } else {
                    let mut iter = meaningful.into_iter();
                    let mut acc = iter.next().unwrap().clone();
                    for weight in iter {
                        acc = self.union(&acc, weight);
                    }
                    acc
                }
            }
        }
    }

    pub fn difference_many<'a>(
        &mut self,
        base: &Weight,
        subtracts: impl IntoIterator<Item = &'a Weight>,
    ) -> Weight {
        let mut acc = base.clone();
        for weight in subtracts {
            acc = self.difference(&acc, weight);
            if acc.is_empty() {
                return acc;
            }
        }
        acc
    }
}

pub type SharedTokenSet = Arc<RangeSetBlaze<u32>>;
type WeightMap = RangeMapBlaze<u32, SharedTokenSet>;

const INTERNER_CLEANUP_INTERVAL: usize = 1024;

// Sharded interner: DashMap provides internal striping (~16 shards) so concurrent
// intern calls can run in parallel on different keys. Previously we used a single
// `Mutex<GlobalWeightInterner>` which serialized all weight-op fresh constructions.
// Store token-set buckets by a compact structural fingerprint instead of using
// the full RangeSetBlaze as the DashMap key.  The old representation forced a
// clone of every newly interned set into the map in addition to the Arc-owned
// copy, which is especially expensive while deserializing large constraints.
// Equality inside a fingerprint bucket preserves exact interning semantics.
static GLOBAL_TOKEN_SETS: Lazy<DashMap<u64, Vec<Weak<RangeSetBlaze<u32>>>>> =
    Lazy::new(DashMap::new);
static GLOBAL_WEIGHTS: Lazy<DashMap<u64, Vec<Weak<WeightMap>>>> = Lazy::new(DashMap::new);
static TOKEN_INSERTS_SINCE_CLEANUP: AtomicUsize = AtomicUsize::new(0);
static WEIGHT_INSERTS_SINCE_CLEANUP: AtomicUsize = AtomicUsize::new(0);
static TOKEN_CLEANUP_IN_PROGRESS: AtomicBool = AtomicBool::new(false);
static WEIGHT_CLEANUP_IN_PROGRESS: AtomicBool = AtomicBool::new(false);
static INTERNER_CLEANUP_DEFERRAL_DEPTH: AtomicUsize = AtomicUsize::new(0);

static EMPTY_RANGESET: Lazy<SharedTokenSet> = Lazy::new(|| Arc::new(RangeSetBlaze::new()));

static EMPTY_WEIGHT: Lazy<Weight> = Lazy::new(|| Weight(Arc::new(WeightMap::new())));

static ALL_WEIGHT: Lazy<Weight> = Lazy::new(|| {
    let mut map = WeightMap::new();
    map.extend_simple(std::iter::once((
        WEIGHT_ALL_SENTINEL..=WEIGHT_ALL_SENTINEL,
        shared_rangeset(sentinel_token_set()),
    )));
    finalize_weight_map(map)
});

/// Defers expensive global interner sweeps until the last active compiler
/// exits. Fresh interning remains exact; only the timing of weak-entry removal
/// changes.
pub struct WeightInternerCleanupDeferral {
    active: bool,
}

pub fn defer_weight_interner_cleanup() -> WeightInternerCleanupDeferral {
    INTERNER_CLEANUP_DEFERRAL_DEPTH.fetch_add(1, Ordering::AcqRel);
    WeightInternerCleanupDeferral { active: true }
}

impl WeightInternerCleanupDeferral {
    pub fn finish(mut self) {
        self.release();
    }

    fn release(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        if INTERNER_CLEANUP_DEFERRAL_DEPTH.fetch_sub(1, Ordering::AcqRel) == 1 {
            interner_clear_stale();
        }
    }
}

impl Drop for WeightInternerCleanupDeferral {
    fn drop(&mut self) {
        self.release();
    }
}

fn prune_dead_token_sets() {
    GLOBAL_TOKEN_SETS.retain(|_, bucket| {
        bucket.retain(|weak| weak.strong_count() > 0);
        !bucket.is_empty()
    });
}

fn prune_dead_weights() {
    GLOBAL_WEIGHTS.retain(|_, bucket| {
        bucket.retain(|weak| weak.strong_count() > 0);
        !bucket.is_empty()
    });
}

/// Count a fresh interner insertion and claim a global sweep exactly once.
///
/// `DashMap::retain` scans every shard. A completed cleanup resets the counter,
/// but producers may cross another interval while that scan is still running.
/// The in-progress gate ensures those later producers never start an overlapping
/// scan; after the owner releases the gate, the accumulated counter makes a
/// subsequent insertion schedule the next sweep normally.
#[inline]
fn claim_interner_cleanup(counter: &AtomicUsize, in_progress: &AtomicBool) -> bool {
    let previous = counter.fetch_add(1, Ordering::Relaxed);
    if previous + 1 < INTERNER_CLEANUP_INTERVAL {
        return false;
    }
    if in_progress
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
        .is_err()
    {
        return false;
    }
    counter.store(0, Ordering::Release);
    true
}

fn maybe_cleanup_token_sets() {
    if INTERNER_CLEANUP_DEFERRAL_DEPTH.load(Ordering::Acquire) != 0 {
        return;
    }
    if claim_interner_cleanup(&TOKEN_INSERTS_SINCE_CLEANUP, &TOKEN_CLEANUP_IN_PROGRESS) {
        prune_dead_token_sets();
        TOKEN_CLEANUP_IN_PROGRESS.store(false, Ordering::Release);
    }
}

fn maybe_cleanup_weights() {
    if INTERNER_CLEANUP_DEFERRAL_DEPTH.load(Ordering::Acquire) != 0 {
        return;
    }
    if claim_interner_cleanup(&WEIGHT_INSERTS_SINCE_CLEANUP, &WEIGHT_CLEANUP_IN_PROGRESS) {
        prune_dead_weights();
        WEIGHT_CLEANUP_IN_PROGRESS.store(false, Ordering::Release);
    }
}

fn interner_clear_all() {
    GLOBAL_TOKEN_SETS.clear();
    GLOBAL_WEIGHTS.clear();
    TOKEN_INSERTS_SINCE_CLEANUP.store(0, Ordering::Relaxed);
    WEIGHT_INSERTS_SINCE_CLEANUP.store(0, Ordering::Relaxed);
    TOKEN_CLEANUP_IN_PROGRESS.store(false, Ordering::Release);
    WEIGHT_CLEANUP_IN_PROGRESS.store(false, Ordering::Release);
}

fn interner_clear_stale() {
    prune_dead_token_sets();
    prune_dead_weights();
    TOKEN_INSERTS_SINCE_CLEANUP.store(0, Ordering::Relaxed);
    WEIGHT_INSERTS_SINCE_CLEANUP.store(0, Ordering::Relaxed);
    TOKEN_CLEANUP_IN_PROGRESS.store(false, Ordering::Release);
    WEIGHT_CLEANUP_IN_PROGRESS.store(false, Ordering::Release);
}


fn rangeset_fingerprint(tokens: &RangeSetBlaze<u32>) -> u64 {
    use std::hash::Hasher;

    let mut hasher = rustc_hash::FxHasher::default();
    for range in tokens.ranges() {
        hasher.write_u32(*range.start());
        hasher.write_u32(*range.end());
    }
    hasher.finish()
}

fn intern_rangeset(tokens: RangeSetBlaze<u32>) -> SharedTokenSet {
    if tokens.is_empty() {
        return Arc::clone(&EMPTY_RANGESET);
    }

    let fingerprint = rangeset_fingerprint(&tokens);
    let mut bucket = GLOBAL_TOKEN_SETS.entry(fingerprint).or_default();
    let mut idx = 0usize;
    while idx < bucket.len() {
        let Some(existing) = bucket[idx].upgrade() else {
            bucket.swap_remove(idx);
            continue;
        };
        if existing.as_ref() == &tokens {
            return existing;
        }
        idx += 1;
    }
    let shared = Arc::new(tokens);
    bucket.push(Arc::downgrade(&shared));
    drop(bucket);
    maybe_cleanup_token_sets();
    shared
}

fn weight_map_fingerprint(map: &WeightMap) -> u64 {
    use std::hash::Hasher;

    let mut hasher = rustc_hash::FxHasher::default();
    for (range, tokens) in map.range_values() {
        hasher.write_u32(*range.start());
        hasher.write_u32(*range.end());
        hasher.write_usize(Arc::as_ptr(tokens) as usize);
    }
    hasher.finish()
}

fn weight_map_eq(left: &WeightMap, right: &WeightMap) -> bool {
    let mut left_iter = left.range_values();
    let mut right_iter = right.range_values();
    loop {
        match (left_iter.next(), right_iter.next()) {
            (None, None) => return true,
            (Some((left_range, left_tokens)), Some((right_range, right_tokens))) => {
                if left_range != right_range {
                    return false;
                }
                if !same_shared_token_set(left_tokens, right_tokens) {
                    return false;
                }
            }
            _ => return false,
        }
    }
}

fn intern_weight_map(map: WeightMap) -> Arc<WeightMap> {
    let fingerprint = weight_map_fingerprint(&map);
    let mut bucket = GLOBAL_WEIGHTS.entry(fingerprint).or_default();
    let mut idx = 0usize;
    while idx < bucket.len() {
        let Some(existing) = bucket[idx].upgrade() else {
            bucket.swap_remove(idx);
            continue;
        };
        if weight_map_eq(existing.as_ref(), &map) {
            return existing;
        }
        idx += 1;
    }
    let shared = Arc::new(map);
    bucket.push(Arc::downgrade(&shared));
    drop(bucket);
    maybe_cleanup_weights();
    shared
}

fn same_shared_token_set(left: &SharedTokenSet, right: &SharedTokenSet) -> bool {
    Arc::ptr_eq(left, right) || left.as_ref() == right.as_ref()
}

fn lookup_memoized_token_set_op(
    kind: TokenSetOpKind,
    left: &SharedTokenSet,
    right: &SharedTokenSet,
) -> Option<SharedTokenSet> {
    with_weight_op_memo(|memo| memo.lookup_token_set(TokenSetOpKey::for_token_sets(kind, left, right)))
}

fn store_memoized_token_set_op(
    kind: TokenSetOpKind,
    left: &SharedTokenSet,
    right: &SharedTokenSet,
    result: &SharedTokenSet,
) {
    with_weight_op_memo(|memo| {
        memo.store_token_set(TokenSetOpKey::for_token_sets(kind, left, right), left, right, result)
    });
}

fn shared_token_union(left: &SharedTokenSet, right: &SharedTokenSet) -> SharedTokenSet {
    if same_shared_token_set(left, right) || left.as_ref().is_subset(right.as_ref()) {
        Arc::clone(right)
    } else if right.as_ref().is_subset(left.as_ref()) {
        Arc::clone(left)
    } else if let Some(existing) = lookup_memoized_token_set_op(TokenSetOpKind::Union, left, right) {
        existing
    } else {
        let result = shared_rangeset(left.as_ref().clone() | right.as_ref().clone());
        store_memoized_token_set_op(TokenSetOpKind::Union, left, right, &result);
        result
    }
}

fn shared_token_union_many(tokens: &[SharedTokenSet]) -> Option<SharedTokenSet> {
    match tokens.len() {
        0 => None,
        1 => Some(Arc::clone(&tokens[0])),
        2 => Some(shared_token_union(&tokens[0], &tokens[1])),
        _ => {
            let mut ranges = Vec::<(u32, u32)>::new();
            for token_set in tokens {
                ranges.extend(
                    token_set
                        .ranges()
                        .map(|range| (*range.start(), *range.end())),
                );
            }
            if ranges.is_empty() {
                return None;
            }
            ranges.sort_unstable();

            let mut merged = Vec::with_capacity(ranges.len());
            let mut current = ranges[0];
            for (start, end) in ranges.into_iter().skip(1) {
                if start <= current.1.saturating_add(1) {
                    current.1 = current.1.max(end);
                } else {
                    merged.push(current.0..=current.1);
                    current = (start, end);
                }
            }
            merged.push(current.0..=current.1);

            Some(shared_rangeset(RangeSetBlaze::from_iter(merged)))
        }
    }
}

fn shared_token_intersection(
    left: &SharedTokenSet,
    right: &SharedTokenSet,
) -> Option<SharedTokenSet> {
    static MODE: Lazy<u8> = Lazy::new(|| {
        token_intersection_mode(std::env::var("GLRMASK_EXPERIMENT_TOKEN_INTERSECTION").ok().as_deref())
    });
    match *MODE {
        1 => shared_token_intersection_cached(left, right, false),
        2 => shared_token_intersection_cached(left, right, true),
        _ => shared_token_intersection_legacy(left, right),
    }
}

fn token_intersection_mode(value: Option<&str>) -> u8 {
    // The one-pass implementation preserves the legacy operand representative
    // as well as the exact intersection. Keep the original algorithm available
    // for differential tests and interleaved performance measurements.
    match value {
        None | Some("sweep") => 2,
        Some("memo") => 1,
        _ => 0,
    }
}

fn shared_token_intersection_legacy(
    left: &SharedTokenSet,
    right: &SharedTokenSet,
) -> Option<SharedTokenSet> {
    if same_shared_token_set(left, right) || left.as_ref().is_subset(right.as_ref()) {
        Some(Arc::clone(left))
    } else if right.as_ref().is_subset(left.as_ref()) {
        Some(Arc::clone(right))
    } else if let Some(existing) =
        lookup_memoized_token_set_op(TokenSetOpKind::Intersection, left, right)
    {
        (!existing.is_empty()).then_some(existing)
    } else {
        let overlap = left.as_ref() & right.as_ref();
        let result = shared_rangeset(overlap);
        store_memoized_token_set_op(TokenSetOpKind::Intersection, left, right, &result);
        (!result.is_empty()).then_some(result)
    }
}

/// Intersect two canonical disjoint interval sequences in one merge pass.
/// Cardinality identifies containment without a second traversal: A∩B is a
/// subset of A and B, so equal finite cardinality proves equality. Arithmetic
/// is u64, including the complete u32 token domain of size 2^32.
fn intersect_token_ranges_once(left: &SharedTokenSet, right: &SharedTokenSet) -> SharedTokenSet {
    let mut li = left.ranges();
    let mut ri = right.ranges();
    let mut l = li.next();
    let mut r = ri.next();
    let mut ranges = SmallVec::<[std::ops::RangeInclusive<u32>; 8]>::new();
    let mut cardinality = 0_u64;
    while let (Some(lr), Some(rr)) = (&l, &r) {
        let start = (*lr.start()).max(*rr.start());
        let end = (*lr.end()).min(*rr.end());
        if start <= end {
            ranges.push(start..=end);
            cardinality += u64::from(end) - u64::from(start) + 1;
        }
        let left_end = *lr.end();
        let right_end = *rr.end();
        if left_end <= right_end { l = li.next(); }
        if right_end <= left_end { r = ri.next(); }
    }
    if cardinality == left.len() {
        Arc::clone(left)
    } else if cardinality == right.len() {
        Arc::clone(right)
    } else if cardinality == 0 {
        Arc::clone(&EMPTY_RANGESET)
    } else {
        shared_rangeset(RangeSetBlaze::from_sorted_disjoint(
            CheckSortedDisjoint::new(ranges.into_iter()),
        ))
    }
}

fn shared_token_intersection_cached(
    left: &SharedTokenSet,
    right: &SharedTokenSet,
    single_pass: bool,
) -> Option<SharedTokenSet> {
    // Preserve the established Some(empty) result for an empty operand.
    if Arc::ptr_eq(left, right) || left.is_empty() { return Some(Arc::clone(left)); }
    if right.is_empty() { return Some(Arc::clone(right)); }
    if let Some(existing) = lookup_memoized_token_set_op(TokenSetOpKind::Intersection, left, right) {
        // The memo key is commutative, but the old containment shortcut picks
        // the current left allocation when the two sets are equal. Preserve
        // that representative in O(1) so pointer-based graph sharing does not
        // depend on which call direction populated the cache first.
        if existing.len() == left.len() { return Some(Arc::clone(left)); }
        if existing.len() == right.len() { return Some(Arc::clone(right)); }
        return (!existing.is_empty()).then_some(existing);
    }
    let result = if single_pass {
        intersect_token_ranges_once(left, right)
    } else if left.len() <= right.len() && left.is_subset(right) {
        Arc::clone(left)
    } else if right.len() <= left.len() && right.is_subset(left) {
        Arc::clone(right)
    } else {
        shared_rangeset(left.as_ref() & right.as_ref())
    };
    // Containment results are worth caching too: in large parser unions the
    // same pair otherwise pays for the subset scan on every invocation.
    store_memoized_token_set_op(TokenSetOpKind::Intersection, left, right, &result);
    (!result.is_empty()).then_some(result)
}

fn shared_token_difference(
    left: &SharedTokenSet,
    right: &SharedTokenSet,
) -> Option<SharedTokenSet> {
    if same_shared_token_set(left, right) || left.as_ref().is_subset(right.as_ref()) {
        None
    } else if left.as_ref().is_disjoint(right.as_ref()) {
        Some(Arc::clone(left))
    } else {
        let difference = left.as_ref().clone() - right.as_ref().clone();
        (!difference.is_empty()).then(|| shared_rangeset(difference))
    }
}

fn union_token_sets(
    left: Option<&SharedTokenSet>,
    right: Option<&SharedTokenSet>,
) -> Option<SharedTokenSet> {
    match (left, right) {
        (Some(left_tokens), Some(right_tokens)) => {
            Some(shared_token_union(left_tokens, right_tokens))
        }
        (Some(tokens), None) | (None, Some(tokens)) => Some(Arc::clone(tokens)),
        (None, None) => None,
    }
}

fn intersect_token_sets(
    left: Option<&SharedTokenSet>,
    right: Option<&SharedTokenSet>,
) -> Option<SharedTokenSet> {
    match (left, right) {
        (Some(left_tokens), Some(right_tokens)) => {
            shared_token_intersection(left_tokens, right_tokens)
        }
        _ => None,
    }
}

fn difference_token_sets(
    left: Option<&SharedTokenSet>,
    right: Option<&SharedTokenSet>,
) -> Option<SharedTokenSet> {
    match (left, right) {
        (Some(left_tokens), Some(right_tokens)) => {
            shared_token_difference(left_tokens, right_tokens)
        }
        (Some(left_tokens), None) => Some(Arc::clone(left_tokens)),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum WeightOpKind {
    Union,
    Intersection,
    Difference,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum TokenSetOpKind {
    Union,
    Intersection,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct WeightOpKey {
    kind: WeightOpKind,
    left: usize,
    right: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TokenSetOpKey {
    kind: TokenSetOpKind,
    left: usize,
    right: usize,
}

#[inline]
fn scoped_commutative_weight_pair_key(left: &Weight, right: &Weight) -> (usize, usize) {
    let left_key = left.ptr_key();
    let right_key = right.ptr_key();
    if left_key <= right_key {
        (left_key, right_key)
    } else {
        (right_key, left_key)
    }
}

#[inline]
fn ordered_commutative_weight_pair<'a>(
    left: &'a Weight,
    right: &'a Weight,
) -> ((usize, usize), &'a Weight, &'a Weight) {
    let left_key = left.ptr_key();
    let right_key = right.ptr_key();
    if left_key <= right_key {
        ((left_key, right_key), left, right)
    } else {
        ((right_key, left_key), right, left)
    }
}

impl WeightOpKey {
    fn new(kind: WeightOpKind, left: usize, right: usize) -> Self {
        match kind {
            WeightOpKind::Union | WeightOpKind::Intersection if left > right => Self {
                kind,
                left: right,
                right: left,
            },
            _ => Self { kind, left, right },
        }
    }

    fn for_weights(kind: WeightOpKind, left: &Weight, right: &Weight) -> Self {
        Self::new(kind, left.ptr_key(), right.ptr_key())
    }
}

impl TokenSetOpKey {
    fn new(kind: TokenSetOpKind, left: usize, right: usize) -> Self {
        if left > right {
            Self {
                kind,
                left: right,
                right: left,
            }
        } else {
            Self { kind, left, right }
        }
    }

    fn for_token_sets(kind: TokenSetOpKind, left: &SharedTokenSet, right: &SharedTokenSet) -> Self {
        Self::new(kind, Arc::as_ptr(left) as usize, Arc::as_ptr(right) as usize)
    }
}

/// Cached memo entry: stores the result AND weak references to both operands.
/// The operand weak refs guard against the ABA problem: if either operand's
/// Arc was dropped and a new Arc reuses the same address, the operand weak
/// ref will fail to upgrade, and the stale entry is discarded.
struct WeightOpMemoEntry {
    result: Weak<WeightMap>,
    left_operand: Weak<WeightMap>,
    right_operand: Weak<WeightMap>,
}

struct TokenSetOpMemoEntry {
    result: Weak<RangeSetBlaze<u32>>,
    left_operand: Weak<RangeSetBlaze<u32>>,
    right_operand: Weak<RangeSetBlaze<u32>>,
}

/// Global generation counter. Incremented by `clear_weight_op_caches()` so
/// that every thread's thread-local memo detects the invalidation on next access.
static WEIGHT_OP_MEMO_GENERATION: AtomicU64 = AtomicU64::new(0);
static WEIGHT_HASH_MEMO_GENERATION: AtomicU64 = AtomicU64::new(0);
const PUBLIC_WEIGHT_INTERSECTION_CACHE_CAP: usize = 262_144;

struct PublicWeightIntersectionMemoEntry {
    left: Weak<WeightMap>,
    right: Weak<WeightMap>,
    result: Weak<WeightMap>,
}

#[derive(Default)]
struct PublicWeightIntersectionMemo {
    results: FxHashMap<(usize, usize), PublicWeightIntersectionMemoEntry>,
    generation: u64,
}

impl PublicWeightIntersectionMemo {
    fn clear_all(&mut self) {
        self.results.clear();
    }

    fn lookup(&mut self, left: &Weight, right: &Weight) -> Option<Weight> {
        let (key, ordered_left, ordered_right) = ordered_commutative_weight_pair(left, right);
        let entry = self.results.get(&key)?;
        let cached_left = entry.left.upgrade();
        let cached_right = entry.right.upgrade();
        let cached_result = entry.result.upgrade();
        if let (Some(cached_left), Some(cached_right), Some(cached_result)) =
            (cached_left, cached_right, cached_result)
        {
            if Arc::ptr_eq(&cached_left, &ordered_left.0)
                && Arc::ptr_eq(&cached_right, &ordered_right.0)
            {
                return Some(Weight(cached_result));
            }
        }
        self.results.remove(&key);
        None
    }

    fn store(&mut self, left: &Weight, right: &Weight, result: &Weight) {
        if self.results.len() >= PUBLIC_WEIGHT_INTERSECTION_CACHE_CAP {
            self.clear_all();
        }
        let (key, ordered_left, ordered_right) = ordered_commutative_weight_pair(left, right);
        self.results.insert(
            key,
            PublicWeightIntersectionMemoEntry {
                left: Arc::downgrade(&ordered_left.0),
                right: Arc::downgrade(&ordered_right.0),
                result: Arc::downgrade(&result.0),
            },
        );
    }
}

#[derive(Default)]
struct WeightOpMemo {
    results: FxHashMap<WeightOpKey, WeightOpMemoEntry>,
    token_set_results: FxHashMap<TokenSetOpKey, TokenSetOpMemoEntry>,
    inserts_since_cleanup: usize,
    /// The generation this memo was last synchronised with.
    generation: u64,
}

impl WeightOpMemo {
    fn maybe_cleanup(&mut self) {
        if self.inserts_since_cleanup < INTERNER_CLEANUP_INTERVAL {
            return;
        }
        self.results.retain(|_, entry| entry.result.strong_count() > 0);
        self.token_set_results
            .retain(|_, entry| entry.result.strong_count() > 0);
        self.inserts_since_cleanup = 0;
    }

    fn clear_all(&mut self) {
        self.results.clear();
        self.token_set_results.clear();
        self.inserts_since_cleanup = 0;
    }

    fn lookup(&mut self, key: WeightOpKey) -> Option<Weight> {
        let entry = self.results.get(&key)?;
        if entry.left_operand.strong_count() == 0 || entry.right_operand.strong_count() == 0 {
            self.results.remove(&key);
            return None;
        }

        entry.result.upgrade().map(Weight)
    }

    fn store(&mut self, key: WeightOpKey, left: &Weight, right: &Weight, result: &Weight) {
        self.maybe_cleanup();
        self.results.insert(
            key,
            WeightOpMemoEntry {
                result: Arc::downgrade(&result.0),
                left_operand: Arc::downgrade(&left.0),
                right_operand: Arc::downgrade(&right.0),
            },
        );
        self.inserts_since_cleanup += 1;
    }

    fn lookup_token_set(&mut self, key: TokenSetOpKey) -> Option<SharedTokenSet> {
        let entry = self.token_set_results.get(&key)?;
        if entry.left_operand.strong_count() == 0 || entry.right_operand.strong_count() == 0 {
            self.token_set_results.remove(&key);
            return None;
        }

        entry.result.upgrade()
    }

    fn store_token_set(
        &mut self,
        key: TokenSetOpKey,
        left: &SharedTokenSet,
        right: &SharedTokenSet,
        result: &SharedTokenSet,
    ) {
        self.maybe_cleanup();
        self.token_set_results.insert(
            key,
            TokenSetOpMemoEntry {
                result: Arc::downgrade(result),
                left_operand: Arc::downgrade(left),
                right_operand: Arc::downgrade(right),
            },
        );
        self.inserts_since_cleanup += 1;
    }
}

thread_local! {
    static WEIGHT_OP_MEMO: RefCell<WeightOpMemo> = RefCell::new(WeightOpMemo::default());
}

thread_local! {
    static PUBLIC_WEIGHT_INTERSECTION_MEMO: RefCell<PublicWeightIntersectionMemo> =
        RefCell::new(PublicWeightIntersectionMemo::default());
}

fn with_weight_op_memo<R>(f: impl FnOnce(&mut WeightOpMemo) -> R) -> R {
    WEIGHT_OP_MEMO.with(|memo| {
        let mut memo = memo.borrow_mut();
        let current_gen = WEIGHT_OP_MEMO_GENERATION.load(Ordering::Acquire);
        if memo.generation != current_gen {
            memo.clear_all();
            memo.generation = current_gen;
        }
        f(&mut memo)
    })
}

fn with_public_weight_intersection_memo<R>(
    f: impl FnOnce(&mut PublicWeightIntersectionMemo) -> R,
) -> R {
    PUBLIC_WEIGHT_INTERSECTION_MEMO.with(|memo| {
        let mut memo = memo.borrow_mut();
        let current_gen = WEIGHT_OP_MEMO_GENERATION.load(Ordering::Acquire);
        if memo.generation != current_gen {
            memo.clear_all();
            memo.generation = current_gen;
        }
        f(&mut memo)
    })
}

/// Cached structural hashes for interned `Weight` maps.
///
/// DWA minimization and merge code hash the same interned weights many times.
/// Computing the structural hash repeatedly walks every outer range and every
/// inner token range, which is especially costly for p0/global terminal-DWA
/// workloads. The cache key is the interned `Arc` pointer, guarded by a weak
/// reference to avoid ABA reuse; the cached value is still the full structural
/// hash, so equal weights keep identical `Hash` output even if a future caller
/// constructs an equal map under a different `Arc`.
struct WeightHashMemoEntry {
    result: u64,
    weight: Weak<WeightMap>,
}

#[derive(Default)]
struct WeightHashMemo {
    results: FxHashMap<usize, WeightHashMemoEntry>,
    inserts_since_cleanup: usize,
    generation: u64,
}

impl WeightHashMemo {
    fn maybe_cleanup(&mut self) {
        if self.inserts_since_cleanup < INTERNER_CLEANUP_INTERVAL {
            return;
        }
        self.results.retain(|_, entry| entry.weight.strong_count() > 0);
        self.inserts_since_cleanup = 0;
    }

    fn clear_all(&mut self) {
        self.results.clear();
        self.inserts_since_cleanup = 0;
    }

    fn get_or_insert(&mut self, weight: &Weight) -> u64 {
        let current_gen = WEIGHT_HASH_MEMO_GENERATION.load(Ordering::Acquire);
        if self.generation != current_gen {
            self.clear_all();
            self.generation = current_gen;
        }
        self.maybe_cleanup();
        let key = weight.ptr_key();
        if let Some(entry) = self.results.get(&key) {
            if let Some(existing) = entry.weight.upgrade() {
                if Arc::ptr_eq(&existing, &weight.0) {
                    return entry.result;
                }
            }
        }

        let result = structural_weight_hash_uncached(weight);
        self.results.insert(
            key,
            WeightHashMemoEntry {
                result,
                weight: Arc::downgrade(&weight.0),
            },
        );
        self.inserts_since_cleanup += 1;
        result
    }
}

thread_local! {
    static WEIGHT_HASH_MEMO: RefCell<WeightHashMemo> = RefCell::new(WeightHashMemo::default());
}

fn structural_weight_hash_uncached(weight: &Weight) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = rustc_hash::FxHasher::default();
    let is_full = weight.is_full();
    is_full.hash(&mut hasher);
    if !is_full {
        for (range, tokens) in weight.0.range_values() {
            range.hash(&mut hasher);
            tokens.as_ref().hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn cached_structural_weight_hash(weight: &Weight) -> u64 {
    WEIGHT_HASH_MEMO.with(|memo| memo.borrow_mut().get_or_insert(weight))
}

/// Prune only dead entries from the global weight interner.
pub fn clear_stale_weights() {
    interner_clear_stale();
}

/// Clear all live entries from the global weight/token-set interners and
/// invalidate memoized weight-operation caches that may retain those weights.
///
/// This is mainly useful for benchmarks that need to prevent interner reuse
/// from contaminating repeated compile measurements.
pub fn clear_weight_interners() {
    interner_clear_all();
    clear_weight_op_caches();
}

/// Clear weight-operation and structural-hash memo caches on **all** threads.
///
/// Increments the global generation counter so that every thread's
/// thread-local memo is lazily cleared on its next access.
pub fn clear_weight_op_caches() {
    WEIGHT_OP_MEMO_GENERATION.fetch_add(1, Ordering::Release);
    WEIGHT_HASH_MEMO_GENERATION.fetch_add(1, Ordering::Release);
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WeightCacheStats {
    pub token_set_entries: usize,
    pub live_token_set_entries: usize,
    pub weight_buckets: usize,
    pub weight_entries: usize,
    pub live_weight_entries: usize,
    pub current_thread_weight_ops: usize,
    pub current_thread_token_set_ops: usize,
    pub current_thread_public_intersections: usize,
    pub current_thread_weight_hashes: usize,
    pub weight_op_generation: u64,
    pub weight_hash_generation: u64,
}

pub fn weight_cache_stats() -> WeightCacheStats {
    let (token_set_entries, live_token_set_entries) = GLOBAL_TOKEN_SETS.iter().fold(
        (0usize, 0usize),
        |(total, live), bucket| {
            let entries = bucket.value();
            (
                total + entries.len(),
                live + entries.iter().filter(|weak| weak.strong_count() > 0).count(),
            )
        },
    );
    let weight_buckets = GLOBAL_WEIGHTS.len();
    let (weight_entries, live_weight_entries) = GLOBAL_WEIGHTS.iter().fold(
        (0usize, 0usize),
        |(total, live), bucket| {
            let entries = bucket.value();
            (
                total + entries.len(),
                live + entries.iter().filter(|weak| weak.strong_count() > 0).count(),
            )
        },
    );
    let (current_thread_weight_ops, current_thread_token_set_ops) = WEIGHT_OP_MEMO.with(|memo| {
        let memo = memo.borrow();
        (memo.results.len(), memo.token_set_results.len())
    });
    let current_thread_public_intersections = PUBLIC_WEIGHT_INTERSECTION_MEMO
        .with(|memo| memo.borrow().results.len());
    let current_thread_weight_hashes = WEIGHT_HASH_MEMO.with(|memo| memo.borrow().results.len());
    WeightCacheStats {
        token_set_entries,
        live_token_set_entries,
        weight_buckets,
        weight_entries,
        live_weight_entries,
        current_thread_weight_ops,
        current_thread_token_set_ops,
        current_thread_public_intersections,
        current_thread_weight_hashes,
        weight_op_generation: WEIGHT_OP_MEMO_GENERATION.load(Ordering::Acquire),
        weight_hash_generation: WEIGHT_HASH_MEMO_GENERATION.load(Ordering::Acquire),
    }
}

fn lookup_memoized_weight_op(kind: WeightOpKind, left: &Weight, right: &Weight) -> Option<Weight> {
    with_weight_op_memo(|memo| memo.lookup(WeightOpKey::for_weights(kind, left, right)))
}

fn store_memoized_weight_op(kind: WeightOpKind, left: &Weight, right: &Weight, result: &Weight) {
    with_weight_op_memo(|memo| {
        memo.store(WeightOpKey::for_weights(kind, left, right), left, right, result)
    });
}

fn with_memoized_weight_op(
    kind: WeightOpKind,
    left: &Weight,
    right: &Weight,
    build: impl FnOnce() -> Weight,
) -> Weight {
    if let Some(existing) = lookup_memoized_weight_op(kind, left, right) {
        return existing;
    }

    let result = build();
    store_memoized_weight_op(kind, left, right, &result);
    result
}

pub fn finalize_weight_map(map: WeightMap) -> Weight {
    if map.ranges().next().is_none() {
        EMPTY_WEIGHT.clone()
    } else {
        Weight(intern_weight_map(map))
    }
}

/// Construct a Weight from an already-canonical artifact-local map without
/// consulting the process-global interner. Packed constraint artifacts already
/// provide their own exact deduplication pool, so re-hashing every decoded map
/// is redundant. Structural Weight equality/hash remain exact across pools.
#[doc(hidden)]
pub fn finalize_weight_map_artifact_local(map: WeightMap) -> Weight {
    if map.ranges().next().is_none() {
        EMPTY_WEIGHT.clone()
    } else {
        Weight(Arc::new(map))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeightSerdeEntry {
    tsid: [u32; 2],
    tokens: Vec<[u32; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeightSerde {
    all: bool,
    entries: Vec<WeightSerdeEntry>,
}

fn sentinel_token_set() -> RangeSetBlaze<u32> {
    std::iter::once(WEIGHT_ALL_SENTINEL..=WEIGHT_ALL_SENTINEL).collect()
}

fn is_sentinel_token_set(tokens: &RangeSetBlaze<u32>) -> bool {
    let mut ranges = tokens.ranges();
    let Some(range) = ranges.next() else {
        return false;
    };
    ranges.next().is_none()
        && *range.start() == WEIGHT_ALL_SENTINEL
        && *range.end() == WEIGHT_ALL_SENTINEL
}

pub fn shared_rangeset(tokens: RangeSetBlaze<u32>) -> SharedTokenSet {
    intern_rangeset(tokens)
}

/// Artifact-local counterpart to [`shared_rangeset`]. The packed artifact pool
/// is already canonical, so its Arc is sufficient identity inside the loaded
/// constraint and avoids a redundant global structural hash/table lookup.
#[doc(hidden)]
pub fn shared_rangeset_artifact_local(tokens: RangeSetBlaze<u32>) -> SharedTokenSet {
    Arc::new(tokens)
}

fn rangeset_from_ranges<I>(ranges: I) -> RangeSetBlaze<u32>
where
    I: IntoIterator<Item = std::ops::RangeInclusive<u32>>,
{
    ranges.into_iter().collect()
}

fn rangeset_to_vec(set: &RangeSetBlaze<u32>) -> Vec<[u32; 2]> {
    set.ranges()
        .map(|range| [*range.start(), *range.end()])
        .collect()
}

fn rangeset_to_string(set: &RangeSetBlaze<u32>) -> String {
    let parts: Vec<String> = set
        .ranges()
        .map(|range| {
            if range.start() == range.end() {
                format!("{}", range.start())
            } else {
                format!("{}..={}", range.start(), range.end())
            }
        })
        .collect();
    format!("{{{}}}", parts.join(","))
}

fn compress_expanded(expanded: &BTreeMap<u32, RangeSetBlaze<u32>>) -> Weight {
    let mut builder = CompactRangeBuilder::new();
    let mut current_start: Option<u32> = None;
    let mut current_end = 0u32;
    let mut current_tokens = RangeSetBlaze::new();

    for (&tsid, tokens) in expanded {
        match current_start {
            Some(_)
                if current_end.checked_add(1) == Some(tsid) && *tokens == current_tokens =>
            {
                current_end = tsid;
            }
            _ => {
                if let Some(start) = current_start.take() {
                    builder.push(
                        start,
                        current_end,
                        shared_rangeset(std::mem::take(&mut current_tokens)),
                    );
                }
                current_start = Some(tsid);
                current_end = tsid;
                current_tokens = tokens.clone();
            }
        }
    }

    if let Some(start) = current_start {
        builder.push(start, current_end, shared_rangeset(current_tokens));
    }

    builder.finish()
}

#[derive(Clone)]
struct WeightRangeEntry {
    start: u32,
    end: u32,
    tokens: SharedTokenSet,
}

#[derive(Clone)]
pub struct WeightIntersectionIndex {
    source: Weight,
    entries: Vec<WeightRangeEntry>,
}

impl WeightIntersectionIndex {
    fn new(source: &Weight) -> Self {
        Self {
            source: source.clone(),
            entries: source
                .0
                .range_values()
                .map(|(range, tokens)| WeightRangeEntry {
                    start: *range.start(),
                    end: *range.end(),
                    tokens: Arc::clone(tokens),
                })
                .collect(),
        }
    }
}

fn intersect_weight_with_index(sparse: &Weight, index: &WeightIntersectionIndex) -> Weight {
    let mut builder = CompactRangeBuilder::new();
    let mut overlap_cache: SmallVec<[(
        *const RangeSetBlaze<u32>,
        *const RangeSetBlaze<u32>,
        Option<SharedTokenSet>,
    ); 8]> = SmallVec::new();

    for (sparse_range, sparse_tokens) in sparse.0.range_values() {
        let sparse_start = *sparse_range.start();
        let sparse_end = *sparse_range.end();
        let mut entry_index = index
            .entries
            .partition_point(|entry| entry.end < sparse_start);

        while let Some(entry) = index.entries.get(entry_index) {
            if entry.start > sparse_end {
                break;
            }

            let start = sparse_start.max(entry.start);
            let end = sparse_end.min(entry.end);
            let sparse_ptr = Arc::as_ptr(sparse_tokens);
            let dense_ptr = Arc::as_ptr(&entry.tokens);
            let tokens = if let Some((_, _, cached)) = overlap_cache.iter().find(
                |(cached_sparse, cached_dense, _)| {
                    *cached_sparse == sparse_ptr && *cached_dense == dense_ptr
                },
            ) {
                cached.clone()
            } else {
                let overlap = shared_token_intersection(sparse_tokens, &entry.tokens);
                overlap_cache.push((sparse_ptr, dense_ptr, overlap.clone()));
                overlap
            };

            if let Some(tokens) = tokens {
                builder.push(start, end, tokens);
            }
            entry_index += 1;
        }
    }

    builder.finish()
}

struct CompactRangeBuilder {
    map: WeightMap,
    pending: Option<WeightRangeEntry>,
}

impl CompactRangeBuilder {
    fn new() -> Self {
        Self {
            map: WeightMap::new(),
            pending: None,
        }
    }

    fn push(&mut self, start: u32, end: u32, tokens: SharedTokenSet) {
        match self.pending.as_mut() {
            Some(pending)
                if pending.end.checked_add(1) == Some(start)
                    && same_shared_token_set(&pending.tokens, &tokens) =>
            {
                pending.end = end;
            }
            _ => {
                self.flush();
                self.pending = Some(WeightRangeEntry { start, end, tokens });
            }
        }
    }

    fn flush(&mut self) {
        // Move the owned pending value directly. An absent range needs no
        // shared empty-set placeholder (and no refcount updates per flush).
        if let Some(pending) = self.pending.take() {
            self.map
                .extend_simple(std::iter::once((pending.start..=pending.end, pending.tokens)));
        }
    }

    fn finish(mut self) -> Weight {
        self.flush();
        finalize_weight_map(self.map)
    }
}

fn intersect_overlay_with_domain(
    overlay: impl IntoIterator<Item = WeightRangeEntry>,
    domain: &Weight,
) -> Weight {
    if domain.is_full() {
        let mut builder = CompactRangeBuilder::new();
        for entry in overlay {
            builder.push(entry.start, entry.end, entry.tokens);
        }
        return builder.finish();
    }

    let mut builder = CompactRangeBuilder::new();
    let mut overlap_cache: SmallVec<[(
        *const RangeSetBlaze<u32>,
        *const RangeSetBlaze<u32>,
        Option<SharedTokenSet>,
    ); 8]> = SmallVec::new();
    let mut domain_iter = domain.0.range_values();
    let mut current_domain = domain_iter.next();
    for entry in overlay {
        while current_domain
            .as_ref()
            .is_some_and(|(range, _)| *range.end() < entry.start)
        {
            current_domain = domain_iter.next();
        }
        while let Some((range, domain_tokens)) = current_domain.as_ref() {
            if *range.start() > entry.end {
                break;
            }
            let start = entry.start.max(*range.start());
            let end = entry.end.min(*range.end());
            let source_ptr = Arc::as_ptr(&entry.tokens);
            let domain_ptr = Arc::as_ptr(domain_tokens);
            let tokens = if let Some((_, _, cached)) =
                overlap_cache
                    .iter()
                    .find(|(cached_source, cached_domain, _)| {
                        *cached_source == source_ptr && *cached_domain == domain_ptr
                    })
            {
                cached.clone()
            } else {
                let overlap = shared_token_intersection(&entry.tokens, domain_tokens);
                overlap_cache.push((source_ptr, domain_ptr, overlap.clone()));
                overlap
            };
            if let Some(tokens) = tokens {
                builder.push(start, end, tokens);
            }
            if *range.end() <= entry.end {
                current_domain = domain_iter.next();
            } else {
                break;
            }
        }
    }
    builder.finish()
}

fn compact_entries(weight: &Weight) -> SmallVec<[WeightRangeEntry; 16]> {
    weight
        .0
        .range_values()
        .map(|(range, tokens)| WeightRangeEntry {
            start: *range.start(),
            end: *range.end(),
            tokens: Arc::clone(tokens),
        })
        .collect()
}

fn single_compact_entry(weight: &Weight) -> Option<WeightRangeEntry> {
    let mut entries = weight.0.range_values();
    let (range, tokens) = entries.next()?;
    if entries.next().is_some() {
        return None;
    }
    Some(WeightRangeEntry {
        start: *range.start(),
        end: *range.end(),
        tokens: Arc::clone(tokens),
    })
}

fn insert_boundary(boundaries: &mut [u64; 4], len: &mut usize, value: u64) {
    let mut pos = 0usize;
    while pos < *len && boundaries[pos] < value {
        pos += 1;
    }
    if pos < *len && boundaries[pos] == value {
        return;
    }
    let mut i = *len;
    while i > pos {
        boundaries[i] = boundaries[i - 1];
        i -= 1;
    }
    boundaries[pos] = value;
    *len += 1;
}

fn combine_single_entries<F>(
    left: &WeightRangeEntry,
    right: &WeightRangeEntry,
    mut combine: F,
) -> Weight
where
    F: FnMut(
        Option<&SharedTokenSet>,
        Option<&SharedTokenSet>,
    ) -> Option<SharedTokenSet>,
{
    let mut boundaries = [0u64; 4];
    let mut len = 0usize;
    insert_boundary(&mut boundaries, &mut len, u64::from(left.start));
    insert_boundary(&mut boundaries, &mut len, u64::from(left.end) + 1);
    insert_boundary(&mut boundaries, &mut len, u64::from(right.start));
    insert_boundary(&mut boundaries, &mut len, u64::from(right.end) + 1);

    if len < 2 {
        return Weight::empty();
    }

    let mut builder = CompactRangeBuilder::new();

    for i in 0..(len - 1) {
        let start = boundaries[i] as u32;
        let end = (boundaries[i + 1] - 1) as u32;
        let left_tokens = (left.start <= start && start <= left.end).then_some(&left.tokens);
        let right_tokens = (right.start <= start && start <= right.end).then_some(&right.tokens);
        let Some(tokens) = combine(left_tokens, right_tokens) else {
            builder.flush();
            continue;
        };
        builder.push(start, end, tokens);
    }

    builder.finish()
}

fn weight_tsid_span(weight: &Weight) -> Option<(u32, u32)> {
    let mut ranges = weight.0.ranges();
    let first = ranges.next()?;
    let mut last_end = *first.end();
    for range in ranges {
        last_end = *range.end();
    }
    Some((*first.start(), last_end))
}

fn append_weight_entries(builder: &mut CompactRangeBuilder, weight: &Weight) {
    for (range, tokens) in weight.0.range_values() {
        builder.push(*range.start(), *range.end(), Arc::clone(tokens));
    }
}

fn coalesce_repeated_token_body_ranges(
    all_entries: Vec<WeightRangeEntry>,
) -> Vec<WeightRangeEntry> {
    const MIN_ENTRIES: usize = 2_048;
    const MIN_OUTPUT_COMPRESSION: usize = 32;

    let input_entries = all_entries.len();
    if input_entries < MIN_ENTRIES {
        return all_entries;
    }

    let mut token_body_ids = FxHashSet::default();
    for entry in &all_entries {
        token_body_ids.insert(Arc::as_ptr(&entry.tokens) as usize);
    }
    if token_body_ids
        .len()
        .saturating_mul(MIN_OUTPUT_COMPRESSION)
        > input_entries
    {
        return all_entries;
    }

    struct TokenBodyGroup {
        tokens: SharedTokenSet,
        ranges: Vec<(u32, u32)>,
    }

    let mut group_by_token_body = FxHashMap::<usize, usize>::default();
    let mut groups = Vec::<TokenBodyGroup>::with_capacity(token_body_ids.len());
    for range_entry in &all_entries {
        let token_body = Arc::as_ptr(&range_entry.tokens) as usize;
        let group_id = match group_by_token_body.entry(token_body) {
            std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let group_id = groups.len();
                entry.insert(group_id);
                groups.push(TokenBodyGroup {
                    tokens: Arc::clone(&range_entry.tokens),
                    ranges: Vec::new(),
                });
                group_id
            }
        };
        groups[group_id].ranges.push((range_entry.start, range_entry.end));
    }

    let mut coalesced = Vec::new();
    for mut group in groups {
        group.ranges.sort_unstable_by_key(|&(start, end)| (start, end));
        let mut merged = Vec::<(u32, u32)>::with_capacity(group.ranges.len());
        for (start, end) in group.ranges {
            if let Some((_, previous_end)) = merged.last_mut()
                && start <= previous_end.saturating_add(1)
            {
                *previous_end = (*previous_end).max(end);
            } else {
                merged.push((start, end));
            }
        }
        coalesced.extend(merged.into_iter().map(|(start, end)| WeightRangeEntry {
            start,
            end,
            tokens: Arc::clone(&group.tokens),
        }));
    }
    let selected = coalesced
        .len()
        .saturating_mul(MIN_OUTPUT_COMPRESSION)
        <= input_entries;
    if selected { coalesced } else { all_entries }
}

fn union_disjoint_tsid_ranges(left: &Weight, right: &Weight) -> Option<Weight> {
    let (left_start, left_end) = weight_tsid_span(left)?;
    let (right_start, right_end) = weight_tsid_span(right)?;

    let mut builder = CompactRangeBuilder::new();
    if left_end < right_start {
        append_weight_entries(&mut builder, left);
        append_weight_entries(&mut builder, right);
        Some(builder.finish())
    } else if right_end < left_start {
        append_weight_entries(&mut builder, right);
        append_weight_entries(&mut builder, left);
        Some(builder.finish())
    } else {
        None
    }
}

/// Direct multi-way union that avoids creating O(N) intermediate Weight objects.
///
/// For large inputs, this uses an event sweep over start and end boundaries.
/// The previous implementation re-scanned every started entry at every
/// boundary, which is quadratic for long overlapping ranges.  The event sweep
/// maintains the active distinct token sets incrementally instead.
fn union_all_multiway_impl(weights: &[&Weight], coalesce_repeated_token_ranges: bool) -> Weight {
    let mut token_union_cache = FxHashMap::default();
    union_all_multiway_impl_with_token_cache(
        weights,
        coalesce_repeated_token_ranges,
        &mut token_union_cache,
    )
}

fn union_all_multiway_with_token_cache(
    weights: &[&Weight],
    token_union_cache: &mut FxHashMap<Vec<usize>, SharedTokenSet>,
) -> Weight {
    union_all_multiway_impl_with_token_cache(weights, false, token_union_cache)
}

fn union_all_multiway_impl_with_token_cache(
    weights: &[&Weight],
    coalesce_repeated_token_ranges: bool,
    token_union_cache: &mut FxHashMap<Vec<usize>, SharedTokenSet>,
) -> Weight {
    let total_entry_hint: usize = weights.iter().map(|w| w.0.range_values_len()).sum();
    let mut all_entries: Vec<WeightRangeEntry> = Vec::with_capacity(total_entry_hint);
    for weight in weights {
        for (range, tokens) in weight.0.range_values() {
            all_entries.push(WeightRangeEntry {
                start: *range.start(),
                end: *range.end(),
                tokens: Arc::clone(tokens),
            });
        }
    }

    if all_entries.is_empty() {
        return Weight::empty();
    }

    if coalesce_repeated_token_ranges {
        all_entries = coalesce_repeated_token_body_ranges(all_entries);
    }

    // Terminal-transport unions frequently contain coordinate-disjoint
    // fragments. Emit them directly instead of building an active-token sweep
    // whose active set can never contain more than one operand.
    all_entries.sort_unstable_by_key(|entry| entry.start);
    if all_entries
        .windows(2)
        .all(|pair| pair[0].end < pair[1].start)
    {
        let mut builder = CompactRangeBuilder::new();
        for entry in all_entries {
            builder.push(entry.start, entry.end, entry.tokens);
        }
        return builder.finish();
    }

    // The compact rescan path avoids tree-map overhead for the small common
    // case; the event sweep avoids pathological repeated scans for larger
    // overlapping unions.
    const INCREMENTAL_SWEEP_MIN_ENTRIES: usize = 64;
    if all_entries.len() < INCREMENTAL_SWEEP_MIN_ENTRIES {
        return union_all_multiway_rescan_with_cache(all_entries, token_union_cache);
    }
    union_all_multiway_incremental_with_cache(all_entries, token_union_cache)
}

fn union_all_multiway(weights: &[&Weight]) -> Weight {
    union_all_multiway_impl(weights, false)
}

fn union_all_multiway_rescan(all_entries: Vec<WeightRangeEntry>) -> Weight {
    let mut token_union_cache = FxHashMap::default();
    union_all_multiway_rescan_with_cache(all_entries, &mut token_union_cache)
}

fn union_all_multiway_rescan_with_cache(
    all_entries: Vec<WeightRangeEntry>,
    token_union_cache: &mut FxHashMap<Vec<usize>, SharedTokenSet>,
) -> Weight {
    let mut boundaries = Vec::with_capacity(all_entries.len() * 2);
    for entry in &all_entries {
        boundaries.push(u64::from(entry.start));
        boundaries.push(u64::from(entry.end) + 1);
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    if boundaries.len() < 2 {
        return Weight::empty();
    }

    let mut builder = CompactRangeBuilder::new();
    let mut scan_start = 0usize;
    let mut active_tokens = Vec::<SharedTokenSet>::new();

    for window in boundaries.windows(2) {
        let interval_start = window[0] as u32;
        let interval_end = (window[1] - 1) as u32;

        while scan_start < all_entries.len() && all_entries[scan_start].end < interval_start {
            scan_start += 1;
        }

        active_tokens.clear();
        for entry in &all_entries[scan_start..] {
            if entry.start > interval_start {
                break;
            }
            if entry.end >= interval_end {
                active_tokens.push(Arc::clone(&entry.tokens));
            }
        }

        active_tokens.sort_unstable_by_key(|tokens| Arc::as_ptr(tokens) as usize);
        active_tokens.dedup_by_key(|tokens| Arc::as_ptr(tokens) as usize);

        let tokens = union_active_token_sets(&active_tokens, token_union_cache);
        if let Some(tokens) = tokens {
            builder.push(interval_start, interval_end, tokens);
        } else {
            builder.flush();
        }
    }

    builder.finish()
}

fn union_all_multiway_incremental(all_entries: Vec<WeightRangeEntry>) -> Weight {
    let mut token_union_cache = FxHashMap::default();
    union_all_multiway_incremental_with_cache(all_entries, &mut token_union_cache)
}

fn union_all_multiway_incremental_with_cache(
    all_entries: Vec<WeightRangeEntry>,
    token_union_cache: &mut FxHashMap<Vec<usize>, SharedTokenSet>,
) -> Weight {
    // Entries are already sorted by start in `union_all_multiway_impl`.
    // End events are exclusive so entries ending at `boundary - 1` leave the
    // active set before entries beginning at `boundary` enter it.
    let mut end_events: Vec<(u64, SharedTokenSet)> = all_entries
        .iter()
        .map(|entry| (u64::from(entry.end) + 1, Arc::clone(&entry.tokens)))
        .collect();
    end_events.sort_unstable_by_key(|(end_exclusive, _)| *end_exclusive);

    let mut builder = CompactRangeBuilder::new();
    let mut active: BTreeMap<usize, (SharedTokenSet, usize)> = BTreeMap::new();
    let mut active_tokens = Vec::<SharedTokenSet>::new();
    let mut start_index = 0usize;
    let mut end_index = 0usize;

    while start_index < all_entries.len() || end_index < end_events.len() {
        let next_start = all_entries
            .get(start_index)
            .map(|entry| u64::from(entry.start));
        let next_end = end_events.get(end_index).map(|(end, _)| *end);
        let boundary = match (next_start, next_end) {
            (Some(start), Some(end)) => start.min(end),
            (Some(start), None) => start,
            (None, Some(end)) => end,
            (None, None) => break,
        };

        while end_index < end_events.len() && end_events[end_index].0 == boundary {
            let tokens = &end_events[end_index].1;
            let key = Arc::as_ptr(tokens) as usize;
            let mut remove = false;
            if let Some((_, count)) = active.get_mut(&key) {
                debug_assert!(*count > 0);
                *count -= 1;
                remove = *count == 0;
            } else {
                debug_assert!(false, "every end event must have an active start event");
            }
            if remove {
                active.remove(&key);
            }
            end_index += 1;
        }

        while start_index < all_entries.len()
            && u64::from(all_entries[start_index].start) == boundary
        {
            let tokens = Arc::clone(&all_entries[start_index].tokens);
            let key = Arc::as_ptr(&tokens) as usize;
            active
                .entry(key)
                .and_modify(|(_, count)| *count += 1)
                .or_insert((tokens, 1));
            start_index += 1;
        }

        let following_start = all_entries
            .get(start_index)
            .map(|entry| u64::from(entry.start));
        let following_end = end_events.get(end_index).map(|(end, _)| *end);
        let Some(next_boundary) = (match (following_start, following_end) {
            (Some(start), Some(end)) => Some(start.min(end)),
            (Some(start), None) => Some(start),
            (None, Some(end)) => Some(end),
            (None, None) => None,
        }) else {
            break;
        };

        if active.is_empty() {
            builder.flush();
            continue;
        }

        active_tokens.clear();
        active_tokens.extend(active.values().map(|(tokens, _)| Arc::clone(tokens)));
        let tokens = union_active_token_sets(&active_tokens, token_union_cache)
            .expect("non-empty active set has a token union");
        builder.push(boundary as u32, (next_boundary - 1) as u32, tokens);
    }

    builder.finish()
}

fn union_active_token_sets(
    active_tokens: &[SharedTokenSet],
    token_union_cache: &mut FxHashMap<Vec<usize>, SharedTokenSet>,
) -> Option<SharedTokenSet> {
    match active_tokens.len() {
        0 => None,
        1 => Some(Arc::clone(&active_tokens[0])),
        2 => Some(shared_token_union(&active_tokens[0], &active_tokens[1])),
        _ => {
            let key: Vec<usize> = active_tokens
                .iter()
                .map(|tokens| Arc::as_ptr(tokens) as usize)
                .collect();
            if let Some(cached) = token_union_cache.get(&key) {
                Some(Arc::clone(cached))
            } else {
                let tokens = shared_token_union_many(active_tokens);
                if let Some(tokens) = &tokens {
                    token_union_cache.insert(key, Arc::clone(tokens));
                }
                tokens
            }
        }
    }
}

fn union_all_single_tsid_entries(weights: &[&Weight]) -> Option<Weight> {
    let mut per_tsid: BTreeMap<u32, SharedTokenSet> = BTreeMap::new();

    for weight in weights {
        let entry = single_compact_entry(weight)?;
        if entry.start != entry.end || entry.start == WEIGHT_ALL_SENTINEL {
            return None;
        }

        per_tsid
            .entry(entry.start)
            .and_modify(|existing| *existing = shared_token_union(existing, &entry.tokens))
            .or_insert(entry.tokens);
    }

    let mut builder = CompactRangeBuilder::new();
    for (tsid, tokens) in per_tsid {
        builder.push(tsid, tsid, tokens);
    }
    Some(builder.finish())
}

// Two-way union via range-set-blaze's `outer_join`. The original hand-written merge is kept
// below as `union_compact_entries_handwritten` (test-only) for differential testing.
fn union_compact_entries(left: &Weight, right: &Weight) -> Weight {
    if left.0.is_empty() {
        return right.clone();
    }
    if right.0.is_empty() {
        return left.clone();
    }
    combine_compact_entries(left, right, union_token_sets)
}

#[cfg(test)]
fn union_compact_entries_handwritten(left: &Weight, right: &Weight) -> Weight {
    let left_entries = compact_entries(left);
    let right_entries = compact_entries(right);

    if left_entries.is_empty() {
        return right.clone();
    }
    if right_entries.is_empty() {
        return left.clone();
    }

    let mut builder = CompactRangeBuilder::new();
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    let mut left_current = Some(left_entries[left_index].clone());
    let mut right_current = Some(right_entries[right_index].clone());

    loop {
        match (&mut left_current, &mut right_current) {
            (Some(left_entry), Some(right_entry)) => {
                if left_entry.end < right_entry.start {
                    builder.push(left_entry.start, left_entry.end, Arc::clone(&left_entry.tokens));
                    left_index += 1;
                    left_current = left_entries.get(left_index).cloned();
                    continue;
                }
                if right_entry.end < left_entry.start {
                    builder.push(right_entry.start, right_entry.end, Arc::clone(&right_entry.tokens));
                    right_index += 1;
                    right_current = right_entries.get(right_index).cloned();
                    continue;
                }

                if left_entry.start < right_entry.start {
                    builder.push(
                        left_entry.start,
                        right_entry.start - 1,
                        Arc::clone(&left_entry.tokens),
                    );
                    left_entry.start = right_entry.start;
                } else if right_entry.start < left_entry.start {
                    builder.push(
                        right_entry.start,
                        left_entry.start - 1,
                        Arc::clone(&right_entry.tokens),
                    );
                    right_entry.start = left_entry.start;
                }

                let overlap_end = left_entry.end.min(right_entry.end);
                builder.push(
                    left_entry.start,
                    overlap_end,
                    shared_token_union(&left_entry.tokens, &right_entry.tokens),
                );

                match (left_entry.end == overlap_end, right_entry.end == overlap_end) {
                    (true, true) => {
                        left_index += 1;
                        right_index += 1;
                        left_current = left_entries.get(left_index).cloned();
                        right_current = right_entries.get(right_index).cloned();
                    }
                    (true, false) => {
                        let next_start = overlap_end + 1;
                        right_entry.start = next_start;
                        left_index += 1;
                        left_current = left_entries.get(left_index).cloned();
                    }
                    (false, true) => {
                        let next_start = overlap_end + 1;
                        left_entry.start = next_start;
                        right_index += 1;
                        right_current = right_entries.get(right_index).cloned();
                    }
                    (false, false) => unreachable!(),
                }
            }
            (Some(left_entry), None) => {
                builder.push(left_entry.start, left_entry.end, Arc::clone(&left_entry.tokens));
                left_index += 1;
                left_current = left_entries.get(left_index).cloned();
            }
            (None, Some(right_entry)) => {
                builder.push(right_entry.start, right_entry.end, Arc::clone(&right_entry.tokens));
                right_index += 1;
                right_current = right_entries.get(right_index).cloned();
            }
            (None, None) => break,
        }
    }

    builder.finish()
}

fn combined_boundaries(left: &[WeightRangeEntry], right: &[WeightRangeEntry]) -> SmallVec<[u64; 32]> {
    let mut boundaries = SmallVec::<[u64; 32]>::with_capacity((left.len() + right.len()) * 2);
    for entry in left.iter().chain(right.iter()) {
        boundaries.push(u64::from(entry.start));
        boundaries.push(u64::from(entry.end) + 1);
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
}

fn active_tokens<'a>(
    entries: &'a [WeightRangeEntry],
    index: &mut usize,
    start: u32,
) -> Option<&'a SharedTokenSet> {
    while *index < entries.len() && entries[*index].end < start {
        *index += 1;
    }
    entries.get(*index).and_then(|entry| {
        (entry.start <= start && start <= entry.end).then_some(&entry.tokens)
    })
}

// Two-way combine via range-set-blaze's `outer_join`: one merge pass over both weights, calling
// `combine` with each side's token set (or `None`) on every range covered by either side. The
// original boundary-sweep version is kept below as `combine_compact_entries_boundary` (test-only).
fn combine_compact_entries<F>(left: &Weight, right: &Weight, mut combine: F) -> Weight
where
    F: FnMut(
        Option<&SharedTokenSet>,
        Option<&SharedTokenSet>,
    ) -> Option<SharedTokenSet>,
{
    let mut builder = CompactRangeBuilder::new();
    for (range, (left_tokens, right_tokens)) in
        left.0.range_values().outer_join(right.0.range_values())
    {
        let Some(tokens) = combine(left_tokens, right_tokens) else {
            builder.flush();
            continue;
        };
        builder.push(*range.start(), *range.end(), tokens);
    }
    builder.finish()
}

#[cfg(test)]
fn combine_compact_entries_boundary<F>(left: &Weight, right: &Weight, mut combine: F) -> Weight
where
    F: FnMut(
        Option<&SharedTokenSet>,
        Option<&SharedTokenSet>,
    ) -> Option<SharedTokenSet>,
{
    let left_entries = compact_entries(left);
    let right_entries = compact_entries(right);
    let boundaries = combined_boundaries(&left_entries, &right_entries);
    if boundaries.len() < 2 {
        return Weight::empty();
    }

    let mut left_index = 0usize;
    let mut right_index = 0usize;
    let mut builder = CompactRangeBuilder::new();

    for window in boundaries.windows(2) {
        let start = window[0] as u32;
        let end = (window[1] - 1) as u32;
        let left_tokens = active_tokens(&left_entries, &mut left_index, start);
        let right_tokens = active_tokens(&right_entries, &mut right_index, start);
        let Some(tokens) = combine(left_tokens, right_tokens) else {
            builder.flush();
            continue;
        };
        builder.push(start, end, tokens);
    }

    builder.finish()
}


fn difference_weights(left: &Weight, right: &Weight) -> Weight {
    let mut right_iter = right.0.range_values();
    let mut right_entry = right_iter.next();
    let mut builder = CompactRangeBuilder::new();
    let mut same_as_left = true;

    for (left_range, left_tokens) in left.0.range_values() {
        let left_start = *left_range.start();
        let left_end = *left_range.end();
        let mut cursor = u64::from(left_start);
        let left_end_u64 = u64::from(left_end);

        while let Some((right_range, _)) = right_entry.as_ref() {
            if *right_range.end() < left_start {
                right_entry = right_iter.next();
            } else {
                break;
            }
        }

        while cursor <= left_end_u64 {
            let Some((right_range, right_tokens)) = right_entry.as_ref() else {
                builder.push(cursor as u32, left_end, Arc::clone(left_tokens));
                break;
            };
            let right_start = u64::from(*right_range.start());
            let right_end = u64::from(*right_range.end());

            if right_start > left_end_u64 {
                builder.push(cursor as u32, left_end, Arc::clone(left_tokens));
                break;
            }
            if right_end < cursor {
                right_entry = right_iter.next();
                continue;
            }

            if cursor < right_start {
                let gap_end = (right_start - 1).min(left_end_u64);
                builder.push(cursor as u32, gap_end as u32, Arc::clone(left_tokens));
                cursor = gap_end + 1;
                if cursor > left_end_u64 {
                    break;
                }
            }

            let overlap_end = left_end_u64.min(right_end);
            if let Some(tokens) = shared_token_difference(left_tokens, right_tokens) {
                if !same_shared_token_set(&tokens, left_tokens) {
                    same_as_left = false;
                }
                builder.push(cursor as u32, overlap_end as u32, tokens);
            } else {
                same_as_left = false;
                builder.flush();
            }
            cursor = overlap_end + 1;

            if right_end <= overlap_end {
                right_entry = right_iter.next();
            }
        }
    }

    if same_as_left {
        left.clone()
    } else {
        builder.finish()
    }
}

fn intersect_weights(left: &Weight, right: &Weight) -> Weight {
    let mut left_iter = left.0.range_values();
    let mut right_iter = right.0.range_values();
    let mut left_entry = left_iter.next();
    let mut right_entry = right_iter.next();

    let mut builder = CompactRangeBuilder::new();
    let mut same_as_left = true;
    let mut same_as_right = true;

    loop {
        let (left_range, left_tokens, right_range, right_tokens) = match (left_entry, right_entry)
        {
            (Some((left_range, left_tokens)), Some((right_range, right_tokens))) => {
                (left_range, left_tokens, right_range, right_tokens)
            }
            (Some(_), None) => {
                same_as_left = false;
                break;
            }
            (None, Some(_)) => {
                same_as_right = false;
                break;
            }
            (None, None) => break,
        };
        let start = (*left_range.start()).max(*right_range.start());
        let end = (*left_range.end()).min(*right_range.end());
        let left_start = *left_range.start();
        let left_end = *left_range.end();
        let right_start = *right_range.start();
        let right_end = *right_range.end();

        if start <= end {
            if start != left_start || end != left_end {
                same_as_left = false;
            }
            if start != right_start || end != right_end {
                same_as_right = false;
            }
            if let Some(tokens) = shared_token_intersection(left_tokens, right_tokens) {
                if !Arc::ptr_eq(&tokens, left_tokens) {
                    same_as_left = false;
                }
                if !Arc::ptr_eq(&tokens, right_tokens) {
                    same_as_right = false;
                }
                builder.push(start, end, tokens);
            } else {
                same_as_left = false;
                same_as_right = false;
            }
        } else if left_end < right_start {
            same_as_left = false;
        } else if right_end < left_start {
            same_as_right = false;
        }

        if left_end <= right_end {
            left_entry = left_iter.next();
        } else {
            left_entry = Some((left_range, left_tokens));
        }
        if right_end <= left_end {
            right_entry = right_iter.next();
        } else {
            right_entry = Some((right_range, right_tokens));
        }
    }

    if same_as_left {
        left.clone()
    } else if same_as_right {
        right.clone()
    } else {
        builder.finish()
    }
}

fn intersect_single_entry_with_weight(single: &WeightRangeEntry, other: &Weight) -> Weight {
    // A very common parser-closure shape is one TSID-wide edge weight intersected
    // with a much larger residual weight. `map_and_set_intersection` is a streaming
    // merge and therefore walks from the first range of `other`; for a point query
    // that needlessly scans every preceding TSID range. RangeMapBlaze::get() uses
    // its BTreeMap index and gives the exact same token-set intersection in O(log n).
    if single.start == single.end {
        let other_tokens = match other.0.range_values().next() {
            Some((range, tokens)) if *range.start() <= single.start && single.start <= *range.end() => {
                tokens
            }
            Some((range, _)) if single.start < *range.start() => return Weight::empty(),
            _ => {
                let Some(tokens) = other.0.get(single.start) else {
                    return Weight::empty();
                };
                tokens
            }
        };
        let tokens = if same_shared_token_set(&single.tokens, other_tokens) {
            Some(Arc::clone(&single.tokens))
        } else {
            shared_token_intersection(&single.tokens, other_tokens)
        };
        let Some(tokens) = tokens else {
            return Weight::empty();
        };
        let mut builder = CompactRangeBuilder::new();
        builder.push(single.start, single.end, tokens);
        return builder.finish();
    }

    let mut builder = CompactRangeBuilder::new();
    let mut overlap_cache: SmallVec<[(
        *const RangeSetBlaze<u32>,
        Option<SharedTokenSet>,
    ); 8]> = SmallVec::new();

    let bounds = CheckSortedDisjoint::new([single.start..=single.end]);
    for (range, other_tokens) in other.0.range_values().map_and_set_intersection(bounds) {
        let start = *range.start();
        let end = *range.end();

        let tokens = if same_shared_token_set(&single.tokens, other_tokens) {
            Some(Arc::clone(&single.tokens))
        } else {
            let cache_key = Arc::as_ptr(other_tokens);
            if let Some((_, cached)) = overlap_cache.iter().find(|(ptr, _)| *ptr == cache_key) {
                cached.clone()
            } else {
                let overlap = shared_token_intersection(&single.tokens, other_tokens);
                overlap_cache.push((cache_key, overlap.clone()));
                overlap
            }
        };

        let Some(tokens) = tokens else {
            builder.flush();
            continue;
        };

        builder.push(start, end, tokens);
    }

    builder.finish()
}

impl Weight {
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn intersection_index(&self) -> WeightIntersectionIndex {
        WeightIntersectionIndex::new(self)
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn intersection_with_index(&self, index: &WeightIntersectionIndex) -> Self {
        let other = &index.source;
        if self.is_empty() || other.is_empty() {
            return Self::empty();
        }
        if Arc::ptr_eq(&self.0, &other.0) {
            return self.clone();
        }
        if self.is_full() {
            return other.clone();
        }
        if other.is_full() {
            return self.clone();
        }

        if let Some(existing) = with_public_weight_intersection_memo(|memo| memo.lookup(self, other)) {
            return existing;
        }

        let result = intersect_weight_with_index(self, index);
        with_public_weight_intersection_memo(|memo| memo.store(self, other, &result));
        result
    }

    pub fn ptr_key(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn raw_range_values(
        &self,
    ) -> impl Iterator<Item = (std::ops::RangeInclusive<u32>, &SharedTokenSet)> {
        self.0.range_values()
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn raw_iter(&self) -> impl Iterator<Item = (u32, &SharedTokenSet)> {
        self.0.iter()
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn token_set_for_tsid_ref(&self, tsid: u32) -> Option<&SharedTokenSet> {
        self.0.get(tsid)
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn storage_ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn structural_hash_cached(&self) -> u64 {
        cached_structural_weight_hash(self)
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn compact_entries(&self) -> Option<Vec<(u32, u32, SharedTokenSet)>> {
        if self.is_full() {
            return None;
        }

        Some(
            self.0
                .range_values()
                .map(|(range, tokens)| (*range.start(), *range.end(), Arc::clone(tokens)))
                .collect(),
        )
    }

    /// Borrowed outer-map entries for consumers that do not retain token sets.
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn range_entries(
        &self,
    ) -> impl Iterator<Item = (u32, u32, &SharedTokenSet)> + '_ {
        self.0
            .range_values()
            .map(|(range, tokens)| (*range.start(), *range.end(), tokens))
    }

    /// Remap only the inner token sets while preserving the already-sorted,
    /// disjoint TSID ranges. This avoids rebuilding a large weight through one
    /// B-tree insertion per range when its outer coordinates are unchanged.
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn remap_token_sets_preserving_tsid_ranges(
        &self,
        mut remap: impl FnMut(&SharedTokenSet) -> SharedTokenSet,
    ) -> Self {
        if self.is_empty() || self.is_full() {
            return self.clone();
        }

        let mut ranges = Vec::<(std::ops::RangeInclusive<u32>, SharedTokenSet)>::new();
        for (start, end, tokens) in self.range_entries() {
            let mapped = remap(tokens);
            if mapped.is_empty() {
                continue;
            }
            if let Some((previous_range, previous_tokens)) = ranges.last_mut()
                && previous_range.end().checked_add(1) == Some(start)
                && same_shared_token_set(previous_tokens, &mapped)
            {
                let previous_start = *previous_range.start();
                *previous_range = previous_start..=end;
            } else {
                ranges.push((start..=end, mapped));
            }
        }
        if ranges.is_empty() {
            return Self::empty();
        }

        let map = WeightMap::from_sorted_disjoint_map(CheckSortedDisjointMap::new(
            ranges
                .iter()
                .map(|(range, tokens)| (range.clone(), tokens)),
        ));
        finalize_weight_map(map)
    }

    pub fn empty() -> Self {
        EMPTY_WEIGHT.clone()
    }

    pub fn all() -> Self {
        ALL_WEIGHT.clone()
    }

    /// Create a weight where all tsids in the range share the same token set.
    pub fn from_uniform(tsid_range: std::ops::RangeInclusive<u32>, tokens: RangeSetBlaze<u32>) -> Self {
        if tokens.is_empty() {
            return Self::empty();
        }
        let mut map = WeightMap::new();
        map.extend_simple(std::iter::once((tsid_range, shared_rangeset(tokens))));
        finalize_weight_map(map)
    }

    /// Build a weight from per-TSID token sets without creating intermediate Weight objects.
    /// Each entry is (tsid, token_set). Entries MUST be sorted by tsid (ascending).
    /// Adjacent TSIDs with identical (Arc-equal) token sets are merged into ranges.
    pub fn from_per_tsid_token_sets(entries: impl IntoIterator<Item = (u32, RangeSetBlaze<u32>)>) -> Self {
        let mut builder = CompactRangeBuilder::new();
        for (tsid, tokens) in entries {
            if tokens.is_empty() {
                continue;
            }
            builder.push(tsid, tsid, shared_rangeset(tokens));
        }
        builder.finish()
    }

    /// Like `from_per_tsid_token_sets` but accepts pre-shared (Arc) token sets.
    /// This allows TSIDs sharing the same representative state to reuse the same
    /// Arc, enabling CompactRangeBuilder to merge them into contiguous ranges.
    pub fn from_per_tsid_shared(entries: impl IntoIterator<Item = (u32, SharedTokenSet)>) -> Self {
        let mut builder = CompactRangeBuilder::new();
        for (tsid, tokens) in entries {
            if tokens.is_empty() {
                continue;
            }
            builder.push(tsid, tsid, tokens);
        }
        builder.finish()
    }

    /// Build a weight from ordered inclusive TSID ranges that already share
    /// one interned token set. Retained as the general range-oriented name;
    /// `from_tsid_runs_shared` is the equivalent run-oriented spelling.
    pub fn from_tsid_ranges_shared(
        entries: impl IntoIterator<Item = (u32, u32, SharedTokenSet)>,
    ) -> Self {
        Self::from_tsid_runs_shared(entries)
    }

    /// Like `from_per_tsid_shared` but accepts pre-grouped, sorted inclusive
    /// `(start_tsid, end_tsid)` runs that each share one Arc token set. This is
    /// byte-identical to pushing every TSID in `start..=end` individually (the
    /// builder merges same-Arc contiguous TSIDs anyway) but skips the per-TSID
    /// iteration when the caller already knows the contiguous coordinate runs.
    pub fn from_tsid_runs_shared(
        runs: impl IntoIterator<Item = (u32, u32, SharedTokenSet)>,
    ) -> Self {
        let mut builder = CompactRangeBuilder::new();
        for (start, end, tokens) in runs {
            if tokens.is_empty() {
                continue;
            }
            builder.push(start, end, tokens);
        }
        builder.finish()
    }

    /// Lift a set of final-TSID runs whose `coordinate` field is sorted
    /// ascending. Byte-identical to
    /// `from_tsid_runs_shared(runs.map(|(s, e, c)| (s, e, self.shared_tokens_for_tsid(c))))`
    /// but replaces the per-run binary-search `get` with a single linear
    /// merge against this weight's ascending core-TSID ranges. Requires that
    /// `self` is neither empty nor full and that `runs` is sorted by
    /// `coordinate` (guaranteed by the post-clustering final-TSID relabel).
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn lift_sorted_coordinate_runs(&self, runs: &[(u32, u32, u32)]) -> Self {
        debug_assert!(!self.is_full());
        debug_assert!(
            runs.windows(2).all(|pair| pair[0].2 <= pair[1].2),
            "lift_sorted_coordinate_runs requires coordinate-sorted runs"
        );
        let mut builder = CompactRangeBuilder::new();
        let mut ranges = self.0.range_values().peekable();
        for &(start, end, coordinate) in runs {
            // Ranges are sorted ascending; coordinates are non-decreasing, so a
            // range whose end is below the coordinate can never match a later
            // run either and can be dropped.
            while let Some((range, _)) = ranges.peek() {
                if *range.end() < coordinate {
                    ranges.next();
                } else {
                    break;
                }
            }
            if let Some((range, tokens)) = ranges.peek() {
                if *range.start() <= coordinate {
                    builder.push(start, end, (*tokens).clone());
                }
            }
        }
        builder.finish()
    }

    /// Return the interned token set at one TSID without cloning its range
    /// contents. This is intentionally crate-visible for sparse coordinate
    /// remaps in the terminal-DWA compiler.
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn shared_tokens_for_tsid(&self, tsid: u32) -> SharedTokenSet {
        if self.is_full() {
            return self
                .0
                .range_values()
                .next()
                .expect("full weight must retain its sentinel token set")
                .1
                .clone();
        }
        self.0
            .get(tsid)
            .cloned()
            .unwrap_or_else(|| Arc::clone(&EMPTY_RANGESET))
    }

    /// Batched, sorted equivalent of [`Self::shared_tokens_for_tsid`] for a slice of
    /// strictly-ascending tsids. A single linear merge over the weight's ranges
    /// replaces one binary-search point lookup per tsid, which is the dominant
    /// cost of the post-DWA group-signature step. `u32::MAX` is the "no
    /// coordinate" sentinel and always yields the empty token set, matching the
    /// point helper's caller contract.
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn shared_tokens_for_sorted_tsids(&self, tsids: &[u32]) -> Vec<SharedTokenSet> {
        debug_assert!(
            tsids.windows(2).all(|pair| pair[0] < pair[1]),
            "shared_tokens_for_sorted_tsids requires strictly-ascending tsids"
        );
        if self.is_full() {
            let sentinel = self
                .0
                .range_values()
                .next()
                .expect("full weight must retain its sentinel token set")
                .1
                .clone();
            return tsids.iter().map(|_| sentinel.clone()).collect();
        }
        let mut result = Vec::with_capacity(tsids.len());
        let mut ranges = self.0.range_values().peekable();
        for &tsid in tsids {
            if tsid == u32::MAX {
                result.push(Arc::clone(&EMPTY_RANGESET));
                continue;
            }
            while let Some((range, _)) = ranges.peek() {
                if *range.end() < tsid {
                    ranges.next();
                } else {
                    break;
                }
            }
            let tokens = match ranges.peek() {
                Some((range, tokens)) if *range.start() <= tsid => (*tokens).clone(),
                _ => Arc::clone(&EMPTY_RANGESET),
            };
            result.push(tokens);
        }
        result
    }

    /// Apply a sorted, unique set of point-TSID token-set overrides without
    /// expanding the unchanged ranges of `self`. Empty override token sets
    /// remove that point from the result.
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn with_sparse_tsid_overrides(
        &self,
        overrides: &[(u32, SharedTokenSet)],
    ) -> Self {
        if overrides.is_empty() || self.is_full() {
            return self.clone();
        }
        debug_assert!(overrides.windows(2).all(|pair| pair[0].0 < pair[1].0));

        let mut builder = CompactRangeBuilder::new();
        let mut override_index = 0usize;

        let push_override = |builder: &mut CompactRangeBuilder,
                             tsid: u32,
                             tokens: &SharedTokenSet| {
            if !tokens.is_empty() {
                builder.push(tsid, tsid, Arc::clone(tokens));
            }
        };

        for (start, end, tokens) in self.range_entries() {
            while override_index < overrides.len() && overrides[override_index].0 < start {
                let (tsid, override_tokens) = &overrides[override_index];
                push_override(&mut builder, *tsid, override_tokens);
                override_index += 1;
            }

            let mut cursor = start;
            while override_index < overrides.len() && overrides[override_index].0 <= end {
                let (tsid, override_tokens) = &overrides[override_index];
                if cursor < *tsid {
                    builder.push(cursor, *tsid - 1, Arc::clone(tokens));
                }
                push_override(&mut builder, *tsid, override_tokens);
                cursor = tsid.saturating_add(1);
                override_index += 1;
            }
            if cursor <= end {
                builder.push(cursor, end, Arc::clone(tokens));
            }
        }

        while override_index < overrides.len() {
            let (tsid, tokens) = &overrides[override_index];
            push_override(&mut builder, *tsid, tokens);
            override_index += 1;
        }

        builder.finish()
    }

    /// Apply sparse TSID overrides and restrict the result to `domain` in a
    /// single range walk. This is exactly equivalent to
    /// `self.with_sparse_tsid_overrides(overrides).intersection(domain)`, but
    /// avoids interning and then immediately re-reading the intermediate
    /// overridden weight.
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn with_sparse_tsid_overrides_intersection(
        &self,
        overrides: &[(u32, SharedTokenSet)],
        domain: &Weight,
    ) -> Self {
        debug_assert!(overrides.windows(2).all(|pair| pair[0].0 < pair[1].0));
        let range_overrides = overrides
            .iter()
            .map(|(tsid, tokens)| (*tsid, *tsid, Arc::clone(tokens)))
            .collect::<SmallVec<[(u32, u32, SharedTokenSet); 16]>>();
        self.with_sparse_tsid_range_overrides_intersection(&range_overrides, domain)
    }

    /// Apply sorted, disjoint TSID-range overrides and restrict the result to
    /// `domain`. This is the ranged analogue of
    /// `with_sparse_tsid_overrides_intersection`: an override replaces the base
    /// token set at every TSID in its inclusive range.
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn with_sparse_tsid_range_overrides_intersection(
        &self,
        overrides: &[(u32, u32, SharedTokenSet)],
        domain: &Weight,
    ) -> Self {
        if self.is_empty() || domain.is_empty() {
            return Self::empty();
        }
        if self.is_full() {
            return self.intersection(domain);
        }
        if overrides.is_empty() {
            return self.intersection(domain);
        }
        debug_assert!(overrides.iter().all(|(start, end, _)| start <= end));
        debug_assert!(
            overrides
                .windows(2)
                .all(|pair| pair[0].1 < pair[1].0)
        );

        let mut overlay = SmallVec::<[WeightRangeEntry; 16]>::new();
        let push_overlay = |entries: &mut SmallVec<[WeightRangeEntry; 16]>,
                            start: u32,
                            end: u32,
                            tokens: &SharedTokenSet| {
            if tokens.is_empty() || start > end {
                return;
            }
            if let Some(previous) = entries.last_mut()
                && previous.end.checked_add(1) == Some(start)
                && same_shared_token_set(&previous.tokens, tokens)
            {
                previous.end = end;
            } else {
                entries.push(WeightRangeEntry {
                    start,
                    end,
                    tokens: Arc::clone(tokens),
                });
            }
        };

        let mut base_iter = self.range_entries();
        let mut current_base = base_iter.next();
        for (override_start, override_end, override_tokens) in overrides {
            loop {
                let Some((start, end, tokens)) = current_base.take() else {
                    break;
                };
                if end < *override_start {
                    push_overlay(&mut overlay, start, end, &tokens);
                    current_base = base_iter.next();
                    continue;
                }
                if start < *override_start {
                    push_overlay(&mut overlay, start, *override_start - 1, &tokens);
                }
                current_base = Some((start.max(*override_start), end, tokens));
                break;
            }

            push_overlay(
                &mut overlay,
                *override_start,
                *override_end,
                override_tokens,
            );

            loop {
                let Some((start, end, tokens)) = current_base.take() else {
                    break;
                };
                if start > *override_end {
                    current_base = Some((start, end, tokens));
                    break;
                }
                if end <= *override_end {
                    current_base = base_iter.next();
                    continue;
                }
                current_base = Some((*override_end + 1, end, tokens));
                break;
            }
        }
        while let Some((start, end, tokens)) = current_base {
            push_overlay(&mut overlay, start, end, &tokens);
            current_base = base_iter.next();
        }

        intersect_overlay_with_domain(overlay, domain)
    }

    /// Build the exact union of sorted point-TSID entries.
    ///
    /// The entries must be nondecreasing by TSID. Equal TSIDs are reduced with
    /// the canonical token-set union before the compact map is built, avoiding
    /// a sequence of whole-weight unions for point-entry workloads.
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn union_sorted_point_entries(
        entries: impl IntoIterator<Item = (u32, SharedTokenSet)>,
    ) -> Self {
        let mut builder = CompactRangeBuilder::new();
        let mut current: Option<(u32, SharedTokenSet)> = None;

        for (tsid, tokens) in entries {
            if tokens.is_empty() {
                continue;
            }
            match current.as_mut() {
                Some((current_tsid, current_tokens)) if *current_tsid == tsid => {
                    *current_tokens = shared_token_union(current_tokens, &tokens);
                }
                Some((current_tsid, current_tokens)) => {
                    builder.push(*current_tsid, *current_tsid, Arc::clone(current_tokens));
                    current = Some((tsid, tokens));
                }
                None => current = Some((tsid, tokens)),
            }
        }

        if let Some((tsid, tokens)) = current {
            builder.push(tsid, tsid, tokens);
        }
        builder.finish()
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn insert(
        &mut self,
        tsid_range: std::ops::RangeInclusive<u32>,
        token_ranges: &[std::ops::RangeInclusive<u32>],
    ) {
        if self.is_full() {
            return;
        }
        let tokens = rangeset_from_ranges(token_ranges.iter().cloned());
        if tokens.is_empty() {
            return;
        }
        let mut expanded = self.expanded_entries();
        for tsid in tsid_range {
            expanded
                .entry(tsid)
                .and_modify(|existing| *existing = existing.clone() | tokens.clone())
                .or_insert_with(|| tokens.clone());
        }
        *self = compress_expanded(&expanded);
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn clear(&mut self) {
        *self = Self::empty();
    }

    pub fn is_full(&self) -> bool {
        let mut entries = self.0.range_values();
        let Some((range, tokens)) = entries.next() else {
            return false;
        };
        entries.next().is_none()
            && *range.start() == WEIGHT_ALL_SENTINEL
            && *range.end() == WEIGHT_ALL_SENTINEL
            && is_sentinel_token_set(tokens.as_ref())
    }

    pub fn is_empty(&self) -> bool {
        self.0.ranges().next().is_none()
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn num_ranges(&self) -> usize {
        self.0.range_values_len()
    }

    /// Return the set of TSID ranges covered by this weight, ignoring token
    /// subranges. `None` denotes the mathematical full weight, whose TSID
    /// coverage may overlap any finite TSID set.
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn tsid_coverage(&self) -> Option<RangeSetBlaze<u32>> {
        if self.is_full() {
            return None;
        }
        Some(self.0.ranges().collect())
    }

    /// Return the cardinality of the finite TSID projection of this weight
    /// without materializing a `RangeSetBlaze`. `None` denotes the full
    /// sentinel weight.
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn tsid_coverage_len(&self) -> Option<usize> {
        if self.is_full() {
            return None;
        }
        Some(
            self.0
                .ranges()
                .map(|range| (*range.end() as usize - *range.start() as usize) + 1)
                .sum(),
        )
    }

    pub fn union(&self, other: &Self) -> Self {
        if self.is_full() || other.is_full() {
            return Self::all();
        }
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        if Arc::ptr_eq(&self.0, &other.0) {
            return self.clone();
        }
        if let Some(existing) = lookup_memoized_weight_op(WeightOpKind::Union, self, other) {
            return existing;
        }
        let result = self.union_uncached(other);
        store_memoized_weight_op(WeightOpKind::Union, self, other, &result);
        result
    }

    fn union_uncached(&self, other: &Self) -> Self {
        if let Some(result) = union_disjoint_tsid_ranges(self, other) {
            return result;
        }

        let left_single = single_compact_entry(self);
        let right_single = single_compact_entry(other);

        if let (Some(left), Some(right)) = (&left_single, &right_single) {
            combine_single_entries(&left, &right, union_token_sets)
        } else {
            union_compact_entries(self, other)
        }
    }

    pub fn union_all<'a>(weights: impl IntoIterator<Item = &'a Self>) -> Self {
        let mut meaningful = SmallVec::<[&Weight; 8]>::new();
        for weight in weights {
            if weight.is_full() {
                return Self::all();
            }
            if weight.is_empty() {
                continue;
            }
            meaningful.push(weight);
        }

        let result = match meaningful.len() {
            0 => Self::empty(),
            1 => meaningful[0].clone(),
            _ if meaningful.len() > 4 => {
            meaningful.sort_unstable_by_key(|w| w.ptr_key());
            meaningful.dedup_by_key(|w| w.ptr_key());
            if meaningful.len() == 1 {
                meaningful[0].clone()
            } else if let Some(result) = union_all_single_tsid_entries(&meaningful) {
                result
            } else if meaningful.len() > 4 {
                union_all_multiway(&meaningful)
            } else {
                let mut iter = meaningful.into_iter();
                let mut acc = iter.next().unwrap().clone();
                for weight in iter {
                    acc = acc.union(weight);
                }
                acc
            }
            }
            _ => {
                if let Some(result) = union_all_single_tsid_entries(&meaningful) {
                    result
                } else {
                    let mut iter = meaningful.into_iter();
                    let mut acc = iter.next().unwrap().clone();
                    for weight in iter {
                        acc = acc.union(weight);
                    }
                    acc
                }
            }
        };
        result
    }

    /// Exact direct multi-way union. This avoids allocating pairwise
    /// intermediate weights when several complex operands are merged once.
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn union_all_direct<'a>(weights: impl IntoIterator<Item = &'a Self>) -> Self {
        let mut meaningful = SmallVec::<[&Weight; 8]>::new();
        for weight in weights {
            if weight.is_full() {
                return Self::all();
            }
            if !weight.is_empty() {
                meaningful.push(weight);
            }
        }
        meaningful.sort_unstable_by_key(|weight| weight.ptr_key());
        meaningful.dedup_by_key(|weight| weight.ptr_key());
        match meaningful.len() {
            0 => Self::empty(),
            1 => meaningful[0].clone(),
            _ => union_all_single_tsid_entries(&meaningful)
                .unwrap_or_else(|| union_all_multiway(&meaningful)),
        }
    }

    /// Exact multi-way union specialized for weighted-DWA reconstruction.
    ///
    /// Reconstruction can accumulate thousands of overlapping TSID intervals
    /// that repeatedly carry the same interned token-set body. Coalesce only
    /// those same-valued coordinate intervals before the ordinary event sweep;
    /// unrelated bulk-union callers retain the normal path and pay no scan.
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn union_all_reconstruction<'a>(
        weights: impl IntoIterator<Item = &'a Self>,
    ) -> Self {
        let mut meaningful = SmallVec::<[&Weight; 8]>::new();
        for weight in weights {
            if weight.is_full() {
                return Self::all();
            }
            if !weight.is_empty() {
                meaningful.push(weight);
            }
        }
        match meaningful.len() {
            0 => Self::empty(),
            1 => meaningful[0].clone(),
            _ if meaningful.len() > 4 => {
                meaningful.sort_unstable_by_key(|weight| weight.ptr_key());
                meaningful.dedup_by_key(|weight| weight.ptr_key());
                if meaningful.len() == 1 {
                    meaningful[0].clone()
                } else if let Some(result) = union_all_single_tsid_entries(&meaningful) {
                    result
                } else if meaningful.len() > 4 {
                    union_all_multiway_impl(&meaningful, true)
                } else {
                    let mut iter = meaningful.into_iter();
                    let mut acc = iter.next().unwrap().clone();
                    for weight in iter {
                        acc = acc.union(weight);
                    }
                    acc
                }
            }
            _ => {
                if let Some(result) = union_all_single_tsid_entries(&meaningful) {
                    result
                } else {
                    let mut iter = meaningful.into_iter();
                    let mut acc = iter.next().unwrap().clone();
                    for weight in iter {
                        acc = acc.union(weight);
                    }
                    acc
                }
            }
        }
    }

    pub fn intersection(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::empty();
        }
        if Arc::ptr_eq(&self.0, &other.0) {
            return self.clone(); // Same weight → intersection is itself
        }
        if self.is_full() {
            return other.clone();
        }
        if other.is_full() {
            return self.clone();
        }

        let existing = with_public_weight_intersection_memo(|memo| memo.lookup(self, other));
        if let Some(existing) = existing {
            return existing;
        }

        let result = self.intersection_uncached_impl(other);
        with_public_weight_intersection_memo(|memo| memo.store(self, other, &result));
        result
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn intersection_uncached(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::empty();
        }
        if Arc::ptr_eq(&self.0, &other.0) {
            return self.clone(); // Same weight → intersection is itself
        }
        if self.is_full() {
            return other.clone();
        }
        if other.is_full() {
            return self.clone();
        }
        self.intersection_uncached_impl(other)
    }

    fn intersection_uncached_impl(&self, other: &Self) -> Self {
        if let (Some(left), Some(right)) = (single_compact_entry(self), single_compact_entry(other))
        {
            combine_single_entries(&left, &right, intersect_token_sets)
        } else if let Some(single) = single_compact_entry(self) {
            intersect_single_entry_with_weight(&single, other)
        } else if let Some(single) = single_compact_entry(other) {
            intersect_single_entry_with_weight(&single, self)
        } else {
            intersect_weights(self, other)
        }
    }

    pub fn difference(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_full() {
            return Self::empty();
        }
        if other.is_empty() {
            return self.clone();
        }
        if Arc::ptr_eq(&self.0, &other.0) {
            return Self::empty();
        }
        if self.is_full() {
            // Cannot compute all \ other without an explicit universe.
            // Return all() as a safe over-approximation.  Callers that need
            // exact complements should use the dedicated complement() method
            // which returns empty() as a no-op sentinel instead.
            return Self::all();
        }
        with_memoized_weight_op(WeightOpKind::Difference, self, other, || self.difference_uncached(other))
    }

    fn difference_uncached(&self, other: &Self) -> Self {
        difference_weights(self, other)
    }

    pub fn complement(&self) -> Self {
        if self.is_full() {
            Self::empty()
        } else if self.is_empty() {
            Self::all()
        } else {
            // Cannot compute a proper per-TSID complement without an explicit
            // token/TSID universe.  Returning empty() makes the determinization
            // normalization step a no-op (target ∪ empty = target), which
            // preserves correctness at the cost of potentially more DWA states
            // (no subset collapsing via normalization).  The previous approach
            // was `all().difference(self)` which always returned `all()` due to
            // the sentinel representation, causing target subsets to collapse
            // into `Weight::all()` and producing false positives.
            Self::empty()
        }
    }

    pub fn from_token_set_for_tsid(tsid: u32, tokens: RangeSetBlaze<u32>) -> Self {
        if tokens.is_empty() {
            return Self::empty();
        }
        Self::from_uniform(tsid..=tsid, tokens)
    }

    pub fn tokens_for_tsid(&self, tsid: u32) -> RangeSetBlaze<u32> {
        if self.is_full() {
            return sentinel_token_set();
        }
        self.0
            .get(tsid)
            .map(|tokens| tokens.as_ref().clone())
            .unwrap_or_else(RangeSetBlaze::new)
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn single_compact_entry_parts(
        &self,
    ) -> Option<(u32, u32, SharedTokenSet)> {
        let entry = single_compact_entry(self)?;
        Some((entry.start, entry.end, entry.tokens))
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn outer_range_count(&self) -> usize {
        self.0.range_values_len()
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn single_tsid_shared_entry(&self) -> Option<(u32, SharedTokenSet)> {
        let (start, end, tokens) = self.single_compact_entry_parts()?;
        if start == end && start != WEIGHT_ALL_SENTINEL {
            Some((start, tokens))
        } else {
            None
        }
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn union_single_tsid_shared_entries(
        entries: impl IntoIterator<Item = (u32, SharedTokenSet)>,
    ) -> Self {
        let mut per_tsid: BTreeMap<u32, SharedTokenSet> = BTreeMap::new();

        for (tsid, tokens) in entries {
            per_tsid
                .entry(tsid)
                .and_modify(|existing| *existing = shared_token_union(existing, &tokens))
                .or_insert(tokens);
        }

        let mut builder = CompactRangeBuilder::new();
        for (tsid, tokens) in per_tsid {
            builder.push(tsid, tsid, tokens);
        }
        builder.finish()
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn intersect_single_parts(
        &self,
        start: u32,
        end: u32,
        tokens: &SharedTokenSet,
    ) -> Self {
        let single = WeightRangeEntry {
            start,
            end,
            tokens: Arc::clone(tokens),
        };
        intersect_single_entry_with_weight(&single, self)
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn for_each_intersection_tokens_with_single<F>(
        &self,
        start: u32,
        end: u32,
        single_tokens: &RangeSetBlaze<u32>,
        mut f: F,
    ) where
        F: FnMut(&RangeSetBlaze<u32>),
    {
        let mut overlap_cache: SmallVec<[(
            *const RangeSetBlaze<u32>,
            Option<SharedTokenSet>,
        ); 8]> = SmallVec::new();

        for (range, other_tokens) in self.0.range_values() {
            if end < *range.start() || *range.end() < start {
                continue;
            }

            if single_tokens == other_tokens.as_ref() {
                f(single_tokens);
                continue;
            }

            let cache_key = Arc::as_ptr(other_tokens);
            if let Some((_, cached)) = overlap_cache.iter().find(|(ptr, _)| *ptr == cache_key) {
                if let Some(cached_tokens) = cached {
                    f(cached_tokens.as_ref());
                }
                continue;
            }

            let overlap = single_tokens & other_tokens.as_ref();
            if overlap.is_empty() {
                overlap_cache.push((cache_key, None));
                continue;
            }

            let overlap_tokens = shared_rangeset(overlap);
            f(overlap_tokens.as_ref());
            overlap_cache.push((cache_key, Some(overlap_tokens)));
        }
    }

    /// Iterate over the unique (Arc-deduplicated) token sets in this weight.
    /// Each token set may cover one or more TSID ranges.
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn unique_token_sets(&self) -> Vec<&RangeSetBlaze<u32>> {
        if self.is_full() || self.is_empty() {
            return Vec::new();
        }
        let mut seen: Vec<*const RangeSetBlaze<u32>> = Vec::new();
        let mut result = Vec::new();
        for (_range, tokens) in self.0.range_values() {
            let ptr = Arc::as_ptr(tokens);
            if !seen.contains(&ptr) {
                seen.push(ptr);
                result.push(tokens.as_ref());
            }
        }
        result
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        if self.is_empty() || other.is_empty() {
            return true;
        }
        if Arc::ptr_eq(&self.0, &other.0) {
            return false; // Same non-empty weight → not disjoint
        }
        if self.is_full() || other.is_full() {
            return false;
        }
        let mut left_iter = self.0.range_values();
        let mut right_iter = other.0.range_values();
        let mut left_entry = left_iter.next();
        let mut right_entry = right_iter.next();

        while let (Some((lr, lt)), Some((rr, rt))) = (&left_entry, &right_entry) {
            let start = (*lr.start()).max(*rr.start());
            let end = (*lr.end()).min(*rr.end());
            if start <= end && !lt.as_ref().is_disjoint(rt.as_ref()) {
                return false;
            }
            if lr.end() <= rr.end() {
                left_entry = left_iter.next();
            } else {
                right_entry = right_iter.next();
            }
        }
        true
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        if self.is_empty() || other.is_full() {
            return true;
        }
        if other.is_empty() || self.is_full() {
            return false;
        }
        let mut self_iter = self.0.range_values();
        let mut other_iter = other.0.range_values();
        let mut self_current = self_iter.next();
        let mut other_current = other_iter.next();
        // Track how far we've verified coverage of the current self entry
        let mut self_verified_up_to: Option<u32> = None;

        while let Some((self_range, self_tokens)) = &self_current {
            let self_start = self_verified_up_to
                .map(|v| v + 1)
                .unwrap_or(*self_range.start());

            if self_start > *self_range.end() {
                self_current = self_iter.next();
                self_verified_up_to = None;
                continue;
            }

            let Some((other_range, other_tokens)) = &other_current else {
                return false;
            };

            if *other_range.end() < self_start {
                other_current = other_iter.next();
                continue;
            }

            if *other_range.start() > self_start {
                return false;
            }

            if !self_tokens.as_ref().is_subset(other_tokens.as_ref()) {
                return false;
            }

            let covered_up_to = (*self_range.end()).min(*other_range.end());
            self_verified_up_to = Some(covered_up_to);

            if covered_up_to >= *self_range.end() {
                self_current = self_iter.next();
                self_verified_up_to = None;
            }
            if covered_up_to >= *other_range.end() {
                other_current = other_iter.next();
            }
        }
        true
    }

    /// Clip all token sets to `0..=max_token`, removing any entries that become empty.
    /// Does nothing to the ALL sentinel.
    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn clip_tokens(&mut self, max_token: u32) {
        if self.is_full() || self.is_empty() {
            return;
        }
        let clip: RangeSetBlaze<u32> = std::iter::once(0..=max_token).collect();
        let mut new_map = WeightMap::new();
        for (tsid_range, tokens) in self.0.range_values() {
            let clipped = tokens.as_ref() & &clip;
            if !clipped.is_empty() {
                new_map.extend_simple(std::iter::once((tsid_range, shared_rangeset(clipped))));
            }
        }
        *self = finalize_weight_map(new_map);
    }

    fn expanded_entries(&self) -> BTreeMap<u32, RangeSetBlaze<u32>> {
        if self.is_full() {
            return BTreeMap::new();
        }
        let mut out = BTreeMap::new();
        for (range, tokens) in self.0.range_values() {
            for tsid in range {
                out.insert(tsid, tokens.as_ref().clone());
            }
        }
        out
    }

    fn to_serde(&self) -> WeightSerde {
        if self.is_full() {
            return WeightSerde {
                all: true,
                entries: Vec::new(),
            };
        }
        WeightSerde {
            all: false,
            entries: self
                .0
                .range_values()
                .map(|(range, tokens)| WeightSerdeEntry {
                    tsid: [*range.start(), *range.end()],
                    tokens: rangeset_to_vec(tokens.as_ref()),
                })
                .collect(),
        }
    }
}

impl PartialEq for Weight {
    fn eq(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        if self.is_full() || other.is_full() {
            return self.is_full() == other.is_full();
        }
        let mut a = self.0.range_values();
        let mut b = other.0.range_values();
        loop {
            match (a.next(), b.next()) {
                (None, None) => return true,
                (Some((ra, ta)), Some((rb, tb))) => {
                    if ra != rb || ta.as_ref() != tb.as_ref() {
                        return false;
                    }
                }
                _ => return false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_range_builder_preserves_stream_and_token_representatives() {
        // Keep the original placeholder-based implementation as an oracle.
        // Comparing raw maps avoids global interning hiding an identity change.
        struct Reference {
            map: WeightMap,
            start: Option<u32>,
            end: u32,
            tokens: SharedTokenSet,
        }
        impl Reference {
            fn flush(&mut self) {
                if let Some(start) = self.start.take() {
                    let tokens = std::mem::replace(&mut self.tokens, Arc::clone(&EMPTY_RANGESET));
                    self.map.extend_simple(std::iter::once((start..=self.end, tokens)));
                }
            }
            fn push(&mut self, start: u32, end: u32, tokens: SharedTokenSet) {
                if self.start.is_some()
                    && self.end.checked_add(1) == Some(start)
                    && same_shared_token_set(&self.tokens, &tokens)
                {
                    self.end = end;
                } else {
                    self.flush();
                    self.start = Some(start);
                    self.end = end;
                    self.tokens = tokens;
                }
            }
        }
        let a = Arc::new(RangeSetBlaze::from_iter([1_u32..=4, 7..=9]));
        let equal_a = Arc::new(a.as_ref().clone());
        let b = Arc::new(RangeSetBlaze::from_iter([3_u32..=8]));
        assert!(!Arc::ptr_eq(&a, &equal_a));
        let entries = [
            (0, 0, Arc::clone(&a)),
            (1, 2, Arc::clone(&equal_a)),
            (3, 6, Arc::clone(&b)),
            (2, 5, Arc::clone(&a)),
            (9, 11, Arc::clone(&b)),
            (u32::MAX - 2, u32::MAX - 1, Arc::clone(&a)),
            (u32::MAX, u32::MAX, Arc::clone(&equal_a)),
            (3, 3, Arc::clone(&EMPTY_RANGESET)),
        ];
        for seed in 0..1024_usize {
            let mut reference = Reference {
                map: WeightMap::new(), start: None, end: 0,
                tokens: Arc::clone(&EMPTY_RANGESET),
            };
            let mut actual = CompactRangeBuilder::new();
            let mut choice = seed;
            for step in 0..(seed % 33) {
                let entry = &entries[choice % entries.len()];
                choice = choice.rotate_right(3).wrapping_add(step * 17 + seed);
                reference.push(entry.0, entry.1, Arc::clone(&entry.2));
                actual.push(entry.0, entry.1, Arc::clone(&entry.2));
                if (seed + step) % 3 == 0 {
                    reference.flush();
                    actual.flush();
                }
            }
            reference.flush();
            actual.flush();
            assert_eq!(actual.map, reference.map, "stream {seed}");
            let identities = |map: &WeightMap| map.range_values()
                .map(|(range, tokens)| (range, Arc::as_ptr(tokens) as usize))
                .collect::<Vec<_>>();
            assert_eq!(identities(&actual.map), identities(&reference.map), "token representatives {seed}");
        }
    }

    #[test]
    fn token_intersection_defaults_to_sweep_and_retains_reference_override() {
        assert_eq!(token_intersection_mode(None), 2);
        assert_eq!(token_intersection_mode(Some("sweep")), 2);
        assert_eq!(token_intersection_mode(Some("memo")), 1);
        for value in ["off", "0", "legacy", "", "unknown"] {
            assert_eq!(token_intersection_mode(Some(value)), 0);
        }
    }

    #[test]
    fn cached_equal_token_sets_keep_the_callers_left_representative() {
        // Equal but separately allocated sets occur after artifact remapping.
        // Canonical subset construction prefers the current left operand,
        // even when a commutative memo was populated in the opposite order.
        for sweep in [false, true] {
            let left = Arc::new(RangeSetBlaze::from_iter([2_u32..=8, 14..=20]));
            let right = Arc::new(left.as_ref().clone());
            assert!(!Arc::ptr_eq(&left, &right));
            let first = shared_token_intersection_cached(&left, &right, sweep).unwrap();
            assert!(Arc::ptr_eq(&first, &left));
            let reversed = shared_token_intersection_cached(&right, &left, sweep).unwrap();
            assert!(Arc::ptr_eq(&reversed, &right), "reversed memo hit changed operand sharing");
        }
    }

    #[test]
    fn token_range_sweep_matches_every_pair_of_small_sets() {
        let sets = (0_u32..256).map(|bits| {
            Arc::new(RangeSetBlaze::from_iter((0_u32..8).filter(|bit| bits & (1 << bit) != 0)))
        }).collect::<Vec<_>>();
        for left in &sets {
            for right in &sets {
                let expected = left.as_ref() & right.as_ref();
                assert_eq!(*intersect_token_ranges_once(left, right), expected);
            }
        }
    }

    #[test]
    fn token_range_sweep_preserves_containment_sharing_and_u32_extremes() {
        let ranges = [
            vec![], vec![0..=0], vec![u32::MAX..=u32::MAX],
            vec![0..=u32::MAX], vec![1..=u32::MAX - 1],
            vec![0..=2, 9..=17, u32::MAX - 2..=u32::MAX],
            vec![2..=9, 16..=19, u32::MAX - 1..=u32::MAX],
        ];
        let sets = ranges.into_iter().map(|ranges| Arc::new(RangeSetBlaze::from_iter(ranges)))
            .collect::<Vec<_>>();
        for left in &sets {
            for right in &sets {
                let expected = left.as_ref() & right.as_ref();
                let actual = intersect_token_ranges_once(left, right);
                assert_eq!(*actual, expected);
                if left.is_subset(right) {
                    assert!(Arc::ptr_eq(&actual, left));
                } else if right.is_subset(left) {
                    assert!(Arc::ptr_eq(&actual, right));
                }
                for sweep in [false, true] {
                    let expected = shared_token_intersection_legacy(left, right);
                    let actual = shared_token_intersection_cached(left, right, sweep);
                    assert_eq!(actual.as_deref(), expected.as_deref());
                }
            }
        }
    }

    #[test]
    fn contained_token_intersection_memo_does_not_retain_operands() {
        let left = Arc::new(RangeSetBlaze::from_iter([2..=4, 8..=10]));
        let right = Arc::new(RangeSetBlaze::from_iter([0..=12]));
        let result = intersect_token_ranges_once(&left, &right);
        assert!(Arc::ptr_eq(&result, &left));
        let key = TokenSetOpKey::for_token_sets(TokenSetOpKind::Intersection, &left, &right);
        let mut memo = WeightOpMemo::default();
        memo.store_token_set(key, &left, &right, &result);
        assert_eq!(memo.lookup_token_set(key).as_deref(), Some(left.as_ref()));
        let weak = Arc::downgrade(&right);
        drop(right);
        assert!(weak.upgrade().is_none());
        assert!(memo.lookup_token_set(key).is_none(), "dead operands invalidate a contained result too");
    }

    fn weight_for_tsid(tsid: u32, ranges: &[(u32, u32)]) -> Weight {
        let token_set = rangeset_from_ranges(ranges.iter().map(|(start, end)| *start..=*end));
        Weight::from_token_set_for_tsid(tsid, token_set)
    }

    #[test]
    fn point_single_intersection_matches_general_intersection() {
        let token_sets = [
            shared_rangeset(RangeSetBlaze::from_iter([0..=7])),
            shared_rangeset(RangeSetBlaze::from_iter([4..=15])),
            shared_rangeset(RangeSetBlaze::from_iter([12..=23])),
        ];
        let other = Weight::from_per_tsid_shared(
            (0..128).map(|tsid| (tsid, Arc::clone(&token_sets[tsid as usize % 3]))),
        );
        let point = Weight::from_token_set_for_tsid(
            117,
            RangeSetBlaze::from_iter([6..=18]),
        );
        let single = single_compact_entry(&point).expect("point weight must be compact");
        let fast = intersect_single_entry_with_weight(&single, &other);
        let reference = intersect_weights(&point, &other);
        assert_eq!(fast, reference);

        let absent = Weight::from_token_set_for_tsid(
            256,
            RangeSetBlaze::from_iter([6..=18]),
        );
        let absent_single = single_compact_entry(&absent).expect("point weight must be compact");
        assert!(intersect_single_entry_with_weight(&absent_single, &other).is_empty());
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

    fn small_exhaustive_weights() -> Vec<Weight> {
        (0u32..64)
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
            .collect()
    }

    #[test]
    fn outer_join_union_matches_handwritten_exhaustively() {
        let weights = small_exhaustive_weights();
        for left in &weights {
            for right in &weights {
                assert_eq!(
                    union_compact_entries(left, right),
                    union_compact_entries_handwritten(left, right),
                    "outer_join union differs for left={left} right={right}",
                );
            }
        }
    }

    #[test]
    fn outer_join_combine_matches_boundary_sweep_exhaustively() {
        type Combine =
            fn(Option<&SharedTokenSet>, Option<&SharedTokenSet>) -> Option<SharedTokenSet>;
        let combines: [(&str, Combine); 3] = [
            ("union", union_token_sets),
            ("intersection", intersect_token_sets),
            ("difference", difference_token_sets),
        ];
        let weights = small_exhaustive_weights();
        for left in &weights {
            for right in &weights {
                for (name, combine) in combines {
                    assert_eq!(
                        combine_compact_entries(left, right, combine),
                        combine_compact_entries_boundary(left, right, combine),
                        "outer_join {name} differs for left={left} right={right}",
                    );
                }
                assert_eq!(
                    intersect_weights(left, right),
                    combine_compact_entries(left, right, intersect_token_sets),
                    "streaming intersection differs for left={left} right={right}",
                );
            }
        }
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
}

impl Eq for Weight {}

impl std::hash::Hash for Weight {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(cached_structural_weight_hash(self));
    }
}

impl std::fmt::Display for Weight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return write!(f, "∅");
        }
        if self.is_full() {
            return write!(f, "ALL");
        }

        let parts: Vec<String> = self
            .0
            .range_values()
            .map(|(range, tokens)| {
                let tsid = if range.start() == range.end() {
                    format!("{}", range.start())
                } else {
                    format!("{}..={}", range.start(), range.end())
                };
                format!("{tsid}→{}", rangeset_to_string(tokens.as_ref()))
            })
            .collect();
        write!(f, "{}", parts.join("; "))
    }
}

const WEIGHT_ALL_SENTINEL: u32 = u32::MAX;

enum PooledWeightSerdeEncoder {
    ByPointer(FxHashMap<usize, u32>),
    // Deferred decoding deliberately stores identical empty Weight values in
    // the structural maps. Their identity lives in a separate packed-ID map,
    // so re-encoding must use that map's ordered IDs rather than pointer keys.
    OrderedIds { ids: Vec<u32>, next: usize },
}

thread_local! {
    /// Versioned constraint serialization can replace ordinary structural
    /// Weight serde with compact pool indices. The context is thread-local so
    /// normal serde users and legacy artifact versions retain the historical
    /// representation.
    static POOLED_WEIGHT_SERDE_ENCODE: RefCell<Option<PooledWeightSerdeEncoder>> =
        RefCell::new(None);
    static POOLED_WEIGHT_SERDE_DECODE: RefCell<Option<Vec<Weight>>> =
        RefCell::new(None);
    static POOLED_WEIGHT_SERDE_DEFER_COUNT: Cell<Option<usize>> = const { Cell::new(None) };
    static POOLED_WEIGHT_SERDE_DEFERRED_IDS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Install a pool-index encoder for subsequent `Weight::serialize` calls on
/// this thread. `weights[index]` must be the weight associated with that index.
pub fn begin_pooled_weight_serde_encode(weights: &[Weight]) {
    let mut by_ptr = FxHashMap::default();
    by_ptr.reserve(weights.len());
    for (index, weight) in weights.iter().enumerate() {
        by_ptr.insert(weight.ptr_key(), index as u32);
    }
    POOLED_WEIGHT_SERDE_ENCODE.with(|slot| {
        let previous = slot.borrow_mut().replace(PooledWeightSerdeEncoder::ByPointer(by_ptr));
        assert!(previous.is_none(), "pooled Weight encode context must not nest");
    });
}

/// Preserve already-packed weight IDs during a single serialization pass.
///
/// Supply one ID per Weight field, in serializer visitation order. Use
/// serialize_into, not a serializer which visits values twice for sizing.
pub fn begin_pooled_weight_serde_encode_ids(ids: Vec<u32>) {
    POOLED_WEIGHT_SERDE_ENCODE.with(|slot| {
        let previous = slot.borrow_mut().replace(PooledWeightSerdeEncoder::OrderedIds { ids, next: 0 });
        assert!(previous.is_none(), "pooled Weight encode context must not nest");
    });
}

pub fn end_pooled_weight_serde_encode() {
    POOLED_WEIGHT_SERDE_ENCODE.with(|slot| {
        if let Some(PooledWeightSerdeEncoder::OrderedIds { ids, next }) = slot.borrow_mut().take() {
            assert_eq!(next, ids.len(), "packed Weight encode visitation count mismatch");
        }
    });
}

/// Install the already-decoded weight pool for subsequent
/// `Weight::deserialize` calls on this thread.
pub fn begin_pooled_weight_serde_decode(weights: Vec<Weight>) {
    POOLED_WEIGHT_SERDE_DECODE.with(|slot| {
        let previous = slot.borrow_mut().replace(weights);
        assert!(previous.is_none(), "pooled Weight decode context must not nest");
    });
}

pub fn end_pooled_weight_serde_decode() {
    POOLED_WEIGHT_SERDE_DECODE.with(|slot| {
        slot.borrow_mut().take();
    });
    POOLED_WEIGHT_SERDE_DEFER_COUNT.with(|slot| slot.set(None));
}

pub fn begin_pooled_weight_serde_deferred_decode(weight_count: usize) {
    POOLED_WEIGHT_SERDE_DEFER_COUNT.with(|slot| {
        assert!(slot.replace(Some(weight_count)).is_none(), "deferred pooled Weight decode context must not nest");
    });
    POOLED_WEIGHT_SERDE_DEFERRED_IDS.with(|slot| slot.borrow_mut().clear());
}

pub fn take_pooled_weight_serde_deferred_ids() -> Vec<u32> {
    POOLED_WEIGHT_SERDE_DEFERRED_IDS.with(|slot| std::mem::take(&mut *slot.borrow_mut()))
}

fn pooled_weight_serde_encode_index(weight: &Weight) -> Option<u32> {
    POOLED_WEIGHT_SERDE_ENCODE.with(|slot| {
        match slot.borrow_mut().as_mut()? {
            PooledWeightSerdeEncoder::ByPointer(by_ptr) => by_ptr.get(&weight.ptr_key()).copied(),
            PooledWeightSerdeEncoder::OrderedIds { ids, next } => {
                let id = ids.get(*next).copied()?;
                *next += 1;
                Some(id)
            }
        }
    })
}

fn pooled_weight_serde_encode_enabled() -> bool {
    POOLED_WEIGHT_SERDE_ENCODE.with(|slot| slot.borrow().is_some())
}

fn pooled_weight_serde_decode_index(index: u32) -> Option<Weight> {
    POOLED_WEIGHT_SERDE_DECODE.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|weights| weights.get(index as usize).cloned())
    })
}

fn pooled_weight_serde_decode_enabled() -> bool {
    POOLED_WEIGHT_SERDE_DECODE.with(|slot| slot.borrow().is_some())
        || POOLED_WEIGHT_SERDE_DEFER_COUNT.with(|slot| slot.get().is_some())
}

fn pooled_weight_serde_deferred_decode_index(index: u32) -> Option<Weight> {
    let valid = POOLED_WEIGHT_SERDE_DEFER_COUNT.with(|slot| {
        slot.get().is_some_and(|count| (index as usize) < count)
    });
    if !valid {
        return None;
    }
    POOLED_WEIGHT_SERDE_DEFERRED_IDS.with(|slot| slot.borrow_mut().push(index));
    Some(Weight::empty())
}

#[inline]
fn pooled_put_var_u32(out: &mut Vec<u8>, mut value: u32) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

#[inline]
fn pooled_put_var_u64(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

#[inline]
fn pooled_take_var_u32(input: &[u8], pos: &mut usize) -> Result<u32, String> {
    let mut value = 0u32;
    let mut shift = 0u32;
    for _ in 0..5 {
        let byte = *input
            .get(*pos)
            .ok_or_else(|| "truncated packed Weight-pool varint".to_owned())?;
        *pos += 1;
        if shift == 28 && byte > 0x0f {
            return Err("overflowing packed Weight-pool u32 varint".to_owned());
        }
        value |= ((byte & 0x7f) as u32) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
    Err("overflowing packed Weight-pool u32 varint".to_owned())
}

#[inline]
fn pooled_take_var_u64(input: &[u8], pos: &mut usize) -> Result<u64, String> {
    let mut value = 0u64;
    let mut shift = 0u32;
    for index in 0..10 {
        let byte = *input
            .get(*pos)
            .ok_or_else(|| "truncated packed Weight-pool varint".to_owned())?;
        *pos += 1;
        if index == 9 && byte > 1 {
            return Err("overflowing packed Weight-pool u64 varint".to_owned());
        }
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
    Err("overflowing packed Weight-pool u64 varint".to_owned())
}

/// Pack a list of already-deduplicated weights into a compact shared
/// token-set/weight pool. The caller's weight order is preserved exactly so
/// ordinary Weight fields can serialize as pool indices.
#[cfg(feature = "internal-api")]
pub fn pack_pooled_weights(weights: &[Weight]) -> Vec<u8> {
    let total_weight_ranges = weights
        .iter()
        .filter(|weight| !weight.is_full())
        .map(|weight| weight.0.range_values_len())
        .sum::<usize>();
    let mut token_set_by_ptr = FxHashMap::<usize, u32>::default();
    let token_set_capacity_hint = weights.len().saturating_mul(2).min(total_weight_ranges);
    token_set_by_ptr.reserve(token_set_capacity_hint);
    let mut token_sets = Vec::<SharedTokenSet>::with_capacity(token_set_capacity_hint);
    let mut weight_body_bytes = Vec::<u8>::with_capacity(
        total_weight_ranges
            .saturating_mul(5)
            .saturating_add(weights.len().saturating_mul(2)),
    );
    let mut weight_body_offsets = Vec::<u32>::with_capacity(weights.len() + 1);
    weight_body_offsets.push(0);
    let mut last_token_set: Option<(usize, u32)> = None;
    for weight in weights {
        if weight.is_full() {
            weight_body_bytes.push(1);
        } else {
            weight_body_bytes.push(0);
            pooled_put_var_u32(&mut weight_body_bytes, weight.0.range_values_len() as u32);
            let mut previous_end_plus_one = 0u64;
            for (range, tokens) in weight.raw_range_values() {
                let start = *range.start();
                let end = *range.end();
                let gap = (start as u64)
                    .checked_sub(previous_end_plus_one)
                    .expect("pooled weight ranges are sorted and disjoint");
                pooled_put_var_u64(&mut weight_body_bytes, gap);
                pooled_put_var_u32(&mut weight_body_bytes, end - start);

                let ptr = Arc::as_ptr(tokens) as usize;
                let token_set = if let Some((last_ptr, last_id)) = last_token_set {
                    if last_ptr == ptr {
                        last_id
                    } else {
                        let next_index = token_sets.len() as u32;
                        *token_set_by_ptr.entry(ptr).or_insert_with(|| {
                            token_sets.push(tokens.clone());
                            next_index
                        })
                    }
                } else {
                    let next_index = token_sets.len() as u32;
                    *token_set_by_ptr.entry(ptr).or_insert_with(|| {
                        token_sets.push(tokens.clone());
                        next_index
                    })
                };
                last_token_set = Some((ptr, token_set));
                pooled_put_var_u32(&mut weight_body_bytes, token_set);
                previous_end_plus_one = end as u64 + 1;
            }
        }
        weight_body_offsets.push(
            u32::try_from(weight_body_bytes.len())
                .expect("packed Weight-pool bodies must fit the WPL3 u32 length domain"),
        );
    }

    // WPL3 keeps the artifact-local token-set identity pool and makes every
    // token-set and Weight record independently decodable.  Unlike WPL2 it does
    // not pool TSID geometry across Weights: real schema artifacts almost never
    // share complete geometry, so geometry interning added hashing/save work and
    // a few bytes without meaningful deduplication. Length-prefixing the Weight
    // records is sufficient to recover parallel decode.
    let mut out = Vec::with_capacity(
        weight_body_bytes
            .len()
            .saturating_add(token_sets.len().saturating_mul(8))
            .saturating_add(weights.len().saturating_mul(2))
            .saturating_add(16),
    );
    out.extend_from_slice(b"WPL3");
    pooled_put_var_u32(&mut out, token_sets.len() as u32);
    let mut token_set_body_bytes = Vec::<u8>::new();
    let mut token_set_body_offsets = Vec::<u32>::with_capacity(token_sets.len() + 1);
    token_set_body_offsets.push(0);
    for tokens in &token_sets {
        let range_count = tokens.ranges().count();
        pooled_put_var_u32(&mut token_set_body_bytes, range_count as u32);
        let mut previous_end_plus_one = 0u64;
        for range in tokens.ranges() {
            let start = *range.start();
            let end = *range.end();
            let gap = (start as u64)
                .checked_sub(previous_end_plus_one)
                .expect("pooled token-set ranges are sorted and disjoint");
            pooled_put_var_u64(&mut token_set_body_bytes, gap);
            pooled_put_var_u32(&mut token_set_body_bytes, end - start);
            previous_end_plus_one = end as u64 + 1;
        }
        token_set_body_offsets.push(
            u32::try_from(token_set_body_bytes.len())
                .expect("packed Weight-pool token-set bodies must fit u32"),
        );
    }
    for offsets in token_set_body_offsets.windows(2) {
        let start = offsets[0] as usize;
        let end = offsets[1] as usize;
        pooled_put_var_u32(&mut out, (end - start) as u32);
        out.extend_from_slice(&token_set_body_bytes[start..end]);
    }

    pooled_put_var_u32(&mut out, weights.len() as u32);
    for offsets in weight_body_offsets.windows(2) {
        let start = offsets[0] as usize;
        let end = offsets[1] as usize;
        pooled_put_var_u32(&mut out, (end - start) as u32);
        out.extend_from_slice(&weight_body_bytes[start..end]);
    }
    out
}

pub fn unpack_pooled_weights(input: &[u8]) -> Result<Vec<Weight>, String> {
    if input.starts_with(b"WPL3") {
        return unpack_pooled_weights_v3(input);
    }
    if input.starts_with(b"WPL2") {
        return unpack_pooled_weights_v2(input);
    }
    unpack_pooled_weights_v1(input)
}

#[derive(Debug, Clone, Copy)]
struct PackedRuntimeWeightSpan {
    body_start: u32,
    body_len: u32,
    entry_count: u32,
    full: bool,
}

#[derive(Debug)]
pub struct PackedRuntimeWeightPool {
    wire: Arc<[u8]>,
    token_spans: Box<[(u32, u32)]>,
    weight_spans: Box<[PackedRuntimeWeightSpan]>,
}

#[derive(Clone, Copy)]
pub struct PackedRuntimePoolTokenSetRef<'a> {
    id: u32,
    body: &'a [u8],
}

impl<'a> PackedRuntimePoolTokenSetRef<'a> {
    #[inline]
    pub fn id(self) -> u32 { self.id }

    /// Read the encoded range count without visiting or materializing ranges.
    /// This lets optional runtime caches reject oversized sets before decoding.
    #[inline]
    pub fn range_count(self) -> usize {
        let mut pos = 0usize;
        pooled_take_var_u32(self.body, &mut pos).map_or(0, |count| count as usize)
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.range_count() == 0
    }

    pub fn for_each_range(self, mut visit: impl FnMut(u32, u32)) {
        let mut pos = 0usize;
        let Ok(range_count) = pooled_take_var_u32(self.body, &mut pos) else {
            return;
        };
        let mut previous_end_plus_one = 0u64;
        for _ in 0..range_count {
            let Ok(gap) = pooled_take_var_u64(self.body, &mut pos) else {
                return;
            };
            let Some(start64) = previous_end_plus_one.checked_add(gap) else {
                return;
            };
            let Ok(start) = u32::try_from(start64) else {
                return;
            };
            let Ok(len) = pooled_take_var_u32(self.body, &mut pos) else {
                return;
            };
            let Some(end) = start.checked_add(len) else {
                return;
            };
            visit(start, end);
            previous_end_plus_one = end as u64 + 1;
        }
    }
}

#[derive(Clone, Copy)]
pub struct PackedRuntimePoolWeightRef<'a> {
    pool: &'a PackedRuntimeWeightPool,
    id: u32,
}

impl<'a> PackedRuntimePoolWeightRef<'a> {
    #[inline]
    pub fn id(self) -> u32 { self.id }
    #[inline]
    pub fn is_full(self) -> bool { self.pool.weight_spans[self.id as usize].full }
    /// Number of encoded TSID intervals, available without allocating the
    /// decoded entry vector. Full weights have no explicit intervals.
    #[inline]
    pub fn entry_count(self) -> usize {
        self.pool.weight_spans[self.id as usize].entry_count as usize
    }
    #[inline]
    pub fn is_empty(self) -> bool {
        let span = self.pool.weight_spans[self.id as usize];
        !span.full && span.entry_count == 0
    }
    pub fn token_set_for_tsid(self, tsid: u32) -> Option<PackedRuntimePoolTokenSetRef<'a>> {
        let span = self.pool.weight_spans[self.id as usize];
        if span.full { return None; }
        let body_start = span.body_start as usize;
        let body = self.pool.wire.get(body_start..body_start + span.body_len as usize)?;
        let mut pos = 1usize;
        let entry_count = pooled_take_var_u32(body, &mut pos).ok()?;
        let mut previous_end_plus_one = 0u64;
        for _ in 0..entry_count {
            let gap = pooled_take_var_u64(body, &mut pos).ok()?;
            let start = u32::try_from(previous_end_plus_one.checked_add(gap)?).ok()?;
            let len = pooled_take_var_u32(body, &mut pos).ok()?;
            let end = start.checked_add(len)?;
            let token_set = pooled_take_var_u32(body, &mut pos).ok()?;
            if tsid < start {
                return None;
            }
            if tsid <= end {
                return self.pool.token_set(token_set);
            }
            previous_end_plus_one = end as u64 + 1;
        }
        None
    }

    pub fn entries(self) -> Vec<((u32, u32), PackedRuntimePoolTokenSetRef<'a>)> {
        let span = self.pool.weight_spans[self.id as usize];
        if span.full {
            return Vec::new();
        }
        let body_start = span.body_start as usize;
        let Some(body) = self.pool.wire.get(body_start..body_start + span.body_len as usize) else {
            return Vec::new();
        };
        let mut pos = 1usize;
        let Ok(entry_count) = pooled_take_var_u32(body, &mut pos) else {
            return Vec::new();
        };
        let mut result = Vec::with_capacity(entry_count as usize);
        let mut previous_end_plus_one = 0u64;
        for _ in 0..entry_count {
            let Ok(gap) = pooled_take_var_u64(body, &mut pos) else { break; };
            let Some(start64) = previous_end_plus_one.checked_add(gap) else { break; };
            let Ok(start) = u32::try_from(start64) else { break; };
            let Ok(len) = pooled_take_var_u32(body, &mut pos) else { break; };
            let Some(end) = start.checked_add(len) else { break; };
            let Ok(token_set) = pooled_take_var_u32(body, &mut pos) else { break; };
            let Some(tokens) = self.pool.token_set(token_set) else { break; };
            result.push(((start, end), tokens));
            previous_end_plus_one = end as u64 + 1;
        }
        result
    }
}

impl PackedRuntimeWeightPool {
    pub fn peek_weight_count(input: &[u8]) -> Result<usize, String> {
        if !input.starts_with(b"WPL3") {
            return Err("packed runtime Weight pool requires WPL3".to_owned());
        }
        let mut pos = 4usize;
        let token_set_count = pooled_take_var_u32(input, &mut pos)? as usize;
        let _ = pooled_take_length_prefixed_slices(
            input, &mut pos, token_set_count, "Weight-pool token set"
        )?;
        Ok(pooled_take_var_u32(input, &mut pos)? as usize)
    }

    pub fn from_packed_bytes(input: &[u8]) -> Result<Self, String> {
        if !input.starts_with(b"WPL3") {
            return Err("packed runtime Weight pool requires WPL3".to_owned());
        }
        let mut pos = 4usize;
        let token_set_count = pooled_take_var_u32(input, &mut pos)? as usize;
        let mut token_spans = Vec::<(u32, u32)>::with_capacity(token_set_count);
        for _ in 0..token_set_count {
            let len = pooled_take_var_u32(input, &mut pos)? as usize;
            let start = pos;
            let end = start.checked_add(len)
                .ok_or_else(|| "overflowing packed Weight-pool token-set length".to_owned())?;
            input.get(start..end)
                .ok_or_else(|| "truncated packed Weight-pool token set".to_owned())?;
            token_spans.push((
                u32::try_from(start).map_err(|_| "packed Weight-pool offset exceeds u32".to_owned())?,
                u32::try_from(len).map_err(|_| "packed Weight-pool length exceeds u32".to_owned())?,
            ));
            pos = end;
        }
        let weight_count = pooled_take_var_u32(input, &mut pos)? as usize;
        let mut weight_spans = Vec::<PackedRuntimeWeightSpan>::with_capacity(weight_count);
        for _ in 0..weight_count {
            let len = pooled_take_var_u32(input, &mut pos)? as usize;
            let start = pos;
            let end = start.checked_add(len)
                .ok_or_else(|| "overflowing packed Weight-pool weight length".to_owned())?;
            let body = input.get(start..end)
                .ok_or_else(|| "truncated packed Weight-pool weight".to_owned())?;
            let tag = *body.first()
                .ok_or_else(|| "truncated packed Weight-pool tag".to_owned())?;
            let (full, entry_count) = match tag {
                1 => {
                    if body.len() != 1 {
                        return Err("trailing bytes in packed full weight".to_owned());
                    }
                    (true, 0)
                }
                0 => {
                    let mut body_pos = 1usize;
                    let count = pooled_take_var_u32(body, &mut body_pos)?;
                    (false, count)
                }
                _ => return Err("invalid packed Weight-pool tag".to_owned()),
            };
            weight_spans.push(PackedRuntimeWeightSpan {
                body_start: u32::try_from(start)
                    .map_err(|_| "packed Weight-pool offset exceeds u32".to_owned())?,
                body_len: u32::try_from(len)
                    .map_err(|_| "packed Weight-pool length exceeds u32".to_owned())?,
                entry_count,
                full,
            });
            pos = end;
        }
        if pos != input.len() {
            return Err("trailing bytes in packed Weight pool".to_owned());
        }
        Ok(Self {
            wire: Arc::from(input.to_vec().into_boxed_slice()),
            token_spans: token_spans.into_boxed_slice(),
            weight_spans: weight_spans.into_boxed_slice(),
        })
    }

    #[inline]
    pub fn weight_count(&self) -> usize { self.weight_spans.len() }

    #[inline]
    pub fn packed_bytes(&self) -> &[u8] { self.wire.as_ref() }

    #[inline]
    pub fn weight(&self, id: u32) -> Option<PackedRuntimePoolWeightRef<'_>> {
        ((id as usize) < self.weight_spans.len()).then_some(PackedRuntimePoolWeightRef { pool: self, id })
    }

    #[inline]
    pub fn token_set(&self, id: u32) -> Option<PackedRuntimePoolTokenSetRef<'_>> {
        let &(start, len) = self.token_spans.get(id as usize)?;
        let start = start as usize;
        Some(PackedRuntimePoolTokenSetRef {
            id,
            body: self.wire.get(start..start + len as usize)?,
        })
    }
}

fn unpack_pooled_weights_v1(input: &[u8]) -> Result<Vec<Weight>, String> {
    let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
    let token_sets_started = profile.then(std::time::Instant::now);
    if !input.starts_with(b"WPL1") {
        return Err("invalid packed Weight-pool header".to_owned());
    }
    let mut pos = 4usize;
    let token_set_count = pooled_take_var_u32(input, &mut pos)? as usize;
    let mut token_sets = Vec::with_capacity(token_set_count);
    let mut token_range_count = 0usize;
    for _ in 0..token_set_count {
        let range_count = pooled_take_var_u32(input, &mut pos)? as usize;
        token_range_count += range_count;
        let mut ranges = Vec::with_capacity(range_count);
        let mut previous_end_plus_one = 0u64;
        for _ in 0..range_count {
            let gap = pooled_take_var_u64(input, &mut pos)?;
            let start64 = previous_end_plus_one
                .checked_add(gap)
                .ok_or_else(|| "overflowing packed token-set range".to_owned())?;
            let start = u32::try_from(start64)
                .map_err(|_| "overflowing packed token-set start".to_owned())?;
            let len = pooled_take_var_u32(input, &mut pos)?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| "overflowing packed token-set end".to_owned())?;
            ranges.push(start..=end);
            previous_end_plus_one = end as u64 + 1;
        }
        let tokens = RangeSetBlaze::from_sorted_disjoint(CheckSortedDisjoint::new(
            ranges.into_iter(),
        ));
        token_sets.push(shared_rangeset_artifact_local(tokens));
    }
    let token_sets_ms = token_sets_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let weights_started = profile.then(std::time::Instant::now);
    let weight_count = pooled_take_var_u32(input, &mut pos)? as usize;
    let mut weights = Vec::with_capacity(weight_count);
    let mut weight_entry_count = 0usize;
    for _ in 0..weight_count {
        let tag = *input
            .get(pos)
            .ok_or_else(|| "truncated packed Weight-pool tag".to_owned())?;
        pos += 1;
        if tag == 1 {
            weights.push(Weight::all());
            continue;
        }
        if tag != 0 {
            return Err("invalid packed Weight-pool tag".to_owned());
        }
        let entry_count = pooled_take_var_u32(input, &mut pos)? as usize;
        weight_entry_count += entry_count;
        if entry_count == 0 {
            weights.push(Weight::empty());
            continue;
        }
        let mut ranges = Vec::with_capacity(entry_count);
        let mut previous_end_plus_one = 0u64;
        for _ in 0..entry_count {
            let gap = pooled_take_var_u64(input, &mut pos)?;
            let start64 = previous_end_plus_one
                .checked_add(gap)
                .ok_or_else(|| "overflowing packed weight range".to_owned())?;
            let start = u32::try_from(start64)
                .map_err(|_| "overflowing packed weight start".to_owned())?;
            let len = pooled_take_var_u32(input, &mut pos)?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| "overflowing packed weight end".to_owned())?;
            let token_set_idx = pooled_take_var_u32(input, &mut pos)? as usize;
            let tokens = token_sets
                .get(token_set_idx)
                .cloned()
                .ok_or_else(|| "invalid packed Weight-pool token-set index".to_owned())?;
            ranges.push((start..=end, tokens));
            previous_end_plus_one = end as u64 + 1;
        }
        let map = WeightMap::from_sorted_disjoint_map(CheckSortedDisjointMap::new(
            ranges
                .iter()
                .map(|(range, tokens)| (range.clone(), tokens)),
        ));
        weights.push(finalize_weight_map_artifact_local(map));
    }
    if pos != input.len() {
        return Err("trailing bytes in packed Weight pool".to_owned());
    }
    if profile {
        eprintln!(
            "[glrmask/profile][pooled_weight_load] bytes={} token_sets={} token_ranges={} weights={} weight_entries={} token_sets_ms={:.3} weights_ms={:.3}",
            input.len(),
            token_set_count,
            token_range_count,
            weight_count,
            weight_entry_count,
            token_sets_ms,
            weights_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Ok(weights)
}

fn pooled_take_length_prefixed_slices<'a>(
    input: &'a [u8],
    pos: &mut usize,
    count: usize,
    label: &str,
) -> Result<Vec<&'a [u8]>, String> {
    let mut bodies = Vec::with_capacity(count);
    for _ in 0..count {
        let len = pooled_take_var_u32(input, pos)? as usize;
        let end = pos
            .checked_add(len)
            .ok_or_else(|| format!("overflowing packed {label} length"))?;
        let body = input
            .get(*pos..end)
            .ok_or_else(|| format!("truncated packed {label}"))?;
        bodies.push(body);
        *pos = end;
    }
    Ok(bodies)
}

fn decode_pooled_token_set_v2(body: &[u8]) -> Result<(SharedTokenSet, usize), String> {
    let mut pos = 0usize;
    let range_count = pooled_take_var_u32(body, &mut pos)? as usize;
    let mut ranges = Vec::with_capacity(range_count);
    let mut previous_end_plus_one = 0u64;
    for _ in 0..range_count {
        let gap = pooled_take_var_u64(body, &mut pos)?;
        let start64 = previous_end_plus_one
            .checked_add(gap)
            .ok_or_else(|| "overflowing packed token-set range".to_owned())?;
        let start = u32::try_from(start64)
            .map_err(|_| "overflowing packed token-set start".to_owned())?;
        let len = pooled_take_var_u32(body, &mut pos)?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| "overflowing packed token-set end".to_owned())?;
        ranges.push(start..=end);
        previous_end_plus_one = end as u64 + 1;
    }
    if pos != body.len() {
        return Err("trailing bytes in packed token set".to_owned());
    }
    let tokens = RangeSetBlaze::from_sorted_disjoint(CheckSortedDisjoint::new(ranges.into_iter()));
    Ok((shared_rangeset_artifact_local(tokens), range_count))
}

fn decode_pooled_weight_v3(
    body: &[u8],
    token_sets: &[SharedTokenSet],
) -> Result<(Weight, usize), String> {
    let mut pos = 0usize;
    let tag = *body
        .get(pos)
        .ok_or_else(|| "truncated packed Weight-pool tag".to_owned())?;
    pos += 1;
    if tag == 1 {
        if pos != body.len() {
            return Err("trailing bytes in packed full weight".to_owned());
        }
        return Ok((Weight::all(), 0));
    }
    if tag != 0 {
        return Err("invalid packed Weight-pool tag".to_owned());
    }
    let entry_count = pooled_take_var_u32(body, &mut pos)? as usize;
    if entry_count == 0 {
        if pos != body.len() {
            return Err("trailing bytes in packed empty weight".to_owned());
        }
        return Ok((Weight::empty(), 0));
    }
    let mut ranges = Vec::with_capacity(entry_count);
    let mut previous_end_plus_one = 0u64;
    for _ in 0..entry_count {
        let gap = pooled_take_var_u64(body, &mut pos)?;
        let start64 = previous_end_plus_one
            .checked_add(gap)
            .ok_or_else(|| "overflowing packed weight range".to_owned())?;
        let start = u32::try_from(start64)
            .map_err(|_| "overflowing packed weight start".to_owned())?;
        let len = pooled_take_var_u32(body, &mut pos)?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| "overflowing packed weight end".to_owned())?;
        let token_set_idx = pooled_take_var_u32(body, &mut pos)? as usize;
        let tokens = token_sets
            .get(token_set_idx)
            .cloned()
            .ok_or_else(|| "invalid packed Weight-pool token-set index".to_owned())?;
        ranges.push((start..=end, tokens));
        previous_end_plus_one = end as u64 + 1;
    }
    if pos != body.len() {
        return Err("trailing bytes in packed weight".to_owned());
    }
    let map = WeightMap::from_sorted_disjoint_map(CheckSortedDisjointMap::new(
        ranges.iter().map(|(range, tokens)| (range.clone(), tokens)),
    ));
    Ok((finalize_weight_map_artifact_local(map), entry_count))
}

fn unpack_pooled_weights_v3(input: &[u8]) -> Result<Vec<Weight>, String> {
    let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
    let token_sets_started = profile.then(std::time::Instant::now);
    let mut pos = 4usize;

    let token_set_count = pooled_take_var_u32(input, &mut pos)? as usize;
    let token_bodies = pooled_take_length_prefixed_slices(
        input,
        &mut pos,
        token_set_count,
        "Weight-pool token set",
    )?;
    let decoded_tokens = if token_set_count >= 256 && rayon::current_num_threads() > 1 {
        token_bodies
            .par_iter()
            .map(|body| decode_pooled_token_set_v2(body))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        token_bodies
            .iter()
            .map(|body| decode_pooled_token_set_v2(body))
            .collect::<Result<Vec<_>, _>>()?
    };
    let token_range_count = decoded_tokens.iter().map(|(_, count)| *count).sum::<usize>();
    let token_sets = decoded_tokens
        .into_iter()
        .map(|(tokens, _)| tokens)
        .collect::<Vec<_>>();
    let token_sets_ms = token_sets_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let weights_started = profile.then(std::time::Instant::now);
    let weight_count = pooled_take_var_u32(input, &mut pos)? as usize;
    let weight_bodies = pooled_take_length_prefixed_slices(
        input,
        &mut pos,
        weight_count,
        "Weight-pool weight",
    )?;
    if pos != input.len() {
        return Err("trailing bytes in packed Weight pool".to_owned());
    }
    let decoded_weights = if weight_count >= 256 && rayon::current_num_threads() > 1 {
        weight_bodies
            .par_iter()
            .map(|body| decode_pooled_weight_v3(body, &token_sets))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        weight_bodies
            .iter()
            .map(|body| decode_pooled_weight_v3(body, &token_sets))
            .collect::<Result<Vec<_>, _>>()?
    };
    let weight_entry_count = decoded_weights.iter().map(|(_, count)| *count).sum::<usize>();
    let weights = decoded_weights
        .into_iter()
        .map(|(weight, _)| weight)
        .collect::<Vec<_>>();
    if profile {
        eprintln!(
            "[glrmask/profile][pooled_weight_load] format=WPL3 bytes={} token_sets={} token_ranges={} weights={} weight_entries={} token_sets_ms={:.3} weights_ms={:.3}",
            input.len(),
            token_set_count,
            token_range_count,
            weight_count,
            weight_entry_count,
            token_sets_ms,
            weights_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Ok(weights)
}

fn decode_pooled_geometry_v2(body: &[u8]) -> Result<Vec<(u32, u32)>, String> {
    let mut pos = 0usize;
    let count = pooled_take_var_u32(body, &mut pos)? as usize;
    let mut ranges = Vec::with_capacity(count);
    let mut previous_end_plus_one = 0u64;
    for _ in 0..count {
        let gap = pooled_take_var_u64(body, &mut pos)?;
        let start64 = previous_end_plus_one
            .checked_add(gap)
            .ok_or_else(|| "overflowing packed weight geometry".to_owned())?;
        let start = u32::try_from(start64)
            .map_err(|_| "overflowing packed weight geometry start".to_owned())?;
        let len = pooled_take_var_u32(body, &mut pos)?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| "overflowing packed weight geometry end".to_owned())?;
        ranges.push((start, end));
        previous_end_plus_one = end as u64 + 1;
    }
    if pos != body.len() {
        return Err("trailing bytes in packed weight geometry".to_owned());
    }
    Ok(ranges)
}

fn decode_pooled_weight_v2(
    body: &[u8],
    token_sets: &[SharedTokenSet],
    geometries: &[Vec<(u32, u32)>],
) -> Result<(Weight, usize), String> {
    let mut pos = 0usize;
    let tag = *body
        .get(pos)
        .ok_or_else(|| "truncated packed Weight-pool tag".to_owned())?;
    pos += 1;
    if tag == 1 {
        if pos != body.len() {
            return Err("trailing bytes in packed full weight".to_owned());
        }
        return Ok((Weight::all(), 0));
    }
    if tag != 0 {
        return Err("invalid packed Weight-pool tag".to_owned());
    }
    let geometry_index = pooled_take_var_u32(body, &mut pos)? as usize;
    let geometry = geometries
        .get(geometry_index)
        .ok_or_else(|| "invalid packed Weight-pool geometry index".to_owned())?;
    if geometry.is_empty() {
        if pos != body.len() {
            return Err("trailing bytes in packed empty weight".to_owned());
        }
        return Ok((Weight::empty(), 0));
    }
    let mut ranges = Vec::with_capacity(geometry.len());
    for &(start, end) in geometry {
        let token_set_idx = pooled_take_var_u32(body, &mut pos)? as usize;
        let tokens = token_sets
            .get(token_set_idx)
            .cloned()
            .ok_or_else(|| "invalid packed Weight-pool token-set index".to_owned())?;
        ranges.push((start..=end, tokens));
    }
    if pos != body.len() {
        return Err("trailing bytes in packed weight".to_owned());
    }
    let map = WeightMap::from_sorted_disjoint_map(CheckSortedDisjointMap::new(
        ranges.iter().map(|(range, tokens)| (range.clone(), tokens)),
    ));
    Ok((finalize_weight_map_artifact_local(map), geometry.len()))
}

fn unpack_pooled_weights_v2(input: &[u8]) -> Result<Vec<Weight>, String> {
    let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
    let token_sets_started = profile.then(std::time::Instant::now);
    let mut pos = 4usize;

    let token_set_count = pooled_take_var_u32(input, &mut pos)? as usize;
    let token_bodies = pooled_take_length_prefixed_slices(
        input,
        &mut pos,
        token_set_count,
        "Weight-pool token set",
    )?;
    let decoded_tokens = if token_set_count >= 256 && rayon::current_num_threads() > 1 {
        token_bodies
            .par_iter()
            .map(|body| decode_pooled_token_set_v2(body))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        token_bodies
            .iter()
            .map(|body| decode_pooled_token_set_v2(body))
            .collect::<Result<Vec<_>, _>>()?
    };
    let token_range_count = decoded_tokens.iter().map(|(_, count)| *count).sum::<usize>();
    let token_sets = decoded_tokens
        .into_iter()
        .map(|(tokens, _)| tokens)
        .collect::<Vec<_>>();
    let token_sets_ms = token_sets_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let geometry_count = pooled_take_var_u32(input, &mut pos)? as usize;
    let geometry_bodies = pooled_take_length_prefixed_slices(
        input,
        &mut pos,
        geometry_count,
        "Weight-pool geometry",
    )?;
    let geometries = if geometry_count >= 256 && rayon::current_num_threads() > 1 {
        geometry_bodies
            .par_iter()
            .map(|body| decode_pooled_geometry_v2(body))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        geometry_bodies
            .iter()
            .map(|body| decode_pooled_geometry_v2(body))
            .collect::<Result<Vec<_>, _>>()?
    };

    let weights_started = profile.then(std::time::Instant::now);
    let weight_count = pooled_take_var_u32(input, &mut pos)? as usize;
    let weight_bodies = pooled_take_length_prefixed_slices(
        input,
        &mut pos,
        weight_count,
        "Weight-pool weight",
    )?;
    if pos != input.len() {
        return Err("trailing bytes in packed Weight pool".to_owned());
    }
    let decoded_weights = if weight_count >= 256 && rayon::current_num_threads() > 1 {
        weight_bodies
            .par_iter()
            .map(|body| decode_pooled_weight_v2(body, &token_sets, &geometries))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        weight_bodies
            .iter()
            .map(|body| decode_pooled_weight_v2(body, &token_sets, &geometries))
            .collect::<Result<Vec<_>, _>>()?
    };
    let weight_entry_count = decoded_weights.iter().map(|(_, count)| *count).sum::<usize>();
    let weights = decoded_weights
        .into_iter()
        .map(|(weight, _)| weight)
        .collect::<Vec<_>>();
    if profile {
        eprintln!(
            "[glrmask/profile][pooled_weight_load] format=WPL2 bytes={} token_sets={} token_ranges={} geometries={} weights={} weight_entries={} token_sets_ms={:.3} weights_ms={:.3}",
            input.len(),
            token_set_count,
            token_range_count,
            geometry_count,
            weight_count,
            weight_entry_count,
            token_sets_ms,
            weights_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Ok(weights)
}

impl Serialize for Weight {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if pooled_weight_serde_encode_enabled() {
            let index = pooled_weight_serde_encode_index(self).ok_or_else(|| {
                <S::Error as serde::ser::Error>::custom(
                    "Weight missing from active pooled serialization context",
                )
            })?;
            return index.serialize(serializer);
        }
        self.to_serde().serialize(serializer)
    }
}

fn weight_from_serde_entries(entries: Vec<WeightSerdeEntry>) -> Weight {
    let mut ranges = Vec::with_capacity(entries.len());
    for entry in entries {
        // WeightSerde is emitted from RangeSetBlaze::ranges(), so these ranges
        // are already sorted and disjoint.  Avoid the generic FromIterator
        // normalization path on every deserialized token set.
        let tokens = RangeSetBlaze::from_sorted_disjoint(CheckSortedDisjoint::new(
            entry.tokens.into_iter().map(|token| token[0]..=token[1]),
        ));
        if tokens.is_empty() {
            continue;
        }
        ranges.push((
            entry.tsid[0]..=entry.tsid[1],
            shared_rangeset(tokens),
        ));
    }
    if ranges.is_empty() {
        return Weight::empty();
    }
    // WeightSerde is emitted directly from WeightMap::range_values(), so the
    // outer TSID ranges are sorted and disjoint too. Avoid incrementally
    // rebuilding and re-normalizing the RangeMapBlaze on load.
    let map = WeightMap::from_sorted_disjoint_map(CheckSortedDisjointMap::new(
        ranges
            .iter()
            .map(|(range, tokens)| (range.clone(), tokens)),
    ));
    finalize_weight_map(map)
}

impl<'de> Deserialize<'de> for Weight {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if pooled_weight_serde_decode_enabled() {
            let index = u32::deserialize(deserializer)?;
            if POOLED_WEIGHT_SERDE_DEFER_COUNT.with(|slot| slot.get().is_some()) {
                return pooled_weight_serde_deferred_decode_index(index)
                    .ok_or_else(|| serde::de::Error::custom("invalid deferred pooled Weight index"));
            }
            return pooled_weight_serde_decode_index(index)
                .ok_or_else(|| serde::de::Error::custom("invalid pooled Weight index"));
        }
        let serde_weight = WeightSerde::deserialize(deserializer)?;
        if serde_weight.all {
            return Ok(Self::all());
        }
        Ok(weight_from_serde_entries(serde_weight.entries))
    }
}

#[cfg(test)]
mod packed_reserialization_tests {
    use super::*;

    #[test]
    fn packed_runtime_counts_match_decoded_weights_and_token_sets() {
        let tokens = RangeSetBlaze::from_iter([0, 2, 3, 5, 17, u32::MAX]);
        let weights = [
            Weight::empty(),
            Weight::all(),
            Weight::from_uniform(7..=13, tokens.clone()),
            Weight::from_per_tsid_token_sets([
                (0, tokens.clone()),
                (2, RangeSetBlaze::from_iter([1, 8])),
                (u32::MAX, tokens),
            ]),
        ];
        let wire = pack_pooled_weights(&weights);
        let pool = PackedRuntimeWeightPool::from_packed_bytes(&wire).unwrap();
        for (id, expected) in weights.iter().enumerate() {
            let packed = pool.weight(id as u32).unwrap();
            assert_eq!(packed.entry_count(), packed.entries().len());
            let expected_entries = if expected.is_full() {
                0
            } else {
                expected.0.range_values_len()
            };
            assert_eq!(packed.entry_count(), expected_entries);
            for (_, tokens) in packed.entries() {
                let mut actual_ranges = 0;
                tokens.for_each_range(|_, _| actual_ranges += 1);
                assert_eq!(tokens.range_count(), actual_ranges);
                assert_eq!(tokens.is_empty(), actual_ranges == 0);
            }
        }
    }

    #[test]
    fn packed_runtime_count_queries_do_not_allocate_from_declared_counts() {
        // A length-framed but truncated payload can advertise a huge count.
        // The cache admission query must remain constant-space; callers can
        // decline it without invoking the allocating entries() decoder.
        let mut wire = b"WPL3".to_vec();
        pooled_put_var_u32(&mut wire, 1);
        let mut token_body = Vec::new();
        pooled_put_var_u32(&mut token_body, u32::MAX);
        pooled_put_var_u32(&mut wire, token_body.len() as u32);
        wire.extend_from_slice(&token_body);
        pooled_put_var_u32(&mut wire, 1);
        let mut weight_body = vec![0];
        pooled_put_var_u32(&mut weight_body, u32::MAX);
        pooled_put_var_u32(&mut wire, weight_body.len() as u32);
        wire.extend_from_slice(&weight_body);

        let pool = PackedRuntimeWeightPool::from_packed_bytes(&wire).unwrap();
        assert_eq!(pool.token_set(0).unwrap().range_count(), u32::MAX as usize);
        assert_eq!(pool.weight(0).unwrap().entry_count(), u32::MAX as usize);
    }

    #[test]
    fn packed_reserialization_preserves_distinct_ids_for_empty_placeholders() {
        let placeholders = vec![Weight::empty(), Weight::empty(), Weight::empty()];
        begin_pooled_weight_serde_encode_ids(vec![2, 0, 1]);
        let mut bytes = Vec::new();
        let result = bincode::serialize_into(&mut bytes, &placeholders);
        end_pooled_weight_serde_encode();
        result.unwrap();
        assert_eq!(bincode::deserialize::<Vec<u32>>(&bytes).unwrap(), vec![2, 0, 1]);
        assert!(!pooled_weight_serde_encode_enabled());
    }

    #[test]
    fn packed_reserialization_rejects_missing_id_instead_of_reusing_empty_weight() {
        begin_pooled_weight_serde_encode_ids(vec![0]);
        let mut bytes = Vec::new();
        let result = bincode::serialize_into(&mut bytes, &[Weight::empty(), Weight::empty()]);
        end_pooled_weight_serde_encode();
        assert!(result.is_err());
        assert!(!pooled_weight_serde_encode_enabled());
    }
}
