//! L1 terminal DWA: direct 2-state construction for terminals with max path
//! length ≤ 1.

use crate::automata::lexer::Lexer;
pub mod implementations;
pub mod max_length;

use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};

/// Exact first-byte target profiles computed for L1 state equivalence.
///
/// The L1 terminal-DWA builder needs the same whole-token walks *and* the same
/// active-terminal signature at each tokenizer state. Reusing both avoids a
/// second full tokenizer-state scan after exact equivalence has already proved
/// the signatures. Profile signature IDs reserve zero for the empty signature;
/// the direct DWA builder numbers non-empty signatures from zero, so the cached
/// direct-builder view stores that shifted numbering explicitly.
type L1WalkProfile = Arc<[(u32, Arc<[(u32, u32)]>)]>;

/// Dense per-worker assembly amortizes its scratch/output reduction only for
/// very large exact L1 quotients. Below this size the established per-state
/// allocation path is cheaper.
const LARGE_EXACT_L1_ASSEMBLY_MIN_STATES: usize = 50_000;
const PARALLEL_L1_GROUP_ASSEMBLY_MIN_VISITS: usize = 20_000;

fn freeze_l1_walk_profile(runs: &[(u32, u32, u32)]) -> L1WalkProfile {
    let mut grouped = Vec::<(u32, Vec<(u32, u32)>)>::new();
    // Most exact target profiles touch only a handful of terminal signatures.
    // Avoid allocating and hashing a map for those cases, but promote to the
    // existing exact lookup strategy once the linear probe stops being cheap.
    const LINEAR_GROUP_LIMIT: usize = 8;
    let mut positions: Option<FxHashMap<u32, usize>> = None;
    for &(signature_id, start, end) in runs {
        if signature_id == 0 {
            continue;
        }
        let direct_signature_id = signature_id - 1;
        let position = if let Some(positions) = positions.as_mut() {
            if let Some(&position) = positions.get(&direct_signature_id) {
                position
            } else {
                let position = grouped.len();
                positions.insert(direct_signature_id, position);
                grouped.push((direct_signature_id, Vec::new()));
                position
            }
        } else if let Some(position) = grouped
            .iter()
            .position(|(signature, _)| *signature == direct_signature_id)
        {
            position
        } else {
            let position = grouped.len();
            grouped.push((direct_signature_id, Vec::new()));
            if grouped.len() == LINEAR_GROUP_LIMIT {
                let mut indexed = FxHashMap::default();
                for (position, (signature, _)) in grouped.iter().enumerate() {
                    indexed.insert(*signature, position);
                }
                positions = Some(indexed);
            }
            position
        };
        grouped[position].1.push((start, end));
    }
    Arc::from(
        grouped
            .into_iter()
            .map(|(signature_id, ranges)| (signature_id, Arc::from(ranges)))
            .collect::<Vec<_>>(),
    )
}

fn freeze_l1_walk_profile_from_direct(profile: Vec<(u32, Vec<(u32, u32)>)>) -> L1WalkProfile {
    Arc::from(
        profile
            .into_iter()
            .map(|(signature_id, ranges)| (signature_id, Arc::from(ranges)))
            .collect::<Vec<_>>(),
    )
}

fn index_l1_walk_profile<'a>(
    results: &'a L1WalkProfile,
) -> Vec<(usize, &'a [(u32, u32)], u64, usize)> {
    results
        .iter()
        .map(|(sig_id, ranges)| {
            let ranges = ranges.as_ref();
            let mut h: u64 = 0;
            for &(start, end) in ranges {
                h = h.wrapping_add(range_hash_val(start, end));
            }
            (
                *sig_id as usize,
                ranges,
                (ranges.len() as u64).wrapping_add(h),
                ranges.len(),
            )
        })
        .collect()
}

#[derive(Debug)]
struct L1ExactProfileReuse {
    target_to_profile_id: Vec<((u8, u32), u32)>,
    walk_profiles_by_id: Vec<L1WalkProfile>,
    /// Exact-profile representative aligned with each pre-isolation L1 TSID
    /// for deterministic epsilon dispatch.
    /// Structured epsilon dispatch may later split the synthetic initial state
    /// into a new TSID, but every deterministic component state still maps
    /// through one of these proved scalar profile representatives.
    profile_representatives_by_internal: Arc<[u32]>,
    /// Exact whole-token behavior for every state retained as an L1 id-map
    /// representative. Entries are aligned with the non-empty first-byte
    /// buckets and use zero for an empty suffix profile.
    representative_profile_ids: FxHashMap<u32, Arc<[u32]>>,
    direct_terminal_signatures: Arc<[Vec<u32>]>,
    direct_state_to_terminal_signature: Arc<[u32]>,
}

impl L1ExactProfileReuse {
    fn materialize_walk_cache(&self) -> FxHashMap<(u8, u32), L1WalkProfile> {
        let profiling = compile_profile_enabled();
        let total_started_at = profiling.then(Instant::now);
        let mut cache = FxHashMap::default();
        for &(target, profile_id) in &self.target_to_profile_id {
            if profile_id != 0 {
                cache.insert(target, Arc::clone(&self.walk_profiles_by_id[profile_id as usize]));
            }
        }
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][l1_exact_profile_materialize] profiles={} targets={} profile_build_ms=0.000 target_clone_ms={:.3} total_ms={:.3}",
                self.walk_profiles_by_id.len(),
                self.target_to_profile_id.len(),
                total_started_at.elapsed().as_secs_f64() * 1000.0,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        cache
    }
}

/// Ranges key with pre-computed hash for O(1) HashMap lookups.
/// The hash is computed in the parallel traversal phase so the sequential
/// interning loop avoids re-hashing large range vectors.
#[derive(Clone)]
struct PreHashedRanges {
    hash: u64,
    ranges: Vec<(u32, u32)>,
}

#[derive(Debug)]
pub struct L1IdentityVocabOrder {
    token_ids_sorted: Arc<[u32]>,
    token_entries_sorted: Arc<[(u32, Arc<[u8]>)]>,
    original_to_internal: Arc<[u32]>,
    token_buckets: L1SortedTokenBuckets,
}

impl crate::vocab::VocabDerivedArtifact for L1IdentityVocabOrder {}
impl crate::vocab::VocabDerivedArtifact
    for super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie
{
}

fn l1_token_bounded_analysis_trie(
    vocab: &Vocab,
) -> Option<Arc<super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie>> {
    if vocab.len() > L1_GENERIC_NFA_TOKEN_BOUNDED_MAX_VOCAB {
        return None;
    }
    if let Some(cached) = vocab.vocab_derived_cache_get::<
        super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie,
    >() {
        return Some(cached);
    }
    let order = l1_identity_vocab_order(vocab);
    let tokens = order
        .token_entries_sorted
        .iter()
        .map(|(_, bytes)| bytes.as_ref())
        .collect::<Vec<_>>();
    let trie = Arc::new(
        super::l2p::equivalence_analysis::state_equivalence::nfa::build_token_bounded_analysis_trie_sorted(
            &tokens,
        ),
    );
    vocab.vocab_derived_cache_set(Arc::clone(&trie));
    Some(trie)
}

pub fn prepare_l1_token_bounded_analysis_trie(vocab: &Vocab) {
    let _ = l1_token_bounded_analysis_trie(vocab);
}

pub fn prepared_l1_token_bounded_analysis_trie(
    vocab: &Vocab,
) -> Option<Arc<super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie>> {
    l1_token_bounded_analysis_trie(vocab)
}

/// Reuse a token-bounded analysis trie only when some earlier vocabulary
/// preparation or branch has already paid to build it.
///
/// Routing callers frequently need only a tiny token subset. Forcing the
/// whole-partition trie on a cold path can cost much more than the exact
/// subset analysis it is intended to accelerate.
pub(crate) fn cached_l1_token_bounded_analysis_trie(
    vocab: &Vocab,
) -> Option<Arc<super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie>> {
    vocab.vocab_derived_cache_get::<
        super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie,
    >()
}

fn l1_identity_vocab_order(vocab: &Vocab) -> Arc<L1IdentityVocabOrder> {
    if let Some(cached) = vocab.vocab_derived_cache_get::<L1IdentityVocabOrder>() {
        return cached;
    }

    let profiling = compile_profile_enabled();
    let total_started_at = profiling.then(Instant::now);
    let collect_started_at = profiling.then(Instant::now);
    let mut token_entries_sorted: Vec<(u32, Arc<[u8]>)> = vocab
        .entries_map()
        .iter()
        .map(|(&id, bytes)| (id, Arc::<[u8]>::from(bytes.clone().into_boxed_slice())))
        .collect();
    let collect_ms = collect_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let sort_started_at = profiling.then(Instant::now);
    token_entries_sorted.sort_unstable_by(|(left_id, left_bytes), (right_id, right_bytes)| {
        left_bytes
            .first()
            .cmp(&right_bytes.first())
            .then(left_bytes.cmp(right_bytes))
            .then(left_id.cmp(right_id))
    });
    let sort_ms = sort_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let map_started_at = profiling.then(Instant::now);
    let mut token_original_to_internal = vec![u32::MAX; vocab.max_token_id() as usize + 1];
    let token_ids_sorted: Vec<u32> = token_entries_sorted
        .iter()
        .enumerate()
        .map(|(internal_id, (original_id, _))| {
            token_original_to_internal[*original_id as usize] = internal_id as u32;
            *original_id
        })
        .collect();
    let map_ms = map_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let buckets_started_at = profiling.then(Instant::now);
    let token_buckets = build_l1_sorted_token_buckets(&token_entries_sorted);
    let eager_subset_tries = std::env::var("GLRMASK_EAGER_SUBSET_PACKED_TRIES")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(false);
    if eager_subset_tries {
        for first_byte in 0..256 {
            let token_ids = &token_buckets.token_indices_by_first_byte[first_byte];
            if token_ids.is_empty() {
                continue;
            }
            token_buckets.packed_suffix_tries_by_first_byte[first_byte].get_or_init(|| {
                Arc::new(L1PackedSuffixTrie::build(
                    &token_entries_sorted,
                    token_ids,
                    &token_buckets.suffix_lcps_by_first_byte[first_byte],
                ))
            });
        }
    }
    let buckets_ms = buckets_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let order = Arc::new(L1IdentityVocabOrder {
        token_ids_sorted: token_ids_sorted.into(),
        token_entries_sorted: token_entries_sorted.into(),
        original_to_internal: token_original_to_internal.into(),
        token_buckets,
    });
    vocab.vocab_derived_cache_set(Arc::clone(&order));
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][l1_identity_order] tokens={} collect_ms={:.3} sort_ms={:.3} map_ms={:.3} buckets_ms={:.3} total_ms={:.3}",
            vocab.len(),
            collect_ms,
            sort_ms,
            map_ms,
            buckets_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    order
}

pub fn prepare_l1_identity_vocab_order(vocab: &Vocab) {
    let order = l1_identity_vocab_order(vocab);
    for first_byte in 0..256 {
        let token_ids = &order.token_buckets.token_indices_by_first_byte[first_byte];
        if token_ids.is_empty() {
            continue;
        }
        order.token_buckets.packed_suffix_tries_by_first_byte[first_byte].get_or_init(|| {
            Arc::new(L1PackedSuffixTrie::build(
                order.token_entries_sorted.as_ref(),
                token_ids,
                &order.token_buckets.suffix_lcps_by_first_byte[first_byte],
            ))
        });
    }
}

pub fn prepared_l1_identity_vocab_order(vocab: &Vocab) -> Arc<L1IdentityVocabOrder> {
    l1_identity_vocab_order(vocab)
}

/// Reuse an already-prepared parent order without forcing a full-vocabulary
/// sort solely to accelerate a much smaller subset build.
pub(crate) fn cached_l1_identity_vocab_order(vocab: &Vocab) -> Option<Arc<L1IdentityVocabOrder>> {
    vocab.vocab_derived_cache_get::<L1IdentityVocabOrder>()
}

/// Resolve exact token-byte groups from the already-prepared identity order.
///
/// This never forces preparation: tiny grammar-specific fast paths may use it
/// when CFA/runtime vocabulary preparation has already paid for the byte order,
/// but cold standalone callers should remain free to choose a cheaper fallback.
pub fn cached_l1_exact_token_groups(
    vocab: &Vocab,
    byte_strings: &[Vec<u8>],
) -> Option<Vec<Vec<u32>>> {
    let order = vocab.vocab_derived_cache_get::<L1IdentityVocabOrder>()?;
    let entries = order.token_entries_sorted.as_ref();
    Some(
        byte_strings
            .iter()
            .map(|bytes| {
                let cmp = |entry: &(u32, Arc<[u8]>)| {
                    entry
                        .1
                        .first()
                        .cmp(&bytes.first())
                        .then_with(|| entry.1.as_ref().cmp(bytes.as_slice()))
                };
                let start = entries.partition_point(|entry| cmp(entry).is_lt());
                let end = entries.partition_point(|entry| !cmp(entry).is_gt());
                entries[start..end]
                    .iter()
                    .map(|(token_id, _)| *token_id)
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
}

/// Reuse the parent L1 byte order for an L2P boundary subset when it is already cached.
pub(crate) fn inherit_l2p_lexical_entry_order(
    parent_vocab: &Vocab,
    child_vocab: &Vocab,
    child_token_ids_in_entry_order: &[u32],
) {
    let Some(parent_order) = parent_vocab.vocab_derived_cache_get::<L1IdentityVocabOrder>() else {
        return;
    };
    debug_assert_eq!(child_token_ids_in_entry_order.len(), child_vocab.len());
    let mut entry_indices = (0..child_token_ids_in_entry_order.len() as u32).collect::<Vec<_>>();
    entry_indices.sort_unstable_by_key(|&entry_index| {
        let token_id = child_token_ids_in_entry_order[entry_index as usize] as usize;
        parent_order.original_to_internal[token_id]
    });
    child_vocab.vocab_derived_cache_set(Arc::new(
        super::l2p::equivalence_analysis::shared::VocabLexicalEntryOrder {
            entry_indices: entry_indices.into(),
        },
    ));
}
pub fn prepare_l1_finite_vocab_projection(vocab: &Vocab) {
    implementations::prepare_finite_vocab_projection(vocab);
}

fn derive_l1_identity_vocab_order_from_parent(
    parent: &L1IdentityVocabOrder,
    subset_vocab: &Vocab,
) -> Arc<L1IdentityVocabOrder> {
    let profiling = compile_profile_enabled();
    let total_started_at = profiling.then(Instant::now);
    let membership_started_at = profiling.then(Instant::now);
    let mut included = vec![false; parent.original_to_internal.len()];
    for &token_id in subset_vocab.entries_map().keys() {
        let token_id = token_id as usize;
        assert!(token_id < included.len(), "subset token missing from parent order");
        included[token_id] = true;
    }
    let membership_ms = membership_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let filter_started_at = profiling.then(Instant::now);
    let mut token_entries_sorted = Vec::with_capacity(subset_vocab.len());
    for (token_id, bytes) in parent.token_entries_sorted.iter() {
        if included[*token_id as usize] {
            token_entries_sorted.push((*token_id, Arc::clone(bytes)));
        }
    }
    assert_eq!(token_entries_sorted.len(), subset_vocab.len());
    let filter_ms = filter_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let map_started_at = profiling.then(Instant::now);
    let mut original_to_internal = vec![u32::MAX; subset_vocab.max_token_id() as usize + 1];
    let token_ids_sorted = token_entries_sorted
        .iter()
        .enumerate()
        .map(|(internal_id, (original_id, _))| {
            original_to_internal[*original_id as usize] = internal_id as u32;
            *original_id
        })
        .collect::<Vec<_>>();
    let map_ms = map_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let buckets_started_at = profiling.then(Instant::now);
    let token_buckets = build_l1_sorted_token_buckets(&token_entries_sorted);
    let buckets_ms = buckets_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][l1_subset_order] tokens={} membership_ms={:.3} filter_ms={:.3} map_ms={:.3} buckets_ms={:.3} total_ms={:.3}",
            subset_vocab.len(), membership_ms, filter_ms, map_ms, buckets_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Arc::new(L1IdentityVocabOrder {
        token_ids_sorted: token_ids_sorted.into(),
        token_entries_sorted: token_entries_sorted.into(),
        original_to_internal: original_to_internal.into(),
        token_buckets,
    })
}

fn skip_max_length_for_partition(partition_label: &str) -> bool {
    if partition_label == "p5" {
        return true;
    }
    static SKIPPED_PARTITIONS: OnceLock<Vec<String>> = OnceLock::new();
    SKIPPED_PARTITIONS
        .get_or_init(|| {
            std::env::var("GLRMASK_SKIP_MAX_LENGTH_PARTITIONS")
                .ok()
                .map(|value| {
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|label| !label.is_empty())
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default()
        })
        .iter()
        .any(|label| label == partition_label)
}

fn skip_l1_max_length_for_partition(partition_label: &str) -> bool {
    if matches!(partition_label, "p1" | "p4" | "p6") {
        return true;
    }
    static SKIPPED_L1_PARTITIONS: OnceLock<Vec<String>> = OnceLock::new();
    SKIPPED_L1_PARTITIONS
        .get_or_init(|| {
            std::env::var("GLRMASK_SKIP_L1_MAX_LENGTH_PARTITIONS")
                .ok()
                .map(|value| {
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|label| !label.is_empty())
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default()
        })
        .iter()
        .any(|label| label == partition_label)
}

fn l1_max_length_min_states() -> usize {
    static MIN_STATES: OnceLock<usize> = OnceLock::new();
    *MIN_STATES.get_or_init(|| {
        std::env::var("GLRMASK_L1_MAX_LENGTH_MIN_STATES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(128)
    })
}

/// Above this many unprojected states, the bounded prepass can cost more than
/// the exact token-signature pass it is meant to shrink. Skipping it remains
/// exact because L1 then classifies every candidate state directly.
fn l1_max_length_large_state_skip_threshold() -> usize {
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("GLRMASK_L1_MAX_LENGTH_LARGE_STATE_SKIP_THRESHOLD")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(16_384)
    })
}

#[inline]
fn should_skip_max_length_for_partition(
    partition_label: &str,
    initial_state_count: usize,
    projected_by_global: bool,
) -> bool {
    skip_max_length_for_partition(partition_label)
        || skip_l1_max_length_for_partition(partition_label)
        || initial_state_count < l1_max_length_min_states()
        || initial_state_count >= l1_max_length_large_state_skip_threshold()
        || (projected_by_global && initial_state_count <= 8192)
}

fn fast_projected_l1_id_map_override() -> Option<bool> {
    static OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("GLRMASK_L1_FAST_PROJECTED_ID_MAP")
            .ok()
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty()
                    || !matches!(
                        trimmed.to_ascii_lowercase().as_str(),
                        "0" | "false" | "no" | "off"
                    )
            })
    })
}

fn fast_projected_l1_id_map_enabled() -> bool {
    fast_projected_l1_id_map_override().unwrap_or(true)
}

pub fn fast_projected_l1_id_map_max_tsids() -> usize {
    static MAX_TSID: OnceLock<usize> = OnceLock::new();
    *MAX_TSID.get_or_init(|| {
        std::env::var("GLRMASK_L1_FAST_PROJECTED_ID_MAP_MAX_TSIDS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(16_384)
    })
}

#[inline]
fn prefer_exact_profiles_over_projected_l1(
    partition_label: &str,
    vocab_count: usize,
    projected_states: usize,
) -> bool {
    partition_label == "p2" && vocab_count >= 50_000 && projected_states >= 1_024
}

#[inline]
fn initial_state_map_contains_original(
    initial_state_map: Option<&ManyToOneIdMap>,
    original: u32,
) -> bool {
    initial_state_map.is_none_or(|map| {
        map.original_to_internal
            .get(original as usize)
            .copied()
            .is_some_and(|internal| internal != u32::MAX)
    })
}

#[inline]
fn initial_state_map_is_sparse(initial_state_map: Option<&ManyToOneIdMap>) -> bool {
    initial_state_map.is_some_and(|map| {
        map.original_to_internal
            .iter()
            .any(|&internal| internal == u32::MAX)
    })
}

#[inline]
fn should_use_fast_projected_l1_id_map(
    partition_label: &str,
    vocab_count: usize,
    initial_state_map: Option<&ManyToOneIdMap>,
    num_dfa_states: usize,
) -> bool {
    if !fast_projected_l1_id_map_enabled() {
        return false;
    }
    let Some(map) = initial_state_map else {
        return false;
    };
    let projected_states = map.num_internal_ids() as usize;
    let structurally_eligible = map.original_to_internal.len() == num_dfa_states
        && projected_states < num_dfa_states
        && projected_states <= fast_projected_l1_id_map_max_tsids();
    if !structurally_eligible {
        return false;
    }

    // A large p2 vocabulary is the measured crossover where retaining the
    // projected classes merely defers exact whole-token equivalence to terminal
    // materialisation. Running the exact classifier here produces reusable walk
    // profiles and avoids a second ~40 ms per-state assembly pass. An explicit
    // true environment override still forces the projected path for diagnostics.
    if fast_projected_l1_id_map_override().is_none()
        && prefer_exact_profiles_over_projected_l1(
            partition_label,
            vocab_count,
            projected_states,
        )
    {
        return false;
    }
    true
}

/// Hash contribution of a single (start, end) range.
#[inline(always)]
fn range_hash_val(s: u32, e: u32) -> u64 {
    let v = (s as u64) | ((e as u64) << 32);
    v.wrapping_mul(0x517cc1b727220a95)
}

impl PreHashedRanges {
    fn new(ranges: Vec<(u32, u32)>) -> Self {
        let mut h: u64 = 0;
        for &(s, e) in &ranges {
            h = h.wrapping_add(range_hash_val(s, e));
        }
        let hash = (ranges.len() as u64).wrapping_add(h);
        Self { hash, ranges }
    }
}

impl PartialEq for PreHashedRanges {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.ranges == other.ranges
    }
}

impl Eq for PreHashedRanges {}

impl Hash for PreHashedRanges {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

/// Lazy range representation: stores references to walk_cache range slices
/// instead of copying. Hash is computed over all referenced ranges using the
/// same commutative scheme as PreHashedRanges, so it matches the hash of the
/// fully-merged range set exactly when no inter-ref adjacency merges occur.
/// For interning, equality is checked via ref identity (ptr + len) — safe
/// because each walk_cache entry's Vec has a unique address and different
/// entry sets always produce different token ID sets.
#[derive(Clone)]
struct LazyRanges<'a> {
    refs: Vec<&'a [(u32, u32)]>,
    hash: u64,
    total_len: usize,
}

impl<'a> LazyRanges<'a> {
    fn new(refs: Vec<&'a [(u32, u32)]>) -> Self {
        // Compute hash over MERGED ranges by streaming through refs.
        // This produces the same hash as hashing the fully materialized
        // merged output, enabling correct interning across different
        // contributing entry sets that merge to the same result.
        let mut h: u64 = 0;
        let mut total_len: usize = 0;
        let mut merged_count: usize = 0;
        let mut current: Option<(u32, u32)> = None;

        for &slice in &refs {
            total_len += slice.len();
            for &(s, e) in slice {
                if let Some((cs, ref mut ce)) = current {
                    if s <= ce.saturating_add(1) {
                        *ce = (*ce).max(e);
                    } else {
                        h = h.wrapping_add(range_hash_val(cs, *ce));
                        merged_count += 1;
                        current = Some((s, e));
                    }
                } else {
                    current = Some((s, e));
                }
            }
        }
        if let Some((s, e)) = current {
            h = h.wrapping_add(range_hash_val(s, e));
            merged_count += 1;
        }
        let hash = (merged_count as u64).wrapping_add(h);
        Self {
            refs,
            hash,
            total_len,
        }
    }

    /// Materialize into merged ranges.
    fn materialize(&self) -> Vec<(u32, u32)> {
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(self.total_len);
        for &slice in &self.refs {
            if let Some((&first, rest)) = slice.split_first() {
                if let Some(last) = merged.last_mut() {
                    if first.0 <= last.1.saturating_add(1) {
                        last.1 = last.1.max(first.1);
                    } else {
                        merged.push(first);
                    }
                } else {
                    merged.push(first);
                }
                merged.extend_from_slice(rest);
            }
        }
        merged
    }
}

impl<'a> PartialEq for LazyRanges<'a> {
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash {
            return false;
        }
        // First try fast path: identical ref lists always produce identical output.
        if self.refs.len() == other.refs.len()
            && self
                .refs
                .iter()
                .zip(other.refs.iter())
                .all(|(&a, &b)| std::ptr::eq(a.as_ptr(), b.as_ptr()) && a.len() == b.len())
        {
            return true;
        }
        // Slow path: streaming merged-range comparison.
        self.materialize() == other.materialize()
    }
}

impl<'a> Eq for LazyRanges<'a> {}

fn append_l1_profile_entry<'a>(
    touched_positions: &mut FxHashMap<usize, usize>,
    touched_signatures: &mut Vec<(usize, Vec<&'a [(u32, u32)]>, u64, usize)>,
    sig_idx: usize,
    ranges: &'a [(u32, u32)],
    entry_hash: u64,
    entry_range_count: usize,
) {
    let position = if let Some(&position) = touched_positions.get(&sig_idx) {
        position
    } else {
        let position = touched_signatures.len();
        touched_positions.insert(sig_idx, position);
        touched_signatures.push((sig_idx, Vec::new(), 0, 0));
        position
    };
    let (_, refs, hash_accum, len_accum) = &mut touched_signatures[position];
    refs.push(ranges);
    *hash_accum = hash_accum.wrapping_add(entry_hash);
    *len_accum += entry_range_count;
}

/// Serial exact-profile collection visits many TSIDs while drawing from the
/// same compact terminal-signature universe. Reuse a dense signature-to-slot
/// scratch table instead of allocating and hashing a fresh sparse map per TSID.
/// `touched_signature_ids` records precisely the entries that must be cleared
/// before the next TSID; the produced range vectors still move into `LazyRanges`.
#[inline]
fn append_l1_profile_entry_dense<'a>(
    touched_positions: &mut [usize],
    touched_signature_ids: &mut Vec<usize>,
    touched_signatures: &mut Vec<(usize, Vec<&'a [(u32, u32)]>, u64, usize)>,
    sig_idx: usize,
    ranges: &'a [(u32, u32)],
    entry_hash: u64,
    entry_range_count: usize,
) {
    let position = touched_positions[sig_idx];
    let position = if position == usize::MAX {
        let position = touched_signatures.len();
        touched_positions[sig_idx] = position;
        touched_signature_ids.push(sig_idx);
        touched_signatures.push((sig_idx, Vec::new(), 0, 0));
        position
    } else {
        position
    };
    let (_, refs, hash_accum, len_accum) = &mut touched_signatures[position];
    refs.push(ranges);
    *hash_accum = hash_accum.wrapping_add(entry_hash);
    *len_accum += entry_range_count;
}

#[inline]
fn append_l1_exact_start_entries<'a>(
    start_state: u32,
    initial_tsids: &[u32],
    empty_token_ranges: &'a [(u32, u32)],
    empty_token_hash: u64,
    state_to_terminal_signature: &[u32],
    reuse: &L1ExactProfileReuse,
    profiles: &[Vec<(usize, &'a [(u32, u32)], u64, usize)>],
    touched_positions: &mut [usize],
    touched_signature_ids: &mut Vec<usize>,
    touched_signatures: &mut Vec<(usize, Vec<&'a [(u32, u32)]>, u64, usize)>,
    entries: &mut Vec<(u32, u32, LazyRanges<'a>)>,
) {
    for &sig_idx in touched_signature_ids.iter() {
        touched_positions[sig_idx] = usize::MAX;
    }
    touched_signature_ids.clear();
    touched_signatures.clear();

    if !empty_token_ranges.is_empty() {
        let sig_id = state_to_terminal_signature[start_state as usize];
        if sig_id != u32::MAX {
            append_l1_profile_entry_dense(
                touched_positions,
                touched_signature_ids,
                touched_signatures,
                sig_id as usize,
                empty_token_ranges,
                empty_token_hash,
                empty_token_ranges.len(),
            );
        }
    }

    let profile_ids = reuse
        .representative_profile_ids
        .get(&start_state)
        .expect("exact L1 profile reuse missing id-map representative");
    for &profile_id in profile_ids.iter() {
        if profile_id == 0 {
            continue;
        }
        for &(sig_idx, ranges, entry_hash, entry_range_count) in &profiles[profile_id as usize] {
            append_l1_profile_entry_dense(
                touched_positions,
                touched_signature_ids,
                touched_signatures,
                sig_idx,
                ranges,
                entry_hash,
                entry_range_count,
            );
        }
    }

    for (sig_idx, refs, hash, total_len) in touched_signatures.drain(..) {
        let lazy = LazyRanges {
            refs,
            hash,
            total_len,
        };
        if initial_tsids.len() > 1 {
            for &tsid in &initial_tsids[..initial_tsids.len() - 1] {
                entries.push((sig_idx as u32, tsid, lazy.clone()));
            }
        }
        entries.push((
            sig_idx as u32,
            *initial_tsids.last().expect("start state has an internal TSID"),
            lazy,
        ));
    }
}

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::stages::mapped_artifact::MappedArtifact;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::id_map_and_terminal_dwa::types::LocalIdMapTerminalDwa;
use crate::ds::weight::{shared_rangeset, Weight};
use crate::grammar::flat::TerminalID;
use crate::Vocab;

use super::l2p::equivalence_analysis::compat::{
    compute_byte_classes, FlatDfa, TokenizerView,
};
use super::types::{compile_profile_enabled, TerminalColoring, TerminalDwaPhaseProfile};

fn l1_exact_profile_reuse_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_L1_EXACT_PROFILE_REUSE")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true)
    })
}

fn l1_remaining_horizon_quotients_enabled(state_count: usize, vocab_count: usize) -> bool {
    // Large vocabularies repeatedly revisit the same residual targets at
    // progressively shorter suffix horizons.  A finite-depth Moore quotient
    // pays one state/alphabet refinement per token byte, then lets every token
    // sharing a target reuse the resulting canonical coordinate.  The target
    // frontier gate below remains the final profitability check, so small or
    // already-collapsed frontiers stay on the direct packed suffix walker.
    let structural_candidate = (4_000..=100_000).contains(&state_count)
        && vocab_count >= 50_000
        && state_count.saturating_mul(vocab_count) >= 100_000_000;
    let explicitly_enabled = std::env::var("GLRMASK_ENABLE_L1_REMAINING_HORIZON_QUOTIENTS")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false);
    (structural_candidate || explicitly_enabled)
        && state_count >= 4_000
        && state_count <= 100_000
        && vocab_count >= 50_000
        && std::env::var_os("GLRMASK_DISABLE_L1_REMAINING_HORIZON_QUOTIENTS").is_none()
}

fn l1_remaining_horizon_target_probe_supports_quotients(raw_unique_targets: usize) -> bool {
    // A depth quotient has to amortize O(depth * states * byte_classes) work.
    // If the unquotiented first-byte frontier is already small, the direct
    // packed suffix walk is cheaper than constructing the quotient family.
    raw_unique_targets >= 20_000
}

fn l1_sequential_group_assembly_override() -> Option<bool> {
    static OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("GLRMASK_L1_SEQUENTIAL_GROUP_ASSEMBLY")
            .ok()
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
            })
    })
}

#[inline]
fn auto_use_sequential_l1_group_assembly(
    contribution_group_visits: usize,
    rayon_threads: usize,
) -> bool {
    rayon_threads <= 1 || contribution_group_visits < PARALLEL_L1_GROUP_ASSEMBLY_MIN_VISITS
}

fn l1_prefers_flat_utf8_lead_bucket(
    first_byte: u8,
    token_count: usize,
    target_count: usize,
    has_empty_suffix: bool,
) -> bool {
    // Valid multi-byte UTF-8 lead-byte buckets often contain one one-byte token
    // plus many longer continuations.  On large exact frontiers, replaying the
    // byte-sorted suffixes directly is cheaper than constructing the packed
    // trie product, whose intermediate state set can grow by several million
    // entries before collapsing to one or a few final profiles.
    has_empty_suffix
        && matches!(first_byte, 0xC2..=0xF4)
        && token_count < 10_000
        && token_count.saturating_mul(target_count) >= 1_000_000
}

fn compact_l1_terminal_dwa_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_COMPACT_L1_TERMINAL_DWA")
            .map(|value| {
                let trimmed = value.trim();
                !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
            })
            .unwrap_or(true)
    })
}

/// Maximum L1 equivalence class count before falling back to L2+.
///
/// When the tokenizer DFA has more than this many distinct equivalence classes
/// for the active L1 terminals, the L1 trie traversal becomes more expensive
/// than L2P's NWA-based approach.
pub const MAX_L1_TSIDS: usize = 50;

/// Quickly count L1 equivalence classes for the given active terminals.
///
/// Used by the partition builder to decide whether L1 should be attempted
/// *before* launching the parallel L1/L2P build, avoiding a wasteful
/// L2P double-build when L1 would be skipped.
pub fn count_l1_equivalence_classes(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    active_terminals: &[bool],
) -> usize {
    let states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();
    let tokenizer_view = TokenizerView::new_filtered(tokenizer, active_terminals);
    let token_bytes: Vec<&[u8]> = vocab.entries_map().values().map(|b| b.as_slice()).collect();
    let mut relevant_bytes = [false; 256];
    for bytes in &token_bytes {
        for &byte in *bytes {
            relevant_bytes[byte as usize] = true;
        }
    }
    let byte_to_class = compute_byte_classes(tokenizer_view.dfa());
    let equiv_mapping = super::l2p::equivalence_analysis::state::max_length::find_state_equivalence_classes_byte_restricted(
        &tokenizer_view,
        &token_bytes,
        &states,
        Some(&byte_to_class),
        Some(active_terminals),
        Some(&relevant_bytes),
    );
    let mut seen = rustc_hash::FxHashSet::default();
    for &rep in &equiv_mapping {
        seen.insert(rep);
    }
    let mut max_length_representatives: Vec<usize> = seen.into_iter().collect();
    max_length_representatives.sort_unstable();

    let order = l1_identity_vocab_order(vocab);
    let flat_trans: Arc<[u32]> = Arc::from(build_flat_transition_table(tokenizer));
    let (exact_mapping, _) = find_l1_exact_state_equivalence_by_token_signatures(
        tokenizer,
        order.as_ref(),
        &max_length_representatives,
        active_terminals,
        &flat_trans,
        None,
    );
    exact_mapping
        .into_iter()
        .collect::<rustc_hash::FxHashSet<_>>()
        .len()
}

/// Build an L1 id map and terminal DWA using the configured implementation plan.
pub fn build_l1_id_map_and_terminal_dwa(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
    flat_trans: &Arc<[u32]>,
    transitions_by_byte: Option<&[u32]>,
    initial_state_map: Option<&ManyToOneIdMap>,
    shared_generic_nfa_topology: Option<&super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTopology>,
    shared_generic_nfa_trie: Option<&super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie>,
    subset_parent_order: Option<&L1IdentityVocabOrder>,
) -> Option<LocalIdMapTerminalDwa> {
    build_l1_id_map_and_terminal_dwa_mode(
        partition_label, tokenizer, vocab, terminal_coloring, use_terminal_coloring,
        ignore_terminal, grammar, active_terminals, flat_trans, transitions_by_byte,
        initial_state_map, shared_generic_nfa_topology, shared_generic_nfa_trie,
        subset_parent_order, false, false,
    )
}

pub fn build_l1_id_map_and_terminal_dwa_mode(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
    flat_trans: &Arc<[u32]>,
    transitions_by_byte: Option<&[u32]>,
    initial_state_map: Option<&ManyToOneIdMap>,
    shared_generic_nfa_topology: Option<&super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTopology>,
    shared_generic_nfa_trie: Option<&super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie>,
    subset_parent_order: Option<&L1IdentityVocabOrder>,
    initial_state_domain_is_exact: bool,
    id_map_only: bool,
) -> Option<LocalIdMapTerminalDwa> {
    let input = implementations::BuildInput {
        partition_label, tokenizer, vocab, terminal_coloring, use_terminal_coloring,
        ignore_terminal, grammar, active_terminals, flat_trans, transitions_by_byte,
        initial_state_map, shared_generic_nfa_topology, shared_generic_nfa_trie,
        subset_parent_order, initial_state_domain_is_exact, id_map_only,
    };
    if id_map_only {
        // Vocabulary-partition compilation consumes only the exact L1 token
        // quotient.  Use the dedicated vocab-equivalence kernel rather than
        // constructing tokenizer-state/DWA-shaped artifacts and discarding
        // them immediately afterwards.
        let result = implementations::build_projected_vocab_equivalence(input)?;
        let tokenizer_states = initial_state_map.cloned().unwrap_or_else(|| {
            let ids = (0..tokenizer.num_states()).collect::<Vec<_>>();
            ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                ids.clone(),
                ids,
            )
        });
        let mut id_map = InternalIdMap {
            tokenizer_states,
            vocab_tokens: result.vocab_map,
            deferred_vocab_singleton_original_ids: None,
        };
        if initial_state_domain_is_exact {
            super::scope::complete_with_continuation_singletons(&mut id_map.tokenizer_states);
        }
        return Some(LocalIdMapTerminalDwa {
            dwa: DWA::new(id_map.num_tsids(), id_map.max_internal_token_id()),
            id_map,
            profile: TerminalDwaPhaseProfile {
                id_map_ms: result.total_wall_ms,
                compact_ms: result.compact_ms,
                ..TerminalDwaPhaseProfile::default()
            },
        });
    }
    let mut result = implementations::build_from_env(input)?;
    if initial_state_domain_is_exact {
        super::scope::complete_with_continuation_singletons(
            &mut result.id_map.tokenizer_states,
        );
    }
    Some(result)
}

/// Build an L1 id_map and terminal DWA for the given vocab and terminal set.
///
/// Uses max-length state equivalence and an identity vocab map, then traverses
/// the vocab tree to accumulate `terminal -> Weight` before building the final
/// 2-state DWA directly.
///
/// Returns `None` if the vocab is empty or no terminal matches exist.
/// The caller should pre-check `count_l1_equivalence_classes()` and merge
/// L1 terminals into L2+ when the count exceeds `MAX_L1_TSIDS`.
fn build_l1_id_map_and_terminal_dwa_production_impl(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    _terminal_coloring: &TerminalColoring,
    _use_terminal_coloring: bool,
    _ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
    flat_trans: &Arc<[u32]>,
    transitions_by_byte: Option<&[u32]>,
    initial_state_map: Option<&ManyToOneIdMap>,
    shared_generic_nfa_topology: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTopology,
    >,
    shared_generic_nfa_trie: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie,
    >,
    subset_parent_order: Option<&L1IdentityVocabOrder>,
    decline_on_budget_abort: bool,
    max_raw_unique_targets: Option<usize>,
) -> Result<Option<LocalIdMapTerminalDwa>, ()> {
    if vocab.is_empty() {
        return Ok(None);
    }

    let generic_epsilon_nfa = tokenizer.has_epsilon_transitions()
        && !tokenizer.has_scalar_deterministic_dispatch();

    let total_started_at = Instant::now();
    let id_map_started_at = Instant::now();
    let sparse_initial_scope = initial_state_map_is_sparse(initial_state_map);
    let (mut id_map, vocab_order, _state_to_rep, id_map_profile, exact_profile_reuse) =
        if generic_epsilon_nfa
            && l1_generic_nfa_exact_profiles_enabled()
            && !sparse_initial_scope
        {
            build_l1_generic_nfa_exact_id_map_impl(
                tokenizer,
                vocab,
                active_terminals,
                flat_trans.as_ref(),
                initial_state_map,
                shared_generic_nfa_topology,
                shared_generic_nfa_trie,
                subset_parent_order,
                decline_on_budget_abort,
                max_raw_unique_targets,
            )?
        } else if generic_epsilon_nfa {
            build_l1_generic_nfa_fallback_id_map(tokenizer, vocab, initial_state_map)
        } else {
            build_l1_id_map(
                partition_label,
                tokenizer,
                vocab,
                active_terminals,
                flat_trans,
                transitions_by_byte,
                initial_state_map,
            )
        };
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    let num_terminals = grammar.num_terminals as u32;
    let dwa_started_at = Instant::now();
    let terminal_result = if generic_epsilon_nfa && exact_profile_reuse.is_none() {
        build_l1_generic_nfa_terminal_dwa(
            tokenizer,
            vocab_order.as_ref(),
            &mut id_map,
            num_terminals,
            active_terminals,
        )
    } else {
        build_l1_terminal_dwa(
            tokenizer,
            vocab_order.as_ref(),
            &mut id_map,
            num_terminals,
            active_terminals,
            flat_trans.as_ref(),
            exact_profile_reuse
                .as_ref()
                .filter(|_| l1_exact_profile_reuse_enabled()),
        )
    };
    let Some((dwa, terminal_profile)) = terminal_result else {
        return Ok(None);
    };
    let dwa_stats_before_compact = dwa.stats();
    let terminal_build_ms = dwa_started_at.elapsed().as_secs_f64() * 1000.0;

    let profiling = compile_profile_enabled();
    let top_profiling = std::env::var_os("GLRMASK_PROFILE_COMPILE_TOP").is_some();
    let tsids_before_compact = id_map.num_tsids();
    let tokens_before_compact = id_map.num_internal_tokens();

    let mut mapped_dwa = MappedArtifact::new(dwa, id_map);
    let (compact_report, compact_ms) = if compact_l1_terminal_dwa_enabled() {
        let compact_started_at = Instant::now();
        let compact_report = if profiling {
            mapped_dwa.compact_dimensions_fast_l1_with_stats()
        } else {
            mapped_dwa.compact_dimensions_fast_l1()
        };
        let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
        (Some(compact_report), compact_ms)
    } else {
        (None, 0.0)
    };
    // Token compaction normally consumes the deferred singleton coordinate.
    // If it was a no-op, materialize the ordinary map before this local
    // artifact escapes the L1 builder.
    mapped_dwa
        .parts_mut()
        .1
        .materialize_deferred_vocab_singletons();
    let dwa_stats_after_compact = mapped_dwa.artifact().stats();
    let tsids_after_compact = mapped_dwa.id_map().num_tsids();
    let tokens_after_compact = mapped_dwa.id_map().num_internal_tokens();
    let (dwa, id_map) = mapped_dwa.into_parts();
    if initial_state_map_is_sparse(initial_state_map)
        && std::env::var_os("GLRMASK_DEBUG_SCOPED_BOUNDARY").is_some()
    {
        eprintln!(
            "SCOPED_L1_OUT partition={} o2i={:?} classes={:?} reps={:?}",
            partition_label,
            id_map.tokenizer_states.original_to_internal,
            id_map.tokenizer_states.internal_to_originals,
            id_map.tokenizer_states.representative_original_ids,
        );
    }
    let compact_tsid_shrink_pct = if tsids_before_compact > 0 {
        (tsids_before_compact as f64 - tsids_after_compact as f64) * 100.0
            / tsids_before_compact as f64
    } else {
        0.0
    };
    let compact_vocab_shrink_pct = if tokens_before_compact > 0 {
        (tokens_before_compact as f64 - tokens_after_compact as f64) * 100.0
            / tokens_before_compact as f64
    } else {
        0.0
    };

    if profiling || top_profiling {
        let stats_str = if let Some(stats) = compact_report.as_ref().and_then(|report| report.profile_stats) {
            format!(
                " compact_tsids_before={} compact_tsids_after={} compact_tokens_before={} compact_tokens_after={} compact_tsid_shrink_pct={:.2} compact_vocab_shrink_pct={:.2} compact_token_ranges_before={} compact_token_ranges_after={}",
                stats.tsids_before, stats.tsids_after,
                stats.tokens_before, stats.tokens_after,
                compact_tsid_shrink_pct, compact_vocab_shrink_pct,
                stats.token_ranges_before, stats.token_ranges_after,
            )
        } else {
            format!(
                " compact_tsids_before={} compact_tsids_after={} compact_tokens_before={} compact_tokens_after={} compact_tsid_shrink_pct={:.2} compact_vocab_shrink_pct={:.2}",
                tsids_before_compact, tsids_after_compact,
                tokens_before_compact, tokens_after_compact,
                compact_tsid_shrink_pct, compact_vocab_shrink_pct,
            )
        };
        eprintln!(
            "[glrmask/profile][l1] partition={} vocab_tokens={} tsids={} rep_states={} initial_states_considered={} max_length_skipped={} max_token_len={} token_len_gt_4={} token_len_gt_8={} token_len_gt_16={} token_len_gt_32={} token_len_gt_64={} state_equiv_ms={:.3} max_length_state_equiv_ms={:.3} exact_state_equiv_ms={:.3} max_length_reps={} exact_reps={} token_identity_map_ms={:.3} id_map_ms={:.3} internal_vocab_ms={:.3} vocab_tree_build_ms={:.3} state_seed_ms={:.3} token_set_intern_ms={:.3} tsid_profile_merge_ms={:.3} tsid_profile_merge_before={} tsid_profile_merge_after={} vocab_tree_traversal_ms={:.3} direct_terminal_dwa_ms={:.3} dwa_states={} dwa_transitions={} dwa_transition_pairs={} dwa_interned_ranges_before_compact={} dwa_interned_ranges_after_compact={} terminal_build_ms={:.3} compact_ms={:.3} determinize=none minimize=none prune=none total_ms={:.3}{}",
            partition_label,
            vocab.entries_map().len(),
            id_map.num_tsids(),
            id_map.tokenizer_states.representative_original_ids.len(),
            id_map_profile.initial_states_considered,
            id_map_profile.max_length_skipped,
            id_map_profile.max_token_len,
            id_map_profile.token_len_gt_4,
            id_map_profile.token_len_gt_8,
            id_map_profile.token_len_gt_16,
            id_map_profile.token_len_gt_32,
            id_map_profile.token_len_gt_64,
            id_map_profile.state_equiv_ms,
            id_map_profile.max_length_state_equiv_ms,
            id_map_profile.exact_state_equiv_ms,
            id_map_profile.max_length_reps,
            id_map_profile.exact_reps,
            id_map_profile.token_identity_map_ms,
            id_map_ms,
            terminal_profile.internal_vocab_ms,
            terminal_profile.vocab_tree_build_ms,
            terminal_profile.state_seed_ms,
            terminal_profile.token_set_intern_ms,
            terminal_profile.tsid_profile_merge_ms,
            terminal_profile.tsid_profile_merge_before,
            terminal_profile.tsid_profile_merge_after,
            terminal_profile.vocab_tree_traversal_ms,
            terminal_profile.direct_terminal_dwa_ms,
            dwa_stats_before_compact.states,
            dwa_stats_before_compact.transitions,
            dwa_stats_before_compact.transition_pairs,
            dwa_stats_before_compact.interned_ranges,
            dwa_stats_after_compact.interned_ranges,
            terminal_build_ms,
            compact_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
            stats_str,
        );
    }

    Ok(Some(LocalIdMapTerminalDwa {
        id_map,
        dwa,
        profile: TerminalDwaPhaseProfile {
            id_map_ms,
            terminal_dwa_ms: terminal_build_ms,
            compact_ms,
            ..TerminalDwaPhaseProfile::default()
        },
    }))
}

pub(super) fn build_l1_id_map_and_terminal_dwa_production(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
    flat_trans: &Arc<[u32]>,
    transitions_by_byte: Option<&[u32]>,
    initial_state_map: Option<&ManyToOneIdMap>,
    shared_generic_nfa_topology: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTopology,
    >,
    shared_generic_nfa_trie: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie,
    >,
    subset_parent_order: Option<&L1IdentityVocabOrder>,
) -> Option<LocalIdMapTerminalDwa> {
    build_l1_id_map_and_terminal_dwa_production_impl(
        partition_label,
        tokenizer,
        vocab,
        terminal_coloring,
        use_terminal_coloring,
        ignore_terminal,
        grammar,
        active_terminals,
        flat_trans,
        transitions_by_byte,
        initial_state_map,
        shared_generic_nfa_topology,
        shared_generic_nfa_trie,
        subset_parent_order,
        false,
        None,
    )
    .expect("ordinary production L1 build cannot decline")
}

pub(super) fn try_build_l1_id_map_and_terminal_dwa_production(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
    flat_trans: &Arc<[u32]>,
    transitions_by_byte: Option<&[u32]>,
    initial_state_map: Option<&ManyToOneIdMap>,
    shared_generic_nfa_topology: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTopology,
    >,
    shared_generic_nfa_trie: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie,
    >,
    subset_parent_order: Option<&L1IdentityVocabOrder>,
    max_raw_unique_targets: Option<usize>,
) -> Option<Option<LocalIdMapTerminalDwa>> {
    build_l1_id_map_and_terminal_dwa_production_impl(
        partition_label,
        tokenizer,
        vocab,
        terminal_coloring,
        use_terminal_coloring,
        ignore_terminal,
        grammar,
        active_terminals,
        flat_trans,
        transitions_by_byte,
        initial_state_map,
        shared_generic_nfa_topology,
        shared_generic_nfa_trie,
        subset_parent_order,
        true,
        max_raw_unique_targets,
    )
    .ok()
}

fn l1_generic_nfa_exact_profiles_enabled() -> bool {
    std::env::var("GLRMASK_L1_GENERIC_NFA_EXACT_PROFILES")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true)
}

const L1_GENERIC_NFA_TOKEN_BOUNDED_MAX_STATE_VOCAB_PAIRS: usize = 350_000_000;
const L1_GENERIC_NFA_TOKEN_BOUNDED_MAX_VOCAB: usize = 20_000;
const L1_GENERIC_NFA_TOKEN_BOUNDED_LARGE_WORK_BUDGET:
    super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisWorkBudget =
    super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisWorkBudget {
        max_configurations: 64_000,
        max_trie_visits: 1_050_000,
    };

fn l1_generic_nfa_token_bounded_large_work_budget_with_override(
    max_trie_visits: Option<usize>,
) -> super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisWorkBudget {
    let mut budget = L1_GENERIC_NFA_TOKEN_BOUNDED_LARGE_WORK_BUDGET;
    if let Some(max_trie_visits) = max_trie_visits {
        budget.max_trie_visits = max_trie_visits;
    }
    budget
}

fn l1_generic_nfa_token_bounded_large_work_budget(
) -> super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisWorkBudget {
    l1_generic_nfa_token_bounded_large_work_budget_with_override(
        std::env::var("GLRMASK_L1_TOKEN_BOUNDED_MAX_TRIE_VISITS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok()),
    )
}
const L1_GENERIC_NFA_RELEVANT_POWERSET_PROBE_MIN_VOCAB: usize = 20_000;
const L1_GENERIC_NFA_RELEVANT_POWERSET_PROBE_MAX_STATES: usize = 4_096;
const L1_GENERIC_NFA_TINY_VOCAB_POWERSET_MIN_RAW_STATES: usize = 60_000;
const L1_GENERIC_NFA_TINY_VOCAB_POWERSET_MAX_VOCAB: usize = 128;
const L1_GENERIC_NFA_SPARSE_MEDIUM_POWERSET_MIN_RAW_STATES: usize = 40_000;
const L1_GENERIC_NFA_SPARSE_MEDIUM_POWERSET_MAX_RAW_STATES: usize = 60_000;
const L1_GENERIC_NFA_SPARSE_MEDIUM_POWERSET_MIN_VOCAB: usize = 512;
const L1_GENERIC_NFA_SPARSE_MEDIUM_POWERSET_MAX_VOCAB: usize = 1_024;
const L1_GENERIC_NFA_SPARSE_MEDIUM_POWERSET_MAX_ACTIVE_TERMINALS: usize = 128;
const L1_GENERIC_NFA_DENSE_MEDIUM_POWERSET_MIN_RAW_STATES: usize = 60_000;
const L1_GENERIC_NFA_DENSE_MEDIUM_POWERSET_MIN_ACTIVE_TERMINALS: usize = 128;
const L1_GENERIC_NFA_DENSE_MEDIUM_POWERSET_MAX_ACTIVE_TERMINALS: usize = 224;
const L1_GENERIC_NFA_SMALL_VOCAB_POWERSET_MAX_STATES: usize = 131_072;

fn l1_generic_nfa_small_vocab_powerset_probe_enabled(
    state_count: usize,
    vocab_count: usize,
    active_terminal_count: usize,
) -> bool {
    let tiny_vocab_large_domain = state_count >= L1_GENERIC_NFA_TINY_VOCAB_POWERSET_MIN_RAW_STATES
        && vocab_count <= L1_GENERIC_NFA_TINY_VOCAB_POWERSET_MAX_VOCAB;
    let sparse_medium_domain =
        (L1_GENERIC_NFA_SPARSE_MEDIUM_POWERSET_MIN_RAW_STATES
            ..=L1_GENERIC_NFA_SPARSE_MEDIUM_POWERSET_MAX_RAW_STATES)
            .contains(&state_count)
            && (L1_GENERIC_NFA_SPARSE_MEDIUM_POWERSET_MIN_VOCAB
                ..=L1_GENERIC_NFA_SPARSE_MEDIUM_POWERSET_MAX_VOCAB)
                .contains(&vocab_count)
            && active_terminal_count
                <= L1_GENERIC_NFA_SPARSE_MEDIUM_POWERSET_MAX_ACTIVE_TERMINALS;
    let dense_medium_large_domain =
        state_count >= L1_GENERIC_NFA_DENSE_MEDIUM_POWERSET_MIN_RAW_STATES
            && (L1_GENERIC_NFA_SPARSE_MEDIUM_POWERSET_MIN_VOCAB
                ..=L1_GENERIC_NFA_SPARSE_MEDIUM_POWERSET_MAX_VOCAB)
                .contains(&vocab_count)
            && (L1_GENERIC_NFA_DENSE_MEDIUM_POWERSET_MIN_ACTIVE_TERMINALS
                ..=L1_GENERIC_NFA_DENSE_MEDIUM_POWERSET_MAX_ACTIVE_TERMINALS)
                .contains(&active_terminal_count);
    tiny_vocab_large_domain || sparse_medium_domain || dense_medium_large_domain
}

pub fn l1_generic_nfa_token_bounded_view_enabled(
    state_count: usize,
    vocab_count: usize,
) -> bool {
    // The token-bounded topology is exact and dramatically cheaper than
    // arbitrary relevant-byte closure when the real token language barely
    // creates any virtual epsilon-closure configurations. Its construction can
    // nevertheless approach the raw-start × token-trie product on large lexer
    // and vocabulary combinations. Keep the bounded route in its profitable
    // regime and use the older exact relevant-powerset proof above that budget.
    vocab_count <= L1_GENERIC_NFA_TOKEN_BOUNDED_MAX_VOCAB
        && state_count.saturating_mul(vocab_count)
            <= L1_GENERIC_NFA_TOKEN_BOUNDED_MAX_STATE_VOCAB_PAIRS
}


fn build_l1_generic_nfa_analysis_view_impl(
    tokenizer: &Tokenizer,
    raw_states: &[usize],
    token_entries: &[(u32, Arc<[u8]>)],
    active_terminals: &[bool],
    flat_trans: &[u32],
    shared_topology: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTopology,
    >,
    shared_token_trie: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie,
    >,
    large_work_budget: super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisWorkBudget,
    decline_on_budget_abort: bool,
) -> Result<(Vec<usize>, TokenizerView, &'static str), ()> {
    let mut relevant_bytes = [false; 256];
    for (_, bytes) in token_entries {
        for &byte in bytes.iter() {
            relevant_bytes[byte as usize] = true;
        }
    }

    let active_terminal_count = active_terminals.iter().filter(|&&active| active).count();
    if l1_generic_nfa_small_vocab_powerset_probe_enabled(
        raw_states.len(),
        token_entries.len(),
        active_terminal_count,
    ) {
        match super::l2p::equivalence_analysis::state_equivalence::nfa::build_relevant_powerset_view_budgeted(
            tokenizer,
            &relevant_bytes,
            Some(active_terminals),
            None,
            super::l2p::equivalence_analysis::state_equivalence::nfa::RelevantPowersetWorkBudget {
                max_configurations: L1_GENERIC_NFA_SMALL_VOCAB_POWERSET_MAX_STATES,
                max_edges: usize::MAX,
            },
        ) {
            Ok(powerset_view) => {
                let state_count = powerset_view.configurations.len();
                let view_states = raw_states
                    .iter()
                    .map(|&raw_state| powerset_view.raw_start_to_view[raw_state] as usize)
                    .collect::<Vec<_>>();
                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][l1_small_vocab_powerset_probe] outcome=selected raw_states={} vocab_tokens={} view_states={} max_states={}",
                        raw_states.len(),
                        token_entries.len(),
                        state_count,
                        L1_GENERIC_NFA_SMALL_VOCAB_POWERSET_MAX_STATES,
                    );
                }
                return Ok((
                    view_states,
                    powerset_view.into_tokenizer_view(),
                    "relevant_powerset_small_vocab",
                ));
            }
            Err(work) => {
                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][l1_small_vocab_powerset_probe] outcome=aborted raw_states={} vocab_tokens={} states={} max_states={}",
                        raw_states.len(),
                        token_entries.len(),
                        work.configurations,
                        L1_GENERIC_NFA_SMALL_VOCAB_POWERSET_MAX_STATES,
                    );
                }
            }
        }
    }

    let use_unbudgeted_token_bounded_view = !decline_on_budget_abort
        && l1_generic_nfa_token_bounded_view_enabled(raw_states.len(), token_entries.len());
    if !use_unbudgeted_token_bounded_view
        && token_entries.len() >= L1_GENERIC_NFA_RELEVANT_POWERSET_PROBE_MIN_VOCAB
    {
        match super::l2p::equivalence_analysis::state_equivalence::nfa::build_relevant_powerset_view_budgeted(
            tokenizer,
            &relevant_bytes,
            Some(active_terminals),
            None,
            super::l2p::equivalence_analysis::state_equivalence::nfa::RelevantPowersetWorkBudget {
                max_configurations: L1_GENERIC_NFA_RELEVANT_POWERSET_PROBE_MAX_STATES,
                max_edges: usize::MAX,
            },
        ) {
            Ok(powerset_view) => {
                let state_count = powerset_view.configurations.len();
                let view_states = raw_states
                    .iter()
                    .map(|&raw_state| powerset_view.raw_start_to_view[raw_state] as usize)
                    .collect::<Vec<_>>();
                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][l1_relevant_powerset_probe] outcome=selected raw_states={} vocab_tokens={} view_states={} max_states={}",
                        raw_states.len(),
                        token_entries.len(),
                        state_count,
                        L1_GENERIC_NFA_RELEVANT_POWERSET_PROBE_MAX_STATES,
                    );
                }
                return Ok((
                    view_states,
                    powerset_view.into_tokenizer_view(),
                    "relevant_powerset_probe",
                ));
            }
            Err(work) => {
                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][l1_relevant_powerset_probe] outcome=aborted raw_states={} vocab_tokens={} states={} max_states={}",
                        raw_states.len(),
                        token_entries.len(),
                        work.configurations,
                        L1_GENERIC_NFA_RELEVANT_POWERSET_PROBE_MAX_STATES,
                    );
                }
            }
        }
    }
    let bounded_view = if use_unbudgeted_token_bounded_view {
        let bounded = if let Some(topology) = shared_topology {
            topology.materialize(tokenizer, Some(active_terminals))
        } else {
            let tokens = token_entries
                .iter()
                .map(|(_, bytes)| bytes.as_ref())
                .collect::<Vec<_>>();
            super::l2p::equivalence_analysis::state_equivalence::nfa::build_token_bounded_analysis_view_projected_sorted_with_raw_transitions_and_trie(
                tokenizer,
                flat_trans,
                raw_states,
                &tokens,
                active_terminals,
                shared_token_trie,
            )
        };
        Some((bounded, "token_bounded"))
    } else {
        let tokens = token_entries
            .iter()
            .map(|(_, bytes)| bytes.as_ref())
            .collect::<Vec<_>>();
        let bounded = super::l2p::equivalence_analysis::state_equivalence::nfa::try_build_token_bounded_analysis_view_projected_sorted_with_raw_transitions_and_trie(
            tokenizer,
            flat_trans,
            raw_states,
            &tokens,
            active_terminals,
            large_work_budget,
            shared_token_trie,
        );
        match bounded {
            Ok((bounded, work)) => {
                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][l1_token_bounded_budget] outcome=complete raw_states={} vocab_tokens={} configurations={} trie_visits={}",
                        raw_states.len(),
                        token_entries.len(),
                        work.configurations,
                        work.trie_visits,
                    );
                }
                Some((bounded, "token_bounded_budgeted"))
            }
            Err(work) => {
                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][l1_token_bounded_budget] outcome=aborted raw_states={} vocab_tokens={} configurations={} trie_visits={}",
                        raw_states.len(),
                        token_entries.len(),
                        work.configurations,
                        work.trie_visits,
                    );
                }
                if decline_on_budget_abort {
                    return Err(());
                }
                None
            }
        }
    };

    if let Some((bounded, analysis_view)) = bounded_view {
        let view_states = raw_states
            .iter()
            .map(|&raw_state| bounded.view_state_for_raw_start(raw_state))
            .collect::<Vec<_>>();
        return Ok((view_states, bounded.tokenizer_view, analysis_view));
    }

    let powerset_view = super::l2p::equivalence_analysis::state_equivalence::nfa::build_relevant_powerset_view(
        tokenizer,
        &relevant_bytes,
        Some(active_terminals),
        None,
    );
    let view_states = raw_states
        .iter()
        .map(|&raw_state| powerset_view.raw_start_to_view[raw_state] as usize)
        .collect::<Vec<_>>();
    Ok((
        view_states,
        powerset_view.into_tokenizer_view(),
        "relevant_powerset",
    ))
}

fn build_l1_generic_nfa_analysis_view(
    tokenizer: &Tokenizer,
    raw_states: &[usize],
    token_entries: &[(u32, Arc<[u8]>)],
    active_terminals: &[bool],
    flat_trans: &[u32],
    shared_topology: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTopology,
    >,
    shared_token_trie: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie,
    >,
    large_work_budget: super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisWorkBudget,
) -> (Vec<usize>, TokenizerView, &'static str) {
    build_l1_generic_nfa_analysis_view_impl(
        tokenizer,
        raw_states,
        token_entries,
        active_terminals,
        flat_trans,
        shared_topology,
        shared_token_trie,
        large_work_budget,
        false,
    )
    .expect("ordinary L1 analysis view cannot decline")
}

fn build_l1_generic_nfa_exact_id_map_impl<'a>(
    tokenizer: &Tokenizer,
    vocab: &'a Vocab,
    active_terminals: &[bool],
    flat_trans: &[u32],
    initial_state_map: Option<&ManyToOneIdMap>,
    shared_topology: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTopology,
    >,
    shared_token_trie: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie,
    >,
    subset_parent_order: Option<&L1IdentityVocabOrder>,
    decline_on_budget_abort: bool,
    max_raw_unique_targets: Option<usize>,
) -> Result<(
    InternalIdMap,
    Arc<L1IdentityVocabOrder>,
    Vec<u32>,
    L1IdMapProfile,
    Option<L1ExactProfileReuse>,
), ()> {
    let num_states = tokenizer.num_states() as usize;
    let token_identity_started_at = Instant::now();
    let vocab_order = subset_parent_order
        .map(|parent| derive_l1_identity_vocab_order_from_parent(parent, vocab))
        .unwrap_or_else(|| l1_identity_vocab_order(vocab));
    let deferred_vocab_singleton_original_ids = Arc::clone(&vocab_order.token_ids_sorted);
    let token_identity_map_ms = token_identity_started_at.elapsed().as_secs_f64() * 1000.0;
    let token_entries = vocab_order.token_entries_sorted.as_ref();
    let token_len_stats = token_length_stats_from_entries(token_entries);
    let max_token_len = token_entries
        .iter()
        .map(|(_, bytes)| bytes.len())
        .max()
        .unwrap_or(0);
    let raw_states = initial_state_map.map_or_else(
        || (0..num_states).collect::<Vec<_>>(),
        |map| {
            assert_eq!(
                map.original_to_internal.len(),
                num_states,
                "generic epsilon L1 seed map must cover the raw tokenizer domain",
            );
            map.representative_original_ids
                .iter()
                .map(|&state| state as usize)
                .collect::<Vec<_>>()
        },
    );

    let state_equiv_started_at = Instant::now();
    let view_started_at = Instant::now();
    let analysis_budget = if decline_on_budget_abort {
        let max_trie_visits = std::env::var("GLRMASK_L1_QUOTIENT_TRY_MAX_TRIE_VISITS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(500_000);
        l1_generic_nfa_token_bounded_large_work_budget_with_override(Some(max_trie_visits))
    } else {
        l1_generic_nfa_token_bounded_large_work_budget()
    };
    let (view_states, tokenizer_view, analysis_view) = build_l1_generic_nfa_analysis_view_impl(
        tokenizer,
        &raw_states,
        token_entries,
        active_terminals,
        flat_trans,
        shared_topology,
        shared_token_trie,
        analysis_budget,
        decline_on_budget_abort,
    )?;
    let view_build_ms = view_started_at.elapsed().as_secs_f64() * 1000.0;

    let terminal_signature_started_at = compile_profile_enabled().then(Instant::now);
    let (state_to_terminal_signature, terminal_signatures) =
        build_l1_flat_state_to_terminal_signatures(tokenizer_view.dfa());
    let terminal_signature_ms = terminal_signature_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    let exact_started_at = Instant::now();
    let (exact_mapping, exact_profile_reuse) =
        find_l1_exact_state_equivalence_by_flat_signatures_maybe_decline(
            vocab_order.as_ref(),
            &view_states,
            state_to_terminal_signature,
            terminal_signatures,
            &tokenizer_view,
            None,
            true,
            terminal_signature_ms,
            None,
            max_raw_unique_targets,
        )?;
    let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(exact_mapping.len(), raw_states.len());
    let mut exact_profile_reuse =
        exact_profile_reuse.expect("generic epsilon L1 exact analysis must retain profiles");

    // Exact classification runs only on representatives from the previously
    // proved seed partition. Compose its result back over the full raw-state
    // domain rather than re-analyzing every member of every seed class.
    let mut exact_rep_to_internal = FxHashMap::<usize, u32>::default();
    let mut seed_rep_to_internal = FxHashMap::<usize, u32>::default();
    let mut seed_rep_to_exact_rep = FxHashMap::<usize, usize>::default();
    let mut seed_rep_to_view_state = FxHashMap::<usize, usize>::default();
    let mut raw_representatives = Vec::<u32>::new();
    for ((&seed_rep, &view_state), &exact_rep) in raw_states
        .iter()
        .zip(&view_states)
        .zip(&exact_mapping)
    {
        let internal = *exact_rep_to_internal.entry(exact_rep).or_insert_with(|| {
            let internal = raw_representatives.len() as u32;
            raw_representatives.push(seed_rep as u32);
            internal
        });
        seed_rep_to_internal.insert(seed_rep, internal);
        seed_rep_to_exact_rep.insert(seed_rep, exact_rep);
        seed_rep_to_view_state.insert(seed_rep, view_state);
    }

    let seed_representative_for_raw = |raw_state: usize| -> usize {
        initial_state_map.map_or(raw_state, |map| {
            let internal = map.original_to_internal[raw_state];
            assert_ne!(
                internal,
                u32::MAX,
                "generic epsilon L1 seed map omitted raw state {raw_state}",
            );
            map.representative_original_ids[internal as usize] as usize
        })
    };
    let mut original_to_internal = vec![u32::MAX; num_states];
    for (raw_state, internal) in original_to_internal.iter_mut().enumerate() {
        let seed_rep = seed_representative_for_raw(raw_state);
        *internal = *seed_rep_to_internal
            .get(&seed_rep)
            .expect("generic epsilon L1 seed representative missing exact class");
    }
    let mut tokenizer_states = ManyToOneIdMap::from_original_to_internal_with_representatives(
        original_to_internal,
        raw_representatives.len() as u32,
        raw_representatives,
    );
    if initial_state_map_contains_original(initial_state_map, tokenizer.initial_state_id()) {
        tokenizer_states.isolate_original(tokenizer.initial_state_id());
    }

    let view_profile_ids = std::mem::take(&mut exact_profile_reuse.representative_profile_ids);
    let view_direct_signatures = Arc::clone(&exact_profile_reuse.direct_state_to_terminal_signature);
    let mut raw_profile_ids = FxHashMap::<u32, Arc<[u32]>>::default();
    for raw_representative in tokenizer_states.iter_representative_ids() {
        let seed_rep = seed_representative_for_raw(raw_representative as usize);
        let exact_rep = *seed_rep_to_exact_rep
            .get(&seed_rep)
            .expect("generic epsilon L1 representative missing exact profile") as u32;
        let profile_ids = view_profile_ids
            .get(&exact_rep)
            .unwrap_or_else(|| panic!("missing generic L1 exact profile for view state {exact_rep}"));
        raw_profile_ids.insert(raw_representative, Arc::clone(profile_ids));
    }
    let raw_direct_signatures = (0..num_states)
        .map(|raw_state| {
            let seed_rep = seed_representative_for_raw(raw_state);
            let view_state = seed_rep_to_view_state[&seed_rep];
            view_direct_signatures[view_state]
        })
        .collect::<Vec<_>>();
    exact_profile_reuse.representative_profile_ids = raw_profile_ids;
    exact_profile_reuse.direct_state_to_terminal_signature = raw_direct_signatures.into();
    exact_profile_reuse.profile_representatives_by_internal = Arc::from([]);

    let exact_reps = tokenizer_states.num_internal_ids() as usize;
    let state_to_rep = state_to_representative_vector(&tokenizer_states, num_states);
    let state_equiv_ms = state_equiv_started_at.elapsed().as_secs_f64() * 1000.0;
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l1_generic_nfa_exact] analysis_view={} raw_states={} view_states={} view_build_ms={:.3} exact_ms={:.3} exact_reps={} total_ms={:.3}",
            analysis_view,
            raw_states.len(),
            tokenizer_view.dfa().states.len(),
            view_build_ms,
            exact_state_equiv_ms,
            exact_reps,
            state_equiv_ms,
        );
    }

    Ok((
        InternalIdMap {
            tokenizer_states,
            vocab_tokens: ManyToOneIdMap::empty(),
            deferred_vocab_singleton_original_ids: Some(deferred_vocab_singleton_original_ids),
        },
        vocab_order,
        state_to_rep,
        L1IdMapProfile {
            initial_states_considered: raw_states.len(),
            max_length_skipped: true,
            max_token_len,
            token_len_gt_4: token_len_stats.gt_4,
            token_len_gt_8: token_len_stats.gt_8,
            token_len_gt_16: token_len_stats.gt_16,
            token_len_gt_32: token_len_stats.gt_32,
            token_len_gt_64: token_len_stats.gt_64,
            state_equiv_ms,
            max_length_state_equiv_ms: 0.0,
            exact_state_equiv_ms,
            max_length_reps: raw_states.len(),
            exact_reps,
            token_identity_map_ms,
        },
        Some(exact_profile_reuse),
    ))
}

fn build_l1_generic_nfa_exact_id_map<'a>(
    tokenizer: &Tokenizer,
    vocab: &'a Vocab,
    active_terminals: &[bool],
    flat_trans: &[u32],
    initial_state_map: Option<&ManyToOneIdMap>,
    shared_topology: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTopology,
    >,
    shared_token_trie: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie,
    >,
    subset_parent_order: Option<&L1IdentityVocabOrder>,
) -> (
    InternalIdMap,
    Arc<L1IdentityVocabOrder>,
    Vec<u32>,
    L1IdMapProfile,
    Option<L1ExactProfileReuse>,
) {
    build_l1_generic_nfa_exact_id_map_impl(
        tokenizer,
        vocab,
        active_terminals,
        flat_trans,
        initial_state_map,
        shared_topology,
        shared_token_trie,
        subset_parent_order,
        false,
        None,
    )
    .expect("ordinary exact L1 build cannot decline")
}

fn build_l1_generic_nfa_fallback_id_map<'a>(
    tokenizer: &Tokenizer,
    vocab: &'a Vocab,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> (
    InternalIdMap,
    Arc<L1IdentityVocabOrder>,
    Vec<u32>,
    L1IdMapProfile,
    Option<L1ExactProfileReuse>,
) {
    let num_states = tokenizer.num_states() as usize;
    let token_bytes = vocab
        .entries_map()
        .values()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let token_len_stats = token_length_stats(&token_bytes);
    let max_token_len = token_bytes.iter().map(|bytes| bytes.len()).max().unwrap_or(0);
    let (deferred_vocab_singleton_original_ids, vocab_order, token_identity_map_ms) =
        build_l1_identity_vocab_order(vocab);

    // Raw TSIDs are the token-boundary coordinate.  Do not quotient them
    // through powerset scanner configurations before the exact L1 profile is
    // known: different raw branches may share a scanner configuration while
    // carrying different parser/GSS histories.  A previously proved global
    // state quotient is safe to reuse; otherwise start from identity.
    let mut tokenizer_states = initial_state_map.cloned().unwrap_or_else(|| {
        let ids = (0..tokenizer.num_states()).collect::<Vec<_>>();
        ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
            ids.clone(),
            ids,
        )
    });
    if initial_state_map_contains_original(initial_state_map, tokenizer.initial_state_id()) {
        tokenizer_states.isolate_original(tokenizer.initial_state_id());
    }
    let state_to_rep = state_to_representative_vector(&tokenizer_states, num_states);
    let exact_reps = tokenizer_states.num_internal_ids() as usize;

    (
        InternalIdMap {
            tokenizer_states,
            vocab_tokens: ManyToOneIdMap::empty(),
            deferred_vocab_singleton_original_ids: Some(deferred_vocab_singleton_original_ids),
        },
        vocab_order,
        state_to_rep,
        L1IdMapProfile {
            initial_states_considered: exact_reps,
            max_length_skipped: true,
            max_token_len,
            token_len_gt_4: token_len_stats.gt_4,
            token_len_gt_8: token_len_stats.gt_8,
            token_len_gt_16: token_len_stats.gt_16,
            token_len_gt_32: token_len_stats.gt_32,
            token_len_gt_64: token_len_stats.gt_64,
            state_equiv_ms: 0.0,
            max_length_state_equiv_ms: 0.0,
            exact_state_equiv_ms: 0.0,
            max_length_reps: exact_reps,
            exact_reps,
            token_identity_map_ms,
        },
        None,
    )
}

const L1_NFA_UNKNOWN_CONFIG: u32 = u32::MAX - 1;
const L1_NFA_DEAD_CONFIG: u32 = u32::MAX;

struct L1NfaPowerset<'a> {
    tokenizer: &'a Tokenizer,
    active_terminals: &'a [bool],
    configs: Vec<Box<[u32]>>,
    config_ids: FxHashMap<Vec<u32>, u32>,
    transitions: Vec<[u32; 256]>,
    signatures: Vec<Box<[u32]>>,
    transition_misses: usize,
}

impl<'a> L1NfaPowerset<'a> {
    fn new(tokenizer: &'a Tokenizer, active_terminals: &'a [bool]) -> Self {
        Self {
            tokenizer,
            active_terminals,
            configs: Vec::new(),
            config_ids: FxHashMap::default(),
            transitions: Vec::new(),
            signatures: Vec::new(),
            transition_misses: 0,
        }
    }

    fn intern(&mut self, mut states: Vec<u32>) -> u32 {
        if states.is_empty() {
            return L1_NFA_DEAD_CONFIG;
        }
        states.sort_unstable();
        states.dedup();
        if let Some(&config) = self.config_ids.get(&states) {
            return config;
        }

        let config = self.configs.len() as u32;
        let mut signature = Vec::<u32>::new();
        for &state in &states {
            signature.extend(collect_active_terminal_signature(
                self.tokenizer,
                state,
                self.active_terminals,
            ));
        }
        signature.sort_unstable();
        signature.dedup();

        self.config_ids.insert(states.clone(), config);
        self.configs.push(states.into_boxed_slice());
        self.transitions.push([L1_NFA_UNKNOWN_CONFIG; 256]);
        self.signatures.push(signature.into_boxed_slice());
        config
    }

    fn start_config(&mut self, raw_state: u32) -> u32 {
        self.intern(
            self.tokenizer
                .execute_from_state_end_only(&[], raw_state)
                .to_vec(),
        )
    }

    fn start_config_scoped(&mut self, raw_state: u32, state_map: &ManyToOneIdMap) -> u32 {
        if raw_state == self.tokenizer.initial_state_id()
            && let Some(dispatch_roots) = self.tokenizer.deterministic_dispatch_roots()
        {
            let mut states = Vec::<u32>::new();
            for &root in dispatch_roots {
                if state_map
                    .original_to_internal
                    .get(root as usize)
                    .copied()
                    .is_none_or(|internal| internal == u32::MAX)
                {
                    continue;
                }
                states.extend(
                    self.tokenizer
                        .execute_from_state_end_only(&[], root)
                        .iter()
                        .copied(),
                );
            }
            return self.intern(states);
        }
        self.start_config(raw_state)
    }

    #[inline]
    fn step(&mut self, config: u32, byte: u8) -> u32 {
        if config == L1_NFA_DEAD_CONFIG {
            return L1_NFA_DEAD_CONFIG;
        }
        let cached = self.transitions[config as usize][byte as usize];
        if cached != L1_NFA_UNKNOWN_CONFIG {
            return cached;
        }
        self.transition_misses += 1;
        let targets = self
            .tokenizer
            .step_all(&self.configs[config as usize], byte)
            .to_vec();
        let target = self.intern(targets);
        self.transitions[config as usize][byte as usize] = target;
        target
    }

    fn step_bytes(&mut self, mut config: u32, bytes: &[u8]) -> u32 {
        for &byte in bytes {
            config = self.step(config, byte);
            if config == L1_NFA_DEAD_CONFIG {
                break;
            }
        }
        config
    }

    #[inline]
    fn signature(&self, config: u32) -> &[u32] {
        if config == L1_NFA_DEAD_CONFIG {
            &[]
        } else {
            &self.signatures[config as usize]
        }
    }
}

fn collect_l1_nfa_profile_from_trie(
    scanner: &mut L1NfaPowerset<'_>,
    node: &VocabPrefixTreeNode,
    config: u32,
    token_aliases: &[Vec<u32>],
    token_ranges_by_terminal: &mut FxHashMap<u32, Vec<(u32, u32)>>,
    node_visits: &mut usize,
) {
    *node_visits += 1;
    if node.has_token() {
        for &terminal in scanner.signature(config) {
            let ranges = token_ranges_by_terminal.entry(terminal).or_default();
            for &token_id in &token_aliases[node.token_id()] {
                append_token_id_range(ranges, token_id);
            }
        }
    }

    for (edge, child) in node.iter_children() {
        let target = scanner.step_bytes(config, edge);
        if target == L1_NFA_DEAD_CONFIG {
            continue;
        }
        collect_l1_nfa_profile_from_trie(
            scanner,
            child,
            target,
            token_aliases,
            token_ranges_by_terminal,
            node_visits,
        );
    }
}

fn build_l1_generic_nfa_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab_order: &L1IdentityVocabOrder,
    id_map: &mut InternalIdMap,
    num_terminals: u32,
    active_terminals: &[bool],
) -> Option<(DWA, L1TerminalBuildProfile)> {
    let tsids_before_merge = id_map.num_tsids() as usize;
    let mut deferred_by_terminal =
        (0..num_terminals).map(|_| Vec::<(u32, Arc<RangeSetBlaze<u32>>)>::new()).collect::<Vec<_>>();

    let vocab_tree_started_at = Instant::now();
    let mut tree_entries = Vec::<(usize, &[u8])>::new();
    let mut token_aliases = Vec::<Vec<u32>>::new();
    for (internal_token_id, (_, bytes)) in vocab_order.token_entries_sorted.iter().enumerate() {
        if tree_entries
            .last()
            .is_some_and(|(_, previous_bytes)| *previous_bytes == bytes.as_ref())
        {
            token_aliases
                .last_mut()
                .expect("duplicate token bytes without an alias group")
                .push(internal_token_id as u32);
            continue;
        }
        let alias_group = token_aliases.len();
        tree_entries.push((alias_group, bytes.as_ref()));
        token_aliases.push(vec![internal_token_id as u32]);
    }
    debug_assert!(tree_entries.windows(2).all(|pair| pair[0].1 < pair[1].1));
    let vocab_tree = VocabPrefixTree::build_presorted(&tree_entries);
    let vocab_tree_build_ms = vocab_tree_started_at.elapsed().as_secs_f64() * 1000.0;

    let state_seed_started_at = Instant::now();
    let mut scanner = L1NfaPowerset::new(tokenizer, active_terminals);
    let mut tsids_by_start_config = FxHashMap::<u32, Vec<u32>>::default();
    for (internal_tsid, raw_state) in id_map
        .tokenizer_states
        .iter_representative_ids()
        .enumerate()
    {
        let start_config = scanner.start_config_scoped(raw_state, &id_map.tokenizer_states);
        if start_config != L1_NFA_DEAD_CONFIG {
            tsids_by_start_config
                .entry(start_config)
                .or_default()
                .push(internal_tsid as u32);
        }
    }
    let state_seed_ms = state_seed_started_at.elapsed().as_secs_f64() * 1000.0;

    let traversal_started_at = Instant::now();
    let mut node_visits = 0usize;
    let mut profiles_built = 0usize;
    let mut profile_entries = 0usize;
    for (start_config, tsids) in tsids_by_start_config {
        let mut token_ranges_by_terminal = FxHashMap::<u32, Vec<(u32, u32)>>::default();
        collect_l1_nfa_profile_from_trie(
            &mut scanner,
            &vocab_tree.root,
            start_config,
            &token_aliases,
            &mut token_ranges_by_terminal,
            &mut node_visits,
        );
        profiles_built += 1;
        for (terminal, ranges) in token_ranges_by_terminal {
            let token_set = shared_rangeset(
                ranges
                    .iter()
                    .map(|&(start, end)| start..=end)
                    .collect(),
            );
            if token_set.is_empty() {
                continue;
            }
            profile_entries += tsids.len();
            for &tsid in &tsids {
                deferred_by_terminal[terminal as usize]
                    .push((tsid, Arc::clone(&token_set)));
            }
        }
    }
    let traversal_ms = traversal_started_at.elapsed().as_secs_f64() * 1000.0;
    let token_set_intern_ms = 0.0;
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l1_nfa_powerset_trie] raw_tsids={} start_configs={} configs={} transition_misses={} node_visits={} profile_entries={} unique_token_strings={} token_aliases={} tree_build_ms={:.3} state_seed_ms={:.3} traversal_ms={:.3}",
            tsids_before_merge,
            profiles_built,
            scanner.configs.len(),
            scanner.transition_misses,
            node_visits,
            profile_entries,
            tree_entries.len(),
            vocab_order.token_entries_sorted.len() - tree_entries.len(),
            vocab_tree_build_ms,
            state_seed_ms,
            traversal_ms,
        );
    }

    let merge_started_at = Instant::now();
    let merge_report = merge_deferred_equivalent_tsids(id_map, &mut deferred_by_terminal);
    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;

    let dwa_started_at = Instant::now();
    let mut dwa = DWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let end_state = dwa.add_state();
    dwa.set_final_weight(end_state, Weight::all());
    let mut transition_count = 0usize;
    for (terminal, entries) in deferred_by_terminal.into_iter().enumerate() {
        if entries.is_empty() {
            continue;
        }
        let weight = Weight::from_per_tsid_shared(entries);
        if weight.is_empty() {
            continue;
        }
        dwa.add_transition(dwa.start_state(), terminal as i32, end_state, weight);
        transition_count += 1;
    }
    if transition_count == 0 {
        return None;
    }
    let dwa_ms = dwa_started_at.elapsed().as_secs_f64() * 1000.0;

    Some((
        dwa,
        L1TerminalBuildProfile {
            internal_vocab_ms: 0.0,
            vocab_tree_build_ms,
            state_seed_ms,
            token_set_intern_ms,
            tsid_profile_merge_ms: merge_ms,
            tsid_profile_merge_before: tsids_before_merge,
            tsid_profile_merge_after: merge_report.tsids_after,
            vocab_tree_traversal_ms: traversal_ms,
            direct_terminal_dwa_ms: traversal_ms + merge_ms + dwa_ms,
        },
    ))
}

fn build_l1_id_map<'a>(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &'a Vocab,
    active_terminals: &[bool],
    flat_trans: &Arc<[u32]>,
    transitions_by_byte: Option<&[u32]>,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> (
    InternalIdMap,
    Arc<L1IdentityVocabOrder>,
    Vec<u32>,
    L1IdMapProfile,
    Option<L1ExactProfileReuse>,
) {
    let num_dfa_states = tokenizer.num_states() as usize;
    let states: Vec<usize> = match initial_state_map {
        Some(map) => map
            .representative_original_ids
            .iter()
            .map(|&s| s as usize)
            .collect(),
        None => (0..num_dfa_states).collect(),
    };
    let projected_by_global = initial_state_map.is_some() && states.len() < num_dfa_states;

    if should_use_fast_projected_l1_id_map(
        partition_label,
        vocab.entries_map().len(),
        initial_state_map,
        num_dfa_states,
    ) {
        let token_bytes: Vec<&[u8]> = vocab
            .entries_map()
            .values()
            .map(|bytes| bytes.as_slice())
            .collect();
        let token_len_stats = token_length_stats(&token_bytes);
        let max_token_len = token_bytes
            .iter()
            .map(|bytes| bytes.len())
            .max()
            .unwrap_or(0);
        let (deferred_vocab_singleton_original_ids, vocab_order, token_identity_map_ms) =
            build_l1_identity_vocab_order(vocab);
        let mut tokenizer_states = initial_state_map
            .expect("checked by should_use_fast_projected_l1_id_map")
            .clone();
        if tokenizer.has_scalar_deterministic_dispatch()
            && initial_state_map_contains_original(
                initial_state_map,
                tokenizer.initial_state_id(),
            )
        {
            tokenizer_states.isolate_original(tokenizer.initial_state_id());
        }
        let state_to_rep = state_to_representative_vector(&tokenizer_states, num_dfa_states);
        let exact_reps = tokenizer_states.num_internal_ids() as usize;

        return (
            InternalIdMap {
                tokenizer_states,
                vocab_tokens: ManyToOneIdMap::empty(),
                deferred_vocab_singleton_original_ids: Some(
                    deferred_vocab_singleton_original_ids,
                ),
            },
            vocab_order,
            state_to_rep,
            L1IdMapProfile {
                initial_states_considered: states.len(),
                max_length_skipped: true,
                max_token_len,
                token_len_gt_4: token_len_stats.gt_4,
                token_len_gt_8: token_len_stats.gt_8,
                token_len_gt_16: token_len_stats.gt_16,
                token_len_gt_32: token_len_stats.gt_32,
                token_len_gt_64: token_len_stats.gt_64,
                state_equiv_ms: 0.0,
                max_length_state_equiv_ms: 0.0,
                exact_state_equiv_ms: 0.0,
                max_length_reps: exact_reps,
                exact_reps,
                token_identity_map_ms,
            },
            None,
        );
    }

    // Max-length bounded state equivalence: merge DFA states that behave
    // identically when only tokens up to the max vocab token length are
    // considered. Filtering by active_terminals lets us also merge states
    // that differ only by inactive terminal finalizers/futures.
    let order = l1_identity_vocab_order(vocab);
    let token_id_bytes = order.token_entries_sorted.as_ref();
    let token_len_stats = token_length_stats_from_entries(token_id_bytes);
    let max_token_len = token_id_bytes
        .iter()
        .map(|(_, bytes)| bytes.len())
        .max()
        .unwrap_or(0);
    let use_remaining_horizon_quotients =
        l1_remaining_horizon_quotients_enabled(states.len(), token_id_bytes.len());
    let max_length_skipped = use_remaining_horizon_quotients
        || should_skip_max_length_for_partition(partition_label, states.len(), projected_by_global);
    let state_equiv_started_at = Instant::now();
    let mut view_ms = 0.0;
    let equiv_mapping = if max_length_skipped {
        states.clone()
    } else {
        let tokenizer_view =
            TokenizerView::new_filtered_from_flat_trans(flat_trans, tokenizer, active_terminals);
        view_ms = state_equiv_started_at.elapsed().as_secs_f64() * 1000.0;
        let token_bytes: Vec<&[u8]> = token_id_bytes
            .iter()
            .map(|(_, bytes)| bytes.as_ref())
            .collect();
        let mut relevant_bytes = [false; 256];
        for bytes in &token_bytes {
            for &byte in *bytes {
                relevant_bytes[byte as usize] = true;
            }
        }
        let byte_to_class = compute_byte_classes(tokenizer_view.dfa());
        super::l2p::equivalence_analysis::state::max_length::find_state_equivalence_classes_byte_restricted(
            &tokenizer_view,
            &token_bytes,
            &states,
            Some(&byte_to_class),
            Some(active_terminals),
            Some(&relevant_bytes),
        )
    };

    // Token IDs are first-byte bucketed and lexicographic within each bucket.
    // This preserves cheap whole-bucket unions and maximizes LCP reuse in the
    // exact suffix-profile walks used by both equivalence and terminal-DWA build.
    let token_sort_ms = 0.0;

    let max_length_ms = if max_length_skipped {
        0.0
    } else {
        state_equiv_started_at.elapsed().as_secs_f64() * 1000.0 - view_ms - token_sort_ms
    };
    let exact_started_at = Instant::now();
    let mut max_length_representatives = equiv_mapping.clone();
    max_length_representatives.sort_unstable();
    max_length_representatives.dedup();
    let (exact_mapping, mut exact_profile_reuse) =
        find_l1_exact_state_equivalence_by_token_signatures(
            tokenizer,
            order.as_ref(),
            &max_length_representatives,
            active_terminals,
            flat_trans,
            transitions_by_byte,
        );
    let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
    let mut max_rep_to_exact_rep = FxHashMap::<usize, usize>::default();
    for (&max_rep, &exact_rep) in max_length_representatives.iter().zip(exact_mapping.iter()) {
        max_rep_to_exact_rep.insert(max_rep, exact_rep);
    }

    // Build representative → internal_id mapping, composing through initial_state_map when present
    let mut rep_to_internal: FxHashMap<usize, u32> = FxHashMap::default();
    let mut state_original_to_internal = vec![u32::MAX; num_dfa_states];
    let mut state_representatives = Vec::new();
    for (i, &rep) in equiv_mapping.iter().enumerate() {
        let state_id = states[i];
        let exact_rep = max_rep_to_exact_rep[&rep];
        let internal_id = *rep_to_internal.entry(exact_rep).or_insert_with(|| {
            let id = state_representatives.len() as u32;
            state_representatives.push(exact_rep as u32);
            id
        });
        state_original_to_internal[state_id] = internal_id;
    }

    // When initial_state_map is present, compose: all DFA states map through
    // initial_state_map → max-length equivalence
    if let Some(init_map) = initial_state_map {
        for (orig_state, &init_internal) in init_map.original_to_internal.iter().enumerate() {
            if init_internal == u32::MAX
                || (init_internal as usize) >= init_map.representative_original_ids.len()
            {
                continue;
            }
            let init_rep = init_map.representative_original_ids[init_internal as usize] as usize;
            let final_internal = state_original_to_internal[init_rep];
            if final_internal != u32::MAX {
                state_original_to_internal[orig_state] = final_internal;
            }
        }
    }

    // Keep this exact: the L1 terminal DWA indexes weights by TSID, so every
    // original state in a TSID must have the same whole-token terminal
    // signature for every token. Sampling tokens here is not a proof and has
    // caused order-sensitive mask/commit mismatches.

    let state_equiv_ms = state_equiv_started_at.elapsed().as_secs_f64() * 1000.0;

    let token_map_started_at = Instant::now();
    let deferred_vocab_singleton_original_ids = Arc::clone(&order.token_ids_sorted);
    let token_identity_map_ms =
        token_sort_ms + token_map_started_at.elapsed().as_secs_f64() * 1000.0;
    if tokenizer.has_scalar_deterministic_dispatch()
        && let Some(reuse) = exact_profile_reuse.as_mut()
    {
        reuse.profile_representatives_by_internal = Arc::from(state_representatives.clone());
    }
    let mut tokenizer_states = ManyToOneIdMap::from_original_to_internal_with_representatives(
        state_original_to_internal,
        state_representatives.len() as u32,
        state_representatives,
    );
    if tokenizer.has_scalar_deterministic_dispatch()
        && initial_state_map_contains_original(initial_state_map, tokenizer.initial_state_id())
    {
        tokenizer_states.isolate_original(tokenizer.initial_state_id());
    }
    let state_to_rep = state_to_representative_vector(&tokenizer_states, num_dfa_states);
    let exact_reps = tokenizer_states.num_internal_ids() as usize;

    (
        InternalIdMap {
            tokenizer_states,
            vocab_tokens: ManyToOneIdMap::empty(),
            deferred_vocab_singleton_original_ids: Some(deferred_vocab_singleton_original_ids),
        },
        order,
        state_to_rep,
        L1IdMapProfile {
            initial_states_considered: states.len(),
            max_length_skipped,
            max_token_len,
            token_len_gt_4: token_len_stats.gt_4,
            token_len_gt_8: token_len_stats.gt_8,
            token_len_gt_16: token_len_stats.gt_16,
            token_len_gt_32: token_len_stats.gt_32,
            token_len_gt_64: token_len_stats.gt_64,
            state_equiv_ms,
            max_length_state_equiv_ms: max_length_ms,
            exact_state_equiv_ms,
            max_length_reps: max_length_representatives.len(),
            exact_reps,
            token_identity_map_ms,
        },
        exact_profile_reuse,
    )
}

fn build_l1_identity_vocab_order(
    vocab: &Vocab,
) -> (Arc<[u32]>, Arc<L1IdentityVocabOrder>, f64) {
    let token_identity_started_at = Instant::now();
    let order = l1_identity_vocab_order(vocab);
    let token_ids_sorted = Arc::clone(&order.token_ids_sorted);

    let token_identity_map_ms = token_identity_started_at.elapsed().as_secs_f64() * 1000.0;
    (token_ids_sorted, order, token_identity_map_ms)
}

fn state_to_representative_vector(state_map: &ManyToOneIdMap, num_dfa_states: usize) -> Vec<u32> {
    let mut state_to_rep = vec![0u32; num_dfa_states];
    for (state_id, &internal) in state_map.original_to_internal.iter().enumerate() {
        if internal != u32::MAX {
            if let Some(&rep) = state_map.representative_original_ids.get(internal as usize) {
                state_to_rep[state_id] = rep;
            }
        }
    }
    state_to_rep
}

struct TokenLengthStats {
    gt_4: usize,
    gt_8: usize,
    gt_16: usize,
    gt_32: usize,
    gt_64: usize,
}

fn token_length_stats(tokens: &[&[u8]]) -> TokenLengthStats {
    let mut stats = TokenLengthStats {
        gt_4: 0,
        gt_8: 0,
        gt_16: 0,
        gt_32: 0,
        gt_64: 0,
    };
    for token in tokens {
        let len = token.len();
        if len > 4 {
            stats.gt_4 += 1;
        }
        if len > 8 {
            stats.gt_8 += 1;
        }
        if len > 16 {
            stats.gt_16 += 1;
        }
        if len > 32 {
            stats.gt_32 += 1;
        }
        if len > 64 {
            stats.gt_64 += 1;
        }
    }
    stats
}

fn token_length_stats_from_entries(tokens: &[(u32, Arc<[u8]>)]) -> TokenLengthStats {
    let mut stats = TokenLengthStats {
        gt_4: 0,
        gt_8: 0,
        gt_16: 0,
        gt_32: 0,
        gt_64: 0,
    };
    for (_, token) in tokens {
        let len = token.len();
        if len > 4 {
            stats.gt_4 += 1;
        }
        if len > 8 {
            stats.gt_8 += 1;
        }
        if len > 16 {
            stats.gt_16 += 1;
        }
        if len > 32 {
            stats.gt_32 += 1;
        }
        if len > 64 {
            stats.gt_64 += 1;
        }
    }
    stats
}

#[inline]
fn l1_canonicalize_target(target: u32, canonical_state: Option<&[u32]>) -> u32 {
    if target == u32::MAX {
        target
    } else {
        canonical_state.map_or(target, |map| map[target as usize])
    }
}

#[inline]
fn l1_transition(
    flat_trans: &[u32],
    transitions_by_byte: Option<&[u32]>,
    num_tokenizer_states: usize,
    active_language: Option<&[bool]>,
    state: u32,
    byte: usize,
    canonical_state: Option<&[u32]>,
) -> u32 {
    if active_language.is_some_and(|active| !active[state as usize]) {
        return u32::MAX;
    }
    let target = if let Some(transitions_by_byte) = transitions_by_byte {
        transitions_by_byte[byte * num_tokenizer_states + state as usize]
    } else {
        flat_trans[state as usize * 256 + byte]
    };
    let target = if target != u32::MAX
        && active_language.is_some_and(|active| !active[target as usize])
    {
        u32::MAX
    } else {
        target
    };
    l1_canonicalize_target(target, canonical_state)
}

fn collect_l1_unique_first_targets(
    states: &[usize],
    nonempty_first_bytes: &[usize],
    suffix_horizon_by_first_byte: &[usize; 256],
    horizon_maps: Option<&[Arc<[u32]>]>,
    flat_trans: &[u32],
    transitions_by_byte: Option<&[u32]>,
    num_tokenizer_states: usize,
    active_language: &[bool],
    cache_targets: bool,
) -> (Vec<(u8, u32)>, Option<Vec<u32>>) {
    let dead = u32::MAX;
    let target_words = num_tokenizer_states.div_ceil(64);
    let row_major = transitions_by_byte.is_none()
        && horizon_maps.is_none()
        && !cache_targets
        && rayon::current_num_threads() > 1
        && states.len() >= 4096;
    if row_major {
        // The historical byte-major loop touches one u32 every 1 KiB in the
        // row-major DFA table. Build small thread-local target bitsets while
        // reading each transition row contiguously, then OR them once. The
        // result is the same exact per-byte target set; this path deliberately
        // declines when callers request the full cached target matrix.
        let threads = rayon::current_num_threads().min(states.len());
        let chunk_size = states.len().div_ceil(threads).max(1);
        let local_seen = states
            .par_chunks(chunk_size)
            .map(|state_chunk| {
                let mut seen = vec![0u64; nonempty_first_bytes.len() * target_words];
                for &state in state_chunk {
                    if !active_language[state] {
                        continue;
                    }
                    let row = &flat_trans[state * 256..state * 256 + 256];
                    for (slot, &byte) in nonempty_first_bytes.iter().enumerate() {
                        let target = row[byte];
                        if target == dead || !active_language[target as usize] {
                            continue;
                        }
                        let target = target as usize;
                        seen[slot * target_words + (target >> 6)] |=
                            1u64 << (target & 63);
                    }
                }
                seen
            })
            .collect::<Vec<_>>();
        let mut target_seen = vec![0u64; nonempty_first_bytes.len() * target_words];
        for local in local_seen {
            for (combined, local) in target_seen.iter_mut().zip(local) {
                *combined |= local;
            }
        }
        let mut unique_targets = Vec::<(u8, u32)>::new();
        for (slot, &byte) in nonempty_first_bytes.iter().enumerate() {
            let words = &target_seen[slot * target_words..(slot + 1) * target_words];
            for (word, &bits) in words.iter().enumerate() {
                let mut bits = bits;
                let base = (word * 64) as u32;
                while bits != 0 {
                    let offset = bits.trailing_zeros();
                    unique_targets.push((byte as u8, base + offset));
                    bits &= bits - 1;
                }
            }
        }
        return (unique_targets, None);
    }
    let mut target_seen = vec![0u64; target_words];
    let mut unique_targets = Vec::<(u8, u32)>::new();
    let mut cached_targets = cache_targets.then(|| {
        Vec::<u32>::with_capacity(states.len().saturating_mul(nonempty_first_bytes.len()))
    });
    for &byte in nonempty_first_bytes {
        let canonical_state = horizon_maps
            .map(|maps| maps[suffix_horizon_by_first_byte[byte]].as_ref());
        for &state in states {
            let target = l1_transition(
                flat_trans,
                transitions_by_byte,
                num_tokenizer_states,
                Some(active_language),
                state as u32,
                byte,
                canonical_state,
            );
            if let Some(cached_targets) = cached_targets.as_mut() {
                cached_targets.push(target);
            }
            if target != dead {
                target_seen[target as usize >> 6] |= 1u64 << (target & 63);
            }
        }
        for word in 0..target_words {
            let mut bits = target_seen[word];
            if bits == 0 {
                continue;
            }
            target_seen[word] = 0;
            let base = (word * 64) as u32;
            while bits != 0 {
                let offset = bits.trailing_zeros();
                unique_targets.push((byte as u8, base + offset));
                bits &= bits - 1;
            }
        }
    }
    (unique_targets, cached_targets)
}

fn find_l1_exact_state_equivalence_by_token_signatures(
    tokenizer: &Tokenizer,
    vocab_order: &L1IdentityVocabOrder,
    states: &[usize],
    active_terminals: &[bool],
    flat_trans: &Arc<[u32]>,
    transitions_by_byte: Option<&[u32]>,
) -> (Vec<usize>, Option<L1ExactProfileReuse>) {
    find_l1_exact_state_equivalence_by_token_signatures_with_first_target_cache(
        tokenizer,
        vocab_order,
        states,
        active_terminals,
        flat_trans,
        transitions_by_byte,
        None,
    )
}

fn find_l1_exact_state_equivalence_by_token_signatures_with_first_target_cache(
    tokenizer: &Tokenizer,
    vocab_order: &L1IdentityVocabOrder,
    states: &[usize],
    active_terminals: &[bool],
    flat_trans: &Arc<[u32]>,
    transitions_by_byte: Option<&[u32]>,
    first_target_cache_override: Option<bool>,
) -> (Vec<usize>, Option<L1ExactProfileReuse>) {
    let terminal_signature_started_at = compile_profile_enabled().then(Instant::now);
    let (state_to_terminal_signature, terminal_signatures, active_language) =
        build_l1_tokenizer_state_to_terminal_signatures(tokenizer, active_terminals);
    let terminal_signature_ms = terminal_signature_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    // `l1_transition` maps inactive sources and targets to dead before the
    // packed builder consults this table. Therefore every non-dead state whose
    // self-loops are queried is active, and its raw self-loop bytes remain
    // exact under terminal projection. Reuse the tokenizer-wide cached table
    // instead of rescanning all 256 bytes for every state in every partition.
    let self_loop_bytes_by_state = tokenizer.all_self_loop_bytes();
    find_l1_exact_state_equivalence_by_components_with_first_target_cache(
        vocab_order,
        states,
        state_to_terminal_signature,
        terminal_signatures,
        flat_trans,
        &active_language,
        None,
        Some((tokenizer, active_terminals)),
        transitions_by_byte,
        true,
        terminal_signature_ms,
        first_target_cache_override,
        Some(self_loop_bytes_by_state.as_ref()),
        None,
    )
    .expect("ordinary exact L1 equivalence cannot decline")
}

fn find_l1_exact_state_equivalence_by_flat_signatures(
    vocab_order: &L1IdentityVocabOrder,
    states: &[usize],
    state_to_terminal_signature: Vec<u32>,
    terminal_signatures: Vec<Vec<u32>>,
    tokenizer_view: &TokenizerView,
    transitions_by_byte: Option<&[u32]>,
    allow_remaining_horizon_quotients: bool,
    terminal_signature_ms: f64,
    self_loop_bytes_by_state: Option<&[U8Set]>,
) -> (Vec<usize>, Option<L1ExactProfileReuse>) {
    find_l1_exact_state_equivalence_by_flat_signatures_maybe_decline(
        vocab_order,
        states,
        state_to_terminal_signature,
        terminal_signatures,
        tokenizer_view,
        transitions_by_byte,
        allow_remaining_horizon_quotients,
        terminal_signature_ms,
        self_loop_bytes_by_state,
        None,
    )
    .expect("ordinary exact L1 equivalence cannot decline")
}

fn find_l1_exact_state_equivalence_by_flat_signatures_maybe_decline(
    vocab_order: &L1IdentityVocabOrder,
    states: &[usize],
    state_to_terminal_signature: Vec<u32>,
    terminal_signatures: Vec<Vec<u32>>,
    tokenizer_view: &TokenizerView,
    transitions_by_byte: Option<&[u32]>,
    allow_remaining_horizon_quotients: bool,
    terminal_signature_ms: f64,
    cached_self_loop_bytes_by_state: Option<&[U8Set]>,
    max_raw_unique_targets: Option<usize>,
) -> Result<(Vec<usize>, Option<L1ExactProfileReuse>), ()> {
    find_l1_exact_state_equivalence_by_flat_signatures_with_first_target_cache_maybe_decline(
        vocab_order,
        states,
        state_to_terminal_signature,
        terminal_signatures,
        tokenizer_view,
        transitions_by_byte,
        allow_remaining_horizon_quotients,
        terminal_signature_ms,
        None,
        cached_self_loop_bytes_by_state,
        max_raw_unique_targets,
    )
}

fn find_l1_exact_state_equivalence_by_flat_signatures_with_first_target_cache(
    vocab_order: &L1IdentityVocabOrder,
    states: &[usize],
    state_to_terminal_signature: Vec<u32>,
    terminal_signatures: Vec<Vec<u32>>,
    tokenizer_view: &TokenizerView,
    transitions_by_byte: Option<&[u32]>,
    allow_remaining_horizon_quotients: bool,
    terminal_signature_ms: f64,
    first_target_cache_override: Option<bool>,
    cached_self_loop_bytes_by_state: Option<&[U8Set]>,
) -> (Vec<usize>, Option<L1ExactProfileReuse>) {
    find_l1_exact_state_equivalence_by_flat_signatures_with_first_target_cache_maybe_decline(
        vocab_order,
        states,
        state_to_terminal_signature,
        terminal_signatures,
        tokenizer_view,
        transitions_by_byte,
        allow_remaining_horizon_quotients,
        terminal_signature_ms,
        first_target_cache_override,
        cached_self_loop_bytes_by_state,
        None,
    )
    .expect("ordinary exact L1 equivalence cannot decline")
}

fn find_l1_exact_state_equivalence_by_flat_signatures_with_first_target_cache_maybe_decline(
    vocab_order: &L1IdentityVocabOrder,
    states: &[usize],
    state_to_terminal_signature: Vec<u32>,
    terminal_signatures: Vec<Vec<u32>>,
    tokenizer_view: &TokenizerView,
    transitions_by_byte: Option<&[u32]>,
    allow_remaining_horizon_quotients: bool,
    terminal_signature_ms: f64,
    first_target_cache_override: Option<bool>,
    cached_self_loop_bytes_by_state: Option<&[U8Set]>,
    max_raw_unique_targets: Option<usize>,
) -> Result<(Vec<usize>, Option<L1ExactProfileReuse>), ()> {
    let dfa = tokenizer_view.dfa();
    let active_language = dfa
        .states
        .iter()
        .map(|state| {
            !state.finalizers.is_empty() || !state.possible_future_group_ids.is_empty()
        })
        .collect::<Vec<_>>();
    find_l1_exact_state_equivalence_by_components_with_first_target_cache(
        vocab_order,
        states,
        state_to_terminal_signature,
        terminal_signatures,
        dfa.transitions.as_ref(),
        &active_language,
        Some(tokenizer_view),
        None,
        transitions_by_byte,
        allow_remaining_horizon_quotients,
        terminal_signature_ms,
        first_target_cache_override,
        cached_self_loop_bytes_by_state,
        max_raw_unique_targets,
    )
}

fn find_l1_exact_state_equivalence_by_components_with_first_target_cache(
    vocab_order: &L1IdentityVocabOrder,
    states: &[usize],
    state_to_terminal_signature: Vec<u32>,
    terminal_signatures: Vec<Vec<u32>>,
    flat_trans: &[u32],
    active_language: &[bool],
    horizon_tokenizer_view: Option<&TokenizerView>,
    horizon_source: Option<(&Tokenizer, &[bool])>,
    transitions_by_byte: Option<&[u32]>,
    allow_remaining_horizon_quotients: bool,
    terminal_signature_ms: f64,
    first_target_cache_override: Option<bool>,
    cached_self_loop_bytes_by_state: Option<&[U8Set]>,
    max_raw_unique_targets: Option<usize>,
) -> Result<(Vec<usize>, Option<L1ExactProfileReuse>), ()> {
    if states.len() <= 1 {
        return Ok((states.to_vec(), None));
    }
    let profile_enabled = compile_profile_enabled();
    let total_started_at = profile_enabled.then(Instant::now);

    // Exact L1 equivalence has a useful factorization.  For a non-empty token
    // b·suffix, the contribution of a start state s depends on s only through
    // delta(s, b); the rest of the token walk is shared by every state with the
    // same first-byte target.  The old implementation re-walked every token for
    // every candidate state.  Here we precompute each distinct
    // (first_byte, first_target) suffix profile once, intern those profiles, and
    // then classify a start state by the small vector of profile IDs reached by
    // its first-byte transitions.
    let sorted_entries = vocab_order.token_entries_sorted.as_ref();
    let token_buckets = &vocab_order.token_buckets;
    let dead = u32::MAX;
    let active_language = active_language.to_vec();
    let num_tokenizer_states = state_to_terminal_signature.len();
    debug_assert_eq!(state_to_terminal_signature.len(), num_tokenizer_states);

    let mut suffix_horizon_by_first_byte = [0usize; 256];
    let mut max_token_len = 0usize;
    let mut relevant_bytes = [false; 256];
    for (byte, token_ids) in token_buckets.token_indices_by_first_byte.iter().enumerate() {
        for &token_id in token_ids {
            let bytes = sorted_entries[token_id].1.as_ref();
            max_token_len = max_token_len.max(bytes.len());
            suffix_horizon_by_first_byte[byte] =
                suffix_horizon_by_first_byte[byte].max(bytes.len().saturating_sub(1));
            for &token_byte in bytes {
                relevant_bytes[token_byte as usize] = true;
            }
        }
    }
    for &token_id in &token_buckets.empty_token_indices {
        let bytes = sorted_entries[token_id].1.as_ref();
        max_token_len = max_token_len.max(bytes.len());
    }

    let nonempty_first_bytes: Vec<usize> = token_buckets
        .token_indices_by_first_byte
        .iter()
        .enumerate()
        .filter_map(|(byte, token_ids)| (!token_ids.is_empty()).then_some(byte))
        .collect();
    // The raw-frontier probe has already paid for every state × first-byte
    // transition. Retaining that compact u32 matrix avoids repeating the same
    // multi-million-cell scan while assembling exact state keys. The large p2
    // diff workloads need roughly 21–26 MiB, which is modest relative to their
    // tokenizer and terminal-DWA working sets.
    const FIRST_TARGET_CACHE_MAX_BYTES: usize = 32 * 1024 * 1024;
    let first_target_cache_bytes = states
        .len()
        .checked_mul(nonempty_first_bytes.len())
        .and_then(|cells| cells.checked_mul(std::mem::size_of::<u32>()));
    let cache_first_targets = first_target_cache_override.unwrap_or_else(|| {
        first_target_cache_bytes.is_some_and(|bytes| bytes <= FIRST_TARGET_CACHE_MAX_BYTES)
    });

    // The finite-horizon quotient prepass is only worthwhile when it avoids a
    // genuinely large number of suffix profiles.  A state×vocab threshold alone
    // cannot see whether the raw first-byte frontier is already compact.  Count
    // that frontier first: the probe is a contiguous byte-major scan and can be
    // reused directly when the quotient is rejected.
    let raw_target_probe_started_at = profile_enabled.then(Instant::now);
    let (raw_unique_targets, raw_first_targets) = collect_l1_unique_first_targets(
        states,
        &nonempty_first_bytes,
        &suffix_horizon_by_first_byte,
        None,
        flat_trans,
        transitions_by_byte,
        num_tokenizer_states,
        &active_language,
        cache_first_targets,
    );
    let raw_unique_targets_len = raw_unique_targets.len();
    let raw_target_probe_ms = raw_target_probe_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    if max_raw_unique_targets.is_some_and(|max_targets| raw_unique_targets_len > max_targets) {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][l1_exact_equiv_decline] raw_unique_targets={} max_raw_unique_targets={} raw_target_probe_ms={:.3}",
                raw_unique_targets_len,
                max_raw_unique_targets.expect("checked above"),
                raw_target_probe_ms,
            );
        }
        return Err(());
    }
    let use_remaining_horizon_quotients = allow_remaining_horizon_quotients
        && l1_remaining_horizon_quotients_enabled(states.len(), sorted_entries.len())
        && l1_remaining_horizon_target_probe_supports_quotients(raw_unique_targets_len);
    let horizon_quotient_started_at = profile_enabled.then(Instant::now);
    let owned_horizon_view = if use_remaining_horizon_quotients
        && horizon_tokenizer_view.is_none()
    {
        let (tokenizer, active_terminals) =
            horizon_source.expect("an L1 horizon source is required");
        Some(TokenizerView::new_filtered(tokenizer, active_terminals))
    } else {
        None
    };
    let horizon_view = use_remaining_horizon_quotients.then(|| {
        horizon_tokenizer_view
            .or(owned_horizon_view.as_ref())
            .expect("L1 horizon tokenizer view must exist")
    });
    let horizon_maps = horizon_view.map(|tokenizer_view| {
        let byte_to_class = compute_byte_classes(tokenizer_view.dfa());
        super::l2p::equivalence_analysis::state::max_length::find_canonical_state_maps_by_depth_from_labels(
            tokenizer_view,
            max_token_len,
            &state_to_terminal_signature,
            Some(&relevant_bytes),
            Some(&byte_to_class),
        )
    });
    let horizon_quotient_ms = horizon_quotient_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    if profile_enabled
        && let Some(horizon_maps) = horizon_maps.as_ref()
    {
        let depths = [0usize, 1, 2, 3, 4, 8, 16, 32, 63, 64];
        let counts = depths
            .iter()
            .filter(|&&depth| depth < horizon_maps.len())
            .map(|&depth| {
                let mut reps = rustc_hash::FxHashSet::default();
                reps.extend(horizon_maps[depth].iter().copied());
                format!("{}:{}", depth, reps.len())
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "[glrmask/profile][l1_terminal_horizon_quotients] states={} max_token_len={} depth_reps={}",
            num_tokenizer_states,
            max_token_len,
            counts,
        );
    }

    let unique_targets_started_at = profile_enabled.then(Instant::now);
    let (unique_targets, first_targets) = if let Some(horizon_maps) = horizon_maps.as_deref() {
        collect_l1_unique_first_targets(
            states,
            &nonempty_first_bytes,
            &suffix_horizon_by_first_byte,
            Some(horizon_maps),
            flat_trans,
            transitions_by_byte,
            num_tokenizer_states,
            &active_language,
            cache_first_targets,
        )
    } else {
        (raw_unique_targets, raw_first_targets)
    };
    let unique_targets_len = unique_targets.len();
    let unique_targets_ms = unique_targets_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });

    let target_profiles_started_at = profile_enabled.then(Instant::now);
    let owned_self_loop_bytes_by_state;
    let self_loop_bytes_by_state = if let Some(cached) = cached_self_loop_bytes_by_state {
        cached
    } else {
        owned_self_loop_bytes_by_state = (0..num_tokenizer_states)
            .map(|state| {
                let row = &flat_trans[state * 256..state * 256 + 256];
                U8Set::from_predicate(|byte| row[byte as usize] == state as u32)
            })
            .collect::<Vec<_>>();
        owned_self_loop_bytes_by_state.as_slice()
    };
    let mut targets_by_first_byte = vec![Vec::<u32>::new(); 256];
    for (byte, target) in unique_targets {
        targets_by_first_byte[byte as usize].push(target);
    }
    let byte_target_groups: Vec<(u8, Vec<u32>)> = targets_by_first_byte
        .into_iter()
        .enumerate()
        .filter_map(|(byte, targets)| (!targets.is_empty()).then_some((byte as u8, targets)))
        .collect();
    let flat_profiles_supported = transitions_by_byte.is_none() && horizon_maps.is_none();
    let build_byte_profiles = |byte: u8, targets: &[u32]| {
        let byte_idx = byte as usize;
        let token_ids = &token_buckets.token_indices_by_first_byte[byte_idx];
        let suffix_lcps = &token_buckets.suffix_lcps_by_first_byte[byte_idx];
        const LARGE_BUCKET_WORK_PRODUCT: usize = 1_000_000;
        let estimated_work = token_ids.len().saturating_mul(targets.len());
        let flat_supported = flat_profiles_supported;
        let prefer_flat_empty_suffix = flat_supported
            && l1_prefers_flat_utf8_lead_bucket(
                byte,
                token_ids.len(),
                targets.len(),
                token_buckets.has_empty_suffix_by_bucket[byte_idx],
            );
        let parallel_bucket = token_ids.len() >= 10_000
            || estimated_work >= LARGE_BUCKET_WORK_PRODUCT;
        let requested = std::env::var("GLRMASK_L1_PROFILE_BUILDER")
            .unwrap_or_else(|_| "auto".to_owned());
        let flat_supported = transitions_by_byte.is_none() && horizon_maps.is_none();
        // Large vocabulary partitions already expose ample parallelism across
        // first-byte buckets. Their flat batched walks retain contiguous
        // transition/token locality, while nested packed-product chunking adds
        // trie construction, allocator, and cross-chunk canonicalization work.
        // Apply this choice before the packed large-bucket branch so both the
        // automatic policy and the explicit `flat` diagnostic override are
        // actually respected.
        // Large vocabularies often favor flat scans, but not when one bucket's
        // token-by-target work is enormous. In that regime the packed/chunked
        // scheduler amortizes suffix sharing and avoids a full Cartesian walk.
        const LARGE_PARTITION_FLAT_MAX_WORK: usize = 8_000_000;
        let large_partition_prefers_flat = sorted_entries.len() >= 20_000
            && estimated_work <= LARGE_PARTITION_FLAT_MAX_WORK;
        if flat_supported
            && (requested == "flat"
                || (requested == "auto" && large_partition_prefers_flat))
        {
            return l1_bucket_suffix_signature_profiles_batched_arc(
                byte,
                targets,
                sorted_entries,
                token_ids,
                suffix_lcps,
                &token_buckets.suffix_subtree_bytes[byte_idx],
                &token_buckets.suffix_first_bytes_by_bucket[byte_idx],
                token_buckets.has_empty_suffix_by_bucket[byte_idx],
                &state_to_terminal_signature,
                flat_trans,
                transitions_by_byte,
                num_tokenizer_states,
                Some(&active_language),
            );
        }

        if parallel_bucket
            && !prefer_flat_empty_suffix
            && targets.len() >= 32
            && rayon::current_num_threads() > 1
        {
            let prebuilt_trie = token_buckets.packed_suffix_tries_by_first_byte[byte_idx]
                .get_or_init(|| {
                    Arc::new(L1PackedSuffixTrie::build(
                        sorted_entries,
                        token_ids,
                        suffix_lcps,
                    ))
                });
            let worker_count = rayon::current_num_threads();
            let default_chunk_count = if token_ids.len() >= 10_000 {
                worker_count.saturating_mul(3).min(24)
            } else {
                worker_count.min(8)
            };
            let chunk_count = default_chunk_count
                .min(if token_ids.len() >= 10_000 {
                    targets.len()
                } else {
                    worker_count
                })
                .min(targets.len());
            let chunk_size = targets.len().div_ceil(chunk_count);
            let build_chunk = |target_chunk: &[u32]| {
                l1_bucket_suffix_signature_profiles_packed(
                    byte,
                    target_chunk,
                    sorted_entries,
                    token_ids,
                    suffix_lcps,
                    &token_buckets.suffix_subtree_bytes[byte_idx],
                    &token_buckets.suffix_first_bytes_by_bucket[byte_idx],
                    token_buckets.has_empty_suffix_by_bucket[byte_idx],
                    &state_to_terminal_signature,
                    self_loop_bytes_by_state,
                    flat_trans,
                    transitions_by_byte,
                    num_tokenizer_states,
                    Some(&active_language),
                    horizon_maps.as_deref(),
                    suffix_horizon_by_first_byte[byte_idx],
                    Some(prebuilt_trie.as_ref()),
                )
            };
            const BLOCK_INTERLEAVE_SIZE: usize = 512;
            let mut chunks = (0..chunk_count)
                .map(|_| Vec::with_capacity(chunk_size))
                .collect::<Vec<_>>();
            for (block_index, block) in targets.chunks(BLOCK_INTERLEAVE_SIZE).enumerate() {
                chunks[block_index % chunk_count].extend_from_slice(block);
            }
            let chunk_build_started_at = profile_enabled.then(Instant::now);
            let mut chunked_profiles: Vec<((u8, u32), Arc<[(u32, u32, u32)]>)> = chunks
                .par_iter()
                .map(|target_chunk| build_chunk(target_chunk))
                .flatten()
                .collect();
            let chunk_build_ms = chunk_build_started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            // The packed builder interns equal behaviors only within one target
            // batch.  Chunking a large bucket for latency must not turn equal
            // cross-chunk profiles into distinct pointer identities, because
            // the outer exact-equivalence pass deliberately uses canonical Arc
            // identity as its O(1) profile key.
            let canonicalize_started_at = profile_enabled.then(Instant::now);
            let mut canonical_profiles =
                FxHashMap::<Arc<[(u32, u32, u32)]>, Arc<[(u32, u32, u32)]>>::default();
            for (_, profile) in &mut chunked_profiles {
                if profile.is_empty() {
                    continue;
                }
                if let Some(canonical) = canonical_profiles.get(profile) {
                    *profile = Arc::clone(canonical);
                } else {
                    canonical_profiles.insert(Arc::clone(profile), Arc::clone(profile));
                }
            }
            if let Some(canonicalize_started_at) = canonicalize_started_at {
                eprintln!(
                    "[glrmask/profile][l1_chunked_profiles] first_byte={} chunks={} targets={} chunk_build_ms={:.3} canonicalize_ms={:.3} canonical_profiles={}",
                    byte,
                    chunk_count,
                    targets.len(),
                    chunk_build_ms,
                    canonicalize_started_at.elapsed().as_secs_f64() * 1000.0,
                    canonical_profiles.len(),
                );
            }
            chunked_profiles
        } else {
            let build_flat = || {
                l1_bucket_suffix_signature_profiles_batched_arc(
                    byte,
                    targets,
                    sorted_entries,
                    token_ids,
                    suffix_lcps,
                    &token_buckets.suffix_subtree_bytes[byte_idx],
                    &token_buckets.suffix_first_bytes_by_bucket[byte_idx],
                    token_buckets.has_empty_suffix_by_bucket[byte_idx],
                    &state_to_terminal_signature,
                    flat_trans,
                    transitions_by_byte,
                    num_tokenizer_states,
                    Some(&active_language),
                )
            };
            let build_packed = || {
                let prebuilt_trie = token_buckets.packed_suffix_tries_by_first_byte[byte_idx]
                    .get_or_init(|| {
                        Arc::new(L1PackedSuffixTrie::build(
                            sorted_entries,
                            token_ids,
                            suffix_lcps,
                        ))
                    });
                l1_bucket_suffix_signature_profiles_packed(
                    byte,
                    targets,
                    sorted_entries,
                    token_ids,
                    suffix_lcps,
                    &token_buckets.suffix_subtree_bytes[byte_idx],
                    &token_buckets.suffix_first_bytes_by_bucket[byte_idx],
                    token_buckets.has_empty_suffix_by_bucket[byte_idx],
                    &state_to_terminal_signature,
                    self_loop_bytes_by_state,
                    flat_trans,
                    transitions_by_byte,
                    num_tokenizer_states,
                    Some(&active_language),
                    horizon_maps.as_deref(),
                    suffix_horizon_by_first_byte[byte_idx],
                    Some(prebuilt_trie.as_ref()),
                )
            };
            // On a large vocabulary partition, flat target walks retain
            // excellent byte/token locality across the many first-byte
            // buckets. Building and traversing a separate packed suffix
            // product for medium buckets instead creates enough allocator and
            // interning traffic to dominate the partition. Reserve the packed
            // auto choice for smaller partitions where its trie compression
            // amortizes that setup cost.
            let auto_prefers_packed = !large_partition_prefers_flat
                && token_ids.len() > 500
                && token_ids.len() <= 7_000
                && token_buckets.packed_suffix_tries_by_first_byte[byte_idx]
                    .get_or_init(|| {
                        Arc::new(L1PackedSuffixTrie::build(
                            sorted_entries,
                            token_ids,
                            suffix_lcps,
                        ))
                    })
                    .nodes
                    .len()
                    > 4_000;
            let use_flat = flat_supported
                && (requested == "flat"
                    || prefer_flat_empty_suffix
                    || (requested == "auto" && !auto_prefers_packed));
            if use_flat { build_flat() } else { build_packed() }
        }
    };
    let target_profile_batches: Vec<Vec<((u8, u32), Arc<[(u32, u32, u32)]>)>> =
        if rayon::current_num_threads() == 1 {
            byte_target_groups
                .iter()
                .map(|(byte, targets)| build_byte_profiles(*byte, targets))
                .collect()
        } else {
            let dominant = byte_target_groups
                .iter()
                .enumerate()
                .max_by_key(|(_, (byte, _))| {
                    token_buckets.token_indices_by_first_byte[*byte as usize].len()
                })
                .filter(|(_, (byte, _))| {
                    token_buckets.token_indices_by_first_byte[*byte as usize].len() >= 10_000
                })
                .map(|(index, _)| index);
            let dominant_started_at = profile_enabled.then(Instant::now);
            let dominant_batch = dominant.map(|index| {
                let (byte, targets) = &byte_target_groups[index];
                build_byte_profiles(*byte, targets)
            });
            let dominant_ms = dominant_started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

            const TASK_FLAT: u8 = 0;
            const TASK_PACKED: u8 = 1;
            const TASK_WHOLE: u8 = 2;
            const LARGE_BUCKET_WORK_PRODUCT: usize = 1_000_000;
            const PACKED_BLOCK_SIZE: usize = 512;
            let worker_count = rayon::current_num_threads();
            let mut tasks = Vec::<(u8, u8, Vec<u32>, usize)>::new();
            let mut flat_buckets = 0usize;
            let mut packed_buckets = 0usize;

            for (index, (byte, targets)) in byte_target_groups.iter().enumerate() {
                if dominant == Some(index) {
                    continue;
                }
                let byte_idx = *byte as usize;
                let token_count = token_buckets.token_indices_by_first_byte[byte_idx].len();
                let work = token_count.saturating_mul(targets.len());
                let use_flat = flat_profiles_supported
                    && l1_prefers_flat_utf8_lead_bucket(
                        *byte,
                        token_count,
                        targets.len(),
                        token_buckets.has_empty_suffix_by_bucket[byte_idx],
                    );
                if use_flat {
                    flat_buckets += 1;
                    let chunk_count = 4usize.min(worker_count).min(targets.len());
                    let chunk_size = targets.len().div_ceil(chunk_count);
                    for chunk in targets.chunks(chunk_size) {
                        tasks.push((
                            TASK_FLAT,
                            *byte,
                            chunk.to_vec(),
                            token_count.saturating_mul(chunk.len()),
                        ));
                    }
                } else if targets.len() >= 32
                    && (token_count >= 10_000 || work >= LARGE_BUCKET_WORK_PRODUCT)
                {
                    packed_buckets += 1;
                    let chunk_count = worker_count.min(8).min(targets.len());
                    let chunk_capacity = targets.len().div_ceil(chunk_count);
                    let mut chunks = (0..chunk_count)
                        .map(|_| Vec::with_capacity(chunk_capacity))
                        .collect::<Vec<_>>();
                    for (block_index, block) in targets.chunks(PACKED_BLOCK_SIZE).enumerate() {
                        chunks[block_index % chunk_count].extend_from_slice(block);
                    }
                    for chunk in chunks.into_iter().filter(|chunk| !chunk.is_empty()) {
                        let chunk_work = token_count.saturating_mul(chunk.len());
                        tasks.push((TASK_PACKED, *byte, chunk, chunk_work));
                    }
                } else {
                    tasks.push((TASK_WHOLE, *byte, targets.clone(), work));
                }
            }

            tasks.sort_unstable_by(|left, right| right.3.cmp(&left.3));
            let task_count = tasks.len();
            let remaining_started_at = profile_enabled.then(Instant::now);
            let task_batches = tasks
                .par_iter()
                .map(|(kind, byte, targets, _)| {
                    let byte_idx = *byte as usize;
                    let token_ids = &token_buckets.token_indices_by_first_byte[byte_idx];
                    let suffix_lcps = &token_buckets.suffix_lcps_by_first_byte[byte_idx];
                    match *kind {
                        TASK_FLAT => l1_bucket_suffix_signature_profiles_batched_arc(
                            *byte,
                            targets,
                            sorted_entries,
                            token_ids,
                            suffix_lcps,
                            &token_buckets.suffix_subtree_bytes[byte_idx],
                            &token_buckets.suffix_first_bytes_by_bucket[byte_idx],
                            token_buckets.has_empty_suffix_by_bucket[byte_idx],
                            &state_to_terminal_signature,
                            flat_trans,
                            transitions_by_byte,
                            num_tokenizer_states,
                            Some(&active_language),
                        ),
                        TASK_PACKED => {
                            let prebuilt_trie = token_buckets
                                .packed_suffix_tries_by_first_byte[byte_idx]
                                .get_or_init(|| {
                                    Arc::new(L1PackedSuffixTrie::build(
                                        sorted_entries,
                                        token_ids,
                                        suffix_lcps,
                                    ))
                                });
                            l1_bucket_suffix_signature_profiles_packed(
                                *byte,
                                targets,
                                sorted_entries,
                                token_ids,
                                suffix_lcps,
                                &token_buckets.suffix_subtree_bytes[byte_idx],
                                &token_buckets.suffix_first_bytes_by_bucket[byte_idx],
                                token_buckets.has_empty_suffix_by_bucket[byte_idx],
                                &state_to_terminal_signature,
                                self_loop_bytes_by_state,
                                flat_trans,
                                transitions_by_byte,
                                num_tokenizer_states,
                                Some(&active_language),
                                horizon_maps.as_deref(),
                                suffix_horizon_by_first_byte[byte_idx],
                                Some(prebuilt_trie.as_ref()),
                            )
                        }
                        TASK_WHOLE => build_byte_profiles(*byte, targets),
                        _ => unreachable!("unknown L1 profile task kind"),
                    }
                })
                .collect::<Vec<_>>();
            let mut remaining_profiles =
                task_batches.into_iter().flatten().collect::<Vec<_>>();

            // Builders canonicalize equal profiles within one task. Restore one
            // canonical Arc per first-byte bucket after joining task slices.
            let mut canonical_by_byte = (0..256)
                .map(|_| {
                    FxHashMap::<Arc<[(u32, u32, u32)]>, Arc<[(u32, u32, u32)]>>::default()
                })
                .collect::<Vec<_>>();
            for ((byte, _), profile) in &mut remaining_profiles {
                if profile.is_empty() {
                    continue;
                }
                let canonical = &mut canonical_by_byte[*byte as usize];
                if let Some(existing) = canonical.get(profile) {
                    *profile = Arc::clone(existing);
                } else {
                    canonical.insert(Arc::clone(profile), Arc::clone(profile));
                }
            }
            let remaining_ms = remaining_started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            if profile_enabled {
                eprintln!(
                    "[glrmask/profile][l1_profile_schedule] dominant_byte={:?} dominant_ms={:.3} tasks={} flat_buckets={} packed_buckets={} remaining_ms={:.3}",
                    dominant.map(|index| byte_target_groups[index].0),
                    dominant_ms,
                    task_count,
                    flat_buckets,
                    packed_buckets,
                    remaining_ms,
                );
            }

            dominant_batch
                .into_iter()
                .chain(std::iter::once(remaining_profiles))
                .collect()
        };
    let target_profiles: Vec<((u8, u32), Arc<[(u32, u32, u32)]>)> =
        target_profile_batches.into_iter().flatten().collect();
    let target_profiles_ms = target_profiles_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });

    let profile_intern_started_at = profile_enabled.then(Instant::now);
    let empty_profile: Arc<[(u32, u32, u32)]> = Arc::from([]);
    let mut profiles_by_id = vec![empty_profile];
    let mut next_profile_id = 1u32;
    let mut profile_id_by_bucket_and_ptr = FxHashMap::<(u8, usize), u32>::default();
    let assert_profile_pointer_partition =
        std::env::var_os("GLRMASK_ASSERT_L1_PROFILE_POINTER_PARTITION").is_some();
    let mut profile_content_to_bucket_and_ptr = assert_profile_pointer_partition
        .then(FxHashMap::<Arc<[(u32, u32, u32)]>, (u8, usize)>::default);
    let mut target_to_profile_id = Vec::<((u8, u32), u32)>::with_capacity(target_profiles.len());
    for (target_key, profile) in target_profiles {
        if !profile.is_empty() {
            if let Some(content_to_pointer) = profile_content_to_bucket_and_ptr.as_mut() {
                let pointer_key = (target_key.0, profile.as_ptr() as usize);
                if let Some(&existing) = content_to_pointer.get(&profile) {
                    assert_eq!(
                        existing, pointer_key,
                        "equal L1 profiles must share their first-byte behavior identity",
                    );
                } else {
                    content_to_pointer.insert(Arc::clone(&profile), pointer_key);
                }
            }
        }
        let profile_id = if profile.is_empty() {
            0
        } else {
            let pointer_key = (target_key.0, profile.as_ptr() as usize);
            if let Some(&profile_id) = profile_id_by_bucket_and_ptr.get(&pointer_key) {
                profile_id
            } else {
                let profile_id = next_profile_id;
                next_profile_id += 1;
                profile_id_by_bucket_and_ptr.insert(pointer_key, profile_id);
                profiles_by_id.push(profile);
                profile_id
            }
        };
        target_to_profile_id.push((target_key, profile_id));
    }
    let profile_ids_len = next_profile_id as usize;
    let profile_id_intern_ms = profile_intern_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    let profile_freeze_started_at = profile_enabled.then(Instant::now);
    let walk_profiles_by_id: Vec<L1WalkProfile> = profiles_by_id
        .iter()
        .map(|profile| freeze_l1_walk_profile(profile.as_ref()))
        .collect();
    let profile_freeze_ms = profile_freeze_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    let profile_intern_ms = profile_intern_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });

    let row_major_u16_keys = transitions_by_byte.is_none()
        && horizon_maps.is_none()
        && profile_ids_len <= u16::MAX as usize
        && state_to_terminal_signature
            .iter()
            .copied()
            .max()
            .is_none_or(|signature| signature <= u16::MAX as u32);
    if row_major_u16_keys {
        let state_keys_started_at = profile_enabled.then(Instant::now);
        let mut byte_slots = [u16::MAX; 256];
        for (slot, &byte) in nonempty_first_bytes.iter().enumerate() {
            byte_slots[byte] = slot as u16;
        }
        let num_slots = nonempty_first_bytes.len();
        let has_empty_tokens = !token_buckets.empty_token_indices.is_empty();
        let sig_cols = usize::from(has_empty_tokens);
        let row_width = num_slots + sig_cols;
        let num_states_in = states.len();
        debug_assert_ne!(row_width, 0);

        // Profile IDs are small in the exact quotient. Store the lookup by
        // (first-byte slot, raw target state), then read each DFA transition row
        // contiguously while generating its exact key.
        let mut profile_by_slot_target =
            vec![0u16; num_slots.saturating_mul(num_tokenizer_states)];
        for &((byte, target), profile_id) in &target_to_profile_id {
            let slot = byte_slots[byte as usize] as usize;
            debug_assert_ne!(slot, usize::from(u16::MAX));
            profile_by_slot_target[slot * num_tokenizer_states + target as usize] =
                profile_id as u16;
        }
        let profile_value = |state: usize, slot: usize, byte: usize| -> u16 {
            if !active_language[state] {
                return 0;
            }
            let target = flat_trans[state * 256 + byte];
            if target == dead {
                0
            } else {
                profile_by_slot_target[slot * num_tokenizer_states + target as usize]
            }
        };
        let hash_value = |mut hash: u64, slot: usize, value: u16| -> u64 {
            hash ^= (value as u64)
                .wrapping_add(0x9e37_79b9_7f4a_7c15)
                .wrapping_add((slot as u64).wrapping_mul(0x517c_c1b7_2722_0a95));
            hash.rotate_left(17).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        };
        let fill_row = |i: usize, row: &mut [u16], hash_slot: &mut u64| {
            let state = states[i];
            let mut hash = 0x9e37_79b9_7f4a_7c15u64;
            if has_empty_tokens {
                let value = state_to_terminal_signature[state] as u16;
                row[0] = value;
                hash = hash_value(hash, 0, value);
            }
            for (slot, &byte) in nonempty_first_bytes.iter().enumerate() {
                let col = sig_cols + slot;
                let value = profile_value(state, slot, byte);
                row[col] = value;
                hash = hash_value(hash, col, value);
            }
            *hash_slot = hash;
        };

        // Generate one exact key row at a time and retain only rows that become
        // class representatives. Full row comparison backs the hash lookup, so
        // collisions affect speed only. This avoids a 46 MB all-state key matrix
        // and its second grouping sweep.
        let mut representatives_by_hash =
            FxHashMap::<u64, Vec<(usize, Box<[u16]>)>>::default();
        let mut mapping = Vec::<usize>::with_capacity(num_states_in);
        let mut representative_profile_ids = FxHashMap::<u32, Arc<[u32]>>::default();
        let mut row = vec![0u16; row_width];
        let mut groups_len = 0usize;
        for i in 0..num_states_in {
            row.fill(0);
            let mut row_hash = 0u64;
            fill_row(i, &mut row, &mut row_hash);
            let bucket = representatives_by_hash.entry(row_hash).or_default();
            let representative_pos = bucket.iter().find_map(|(rep_pos, rep_row)| {
                (rep_row.as_ref() == row.as_slice()).then_some(*rep_pos)
            });
            let representative = match representative_pos {
                Some(rep_pos) => states[rep_pos],
                None => {
                    bucket.push((i, row.clone().into_boxed_slice()));
                    groups_len += 1;
                    let representative = states[i];
                    representative_profile_ids
                        .entry(representative as u32)
                        .or_insert_with(|| {
                            Arc::from(
                                row[sig_cols..]
                                    .iter()
                                    .map(|&profile_id| profile_id as u32)
                                    .collect::<Vec<_>>(),
                            )
                        });
                    representative
                }
            };
            mapping.push(representative);
        }
        let state_keys_ms = state_keys_started_at.map_or(0.0, |started| {
            started.elapsed().as_secs_f64() * 1000.0
        });
        let group_ms = 0.0;

        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][l1_streaming_state_keys] states={} slots={} lookup_bytes={} representative_key_bytes={} state_keys_ms={:.3}",
                num_states_in,
                num_slots,
                profile_by_slot_target.len() * std::mem::size_of::<u16>(),
                groups_len * row_width * std::mem::size_of::<u16>(),
                state_keys_ms,
            );
            eprintln!(
                "[glrmask/profile][l1_exact_equiv_detail] states={} first_bytes={} raw_unique_targets={} unique_targets={} horizon_selected={} profile_ids={} groups={} terminal_signature_ms={:.3} raw_target_probe_ms={:.3} horizon_quotient_ms={:.3} unique_targets_ms={:.3} target_profiles_ms={:.3} profile_id_intern_ms={:.3} profile_freeze_ms={:.3} profile_intern_ms={:.3} state_keys_ms={:.3} group_ms={:.3} total_ms={:.3}",
                states.len(),
                nonempty_first_bytes.len(),
                raw_unique_targets_len,
                unique_targets_len,
                use_remaining_horizon_quotients,
                profile_ids_len,
                groups_len,
                terminal_signature_ms,
                raw_target_probe_ms,
                horizon_quotient_ms,
                unique_targets_ms,
                target_profiles_ms,
                profile_id_intern_ms,
                profile_freeze_ms,
                profile_intern_ms,
                state_keys_ms,
                group_ms,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        return Ok((
            mapping,
            Some(L1ExactProfileReuse {
                target_to_profile_id,
                walk_profiles_by_id,
                profile_representatives_by_internal: Arc::from([]),
                representative_profile_ids,
                direct_terminal_signatures: terminal_signatures[1..].to_vec().into(),
                direct_state_to_terminal_signature: state_to_terminal_signature
                    .into_iter()
                    .map(|signature_id| {
                        if signature_id == 0 {
                            u32::MAX
                        } else {
                            signature_id - 1
                        }
                    })
                    .collect::<Vec<_>>()
                    .into(),
            }),
        ));
    }

    // Materialize each state's exact equivalence key as a contiguous row, then
    // group by a 64-bit fingerprint backed by full row equality. The key is the
    // per-first-byte profile-id vector, optionally prefixed by the terminal
    // signature when empty tokens exist. Columns are filled one first byte at a
    // time through a single reused, state-sized profile column (interned from
    // the small per-byte target list), so there is no num_slots*num_states dense
    // profile table to allocate, zero, and stride through under partition-level
    // memory contention. The byte-major transition column read and the column
    // write are both contiguous, and exact equality / representative profile
    // vectors read the same dense key matrix instead of re-walking transitions.
    let state_keys_started_at = profile_enabled.then(Instant::now);
    let mut byte_slots = [u16::MAX; 256];
    for (slot, &byte) in nonempty_first_bytes.iter().enumerate() {
        byte_slots[byte] = slot as u16;
    }
    let num_slots = nonempty_first_bytes.len();
    let mut slot_targets: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_slots];
    for &((byte, target), profile_id) in &target_to_profile_id {
        let slot = byte_slots[byte as usize] as usize;
        debug_assert_ne!(slot, usize::from(u16::MAX));
        slot_targets[slot].push((target, profile_id));
    }
    let has_empty_tokens = !token_buckets.empty_token_indices.is_empty();
    let sig_cols = usize::from(has_empty_tokens);
    let row_width = num_slots + sig_cols;
    let num_states_in = states.len();
    let mut keys = vec![0u32; num_states_in.saturating_mul(row_width).max(1)];
    // Tile over states so a block of rows stays cache-resident while all of its
    // first-byte columns are filled. Very wide key matrices can optionally fill
    // independent tiles in parallel; each worker owns one reusable profile
    // column, so there is no shared mutation or per-cell synchronization.
    const FILL_TILE: usize = 512;
    let fill_tile = |tile_start: usize, tile: &mut [u32], profile_col: &mut [u32]| {
        let tile_rows = tile.len() / row_width;
        if has_empty_tokens {
            for local_i in 0..tile_rows {
                let state = states[tile_start + local_i];
                tile[local_i * row_width] = state_to_terminal_signature[state];
            }
        }
        for (slot, &byte) in nonempty_first_bytes.iter().enumerate() {
            let col = sig_cols + slot;
            let canonical_state = horizon_maps
                .as_ref()
                .map(|maps| maps[suffix_horizon_by_first_byte[byte]].as_ref());
            for &(target, profile_id) in &slot_targets[slot] {
                profile_col[target as usize] = profile_id;
            }
            for local_i in 0..tile_rows {
                let i = tile_start + local_i;
                let target = if let Some(first_targets) = first_targets.as_ref() {
                    first_targets[slot * num_states_in + i]
                } else {
                    l1_transition(
                        flat_trans,
                        transitions_by_byte,
                        num_tokenizer_states,
                        Some(&active_language),
                        states[i] as u32,
                        byte,
                        canonical_state,
                    )
                };
                tile[local_i * row_width + col] = if target == dead {
                    0
                } else {
                    profile_col[target as usize]
                };
            }
            for &(target, _) in &slot_targets[slot] {
                profile_col[target as usize] = 0;
            }
        }
    };
    // Finite-horizon quotients already spend parallel work constructing their
    // canonical state maps, and this key fill runs inside an outer parallel
    // partition. Fanning out again here competes with sibling partitions and
    // lengthens the build tail. Ordinary exact-profile rows do not have that
    // prior quotient cost and retain the profitable parallel fill.
    let parallel_key_fill = horizon_maps.is_none()
        && std::env::var_os("GLRMASK_DISABLE_PARALLEL_L1_STATE_KEYS").is_none()
        && rayon::current_num_threads() > 1
        && row_width > 0
        && num_states_in.saturating_mul(row_width) >= 1_000_000;
    if parallel_key_fill {
        keys[..num_states_in * row_width]
            .par_chunks_mut(FILL_TILE * row_width)
            .enumerate()
            .for_each_init(
                || vec![0u32; num_tokenizer_states],
                |profile_col, (tile_index, tile)| {
                    fill_tile(tile_index * FILL_TILE, tile, profile_col);
                },
            );
    } else {
        let mut profile_col = vec![0u32; num_tokenizer_states];
        for (tile_index, tile) in keys[..num_states_in * row_width]
            .chunks_mut(FILL_TILE * row_width)
            .enumerate()
        {
            fill_tile(tile_index * FILL_TILE, tile, &mut profile_col);
        }
    }
    let row_hash = |row: &[u32]| -> u64 {
        let mut hash = 0x9e37_79b9_7f4a_7c15u64;
        for (slot, &value) in row.iter().enumerate() {
            hash ^= (value as u64)
                .wrapping_add(0x9e37_79b9_7f4a_7c15)
                .wrapping_add((slot as u64).wrapping_mul(0x517c_c1b7_2722_0a95));
            hash = hash.rotate_left(17).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        }
        hash
    };
    let state_key_hashes: Vec<u64> = if rayon::current_num_threads() == 1 || num_states_in < 4096 {
        (0..num_states_in)
            .map(|i| row_hash(&keys[i * row_width..i * row_width + row_width]))
            .collect()
    } else {
        (0..num_states_in)
            .into_par_iter()
            .map(|i| row_hash(&keys[i * row_width..i * row_width + row_width]))
            .collect()
    };
    let state_keys_ms = state_keys_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });

    let group_started_at = profile_enabled.then(Instant::now);
    let mut representatives_by_hash = FxHashMap::<u64, Vec<usize>>::default();
    let mut mapping = Vec::<usize>::with_capacity(num_states_in);
    // Exact-class representatives become the L1 id-map representatives. Preserve
    // their already-proved first-byte profile vectors (the slot columns of the
    // key row) so the direct terminal-DWA builder does not repeat the work.
    let mut representative_profile_ids = FxHashMap::<u32, Arc<[u32]>>::default();
    let mut groups_len = 0usize;
    for i in 0..num_states_in {
        let row = &keys[i * row_width..i * row_width + row_width];
        let bucket = representatives_by_hash
            .entry(state_key_hashes[i])
            .or_default();
        let mut representative_pos = None;
        for &rep_pos in bucket.iter() {
            if &keys[rep_pos * row_width..rep_pos * row_width + row_width] == row {
                representative_pos = Some(rep_pos);
                break;
            }
        }
        let representative = match representative_pos {
            Some(rep_pos) => states[rep_pos],
            None => {
                bucket.push(i);
                groups_len += 1;
                let representative = states[i];
                representative_profile_ids
                    .entry(representative as u32)
                    .or_insert_with(|| Arc::from(&row[sig_cols..]));
                representative
            }
        };
        mapping.push(representative);
    }
    let group_ms = group_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][l1_exact_equiv_detail] states={} first_bytes={} raw_unique_targets={} unique_targets={} horizon_selected={} profile_ids={} groups={} terminal_signature_ms={:.3} raw_target_probe_ms={:.3} horizon_quotient_ms={:.3} unique_targets_ms={:.3} target_profiles_ms={:.3} profile_id_intern_ms={:.3} profile_freeze_ms={:.3} profile_intern_ms={:.3} state_keys_ms={:.3} group_ms={:.3} total_ms={:.3}",
            states.len(),
            nonempty_first_bytes.len(),
            raw_unique_targets_len,
            unique_targets_len,
            use_remaining_horizon_quotients,
            profile_ids_len,
            groups_len,
            terminal_signature_ms,
            raw_target_probe_ms,
            horizon_quotient_ms,
            unique_targets_ms,
            target_profiles_ms,
            profile_id_intern_ms,
            profile_freeze_ms,
            profile_intern_ms,
            state_keys_ms,
            group_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Ok((
        mapping,
        Some(L1ExactProfileReuse {
            target_to_profile_id,
            walk_profiles_by_id,
            profile_representatives_by_internal: Arc::from([]),
            representative_profile_ids,
            // Exact-equivalence signature ids reserve zero for empty. The
            // direct DWA builder uses `u32::MAX` for empty and zero-based ids
            // for non-empty signatures, matching `materialize_walk_cache`.
            direct_terminal_signatures: terminal_signatures[1..].to_vec().into(),
            direct_state_to_terminal_signature: state_to_terminal_signature
                .into_iter()
                .map(|signature_id| {
                    if signature_id == 0 {
                        u32::MAX
                    } else {
                        signature_id - 1
                    }
                })
                .collect::<Vec<_>>()
                .into(),
        }),
    ))
}

fn l1_target_self_loop_covers_suffix_subtree(
    target: u32,
    suffix_subtree: &[u64; 4],
    flat_trans: &[u32],
) -> bool {
    for word in 0..4usize {
        let mut bits = suffix_subtree[word];
        while bits != 0 {
            let offset = bits.trailing_zeros() as usize;
            let byte = word * 64 + offset;
            if flat_trans[target as usize * 256 + byte] != target {
                return false;
            }
            bits &= bits - 1;
        }
    }
    true
}

#[derive(Debug, Clone, Copy, Default)]
struct L1BucketCompressionEstimate {
    self_loop_targets: usize,
    first_step_dead_targets: usize,
    first_step_groups: usize,
    effective_targets: usize,
    first_step_live_edges: usize,
    first_step_unique_states: usize,
}

fn l1_bucket_compression_estimate(
    targets: &[u32],
    suffix_subtree: &[u64; 4],
    suffix_first_bytes: &[u64; 4],
    has_empty_suffix: bool,
    flat_trans: &[u32],
) -> L1BucketCompressionEstimate {
    let dead = u32::MAX;
    let mut estimate = L1BucketCompressionEstimate::default();
    let mut walk_targets = Vec::new();
    for &target in targets {
        if l1_target_self_loop_covers_suffix_subtree(target, suffix_subtree, flat_trans) {
            estimate.self_loop_targets += 1;
        } else {
            walk_targets.push(target);
        }
    }
    if has_empty_suffix {
        estimate.effective_targets = walk_targets.len();
        return estimate;
    }
    let mut first_suffix_bytes = Vec::new();
    for word in 0..4u8 {
        let mut bits = suffix_first_bytes[word as usize];
        while bits != 0 {
            let offset = bits.trailing_zeros() as u8;
            first_suffix_bytes.push(word * 64 + offset);
            bits &= bits - 1;
        }
    }
    if first_suffix_bytes.is_empty() {
        estimate.effective_targets = walk_targets.len();
        return estimate;
    }
    let mut groups = FxHashMap::<Vec<u32>, usize>::default();
    let mut unique_states = rustc_hash::FxHashSet::default();
    for target in walk_targets {
        let fp = first_suffix_bytes
            .iter()
            .map(|&byte| flat_trans[target as usize * 256 + byte as usize])
            .collect::<Vec<_>>();
        if fp.iter().all(|&state| state == dead) {
            estimate.first_step_dead_targets += 1;
            continue;
        }
        estimate.first_step_live_edges += fp.iter().filter(|&&state| state != dead).count();
        unique_states.extend(fp.iter().copied().filter(|&state| state != dead));
        groups.entry(fp).or_insert(0);
    }
    estimate.first_step_groups = groups.len();
    estimate.effective_targets = groups.len();
    estimate.first_step_unique_states = unique_states.len();
    estimate
}

/// Group first-step-equivalent suffix targets without allocating a temporary
/// fingerprint vector for every target.
///
/// Only one owned fingerprint is retained per distinct group. The common case
/// reuses `scratch` and appends the target to an existing group after an exact
/// collision check.
fn l1_group_targets_by_first_suffix_fingerprint(
    targets: &[u32],
    first_suffix_bytes: &[u8],
    flat_trans: &[u32],
    transitions_by_byte: Option<&[u32]>,
    num_tokenizer_states: usize,
    active_language: Option<&[bool]>,
    canonical_state: Option<&[u32]>,
) -> (Vec<u32>, Vec<Vec<u32>>, Vec<u32>) {
    let dead = u32::MAX;
    let mut representatives = Vec::<u32>::new();
    let mut others = Vec::<Vec<u32>>::new();
    let mut fingerprints = Vec::<Vec<u32>>::new();
    let mut group_indices_by_hash = FxHashMap::<u64, Vec<usize>>::default();
    let mut dead_targets = Vec::<u32>::new();
    let mut scratch = Vec::<u32>::with_capacity(first_suffix_bytes.len());

    for &target in targets {
        scratch.clear();
        let mut all_dead = true;
        for &byte in first_suffix_bytes {
            let next = l1_transition(
                flat_trans,
                transitions_by_byte,
                num_tokenizer_states,
                active_language,
                target,
                byte as usize,
                canonical_state,
            );
            all_dead &= next == dead;
            scratch.push(next);
        }
        if all_dead {
            dead_targets.push(target);
            continue;
        }

        let mut hasher = rustc_hash::FxHasher::default();
        scratch.hash(&mut hasher);
        let hash = hasher.finish();
        let candidates = group_indices_by_hash.entry(hash).or_default();
        if let Some(group_index) = candidates
            .iter()
            .copied()
            .find(|&group_index| fingerprints[group_index] == scratch)
        {
            others[group_index].push(target);
            continue;
        }

        let group_index = representatives.len();
        representatives.push(target);
        others.push(Vec::new());
        fingerprints.push(std::mem::replace(
            &mut scratch,
            Vec::with_capacity(first_suffix_bytes.len()),
        ));
        candidates.push(group_index);
    }

    (representatives, others, dead_targets)
}

fn l1_bucket_suffix_signature_profiles_batched_arc(
    first_byte: u8,
    targets: &[u32],
    sorted_entries: &[(u32, Arc<[u8]>)],
    token_ids: &[usize],
    suffix_lcps: &[usize],
    suffix_subtree: &[u64; 4],
    suffix_first_bytes: &[u64; 4],
    has_empty_suffix: bool,
    state_to_terminal_signature: &[u32],
    flat_trans: &[u32],
    transitions_by_byte: Option<&[u32]>,
    num_tokenizer_states: usize,
    active_language: Option<&[bool]>,
) -> Vec<((u8, u32), Arc<[(u32, u32, u32)]>)> {
    let profiles = l1_bucket_suffix_signature_profiles_batched(
        first_byte,
        targets,
        sorted_entries,
        token_ids,
        suffix_lcps,
        suffix_subtree,
        suffix_first_bytes,
        has_empty_suffix,
        state_to_terminal_signature,
        flat_trans,
        transitions_by_byte,
        num_tokenizer_states,
        active_language,
    );
    let mut canonical =
        FxHashMap::<Arc<[(u32, u32, u32)]>, Arc<[(u32, u32, u32)]>>::default();
    profiles
        .into_iter()
        .map(|(key, profile)| {
            if profile.is_empty() {
                return (key, Arc::from([]));
            }
            let profile: Arc<[(u32, u32, u32)]> = Arc::from(profile);
            if let Some(existing) = canonical.get(&profile) {
                (key, Arc::clone(existing))
            } else {
                canonical.insert(Arc::clone(&profile), Arc::clone(&profile));
                (key, profile)
            }
        })
        .collect()
}

fn l1_bucket_suffix_signature_profiles_batched(
    first_byte: u8,
    targets: &[u32],
    sorted_entries: &[(u32, Arc<[u8]>)],
    token_ids: &[usize],
    suffix_lcps: &[usize],
    suffix_subtree: &[u64; 4],
    suffix_first_bytes: &[u64; 4],
    has_empty_suffix: bool,
    state_to_terminal_signature: &[u32],
    flat_trans: &[u32],
    transitions_by_byte: Option<&[u32]>,
    num_tokenizer_states: usize,
    active_language: Option<&[bool]>,
) -> Vec<((u8, u32), Vec<(u32, u32, u32)>)> {
    let dead = u32::MAX;
    let mut results = Vec::<((u8, u32), Vec<(u32, u32, u32)>)>::with_capacity(targets.len());
    if token_ids.is_empty() {
        for &target in targets {
            results.push(((first_byte, target), Vec::new()));
        }
        return results;
    }

    let mut walk_targets = Vec::<u32>::new();
    for &target in targets {
        if l1_target_self_loop_covers_suffix_subtree(target, suffix_subtree, flat_trans) {
            let mut profile = Vec::<(u32, u32, u32)>::new();
            let sig_id = state_to_terminal_signature[target as usize];
            if sig_id != 0 {
                if let (Some(&first), Some(&last)) = (token_ids.first(), token_ids.last()) {
                    profile.push((sig_id, first as u32, last as u32));
                }
            }
            results.push(((first_byte, target), profile));
        } else {
            walk_targets.push(target);
        }
    }

    if walk_targets.is_empty() {
        return results;
    }

    let mut dedup_others: Option<Vec<Vec<u32>>> = None;
    if !has_empty_suffix {
        let mut first_suffix_bytes = Vec::<u8>::new();
        for word in 0..4u8 {
            let mut bits = suffix_first_bytes[word as usize];
            while bits != 0 {
                let offset = bits.trailing_zeros() as u8;
                first_suffix_bytes.push(word * 64 + offset);
                bits &= bits - 1;
            }
        }

        if !first_suffix_bytes.is_empty() {
            let (deduped_targets, others, dead_targets) =
                l1_group_targets_by_first_suffix_fingerprint(
                    &walk_targets,
                    &first_suffix_bytes,
                    flat_trans,
                    transitions_by_byte,
                    num_tokenizer_states,
                    active_language,
                    None,
                );
            for target in dead_targets {
                results.push(((first_byte, target), Vec::new()));
            }
            if deduped_targets.len() < walk_targets.len() {
                walk_targets = deduped_targets;
                dedup_others = Some(others);
            }
        }
    }


    if walk_targets.is_empty() {
        return results;
    }

    let num_walk = walk_targets.len();
    let mut profiles = vec![Vec::<(u32, u32, u32)>::new(); num_walk];
    let mut suffix_states = walk_targets.clone();

    for (bucket_pos, &internal_token_id) in token_ids.iter().enumerate() {
        let suffix_bytes = &sorted_entries[internal_token_id].1[1..];
        let lcp_len = suffix_lcps[bucket_pos].min(suffix_states.len() / num_walk - 1);
        suffix_states.truncate((lcp_len + 1) * num_walk);

        for byte_pos in lcp_len..suffix_bytes.len() {
            let byte = suffix_bytes[byte_pos];
            let base = byte_pos * num_walk;
            for target_idx in 0..num_walk {
                let previous_state = suffix_states[base + target_idx];
                let next_state = if previous_state == dead {
                    dead
                } else {
                    l1_transition(
                        flat_trans,
                        transitions_by_byte,
                        num_tokenizer_states,
                        active_language,
                        previous_state,
                        byte as usize,
                        None,
                    )
                };
                suffix_states.push(next_state);
            }
        }

        let end_base = suffix_bytes.len() * num_walk;
        let token_id = internal_token_id as u32;
        for target_idx in 0..num_walk {
            let final_state = suffix_states[end_base + target_idx];
            if final_state == dead {
                continue;
            }
            let sig_id = state_to_terminal_signature[final_state as usize];
            if sig_id != 0 {
                append_l1_signature_profile_run(&mut profiles[target_idx], sig_id, token_id);
            }
        }
    }

    for (target_idx, (target, profile)) in walk_targets.into_iter().zip(profiles).enumerate() {
        if let Some(ref others) = dedup_others {
            for &other_target in &others[target_idx] {
                results.push(((first_byte, other_target), profile.clone()));
            }
        }
        results.push(((first_byte, target), profile));
    }
    results
}


const L1_NONE: u32 = u32::MAX;

/// Compact suffix trie for one first-byte vocabulary bucket. Token IDs are
/// byte-sorted, so each node's subtree is one contiguous token interval.
#[derive(Debug, Clone, Copy)]
struct L1PackedSuffixTrieNode {
    incoming_byte: u8,
    terminal_token: u32,
    terminal_token_end: u32,
    subtree_start: u32,
    subtree_end: u32,
    first_child: u32,
    last_child: u32,
    next_sibling: u32,
    first_edge: u32,
    edge_len: u32,
}

impl L1PackedSuffixTrieNode {
    fn root() -> Self {
        Self {
            incoming_byte: 0,
            terminal_token: L1_NONE,
            terminal_token_end: L1_NONE,
            subtree_start: L1_NONE,
            subtree_end: L1_NONE,
            first_child: L1_NONE,
            last_child: L1_NONE,
            next_sibling: L1_NONE,
            first_edge: 0,
            edge_len: 0,
        }
    }

    fn child(byte: u8) -> Self {
        Self {
            incoming_byte: byte,
            ..Self::root()
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct L1PackedSuffixTrieEdge {
    byte: u8,
    child: u32,
}

#[derive(Debug)]
struct L1PackedSuffixTrie {
    nodes: Vec<L1PackedSuffixTrieNode>,
    edges: Vec<L1PackedSuffixTrieEdge>,
    remaining_horizon_by_node: Vec<usize>,
    subtree_bytes_by_node: Vec<[u64; 4]>,
}

impl L1PackedSuffixTrie {
    fn build(
        sorted_entries: &[(u32, Arc<[u8]>)],
        token_ids: &[usize],
        suffix_lcps: &[usize],
    ) -> Self {
        // The LCP walk determines the exact number of nodes appended by each
        // token. Reserve once so large vocabulary buckets do not repeatedly
        // copy the packed trie while it is being constructed.
        let mut node_capacity = 1usize;
        let mut previous_suffix_len = 0usize;
        for (bucket_pos, &internal_token_id) in token_ids.iter().enumerate() {
            let suffix_len = sorted_entries[internal_token_id].1.len().saturating_sub(1);
            let lcp = suffix_lcps[bucket_pos]
                .min(suffix_len)
                .min(previous_suffix_len);
            node_capacity += suffix_len - lcp;
            previous_suffix_len = suffix_len;
        }
        let mut nodes = Vec::with_capacity(node_capacity);
        nodes.push(L1PackedSuffixTrieNode::root());
        let mut path = vec![0u32];

        for (bucket_pos, &internal_token_id) in token_ids.iter().enumerate() {
            let suffix = &sorted_entries[internal_token_id].1[1..];
            let lcp = suffix_lcps[bucket_pos]
                .min(suffix.len())
                .min(path.len().saturating_sub(1));
            path.truncate(lcp + 1);
            for &byte in &suffix[lcp..] {
                let parent = *path.last().expect("suffix trie path") as usize;
                let child = nodes.len() as u32;
                nodes.push(L1PackedSuffixTrieNode::child(byte));
                if nodes[parent].first_child == L1_NONE {
                    nodes[parent].first_child = child;
                } else {
                    let last = nodes[parent].last_child as usize;
                    nodes[last].next_sibling = child;
                }
                nodes[parent].last_child = child;
                path.push(child);
            }

            let token_id = internal_token_id as u32;
            let terminal = *path.last().expect("suffix trie terminal path") as usize;
            if nodes[terminal].terminal_token == L1_NONE {
                nodes[terminal].terminal_token = token_id;
                nodes[terminal].terminal_token_end = token_id;
            } else {
                debug_assert_eq!(token_id, nodes[terminal].terminal_token_end + 1);
                nodes[terminal].terminal_token_end = token_id;
            }
        }
        debug_assert_eq!(nodes.len(), node_capacity);

        // Token IDs follow the byte-sorted vocabulary order. Every suffix-trie
        // subtree therefore spans one contiguous leaf-token interval. Computing
        // that interval once bottom-up avoids rewriting it along each token path.
        for node_index in (0..nodes.len()).rev() {
            let mut subtree_start = nodes[node_index].terminal_token;
            let mut subtree_end = nodes[node_index].terminal_token_end;
            let mut child = nodes[node_index].first_child;
            while child != L1_NONE {
                let child_node = nodes[child as usize];
                if subtree_start == L1_NONE || child_node.subtree_start < subtree_start {
                    subtree_start = child_node.subtree_start;
                }
                if subtree_end == L1_NONE || child_node.subtree_end > subtree_end {
                    subtree_end = child_node.subtree_end;
                }
                child = child_node.next_sibling;
            }
            debug_assert_ne!(subtree_start, L1_NONE);
            debug_assert_ne!(subtree_end, L1_NONE);
            nodes[node_index].subtree_start = subtree_start;
            nodes[node_index].subtree_end = subtree_end;
        }

        let mut edges = Vec::with_capacity(nodes.len().saturating_sub(1));
        for node_index in 0..nodes.len() {
            let first_edge = edges.len() as u32;
            let mut child = nodes[node_index].first_child;
            while child != L1_NONE {
                let child_node = nodes[child as usize];
                edges.push(L1PackedSuffixTrieEdge {
                    byte: child_node.incoming_byte,
                    child,
                });
                child = child_node.next_sibling;
            }
            nodes[node_index].first_edge = first_edge;
            nodes[node_index].edge_len = edges.len() as u32 - first_edge;
        }
        let mut remaining_horizon_by_node = vec![0usize; nodes.len()];
        let mut subtree_bytes_by_node = vec![[0u64; 4]; nodes.len()];
        for node_index in (0..nodes.len()).rev() {
            let node = nodes[node_index];
            let mut remaining_horizon = 0usize;
            for edge_offset in 0..node.edge_len as usize {
                let edge = edges[node.first_edge as usize + edge_offset];
                let child = edge.child as usize;
                remaining_horizon =
                    remaining_horizon.max(1 + remaining_horizon_by_node[child]);
                subtree_bytes_by_node[node_index][edge.byte as usize / 64] |=
                    1u64 << (edge.byte as usize % 64);
                for word in 0..4 {
                    subtree_bytes_by_node[node_index][word] |= subtree_bytes_by_node[child][word];
                }
            }
            remaining_horizon_by_node[node_index] = remaining_horizon;
        }
        Self {
            nodes,
            edges,
            remaining_horizon_by_node,
            subtree_bytes_by_node,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct L1PackedProductNodeData {
    states_start: u32,
    states_len: u32,
    behaviors_start: u32,
    records_start: u32,
}

#[derive(Clone, Copy, Default)]
struct L1PackedProductEdgeData {
    map_start: u32,
}

#[derive(Clone, Copy)]
struct L1PackedProductBehaviorRecord {
    terminal_signature: u32,
    child_behaviors_start: u32,
    uniform_signature: u32,
    hash_next: u32,
}

const L1_UNIFORM_BEHAVIOR_FLAG: u32 = 1 << 31;

#[inline]
fn l1_encode_uniform_behavior(signature: u32) -> u32 {
    debug_assert!(signature != 0 && signature < L1_UNIFORM_BEHAVIOR_FLAG);
    L1_UNIFORM_BEHAVIOR_FLAG | signature
}

#[inline]
fn l1_uniform_behavior_signature(behavior: u32) -> Option<u32> {
    (behavior & L1_UNIFORM_BEHAVIOR_FLAG != 0)
        .then_some(behavior & !L1_UNIFORM_BEHAVIOR_FLAG)
}

#[inline]
fn l1_packed_hash_behavior(terminal_signature: u32, child_behaviors: &[u32]) -> u64 {
    let mut hash = terminal_signature as u64 ^ 0x9e37_79b9_7f4a_7c15;
    for &child in child_behaviors {
        hash = hash.rotate_left(13) ^ (child as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
        hash = hash.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    hash
}

fn l1_packed_uniform_signature(
    trie: &L1PackedSuffixTrie,
    node_index: usize,
    behavior_id: u32,
    data: &[L1PackedProductNodeData],
    records: &[L1PackedProductBehaviorRecord],
) -> u32 {
    if behavior_id == 0 {
        return 0;
    }
    if let Some(signature) = l1_uniform_behavior_signature(behavior_id) {
        return signature;
    }
    let node = trie.nodes[node_index];
    if node.edge_len == 0 {
        return behavior_id;
    }
    if node.terminal_token == L1_NONE && node.edge_len == 1 {
        let child = trie.edges[node.first_edge as usize].child as usize;
        return l1_packed_uniform_signature(trie, child, behavior_id, data, records);
    }
    records[data[node_index].records_start as usize + behavior_id as usize - 1].uniform_signature
}

fn l1_packed_append_behavior_reference(
    trie: &L1PackedSuffixTrie,
    node_index: usize,
    behavior_id: u32,
    data: &[L1PackedProductNodeData],
    records: &[L1PackedProductBehaviorRecord],
    record_child_behaviors: &[u32],
    profile: &mut Vec<(u32, u32, u32)>,
) {
    if behavior_id == 0 {
        return;
    }
    let node = trie.nodes[node_index];
    if let Some(signature) = l1_uniform_behavior_signature(behavior_id) {
        append_l1_signature_profile_range(profile, signature, node.subtree_start, node.subtree_end);
        return;
    }
    if node.edge_len == 0 {
        append_l1_signature_profile_range(profile, behavior_id, node.subtree_start, node.subtree_end);
        return;
    }
    if node.terminal_token == L1_NONE && node.edge_len == 1 {
        let child = trie.edges[node.first_edge as usize].child as usize;
        l1_packed_append_behavior_reference(
            trie,
            child,
            behavior_id,
            data,
            records,
            record_child_behaviors,
            profile,
        );
        return;
    }

    let record = records[data[node_index].records_start as usize + behavior_id as usize - 1];
    if record.uniform_signature != 0 {
        append_l1_signature_profile_range(
            profile,
            record.uniform_signature,
            node.subtree_start,
            node.subtree_end,
        );
        return;
    }
    if record.terminal_signature != 0 {
        append_l1_signature_profile_range(
            profile,
            record.terminal_signature,
            node.terminal_token,
            node.terminal_token_end,
        );
    }
    let children_start = record.child_behaviors_start as usize;
    for edge_offset in 0..node.edge_len as usize {
        let edge = trie.edges[node.first_edge as usize + edge_offset];
        l1_packed_append_behavior_reference(
            trie,
            edge.child as usize,
            record_child_behaviors[children_start + edge_offset],
            data,
            records,
            record_child_behaviors,
            profile,
        );
    }
}


fn l1_uniform_bucket_profile(
    profiles_by_signature: &mut FxHashMap<u32, Arc<[(u32, u32, u32)]>>,
    signature_id: u32,
    token_start: u32,
    token_end: u32,
) -> Arc<[(u32, u32, u32)]> {
    Arc::clone(
        profiles_by_signature
            .entry(signature_id)
            .or_insert_with(|| Arc::from(vec![(signature_id, token_start, token_end)])),
    )
}

fn l1_bucket_suffix_signature_profiles_packed(
    first_byte: u8,
    targets: &[u32],
    sorted_entries: &[(u32, Arc<[u8]>)],
    token_ids: &[usize],
    suffix_lcps: &[usize],
    suffix_subtree: &[u64; 4],
    suffix_first_bytes: &[u64; 4],
    has_empty_suffix: bool,
    state_to_terminal_signature: &[u32],
    self_loop_bytes_by_state: &[U8Set],
    flat_trans: &[u32],
    transitions_by_byte: Option<&[u32]>,
    num_lexer_states: usize,
    active_language: Option<&[bool]>,
    horizon_maps: Option<&[Arc<[u32]>]>,
    suffix_horizon: usize,
    prebuilt_trie: Option<&L1PackedSuffixTrie>,
) -> Vec<((u8, u32), Arc<[(u32, u32, u32)]>)> {
    let profiling = compile_profile_enabled();
    let total_started_at = profiling.then(Instant::now);
    let dead = u32::MAX;
    let mut results = Vec::<((u8, u32), Arc<[(u32, u32, u32)]>)>::with_capacity(targets.len());
    if token_ids.is_empty() {
        for &target in targets {
            results.push(((first_byte, target), Arc::from([])));
        }
        return results;
    }
    let token_start = token_ids[0] as u32;
    let token_end = *token_ids.last().expect("non-empty suffix bucket") as u32;
    let mut uniform_profiles_by_signature =
        FxHashMap::<u32, Arc<[(u32, u32, u32)]>>::default();

    let mut walk_targets = Vec::<u32>::new();
    for &target in targets {
        if l1_target_self_loop_covers_suffix_subtree(target, suffix_subtree, flat_trans) {
            let sig_id = state_to_terminal_signature[target as usize];
            let profile = if sig_id == 0 {
                Arc::from([])
            } else {
                l1_uniform_bucket_profile(
                    &mut uniform_profiles_by_signature,
                    sig_id,
                    token_start,
                    token_end,
                )
            };
            results.push(((first_byte, target), profile));
        } else {
            walk_targets.push(target);
        }
    }
    if walk_targets.is_empty() {
        return results;
    }

    let mut dedup_others: Option<Vec<Vec<u32>>> = None;
    let mut first_suffix_bytes = Vec::<u8>::new();
    for word in 0..4u8 {
        let mut bits = suffix_first_bytes[word as usize];
        while bits != 0 {
            let offset = bits.trailing_zeros() as u8;
            first_suffix_bytes.push(word * 64 + offset);
            bits &= bits - 1;
        }
    }
    // Without a one-byte token, matching the first-step successor vector is an
    // exact quotient for the whole suffix trie. Empty-suffix buckets also need
    // the root terminal observation, so those buckets retain their full target
    // set rather than using this first-step-only fingerprint.
    if !has_empty_suffix && !first_suffix_bytes.is_empty() {
        let canonical_state = horizon_maps
            .map(|maps| maps[suffix_horizon.saturating_sub(1)].as_ref());
        let (deduped_targets, others, dead_targets) =
            l1_group_targets_by_first_suffix_fingerprint(
                &walk_targets,
                &first_suffix_bytes,
                flat_trans,
                transitions_by_byte,
                num_lexer_states,
                active_language,
                canonical_state,
            );
        for target in dead_targets {
            results.push(((first_byte, target), Arc::from([])));
        }
        if deduped_targets.len() < walk_targets.len() {
            walk_targets = deduped_targets;
            dedup_others = Some(others);
        }
    }

    if walk_targets.is_empty() {
        return results;
    }

    let trie_started_at = profiling.then(Instant::now);
    let owned_trie;
    let trie = if let Some(trie) = prebuilt_trie {
        trie
    } else {
        owned_trie = L1PackedSuffixTrie::build(sorted_entries, token_ids, suffix_lcps);
        &owned_trie
    };
    debug_assert_eq!(trie.remaining_horizon_by_node[0], suffix_horizon);
    let trie_ms = trie_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let propagate_started_at = profiling.then(Instant::now);
    let mut data = vec![L1PackedProductNodeData::default(); trie.nodes.len()];
    let mut edge_data = vec![L1PackedProductEdgeData::default(); trie.edges.len()];
    let reserve_packed_product = token_ids.len() >= 10_000;
    let estimated_states = if reserve_packed_product {
        walk_targets.len().saturating_mul(2_048)
    } else {
        0
    };
    let mut states = Vec::<u32>::with_capacity(estimated_states);
    let mut transition_maps =
        Vec::<u32>::with_capacity(estimated_states.saturating_mul(2));
    states.extend_from_slice(&walk_targets);
    data[0].states_start = 0;
    data[0].states_len = walk_targets.len() as u32;

    let mut seen_stamp = vec![0u32; num_lexer_states];
    let mut seen_index = vec![0u32; num_lexer_states];
    let mut stamp = 0u32;
    let mut uniform_subtree_transitions = 0usize;
    let mut active_nodes = Vec::<usize>::with_capacity(trie.nodes.len().min(states.len().max(1)));
    active_nodes.push(0);
    let mut active_node_cursor = 0usize;
    while active_node_cursor < active_nodes.len() {
        let node_index = active_nodes[active_node_cursor];
        active_node_cursor += 1;
        let node = trie.nodes[node_index];
        let parent_start = data[node_index].states_start as usize;
        let parent_len = data[node_index].states_len as usize;
        debug_assert_ne!(parent_len, 0);
        for edge_offset in 0..node.edge_len as usize {
            let edge_index = node.first_edge as usize + edge_offset;
            let edge = trie.edges[edge_index];
            let child = edge.child as usize;
            let canonical_state = horizon_maps
                .map(|maps| maps[trie.remaining_horizon_by_node[child]].as_ref());
            edge_data[edge_index].map_start = transition_maps.len() as u32;
            let child_start = states.len() as u32;
            let uniform_transition = |next: u32| -> Option<u32> {
                if next == dead {
                    return Some(L1_NONE);
                }
                let self_loops = &self_loop_bytes_by_state[next as usize];
                let subtree = U8Set::from_words(trie.subtree_bytes_by_node[child]);
                if subtree.is_subset(self_loops) {
                    let signature = state_to_terminal_signature[next as usize];
                    Some(if signature == 0 {
                        L1_NONE
                    } else {
                        l1_encode_uniform_behavior(signature)
                    })
                } else {
                    None
                }
            };
            if parent_len == 1 {
                let next = l1_transition(
                    flat_trans,
                    transitions_by_byte,
                    num_lexer_states,
                    active_language,
                    states[parent_start],
                    edge.byte as usize,
                    canonical_state,
                );
                if let Some(encoded) = uniform_transition(next) {
                    uniform_subtree_transitions += usize::from(next != dead);
                    transition_maps.push(encoded);
                } else {
                    states.push(next);
                    transition_maps.push(0);
                }
            } else if parent_len == 2 {
                let first = l1_transition(
                    flat_trans,
                    transitions_by_byte,
                    num_lexer_states,
                    active_language,
                    states[parent_start],
                    edge.byte as usize,
                    canonical_state,
                );
                let second = l1_transition(
                    flat_trans,
                    transitions_by_byte,
                    num_lexer_states,
                    active_language,
                    states[parent_start + 1],
                    edge.byte as usize,
                    canonical_state,
                );
                let mut first_added_raw = None;
                let first_index = if let Some(encoded) = uniform_transition(first) {
                    uniform_subtree_transitions += usize::from(first != dead);
                    encoded
                } else {
                    states.push(first);
                    first_added_raw = Some(first);
                    0
                };
                let second_index = if let Some(encoded) = uniform_transition(second) {
                    uniform_subtree_transitions += usize::from(second != dead);
                    encoded
                } else if first_added_raw == Some(second) {
                    0
                } else {
                    let index = states.len() as u32 - child_start;
                    states.push(second);
                    index
                };
                transition_maps.push(first_index);
                transition_maps.push(second_index);
            } else {
                stamp = stamp.wrapping_add(1);
                if stamp == 0 {
                    seen_stamp.fill(0);
                    stamp = 1;
                }
                for state_offset in 0..parent_len {
                    let state = states[parent_start + state_offset];
                    let next = l1_transition(
                        flat_trans,
                        transitions_by_byte,
                        num_lexer_states,
                        active_language,
                        state,
                        edge.byte as usize,
                        canonical_state,
                    );
                    if let Some(encoded) = uniform_transition(next) {
                        uniform_subtree_transitions += usize::from(next != dead);
                        transition_maps.push(encoded);
                    } else if seen_stamp[next as usize] == stamp {
                        transition_maps.push(seen_index[next as usize]);
                    } else {
                        let child_index = states.len() as u32 - child_start;
                        seen_stamp[next as usize] = stamp;
                        seen_index[next as usize] = child_index;
                        states.push(next);
                        transition_maps.push(child_index);
                    }
                }
            }
            data[child].states_start = child_start;
            data[child].states_len = states.len() as u32 - child_start;
            if data[child].states_len != 0 {
                active_nodes.push(child);
            }
        }
    }

    let propagate_ms = propagate_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let behavior_started_at = profiling.then(Instant::now);
    let mut behavior_ids = Vec::<u32>::with_capacity(states.len());
    let mut records = Vec::<L1PackedProductBehaviorRecord>::new();
    let mut record_child_behaviors = Vec::<u32>::new();
    // Behavior interning is local to each trie node. Reusing the table avoids
    // an allocation and rehash-table growth for every branching node while
    // preserving the node-local behavior IDs and collision chains.
    let mut behavior_hash_heads = FxHashMap::<u64, u32>::default();
    let mut unary_behavior_ids = FxHashMap::<(u32, u32), u32>::default();
    let mut binary_behavior_ids = FxHashMap::<(u32, u32, u32), u32>::default();
    let mut small_unary_behavior_ids = Vec::<((u32, u32), u32)>::new();
    let mut small_binary_behavior_ids = Vec::<((u32, u32, u32), u32)>::new();
    let mut scratch_children = Vec::<u32>::new();
    const SMALL_NODE_LINEAR_INTERN_LIMIT: usize = 16;
    for &node_index in active_nodes.iter().rev() {
        let node = trie.nodes[node_index];
        let state_start = data[node_index].states_start as usize;
        let state_len = data[node_index].states_len as usize;
        data[node_index].behaviors_start = behavior_ids.len() as u32;
        data[node_index].records_start = records.len() as u32;
        if state_len == 0 {
            continue;
        }
        if node.edge_len == 0 {
            for state_offset in 0..state_len {
                behavior_ids.push(state_to_terminal_signature[states[state_start + state_offset] as usize]);
            }
            continue;
        }
        if node.terminal_token == L1_NONE && node.edge_len == 1 {
            let edge_index = node.first_edge as usize;
            let child = trie.edges[edge_index].child as usize;
            let child_behavior_start = data[child].behaviors_start as usize;
            let map_start = edge_data[edge_index].map_start as usize;
            for state_offset in 0..state_len {
                let child_state_index = transition_maps[map_start + state_offset];
                behavior_ids.push(if child_state_index == L1_NONE {
                    0
                } else if l1_uniform_behavior_signature(child_state_index).is_some() {
                    child_state_index
                } else {
                    behavior_ids[child_behavior_start + child_state_index as usize]
                });
            }
            continue;
        }
        if node.edge_len == 1 {
            let edge_index = node.first_edge as usize;
            let child = trie.edges[edge_index].child as usize;
            let child_behavior_start = data[child].behaviors_start as usize;
            let map_start = edge_data[edge_index].map_start as usize;
            let use_linear_intern = state_len <= SMALL_NODE_LINEAR_INTERN_LIMIT;
            if use_linear_intern {
                small_unary_behavior_ids.clear();
            } else {
                unary_behavior_ids.clear();
            }
            for state_offset in 0..state_len {
                let state = states[state_start + state_offset];
                let terminal_signature = state_to_terminal_signature[state as usize];
                let child_state_index = transition_maps[map_start + state_offset];
                let child_behavior = if child_state_index == L1_NONE {
                    0
                } else if l1_uniform_behavior_signature(child_state_index).is_some() {
                    child_state_index
                } else {
                    behavior_ids[child_behavior_start + child_state_index as usize]
                };
                if terminal_signature == 0 && child_behavior == 0 {
                    behavior_ids.push(0);
                    continue;
                }
                let key = (terminal_signature, child_behavior);
                let existing = if use_linear_intern {
                    small_unary_behavior_ids
                        .iter()
                        .find_map(|(candidate, id)| (*candidate == key).then_some(*id))
                } else {
                    unary_behavior_ids.get(&key).copied()
                };
                let behavior_id = if let Some(id) = existing {
                    id
                } else {
                    let child_behaviors_start = record_child_behaviors.len() as u32;
                    record_child_behaviors.push(child_behavior);
                    let child_uniform = l1_packed_uniform_signature(
                        &trie,
                        child,
                        child_behavior,
                        &data,
                        &records,
                    );
                    let uniform_signature = if terminal_signature != 0
                        && child_uniform == terminal_signature
                    {
                        terminal_signature
                    } else {
                        0
                    };
                    let id = records.len() as u32 - data[node_index].records_start + 1;
                    records.push(L1PackedProductBehaviorRecord {
                        terminal_signature,
                        child_behaviors_start,
                        uniform_signature,
                        hash_next: L1_NONE,
                    });
                    if use_linear_intern {
                        small_unary_behavior_ids.push((key, id));
                    } else {
                        unary_behavior_ids.insert(key, id);
                    }
                    id
                };
                behavior_ids.push(behavior_id);
            }
            continue;
        }
        if node.edge_len == 2 {
            let first_edge_index = node.first_edge as usize;
            let second_edge_index = first_edge_index + 1;
            let first_child = trie.edges[first_edge_index].child as usize;
            let second_child = trie.edges[second_edge_index].child as usize;
            let first_behavior_start = data[first_child].behaviors_start as usize;
            let second_behavior_start = data[second_child].behaviors_start as usize;
            let first_map_start = edge_data[first_edge_index].map_start as usize;
            let second_map_start = edge_data[second_edge_index].map_start as usize;
            let use_linear_intern = state_len <= SMALL_NODE_LINEAR_INTERN_LIMIT;
            if use_linear_intern {
                small_binary_behavior_ids.clear();
            } else {
                binary_behavior_ids.clear();
            }
            for state_offset in 0..state_len {
                let state = states[state_start + state_offset];
                let terminal_signature = if node.terminal_token == L1_NONE {
                    0
                } else {
                    state_to_terminal_signature[state as usize]
                };
                let first_state_index = transition_maps[first_map_start + state_offset];
                let first_behavior = if first_state_index == L1_NONE {
                    0
                } else if l1_uniform_behavior_signature(first_state_index).is_some() {
                    first_state_index
                } else {
                    behavior_ids[first_behavior_start + first_state_index as usize]
                };
                let second_state_index = transition_maps[second_map_start + state_offset];
                let second_behavior = if second_state_index == L1_NONE {
                    0
                } else if l1_uniform_behavior_signature(second_state_index).is_some() {
                    second_state_index
                } else {
                    behavior_ids[second_behavior_start + second_state_index as usize]
                };
                if terminal_signature == 0 && first_behavior == 0 && second_behavior == 0 {
                    behavior_ids.push(0);
                    continue;
                }
                let key = (terminal_signature, first_behavior, second_behavior);
                let existing = if use_linear_intern {
                    small_binary_behavior_ids
                        .iter()
                        .find_map(|(candidate, id)| (*candidate == key).then_some(*id))
                } else {
                    binary_behavior_ids.get(&key).copied()
                };
                let behavior_id = if let Some(id) = existing {
                    id
                } else {
                    let child_behaviors_start = record_child_behaviors.len() as u32;
                    record_child_behaviors.push(first_behavior);
                    record_child_behaviors.push(second_behavior);
                    let first_uniform = l1_packed_uniform_signature(
                        &trie,
                        first_child,
                        first_behavior,
                        &data,
                        &records,
                    );
                    let second_uniform = l1_packed_uniform_signature(
                        &trie,
                        second_child,
                        second_behavior,
                        &data,
                        &records,
                    );
                    let uniform_signature = if node.terminal_token == L1_NONE {
                        if first_uniform != 0 && first_uniform == second_uniform {
                            first_uniform
                        } else {
                            0
                        }
                    } else if terminal_signature != 0
                        && first_uniform == terminal_signature
                        && second_uniform == terminal_signature
                    {
                        terminal_signature
                    } else {
                        0
                    };
                    let id = records.len() as u32 - data[node_index].records_start + 1;
                    records.push(L1PackedProductBehaviorRecord {
                        terminal_signature,
                        child_behaviors_start,
                        uniform_signature,
                        hash_next: L1_NONE,
                    });
                    if use_linear_intern {
                        small_binary_behavior_ids.push((key, id));
                    } else {
                        binary_behavior_ids.insert(key, id);
                    }
                    id
                };
                behavior_ids.push(behavior_id);
            }
            continue;
        }

        let child_count = node.edge_len as usize;
        let use_linear_intern = state_len <= SMALL_NODE_LINEAR_INTERN_LIMIT;
        if !use_linear_intern {
            behavior_hash_heads.clear();
        }
        scratch_children.clear();
        if scratch_children.capacity() < child_count {
            scratch_children.reserve(child_count - scratch_children.capacity());
        }
        for state_offset in 0..state_len {
            let state = states[state_start + state_offset];
            let terminal_signature = if node.terminal_token == L1_NONE {
                0
            } else {
                state_to_terminal_signature[state as usize]
            };
            scratch_children.clear();
            for edge_offset in 0..child_count {
                let edge_index = node.first_edge as usize + edge_offset;
                let child = trie.edges[edge_index].child as usize;
                let child_state_index = transition_maps[edge_data[edge_index].map_start as usize + state_offset];
                scratch_children.push(if child_state_index == L1_NONE {
                    0
                } else if l1_uniform_behavior_signature(child_state_index).is_some() {
                    child_state_index
                } else {
                    behavior_ids[data[child].behaviors_start as usize + child_state_index as usize]
                });
            }
            if terminal_signature == 0 && scratch_children.iter().all(|&id| id == 0) {
                behavior_ids.push(0);
                continue;
            }
            let hash = (!use_linear_intern)
                .then(|| l1_packed_hash_behavior(terminal_signature, &scratch_children));
            let mut found = None;
            if use_linear_intern {
                let records_start = data[node_index].records_start as usize;
                for (offset, record) in records[records_start..].iter().enumerate() {
                    if record.terminal_signature == terminal_signature
                        && record_child_behaviors[record.child_behaviors_start as usize
                            ..record.child_behaviors_start as usize + child_count]
                            == scratch_children[..]
                    {
                        found = Some(offset as u32 + 1);
                        break;
                    }
                }
            } else {
                let hash = hash.expect("hashed behavior interning");
                let mut candidate = behavior_hash_heads.get(&hash).copied().unwrap_or(L1_NONE);
                while candidate != L1_NONE {
                    let record = records
                        [data[node_index].records_start as usize + candidate as usize - 1];
                    if record.terminal_signature == terminal_signature
                        && record_child_behaviors[record.child_behaviors_start as usize
                            ..record.child_behaviors_start as usize + child_count]
                            == scratch_children[..]
                    {
                        found = Some(candidate);
                        break;
                    }
                    candidate = record.hash_next;
                }
            }
            let behavior_id = if let Some(id) = found {
                id
            } else {
                let child_behaviors_start = record_child_behaviors.len() as u32;
                record_child_behaviors.extend_from_slice(&scratch_children);
                let mut uniform_signature = if node.terminal_token == L1_NONE {
                    0
                } else {
                    terminal_signature
                };
                if uniform_signature != 0 {
                    for edge_offset in 0..child_count {
                        let edge_index = node.first_edge as usize + edge_offset;
                        let child = trie.edges[edge_index].child as usize;
                        let child_uniform = l1_packed_uniform_signature(
                            &trie,
                            child,
                            scratch_children[edge_offset],
                            &data,
                            &records,
                        );
                        if child_uniform != uniform_signature {
                            uniform_signature = 0;
                            break;
                        }
                    }
                } else if node.terminal_token == L1_NONE {
                    let mut candidate_uniform = 0u32;
                    for edge_offset in 0..child_count {
                        let edge_index = node.first_edge as usize + edge_offset;
                        let child = trie.edges[edge_index].child as usize;
                        let child_uniform = l1_packed_uniform_signature(
                            &trie,
                            child,
                            scratch_children[edge_offset],
                            &data,
                            &records,
                        );
                        if child_uniform == 0
                            || (candidate_uniform != 0 && candidate_uniform != child_uniform)
                        {
                            candidate_uniform = 0;
                            break;
                        }
                        candidate_uniform = child_uniform;
                    }
                    uniform_signature = candidate_uniform;
                }
                let id = records.len() as u32 - data[node_index].records_start + 1;
                let hash_next = if let Some(hash) = hash {
                    behavior_hash_heads.get(&hash).copied().unwrap_or(L1_NONE)
                } else {
                    L1_NONE
                };
                records.push(L1PackedProductBehaviorRecord {
                    terminal_signature,
                    child_behaviors_start,
                    uniform_signature,
                    hash_next,
                });
                if let Some(hash) = hash {
                    behavior_hash_heads.insert(hash, id);
                }
                id
            };
            behavior_ids.push(behavior_id);
        }
    }

    let behavior_ms = behavior_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let materialize_started_at = profiling.then(Instant::now);
    let root_behavior_start = data[0].behaviors_start as usize;
    let mut profiles_by_behavior = FxHashMap::<u32, Arc<[(u32, u32, u32)]>>::default();
    for (target_index, &target) in walk_targets.iter().enumerate() {
        let behavior_id = behavior_ids[root_behavior_start + target_index];
        let profile = if let Some(profile) = profiles_by_behavior.get(&behavior_id) {
            Arc::clone(profile)
        } else {
            let uniform_signature = l1_packed_uniform_signature(
                &trie,
                0,
                behavior_id,
                &data,
                &records,
            );
            let profile = if uniform_signature != 0 {
                l1_uniform_bucket_profile(
                    &mut uniform_profiles_by_signature,
                    uniform_signature,
                    token_start,
                    token_end,
                )
            } else {
                let mut profile = Vec::new();
                l1_packed_append_behavior_reference(
                    &trie,
                    0,
                    behavior_id,
                    &data,
                    &records,
                    &record_child_behaviors,
                    &mut profile,
                );
                Arc::from(profile)
            };
            profiles_by_behavior.insert(behavior_id, Arc::clone(&profile));
            profile
        };
        if let Some(ref others) = dedup_others {
            for &other_target in &others[target_index] {
                results.push(((first_byte, other_target), Arc::clone(&profile)));
            }
        }
        results.push(((first_byte, target), profile));
    }
    let materialize_ms = materialize_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][l1_packed_product] first_byte={} suffix_horizon={} tokens={} targets={} trie_nodes={} active_nodes={} states={} states_capacity={} transition_maps={} transition_maps_capacity={} behavior_ids={} behavior_ids_capacity={} records={} record_child_behaviors={} uniform_subtree_transitions={} trie_ms={:.3} propagate_ms={:.3} behavior_ms={:.3} materialize_ms={:.3} total_ms={:.3}",
            first_byte,
            suffix_horizon,
            token_ids.len(),
            walk_targets.len(),
            trie.nodes.len(),
            active_nodes.len(),
            states.len(),
            states.capacity(),
            transition_maps.len(),
            transition_maps.capacity(),
            behavior_ids.len(),
            behavior_ids.capacity(),
            records.len(),
            record_child_behaviors.len(),
            uniform_subtree_transitions,
            trie_ms,
            propagate_ms,
            behavior_ms,
            materialize_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    results
}


#[derive(Debug)]
struct L1SortedTokenBuckets {
    empty_token_indices: Vec<usize>,
    token_indices_by_first_byte: Vec<Vec<usize>>,
    suffix_lcps_by_first_byte: Vec<Vec<usize>>,
    suffix_subtree_bytes: Vec<[u64; 4]>,
    suffix_first_bytes_by_bucket: Vec<[u64; 4]>,
    has_empty_suffix_by_bucket: Vec<bool>,
    packed_suffix_tries_by_first_byte: Vec<OnceLock<Arc<L1PackedSuffixTrie>>>,
}

fn build_l1_sorted_token_buckets(sorted_entries: &[(u32, Arc<[u8]>)]) -> L1SortedTokenBuckets {
    let mut empty_token_indices = Vec::<usize>::new();
    let mut token_indices_by_first_byte = vec![Vec::<usize>::new(); 256];
    for (internal_token_id, (_original_id, token_bytes)) in sorted_entries.iter().enumerate() {
        if let Some(&first_byte) = token_bytes.first() {
            token_indices_by_first_byte[first_byte as usize].push(internal_token_id);
        } else {
            empty_token_indices.push(internal_token_id);
        }
    }

    let mut suffix_lcps_by_first_byte = vec![Vec::<usize>::new(); 256];
    let mut suffix_subtree_bytes: Vec<[u64; 4]> = vec![[0u64; 4]; 256];
    let mut suffix_first_bytes_by_bucket: Vec<[u64; 4]> = vec![[0u64; 4]; 256];
    let mut has_empty_suffix_by_bucket = vec![false; 256];
    for first_byte in 0..256 {
        let token_ids = &token_indices_by_first_byte[first_byte];
        if token_ids.is_empty() {
            continue;
        }

        let lcps = &mut suffix_lcps_by_first_byte[first_byte];
        lcps.reserve(token_ids.len());

        let mut previous_suffix: &[u8] = &[];
        for &internal_token_id in token_ids {
            let token_bytes = &sorted_entries[internal_token_id].1;
            let suffix_bytes = &token_bytes[1..];
            lcps.push(common_prefix_len(previous_suffix, suffix_bytes));
            if suffix_bytes.is_empty() {
                has_empty_suffix_by_bucket[first_byte] = true;
            } else {
                let b = suffix_bytes[0];
                suffix_first_bytes_by_bucket[first_byte][b as usize >> 6] |= 1u64 << (b & 63);
                for &byte in suffix_bytes {
                    suffix_subtree_bytes[first_byte][byte as usize >> 6] |= 1u64 << (byte & 63);
                }
            }
            previous_suffix = suffix_bytes;
        }
    }

    let packed_suffix_tries_by_first_byte =
        (0..256).map(|_| OnceLock::new()).collect();

    L1SortedTokenBuckets {
        empty_token_indices,
        token_indices_by_first_byte,
        suffix_lcps_by_first_byte,
        suffix_subtree_bytes,
        suffix_first_bytes_by_bucket,
        has_empty_suffix_by_bucket,
        packed_suffix_tries_by_first_byte,
    }
}

fn collect_active_terminal_signature(
    tokenizer: &Tokenizer,
    state: u32,
    active_terminals: &[bool],
) -> Vec<u32> {
    let mut signature = Vec::<u32>::new();
    for tid in tokenizer.matched_terminals_iter(state) {
        if active_terminals.get(tid as usize).copied().unwrap_or(false) {
            signature.push(tid);
        }
    }
    for tid in tokenizer.tokens_accessible_from_state(state).iter() {
        if active_terminals.get(tid).copied().unwrap_or(false) {
            signature.push(tid as u32);
        }
    }
    signature.sort_unstable();
    signature.dedup();
    signature
}

/// Intern the exact active-terminal signature of every tokenizer state.
///
/// Signature zero is the empty signature. Keeping the reverse table lets the
/// L1 terminal-DWA builder reuse the same proof object rather than redoing this
/// full state scan with a separately numbered signature map.
fn build_l1_state_to_terminal_signatures(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
) -> (Vec<u32>, Vec<Vec<u32>>) {
    let mut signature_to_id = FxHashMap::<Vec<u32>, u32>::default();
    signature_to_id.insert(Vec::new(), 0);
    let mut terminal_signatures = vec![Vec::new()];
    let mut state_to_terminal_signature = vec![0u32; tokenizer.num_states() as usize];

    for state in 0..tokenizer.num_states() as usize {
        let signature =
            collect_active_terminal_signature(tokenizer, state as u32, active_terminals);
        let sig_id = match signature_to_id.get(&signature) {
            Some(&id) => id,
            None => {
                let next_signature_id = terminal_signatures.len() as u32;
                signature_to_id.insert(signature.clone(), next_signature_id);
                terminal_signatures.push(signature);
                next_signature_id
            }
        };
        state_to_terminal_signature[state] = sig_id;
    }

    if compile_profile_enabled() {
        let mut nonempty_states = 0usize;
        let mut singleton_states = 0usize;
        let mut membership_sum = 0usize;
        let mut max_membership = 0usize;
        for &signature_id in &state_to_terminal_signature {
            let len = terminal_signatures[signature_id as usize].len();
            if len != 0 {
                nonempty_states += 1;
                membership_sum += len;
                max_membership = max_membership.max(len);
                if len == 1 {
                    singleton_states += 1;
                }
            }
        }
        let mut signature_histogram = [0usize; 6];
        for signature in &terminal_signatures {
            let bucket = match signature.len() {
                0 => 0,
                1 => 1,
                2..=4 => 2,
                5..=16 => 3,
                17..=64 => 4,
                _ => 5,
            };
            signature_histogram[bucket] += 1;
        }
        eprintln!(
            "[glrmask/profile][l1_terminal_signatures] states={} signature_ids={} nonempty_states={} singleton_states={} membership_sum={} max_membership={} signature_histogram={:?}",
            state_to_terminal_signature.len(),
            terminal_signatures.len(),
            nonempty_states,
            singleton_states,
            membership_sum,
            max_membership,
            signature_histogram,
        );
    }

    (state_to_terminal_signature, terminal_signatures)
}

fn build_l1_flat_state_to_terminal_signatures(
    dfa: &FlatDfa,
) -> (Vec<u32>, Vec<Vec<u32>>) {
    let mut signature_to_id = FxHashMap::<Vec<u32>, u32>::default();
    signature_to_id.insert(Vec::new(), 0);
    let mut terminal_signatures = vec![Vec::new()];
    let mut state_to_terminal_signature = vec![0u32; dfa.states.len()];

    for (state, metadata) in dfa.states.iter().enumerate() {
        let mut signature = metadata
            .finalizers
            .iter()
            .chain(&metadata.possible_future_group_ids)
            .map(|&terminal| terminal as u32)
            .collect::<Vec<_>>();
        signature.sort_unstable();
        signature.dedup();
        let signature_id = if let Some(&id) = signature_to_id.get(&signature) {
            id
        } else {
            let id = terminal_signatures.len() as u32;
            signature_to_id.insert(signature.clone(), id);
            terminal_signatures.push(signature);
            id
        };
        state_to_terminal_signature[state] = signature_id;
    }

    (state_to_terminal_signature, terminal_signatures)
}

fn build_l1_tokenizer_state_to_terminal_signatures(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
) -> (Vec<u32>, Vec<Vec<u32>>, Vec<bool>) {
    let mut signature_to_id = FxHashMap::<Vec<u32>, u32>::default();
    signature_to_id.insert(Vec::new(), 0);
    let mut terminal_signatures = vec![Vec::new()];
    let mut state_to_terminal_signature = vec![0u32; tokenizer.num_states() as usize];
    let mut active_language = vec![false; tokenizer.num_states() as usize];

    for state in 0..tokenizer.num_states() {
        let mut signature = tokenizer
            .matched_terminals_iter(state)
            .chain(tokenizer.possible_future_terminals_iter(state))
            .filter(|&terminal| {
                active_terminals
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        signature.sort_unstable();
        signature.dedup();
        active_language[state as usize] = !signature.is_empty();
        let signature_id = if let Some(&id) = signature_to_id.get(&signature) {
            id
        } else {
            let id = terminal_signatures.len() as u32;
            signature_to_id.insert(signature.clone(), id);
            terminal_signatures.push(signature);
            id
        };
        state_to_terminal_signature[state as usize] = signature_id;
    }

    (state_to_terminal_signature, terminal_signatures, active_language)
}


fn l1_token_signature_profile_for_state(
    start_state: u32,
    sorted_entries: &[(u32, Arc<[u8]>)],
    buckets: &L1SortedTokenBuckets,
    state_to_terminal_signature: &[u32],
    flat_trans: &[u32],
) -> Vec<(u32, u32, u32)> {
    let dead = u32::MAX;
    let mut profile = Vec::<(u32, u32, u32)>::new();
    let start_sig = state_to_terminal_signature[start_state as usize];
    for &internal_token_id in &buckets.empty_token_indices {
        append_l1_signature_profile_run(&mut profile, start_sig, internal_token_id as u32);
    }

    for (first_byte, token_ids) in buckets.token_indices_by_first_byte.iter().enumerate() {
        if token_ids.is_empty() {
            continue;
        }

        let first_target = flat_trans[start_state as usize * 256 + first_byte];
        if first_target == dead {
            for &internal_token_id in token_ids {
                append_l1_signature_profile_run(&mut profile, 0, internal_token_id as u32);
            }
            continue;
        }

        let suffix_lcps = &buckets.suffix_lcps_by_first_byte[first_byte];
        let mut suffix_states = vec![first_target];
        for (bucket_pos, &internal_token_id) in token_ids.iter().enumerate() {
            let suffix_bytes = &sorted_entries[internal_token_id].1[1..];
            let lcp_len = suffix_lcps[bucket_pos].min(suffix_states.len().saturating_sub(1));
            suffix_states.truncate(lcp_len + 1);

            let mut state = *suffix_states.last().unwrap_or(&first_target);
            if state == dead {
                suffix_states.resize(suffix_bytes.len() + 1, dead);
            } else {
                for &byte in &suffix_bytes[lcp_len..] {
                    state = flat_trans[state as usize * 256 + byte as usize];
                    suffix_states.push(state);
                    if state == dead {
                        suffix_states.resize(suffix_bytes.len() + 1, dead);
                        break;
                    }
                }
            }

            let final_state = suffix_states[suffix_bytes.len()];
            let sig_id = if final_state == dead {
                0
            } else {
                state_to_terminal_signature[final_state as usize]
            };
            append_l1_signature_profile_run(&mut profile, sig_id, internal_token_id as u32);
        }
    }

    profile
}

fn append_l1_signature_profile_run(profile: &mut Vec<(u32, u32, u32)>, sig_id: u32, token_id: u32) {
    if let Some((last_sig, _start, end)) = profile.last_mut() {
        if *last_sig == sig_id && end.wrapping_add(1) == token_id {
            *end = token_id;
            return;
        }
    }
    profile.push((sig_id, token_id, token_id));
}

#[inline]
fn append_l1_signature_profile_range(
    profile: &mut Vec<(u32, u32, u32)>,
    sig_id: u32,
    token_start: u32,
    token_end: u32,
) {
    debug_assert!(token_start != L1_NONE);
    debug_assert!(token_end >= token_start);
    append_l1_signature_profile_run(profile, sig_id, token_start);
    if token_end != token_start {
        profile.last_mut().expect("packed profile range").2 = token_end;
    }
}

fn build_l1_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab_order: &L1IdentityVocabOrder,
    id_map: &mut InternalIdMap,
    num_terminals: u32,
    active_terminals: &[bool],
    flat_trans: &[u32],
    exact_profile_reuse: Option<&L1ExactProfileReuse>,
) -> Option<(DWA, L1TerminalBuildProfile)> {
    let total_started_at = std::time::Instant::now();
    let internal_vocab_ms = 0.0;
    let sorted_entries = vocab_order.token_entries_sorted.as_ref();
    let token_buckets = &vocab_order.token_buckets;

    if sorted_entries.is_empty() {
        return None;
    }

    let vocab_tree_build_ms = 0.0;

    let state_seed_started_at = Instant::now();
    let mut states_to_initial_tsids = FxHashMap::<u32, Vec<u32>>::default();
    let scalar_dispatch_roots = tokenizer
        .has_scalar_deterministic_dispatch()
        .then(|| tokenizer.deterministic_dispatch_roots())
        .flatten();
    let dispatch_profile_reuse = exact_profile_reuse.filter(|_| scalar_dispatch_roots.is_some());
    for (internal_tsid, representative_state) in id_map
        .tokenizer_states
        .iter_representative_ids()
        .enumerate()
    {
        if id_map.tokenizer_states.internal_to_originals[internal_tsid]
            .contains(&tokenizer.initial_state_id())
            && let Some(dispatch_roots) = scalar_dispatch_roots
        {
            for &dispatch_root in dispatch_roots {
                // A scoped boundary map deliberately omits reset-dispatch
                // roots owned only by other immediate components.  The
                // ordinary L1 builder historically expanded the global reset
                // to every dispatch root, which turns a component-local token
                // start into the link-wide post-commit reset.  Keep only roots
                // represented by this analysis domain.  Unscoped maps cover
                // every root, so ordinary compilation is unchanged.
                let root_internal = id_map.tokenizer_states.original_to_internal
                    [dispatch_root as usize];
                if root_internal == u32::MAX {
                    continue;
                }
                let start_state = if let Some(reuse) = dispatch_profile_reuse {
                    *reuse
                        .profile_representatives_by_internal
                        .get(root_internal as usize)
                        .expect("dispatch root refers to post-isolation-only L1 TSID")
                } else {
                    dispatch_root
                };
                states_to_initial_tsids
                    .entry(start_state)
                    .or_default()
                    .push(internal_tsid as u32);
            }
            continue;
        }
        let start_state = dispatch_profile_reuse.map_or(representative_state, |reuse| {
            *reuse
                .profile_representatives_by_internal
                .get(internal_tsid)
                .expect("ordinary structured-dispatch TSID missing pre-isolation profile representative")
        });
        states_to_initial_tsids
            .entry(start_state)
            .or_default()
            .push(internal_tsid as u32);
    }
    if dispatch_profile_reuse.is_some() {
        for tsids in states_to_initial_tsids.values_mut() {
            tsids.sort_unstable();
            tsids.dedup();
        }
    }
    let state_seed_ms = state_seed_started_at.elapsed().as_secs_f64() * 1000.0;
    let dead = u32::MAX;
    // Exact equivalence has already built this state-signature index on the
    // normal L1 path. Reuse it verbatim. The fallback keeps the direct builder
    // self-contained for callers that intentionally bypass exact-profile reuse.
    let fallback_signatures = exact_profile_reuse.is_none().then(|| {
        let (state_to_exact_signature, signatures_with_empty) =
            build_l1_state_to_terminal_signatures(tokenizer, active_terminals);
        let terminal_signatures: Vec<Vec<u32>> =
            signatures_with_empty.into_iter().skip(1).collect();
        let state_to_terminal_signature: Vec<u32> = state_to_exact_signature
            .into_iter()
            .map(|signature_id| {
                if signature_id == 0 {
                    u32::MAX
                } else {
                    signature_id - 1
                }
            })
            .collect();
        (terminal_signatures, state_to_terminal_signature)
    });
    let (terminal_signatures, state_to_terminal_signature): (&[Vec<u32>], &[u32]) =
        if let Some(reuse) = exact_profile_reuse {
            (
                reuse.direct_terminal_signatures.as_ref(),
                reuse.direct_state_to_terminal_signature.as_ref(),
            )
        } else {
            let (terminal_signatures, state_to_terminal_signature) = fallback_signatures
                .as_ref()
                .expect("missing fallback L1 terminal signatures");
            (terminal_signatures.as_slice(), state_to_terminal_signature.as_slice())
        };

    // Batch simulation: for each unique start state, simulate all tokens through
    // the DFA and accumulate terminal_signature(final concrete state) → (tsid → token_ids).
    // Parallelized across start states using rayon.

    let traversal_started_at = Instant::now();

    // Parallel traversal: each start_state processed independently.
    // Each (terminal_signature, tsid) pair is unique across start groups since TSIDs
    // partition deterministically into start groups. We exploit this by using
    // Arc from the start and skipping merging entirely.
    let empty_token_indices = token_buckets.empty_token_indices.as_slice();
    let token_indices_by_first_byte = &token_buckets.token_indices_by_first_byte;
    let suffix_lcps_by_first_byte = &token_buckets.suffix_lcps_by_first_byte;
    let suffix_subtree_bytes = &token_buckets.suffix_subtree_bytes;
    let suffix_first_bytes_by_bucket = &token_buckets.suffix_first_bytes_by_bucket;
    let has_empty_suffix_by_bucket = &token_buckets.has_empty_suffix_by_bucket;
    // Walk cache: compute once per unique (first_byte, target) and cache
    // the raw merged ranges. Self-loop optimization: if the target state
    // has self-loops on all suffix bytes, all tokens end at the target
    // state and the walk can be skipped entirely.
    let walk_cache: Option<FxHashMap<(u8, u32), L1WalkProfile>> =
        if exact_profile_reuse.is_some() {
            None
        } else {
            Some({
    // Phase 1: Identify unique concrete (first_byte, target_state) pairs.
    // State equivalence is valid for whole-token walks from a start state,
    // but it is not necessarily closed over suffixes after the first byte.
    // Walk token suffixes from the concrete post-first-byte DFA state and
    // only map the final state back to a representative.
    let mut unique_targets: FxHashMap<(u8, u32), ()> = FxHashMap::default();
    for (&start_state, _) in &states_to_initial_tsids {
        for (byte, token_ids) in token_indices_by_first_byte.iter().enumerate() {
            if token_ids.is_empty() {
                continue;
            }
            let target_state = flat_trans[start_state as usize * 256 + byte];
            if target_state != dead {
                unique_targets
                    .entry((byte as u8, target_state))
                    .or_default();
            }
        }
    }
    let unique_walk_keys: Vec<(u8, u32)> = unique_targets.into_keys().collect();

    // Precompute self-loop mask per target state.
    let mut self_loop_masks: FxHashMap<u32, [u64; 4]> = FxHashMap::default();
    for &(_, target) in &unique_walk_keys {
        self_loop_masks.entry(target).or_insert_with(|| {
            let mut mask = [0u64; 4];
            for byte in 0..=255u8 {
                if flat_trans[target as usize * 256 + byte as usize] == target {
                    mask[byte as usize >> 6] |= 1u64 << (byte & 63);
                }
            }
            mask
        });
    }

    // Parallel walk batched by first_byte: all targets for the same byte
    // are walked simultaneously in one pass over the token list.
    // This breaks the serial dependency chain across targets, enabling
    // memory-level parallelism (independent L2 accesses can overlap).
    {
        // Group unique walk keys by first_byte.
        let mut walks_by_byte: FxHashMap<u8, Vec<u32>> = FxHashMap::default();
        for &(byte, target) in &unique_walk_keys {
            walks_by_byte.entry(byte).or_default().push(target);
        }
        let byte_groups: Vec<(u8, Vec<u32>)> = walks_by_byte.into_iter().collect();

        let build_byte_batch = |(first_byte, all_targets): &(u8, Vec<u32>)| {
                let byte = *first_byte;
                let bucket_idx = byte as usize;
                let token_ids = &token_indices_by_first_byte[bucket_idx];
                let suffix_lcps = &suffix_lcps_by_first_byte[bucket_idx];
                let subtree = &suffix_subtree_bytes[byte as usize];

                // Separate self-loop targets from targets that need walking.
                let mut selfloop_targets: Vec<u32> = Vec::new();
                let mut walk_targets: Vec<u32> = Vec::new();
                for &target in all_targets {
                    let mask = &self_loop_masks[&target];
                    let can_skip = (subtree[0] & !mask[0]) == 0
                        && (subtree[1] & !mask[1]) == 0
                        && (subtree[2] & !mask[2]) == 0
                        && (subtree[3] & !mask[3]) == 0;
                    if can_skip {
                        selfloop_targets.push(target);
                    } else {
                        walk_targets.push(target);
                    }
                }

                let mut results: Vec<((u8, u32), Vec<(u32, Vec<(u32, u32)>)>)> = Vec::new();

                // Handle self-loop targets.
                if !selfloop_targets.is_empty() {
                    let first = *token_ids.first().unwrap() as u32;
                    let last = *token_ids.last().unwrap() as u32;
                    for &target in &selfloop_targets {
                        let sig_id = state_to_terminal_signature[target as usize];
                        if sig_id != u32::MAX {
                            results.push(((byte, target), vec![(sig_id, vec![(first, last)])]));
                        }
                    }
                }

                if walk_targets.is_empty() {
                    return results;
                }

                // Fingerprint dedup: group walk targets by their
                // first-suffix-byte transition pattern. Two targets that
                // transition to the same next-state for every first-suffix-byte
                // produce identical walk results (all subsequent walk steps
                // proceed from the same state). Targets that are dead on ALL
                // first-suffix-bytes produce empty results and are eliminated.
                // This is only valid when no single-byte tokens exist in this
                // bucket (empty suffix → final state = target state, which
                // differs between targets).
                let sfb = &suffix_first_bytes_by_bucket[bucket_idx];
                let has_empty = has_empty_suffix_by_bucket[bucket_idx];
                // rep_idx → list of non-representative targets in the same
                // fingerprint group (excludes the representative itself)
                let mut dedup_others: Option<Vec<Vec<u32>>> = None;
                if !has_empty {
                    // Collect unique suffix first bytes for fingerprint keys
                    let mut sfb_list: Vec<u8> = Vec::new();
                    for w in 0..4u8 {
                        let mut bits = sfb[w as usize];
                        while bits != 0 {
                            let offset = bits.trailing_zeros() as u8;
                            sfb_list.push(w * 64 + offset);
                            bits &= bits - 1;
                        }
                    }

                    if !sfb_list.is_empty() {
                        // Compute fingerprint for each target and group
                        let mut fp_groups: FxHashMap<Vec<u32>, Vec<u32>> = FxHashMap::default();
                        for &target in &walk_targets {
                            let fp: Vec<u32> = sfb_list
                                .iter()
                                .map(|&b| flat_trans[target as usize * 256 + b as usize])
                                .collect();
                            fp_groups.entry(fp).or_default().push(target);
                        }

                        // Separate dead groups (all entries are dead)
                        // from live groups
                        let mut deduped_targets: Vec<u32> = Vec::new();
                        let mut others: Vec<Vec<u32>> = Vec::new();
                        let mut dead_eliminated = 0usize;
                        let mut dup_eliminated = 0usize;

                        for (fp, group) in &fp_groups {
                            let all_dead = fp.iter().all(|&s| s == dead);
                            if all_dead {
                                // All targets in this group produce empty walk results
                                dead_eliminated += group.len();
                                continue;
                            }
                            let rep = group[0];
                            deduped_targets.push(rep);
                            let group_others: Vec<u32> = group[1..].to_vec();
                            dup_eliminated += group_others.len();
                            others.push(group_others);
                        }

                        let total_eliminated = dead_eliminated + dup_eliminated;
                        if total_eliminated > 0 {
                            dedup_others = Some(others);
                            walk_targets = deduped_targets;
                        }
                    }
                }

                if walk_targets.is_empty() {
                    return results;
                }

                let num_walk = walk_targets.len();

                // suffix_states: flat [pos * num_walk + target_idx]
                // Position 0 = initial target states (before any suffix bytes).
                let mut suffix_states: Vec<u32> = walk_targets.clone();

                // Per-target run-flush state.
                let mut run_signature_ids: Vec<u32> = vec![u32::MAX; num_walk];
                let mut run_starts: Vec<u32> = vec![0; num_walk];
                let mut run_ends: Vec<u32> = vec![0; num_walk];
                let mut signature_maps: Vec<FxHashMap<u32, Vec<(u32, u32)>>> =
                    (0..num_walk).map(|_| FxHashMap::default()).collect();

                for (bucket_pos, &internal_token_id) in token_ids.iter().enumerate() {
                    let suffix_bytes = &sorted_entries[internal_token_id].1[1..];
                    let lcp_len = suffix_lcps[bucket_pos];

                    // Truncate all targets to lcp_len + 1 positions.
                    suffix_states.truncate((lcp_len + 1) * num_walk);

                    // Walk remaining suffix bytes with all targets in parallel.
                    for byte_pos in lcp_len..suffix_bytes.len() {
                        let b = suffix_bytes[byte_pos];
                        let base = byte_pos * num_walk;
                        for t in 0..num_walk {
                            let prev_state = suffix_states[base + t];
                            let next_state = if prev_state == dead {
                                dead
                            } else {
                                flat_trans[prev_state as usize * 256 + b as usize]
                            };
                            suffix_states.push(next_state);
                        }
                    }

                    // Record final states for each target.
                    let end_base = suffix_bytes.len() * num_walk;
                    let token_id = internal_token_id as u32;
                    for t in 0..num_walk {
                        let final_state = suffix_states[end_base + t];
                        if final_state != dead {
                            let sig_id = state_to_terminal_signature[final_state as usize];
                            if sig_id == u32::MAX {
                                if run_signature_ids[t] != u32::MAX {
                                    signature_maps[t]
                                        .entry(run_signature_ids[t])
                                        .or_default()
                                        .push((run_starts[t], run_ends[t]));
                                    run_signature_ids[t] = u32::MAX;
                                }
                                continue;
                            }
                            if run_signature_ids[t] == sig_id
                                && run_ends[t].wrapping_add(1) == token_id
                            {
                                run_ends[t] = token_id;
                            } else {
                                // Flush previous run for this target.
                                if run_signature_ids[t] != u32::MAX {
                                    signature_maps[t]
                                        .entry(run_signature_ids[t])
                                        .or_default()
                                        .push((run_starts[t], run_ends[t]));
                                }
                                run_signature_ids[t] = sig_id;
                                run_starts[t] = token_id;
                                run_ends[t] = token_id;
                            }
                        } else {
                            // Dead: flush current run for this target.
                            if run_signature_ids[t] != u32::MAX {
                                signature_maps[t]
                                    .entry(run_signature_ids[t])
                                    .or_default()
                                    .push((run_starts[t], run_ends[t]));
                                run_signature_ids[t] = u32::MAX;
                            }
                        }
                    }
                }

                // Flush remaining runs.
                for t in 0..num_walk {
                    if run_signature_ids[t] != u32::MAX {
                        signature_maps[t]
                            .entry(run_signature_ids[t])
                            .or_default()
                            .push((run_starts[t], run_ends[t]));
                    }
                }

                // Package per-target results.
                for (t, map) in signature_maps.into_iter().enumerate() {
                    let target = walk_targets[t];
                    let entries: Vec<(u32, Vec<(u32, u32)>)> = map
                        .into_iter()
                        .map(|(sig_id, ranges)| {
                            debug_assert!(
                                ranges.windows(2).all(|w| w[0].1 < w[1].0),
                                "Phase 1 ranges should be sorted and non-overlapping"
                            );
                            (sig_id, ranges)
                        })
                        .collect();

                    // Expand results to all targets in the same fingerprint
                    // group if fingerprint dedup was applied.
                    if let Some(ref others) = dedup_others {
                        for &other_target in &others[t] {
                            results.push(((byte, other_target), entries.clone()));
                        }
                    }

                    results.push(((byte, target), entries));
                }

                results
            };
        let all_batches: Vec<Vec<((u8, u32), Vec<(u32, Vec<(u32, u32)>)>)>> =
            if rayon::current_num_threads() == 1 {
                byte_groups.iter().map(build_byte_batch).collect()
            } else {
                byte_groups.par_iter().map(build_byte_batch).collect()
            };

        let mut cache: FxHashMap<(u8, u32), Vec<(u32, Vec<(u32, u32)>)>> = FxHashMap::default();
        for batch in all_batches {
            for (key, value) in batch {
                cache.insert(key, value);
            }
        }
              cache
                  .into_iter()
                  .map(|(target, profile)| (target, freeze_l1_walk_profile_from_direct(profile)))
                  .collect()
          }
              })
          };

    // Build indexed walk_cache: (byte, target) → Vec of (signature_id, &ranges, entry_hash, entry_range_count).
    // entry_hash is precomputed from the ranges so Phase 2 can combine hashes
    // in O(entries) instead of O(ranges).
    let indexed_cache_started_at = compile_profile_enabled().then(Instant::now);
    let indexed_reuse_profiles: Option<Vec<Vec<(usize, &[(u32, u32)], u64, usize)>>> =
        exact_profile_reuse.map(|reuse| {
            reuse
                .walk_profiles_by_id
                .iter()
                .map(index_l1_walk_profile)
                .collect()
        });
    let indexed_walk_cache: Option<FxHashMap<(u8, u32), Vec<(usize, &[(u32, u32)], u64, usize)>>> =
        walk_cache.as_ref().map(|walk_cache| {
            walk_cache
                .iter()
                .map(|(&key, results)| (key, index_l1_walk_profile(results)))
                .collect()
        });
    if let Some(indexed_cache_started_at) = indexed_cache_started_at {
        eprintln!(
            "[glrmask/profile][l1_indexed_walk_cache] targets={} total_ms={:.3}",
            indexed_walk_cache
                .as_ref()
                .map_or_else(
                    || indexed_reuse_profiles.as_ref().map_or(0, Vec::len),
                    FxHashMap::len,
                ),
            indexed_cache_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    // Phase 2: For each start_state, collect walk_cache references per

    let start_states_list: Vec<(&u32, &Vec<u32>)> =
        states_to_initial_tsids.iter().collect();

    // Pre-build empty token ranges (shared across all start_states).
    let empty_token_ranges: Vec<(u32, u32)> = {
        let mut ranges = Vec::new();
        for &internal_token_id in empty_token_indices {
            append_token_id_range(&mut ranges, internal_token_id as u32);
        }
        ranges
    };
    // Precompute hash for empty token ranges.
    let empty_token_hash: u64 = {
        let mut h: u64 = 0;
        for &(s, e) in &empty_token_ranges {
            h = h.wrapping_add(range_hash_val(s, e));
        }
        (empty_token_ranges.len() as u64).wrapping_add(h)
    };

    let build_start_state_results = |&(&start_state, ref initial_tsids): &(&u32, &Vec<u32>)| {
            let collect_start = Instant::now();

            // Track only touched terminal signatures for this start state instead of
            // allocating all signature buckets every time.
            let mut touched_positions: FxHashMap<usize, usize> = FxHashMap::default();
            let mut touched_signatures: Vec<(usize, Vec<&[(u32, u32)]>, u64, usize)> = Vec::new();

            // Empty tokens: terminal signature at the start state.
            if !empty_token_ranges.is_empty() {
                let sig_id = state_to_terminal_signature[start_state as usize];
                if sig_id != u32::MAX {
                    append_l1_profile_entry(
                        &mut touched_positions,
                        &mut touched_signatures,
                        sig_id as usize,
                        empty_token_ranges.as_slice(),
                        empty_token_hash,
                        empty_token_ranges.len(),
                    );
                }
            }

            if let Some(reuse) = exact_profile_reuse {
                let profile_ids = reuse
                    .representative_profile_ids
                    .get(&start_state)
                    .expect("exact L1 profile reuse missing id-map representative");
                let profiles = indexed_reuse_profiles
                    .as_ref()
                    .expect("missing indexed exact L1 profiles");
                for &profile_id in profile_ids.iter() {
                    if profile_id == 0 {
                        continue;
                    }
                    for &(sig_idx, ranges, entry_hash, entry_mc) in
                        &profiles[profile_id as usize]
                    {
                        append_l1_profile_entry(
                            &mut touched_positions,
                            &mut touched_signatures,
                            sig_idx,
                            ranges,
                            entry_hash,
                            entry_mc,
                        );
                    }
                }
            } else {
                let cache = indexed_walk_cache
                    .as_ref()
                    .expect("missing fallback indexed L1 walk cache");
                for (byte, token_ids) in token_indices_by_first_byte.iter().enumerate() {
                    if token_ids.is_empty() {
                        continue;
                    }
                    let target_state = flat_trans[start_state as usize * 256 + byte];
                    if target_state == dead {
                        continue;
                    }
                    if let Some(results) = cache.get(&(byte as u8, target_state)) {
                        for &(sig_idx, ranges, entry_hash, entry_mc) in results {
                            append_l1_profile_entry(
                                &mut touched_positions,
                                &mut touched_signatures,
                                sig_idx,
                                ranges,
                                entry_hash,
                                entry_mc,
                            );
                        }
                    }
                }
            }

            // Finalize hashes and build LazyRanges entries.
            let mut result: Vec<(u32, u32, LazyRanges)> = Vec::new();
            for (sig_idx, refs, hash, total_len) in touched_signatures.into_iter() {
                let lazy = LazyRanges {
                    refs,
                    hash,
                    total_len,
                };
                if initial_tsids.len() > 1 {
                    for &tsid in &initial_tsids[..initial_tsids.len() - 1] {
                        result.push((sig_idx as u32, tsid, lazy.clone()));
                    }
                }
                result.push((sig_idx as u32, *initial_tsids.last().unwrap(), lazy));
            }
            result
        };
    // Large exact L1 quotients amortize dense reusable worker scratch: it avoids
    // one FxHashMap and one result Vec allocation per representative state. On
    // smaller quotients the fold/reduce setup costs more than those allocations,
    // so retain the established per-state path. Scalar deterministic dispatch
    // retains its newer dedicated path from main.
    let dense_exact_collection = exact_profile_reuse.is_some()
        && !tokenizer.has_scalar_deterministic_dispatch()
        && (rayon::current_num_threads() == 1
            || start_states_list.len() >= LARGE_EXACT_L1_ASSEMBLY_MIN_STATES);
    let mut all_entries: Vec<(u32, u32, LazyRanges<'_>)> = if dense_exact_collection {
        let reuse = exact_profile_reuse.expect("missing exact L1 profile reuse");
        let profiles = indexed_reuse_profiles
            .as_ref()
            .expect("missing indexed exact L1 profiles");
        if rayon::current_num_threads() == 1 {
            let mut touched_positions = vec![usize::MAX; terminal_signatures.len()];
            let mut touched_signature_ids = Vec::<usize>::new();
            let mut touched_signatures =
                Vec::<(usize, Vec<&[(u32, u32)]>, u64, usize)>::new();
            let mut entries = Vec::new();
            for &(&start_state, initial_tsids) in &start_states_list {
                append_l1_exact_start_entries(
                    start_state,
                    initial_tsids,
                    empty_token_ranges.as_slice(),
                    empty_token_hash,
                    state_to_terminal_signature,
                    reuse,
                    profiles,
                    &mut touched_positions,
                    &mut touched_signature_ids,
                    &mut touched_signatures,
                    &mut entries,
                );
            }
            entries
        } else {
            start_states_list
                .par_iter()
                .fold(
                    || {
                        (
                            vec![usize::MAX; terminal_signatures.len()],
                            Vec::<usize>::new(),
                            Vec::<(usize, Vec<&[(u32, u32)]>, u64, usize)>::new(),
                            Vec::<(u32, u32, LazyRanges<'_>)>::new(),
                        )
                    },
                    |(
                        mut touched_positions,
                        mut touched_signature_ids,
                        mut touched_signatures,
                        mut entries,
                    ),
                     item| {
                        let &(&start_state, initial_tsids) = item;
                        append_l1_exact_start_entries(
                            start_state,
                            initial_tsids,
                            empty_token_ranges.as_slice(),
                            empty_token_hash,
                            state_to_terminal_signature,
                            reuse,
                            profiles,
                            &mut touched_positions,
                            &mut touched_signature_ids,
                            &mut touched_signatures,
                            &mut entries,
                        );
                        (
                            touched_positions,
                            touched_signature_ids,
                            touched_signatures,
                            entries,
                        )
                    },
                )
                .map(|(_, _, _, entries)| entries)
                .reduce(Vec::new, |mut left, mut right| {
                    left.append(&mut right);
                    left
                })
        }
    } else {
        let per_thread_results: Vec<Vec<(u32, u32, LazyRanges<'_>)>> =
            if rayon::current_num_threads() == 1 {
                start_states_list.iter().map(build_start_state_results).collect()
            } else {
                start_states_list
                    .par_iter()
                    .map(build_start_state_results)
                    .collect()
            };
        per_thread_results.into_iter().flatten().collect()
    };

    let start_state_collect_ms = traversal_started_at.elapsed().as_secs_f64() * 1000.0;

    // Sort-based intern: sort entries by hash, find hash-group boundaries,
    // then verify and build Arcs in parallel. LazyRanges are compared by
    // ref identity (fast pointer comparison) and materialized only for
    // unique groups.
    let token_set_intern_started_at = Instant::now();

    // Sort by hash (fast u64 comparison). Equal hashes → same group candidate.
    all_entries.sort_unstable_by_key(|entry| entry.2.hash);

    // Find hash-group boundaries (sequential, fast — u64 comparison only).
    let mut hash_group_starts: Vec<usize> = vec![0];
    for k in 1..all_entries.len() {
        if all_entries[k].2.hash != all_entries[k - 1].2.hash {
            hash_group_starts.push(k);
        }
    }
    hash_group_starts.push(all_entries.len());

    // Process hash groups in parallel. Within each group, sub-group by
    // range equality. Cache the representative's materialization to avoid
    // re-materializing it for every comparison.
    let process_hash_group = |w: &[usize]| {
            let start = w[0];
            let end = w[1];
            let mut out = Vec::new();
            let mut sub_start = start;
            while sub_start < end {
                // Materialize the sub-group representative once.
                let rep_materialized = all_entries[sub_start].2.materialize();
                let mut sub_end = sub_start + 1;
                while sub_end < end {
                    // Fast path: ref pointer identity.
                    let candidate = &all_entries[sub_end].2;
                    let representative = &all_entries[sub_start].2;
                    let fast_match =
                        candidate.refs.len() == representative.refs.len()
                            && candidate.refs.iter().zip(representative.refs.iter()).all(
                                |(&a, &b)| {
                                    std::ptr::eq(a.as_ptr(), b.as_ptr()) && a.len() == b.len()
                                },
                            );
                    if fast_match {
                        sub_end += 1;
                        continue;
                    }
                    // Slow path: materialize candidate, compare with cached rep.
                    if candidate.materialize() == rep_materialized {
                        sub_end += 1;
                    } else {
                        break;
                    }
                }
                let arc: Arc<RangeSetBlaze<u32>> =
                    Arc::new(rep_materialized.iter().map(|&(s, e)| s..=e).collect());
                for k in sub_start..sub_end {
                    out.push((all_entries[k].0 as usize, all_entries[k].1, arc.clone()));
                }
                sub_start = sub_end;
            }
            out
        };
    let group_results: Vec<Vec<(usize, u32, Arc<RangeSetBlaze<u32>>)>> =
        if rayon::current_num_threads() == 1 {
            hash_group_starts.windows(2).map(process_hash_group).collect()
        } else {
            hash_group_starts
                .par_windows(2)
                .map(process_hash_group)
                .collect()
        };

    // Merge parallel results into deferred_arced (sequential).
    let mut deferred_arced: Vec<Vec<(u32, Arc<RangeSetBlaze<u32>>)>> =
        vec![Vec::new(); terminal_signatures.len()];
    for result in group_results {
        for (sig_id, tsid, arc) in result {
            deferred_arced[sig_id as usize].push((tsid, arc));
        }
    }

    let token_set_intern_ms = token_set_intern_started_at.elapsed().as_secs_f64() * 1000.0;
    let traversal_ms = traversal_started_at.elapsed().as_secs_f64() * 1000.0;

    let tsid_profile_merge_started_at = Instant::now();
    let tsid_profile_merge_before = id_map.num_tsids() as usize;
    let tsid_profile_merge_report = merge_deferred_equivalent_tsids(id_map, &mut deferred_arced);
    let tsid_profile_merge_after = tsid_profile_merge_report.tsids_after;
    let tsid_profile_merge_ms = tsid_profile_merge_started_at.elapsed().as_secs_f64() * 1000.0;

    let distribute_started_at = Instant::now();

    // Build terminal -> terminal-signature ids. Each signature is the exact
    // set of active terminals produced by the full-token end state.
    let mut terminal_to_signatures: Vec<Vec<u32>> = vec![Vec::new(); num_terminals as usize];
    for (sig_id, terminals) in terminal_signatures.iter().enumerate() {
        for &terminal in terminals {
            terminal_to_signatures[terminal as usize].push(sig_id as u32);
        }
    }

    // Pre-compute per-TSID full token set unions and contributing signatures.
    // For each terminal, TSIDs whose contributing signatures are all active reuse
    // the precomputed Arc; only TSIDs with some inactive signatures are recomputed.
    let merge_started_at = Instant::now();

    let num_tsids = id_map.num_tsids() as usize;

    // Build per-TSID: full ranges union + contributing signature count.
    let full_ranges_started_at = Instant::now();
    let mut tsid_full_ranges: Vec<Vec<(u32, u32)>> = (0..num_tsids).map(|_| Vec::new()).collect();
    let mut tsid_total_rep_counts = vec![0usize; num_tsids];
    for entries in &deferred_arced {
        for &(tsid, ref arc) in entries {
            tsid_total_rep_counts[tsid as usize] += 1;
            for r in arc.ranges() {
                tsid_full_ranges[tsid as usize].push((*r.start(), *r.end()));
            }
        }
    }
    let tsid_full_arc_cache: Vec<std::sync::OnceLock<Option<Arc<RangeSetBlaze<u32>>>>> =
        (0..num_tsids).map(|_| std::sync::OnceLock::new()).collect();

    let full_ranges_ms = full_ranges_started_at.elapsed().as_secs_f64() * 1000.0;

    // Group terminals by active_states to deduplicate identical computation
    let terminal_group_started_at = Instant::now();
    let mut active_tids: Vec<usize> = (0..terminal_to_signatures.len())
        .filter(|&i| !terminal_to_signatures[i].is_empty())
        .collect();
    active_tids
        .sort_unstable_by(|&a, &b| terminal_to_signatures[a].cmp(&terminal_to_signatures[b]));
    let mut unique_groups: Vec<Vec<usize>> = Vec::new();
    for &tid in &active_tids {
        if let Some(last) = unique_groups.last_mut() {
            if terminal_to_signatures[last[0]] == terminal_to_signatures[tid] {
                last.push(tid);
                continue;
            }
        }
        unique_groups.push(vec![tid]);
    }

    let terminal_group_ms = terminal_group_started_at.elapsed().as_secs_f64() * 1000.0;
    let num_groups = unique_groups.len();
    let contribution_seed_started_at = Instant::now();
    let signature_groups = build_end_rep_groups(
        &unique_groups,
        &terminal_to_signatures,
        deferred_arced.len(),
    );
    let mut tsid_group_contributions: Vec<Vec<(usize, Arc<RangeSetBlaze<u32>>)>> =
        (0..num_tsids).map(|_| Vec::new()).collect();
    for (sig_id, entries) in deferred_arced.iter().enumerate() {
        if signature_groups[sig_id].is_empty() {
            continue;
        }
        for &(tsid, ref arc) in entries {
            tsid_group_contributions[tsid as usize].push((sig_id, Arc::clone(arc)));
        }
    }

    let contribution_seed_ms = contribution_seed_started_at.elapsed().as_secs_f64() * 1000.0;
    let contribution_group_visits = tsid_group_contributions
        .iter()
        .flat_map(|contributions| contributions.iter())
        .map(|(sig_id, _)| signature_groups[*sig_id].len())
        .sum::<usize>();
    let per_tsid_group_entries_started_at = Instant::now();
    let direct_group_assembly = l1_sequential_group_assembly_override().unwrap_or_else(|| {
        auto_use_sequential_l1_group_assembly(
            contribution_group_visits,
            rayon::current_num_threads(),
        )
    });
    let group_weight_entries: Vec<Vec<(u32, Arc<RangeSetBlaze<u32>>)>> =
        if direct_group_assembly {
            // In the single-threaded case, avoid allocating `num_groups` fresh
            // counters and range buffers for every TSID. Reuse sparse scratch
            // and emit directly into each group, which is already ordered by
            // ascending TSID because this loop is sequential.
            let mut group_weight_entries: Vec<Vec<(u32, Arc<RangeSetBlaze<u32>>)>> =
                (0..num_groups).map(|_| Vec::new()).collect();
            let mut group_counts = vec![0usize; num_groups];
            let mut group_ranges: Vec<Vec<(u32, u32)>> =
                (0..num_groups).map(|_| Vec::new()).collect();
            // Most TSIDs contribute one signature, or have a terminal group
            // that sees only one of their few signatures. In that case the
            // group weight is exactly the existing shared range set: preserve
            // that Arc instead of copying, sorting, and rebuilding it.
            let mut group_single_arc: Vec<Option<Arc<RangeSetBlaze<u32>>>> =
                (0..num_groups).map(|_| None).collect();
            let mut touched_groups = Vec::<usize>::new();

            for (tsid, contributions) in tsid_group_contributions.iter().enumerate() {
                if contributions.is_empty() {
                    continue;
                }

                for &group_idx in &touched_groups {
                    group_counts[group_idx] = 0;
                    group_ranges[group_idx].clear();
                    group_single_arc[group_idx] = None;
                }
                touched_groups.clear();

                for &(sig_id, ref arc) in contributions {
                    for &group_idx in &signature_groups[sig_id] {
                        if group_counts[group_idx] == 0 {
                            touched_groups.push(group_idx);
                            group_counts[group_idx] = 1;
                            group_single_arc[group_idx] = Some(Arc::clone(arc));
                            continue;
                        }
                        if group_counts[group_idx] == 1
                            && tsid_total_rep_counts[tsid] != 2
                        {
                            group_ranges[group_idx].extend(
                                group_single_arc[group_idx]
                                    .as_ref()
                                    .expect("single group contribution")
                                    .ranges()
                                    .map(|range| (*range.start(), *range.end())),
                            );
                        }
                        group_single_arc[group_idx] = None;
                        group_counts[group_idx] += 1;
                        if tsid_total_rep_counts[tsid] != 2 {
                            group_ranges[group_idx].extend(
                                arc.ranges().map(|range| (*range.start(), *range.end())),
                            );
                        }
                    }
                }

                for &group_idx in &touched_groups {
                    let shared = if group_counts[group_idx] == 1 {
                        Some(Arc::clone(
                            group_single_arc[group_idx]
                                .as_ref()
                                .expect("single group contribution"),
                        ))
                    } else if group_counts[group_idx] == tsid_total_rep_counts[tsid] {
                        tsid_full_arc_cache[tsid]
                            .get_or_init(|| {
                                shared_rangeset_from_unsorted_pairs(
                                    tsid_full_ranges[tsid].as_slice(),
                                )
                            })
                            .clone()
                    } else {
                        shared_rangeset_from_unsorted_pairs(group_ranges[group_idx].as_slice())
                    };
                    if let Some(tokens) = shared {
                        group_weight_entries[group_idx].push((tsid as u32, tokens));
                    }
                }
            }

            group_weight_entries
        } else if exact_profile_reuse.is_some()
            && num_tsids >= LARGE_EXACT_L1_ASSEMBLY_MIN_STATES
        {
            // Reuse one sparse group-assembly scratch per Rayon worker and
            // accumulate directly into per-group outputs. The previous path
            // allocated `num_groups` counters/range vectors plus one result
            // vector for every TSID, even though each TSID touches only a
            // small subset of groups.
            let mut group_weight_entries = tsid_group_contributions
                .par_iter()
                .enumerate()
                .fold(
                    || {
                        (
                            vec![0usize; num_groups],
                            (0..num_groups)
                                .map(|_| Vec::<(u32, u32)>::new())
                                .collect::<Vec<_>>(),
                            (0..num_groups)
                                .map(|_| None::<Arc<RangeSetBlaze<u32>>>)
                                .collect::<Vec<_>>(),
                            Vec::<usize>::new(),
                            (0..num_groups)
                                .map(|_| Vec::<(u32, Arc<RangeSetBlaze<u32>>)>::new())
                                .collect::<Vec<_>>(),
                        )
                    },
                    |(
                        mut group_counts,
                        mut group_ranges,
                        mut group_single_arc,
                        mut touched_groups,
                        mut output,
                    ),
                     (tsid, contributions)| {
                        for &group_idx in &touched_groups {
                            group_counts[group_idx] = 0;
                            group_ranges[group_idx].clear();
                            group_single_arc[group_idx] = None;
                        }
                        touched_groups.clear();

                        for &(sig_id, ref arc) in contributions {
                            for &group_idx in &signature_groups[sig_id] {
                                if group_counts[group_idx] == 0 {
                                    touched_groups.push(group_idx);
                                    group_counts[group_idx] = 1;
                                    group_single_arc[group_idx] = Some(Arc::clone(arc));
                                    continue;
                                }

                                if group_counts[group_idx] == 1 {
                                    group_ranges[group_idx].extend(
                                        group_single_arc[group_idx]
                                            .as_ref()
                                            .expect("single group contribution")
                                            .ranges()
                                            .map(|range| (*range.start(), *range.end())),
                                    );
                                    group_single_arc[group_idx] = None;
                                }
                                group_counts[group_idx] += 1;
                                group_ranges[group_idx].extend(
                                    arc.ranges().map(|range| (*range.start(), *range.end())),
                                );
                            }
                        }

                        for &group_idx in &touched_groups {
                            let shared = if group_counts[group_idx] == 1 {
                                Some(Arc::clone(
                                    group_single_arc[group_idx]
                                        .as_ref()
                                        .expect("single group contribution"),
                                ))
                            } else if group_counts[group_idx] == tsid_total_rep_counts[tsid] {
                                tsid_full_arc_cache[tsid]
                                    .get_or_init(|| {
                                        shared_rangeset_from_unsorted_pairs(
                                            tsid_full_ranges[tsid].as_slice(),
                                        )
                                    })
                                    .clone()
                            } else {
                                shared_rangeset_from_unsorted_pairs(
                                    group_ranges[group_idx].as_slice(),
                                )
                            };
                            if let Some(tokens) = shared {
                                output[group_idx].push((tsid as u32, tokens));
                            }
                        }

                        (
                            group_counts,
                            group_ranges,
                            group_single_arc,
                            touched_groups,
                            output,
                        )
                    },
                )
                .map(|(_, _, _, _, output)| output)
                .reduce(
                    || {
                        (0..num_groups)
                            .map(|_| Vec::<(u32, Arc<RangeSetBlaze<u32>>)>::new())
                            .collect::<Vec<_>>()
                    },
                    |mut left, right| {
                        for (left_group, mut right_group) in left.iter_mut().zip(right) {
                            left_group.append(&mut right_group);
                        }
                        left
                    },
                );
            for entries in &mut group_weight_entries {
                entries.sort_unstable_by_key(|&(tsid, _)| tsid);
            }
            group_weight_entries
        } else {
            let per_tsid_group_entries: Vec<Vec<(usize, u32, Arc<RangeSetBlaze<u32>>)>> =
                tsid_group_contributions
                    .par_iter()
                    .enumerate()
                    .map(|(tsid, contributions)| {
                        if contributions.is_empty() {
                            return Vec::new();
                        }

                        let mut group_counts = vec![0usize; num_groups];
                        let mut group_ranges: Vec<Vec<(u32, u32)>> =
                            (0..num_groups).map(|_| Vec::new()).collect();
                        let mut touched_groups: Vec<usize> = Vec::new();

                        for &(sig_id, ref arc) in contributions {
                            for &group_idx in &signature_groups[sig_id] {
                                if group_counts[group_idx] == 0 {
                                    touched_groups.push(group_idx);
                                }
                                group_counts[group_idx] += 1;
                                for range in arc.ranges() {
                                    group_ranges[group_idx]
                                        .push((*range.start(), *range.end()));
                                }
                            }
                        }

                        touched_groups.sort_unstable();

                        let mut out: Vec<(usize, u32, Arc<RangeSetBlaze<u32>>)> = Vec::new();
                        out.reserve(touched_groups.len());
                        for group_idx in touched_groups {
                            let shared = if group_counts[group_idx] == tsid_total_rep_counts[tsid] {
                                tsid_full_arc_cache[tsid]
                                    .get_or_init(|| {
                                        shared_rangeset_from_unsorted_pairs(
                                            tsid_full_ranges[tsid].as_slice(),
                                        )
                                    })
                                    .clone()
                            } else {
                                shared_rangeset_from_unsorted_pairs(
                                    group_ranges[group_idx].as_slice(),
                                )
                            };
                            if let Some(tokens) = shared {
                                out.push((group_idx, tsid as u32, tokens));
                            }
                        }
                        out
                    })
                    .collect();

            let mut group_weight_entries: Vec<Vec<(u32, Arc<RangeSetBlaze<u32>>)>> =
                (0..unique_groups.len()).map(|_| Vec::new()).collect();
            for entries in per_tsid_group_entries {
                for (group_idx, tsid, tokens) in entries {
                    group_weight_entries[group_idx].push((tsid, tokens));
                }
            }
            for entries in &mut group_weight_entries {
                entries.sort_unstable_by_key(|&(tsid, _)| tsid);
            }
            group_weight_entries
        };
    let per_tsid_group_entries_ms =
        per_tsid_group_entries_started_at.elapsed().as_secs_f64() * 1000.0;
    let group_weight_entries_ms = 0.0;

    let group_results: Vec<Option<(Vec<usize>, Weight)>> = unique_groups
        .iter()
        .enumerate()
        .map(|(group_idx, tids)| {
            let weight_entries = &group_weight_entries[group_idx];
            if weight_entries.is_empty() {
                return None;
            }
            let weight = Weight::from_per_tsid_shared(
                weight_entries.iter().map(|(t, a)| (*t, Arc::clone(a))),
            );
            if weight.is_empty() {
                return None;
            }
            Some((tids.clone(), weight))
        })
        .collect();

    // Sequential DWA construction from grouped results
    let dwa_build_started_at = Instant::now();
    let mut dwa = DWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let end_state = dwa.add_state();
    dwa.set_final_weight(end_state, Weight::all());
    let mut num_transitions = 0usize;

    for result in group_results.into_iter().flatten() {
        let (tids, weight) = result;
        for &tid in &tids {
            dwa.add_transition(dwa.start_state(), tid as i32, end_state, weight.clone());
            num_transitions += 1;
        }
    }

    if num_transitions == 0 {
        return None;
    }

    let dwa_stats = dwa.stats();
    let dwa_build_ms = dwa_build_started_at.elapsed().as_secs_f64() * 1000.0;

    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l1_terminal_assembly] mode={} num_tsids={} num_groups={} contribution_group_visits={} start_collect_ms={:.3} token_intern_ms={:.3} full_ranges_ms={:.3} terminal_group_ms={:.3} contribution_seed_ms={:.3} per_tsid_group_entries_ms={:.3} group_weight_entries_ms={:.3} dwa_build_ms={:.3} total_direct_ms={:.3}",
            if direct_group_assembly { "sequential" } else { "parallel" },
            num_tsids,
            num_groups,
            contribution_group_visits,
            start_state_collect_ms,
            token_set_intern_ms,
            full_ranges_ms,
            terminal_group_ms,
            contribution_seed_ms,
            per_tsid_group_entries_ms,
            group_weight_entries_ms,
            dwa_build_ms,
            merge_ms,
        );
    }
    let direct_terminal_dwa_ms = merge_ms;
    let distribute_ms = distribute_started_at.elapsed().as_secs_f64() * 1000.0;
    let vocab_tree_traversal_ms = traversal_ms;

    Some((
        dwa,
        L1TerminalBuildProfile {
            internal_vocab_ms,
            vocab_tree_build_ms,
            state_seed_ms,
            token_set_intern_ms,
            tsid_profile_merge_ms,
            tsid_profile_merge_before,
            tsid_profile_merge_after,
            vocab_tree_traversal_ms,
            direct_terminal_dwa_ms,
        },
    ))
}

pub fn build_flat_transition_table(tokenizer: &Tokenizer) -> Vec<u32> {
    let mut flat_trans = vec![0; tokenizer.num_states() as usize * 256];
    for (state_idx, row) in flat_trans.chunks_exact_mut(256).enumerate() {
        let row: &mut [u32; 256] = row
            .try_into()
            .expect("flat tokenizer transition rows have fixed width");
        tokenizer.fill_transition_row(state_idx as u32, row);
    }
    flat_trans
}

fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    let limit = left.len().min(right.len());
    let mut index = 0usize;
    while index < limit && left[index] == right[index] {
        index += 1;
    }
    index
}

fn append_token_id_range(token_ranges: &mut Vec<(u32, u32)>, token_id: u32) {
    append_token_id_span(token_ranges, token_id, token_id);
}

fn append_token_id_span(token_ranges: &mut Vec<(u32, u32)>, start: u32, end: u32) {
    if let Some((_, last_end)) = token_ranges.last_mut() {
        if start <= last_end.saturating_add(1) {
            *last_end = (*last_end).max(end);
            return;
        }
    }
    token_ranges.push((start, end));
}

fn flush_end_rep_run(
    end_rep_token_ranges: &mut FxHashMap<u32, Vec<(u32, u32)>>,
    current_run_end_rep: &mut Option<u32>,
    current_run_start: &mut u32,
    current_run_end: &mut u32,
) {
    if let Some(end_rep) = current_run_end_rep.take() {
        append_token_id_span(
            end_rep_token_ranges.entry(end_rep).or_default(),
            *current_run_start,
            *current_run_end,
        );
    }
}

fn collect_l1_root_ranges_by_first_byte_lcp(
    start_state: u32,
    sorted_entries: &[(u32, Arc<[u8]>)],
    empty_token_indices: &[usize],
    token_indices_by_first_byte: &[Vec<usize>],
    flat_trans: &[u32],
    state_to_rep: &[u32],
    end_rep_token_ranges: &mut FxHashMap<u32, Vec<(u32, u32)>>,
) {
    let dead = u32::MAX;
    let start_rep = state_to_rep[start_state as usize];
    for &internal_token_id in empty_token_indices {
        append_token_id_range(
            end_rep_token_ranges.entry(start_rep).or_default(),
            internal_token_id as u32,
        );
    }

    for (first_byte, token_ids) in token_indices_by_first_byte.iter().enumerate() {
        if token_ids.is_empty() {
            continue;
        }

        let first_target = flat_trans[start_state as usize * 256 + first_byte];
        if first_target == dead {
            continue;
        }

        let mut previous_suffix: &[u8] = &[];
        let mut suffix_states = vec![first_target];
        for &internal_token_id in token_ids {
            let token_bytes = sorted_entries[internal_token_id].1.as_ref();
            let suffix_bytes = &token_bytes[1..];
            let lcp_len = common_prefix_len(previous_suffix, suffix_bytes);
            suffix_states.truncate(lcp_len + 1);

            let mut state = *suffix_states.last().unwrap_or(&first_target);
            if state == dead {
                suffix_states.resize(suffix_bytes.len() + 1, dead);
            } else {
                for &byte in &suffix_bytes[lcp_len..] {
                    state = flat_trans[state as usize * 256 + byte as usize];
                    suffix_states.push(state);
                    if state == dead {
                        suffix_states.resize(suffix_bytes.len() + 1, dead);
                        break;
                    }
                }
            }

            let final_state = suffix_states[suffix_bytes.len()];
            if final_state != dead {
                let end_rep = state_to_rep[final_state as usize];
                append_token_id_range(
                    end_rep_token_ranges.entry(end_rep).or_default(),
                    internal_token_id as u32,
                );
            }

            previous_suffix = suffix_bytes;
        }
    }
}

fn merge_ranges_in_place(ranges: &mut Vec<(u32, u32)>) {
    if ranges.is_empty() {
        return;
    }

    ranges.sort_unstable();
    let mut write_index = 0usize;
    for read_index in 1..ranges.len() {
        if ranges[read_index].0 <= ranges[write_index].1.saturating_add(1) {
            ranges[write_index].1 = ranges[write_index].1.max(ranges[read_index].1);
        } else {
            write_index += 1;
            ranges[write_index] = ranges[read_index];
        }
    }
    ranges.truncate(write_index + 1);
}

fn shared_rangeset_from_unsorted_pairs(ranges: &[(u32, u32)]) -> Option<Arc<RangeSetBlaze<u32>>> {
    if ranges.is_empty() {
        return None;
    }

    let mut merged = ranges.to_vec();
    merge_ranges_in_place(&mut merged);
    Some(shared_rangeset(
        merged.iter().map(|&(start, end)| start..=end).collect(),
    ))
}

fn build_end_rep_groups(
    unique_groups: &[Vec<usize>],
    terminal_to_active_states: &[Vec<u32>],
    num_end_reps: usize,
) -> Vec<Vec<usize>> {
    let mut groups_by_end_rep = vec![Vec::new(); num_end_reps];
    for (group_idx, tids) in unique_groups.iter().enumerate() {
        for &state in &terminal_to_active_states[tids[0]] {
            groups_by_end_rep[state as usize].push(group_idx);
        }
    }
    groups_by_end_rep
}

fn merge_deferred_equivalent_tsids(
    id_map: &mut InternalIdMap,
    deferred_arced: &mut [Vec<(u32, Arc<RangeSetBlaze<u32>>)>],
) -> L1TsidProfileMergeReport {
    let num_tsids = id_map.num_tsids() as usize;
    if num_tsids <= 1 {
        return L1TsidProfileMergeReport {
            tsids_after: num_tsids,
            unique_arc_token_sets: 0,
            unique_range_token_sets: 0,
            profile_build_ms: 0.0,
            group_ms: 0.0,
            remap_ms: 0.0,
        };
    }

    let profile_build_started_at = Instant::now();
    let mut profiles = vec![Vec::<(u32, u32)>::new(); num_tsids];
    let mut token_ctx_by_arc = FxHashMap::<usize, u32>::default();
    let mut next_token_ctx = 0u32;
    for (end_rep, entries) in deferred_arced.iter().enumerate() {
        for &(tsid, ref token_set) in entries {
            let arc_ptr = Arc::as_ptr(token_set) as usize;
            let token_ctx = *token_ctx_by_arc.entry(arc_ptr).or_insert_with(|| {
                let ctx = next_token_ctx;
                next_token_ctx += 1;
                ctx
            });
            profiles[tsid as usize].push((end_rep as u32, token_ctx));
        }
    }
    let profile_build_ms = profile_build_started_at.elapsed().as_secs_f64() * 1000.0;

    let group_started_at = Instant::now();
    let mut sorted_tsids: Vec<usize> = (0..num_tsids).collect();
    sorted_tsids.sort_by(|&left, &right| profiles[left].cmp(&profiles[right]));

    let mut tsid_perm = vec![0u32; num_tsids];
    let mut new_count = 1usize;
    tsid_perm[sorted_tsids[0]] = 0;
    for pair in sorted_tsids.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        if profiles[previous] != profiles[current] {
            new_count += 1;
        }
        tsid_perm[current] = (new_count - 1) as u32;
    }
    let group_ms = group_started_at.elapsed().as_secs_f64() * 1000.0;

    if new_count == num_tsids {
        return L1TsidProfileMergeReport {
            tsids_after: num_tsids,
            unique_arc_token_sets: token_ctx_by_arc.len(),
            unique_range_token_sets: token_ctx_by_arc.len(),
            profile_build_ms,
            group_ms,
            remap_ms: 0.0,
        };
    }

    let remap_started_at = Instant::now();
    apply_tsid_perm_to_id_map(&mut id_map.tokenizer_states, &tsid_perm, new_count);
    remap_deferred_arced_tsids(deferred_arced, &tsid_perm);
    let remap_ms = remap_started_at.elapsed().as_secs_f64() * 1000.0;

    L1TsidProfileMergeReport {
        tsids_after: new_count,
        unique_arc_token_sets: token_ctx_by_arc.len(),
        unique_range_token_sets: token_ctx_by_arc.len(),
        profile_build_ms,
        group_ms,
        remap_ms,
    }
}

fn remap_deferred_arced_tsids(
    deferred_arced: &mut [Vec<(u32, Arc<RangeSetBlaze<u32>>)>],
    tsid_perm: &[u32],
) {
    for entries in deferred_arced {
        if entries.is_empty() {
            continue;
        }

        let mut remapped: Vec<(u32, Arc<RangeSetBlaze<u32>>)> = std::mem::take(entries)
            .into_iter()
            .map(|(tsid, token_set)| (tsid_perm[tsid as usize], token_set))
            .collect();
        remapped.sort_unstable_by_key(|(tsid, _)| *tsid);

        let mut merged_entries = Vec::with_capacity(remapped.len());
        let mut idx = 0usize;
        while idx < remapped.len() {
            let tsid = remapped[idx].0;
            let token_set = Arc::clone(&remapped[idx].1);
            idx += 1;
            while idx < remapped.len() && remapped[idx].0 == tsid {
                idx += 1;
            }
            merged_entries.push((tsid, token_set));
        }

        *entries = merged_entries;
    }
}

fn apply_tsid_perm_to_id_map(id_map: &mut ManyToOneIdMap, perm: &[u32], new_count: usize) {
    let old_internal_to_originals = std::mem::take(&mut id_map.internal_to_originals);
    let old_representatives = std::mem::take(&mut id_map.representative_original_ids);

    for internal in &mut id_map.original_to_internal {
        if *internal != u32::MAX {
            *internal = perm[*internal as usize];
        }
    }

    let mut new_internal_to_originals = vec![Vec::new(); new_count];
    let mut new_representatives = vec![u32::MAX; new_count];
    for (old_internal, originals) in old_internal_to_originals.into_iter().enumerate() {
        let new_internal = perm[old_internal] as usize;
        new_internal_to_originals[new_internal].extend(originals);
        if new_representatives[new_internal] == u32::MAX {
            new_representatives[new_internal] = old_representatives[old_internal];
        }
    }

    id_map.internal_to_originals = new_internal_to_originals;
    id_map.representative_original_ids = new_representatives;
}

struct L1IdMapProfile {
    initial_states_considered: usize,
    max_length_skipped: bool,
    max_token_len: usize,
    token_len_gt_4: usize,
    token_len_gt_8: usize,
    token_len_gt_16: usize,
    token_len_gt_32: usize,
    token_len_gt_64: usize,
    state_equiv_ms: f64,
    max_length_state_equiv_ms: f64,
    exact_state_equiv_ms: f64,
    max_length_reps: usize,
    exact_reps: usize,
    token_identity_map_ms: f64,
}

struct L1TsidProfileMergeReport {
    tsids_after: usize,
    unique_arc_token_sets: usize,
    unique_range_token_sets: usize,
    profile_build_ms: f64,
    group_ms: f64,
    remap_ms: f64,
}

struct L1TerminalBuildProfile {
    internal_vocab_ms: f64,
    vocab_tree_build_ms: f64,
    state_seed_ms: f64,
    token_set_intern_ms: f64,
    tsid_profile_merge_ms: f64,
    tsid_profile_merge_before: usize,
    tsid_profile_merge_after: usize,
    vocab_tree_traversal_ms: f64,
    direct_terminal_dwa_ms: f64,
}

#[cfg(test)]
mod generic_nfa_tests ;

#[cfg(test)]
mod packed_suffix_product_tests ;
