use crate::automata::lexer::Lexer;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use super::state_equivalence::global_token_position::GlobalTokenPositionStatePartition;
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::ignore_transparent_disallowed_follows;
use super::state_equivalence::{
    build_state_map_from_subset_representatives, identity_state_map, resolve_l2p_pipeline_config,
    run_state_equivalence_pipeline,
    run_state_equivalence_pipeline_with_initial_restricted_observation_certificate,
    StateEquivalenceScope,
};
use super::state_equivalence::pipeline::StateEquivalencePipelineProfile;
use super::state_equivalence::nfa::{
    PrebuiltSparsePowersetRefinement, TokenBoundedAnalysisTrie, build_bounded_analysis_view,
    build_bounded_analysis_view_with_trie,
    RelevantPowersetView, RelevantPowersetWork, RelevantPowersetWorkBudget,
    build_bounded_analysis_view_from_relevant_powerset, build_relevant_powerset_view,
    build_relevant_powerset_view_budgeted, powerset_output_class_ids,
};
use crate::ds::bitset::BitSet;
use super::compat::TokenizerView;
use super::disallowed_follows::normalize_disallowed_follows;
use super::shared::{
    TokenDedup, VocabLexicalEntryOrder,
    expand_vocab_classes,
    hash_byte_class_seq,
    representative_tokens_for_vocab_classes,
    tokenizer_group_count,
};

fn assert_same_many_to_one_partition(
    left: &ManyToOneIdMap,
    right: &ManyToOneIdMap,
    message: &str,
) {
    assert_eq!(
        left.original_to_internal.len(),
        right.original_to_internal.len(),
        "{message}: raw domains differ",
    );
    let mut left_to_right = FxHashMap::<u32, u32>::default();
    let mut right_to_left = FxHashMap::<u32, u32>::default();
    for (&left_class, &right_class) in left
        .original_to_internal
        .iter()
        .zip(&right.original_to_internal)
    {
        assert_eq!(
            left_class == u32::MAX,
            right_class == u32::MAX,
            "{message}: mapped raw domains differ",
        );
        if left_class == u32::MAX {
            continue;
        }
        if let Some(previous) = left_to_right.insert(left_class, right_class) {
            assert_eq!(previous, right_class, "{message}: left class split");
        }
        if let Some(previous) = right_to_left.insert(right_class, left_class) {
            assert_eq!(previous, left_class, "{message}: right class split");
        }
    }
}
use super::state::fast as state_equivalence_analysis;
use super::vocab::common_atom as common_atom_preclass;
use super::vocab::fast as vocab_equivalence_analysis;

// Rebuilding the selected terminal languages is a fixed-cost semantic
// prequotient. Restrict it to vocabularies large enough to amortize that build.
const ACTIVE_LANGUAGE_BYTE_DEDUP_MIN_TOKENS: usize = 50_000;

fn common_atom_preclass_strict_reference_enabled(partition_label: &str) -> bool {
    std::env::var("GLRMASK_L2P_COMMON_ATOM_PRECLASS_STRICT_REFERENCE")
        .ok()
        .is_some_and(|value| value == "1" || value == partition_label)
}

fn common_atom_preclass_enabled() -> bool {
    std::env::var("GLRMASK_ENABLE_L2P_COMMON_ATOM_PRECLASS")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true)
}

fn first_byte_vocab_factor_enabled(partition_label: &str) -> bool {
    super::super::l2p_first_byte_vocab_factor_enabled_for_partition(partition_label)
}

fn first_byte_vocab_factor_strict_reference_enabled(partition_label: &str) -> bool {
    super::super::l2p_first_byte_vocab_factor_strict_reference_enabled_for_partition(
        partition_label,
    )
}

fn staged_vocab_state_refinement_enabled(partition_label: &str) -> bool {
    std::env::var("GLRMASK_STAGED_VOCAB_STATE_REFINEMENT")
        .ok()
        .is_some_and(|value| {
            let value = value.trim();
            value == "1" || value == partition_label
        })
}

fn staged_vocab_state_refinement_strict_reference_enabled(partition_label: &str) -> bool {
    std::env::var("GLRMASK_STAGED_VOCAB_STATE_REFINEMENT_STRICT_REFERENCE")
        .ok()
        .is_some_and(|value| {
            let value = value.trim();
            value == "1" || value == partition_label
        })
}

#[derive(Debug, Clone, Copy, Default)]
struct FirstByteVocabFactorProfile {
    buckets: usize,
    factored_buckets: usize,
    min_bucket_tokens: usize,
    source_states_before: usize,
    source_states_after: usize,
    preliminary_classes: usize,
    bucket_ms: f64,
    final_ms: f64,
    total_ms: f64,
    dfa_build_ms: f64,
}

/// Exact vocabulary partition through deterministic first-byte successors.
///
/// For one nonempty token `b + suffix`, every observable action from source
/// state `q` after consuming the first byte depends only on `delta(q, b)`.
/// Therefore one source representative per distinct `b` successor is enough
/// to compute exact token classes *within* a first-byte bucket. Those bucket
/// classes are only a prepartition: one final exact scan over all original
/// source states and one representative per preliminary class merges classes
/// that are equivalent across different first-byte buckets.
fn first_byte_factored_vocab_classes(
    tokenizer: &TokenizerView,
    tokens: &[&[u8]],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    byte_to_class: &[u8; 256],
) -> Option<(
    vocab_equivalence_analysis::VocabEquivalenceResult,
    FirstByteVocabFactorProfile,
)> {
    if tokens.len() < 2 || initial_states.len() < 2 || tokens.iter().any(|token| token.is_empty()) {
        return None;
    }
    let min_bucket_tokens = std::env::var("GLRMASK_L2P_FIRST_BYTE_VOCAB_FACTOR_MIN_BUCKET_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value >= 2)
        .unwrap_or(2);

    let total_started_at = Instant::now();
    // A DFA byte class has one transition row across every analysis state.
    // Tokens whose leading bytes share that class therefore admit the same
    // first-successor quotient and can be classified in one exact bucket.
    let mut token_buckets = (0..256).map(|_| Vec::<usize>::new()).collect::<Vec<_>>();
    for (token, bytes) in tokens.iter().enumerate() {
        token_buckets[byte_to_class[bytes[0] as usize] as usize].push(token);
    }

    let analysis_dfa_cache = vocab_equivalence_analysis::SharedVocabAnalysisDfaCache::default();
    let bucket_started_at = Instant::now();
    let bucket_results = token_buckets
        .par_iter()
        .enumerate()
        .filter(|(_, bucket)| !bucket.is_empty())
        .map(|(_first_class, bucket)| {
            if bucket.len() < min_bucket_tokens {
                return (
                    bucket.iter().copied().map(|token| vec![token]).collect(),
                    0usize,
                    0.0f64,
                    false,
                );
            }

            let first_byte = tokens[bucket[0]][0] as usize;
            let mut source_for_target = BTreeMap::<u32, usize>::new();
            for &source in initial_states {
                let target = tokenizer.dfa().trans(source, first_byte);
                source_for_target.entry(target).or_insert(source);
            }
            let reduced_sources = source_for_target.into_values().collect::<Vec<_>>();
            let bucket_tokens = bucket.iter().map(|&token| tokens[token]).collect::<Vec<_>>();
            let (bucket_classes, build_ms) =
                vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                    tokenizer,
                    &bucket_tokens,
                    &reduced_sources,
                    disallowed_follows,
                    Some(byte_to_class),
                    None,
                    None,
                    Some(&analysis_dfa_cache),
                );
            let classes = bucket_classes
                .into_iter()
                .map(|local_class| {
                    let mut global_class = local_class
                        .into_iter()
                        .map(|local_token| bucket[local_token])
                        .collect::<Vec<_>>();
                    global_class.sort_unstable();
                    global_class
                })
                .collect::<Vec<_>>();
            (classes, reduced_sources.len(), build_ms, true)
        })
        .collect::<Vec<_>>();
    let buckets = bucket_results.len();
    let factored_buckets = bucket_results
        .iter()
        .filter(|(_, _, _, factored)| *factored)
        .count();
    let source_states_after = bucket_results
        .iter()
        .map(|(_, states, _, _)| *states)
        .sum::<usize>();
    let mut dfa_build_ms = bucket_results
        .iter()
        .map(|(_, _, build_ms, _)| *build_ms)
        .fold(0.0f64, f64::max);
    let mut preliminary_classes = bucket_results
        .into_iter()
        .flat_map(|(classes, _, _, _)| classes)
        .collect::<Vec<_>>();
    let bucket_ms = bucket_started_at.elapsed().as_secs_f64() * 1000.0;
    preliminary_classes.sort_unstable_by_key(|class| class[0]);

    // No reduction means the final representative scan would only duplicate
    // the ordinary exact pass.
    if preliminary_classes.len() >= tokens.len() {
        return None;
    }

    let representative_tokens = preliminary_classes
        .iter()
        .map(|class| tokens[class[0]])
        .collect::<Vec<_>>();
    let final_started_at = Instant::now();
    let (representative_classes, final_build_ms) =
        vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
            tokenizer,
            &representative_tokens,
            initial_states,
            disallowed_follows,
            Some(byte_to_class),
            None,
            None,
            Some(&analysis_dfa_cache),
        );
    dfa_build_ms = dfa_build_ms.max(final_build_ms);
    let final_ms = final_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut classes = vocab_equivalence_analysis::VocabEquivalenceResult::new();
    for representative_class in representative_classes {
        let total_len = representative_class
            .iter()
            .map(|&preliminary| preliminary_classes[preliminary].len())
            .sum();
        let mut class = Vec::with_capacity(total_len);
        for preliminary in representative_class {
            class.extend_from_slice(&preliminary_classes[preliminary]);
        }
        class.sort_unstable();
        classes.insert(class);
    }

    let total_ms = total_started_at.elapsed().as_secs_f64() * 1000.0;
    Some((
        classes,
        FirstByteVocabFactorProfile {
            buckets,
            factored_buckets,
            min_bucket_tokens,
            source_states_before: initial_states.len().saturating_mul(factored_buckets),
            source_states_after,
            preliminary_classes: preliminary_classes.len(),
            bucket_ms,
            final_ms,
            total_ms,
            dfa_build_ms,
        },
    ))
}

fn deduplicate_tokens_by_byte_class<'a, S: AsRef<[u8]>>(
    tokens: &'a [S],
    byte_to_class: &[u8; 256],
    lexical_entry_order: Option<&[u32]>,
) -> TokenDedup<'a> {
    #[inline]
    fn same_projected_bytes(left: &[u8], right: &[u8], byte_to_class: &[u8; 256]) -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(&left, &right)| byte_to_class[left as usize] == byte_to_class[right as usize])
    }

    // `hash_byte_class_seq` is already the expensive sequence hash we want.
    // Feeding that u128 through std::HashMap's SipHash a second time was a large
    // p90 cost on 80k-token partitions. Use FxHash for the outer lookup, but do
    // not treat the sequence hash as an equality oracle: verify projected bytes
    // exactly, with a collision side table allocated only if a real hash
    // collision occurs.
    let mut hash_to_repr = FxHashMap::<u128, usize>::with_capacity_and_hasher(
        tokens.len() / 2,
        Default::default(),
    );
    let mut collisions = FxHashMap::<u128, Vec<usize>>::default();
    let mut representative_token_bytes = Vec::new();
    let mut representative_original_indices = Vec::new();
    let mut original_to_repr = Vec::with_capacity(tokens.len());

    for (original_index, token) in tokens.iter().enumerate() {
        let bytes = token.as_ref();
        let hash = hash_byte_class_seq(bytes, byte_to_class);
        let repr_idx = if let Some(&primary) = hash_to_repr.get(&hash) {
            if same_projected_bytes(bytes, representative_token_bytes[primary], byte_to_class) {
                primary
            } else if let Some(existing) = collisions.get(&hash).and_then(|candidates| {
                candidates.iter().copied().find(|&candidate| {
                    same_projected_bytes(
                        bytes,
                        representative_token_bytes[candidate],
                        byte_to_class,
                    )
                })
            }) {
                existing
            } else {
                let idx = representative_token_bytes.len();
                representative_token_bytes.push(bytes);
                representative_original_indices.push(original_index);
                collisions.entry(hash).or_default().push(idx);
                idx
            }
        } else {
            let idx = representative_token_bytes.len();
            representative_token_bytes.push(bytes);
            representative_original_indices.push(original_index);
            hash_to_repr.insert(hash, idx);
            idx
        };
        original_to_repr.push(repr_idx);
    }

    if let Some(lexical_entry_order) = lexical_entry_order {
        debug_assert_eq!(lexical_entry_order.len(), tokens.len());
        let mut repr_at_original = vec![usize::MAX; tokens.len()];
        for (repr, &original) in representative_original_indices.iter().enumerate() {
            repr_at_original[original] = repr;
        }
        let lexical_reprs = lexical_entry_order
            .iter()
            .filter_map(|&original| {
                let repr = repr_at_original[original as usize];
                (repr != usize::MAX).then_some(repr)
            })
            .collect::<Vec<_>>();
        debug_assert_eq!(lexical_reprs.len(), representative_token_bytes.len());
        let mut old_to_new = vec![usize::MAX; representative_token_bytes.len()];
        let old_bytes = representative_token_bytes;
        let old_originals = representative_original_indices;
        representative_token_bytes = Vec::with_capacity(old_bytes.len());
        representative_original_indices = Vec::with_capacity(old_originals.len());
        for (new_repr, old_repr) in lexical_reprs.into_iter().enumerate() {
            old_to_new[old_repr] = new_repr;
            representative_token_bytes.push(old_bytes[old_repr]);
            representative_original_indices.push(old_originals[old_repr]);
        }
        for repr in &mut original_to_repr {
            *repr = old_to_new[*repr];
        }
    }

    TokenDedup {
        representative_token_bytes,
        representative_original_indices,
        original_to_repr,
    }
}

fn representative_byte_mask(
    relevant_bytes: &[bool; 256],
    byte_to_class: Option<&[u8; 256]>,
) -> [bool; 256] {
    let mut representative_bytes = [false; 256];
    for byte in super::state_equivalence::max_length::active_byte_representatives(
        Some(relevant_bytes),
        byte_to_class,
    ) {
        representative_bytes[byte as usize] = true;
    }
    representative_bytes
}

fn build_state_map(
    state_classes: &BTreeSet<BTreeSet<usize>>,
    num_dfa_states: usize,
) -> ManyToOneIdMap {
    let mut original_to_internal = vec![u32::MAX; num_dfa_states];
    let mut internal_to_originals = Vec::new();
    let mut representative_original_ids = Vec::new();

    for class in state_classes {
        let internal_id = internal_to_originals.len() as u32;
        let originals: Vec<u32> = class.iter().map(|&state| state as u32).collect();
        for &state in &originals {
            original_to_internal[state as usize] = internal_id;
        }
        representative_original_ids
            .push(*originals.first().expect("state class must be non-empty"));
        internal_to_originals.push(originals);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn build_state_map_composed(
    state_classes: &BTreeSet<BTreeSet<usize>>,
    num_dfa_states: usize,
    initial_state_map: &ManyToOneIdMap,
) -> ManyToOneIdMap {
    let mut rep_to_new_internal = vec![u32::MAX; num_dfa_states];
    let mut new_internal_to_originals: Vec<Vec<u32>> = Vec::new();
    for class in state_classes {
        let internal_id = new_internal_to_originals.len() as u32;
        let originals: Vec<u32> = class.iter().map(|&state| state as u32).collect();
        for &state in &originals {
            rep_to_new_internal[state as usize] = internal_id;
        }
        new_internal_to_originals.push(originals);
    }

    let mut composed_original_to_internal = vec![u32::MAX; num_dfa_states];
    let mut composed_internal_to_originals = vec![Vec::new(); new_internal_to_originals.len()];
    let mut composed_reps = vec![u32::MAX; new_internal_to_originals.len()];

    for (orig_state, &init_internal) in initial_state_map.original_to_internal.iter().enumerate() {
        if init_internal == u32::MAX
            || (init_internal as usize) >= initial_state_map.representative_original_ids.len()
        {
            continue;
        }
        let init_rep = initial_state_map.representative_original_ids[init_internal as usize] as usize;
        let new_internal = rep_to_new_internal[init_rep];
        if new_internal != u32::MAX {
            composed_original_to_internal[orig_state] = new_internal;
            let bucket = &mut composed_internal_to_originals[new_internal as usize];
            if bucket.is_empty() {
                composed_reps[new_internal as usize] = init_rep as u32;
            }
            bucket.push(orig_state as u32);
        }
    }

    ManyToOneIdMap {
        original_to_internal: composed_original_to_internal,
        internal_to_originals: composed_internal_to_originals,
        representative_original_ids: composed_reps,
    }
}

fn build_vocab_map(
    vocab_classes: &BTreeSet<Vec<usize>>,
    token_ids: &[u32],
    max_token_id: u32,
) -> ManyToOneIdMap {
    let mut ordered_vocab_classes: Vec<(u32, Vec<u32>)> = vocab_classes
        .iter()
        .map(|class| {
            let mut min_tid = u32::MAX;
            let mut originals: Vec<u32> = Vec::with_capacity(class.len());
            for &idx in class {
                let tid = token_ids[idx];
                originals.push(tid);
                if tid < min_tid {
                    min_tid = tid;
                }
            }
            (min_tid, originals)
        })
        .collect();
    ordered_vocab_classes.sort_unstable_by_key(|(rep_tid, _)| *rep_tid);

    let mut original_to_internal = vec![u32::MAX; (max_token_id + 1) as usize];
    let mut internal_to_originals = Vec::with_capacity(ordered_vocab_classes.len());
    let mut representative_original_ids = Vec::with_capacity(ordered_vocab_classes.len());

    for (internal_id, (rep_tid, originals)) in ordered_vocab_classes.into_iter().enumerate() {
        for &token_id in &originals {
            original_to_internal[token_id as usize] = internal_id as u32;
        }
        representative_original_ids.push(rep_tid);
        internal_to_originals.push(originals);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn build_vocab_group_map(
    vocab_classes: &BTreeSet<Vec<usize>>,
    token_ids: &[u32],
) -> ManyToOneIdMap {
    let mut ordered_vocab_classes: Vec<(u32, Vec<u32>)> = vocab_classes
        .iter()
        .map(|class| {
            let mut min_tid = u32::MAX;
            let mut originals = Vec::with_capacity(class.len());
            for &idx in class {
                let tid = token_ids[idx];
                originals.push(tid);
                min_tid = min_tid.min(tid);
            }
            originals.sort_unstable();
            (min_tid, originals)
        })
        .collect();
    ordered_vocab_classes.sort_unstable_by_key(|(rep_tid, _)| *rep_tid);
    ManyToOneIdMap {
        original_to_internal: Vec::new(),
        representative_original_ids: ordered_vocab_classes
            .iter()
            .map(|(representative, _)| *representative)
            .collect(),
        internal_to_originals: ordered_vocab_classes
            .into_iter()
            .map(|(_, originals)| originals)
            .collect(),
    }
}

/// An identity map is an exact conservative token quotient. It is useful for
/// small structural partitions where vocab equivalence has no materialization
/// benefit, while its analysis cost dominates the enclosing partition wall.
fn build_identity_vocab_map(token_ids: &[u32], max_token_id: u32) -> ManyToOneIdMap {
    let mut token_ids = token_ids.to_vec();
    token_ids.sort_unstable();
    token_ids.dedup();

    let mut original_to_internal = vec![u32::MAX; max_token_id as usize + 1];
    let mut internal_to_originals = Vec::with_capacity(token_ids.len());
    let mut representatives = Vec::with_capacity(token_ids.len());
    for (internal, token_id) in token_ids.into_iter().enumerate() {
        original_to_internal[token_id as usize] = internal as u32;
        internal_to_originals.push(vec![token_id]);
        representatives.push(token_id);
    }
    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids: representatives,
    }
}

/// Conservative vocabulary quotient for epsilon tokenizers: only byte-for-byte
/// aliases are merged. They necessarily have identical lexer behaviour, and
/// merging them is required because the vocabulary prefix tree has one leaf
/// per byte string. The final mask mapping expands the class back to every
/// original token id.
fn build_exact_byte_vocab_map(
    token_ids: &[u32],
    token_bytes: &[&[u8]],
    max_token_id: u32,
) -> ManyToOneIdMap {
    debug_assert_eq!(token_ids.len(), token_bytes.len());
    let mut classes = BTreeMap::<Vec<u8>, Vec<u32>>::new();
    for (&token_id, &bytes) in token_ids.iter().zip(token_bytes) {
        classes.entry(bytes.to_vec()).or_default().push(token_id);
    }

    let mut ordered = classes.into_values().collect::<Vec<_>>();
    for class in &mut ordered {
        class.sort_unstable();
    }
    ordered.sort_unstable_by_key(|class| class[0]);

    let mut original_to_internal = vec![u32::MAX; max_token_id as usize + 1];
    let mut representatives = Vec::with_capacity(ordered.len());
    for (internal, class) in ordered.iter().enumerate() {
        representatives.push(class[0]);
        for &original in class {
            original_to_internal[original as usize] = internal as u32;
        }
    }
    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals: ordered,
        representative_original_ids: representatives,
    }
}

struct PreparedEquivalenceInputs<'a> {
    max_token_id: u32,
    token_ids: Vec<u32>,
    token_bytes: Vec<&'a [u8]>,
    initial_states: Vec<usize>,
    lexical_entry_order: Option<Arc<VocabLexicalEntryOrder>>,
}

impl PreparedEquivalenceInputs<'_> {
    #[inline]
    fn lexical_entry_indices(&self) -> Option<&[u32]> {
        self.lexical_entry_order.as_deref().map(|order| order.entry_indices.as_ref())
    }
}

fn prepare_equivalence_inputs<'a>(
    tokenizer: &Tokenizer,
    vocab: &'a Vocab,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> PreparedEquivalenceInputs<'a> {
    let max_token_id = vocab.max_token_id();
    let mut token_bytes: Vec<&[u8]> = Vec::with_capacity(vocab.len());
    let mut token_ids: Vec<u32> = Vec::with_capacity(vocab.len());
    for (&tid, bytes) in vocab.entries_map().iter() {
        token_ids.push(tid);
        token_bytes.push(bytes.as_slice());
    }

    let initial_states = match initial_state_map {
        Some(map) => map.representative_original_ids.iter().map(|&s| s as usize).collect(),
        None => (0..tokenizer.num_states() as usize).collect(),
    };

    PreparedEquivalenceInputs {
        max_token_id,
        token_ids,
        token_bytes,
        initial_states,
        lexical_entry_order: vocab.vocab_derived_cache_get::<VocabLexicalEntryOrder>(),
    }
}


struct CombinedEquivalenceResult {
    vocab_classes: BTreeSet<Vec<usize>>,
    state_classes: BTreeSet<BTreeSet<usize>>,
}

#[derive(Debug, Clone)]
pub struct CombinedEquivalenceProfile {
    pub initial_states_considered: usize,
    pub max_length_skipped: bool,
    pub max_token_len: usize,
    pub token_len_gt_4: usize,
    pub token_len_gt_8: usize,
    pub token_len_gt_16: usize,
    pub token_len_gt_32: usize,
    pub token_len_gt_64: usize,
    pub raw_analysis_base_init_ms: f64,
    pub analysis_view_build_ms: f64,
    /// Sub-timing within `analysis_view_build_ms`; do not add separately.
    pub active_mask_filter_ms: f64,
    pub effective_follows_normalize_ms: f64,
    pub prepare_inputs_ms: f64,
    pub byte_class_setup_ms: f64,
    pub vocab_analysis_dfa_build_ms: f64,
    pub token_dedup_ms: f64,
    pub restricted_observation_state_equiv_ms: f64,
    pub max_length_state_equiv_ms: f64,
    pub vocab_equiv_ms: f64,
    pub exact_state_equiv_ms: f64,
    pub id_map_finalize_ms: f64,
    pub restricted_observation_reps: usize,
    pub max_length_reps: usize,
    pub exact_reps: usize,
    pub exact_rep_confirmation_used: bool,
}

struct TokenLengthStats {
    gt_4: usize,
    gt_8: usize,
    gt_16: usize,
    gt_32: usize,
    gt_64: usize,
}

fn skip_max_length_for_partition(partition_label: &str) -> bool {
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

#[inline]
fn direct_refinement_work_is_no_larger(
    direct_token_bytes: usize,
    max_token_len: usize,
    active_byte_count: usize,
) -> bool {
    direct_token_bytes <= max_token_len.saturating_mul(active_byte_count)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum L2pNfaAnalysisViewPolicy {
    Adaptive,
    Bounded,
    Powerset,
}

impl L2pNfaAnalysisViewPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::Bounded => "bounded",
            Self::Powerset => "powerset",
        }
    }
}

#[inline]
fn l2p_nfa_analysis_view_policy(partition_label: &str) -> L2pNfaAnalysisViewPolicy {
    if std::env::var_os("GLRMASK_L2P_NFA_RELEVANT_POWERSET_VIEW").is_none()
        && partition_label.starts_with("boundary_shard")
    {
        // Static-link boundary shards always run the full 128k-vocab analysis
        // over a ~26k-state merged tokenizer (pair estimate ~3.3B): the
        // adaptive powerset probe unconditionally aborts at the 40K cap after
        // ~0.6 s of wasted construction and the bounded view then computes the
        // identical exact quotient. Both views are exact; defaulting this path
        // to bounded skips the doomed probe. An explicit env override is
        // still honored for diagnostics.
        return L2pNfaAnalysisViewPolicy::Bounded;
    }
    let Ok(value) = std::env::var("GLRMASK_L2P_NFA_RELEVANT_POWERSET_VIEW") else {
        return L2pNfaAnalysisViewPolicy::Adaptive;
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" | "adaptive" => L2pNfaAnalysisViewPolicy::Adaptive,
        "" | "1" | "true" | "yes" | "on" | "powerset" => {
            L2pNfaAnalysisViewPolicy::Powerset
        }
        "0" | "false" | "no" | "off" | "bounded" => L2pNfaAnalysisViewPolicy::Bounded,
        other => panic!(
            "invalid GLRMASK_L2P_NFA_RELEVANT_POWERSET_VIEW={other:?}; expected auto/adaptive, powerset/1/true/on, or bounded/0/false/off"
        ),
    }
}

#[inline]
fn l2p_nfa_relevant_powerset_max_states() -> usize {
    // Keep a narrow allowance above 32K for projected powersets that only
    // slightly overflow the historical cap. This admits useful ~38K views
    // (notably JSB o21456 p1) without selecting the 41K+ views whose larger
    // downstream state domain is slower than the bounded fallback.
    const DEFAULT_MAX_STATES: usize = 40_960;
    std::env::var("GLRMASK_L2P_NFA_RELEVANT_POWERSET_MAX_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_MAX_STATES)
}

#[inline]
fn l2p_nfa_relevant_powerset_min_bounded_pairs(partition_label: &str) -> usize {
    if let Some(value) = std::env::var("GLRMASK_L2P_NFA_RELEVANT_POWERSET_MIN_BOUNDED_PAIRS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
    {
        return value;
    }
    if partition_label == "p0" {
        return std::env::var("GLRMASK_P0_NFA_RELEVANT_POWERSET_MIN_BOUNDED_PAIRS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(0);
    }
    500_000
}

#[inline]
fn should_probe_l2p_nfa_powerset(
    policy: L2pNfaAnalysisViewPolicy,
    bounded_pair_estimate: usize,
    min_bounded_pairs: usize,
) -> bool {
    match policy {
        L2pNfaAnalysisViewPolicy::Adaptive => bounded_pair_estimate >= min_bounded_pairs,
        L2pNfaAnalysisViewPolicy::Bounded => false,
        L2pNfaAnalysisViewPolicy::Powerset => true,
    }
}

fn build_l2p_nfa_powerset_candidate(
    policy: L2pNfaAnalysisViewPolicy,
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    active_groups: Option<&[bool]>,
    state_map: Option<&ManyToOneIdMap>,
    max_states: usize,
) -> (Option<RelevantPowersetView>, Option<RelevantPowersetWork>) {
    match policy {
        L2pNfaAnalysisViewPolicy::Bounded => (None, None),
        // Explicit forced-powerset mode remains diagnostic and unbounded.
        L2pNfaAnalysisViewPolicy::Powerset => (
            Some(build_relevant_powerset_view(
                tokenizer,
                relevant_bytes,
                active_groups,
                state_map,
            )),
            None,
        ),
        L2pNfaAnalysisViewPolicy::Adaptive => {
            let active_bytes = relevant_bytes.iter().filter(|&&active| active).count().max(1);
            let budget = RelevantPowersetWorkBudget {
                max_configurations: max_states,
                max_edges: max_states.saturating_mul(active_bytes),
            };
            match build_relevant_powerset_view_budgeted(
                tokenizer,
                relevant_bytes,
                active_groups,
                state_map,
                budget,
            ) {
                Ok(view) => (Some(view), None),
                Err(work) => (None, Some(work)),
            }
        }
    }
}

#[inline]
fn should_use_l2p_nfa_powerset(
    policy: L2pNfaAnalysisViewPolicy,
    candidate_present: bool,
    powerset_states: usize,
    max_states: usize,
) -> bool {
    match policy {
        L2pNfaAnalysisViewPolicy::Adaptive => {
            candidate_present && powerset_states <= max_states
        }
        L2pNfaAnalysisViewPolicy::Bounded => false,
        L2pNfaAnalysisViewPolicy::Powerset => true,
    }
}

#[inline]
fn should_use_l2p_nfa_prepass_powerset(
    policy: L2pNfaAnalysisViewPolicy,
    candidate_present: bool,
) -> bool {
    match policy {
        // Adaptive probing already gates construction on the estimated cost of
        // the bounded alternative. Once the exact sparse candidate has been
        // built successfully, do not throw it away because of a second,
        // unrelated state-count cliff. The prepass is consumed only by bounded
        // max-length refinement; stable restricted observation uses the direct
        // incremental NFA engine.
        L2pNfaAnalysisViewPolicy::Adaptive => candidate_present,
        L2pNfaAnalysisViewPolicy::Bounded => false,
        L2pNfaAnalysisViewPolicy::Powerset => true,
    }
}

fn should_skip_max_length_for_partition(
    partition_label: &str,
    initial_state_count: usize,
    projected_by_global: bool,
    direct_token_bytes: usize,
    max_token_len: usize,
    active_byte_count: usize,
) -> bool {
    if skip_max_length_for_partition(partition_label)
        || (projected_by_global && initial_state_count <= 8192)
    {
        return true;
    }

    // Both routes are exact. The k-bounded prepass performs up to one
    // state-transition refinement for every (length, active-byte) pair;
    // direct finite-vocabulary refinement must at least read every token byte.
    // Prefer the direct route when its input bound is no larger, avoiding a
    // prepass that cannot amortize its own full-DFA scans.
    direct_refinement_work_is_no_larger(
        direct_token_bytes,
        max_token_len,
        active_byte_count,
    )
}

const EXACT_REP_CONFIRMATION_MIN_STATES: usize = 2_000;
const EXACT_REP_CONFIRMATION_MIN_TOKENS: usize = 200;
const VOCAB_FIRST_MIN_TOKENS: usize = 512;
const VOCAB_FIRST_MIN_STATES: usize = 256;
const VOCAB_FIRST_WORK_MIN_TOKENS: usize = 10_000;
const VOCAB_FIRST_WORK_MIN_PAIRS: usize = 500_000;
const RAW_QUOTIENT_TINY_VOCAB_MAX_TOKENS: usize = 16;
const RAW_QUOTIENT_TINY_VOCAB_MAX_BYTES: usize = 64;
const RAW_QUOTIENT_STRUCTURAL_BOUNDARY_MAX_TOKENS: usize = 48;
const RAW_QUOTIENT_STRUCTURAL_BOUNDARY_MAX_BYTES: usize = 256;

fn prefer_vocab_first_equivalence(token_count: usize, state_count: usize) -> bool {
    (token_count >= VOCAB_FIRST_MIN_TOKENS && state_count >= VOCAB_FIRST_MIN_STATES)
        || (token_count >= VOCAB_FIRST_WORK_MIN_TOKENS
            && token_count.saturating_mul(state_count) >= VOCAB_FIRST_WORK_MIN_PAIRS)
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

/// Compose the raw restricted-observation quotient with its final exact
/// representative map without materializing ordered sets of raw states.
///
/// Scanning raw states in ascending order assigns dense IDs in the same order
/// as the generic `BTreeSet` path: each class is ordered by its first raw
/// member, and that member is also the retained raw representative.
fn compose_raw_quotient_state_map(
    pre_state_map: &ManyToOneIdMap,
    final_representative_for_preclass: &[usize],
) -> ManyToOneIdMap {
    compose_raw_quotient_state_map_impl(
        pre_state_map,
        final_representative_for_preclass,
        None,
    )
}

fn compose_raw_quotient_state_map_preserving_directional_representatives(
    pre_state_map: &ManyToOneIdMap,
    final_representative_for_preclass: &[usize],
) -> ManyToOneIdMap {
    compose_raw_quotient_state_map_impl(
        pre_state_map,
        final_representative_for_preclass,
        Some(&pre_state_map.representative_original_ids),
    )
}

/// Deduplicate exact representatives that are already expressed in analysis-
/// view coordinates.
///
/// A composed raw quotient may retain the first raw member of each final class,
/// but a bounded analysis view only maps the raw representatives that seeded the
/// view. Re-projecting the composed raw representative can therefore produce the
/// unmapped-state sentinel. The exact representatives are the authoritative
/// coordinates for a subsequent vocabulary pass.
fn exact_analysis_view_representatives(representatives: &[usize]) -> Vec<usize> {
    let mut representatives = representatives.to_vec();
    representatives.sort_unstable();
    representatives.dedup();
    representatives
}

fn compose_raw_quotient_state_map_impl(
    pre_state_map: &ManyToOneIdMap,
    final_representative_for_preclass: &[usize],
    representative_original_for_final_key: Option<&[u32]>,
) -> ManyToOneIdMap {
    assert_eq!(
        pre_state_map.internal_to_originals.len(),
        final_representative_for_preclass.len(),
        "raw quotient and exact representative map disagree",
    );
    let preclass_count = pre_state_map.internal_to_originals.len();
    let mut final_key_to_internal = vec![u32::MAX; preclass_count];
    let mut original_to_internal = vec![u32::MAX; pre_state_map.original_to_internal.len()];
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representative_original_ids = Vec::<u32>::new();

    for (raw_state, &preclass) in pre_state_map.original_to_internal.iter().enumerate() {
        if preclass == u32::MAX {
            continue;
        }
        let preclass = preclass as usize;
        assert!(preclass < preclass_count, "invalid raw quotient class");
        let final_key = final_representative_for_preclass[preclass];
        assert!(final_key < preclass_count, "invalid exact representative");
        let internal = if final_key_to_internal[final_key] == u32::MAX {
            let next = internal_to_originals.len() as u32;
            final_key_to_internal[final_key] = next;
            internal_to_originals.push(Vec::new());
            representative_original_ids.push(
                representative_original_for_final_key
                    .map_or(raw_state as u32, |representatives| representatives[final_key]),
            );
            next
        } else {
            final_key_to_internal[final_key]
        };
        original_to_internal[raw_state] = internal;
        internal_to_originals[internal as usize].push(raw_state as u32);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn build_internal_id_map_from_combined_result(
    tokenizer: &Tokenizer,
    initial_state_map: Option<&ManyToOneIdMap>,
    prepared: &PreparedEquivalenceInputs<'_>,
    result: &CombinedEquivalenceResult,
) -> InternalIdMap {
    let num_dfa_states = tokenizer.num_states() as usize;
    let state_map = match initial_state_map {
        Some(init_map) => build_state_map_composed(&result.state_classes, num_dfa_states, init_map),
        None => build_state_map(&result.state_classes, num_dfa_states),
    };
    let vocab_map = build_vocab_map(&result.vocab_classes, &prepared.token_ids, prepared.max_token_id);

    InternalIdMap {
        tokenizer_states: state_map,
        vocab_tokens: vocab_map,
        deferred_vocab_singleton_original_ids: None,
    }
}

fn try_analyze_equivalences_with_token_position_partition(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    effective_disallowed: &BTreeMap<u32, BitSet>,
    active_groups: Option<&[bool]>,
    _shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    _shared_analysis_dfa_cache: Option<&super::vocab::fast::SharedVocabAnalysisDfaCache>,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    token_position_partition: &GlobalTokenPositionStatePartition,
    effective_follows_prepare_ms: f64,
    pre_normalized_disallowed_follows: Option<&[BitSet]>,
) -> Option<(InternalIdMap, CombinedEquivalenceProfile)> {
    let active_groups = active_groups?;
    let seed = token_position_partition.as_many_to_one();
    let num_states = tokenizer.num_states() as usize;
    if seed.original_to_internal.len() != num_states
        || seed.representative_original_ids.is_empty()
        || vocab.entries_map().values().any(Vec::is_empty)
    {
        return None;
    }

    let prepare_inputs_started_at = Instant::now();
    let prepared = prepare_equivalence_inputs(tokenizer, vocab, None);
    if prepared.token_bytes.is_empty() {
        return None;
    }
    let token_len_stats = token_length_stats(&prepared.token_bytes);
    let max_token_len = prepared
        .token_bytes
        .iter()
        .map(|token| token.len())
        .max()
        .unwrap_or(0);
    let prepare_inputs_ms = prepare_inputs_started_at.elapsed().as_secs_f64() * 1000.0;

    let seed_states = seed
        .representative_original_ids
        .iter()
        .map(|&state| state as usize)
        .collect::<Vec<_>>();
    if seed_states.iter().any(|&state| state >= num_states) {
        return None;
    }

    let compatible_flat_trans = flat_trans.filter(|transitions| {
        transitions.len() == tokenizer.num_states() as usize * 256
    });
    let analysis_view_started_at = Instant::now();
    // C is a token-start partition, not an absolute lexer-state equivalence.
    // For epsilon lexers, analyze its representatives as epsilon-closed scanner
    // configurations in a bounded powerset view. These view IDs stay local to
    // token-boundary analysis and are projected back through C before returning.
    let bounded_analysis = tokenizer.has_epsilon_transitions().then(|| {
        build_bounded_analysis_view(
            tokenizer,
            &seed_states,
            &prepared.token_bytes,
            Some(active_groups),
        )
    });
    let raw_analysis_view = bounded_analysis.is_none().then(|| match compatible_flat_trans {
        Some(transitions) => TokenizerView::new_filtered_from_flat_trans(
            transitions,
            tokenizer,
            active_groups,
        ),
        None => TokenizerView::new_filtered(tokenizer, active_groups),
    });
    let tokenizer_view = bounded_analysis
        .as_ref()
        .map(|bounded| &bounded.tokenizer_view)
        .or(raw_analysis_view.as_ref())
        .expect("token-boundary analysis view must exist");
    let analysis_view_build_ms = analysis_view_started_at.elapsed().as_secs_f64() * 1000.0;

    // The C-seeded state/vocab equivalence walks only ever follow the
    // partition's token bytes, so byte classes only need to be exact for the
    // bytes that actually appear in the vocabulary. Restricting the byte-class
    // scan to those relevant bytes makes base construction proportional to the
    // partition's tiny alphabet instead of the full 18k-state x 256-byte table.
    // The base is partition-local (its byte classes depend on this partition's
    // relevant bytes), so it uses local caches rather than the cross-partition
    // shared ones to avoid contaminating other partitions.
    let mut relevant_bytes = [false; 256];
    for token in &prepared.token_bytes {
        for &byte in token.iter() {
            relevant_bytes[byte as usize] = true;
        }
    }
    let local_vocab_dfa_cache = vocab_equivalence_analysis::SharedVocabDfaCache::new();
    let local_analysis_dfa_cache =
        vocab_equivalence_analysis::SharedVocabAnalysisDfaCache::default();
    let compatible_shared_base = Some(local_vocab_dfa_cache.get_or_init(|| {
        vocab_equivalence_analysis::SharedVocabDfaBase::build_from_dfa_relevant(
            tokenizer_view.dfa(),
            &relevant_bytes,
        )
    }));

    let follows_normalize_started_at = Instant::now();
    let owned_normalized_disallowed_follows;
    let normalized_disallowed_follows = if let Some(rows) = pre_normalized_disallowed_follows {
        rows
    } else {
        owned_normalized_disallowed_follows =
            normalize_disallowed_follows(tokenizer_group_count(&tokenizer_view), effective_disallowed);
        &owned_normalized_disallowed_follows
    };
    let effective_follows_normalize_ms = effective_follows_prepare_ms
        + follows_normalize_started_at.elapsed().as_secs_f64() * 1000.0;

    let byte_class_started_at = Instant::now();
    let byte_to_class = compatible_shared_base
        .map(vocab_equivalence_analysis::SharedVocabDfaBase::byte_to_class)
        .unwrap_or_else(|| super::compat::compute_byte_classes(tokenizer_view.dfa()));
    let byte_class_setup_ms = byte_class_started_at.elapsed().as_secs_f64() * 1000.0;
    let token_dedup_started_at = Instant::now();
    let dedup = deduplicate_tokens_by_byte_class(&prepared.token_bytes, &byte_to_class, prepared.lexical_entry_indices());
    let token_dedup_ms = token_dedup_started_at.elapsed().as_secs_f64() * 1000.0;

    let seed_analysis_states = if let Some(bounded) = bounded_analysis.as_ref() {
        seed_states
            .iter()
            .map(|&raw| bounded.view_state_for_raw_start(raw))
            .collect::<Vec<_>>()
    } else {
        seed_states.clone()
    };
    let mut query_states = seed_analysis_states.clone();
    query_states.sort_unstable();
    query_states.dedup();
    let exact_rep_confirmation_used = query_states.len() >= EXACT_REP_CONFIRMATION_MIN_STATES
        && dedup.representative_token_bytes.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
    let exact_started_at = Instant::now();
    let shared_first_byte = bounded_analysis.as_ref().and_then(|_| {
        let first = dedup
            .representative_token_bytes
            .first()
            .and_then(|token| token.first())
            .copied()?;
        dedup
            .representative_token_bytes
            .iter()
            .all(|token| token.len() > 1 && token.first().copied() == Some(first))
            .then_some(first)
    });
    let state_representatives = if let Some(common_first) = shared_first_byte {
        let mut prefix_targets = BTreeSet::<usize>::new();
        let mut target_by_source = Vec::<u32>::with_capacity(query_states.len());
        for &source in &query_states {
            let target = tokenizer_view.dfa().trans(source, common_first as usize);
            target_by_source.push(target);
            if target != u32::MAX {
                prefix_targets.insert(target as usize);
            }
        }
        let prefix_targets = prefix_targets.into_iter().collect::<Vec<_>>();
        let suffix_tokens = dedup
            .representative_token_bytes
            .iter()
            .map(|token| &token[1..])
            .collect::<Vec<_>>();
        let prefix_target_representatives = if exact_rep_confirmation_used {
            state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base_with_initial_finalizers(
                    &tokenizer_view,
                    &suffix_tokens,
                    &prefix_targets,
                    &normalized_disallowed_follows,
                    None,
                    None,
                    Some(true),
                    compatible_shared_base,
                )
        } else {
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base_with_initial_finalizers(
                    &tokenizer_view,
                    &suffix_tokens,
                    &prefix_targets,
                    &normalized_disallowed_follows,
                    compatible_shared_base,
                )
        };
        let behavior_for_target = prefix_targets
            .iter()
            .copied()
            .zip(prefix_target_representatives)
            .collect::<BTreeMap<_, _>>();
        let mut representative_for_behavior = BTreeMap::<Option<usize>, usize>::new();
        let behavior_by_source = target_by_source
            .iter()
            .map(|&target| {
                (target != u32::MAX).then(|| behavior_for_target[&(target as usize)])
            })
            .collect::<Vec<_>>();
        for (&source, &behavior) in query_states.iter().zip(&behavior_by_source) {
            representative_for_behavior.entry(behavior).or_insert(source);
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][token_boundary_first_byte_factorization] partition={} first_byte={} source_states={} prefix_targets={} behavior_classes={}",
                partition_label,
                common_first,
                query_states.len(),
                prefix_targets.len(),
                representative_for_behavior.len(),
            );
        }
        behavior_by_source
            .into_iter()
            .map(|behavior| representative_for_behavior[&behavior])
            .collect()
    } else if exact_rep_confirmation_used {
        state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
            &tokenizer_view,
            &dedup.representative_token_bytes,
            &query_states,
            &normalized_disallowed_follows,
            None,
            None,
            Some(true),
            compatible_shared_base,
        )
    } else {
        state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
            &tokenizer_view,
            &dedup.representative_token_bytes,
            &query_states,
            &normalized_disallowed_follows,
            compatible_shared_base,
        )
    };
    let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;

    let tokenizer_states = if bounded_analysis.is_some() {
        let analysis_to_exact = query_states
            .iter()
            .copied()
            .zip(state_representatives.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let first_seed_class_for_analysis = seed_analysis_states
            .iter()
            .copied()
            .enumerate()
            .fold(
                BTreeMap::<usize, usize>::new(),
                |mut map, (seed_class, analysis_state)| {
                    map.entry(analysis_state).or_insert(seed_class);
                    map
                },
            );
        let final_seed_class_representatives = seed_analysis_states
            .iter()
            .map(|analysis_state| {
                let exact_analysis_state = analysis_to_exact[analysis_state];
                first_seed_class_for_analysis[&exact_analysis_state]
            })
            .collect::<Vec<_>>();
        // C is directional: each seed representative extends every member's
        // positional observations. Exact powerset refinement may merge C seed
        // classes, but the resulting class must retain the chosen C
        // representative. Picking the first raw member, as ordinary
        // equivalence composition does, can replace the representative with a
        // strictly less-defined member and lose terminal-NWA behavior.
        compose_raw_quotient_state_map_preserving_directional_representatives(
            seed,
            &final_seed_class_representatives,
        )
    } else {
        build_state_map_from_subset_representatives(
            &seed_states,
            &state_representatives,
            num_states,
            Some(seed),
        )
    };
    let mut final_state_representatives = if bounded_analysis.is_some() {
        // Exact refinement above is already expressed in bounded powerset-view
        // coordinates. The composed raw quotient deliberately chooses the first
        // raw member of each final class as its stored representative; that raw
        // member need not itself be one of C's seeded token-start representatives.
        // Re-projecting it into the bounded view is therefore invalid. Use the
        // exact analysis representatives directly for the subsequent vocab pass.
        state_representatives.clone()
    } else {
        tokenizer_states
            .representative_original_ids
            .iter()
            .map(|&state| state as usize)
            .collect::<Vec<_>>()
    };
    final_state_representatives.sort_unstable();
    final_state_representatives.dedup();

    let vocab_equiv_started_at = Instant::now();
    let (dedup_vocab_classes, vocab_analysis_dfa_build_ms) =
        vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
            &tokenizer_view,
            &dedup.representative_token_bytes,
            &final_state_representatives,
            effective_disallowed,
            Some(&byte_to_class),
            None,
            Some(&local_vocab_dfa_cache),
            Some(&local_analysis_dfa_cache),
        );
    let vocab_classes = expand_vocab_classes(
        dedup_vocab_classes,
        &dedup.original_to_repr,
        dedup.representative_token_bytes.len(),
    );
    let vocab_equiv_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;

    let finalize_started_at = Instant::now();
    let internal_id_map = InternalIdMap {
        tokenizer_states,
        vocab_tokens: build_vocab_map(&vocab_classes, &prepared.token_ids, prepared.max_token_id),
        deferred_vocab_singleton_original_ids: None,
    };
    let id_map_finalize_ms = finalize_started_at.elapsed().as_secs_f64() * 1000.0;
    let exact_reps = internal_id_map.tokenizer_states.num_internal_ids() as usize;

    Some((
        internal_id_map,
        CombinedEquivalenceProfile {
            initial_states_considered: seed_states.len(),
            max_length_skipped: true,
            max_token_len,
            token_len_gt_4: token_len_stats.gt_4,
            token_len_gt_8: token_len_stats.gt_8,
            token_len_gt_16: token_len_stats.gt_16,
            token_len_gt_32: token_len_stats.gt_32,
            token_len_gt_64: token_len_stats.gt_64,
            raw_analysis_base_init_ms: 0.0,
            analysis_view_build_ms,
            active_mask_filter_ms: 0.0,
            effective_follows_normalize_ms,
            prepare_inputs_ms,
            byte_class_setup_ms,
            vocab_analysis_dfa_build_ms,
            token_dedup_ms,
            restricted_observation_state_equiv_ms: 0.0,
            max_length_state_equiv_ms: 0.0,
            vocab_equiv_ms,
            exact_state_equiv_ms,
            id_map_finalize_ms,
            restricted_observation_reps: seed_states.len(),
            max_length_reps: seed_states.len(),
            exact_reps,
            exact_rep_confirmation_used,
        },
    ))
}

fn try_analyze_equivalences_with_raw_quotient(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    effective_disallowed: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    active_groups: Option<&[bool]>,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    initial_state_map: Option<&ManyToOneIdMap>,
    effective_follows_prepare_ms: f64,
    precomputed_raw_observations: Option<(&[u32], &[u32])>,
) -> Option<(InternalIdMap, CombinedEquivalenceProfile)> {
    let active_groups = active_groups?;
    let flat_trans = flat_trans?;
    if flat_trans.len() != tokenizer.num_states() as usize * 256 {
        return None;
    }

    let prepare_inputs_started_at = Instant::now();
    let prepared = prepare_equivalence_inputs(tokenizer, vocab, initial_state_map);
    let token_len_stats = token_length_stats(&prepared.token_bytes);
    let prepare_inputs_ms = prepare_inputs_started_at.elapsed().as_secs_f64() * 1000.0;

    // The direct raw refinement sees every byte that can occur in this local
    // vocabulary. It is deliberately at least as fine as the later deduped
    // quotient analysis.
    let mut raw_relevant_bytes = [false; 256];
    let mut direct_token_bytes = 0usize;
    let mut max_token_len = 0usize;
    for token in &prepared.token_bytes {
        direct_token_bytes += token.len();
        max_token_len = max_token_len.max(token.len());
        for &byte in *token {
            raw_relevant_bytes[byte as usize] = true;
        }
    }
    let active_byte_count = raw_relevant_bytes.iter().filter(|&&active| active).count();
    let projected_by_global = prepared.initial_states.len() < tokenizer.num_states() as usize;
    let skip_raw_quotient = should_skip_max_length_for_partition(
        partition_label,
        prepared.initial_states.len(),
        projected_by_global,
        direct_token_bytes,
        max_token_len,
        active_byte_count,
    );
    // The generic route still performs restricted observation, but for a tiny
    // local vocabulary it then keeps the full raw analysis DFA alive for the
    // exact state and token phases.  Materializing the exact restricted raw
    // quotient is cheaper in that regime: it avoids the 18k-state shared-base
    // setup and lets both remaining phases operate on the quotient.  Keep the
    // bound deliberately narrow; larger short-token partitions (notably p7)
    // can retain too much raw topology for this to win.
    let tiny_raw_quotient = prepared.token_bytes.len() <= RAW_QUOTIENT_TINY_VOCAB_MAX_TOKENS
        && direct_token_bytes <= RAW_QUOTIENT_TINY_VOCAB_MAX_BYTES;
    let structural_boundary_raw_quotient = matches!(partition_label, "p7" | "p8")
        && prepared.token_bytes.len() <= RAW_QUOTIENT_STRUCTURAL_BOUNDARY_MAX_TOKENS
        && direct_token_bytes <= RAW_QUOTIENT_STRUCTURAL_BOUNDARY_MAX_BYTES;
    if skip_raw_quotient && !tiny_raw_quotient && !structural_boundary_raw_quotient {
        return None;
    }

    // P8's local vocabulary has one common leading quote byte. For this one
    // partition, the full behavior of a source state on every local token is
    // determined by its quote successor. Evaluate one source representative
    // per distinct successor, then lift the exact result back to all raw
    // sources. This is a conservative factorization: equal quote successors
    // have identical complete token paths by definition.
    if partition_label == "p8" && super::super::p8_first_byte_factorization_allowed() {
        let common_first = prepared
            .token_bytes
            .first()
            .and_then(|token| token.first())
            .copied()?;
        if prepared
            .token_bytes
            .iter()
            .any(|token| token.first().copied() != Some(common_first))
        {
            return None;
        }

        // Every production P8 token has a nonempty identifier prefix after
        // the quote. Model terminal matches at the quote successor as
        // zero-position events, then analyze only the remaining suffix.
        // A one-byte quote token invalidates that factorization: its acceptance
        // is observed at the quote successor itself, not after a nonempty
        // suffix. Let the generic exact route handle such atypical partitions.
        let use_seeded_suffix_factorization =
            prepared.token_bytes.iter().all(|token| token.len() > 1);
        if !use_seeded_suffix_factorization {
            return None;
        }
        let mut target_to_source = BTreeMap::<u32, usize>::new();
        let mut source_representative = vec![usize::MAX; tokenizer.num_states() as usize];
        let mut target_by_source = vec![u32::MAX; tokenizer.num_states() as usize];
        for source in 0..tokenizer.num_states() as usize {
            let target = flat_trans[source * 256 + common_first as usize];
            if target == u32::MAX {
                continue;
            }
            let representative = *target_to_source.entry(target).or_insert(source);
            source_representative[source] = representative;
            target_by_source[source] = target;
        }
        let representative_sources: Vec<usize> = target_to_source.values().copied().collect();
        let quote_targets: Vec<usize> = target_to_source.keys().map(|&target| target as usize).collect();
        let suffix_tokens: Vec<&[u8]> = prepared
            .token_bytes
            .iter()
            .map(|token| &token[1..])
            .collect();
        debug_assert!(suffix_tokens.iter().all(|suffix| !suffix.is_empty()));
        let view_started_at = Instant::now();
        let analysis_view = TokenizerView::new_filtered_from_flat_trans(
            flat_trans,
            tokenizer,
            active_groups,
        );
        let analysis_view_build_ms = view_started_at.elapsed().as_secs_f64() * 1000.0;
        let exact_started_at = Instant::now();
        let representative_state_reps = if use_seeded_suffix_factorization {
            state_equivalence_analysis::find_state_equivalence_classes_with_sparse_disallowed_and_raw_transitions_with_initial_finalizers(
                &analysis_view,
                &suffix_tokens,
                &quote_targets,
                effective_disallowed,
            )
        } else {
            state_equivalence_analysis::find_state_equivalence_classes_with_sparse_disallowed_and_raw_transitions(
                &analysis_view,
                &prepared.token_bytes,
                &representative_sources,
                effective_disallowed,
            )
        };
        let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
        let mut source_rep_to_behavior_rep = HashMap::<usize, usize>::new();
        if use_seeded_suffix_factorization {
            for (&target, &behavior_rep) in quote_targets.iter().zip(&representative_state_reps) {
                source_rep_to_behavior_rep.insert(target, behavior_rep);
            }
        } else {
            for (&source, &behavior_rep) in representative_sources
                .iter()
                .zip(&representative_state_reps)
            {
                source_rep_to_behavior_rep.insert(source, behavior_rep);
            }
        }

        let id_map_finalize_started_at = Instant::now();
        let mut behavior_rep_to_internal = HashMap::<Option<usize>, u32>::new();
        let mut original_to_internal = vec![u32::MAX; tokenizer.num_states() as usize];
        let mut internal_to_originals = Vec::<Vec<u32>>::new();
        let mut representative_original_ids = Vec::<u32>::new();
        for source in 0..tokenizer.num_states() as usize {
            let behavior_rep = if use_seeded_suffix_factorization {
                (target_by_source[source] != u32::MAX)
                    .then(|| source_rep_to_behavior_rep[&(target_by_source[source] as usize)])
            } else {
                (source_representative[source] != usize::MAX).then(|| {
                    source_rep_to_behavior_rep[&source_representative[source]]
                })
            };
            let next = internal_to_originals.len() as u32;
            let internal = *behavior_rep_to_internal.entry(behavior_rep).or_insert_with(|| {
                internal_to_originals.push(Vec::new());
                representative_original_ids.push(source as u32);
                next
            });
            original_to_internal[source] = internal;
            internal_to_originals[internal as usize].push(source as u32);
        }
        let tokenizer_states = ManyToOneIdMap {
            original_to_internal,
            internal_to_originals,
            representative_original_ids,
        };
        let exact_reps = tokenizer_states.num_internal_ids() as usize;
        let internal_id_map = InternalIdMap {
            tokenizer_states,
            vocab_tokens: build_identity_vocab_map(&prepared.token_ids, prepared.max_token_id),
            deferred_vocab_singleton_original_ids: None,
        };
        let id_map_finalize_ms = id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][p8_first_byte_factorization] first_byte={} quote_targets={} seeded_suffix_factorized={} exact_reps={} view_ms={:.3} state_equiv_ms={:.3}",
                common_first,
                representative_sources.len(),
                use_seeded_suffix_factorization,
                exact_reps,
                analysis_view_build_ms,
                exact_state_equiv_ms,
            );
        }
        return Some((
            internal_id_map,
            CombinedEquivalenceProfile {
                initial_states_considered: prepared.initial_states.len(),
                max_length_skipped: false,
                max_token_len,
                token_len_gt_4: token_len_stats.gt_4,
                token_len_gt_8: token_len_stats.gt_8,
                token_len_gt_16: token_len_stats.gt_16,
                token_len_gt_32: token_len_stats.gt_32,
                token_len_gt_64: token_len_stats.gt_64,
                raw_analysis_base_init_ms: 0.0,
                analysis_view_build_ms,
                active_mask_filter_ms: 0.0,
                effective_follows_normalize_ms: effective_follows_prepare_ms,
                prepare_inputs_ms,
                byte_class_setup_ms: 0.0,
                vocab_analysis_dfa_build_ms: 0.0,
                token_dedup_ms: 0.0,
                restricted_observation_state_equiv_ms: 0.0,
                max_length_state_equiv_ms: 0.0,
                vocab_equiv_ms: 0.0,
                exact_state_equiv_ms,
                id_map_finalize_ms,
                restricted_observation_reps: representative_sources.len(),
                max_length_reps: representative_sources.len(),
                exact_reps,
                exact_rep_confirmation_used: false,
            },
        ));
    }

    let restricted_started_at = Instant::now();
    let raw_restricted = super::state_equivalence::restricted_observation::compute_state_map_raw(
        tokenizer,
        flat_trans,
        active_groups,
        &raw_relevant_bytes,
        precomputed_raw_observations,
    )?;
    let restricted_observation_state_equiv_ms =
        restricted_started_at.elapsed().as_secs_f64() * 1000.0;
    if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][raw_restricted_observation] partition={} labels_ms={:.3} refine_ms={:.3} certificate_ms={:.3} rounds={} reps={}",
            partition_label,
            raw_restricted.label_ms,
            raw_restricted.refine_ms,
            raw_restricted.certificate_ms,
            raw_restricted.rounds,
            raw_restricted.state_map.num_internal_ids(),
        );
    }
    let raw_observation_ids = raw_restricted.raw_observation_ids;
    let observation_representatives = raw_restricted.observation_representatives;
    let pre_state_map = raw_restricted.state_map;

    let use_compact_quotient_view = partition_label == "p7"
        && !super::super::l2p_terminal_interchangeability_strict_reference_enabled();
    let view_build_started_at = Instant::now();
    let (analysis_view, prebuilt_local_base, active_mask_filter_ms) = if use_compact_quotient_view {
        if let Some((view, base, build_ms)) =
            TokenizerView::new_filtered_quotient_from_flat_trans_with_observation_cache_and_relevant_base(
                flat_trans,
                tokenizer,
                active_groups,
                &pre_state_map,
                &raw_observation_ids,
                &observation_representatives,
                &raw_relevant_bytes,
            )
        {
            (view, Some(base), build_ms)
        } else {
            let (view, build_ms) =
                TokenizerView::new_filtered_quotient_from_flat_trans_with_observation_cache(
                    flat_trans,
                    tokenizer,
                    active_groups,
                    &pre_state_map,
                    &raw_observation_ids,
                    &observation_representatives,
                );
            (view, None, build_ms)
        }
    } else {
        let (view, build_ms) =
            TokenizerView::new_filtered_quotient_from_flat_trans_with_observation_cache(
                flat_trans,
                tokenizer,
                active_groups,
                &pre_state_map,
                &raw_observation_ids,
                &observation_representatives,
            );
        (view, None, build_ms)
    };
    let analysis_view_build_ms = view_build_started_at.elapsed().as_secs_f64() * 1000.0;

    // This exact byte partition is now over at most a few hundred quotient
    // rows. It replaces the previous 18k-state shared-base cold setup.
    let byte_class_setup_started_at = Instant::now();
    let local_shared_base = prebuilt_local_base.or_else(|| {
        super::vocab::fast::SharedVocabDfaBase::build_from_dfa_for_relevant_bytes(
            analysis_view.dfa(),
            &raw_relevant_bytes,
        )
    });
    let byte_to_class = local_shared_base
        .as_ref()
        .map(|base| base.byte_to_class())
        .unwrap_or_else(|| super::compat::compute_byte_classes(analysis_view.dfa()));
    let byte_class_setup_ms = byte_class_setup_started_at.elapsed().as_secs_f64() * 1000.0;

    let token_dedup_started_at = Instant::now();
    let dedup = deduplicate_tokens_by_byte_class(&prepared.token_bytes, &byte_to_class, prepared.lexical_entry_indices());
    let token_dedup_ms = token_dedup_started_at.elapsed().as_secs_f64() * 1000.0;

    // In strict-reference mode the primary build retains the old cloned dense
    // representation. The recursive reference runs with the guard suppressed
    // and therefore exercises the production borrowed/sparse form, so the
    // existing terminal-DWA equality check crosses the representations.
    let strict_reference_mode =
        super::super::l2p_terminal_interchangeability_strict_reference_enabled();
    let use_sparse_follow_rows = partition_label == "p8"
        && !strict_reference_mode;
    let use_borrowed_follow_rows = partition_label == "p7" && !strict_reference_mode;
    let follows_normalize_started_at = Instant::now();
    let group_count = tokenizer_group_count(&analysis_view);
    let normalized_disallowed_follows = (!use_sparse_follow_rows && !use_borrowed_follow_rows).then(|| {
        normalize_disallowed_follows(group_count, effective_disallowed)
    });
    let borrowed_disallowed_follows = use_borrowed_follow_rows.then(|| {
        (0..group_count)
            .map(|terminal| {
                effective_disallowed
                    .get(&(terminal as u32))
                    .filter(|bits| !bits.is_zero())
            })
            .collect::<Vec<_>>()
    });
    let effective_follows_normalize_ms = if use_sparse_follow_rows {
        effective_follows_prepare_ms
    } else {
        effective_follows_prepare_ms
            + follows_normalize_started_at.elapsed().as_secs_f64() * 1000.0
    };
    let pre_reduced_states: Vec<usize> = (0..pre_state_map.internal_to_originals.len()).collect();
    let run_state_refinement = |tokens: &[&[u8]]| {
        let exact_rep_confirmation_used = pre_reduced_states.len()
            >= EXACT_REP_CONFIRMATION_MIN_STATES
            && tokens.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
        let representatives = if use_sparse_follow_rows {
            if exact_rep_confirmation_used {
                state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_sparse_disallowed_and_shared_base(
                    &analysis_view,
                    tokens,
                    &pre_reduced_states,
                    effective_disallowed,
                    None,
                    None,
                    Some(true),
                    local_shared_base.as_ref(),
                )
            } else {
                state_equivalence_analysis::find_state_equivalence_classes_with_sparse_disallowed_and_shared_base(
                    &analysis_view,
                    tokens,
                    &pre_reduced_states,
                    effective_disallowed,
                    local_shared_base.as_ref(),
                )
            }
        } else if use_borrowed_follow_rows && exact_rep_confirmation_used {
            state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_borrowed_disallowed_and_shared_base(
                &analysis_view,
                tokens,
                &pre_reduced_states,
                borrowed_disallowed_follows
                    .as_deref()
                    .expect("borrowed follow rows must be present"),
                None,
                None,
                Some(true),
                local_shared_base.as_ref(),
            )
        } else if use_borrowed_follow_rows {
            state_equivalence_analysis::find_state_equivalence_classes_with_borrowed_disallowed_and_shared_base(
                &analysis_view,
                tokens,
                &pre_reduced_states,
                borrowed_disallowed_follows
                    .as_deref()
                    .expect("borrowed follow rows must be present"),
                local_shared_base.as_ref(),
            )
        } else if exact_rep_confirmation_used {
            state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
                &analysis_view,
                tokens,
                &pre_reduced_states,
                normalized_disallowed_follows
                    .as_deref()
                    .expect("dense follow rows must be present"),
                None,
                None,
                Some(true),
                local_shared_base.as_ref(),
            )
        } else {
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
                &analysis_view,
                tokens,
                &pre_reduced_states,
                normalized_disallowed_follows
                    .as_deref()
                    .expect("dense follow rows must be present"),
                local_shared_base.as_ref(),
            )
        };
        (representatives, exact_rep_confirmation_used)
    };

    // For vocab-only compilation the final tokenizer-state quotient is dead
    // output: callers retain only `vocab_tokens`.  The exact token quotient on
    // the restricted-observation pre-state quotient is already sufficient, and
    // state/vocab refinement commute.  Therefore always take vocabulary-first
    // order here and return before the final state refinement.  Full terminal
    // builds keep the historical selector because they consume both quotients.
    let vocab_first = !matches!(partition_label, "p7" | "p8")
        && dedup.representative_token_bytes.len() >= 512
        && pre_reduced_states.len() >= 256;
    if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][raw_quotient_equivalence_order] partition={} dedup_tokens={} pre_states={} vocab_first={}",
            partition_label,
            dedup.representative_token_bytes.len(),
            pre_reduced_states.len(),
            vocab_first,
        );
    }

    let (
        reduced_state_reps_for_pre_reduced,
        exact_rep_confirmation_used,
        precomputed_vocab,
        exact_state_equiv_ms,
        vocab_equiv_ms,
    ) = if vocab_first {
        let vocab_equiv_started_at = Instant::now();
        let precomputed_vocab =
            vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                &analysis_view,
                &dedup.representative_token_bytes,
                &pre_reduced_states,
                effective_disallowed,
                Some(&byte_to_class),
                None,
                None,
                None,
            );
        let vocab_equiv_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;
        let representative_tokens = representative_tokens_for_vocab_classes(
            &precomputed_vocab.0,
            &dedup.representative_token_bytes,
        );
        let exact_started_at = Instant::now();
        let (state_reps, exact_rep_confirmation_used) =
            run_state_refinement(&representative_tokens);
        let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
        (
            state_reps,
            exact_rep_confirmation_used,
            Some(precomputed_vocab),
            exact_state_equiv_ms,
            vocab_equiv_ms,
        )
    } else {
        let exact_started_at = Instant::now();
        let (state_reps, exact_rep_confirmation_used) =
            run_state_refinement(&dedup.representative_token_bytes);
        let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
        (
            state_reps,
            exact_rep_confirmation_used,
            None,
            exact_state_equiv_ms,
            0.0,
        )
    };

    let mut final_state_representatives = reduced_state_reps_for_pre_reduced.clone();
    final_state_representatives.sort_unstable();
    final_state_representatives.dedup();

    // P8 is the structural-boundary route. Its local source vocabulary is
    // deliberately tiny, while the generic token quotient can cost more than
    // the entire NWA/DWA construction and frequently leaves every local token
    // distinct. Retaining the identity token map is exact by construction: it
    // only declines a possible compression, and leaves the independently exact
    // scanner-state quotient unchanged.
    if matches!(partition_label, "p7" | "p8") {
        let id_map_finalize_started_at = Instant::now();
        let tokenizer_states = compose_raw_quotient_state_map(
            &pre_state_map,
            &reduced_state_reps_for_pre_reduced,
        );
        let exact_reps = tokenizer_states.num_internal_ids() as usize;
        let internal_id_map = InternalIdMap {
            tokenizer_states,
            vocab_tokens: build_identity_vocab_map(&prepared.token_ids, prepared.max_token_id),
            deferred_vocab_singleton_original_ids: None,
        };
        let id_map_finalize_ms = id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;
        return Some((
            internal_id_map,
            CombinedEquivalenceProfile {
                initial_states_considered: prepared.initial_states.len(),
                max_length_skipped: false,
                max_token_len,
                token_len_gt_4: token_len_stats.gt_4,
                token_len_gt_8: token_len_stats.gt_8,
                token_len_gt_16: token_len_stats.gt_16,
                token_len_gt_32: token_len_stats.gt_32,
                token_len_gt_64: token_len_stats.gt_64,
                raw_analysis_base_init_ms: 0.0,
                analysis_view_build_ms,
                active_mask_filter_ms,
                effective_follows_normalize_ms,
                prepare_inputs_ms,
                byte_class_setup_ms,
                vocab_analysis_dfa_build_ms: 0.0,
                token_dedup_ms,
                restricted_observation_state_equiv_ms,
                max_length_state_equiv_ms: 0.0,
                vocab_equiv_ms: 0.0,
                exact_state_equiv_ms,
                id_map_finalize_ms,
                restricted_observation_reps: pre_state_map.num_internal_ids() as usize,
                max_length_reps: pre_state_map.num_internal_ids() as usize,
                exact_reps,
                exact_rep_confirmation_used,
            },
        ));
    }

    let (dedup_vocab_classes, vocab_analysis_dfa_build_ms, vocab_equiv_ms) =
        if let Some((classes, build_ms)) = precomputed_vocab {
            (classes, build_ms, vocab_equiv_ms)
        } else {
            let vocab_equiv_started_at = Instant::now();
            let (classes, build_ms) =
                vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                    &analysis_view,
                    &dedup.representative_token_bytes,
                    &final_state_representatives,
                    effective_disallowed,
                    Some(&byte_to_class),
                    None,
                    None,
                    None,
                );
            (
                classes,
                build_ms,
                vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0,
            )
        };

    let id_map_finalize_started_at = Instant::now();
    let vocab_classes = expand_vocab_classes(
        dedup_vocab_classes,
        &dedup.original_to_repr,
        dedup.representative_token_bytes.len(),
    );
    let tokenizer_states = compose_raw_quotient_state_map(
        &pre_state_map,
        &reduced_state_reps_for_pre_reduced,
    );
    let exact_reps = tokenizer_states.num_internal_ids() as usize;
    let internal_id_map = InternalIdMap {
        tokenizer_states,
        vocab_tokens: build_vocab_map(&vocab_classes, &prepared.token_ids, prepared.max_token_id),
        deferred_vocab_singleton_original_ids: None,
    };
    let id_map_finalize_ms = id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;

    Some((
        internal_id_map,
        CombinedEquivalenceProfile {
            initial_states_considered: prepared.initial_states.len(),
            max_length_skipped: false,
            max_token_len,
            token_len_gt_4: token_len_stats.gt_4,
            token_len_gt_8: token_len_stats.gt_8,
            token_len_gt_16: token_len_stats.gt_16,
            token_len_gt_32: token_len_stats.gt_32,
            token_len_gt_64: token_len_stats.gt_64,
            raw_analysis_base_init_ms: 0.0,
            analysis_view_build_ms,
            active_mask_filter_ms,
            effective_follows_normalize_ms,
            prepare_inputs_ms,
            byte_class_setup_ms,
            vocab_analysis_dfa_build_ms,
            token_dedup_ms,
            restricted_observation_state_equiv_ms,
            max_length_state_equiv_ms: 0.0,
            vocab_equiv_ms,
            exact_state_equiv_ms,
            id_map_finalize_ms,
            restricted_observation_reps: pre_state_map.num_internal_ids() as usize,
            max_length_reps: pre_state_map.num_internal_ids() as usize,
            exact_reps,
            exact_rep_confirmation_used,
        },
    ))
}

/// A finite, explicitly scoped observation query. The optional total quotient
/// certifies continuation topology; it never expands the initial query set.
#[derive(Clone, Copy)]
pub struct ScopedEquivalenceDomain<'a> {
    pub continuation_state_map: Option<&'a ManyToOneIdMap>,
}

pub fn analyze_equivalences_with_group_filter(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    disallowed_follows_are_ignore_transparent: bool,
    pre_normalized_disallowed_follows: Option<&[BitSet]>,
    active_groups: Option<&[bool]>,
    shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&super::vocab::fast::SharedVocabAnalysisDfaCache>,
    shared_base_setup_ms: f64,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    shared_transition_cache: Option<&std::sync::OnceLock<super::compat::FlatTransitionCache>>,
    initial_state_map: Option<&ManyToOneIdMap>,
    initial_state_map_has_stable_restricted_observation: bool,
    scoped_domain: Option<ScopedEquivalenceDomain<'_>>,
    token_position_partition: Option<&GlobalTokenPositionStatePartition>,
    precomputed_raw_observations: Option<(&[u32], &[u32])>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> (InternalIdMap, CombinedEquivalenceProfile) {
    let use_prepartition = std::env::var("GLRMASK_VOCAB_PARTITION_L2P_PRECLASS")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(false);
    if use_prepartition {
        return analyze_vocab_first_transition_prepartition(
            partition_label,
            tokenizer,
            vocab,
            disallowed_follows,
            ignore_terminal,
            disallowed_follows_are_ignore_transparent,
            active_groups,
            shared_vocab_dfa_cache,
            shared_analysis_dfa_cache,
            flat_trans,
            initial_state_map,
            prebuilt_token_trie,
        );
    }
    analyze_equivalences_impl(
        partition_label,
        tokenizer,
        vocab,
        disallowed_follows,
        ignore_terminal,
        disallowed_follows_are_ignore_transparent,
        pre_normalized_disallowed_follows,
        active_groups,
        shared_vocab_dfa_cache,
        shared_analysis_dfa_cache,
        shared_base_setup_ms,
        flat_trans,
        shared_transition_cache,
        initial_state_map,
        initial_state_map_has_stable_restricted_observation,
        scoped_domain,
        token_position_partition,
        precomputed_raw_observations,
        prebuilt_token_trie,
        false,
    )
}

/// Vocabulary-only variant of the exact combined analysis.
///
/// When the commuting quotient pipeline selects vocabulary-first order, the
/// token partition is already exact before the final tokenizer-state quotient
/// is computed.  In that case this returns immediately after materializing the
/// token map.  Otherwise it deliberately falls back to the full combined
/// analysis rather than weakening the equivalence relation.
pub fn analyze_vocab_equivalences_with_group_filter(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    disallowed_follows_are_ignore_transparent: bool,
    pre_normalized_disallowed_follows: Option<&[BitSet]>,
    active_groups: Option<&[bool]>,
    shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&super::vocab::fast::SharedVocabAnalysisDfaCache>,
    shared_base_setup_ms: f64,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    shared_transition_cache: Option<&std::sync::OnceLock<super::compat::FlatTransitionCache>>,
    initial_state_map: Option<&ManyToOneIdMap>,
    initial_state_map_has_stable_restricted_observation: bool,
    scoped_domain: Option<ScopedEquivalenceDomain<'_>>,
    token_position_partition: Option<&GlobalTokenPositionStatePartition>,
    precomputed_raw_observations: Option<(&[u32], &[u32])>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> (InternalIdMap, CombinedEquivalenceProfile) {
    analyze_equivalences_impl(
        partition_label,
        tokenizer,
        vocab,
        disallowed_follows,
        ignore_terminal,
        disallowed_follows_are_ignore_transparent,
        pre_normalized_disallowed_follows,
        active_groups,
        shared_vocab_dfa_cache,
        shared_analysis_dfa_cache,
        shared_base_setup_ms,
        flat_trans,
        shared_transition_cache,
        initial_state_map,
        initial_state_map_has_stable_restricted_observation,
        scoped_domain,
        token_position_partition,
        precomputed_raw_observations,
        prebuilt_token_trie,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn analyze_vocab_first_transition_prepartition(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    disallowed_follows_are_ignore_transparent: bool,
    active_groups: Option<&[bool]>,
    shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&super::vocab::fast::SharedVocabAnalysisDfaCache>,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    initial_state_map: Option<&ManyToOneIdMap>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> (InternalIdMap, CombinedEquivalenceProfile) {
    let total_started_at = Instant::now();
    let follows_started_at = Instant::now();
    let token_path_disallowed_follows = (!disallowed_follows_are_ignore_transparent)
        .then(|| ignore_transparent_disallowed_follows(disallowed_follows, ignore_terminal));
    let effective_disallowed = token_path_disallowed_follows
        .as_ref()
        .unwrap_or(disallowed_follows);
    let effective_follows_normalize_ms = follows_started_at.elapsed().as_secs_f64() * 1000.0;

    let prepare_started_at = Instant::now();
    let prepared = prepare_equivalence_inputs(tokenizer, vocab, initial_state_map);
    let token_len_stats = token_length_stats(&prepared.token_bytes);
    let max_token_len = prepared.token_bytes.iter().map(|token| token.len()).max().unwrap_or(0);
    let prepare_inputs_ms = prepare_started_at.elapsed().as_secs_f64() * 1000.0;

    let view_started_at = Instant::now();
    let owned_view;
    let mut bounded_initial_states = Vec::new();
    let tokenizer_view = if tokenizer.has_epsilon_transitions() {
        let bounded_prebuilt_trie = prebuilt_token_trie
            .filter(|trie| trie.is_reasonable_superset_for(prepared.token_bytes.len()));
        let bounded = build_bounded_analysis_view_with_trie(
            tokenizer,
            &prepared.initial_states,
            &prepared.token_bytes,
            active_groups,
            bounded_prebuilt_trie,
        );
        bounded_initial_states = prepared
            .initial_states
            .iter()
            .map(|&raw| bounded.view_state_for_raw_start(raw))
            .collect();
        bounded_initial_states.sort_unstable();
        bounded_initial_states.dedup();
        owned_view = bounded.tokenizer_view;
        &owned_view
    } else {
        let compatible_flat_trans =
            flat_trans.filter(|ft| ft.len() == tokenizer.num_states() as usize * 256);
        owned_view = match (active_groups, compatible_flat_trans) {
            (Some(groups), Some(ft)) => {
                TokenizerView::new_filtered_from_flat_trans(ft, tokenizer, groups)
            }
            (Some(groups), None) => TokenizerView::new_filtered(tokenizer, groups),
            (None, Some(ft)) => TokenizerView::new_from_flat_trans(ft, tokenizer),
            _ => TokenizerView::new(tokenizer),
        };
        &owned_view
    };
    let analysis_view_build_ms = view_started_at.elapsed().as_secs_f64() * 1000.0;
    let analysis_initial_states = if tokenizer.has_epsilon_transitions() {
        bounded_initial_states.as_slice()
    } else {
        prepared.initial_states.as_slice()
    };

    // Bounded epsilon views use partition-local state coordinates, so never
    // reuse a grammar-wide DFA cache whose key does not encode that topology.
    let (analysis_vocab_dfa_cache, analysis_dfa_cache) = if tokenizer.has_epsilon_transitions() {
        (None, None)
    } else {
        (shared_vocab_dfa_cache, shared_analysis_dfa_cache)
    };

    let vocab_started_at = Instant::now();
    let (vocab_classes, vocab_analysis_dfa_build_ms) =
        vocab_equivalence_analysis::find_vocab_first_transition_preclasses_with_group_filter_profiled(
            tokenizer_view,
            &prepared.token_bytes,
            analysis_initial_states,
            effective_disallowed,
            None,
            active_groups,
            analysis_vocab_dfa_cache,
            analysis_dfa_cache,
        );
    let vocab_equiv_ms = vocab_started_at.elapsed().as_secs_f64() * 1000.0;

    let finalize_started_at = Instant::now();
    let vocab_tokens = build_vocab_map(&vocab_classes, &prepared.token_ids, prepared.max_token_id);
    let tokenizer_states = initial_state_map.cloned().unwrap_or_else(|| {
        let count = tokenizer.num_states();
        ManyToOneIdMap::from_original_to_internal_allowing_unmapped((0..count).collect(), count)
    });
    let reps = tokenizer_states.num_internal_ids() as usize;
    let id_map_finalize_ms = finalize_started_at.elapsed().as_secs_f64() * 1000.0;
    if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
    {
        eprintln!(
            "[glrmask/profile][vocab_l2p_first_transition_prepartition] partition={} tokens={} classes={} states={} total_ms={:.3} view_ms={:.3} prepare_ms={:.3} factor_ms={:.3} finalize_ms={:.3}",
            partition_label,
            vocab.len(),
            vocab_tokens.num_internal_ids(),
            analysis_initial_states.len(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
            analysis_view_build_ms,
            prepare_inputs_ms,
            vocab_equiv_ms,
            id_map_finalize_ms,
        );
    }
    (
        InternalIdMap {
            tokenizer_states,
            vocab_tokens,
            deferred_vocab_singleton_original_ids: None,
        },
        CombinedEquivalenceProfile {
            initial_states_considered: analysis_initial_states.len(),
            max_length_skipped: true,
            max_token_len,
            token_len_gt_4: token_len_stats.gt_4,
            token_len_gt_8: token_len_stats.gt_8,
            token_len_gt_16: token_len_stats.gt_16,
            token_len_gt_32: token_len_stats.gt_32,
            token_len_gt_64: token_len_stats.gt_64,
            raw_analysis_base_init_ms: 0.0,
            analysis_view_build_ms,
            active_mask_filter_ms: 0.0,
            effective_follows_normalize_ms,
            prepare_inputs_ms,
            byte_class_setup_ms: 0.0,
            vocab_analysis_dfa_build_ms,
            token_dedup_ms: 0.0,
            restricted_observation_state_equiv_ms: 0.0,
            max_length_state_equiv_ms: 0.0,
            vocab_equiv_ms,
            exact_state_equiv_ms: 0.0,
            id_map_finalize_ms,
            restricted_observation_reps: reps,
            max_length_reps: reps,
            exact_reps: reps,
            exact_rep_confirmation_used: false,
        },
    )
}

/// Combined equivalence analysis over a flattened tokenizer DFA.
///
/// Uses state equivalence (k-step hashing plus token-based refinement) followed
/// by vocab equivalence (parallel batched with byte-class compression).
fn analyze_equivalences_impl(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    disallowed_follows_are_ignore_transparent: bool,
    pre_normalized_disallowed_follows: Option<&[BitSet]>,
    active_groups: Option<&[bool]>,
    shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&super::vocab::fast::SharedVocabAnalysisDfaCache>,
    shared_base_setup_ms: f64,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    shared_transition_cache: Option<&std::sync::OnceLock<super::compat::FlatTransitionCache>>,
    initial_state_map: Option<&ManyToOneIdMap>,
    initial_state_map_has_stable_restricted_observation: bool,
    scoped_domain: Option<ScopedEquivalenceDomain<'_>>,
    token_position_partition: Option<&GlobalTokenPositionStatePartition>,
    precomputed_raw_observations: Option<(&[u32], &[u32])>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
    vocab_only: bool,
) -> (InternalIdMap, CombinedEquivalenceProfile) {
    let continuation_state_map = scoped_domain.and_then(|domain| domain.continuation_state_map);
    let prebuilt_token_trie = std::env::var("GLRMASK_USE_PREBUILT_L2P_TOKEN_TRIE")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true)
        .then_some(prebuilt_token_trie)
        .flatten();
    if tokenizer.has_epsilon_transitions() {
        let follows_prepare_started_at = Instant::now();
        let token_path_disallowed_follows = (!disallowed_follows_are_ignore_transparent)
            .then(|| ignore_transparent_disallowed_follows(disallowed_follows, ignore_terminal));
        let effective_follows_prepare_ms =
            follows_prepare_started_at.elapsed().as_secs_f64() * 1000.0;
        let effective_disallowed = token_path_disallowed_follows
            .as_ref()
            .unwrap_or(disallowed_follows);

        // C is valid at LLM-token start boundaries. The C-seeded route performs
        // only whole-token analysis, so it is sound to use before the absolute
        // restricted-observation pipeline. Epsilon lexers use a bounded
        // configuration view inside that route and project back to raw TSIDs.
        if let Some(token_position_partition) = token_position_partition {
            if let Some(result) = try_analyze_equivalences_with_token_position_partition(
                partition_label,
                tokenizer,
                vocab,
                effective_disallowed,
                active_groups,
                shared_vocab_dfa_cache,
                shared_analysis_dfa_cache,
                flat_trans,
                token_position_partition,
                effective_follows_prepare_ms,
                pre_normalized_disallowed_follows,
            ) {
                return result;
            }
        }

        let prepare_started_at = Instant::now();
        let prepared = prepare_equivalence_inputs(tokenizer, vocab, initial_state_map);
        let token_len_stats = token_length_stats(&prepared.token_bytes);
        let prepare_inputs_ms = prepare_started_at.elapsed().as_secs_f64() * 1000.0;

        let token_dedup_started_at = Instant::now();
        let identity_token_count = prepared.token_bytes.len();
        let active_byte_class_started_at = Instant::now();
        let active_language_byte_classes = (prepared.token_bytes.len()
            >= ACTIVE_LANGUAGE_BYTE_DEDUP_MIN_TOKENS)
            .then(|| {
                active_groups.map(|active_groups| {
                    if let Some(flat_trans) = flat_trans {
                        super::compat::compute_active_physical_byte_classes_from_flat_trans(
                            tokenizer,
                            active_groups,
                            flat_trans.as_ref(),
                        )
                    } else {
                        let view = TokenizerView::new_filtered(tokenizer, active_groups);
                        super::compat::compute_byte_classes(view.dfa())
                    }
                })
            })
            .flatten();
        let active_mask_filter_ms = active_byte_class_started_at.elapsed().as_secs_f64() * 1000.0;
        // Active-byte equivalence subsumes exact-byte equivalence: equal byte
        // strings necessarily have equal projected class strings.  When the
        // active congruence exists, one projected dedup pass therefore replaces
        // the old identity-dedup + projected-dedup pair exactly.
        let dedup = if let Some(byte_to_class) = active_language_byte_classes.as_ref() {
            deduplicate_tokens_by_byte_class(&prepared.token_bytes, byte_to_class, prepared.lexical_entry_indices())
        } else {
            let identity_byte_class: [u8; 256] = std::array::from_fn(|byte| byte as u8);
            deduplicate_tokens_by_byte_class(&prepared.token_bytes, &identity_byte_class, prepared.lexical_entry_indices())
        };
        let token_dedup_ms = token_dedup_started_at.elapsed().as_secs_f64() * 1000.0
            - active_mask_filter_ms;
        if (std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some())
            && let Some(byte_to_class) = active_language_byte_classes.as_ref()
        {
            eprintln!(
                "[glrmask/profile][active_language_byte_dedup] partition={} active_terminals={} byte_classes={} identity_tokens={} canonical_tokens={} build_ms={:.3}",
                partition_label,
                active_groups.map_or(0, |active| active.iter().filter(|&&value| value).count()),
                byte_to_class
                    .iter()
                    .copied()
                    .max()
                    .map_or(0, |class| class as usize + 1),
                identity_token_count,
                dedup.representative_token_bytes.len(),
                active_mask_filter_ms,
            );
        }
        let max_token_len = dedup
            .representative_token_bytes
            .iter()
            .map(|token| token.len())
            .max()
            .unwrap_or(0);
        let mut relevant_bytes = [false; 256];
        for token in &dedup.representative_token_bytes {
            for &byte in *token {
                relevant_bytes[byte as usize] = true;
            }
        }
        let direct_token_bytes = dedup
            .representative_token_bytes
            .iter()
            .map(|token| token.len())
            .sum();
        let active_byte_count = relevant_bytes.iter().filter(|&&active| active).count();
        let projected_by_global = prepared.initial_states.len() < tokenizer.num_states() as usize;
        let max_length_skipped = should_skip_max_length_for_partition(
            partition_label,
            prepared.initial_states.len(),
            projected_by_global,
            direct_token_bytes,
            max_token_len,
            active_byte_count,
        );
        // Query only the actual candidate token trajectories. A whole-byte-
        // alphabet powerset may be much larger despite fewer initial roots,
        // and later exact analysis pays for that unnecessary topology. Both
        // existing views compute the same exact relation. Scope is explicit,
        // not inferred from a partition label (ordinary p0/p1 names survive
        // composition). Explicit diagnostic overrides remain authoritative.
        let analysis_view_policy = if scoped_domain.is_some()
            && std::env::var_os("GLRMASK_L2P_NFA_RELEVANT_POWERSET_VIEW").is_none()
        {
            L2pNfaAnalysisViewPolicy::Bounded
        } else {
            l2p_nfa_analysis_view_policy(partition_label)
        };
        let powerset_max_states = l2p_nfa_relevant_powerset_max_states();
        let powerset_min_bounded_pairs =
            l2p_nfa_relevant_powerset_min_bounded_pairs(partition_label);
        let prepass_pair_estimate = prepared
            .initial_states
            .len()
            .saturating_mul(dedup.representative_token_bytes.len());
        // Even when the incoming TI map already certifies the mandatory
        // restricted-observation fixed point, keep constructing this topology
        // when the normal policy selects it: the later exact analysis view can
        // reuse the same sparse powerset.  The certificate removes only the
        // redundant refinement pass, not a topology that remains useful.
        let should_probe_prepass_powerset = should_probe_l2p_nfa_powerset(
            analysis_view_policy,
            prepass_pair_estimate,
            powerset_min_bounded_pairs,
        );
        let prepass_relevant_bytes = representative_byte_mask(
            &relevant_bytes,
            active_language_byte_classes.as_ref(),
        );
        let prepass_powerset_started_at = Instant::now();
        // The TI seed is already a stable restricted-observation right
        // congruence for this exact active-terminal projection and vocabulary
        // byte alphabet. Building the sparse relevant powerset over those
        // certified classes is therefore exact and avoids reconstructing the
        // same quotient topology over raw epsilon-NFA states.
        // The incoming scoped map is deliberately partial: it quotients only
        // token-start states in the requested component.  The powerset view,
        // however, is a continuation topology and may step into any raw state
        // reachable after the current link reset.  Give those omitted states
        // private singleton classes for this topology only.  This preserves
        // the scoped initial equivalence relation while satisfying the
        // powerset builder's total-state-map contract and keeps post-commit
        // continuation exact.
        // An explicitly supplied total continuation certificate may contain
        // classes outside the token-start query domain. It is used ONLY to
        // build transitions, never to widen prepared.initial_states. Keeping
        // these domains separate preserves the proof while making query cost
        // proportional to represented start classes rather than the whole Q.
        let completed_certified_powerset_seed_map =
            (initial_state_map_has_stable_restricted_observation && continuation_state_map.is_none())
                .then(|| {
                    initial_state_map.and_then(|map| {
                        map.original_to_internal
                            .iter()
                            .any(|&class| class == u32::MAX)
                            .then(|| {
                                let mut completed = map.clone();
                                super::super::super::scope::complete_with_continuation_singletons(
                                    &mut completed,
                                );
                                completed
                            })
                    })
                })
                .flatten();
        let certified_powerset_seed_map = if initial_state_map_has_stable_restricted_observation {
            continuation_state_map.or(completed_certified_powerset_seed_map
                .as_ref()
                .or(initial_state_map))
        } else {
            None
        };
        let (mut prepass_powerset_candidate, prepass_powerset_aborted) =
            if should_probe_prepass_powerset {
                build_l2p_nfa_powerset_candidate(
                    analysis_view_policy,
                    tokenizer,
                    &prepass_relevant_bytes,
                    active_groups,
                    certified_powerset_seed_map,
                    powerset_max_states,
                )
            } else {
                (None, None)
            };
        let prepass_powerset_build_ms =
            prepass_powerset_started_at.elapsed().as_secs_f64() * 1000.0;
        let prepass_powerset_states = prepass_powerset_candidate
            .as_ref()
            .map_or(0, |powerset| powerset.states.len());
        let use_prebuilt_nfa_refinement = should_use_l2p_nfa_prepass_powerset(
            analysis_view_policy,
            prepass_powerset_candidate.is_some(),
        );
        let prepass_output_classes = use_prebuilt_nfa_refinement
            .then(|| {
                prepass_powerset_candidate
                    .as_ref()
                    .map(powerset_output_class_ids)
            })
            .flatten();
        let prebuilt_nfa_refinement = prepass_powerset_candidate
            .as_ref()
            .zip(prepass_output_classes.as_ref())
            .map(|(powerset, output_class_by_config)| PrebuiltSparsePowersetRefinement {
                raw_start_to_view: powerset.raw_start_to_view.as_ref(),
                configurations: powerset.configurations.as_ref(),
                output_class_by_config,
                edge_offsets: &powerset.edge_offsets,
                edges: &powerset.edges,
            });
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][l2p_nfa_prepass_view] partition={} policy={} selected={} active_bytes={} pair_estimate={} powerset_states={} powerset_max_states={} aborted_configurations={} aborted_edges={} build_ms={:.3}",
                partition_label,
                analysis_view_policy.as_str(),
                if prebuilt_nfa_refinement.is_some() {
                    "projected_sparse_powerset"
                } else {
                    "direct_nfa"
                },
                prepass_relevant_bytes.iter().filter(|&&active| active).count(),
                prepass_pair_estimate,
                prepass_powerset_states,
                powerset_max_states,
                prepass_powerset_aborted.map_or(0, |work| work.configurations),
                prepass_powerset_aborted.map_or(0, |work| work.edges),
                prepass_powerset_build_ms,
            );
        }

        let mut pipeline_config = resolve_l2p_pipeline_config(!max_length_skipped);
        // On large p1 domains the exact finite-vocabulary refinement already
        // collapses the state space aggressively. The preliminary stable
        // restricted-observation pass walks tens of thousands of states but saves
        // less work downstream than it costs. The exact vocabulary/state passes
        // remain authoritative, so omitting this optional prequotient is semantic-
        // preserving.
        let auto_skip_pipeline_prepass = partition_label == "p1"
            && prepared.initial_states.len() >= 20_000
            && dedup.representative_token_bytes.len() >= 10_000;
        // Static-link boundary shards analyze a merged tokenizer under the
        // full vocabulary: restricted observation provably finds nothing
        // (26,046 classes on selected10) and max-length refinement is empty,
        // so the pipeline is pure overhead (~0.5 s). The exact
        // vocabulary/state passes remain authoritative, so omitting this
        // optional prequotient is semantic-preserving (same argument as the
        // p1 auto-skip above).
        let boundary_shard_skip_pipeline_prepass =
            partition_label.starts_with("boundary_shard");
        // A vocab-only caller never consumes the final tokenizer-state
        // quotient.  When the incoming state domain is already small, keeping
        // it exact (identity / inherited map) is cheaper than computing a
        // preliminary state quotient and is strictly conservative for the
        // subsequent exact vocabulary relation.
        let vocab_only_direct_states = vocab_only && prepared.initial_states.len() <= 256;
        let skip_pipeline_prepass = vocab_only_direct_states
            || auto_skip_pipeline_prepass
            || boundary_shard_skip_pipeline_prepass
            || std::env::var("GLRMASK_SKIP_L2P_STATE_EQUIV_PIPELINE_PARTITIONS")
                .ok()
                .is_some_and(|scope| {
                    scope
                        .split(',')
                        .map(str::trim)
                        .any(|candidate| candidate == partition_label)
                });
        if skip_pipeline_prepass {
            pipeline_config.passes.clear();
        }
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][l2p_state_pipeline_plan] partition={} skip={} passes={:?}",
                partition_label,
                skip_pipeline_prepass,
                pipeline_config.passes,
            );
        }
        let (tokenizer_states, pipeline_profile) = if vocab_only_direct_states {
            // `vocab_only_direct_states` deliberately keeps the incoming state
            // coordinate exact: there are no state-equivalence passes to run.
            // Avoid entering the generic pipeline here. Even with an empty
            // pass list it computes the cached max-length vocabulary statistic,
            // which is pure setup overhead for this vocab-only result and is
            // never consumed before returning.
            let tokenizer_states = initial_state_map
                .cloned()
                .unwrap_or_else(|| identity_state_map(tokenizer.num_states() as usize));
            let reps = tokenizer_states.num_internal_ids() as usize;
            (
                tokenizer_states,
                StateEquivalencePipelineProfile {
                    restricted_observation_reps: reps,
                    max_length_skipped: true,
                    max_length_reps: reps,
                    ..StateEquivalencePipelineProfile::default()
                },
            )
        } else {
            run_state_equivalence_pipeline_with_initial_restricted_observation_certificate(
                tokenizer,
                vocab,
                initial_state_map,
                active_groups,
                StateEquivalenceScope::L2p,
                &pipeline_config,
                prebuilt_nfa_refinement.as_ref(),
                None,
                None,
                initial_state_map_has_stable_restricted_observation,
            )
        };
        if initial_state_map_has_stable_restricted_observation
            && std::env::var_os("GLRMASK_VALIDATE_TI_RESTRICTED_OBSERVATION_REUSE").is_some()
        {
            let (reference, _) = run_state_equivalence_pipeline(
                tokenizer,
                vocab,
                initial_state_map,
                active_groups,
                StateEquivalenceScope::L2p,
                &pipeline_config,
                None,
                None,
                None,
            );
            // The legacy reference expands excluded raw states into additional
            // classes. Compare both partitions on the actual requested seed
            // domain, not on that unrelated continuation-only expansion.
            // Every equivalence class/member within the query is still checked.
            let reference = if scoped_domain.is_some() {
                crate::terminal_dwa::scope::restrict_quotient_to_initial_domain(
                    &reference, initial_state_map.expect("certified scoped initial domain"),
                ).expect("reference quotient must cover the scoped raw domain")
            } else { reference };
            assert_same_many_to_one_partition(
                &tokenizer_states,
                &reference,
                "TI restricted-observation reuse changed the state partition",
            );
            eprintln!(
                "[glrmask/profile][ti_restricted_observation_reuse_validation] partition={} result=equivalent reps={}",
                partition_label,
                tokenizer_states.num_internal_ids(),
            );
        }

        let raw_pre_representatives = tokenizer_states
            .representative_original_ids
            .iter()
            .map(|&state| state as usize)
            .collect::<Vec<_>>();
        let analysis_view_started_at = Instant::now();
        let bounded_pair_estimate = raw_pre_representatives
            .len()
            .saturating_mul(dedup.representative_token_bytes.len());
        let should_probe_powerset = should_probe_l2p_nfa_powerset(
            analysis_view_policy,
            bounded_pair_estimate,
            powerset_min_bounded_pairs,
        );
        let powerset_started_at = Instant::now();
        let can_reuse_prepass_powerset = prepass_relevant_bytes == relevant_bytes;
        let (mut powerset_candidate, powerset_aborted) = if !should_probe_powerset {
            (None, None)
        } else if can_reuse_prepass_powerset {
            if let Some(candidate) = prepass_powerset_candidate.take() {
                (Some(candidate), None)
            } else if prepass_powerset_aborted.is_some() {
                // The same alphabet already exceeded the adaptive budget.
                // Do not repeat the failed construction after refinement.
                (None, prepass_powerset_aborted)
            } else {
                build_l2p_nfa_powerset_candidate(
                    analysis_view_policy,
                    tokenizer,
                    &relevant_bytes,
                    active_groups,
                    certified_powerset_seed_map,
                    powerset_max_states,
                )
            }
        } else {
            build_l2p_nfa_powerset_candidate(
                analysis_view_policy,
                tokenizer,
                &relevant_bytes,
                active_groups,
                certified_powerset_seed_map,
                powerset_max_states,
            )
        };
        let powerset_build_ms = powerset_started_at.elapsed().as_secs_f64() * 1000.0;
        let powerset_probed = powerset_candidate.is_some();
        let powerset_states = powerset_candidate
            .as_ref()
            .map_or(0, |powerset| powerset.states.len());
        let use_powerset = should_use_l2p_nfa_powerset(
            analysis_view_policy,
            powerset_probed,
            powerset_states,
            powerset_max_states,
        );
        let (analysis_view_owned, preclass_view_states, analysis_view_kind) = if use_powerset {
            let powerset = powerset_candidate
                .take()
                .expect("powerset analysis policy must build a powerset candidate");
            let preclass_view_states = raw_pre_representatives
                .iter()
                .map(|&raw| powerset.view_state_for_raw_start(raw))
                .collect::<Vec<_>>();
            (
                powerset.into_tokenizer_view(),
                preclass_view_states,
                "powerset",
            )
        } else if let Some(powerset) = powerset_candidate.as_ref() {
            let bounded = build_bounded_analysis_view_from_relevant_powerset(
                powerset,
                &raw_pre_representatives,
                &dedup.representative_token_bytes,
            );
            let preclass_view_states = raw_pre_representatives
                .iter()
                .map(|&raw| bounded.view_state_for_raw_start(raw))
                .collect::<Vec<_>>();
            (
                bounded.tokenizer_view,
                preclass_view_states,
                "bounded_from_powerset",
            )
        } else {
            let bounded_prebuilt_trie = prebuilt_token_trie.filter(|trie| {
                trie.is_reasonable_superset_for(dedup.representative_token_bytes.len())
            });
            let bounded = build_bounded_analysis_view_with_trie(
                tokenizer,
                &raw_pre_representatives,
                &dedup.representative_token_bytes,
                active_groups,
                bounded_prebuilt_trie,
            );
            let preclass_view_states = raw_pre_representatives
                .iter()
                .map(|&raw| bounded.view_state_for_raw_start(raw))
                .collect::<Vec<_>>();
            (bounded.tokenizer_view, preclass_view_states, "bounded")
        };
        let analysis_view_build_ms =
            analysis_view_started_at.elapsed().as_secs_f64() * 1000.0;
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][l2p_nfa_analysis_view] partition={} policy={} selected={} bounded_pair_estimate={} powerset_min_bounded_pairs={} powerset_probed={} powerset_states={} powerset_max_states={} aborted_configurations={} aborted_edges={} powerset_build_ms={:.3} total_build_ms={:.3}",
                partition_label,
                analysis_view_policy.as_str(),
                analysis_view_kind,
                bounded_pair_estimate,
                powerset_min_bounded_pairs,
                powerset_probed,
                powerset_states,
                powerset_max_states,
                powerset_aborted.map_or(0, |work| work.configurations),
                powerset_aborted.map_or(0, |work| work.edges),
                powerset_build_ms,
                analysis_view_build_ms,
            );
        }
        let analysis_view = &analysis_view_owned;

        let byte_class_started_at = Instant::now();
        // The full active-language classes remain sound for token-spelling
        // canonicalization above, but the bounded analysis view materializes
        // only vocabulary-relevant byte trajectories. That pruning can split a
        // full-language byte class, so transition compression must use classes
        // computed from this exact view rather than reusing the earlier table.
        let use_local_shared_base = super::super::local_shared_equiv_base_enabled();
        let local_vocab_dfa_cache = vocab_equivalence_analysis::SharedVocabDfaCache::new();
        if use_local_shared_base {
            let _ = local_vocab_dfa_cache.set(
                vocab_equivalence_analysis::SharedVocabDfaBase::build_from_dfa_relevant(
                    analysis_view.dfa(),
                    &relevant_bytes,
                ),
            );
        }
        let local_shared_base = local_vocab_dfa_cache.get();
        let local_shared_cache = local_shared_base.map(|_| &local_vocab_dfa_cache);
        let byte_to_class = local_shared_base
            .map(vocab_equivalence_analysis::SharedVocabDfaBase::byte_to_class)
            .unwrap_or_else(|| super::compat::compute_byte_classes(analysis_view.dfa()));
        let byte_class_setup_ms = byte_class_started_at.elapsed().as_secs_f64() * 1000.0;

        let follows_normalize_started_at = Instant::now();
        let normalized_disallowed_follows =
            normalize_disallowed_follows(tokenizer_group_count(analysis_view), effective_disallowed);
        let effective_follows_normalize_ms = effective_follows_prepare_ms
            + follows_normalize_started_at.elapsed().as_secs_f64() * 1000.0;

        let mut query_view_states = preclass_view_states.clone();
        query_view_states.sort_unstable();
        query_view_states.dedup();
        if std::env::var_os("GLRMASK_PROFILE_L2P_FIRST_BYTE_VOCAB_FACTOR").is_some() {
            let mut token_counts = [0usize; 256];
            let mut one_byte_counts = [0usize; 256];
            for token in &dedup.representative_token_bytes {
                if let Some(&first) = token.first() {
                    token_counts[first as usize] += 1;
                    one_byte_counts[first as usize] += usize::from(token.len() == 1);
                }
            }
            let mut factored_pairs = 0usize;
            let mut active_buckets = 0usize;
            let mut rows = Vec::<(usize, usize, usize, usize)>::new();
            for byte in 0..256usize {
                let tokens = token_counts[byte];
                if tokens == 0 {
                    continue;
                }
                active_buckets += 1;
                let mut targets = query_view_states
                    .iter()
                    .map(|&state| analysis_view.dfa().trans(state, byte))
                    .collect::<Vec<_>>();
                targets.sort_unstable();
                targets.dedup();
                factored_pairs = factored_pairs.saturating_add(tokens.saturating_mul(targets.len()));
                rows.push((tokens, targets.len(), byte, one_byte_counts[byte]));
            }
            rows.sort_unstable_by(|left, right| {
                right
                    .0
                    .saturating_mul(right.1)
                    .cmp(&left.0.saturating_mul(left.1))
            });
            let full_pairs = dedup
                .representative_token_bytes
                .len()
                .saturating_mul(query_view_states.len());
            eprintln!(
                "[glrmask/profile][l2p_first_byte_vocab_factor] partition={} tokens={} states={} active_buckets={} full_pairs={} factored_pairs={} reduction_pct={:.2} top={:?}",
                partition_label,
                dedup.representative_token_bytes.len(),
                query_view_states.len(),
                active_buckets,
                full_pairs,
                factored_pairs,
                if full_pairs == 0 {
                    0.0
                } else {
                    100.0 * (1.0 - factored_pairs as f64 / full_pairs as f64)
                },
                rows.into_iter().take(16).collect::<Vec<_>>(),
            );
        }
        let p0_vocab_first = partition_label == "p0"
            && std::env::var("GLRMASK_P0_VOCAB_FIRST")
                .map(|value| {
                    let value = value.trim();
                    value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
                })
                .unwrap_or(true);
        // Computing vocabulary equivalence first is exact: tokens merged here
        // are indistinguishable from every query state, so one representative
        // per vocabulary class is sufficient for the subsequent state scan.
        // Keep the established large-state route, and also use vocab-first when
        // the direct state×token scan is large enough to amortize the vocab pass.
        let small_state_p2_vocab_first = partition_label == "p2"
            && dedup.representative_token_bytes.len() >= 16_000
            && query_view_states.len() <= 64;
        let vocab_first = vocab_only
            || p0_vocab_first
            || small_state_p2_vocab_first
            || prefer_vocab_first_equivalence(
                dedup.representative_token_bytes.len(),
                query_view_states.len(),
            );
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][epsilon_equivalence_order] partition={} dedup_tokens={} pre_states={} vocab_first={}",
                partition_label,
                dedup.representative_token_bytes.len(),
                query_view_states.len(),
                vocab_first,
            );
        }
        let mut staged_query_representatives = None::<Vec<usize>>;
        let mut staged_exact_state_equiv_ms = 0.0f64;
        let mut staged_exact_rep_confirmation_used = false;
        let staged_requested = vocab_first
            && !vocab_only
            && staged_vocab_state_refinement_enabled(partition_label);
        let (precomputed_vocab, state_tokens, vocab_equiv_ms) = if staged_requested {
            let vocab_equiv_started_at = Instant::now();
            let precomputed_vocab =
                vocab_equivalence_analysis::find_vocab_equivalence_classes_with_factor_state_refinement_profiled(
                    analysis_view,
                    &dedup.representative_token_bytes,
                    &query_view_states,
                    effective_disallowed,
                    Some(&byte_to_class),
                    None,
                    local_shared_cache,
                    None,
                    |preliminary_token_indices| {
                        let state_tokens = preliminary_token_indices
                            .iter()
                            .map(|&token| dedup.representative_token_bytes[token])
                            .collect::<Vec<_>>();
                        staged_exact_rep_confirmation_used = query_view_states.len()
                            >= EXACT_REP_CONFIRMATION_MIN_STATES
                            && state_tokens.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
                        let exact_started_at = Instant::now();
                        let query_representatives = if staged_exact_rep_confirmation_used {
                            state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
                                analysis_view,
                                &state_tokens,
                                &query_view_states,
                                &normalized_disallowed_follows,
                                None,
                                None,
                                Some(true),
                                local_shared_base,
                            )
                        } else {
                            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
                                analysis_view,
                                &state_tokens,
                                &query_view_states,
                                &normalized_disallowed_follows,
                                local_shared_base,
                            )
                        };
                        staged_exact_state_equiv_ms =
                            exact_started_at.elapsed().as_secs_f64() * 1000.0;
                        let mut first_position_for_representative =
                            BTreeMap::<usize, usize>::new();
                        for (position, &representative) in
                            query_representatives.iter().enumerate()
                        {
                            first_position_for_representative
                                .entry(representative)
                                .or_insert(position);
                        }
                        staged_query_representatives = Some(query_representatives);
                        first_position_for_representative.into_values().collect()
                    },
                );
            let total_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;
            let state_tokens = if staged_query_representatives.is_some() {
                Vec::new()
            } else {
                representative_tokens_for_vocab_classes(
                    &precomputed_vocab.0,
                    &dedup.representative_token_bytes,
                )
            };
            if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some()
                || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
                || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
            {
                eprintln!(
                    "[glrmask/profile][staged_vocab_state_refinement] partition={} selected={} input_states={} authority_states={} exact_classes={} exact_state_ms={:.3} vocab_ms={:.3}",
                    partition_label,
                    staged_query_representatives.is_some(),
                    query_view_states.len(),
                    staged_query_representatives.as_ref().map_or(query_view_states.len(), |mapping| {
                        mapping.iter().copied().collect::<BTreeSet<_>>().len()
                    }),
                    precomputed_vocab.0.len(),
                    staged_exact_state_equiv_ms,
                    (total_ms - staged_exact_state_equiv_ms).max(0.0),
                );
            }
            (
                Some(precomputed_vocab),
                state_tokens,
                (total_ms - staged_exact_state_equiv_ms).max(0.0),
            )
        } else if vocab_first {
            let vocab_equiv_started_at = Instant::now();
            let first_byte_factored = first_byte_vocab_factor_enabled(partition_label)
                .then(|| {
                    first_byte_factored_vocab_classes(
                        analysis_view,
                        &dedup.representative_token_bytes,
                        &query_view_states,
                        effective_disallowed,
                        &byte_to_class,
                    )
                })
                .flatten();
            let common_atom_preclasses = common_atom_preclass_enabled().then(|| {
                common_atom_preclass::try_find_common_atom_preclasses_with_vocab(
                    tokenizer,
                    active_groups,
                    &dedup.representative_token_bytes,
                    vocab,
                    &dedup.representative_original_indices,
                )
            });
            let precomputed_vocab = if let Some((classes, factor_profile)) = first_byte_factored {
                if first_byte_vocab_factor_strict_reference_enabled(partition_label) {
                    let strict_started_at = Instant::now();
                    let (strict_classes, _) =
                        vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                            analysis_view,
                            &dedup.representative_token_bytes,
                            &query_view_states,
                            effective_disallowed,
                            Some(&byte_to_class),
                            None,
                            local_shared_cache,
                            None,
                        );
                    assert_eq!(
                        classes, strict_classes,
                        "first-byte vocabulary factorization changed exact classes in {partition_label}",
                    );
                    eprintln!(
                        "[glrmask/profile][l2p_first_byte_vocab_factor_strict_reference] partition={} exact_classes={} compare_ms={:.3} differs=false",
                        partition_label,
                        strict_classes.len(),
                        strict_started_at.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some()
                    || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
                    || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
                {
                    eprintln!(
                        "[glrmask/profile][l2p_first_byte_vocab_factor_selected] partition={} tokens={} buckets={} factored_buckets={} min_bucket_tokens={} source_state_buckets_before={} source_state_buckets_after={} preliminary_classes={} final_classes={} bucket_ms={:.3} final_ms={:.3} dfa_build_ms={:.3} total_ms={:.3}",
                        partition_label,
                        dedup.representative_token_bytes.len(),
                        factor_profile.buckets,
                        factor_profile.factored_buckets,
                        factor_profile.min_bucket_tokens,
                        factor_profile.source_states_before,
                        factor_profile.source_states_after,
                        factor_profile.preliminary_classes,
                        classes.len(),
                        factor_profile.bucket_ms,
                        factor_profile.final_ms,
                        factor_profile.dfa_build_ms,
                        factor_profile.total_ms,
                    );
                }
                (classes, factor_profile.dfa_build_ms)
            } else if let Some(preclasses) = common_atom_preclasses.flatten() {
                let representative_tokens =
                    preclasses.representative_tokens(&dedup.representative_token_bytes);
                let exact_started_at = Instant::now();
                let (representative_classes, build_ms) =
                    vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                        analysis_view,
                        &representative_tokens,
                        &query_view_states,
                        effective_disallowed,
                        Some(&byte_to_class),
                        None,
                        local_shared_cache,
                        None,
                    );
                let exact_scan_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
                let classes = preclasses.expand_exact_classes(&representative_classes);
                if common_atom_preclass_strict_reference_enabled(partition_label) {
                    let strict_started_at = Instant::now();
                    let (strict_classes, _) =
                        vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                            analysis_view,
                            &dedup.representative_token_bytes,
                            &query_view_states,
                            effective_disallowed,
                            Some(&byte_to_class),
                            None,
                            local_shared_cache,
                            None,
                        );
                    assert_eq!(
                        classes, strict_classes,
                        "common-atom vocabulary preclasses changed exact classes in {partition_label}",
                    );
                    eprintln!(
                        "[glrmask/profile][l2p_vocab_preclass_strict_reference] partition={} full_exact_classes={} compare_ms={:.3} differs=false",
                        partition_label,
                        strict_classes.len(),
                        strict_started_at.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some()
                    || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
                    || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
                {
                    eprintln!(
                        "[glrmask/profile][l2p_vocab_preclass] partition={} kind=common_atom active_terminals={} atom_states={} tokens={} preclasses={} final_exact_classes={} build_ms={:.3} classify_ms={:.3} exact_scan_ms={:.3} exact_dfa_build_ms={:.3}",
                        partition_label,
                        preclasses.active_terminals,
                        preclasses.atom_states,
                        dedup.representative_token_bytes.len(),
                        preclasses.len(),
                        classes.len(),
                        preclasses.build_ms,
                        preclasses.classify_ms,
                        exact_scan_ms,
                        build_ms,
                    );
                }
                (classes, build_ms)
            } else {
                vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                    analysis_view,
                    &dedup.representative_token_bytes,
                    &query_view_states,
                    effective_disallowed,
                    Some(&byte_to_class),
                    None,
                    local_shared_cache,
                    None,
                )
            };
            let vocab_equiv_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;
            if vocab_only {
                let id_map_finalize_started_at = Instant::now();
                let vocab_classes = expand_vocab_classes(
                    precomputed_vocab.0,
                    &dedup.original_to_repr,
                    dedup.representative_token_bytes.len(),
                );
                let vocab_tokens = build_vocab_group_map(&vocab_classes, &prepared.token_ids);
                let exact_reps = tokenizer_states.num_internal_ids() as usize;
                let id_map_finalize_ms =
                    id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;
                return (
                    InternalIdMap {
                        tokenizer_states,
                        vocab_tokens,
                        deferred_vocab_singleton_original_ids: None,
                    },
                    CombinedEquivalenceProfile {
                        initial_states_considered: prepared.initial_states.len(),
                        max_length_skipped: pipeline_profile.max_length_skipped,
                        max_token_len,
                        token_len_gt_4: token_len_stats.gt_4,
                        token_len_gt_8: token_len_stats.gt_8,
                        token_len_gt_16: token_len_stats.gt_16,
                        token_len_gt_32: token_len_stats.gt_32,
                        token_len_gt_64: token_len_stats.gt_64,
                        raw_analysis_base_init_ms: 0.0,
                        analysis_view_build_ms,
                        active_mask_filter_ms,
                        effective_follows_normalize_ms,
                        prepare_inputs_ms,
                        byte_class_setup_ms,
                        vocab_analysis_dfa_build_ms: precomputed_vocab.1,
                        token_dedup_ms,
                        restricted_observation_state_equiv_ms: pipeline_profile
                            .restricted_observation_state_equiv_ms,
                        max_length_state_equiv_ms: pipeline_profile.max_length_state_equiv_ms,
                        vocab_equiv_ms,
                        exact_state_equiv_ms: 0.0,
                        id_map_finalize_ms,
                        restricted_observation_reps: pipeline_profile.restricted_observation_reps,
                        max_length_reps: pipeline_profile.max_length_reps,
                        exact_reps,
                        exact_rep_confirmation_used: false,
                    },
                );
            }
            let state_tokens = representative_tokens_for_vocab_classes(
                &precomputed_vocab.0,
                &dedup.representative_token_bytes,
            );
            (Some(precomputed_vocab), state_tokens, vocab_equiv_ms)
        } else {
            (None, dedup.representative_token_bytes.clone(), 0.0)
        };
        let (query_representatives, exact_state_equiv_ms, exact_rep_confirmation_used) =
            if let Some(query_representatives) = staged_query_representatives {
                (
                    query_representatives,
                    staged_exact_state_equiv_ms,
                    staged_exact_rep_confirmation_used,
                )
            } else {
                let exact_rep_confirmation_used = query_view_states.len()
                    >= EXACT_REP_CONFIRMATION_MIN_STATES
                    && state_tokens.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
                let exact_started_at = Instant::now();
                let query_representatives = if exact_rep_confirmation_used {
                    state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
                        analysis_view,
                        &state_tokens,
                        &query_view_states,
                        &normalized_disallowed_follows,
                        None,
                        None,
                        Some(true),
                        local_shared_base,
                    )
                } else {
                    state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
                        analysis_view,
                        &state_tokens,
                        &query_view_states,
                        &normalized_disallowed_follows,
                        local_shared_base,
                    )
                };
                (
                    query_representatives,
                    exact_started_at.elapsed().as_secs_f64() * 1000.0,
                    exact_rep_confirmation_used,
                )
            };

        let view_to_exact_view = query_view_states
            .iter()
            .copied()
            .zip(query_representatives.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let first_preclass_for_view = preclass_view_states
            .iter()
            .copied()
            .enumerate()
            .fold(BTreeMap::<usize, usize>::new(), |mut map, (preclass, view_state)| {
                map.entry(view_state).or_insert(preclass);
                map
            });
        let final_preclass_representatives = preclass_view_states
            .iter()
            .map(|view_state| {
                let exact_view = view_to_exact_view[view_state];
                first_preclass_for_view[&exact_view]
            })
            .collect::<Vec<_>>();
        let tokenizer_states =
            compose_raw_quotient_state_map(&tokenizer_states, &final_preclass_representatives);

        // Exact refinement above is expressed in bounded analysis-view
        // coordinates. The composed raw quotient may choose an unseeded raw
        // member as the stored representative, so projecting that representative
        // through the sparse raw-start map is invalid. Reuse the exact view
        // representatives directly for the subsequent vocabulary pass.
        let final_view_states =
            exact_analysis_view_representatives(&query_representatives);

        let (dedup_vocab_classes, vocab_analysis_dfa_build_ms, vocab_equiv_ms) =
            if let Some((classes, build_ms)) = precomputed_vocab {
                (classes, build_ms, vocab_equiv_ms)
            } else {
                let vocab_equiv_started_at = Instant::now();
                let (classes, build_ms) =
                    vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                        analysis_view,
                        &dedup.representative_token_bytes,
                        &final_view_states,
                        effective_disallowed,
                        Some(&byte_to_class),
                        None,
                        local_shared_cache,
                        None,
                    );
                (
                    classes,
                    build_ms,
                    vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0,
                )
            };

        if staged_requested
            && staged_vocab_state_refinement_strict_reference_enabled(partition_label)
        {
            let strict_started_at = Instant::now();
            let (strict_classes, _) =
                vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                    analysis_view,
                    &dedup.representative_token_bytes,
                    &query_view_states,
                    effective_disallowed,
                    Some(&byte_to_class),
                    None,
                    local_shared_cache,
                    None,
                );
            assert_eq!(
                dedup_vocab_classes, strict_classes,
                "staged vocabulary/state refinement changed exact classes in {partition_label}",
            );
            eprintln!(
                "[glrmask/profile][staged_vocab_state_refinement_strict_reference] partition={} exact_classes={} compare_ms={:.3} differs=false",
                partition_label,
                strict_classes.len(),
                strict_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        let id_map_finalize_started_at = Instant::now();
        let vocab_classes = expand_vocab_classes(
            dedup_vocab_classes,
            &dedup.original_to_repr,
            dedup.representative_token_bytes.len(),
        );
        let vocab_tokens = build_vocab_map(
            &vocab_classes,
            &prepared.token_ids,
            prepared.max_token_id,
        );
        let id_map = InternalIdMap {
            tokenizer_states,
            vocab_tokens,
            deferred_vocab_singleton_original_ids: None,
        };
        let id_map_finalize_ms = id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;
        let exact_reps = id_map.tokenizer_states.num_internal_ids() as usize;
        return (
            id_map,
            CombinedEquivalenceProfile {
                initial_states_considered: prepared.initial_states.len(),
                max_length_skipped: pipeline_profile.max_length_skipped,
                max_token_len,
                token_len_gt_4: token_len_stats.gt_4,
                token_len_gt_8: token_len_stats.gt_8,
                token_len_gt_16: token_len_stats.gt_16,
                token_len_gt_32: token_len_stats.gt_32,
                token_len_gt_64: token_len_stats.gt_64,
                raw_analysis_base_init_ms: 0.0,
                analysis_view_build_ms,
                active_mask_filter_ms,
                effective_follows_normalize_ms,
                prepare_inputs_ms,
                byte_class_setup_ms,
                vocab_analysis_dfa_build_ms,
                token_dedup_ms,
                restricted_observation_state_equiv_ms: pipeline_profile
                    .restricted_observation_state_equiv_ms,
                max_length_state_equiv_ms: pipeline_profile.max_length_state_equiv_ms,
                vocab_equiv_ms,
                exact_state_equiv_ms,
                id_map_finalize_ms,
                restricted_observation_reps: pipeline_profile.restricted_observation_reps,
                max_length_reps: pipeline_profile.max_length_reps,
                exact_reps,
                exact_rep_confirmation_used,
            },
        );
    }

    let follows_prepare_started_at = Instant::now();
    let token_path_disallowed_follows = (!disallowed_follows_are_ignore_transparent)
        .then(|| ignore_transparent_disallowed_follows(disallowed_follows, ignore_terminal));
    let effective_follows_prepare_ms = follows_prepare_started_at.elapsed().as_secs_f64() * 1000.0;
    let effective_disallowed = token_path_disallowed_follows
        .as_ref()
        .unwrap_or(disallowed_follows);
    if let Some(token_position_partition) = token_position_partition {
        if let Some(result) = try_analyze_equivalences_with_token_position_partition(
            partition_label,
            tokenizer,
            vocab,
            effective_disallowed,
            active_groups,
            shared_vocab_dfa_cache,
            shared_analysis_dfa_cache,
            flat_trans,
            token_position_partition,
            effective_follows_prepare_ms,
            pre_normalized_disallowed_follows,
        ) {
            return result;
        }
    }
    if let Some(result) = try_analyze_equivalences_with_raw_quotient(
        partition_label,
        tokenizer,
        vocab,
        effective_disallowed,
        ignore_terminal,
        active_groups,
        flat_trans,
        initial_state_map,
        effective_follows_prepare_ms,
        precomputed_raw_observations,
    ) {
        return result;
    }
    // The raw tokenizer is the only lexer coordinate in L2P. Retain the
    // compatibility check defensively, since this cache is shared by callers.
    let compatible_flat_trans = flat_trans.filter(|ft| {
        ft.len() == tokenizer.num_states() as usize * 256
    });
    let analysis_view_build_started_at = Instant::now();
    let tokenizer_view = match (active_groups, compatible_flat_trans) {
        (Some(active_groups), Some(ft)) => TokenizerView::new_filtered_from_flat_trans(ft, tokenizer, active_groups),
        (Some(active_groups), None) => TokenizerView::new_filtered(tokenizer, active_groups),
        (None, Some(ft)) => TokenizerView::new_from_flat_trans(ft, tokenizer),
        _ => TokenizerView::new(tokenizer),
    };
    let analysis_view_build_ms = analysis_view_build_started_at.elapsed().as_secs_f64() * 1000.0;

    let prepare_inputs_started_at = Instant::now();
    let prepared = prepare_equivalence_inputs(tokenizer, vocab, initial_state_map);
    let token_len_stats = token_length_stats(&prepared.token_bytes);
    let prepare_inputs_ms = prepare_inputs_started_at.elapsed().as_secs_f64() * 1000.0;

    let raw_analysis_base_started_at = Instant::now();
    if let Some(cache) = shared_vocab_dfa_cache {
        cache.get_or_init(|| vocab_equivalence_analysis::SharedVocabDfaBase::build_from_dfa(tokenizer_view.dfa()));
    }
    let raw_analysis_base_init_ms = shared_base_setup_ms
        + raw_analysis_base_started_at.elapsed().as_secs_f64() * 1000.0;

    let byte_class_setup_started_at = Instant::now();
    let compatible_cache = shared_vocab_dfa_cache
        .and_then(|cache| cache.get())
        .filter(|base| base.is_compatible_with_dfa(tokenizer_view.dfa()));
    let byte_to_class = compatible_cache
        .map(|base| base.byte_to_class())
        .unwrap_or_else(|| super::compat::compute_byte_classes(tokenizer_view.dfa()));
    let byte_class_setup_ms = byte_class_setup_started_at.elapsed().as_secs_f64() * 1000.0;

    let token_dedup_started_at = Instant::now();
    let dedup = deduplicate_tokens_by_byte_class(&prepared.token_bytes, &byte_to_class, prepared.lexical_entry_indices());
    let token_dedup_ms = token_dedup_started_at.elapsed().as_secs_f64() * 1000.0;
    let max_token_len = dedup
        .representative_token_bytes
        .iter()
        .map(|token| token.len())
        .max()
        .unwrap_or(0);
    let mut relevant_bytes = [false; 256];
    for token in &dedup.representative_token_bytes {
        for &byte in *token {
            relevant_bytes[byte as usize] = true;
        }
    }
    let direct_token_bytes: usize = dedup
        .representative_token_bytes
        .iter()
        .map(|token| token.len())
        .sum();
    let active_byte_count = relevant_bytes.iter().filter(|&&active| active).count();
    let projected_by_global = prepared.initial_states.len() < tokenizer.num_states() as usize;
    let max_length_skipped = should_skip_max_length_for_partition(
        partition_label,
        prepared.initial_states.len(),
        projected_by_global,
        direct_token_bytes,
        max_token_len,
        active_byte_count,
    );
    let pipeline_config = resolve_l2p_pipeline_config(!max_length_skipped);
    let (pre_state_map, pipeline_profile) = run_state_equivalence_pipeline(
        tokenizer,
        vocab,
        initial_state_map,
        active_groups,
        StateEquivalenceScope::L2p,
        &pipeline_config,
        None,
        Some(&tokenizer_view),
        Some(&byte_to_class),
    );
    // Restricted observation is a fixed point over the vocabulary byte
    // alphabet. When its map is also output-labelled congruent, token paths can
    // be evaluated exactly on the quotient instead of reinitializing 18k raw
    // lexer states for the exact and vocabulary phases.
    let quotient_cert_started_at = Instant::now();
    let quotient_certified = pipeline_profile.relevant_byte_congruence_certified
        || tokenizer_view.is_relevant_byte_congruent(&pre_state_map, &relevant_bytes);
    let quotient_cert_ms = quotient_cert_started_at.elapsed().as_secs_f64() * 1000.0;
    let quotient_materialize_started_at = Instant::now();
    let analysis_quotient = quotient_certified
        .then(|| tokenizer_view.quotient_by_state_map(&pre_state_map));
    let quotient_materialize_ms =
        quotient_materialize_started_at.elapsed().as_secs_f64() * 1000.0;
    if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][analysis_quotient] partition={} certified_by_construction={} certified={} source_states={} quotient_states={} certify_ms={:.3} materialize_ms={:.3}",
            partition_label,
            pipeline_profile.relevant_byte_congruence_certified,
            quotient_certified,
            tokenizer_view.dfa().states.len(),
            pre_state_map.num_internal_ids(),
            quotient_cert_ms,
            quotient_materialize_ms,
        );
    }
    let uses_analysis_quotient = analysis_quotient.is_some();
    let analysis_view = analysis_quotient.as_ref().unwrap_or(&tokenizer_view);
    let pre_reduced_states: Vec<usize> = if uses_analysis_quotient {
        (0..pre_state_map.internal_to_originals.len()).collect()
    } else {
        pre_state_map
            .representative_original_ids
            .iter()
            .map(|&state| state as usize)
            .collect()
    };
    let follows_normalize_started_at = Instant::now();
    let normalized_disallowed_follows =
        normalize_disallowed_follows(tokenizer_group_count(analysis_view), effective_disallowed);
    let effective_follows_normalize_ms = effective_follows_prepare_ms
        + follows_normalize_started_at.elapsed().as_secs_f64() * 1000.0;

    // State and vocabulary equivalence are commuting exact quotients.  For a
    // large vocabulary, classify tokens on the pre-state quotient first, then
    // refine states using one representative token per exact token class.  The
    // final state partition and final token partition are identical to the
    // historical state-then-vocab order (see the commutativity regression
    // below), but the expensive state trellis sees hundreds of tokens instead
    // of tens of thousands.
    // A vocabulary-only caller never consumes the final tokenizer-state
    // quotient.  State and vocabulary equivalence commute, and the vocabulary
    // relation computed against the complete pre-state quotient is already
    // exact before the final state refinement.  Prefer that order regardless
    // of problem size instead of paying to compute a state quotient that is
    // immediately discarded.  Full L2P builds retain the historical
    // size-based selector because they do consume both quotients.
    let vocab_first = vocab_only
        || (dedup.representative_token_bytes.len() >= 8_192
            && pre_reduced_states.len() >= 256);
    if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][combined_equivalence_order] partition={} dedup_tokens={} pre_states={} vocab_first={}",
            partition_label,
            dedup.representative_token_bytes.len(),
            pre_reduced_states.len(),
            vocab_first,
        );
    }
    let (
        reduced_state_reps_for_pre_reduced,
        dedup_vocab_classes,
        vocab_analysis_dfa_build_ms,
        exact_state_equiv_ms,
        vocab_equiv_ms,
        exact_rep_confirmation_used,
    ) = if vocab_first {
        let vocab_equiv_started_at = Instant::now();
        let (dedup_vocab_classes, vocab_analysis_dfa_build_ms) =
            vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                analysis_view,
                &dedup.representative_token_bytes,
                &pre_reduced_states,
                effective_disallowed,
                Some(&byte_to_class),
                if uses_analysis_quotient { None } else { active_groups },
                if uses_analysis_quotient { None } else { shared_vocab_dfa_cache },
                if uses_analysis_quotient { None } else { shared_analysis_dfa_cache },
            );
        let vocab_equiv_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;
        if vocab_only {
            let id_map_finalize_started_at = Instant::now();
            let vocab_classes = expand_vocab_classes(
                dedup_vocab_classes,
                &dedup.original_to_repr,
                dedup.representative_token_bytes.len(),
            );
            let vocab_tokens = build_vocab_group_map(&vocab_classes, &prepared.token_ids);
            let tokenizer_states = pre_state_map.clone();
            let exact_reps = tokenizer_states.num_internal_ids() as usize;
            let internal_id_map = InternalIdMap {
                tokenizer_states,
                vocab_tokens,
                deferred_vocab_singleton_original_ids: None,
            };
            let id_map_finalize_ms =
                id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;
            return (
                internal_id_map,
                CombinedEquivalenceProfile {
                    initial_states_considered: prepared.initial_states.len(),
                    max_length_skipped: pipeline_profile.max_length_skipped,
                    max_token_len,
                    token_len_gt_4: token_len_stats.gt_4,
                    token_len_gt_8: token_len_stats.gt_8,
                    token_len_gt_16: token_len_stats.gt_16,
                    token_len_gt_32: token_len_stats.gt_32,
                    token_len_gt_64: token_len_stats.gt_64,
                    raw_analysis_base_init_ms,
                    analysis_view_build_ms,
                    active_mask_filter_ms: 0.0,
                    effective_follows_normalize_ms,
                    prepare_inputs_ms,
                    byte_class_setup_ms,
                    vocab_analysis_dfa_build_ms,
                    token_dedup_ms,
                    restricted_observation_state_equiv_ms: pipeline_profile
                        .restricted_observation_state_equiv_ms,
                    max_length_state_equiv_ms: pipeline_profile.max_length_state_equiv_ms,
                    vocab_equiv_ms,
                    exact_state_equiv_ms: 0.0,
                    id_map_finalize_ms,
                    restricted_observation_reps: pipeline_profile.restricted_observation_reps,
                    max_length_reps: pipeline_profile.max_length_reps,
                    exact_reps,
                    exact_rep_confirmation_used: false,
                },
            );
        }
        let representative_tokens = representative_tokens_for_vocab_classes(
            &dedup_vocab_classes,
            &dedup.representative_token_bytes,
        );
        let exact_rep_confirmation_used = pre_reduced_states.len()
            >= EXACT_REP_CONFIRMATION_MIN_STATES
            && representative_tokens.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
        let exact_started_at = Instant::now();
        let reduced_state_reps_for_pre_reduced = if exact_rep_confirmation_used {
            state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
                analysis_view,
                &representative_tokens,
                &pre_reduced_states,
                &normalized_disallowed_follows,
                None,
                None,
                Some(true),
                (!uses_analysis_quotient).then_some(compatible_cache).flatten(),
            )
        } else {
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
                analysis_view,
                &representative_tokens,
                &pre_reduced_states,
                &normalized_disallowed_follows,
                (!uses_analysis_quotient).then_some(compatible_cache).flatten(),
            )
        };
        let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
        (
            reduced_state_reps_for_pre_reduced,
            dedup_vocab_classes,
            vocab_analysis_dfa_build_ms,
            exact_state_equiv_ms,
            vocab_equiv_ms,
            exact_rep_confirmation_used,
        )
    } else {
        let exact_rep_confirmation_used = pre_reduced_states.len()
            >= EXACT_REP_CONFIRMATION_MIN_STATES
            && dedup.representative_token_bytes.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
        let exact_started_at = Instant::now();
        let reduced_state_reps_for_pre_reduced = if exact_rep_confirmation_used {
            state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
                analysis_view,
                &dedup.representative_token_bytes,
                &pre_reduced_states,
                &normalized_disallowed_follows,
                None,
                None,
                Some(true),
                (!uses_analysis_quotient).then_some(compatible_cache).flatten(),
            )
        } else {
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
                analysis_view,
                &dedup.representative_token_bytes,
                &pre_reduced_states,
                &normalized_disallowed_follows,
                (!uses_analysis_quotient).then_some(compatible_cache).flatten(),
            )
        };
        let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
        let mut final_state_representatives = reduced_state_reps_for_pre_reduced.clone();
        final_state_representatives.sort_unstable();
        final_state_representatives.dedup();
        let vocab_equiv_started_at = Instant::now();
        let (dedup_vocab_classes, vocab_analysis_dfa_build_ms) =
            vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                analysis_view,
                &dedup.representative_token_bytes,
                &final_state_representatives,
                effective_disallowed,
                Some(&byte_to_class),
                if uses_analysis_quotient { None } else { active_groups },
                if uses_analysis_quotient { None } else { shared_vocab_dfa_cache },
                if uses_analysis_quotient { None } else { shared_analysis_dfa_cache },
            );
        let vocab_equiv_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;
        (
            reduced_state_reps_for_pre_reduced,
            dedup_vocab_classes,
            vocab_analysis_dfa_build_ms,
            exact_state_equiv_ms,
            vocab_equiv_ms,
            exact_rep_confirmation_used,
        )
    };

    let representative_states = if uses_analysis_quotient {
        prepared
            .initial_states
            .iter()
            .map(|&state| {
                let pre_internal = pre_state_map.original_to_internal[state] as usize;
                let final_internal = reduced_state_reps_for_pre_reduced[pre_internal];
                pre_state_map.representative_original_ids[final_internal] as usize
            })
            .collect::<Vec<_>>()
    } else {
        let rep_to_final: BTreeMap<usize, usize> = pre_reduced_states
            .iter()
            .copied()
            .zip(reduced_state_reps_for_pre_reduced.iter().copied())
            .collect();
        prepared
            .initial_states
            .iter()
            .map(|&state| {
                let pre_internal = pre_state_map.original_to_internal[state];
                let pre_rep = pre_state_map.representative_original_ids[pre_internal as usize] as usize;
                rep_to_final[&pre_rep]
            })
            .collect::<Vec<_>>()
    };
    let id_map_finalize_started_at = Instant::now();
    let vocab_classes = expand_vocab_classes(
        dedup_vocab_classes,
        &dedup.original_to_repr,
        dedup.representative_token_bytes.len(),
    );
    let state_classes =
        state_equivalence_analysis::mapping_to_equivalence_classes(&prepared.initial_states, &representative_states);
    let exact_reps = state_classes.len();
    let result = CombinedEquivalenceResult {
        vocab_classes,
        state_classes,
    };
    let internal_id_map =
        build_internal_id_map_from_combined_result(tokenizer, initial_state_map, &prepared, &result);
    let id_map_finalize_ms = id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;

    (
        internal_id_map,
        CombinedEquivalenceProfile {
            initial_states_considered: prepared.initial_states.len(),
            max_length_skipped: pipeline_profile.max_length_skipped,
            max_token_len,
            token_len_gt_4: token_len_stats.gt_4,
            token_len_gt_8: token_len_stats.gt_8,
            token_len_gt_16: token_len_stats.gt_16,
            token_len_gt_32: token_len_stats.gt_32,
            token_len_gt_64: token_len_stats.gt_64,
            raw_analysis_base_init_ms,
            analysis_view_build_ms,
            active_mask_filter_ms: 0.0,
            effective_follows_normalize_ms,
            prepare_inputs_ms,
            byte_class_setup_ms,
            vocab_analysis_dfa_build_ms,
            token_dedup_ms,
            restricted_observation_state_equiv_ms: pipeline_profile.restricted_observation_state_equiv_ms,
            max_length_state_equiv_ms: pipeline_profile.max_length_state_equiv_ms,
            vocab_equiv_ms,
            exact_state_equiv_ms,
            id_map_finalize_ms,
            restricted_observation_reps: pipeline_profile.restricted_observation_reps,
            max_length_reps: pipeline_profile.max_length_reps,
            exact_reps,
            exact_rep_confirmation_used,
        },
    )
}

#[cfg(test)]
mod prepass_selection_tests {
    use super::*;
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::compat::{
        FlatDfa, FlatDfaState,
    };
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::shared::representative_tokens_for_vocab_classes;
    use std::sync::Arc;

    #[test]
    fn vocab_first_selection_uses_large_state_or_large_pair_work() {
        assert!(prefer_vocab_first_equivalence(512, 256));
        assert!(prefer_vocab_first_equivalence(56_610, 40));
        assert!(prefer_vocab_first_equivalence(47_171, 61));
        assert!(prefer_vocab_first_equivalence(29_801, 31));

        assert!(!prefer_vocab_first_equivalence(4_657, 182));
        assert!(!prefer_vocab_first_equivalence(10_142, 40));
        assert!(!prefer_vocab_first_equivalence(511, usize::MAX));
    }

    #[test]
    fn adaptive_nfa_view_skips_powerset_probe_below_bounded_work_threshold() {
        assert!(!should_probe_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Adaptive,
            499_999,
            500_000,
        ));
        assert!(should_probe_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Adaptive,
            500_000,
            500_000,
        ));
    }

    #[test]
    fn adaptive_nfa_view_uses_only_small_probed_powersets() {
        assert!(should_use_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Adaptive,
            true,
            8_192,
            8_192,
        ));
        assert!(!should_use_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Adaptive,
            true,
            8_193,
            8_192,
        ));
        assert!(!should_use_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Adaptive,
            false,
            0,
            8_192,
        ));
    }

    #[test]
    fn forced_nfa_view_policies_override_adaptive_gates() {
        assert!(!should_probe_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Bounded,
            usize::MAX,
            500_000,
        ));
        assert!(!should_use_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Bounded,
            true,
            1,
            8_192,
        ));
        assert!(should_probe_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Powerset,
            0,
            500_000,
        ));
        assert!(should_use_l2p_nfa_powerset(
            L2pNfaAnalysisViewPolicy::Powerset,
            true,
            usize::MAX,
            8_192,
        ));
    }

    fn partition_from_representatives<T: Ord + Copy>(
        values: &[T],
        representatives: &[T],
    ) -> BTreeSet<BTreeSet<T>> {
        let mut by_representative = BTreeMap::<T, BTreeSet<T>>::new();
        for (&value, &representative) in values.iter().zip(representatives) {
            by_representative.entry(representative).or_default().insert(value);
        }
        by_representative.into_values().collect()
    }

    fn synthetic_view() -> TokenizerView {
        let state_count = 5usize;
        let mut transitions = vec![u32::MAX; state_count * 256];
        let set = |transitions: &mut [u32], state: usize, byte: u8, target: u32| {
            transitions[state * 256 + byte as usize] = target;
        };
        set(&mut transitions, 0, b'a', 1);
        set(&mut transitions, 0, b'b', 2);
        set(&mut transitions, 1, b'a', 1);
        set(&mut transitions, 1, b'b', 3);
        set(&mut transitions, 2, b'a', 3);
        set(&mut transitions, 2, b'b', 2);
        set(&mut transitions, 3, b'a', 3);
        set(&mut transitions, 3, b'b', 3);
        // State 4 is behaviorally identical to state 1.
        set(&mut transitions, 4, b'a', 4);
        set(&mut transitions, 4, b'b', 3);

        TokenizerView {
            flat_dfa: FlatDfa {
                start_state: 0,
                transitions: Arc::from(transitions),
                states: vec![
                    FlatDfaState {
                        finalizers: vec![],
                        possible_future_group_ids: vec![0, 1],
                    },
                    FlatDfaState {
                        finalizers: vec![0],
                        possible_future_group_ids: vec![0, 1],
                    },
                    FlatDfaState {
                        finalizers: vec![1],
                        possible_future_group_ids: vec![0, 1],
                    },
                    FlatDfaState {
                        finalizers: vec![0, 1],
                        possible_future_group_ids: vec![0, 1],
                    },
                    FlatDfaState {
                        finalizers: vec![0],
                        possible_future_group_ids: vec![0, 1],
                    },
                ],
            },
        }
    }

    #[test]
    fn state_then_vocab_equivalence_matches_vocab_then_state() {
        let view = synthetic_view();
        let tokens: Vec<&[u8]> = vec![
            b"a", b"b", b"aa", b"ab", b"ba", b"bb", b"x", b"y",
        ];
        let states: Vec<usize> = (0..view.dfa().states.len()).collect();
        let byte_to_class = super::super::compat::compute_byte_classes(view.dfa());
        let disallowed = BTreeMap::<u32, BitSet>::new();
        let normalized = normalize_disallowed_follows(2, &disallowed);

        let old_vocab = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &states,
            &disallowed,
            Some(&byte_to_class),
            None,
            None,
            None,
        );
        let old_token_reps = representative_tokens_for_vocab_classes(&old_vocab, &tokens);
        let old_state_reps =
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed(
                &view,
                &old_token_reps,
                &states,
                &normalized,
            );

        let reversed_state_reps =
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed(
                &view,
                &tokens,
                &states,
                &normalized,
            );
        let final_state_reps: Vec<usize> = {
            let mut reps = reversed_state_reps.clone();
            reps.sort_unstable();
            reps.dedup();
            reps
        };
        let reversed_vocab = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &final_state_reps,
            &disallowed,
            Some(&byte_to_class),
            None,
            None,
            None,
        );

        assert_eq!(
            partition_from_representatives(&states, &old_state_reps),
            partition_from_representatives(&states, &reversed_state_reps),
        );
        assert_eq!(old_vocab, reversed_vocab);
    }

    #[test]
    fn first_byte_vocab_factorization_matches_full_exact_partition() {
        let view = synthetic_view();
        let tokens: Vec<&[u8]> = vec![
            b"a", b"b", b"ax", b"ay", b"bx", b"by", b"x", b"y",
        ];
        let states: Vec<usize> = (0..view.dfa().states.len()).collect();
        let byte_to_class = super::super::compat::compute_byte_classes(view.dfa());
        let mut disallowed = BTreeMap::<u32, BitSet>::new();
        let mut blocked = BitSet::new(2);
        blocked.set(1);
        disallowed.insert(0, blocked);

        let full = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &states,
            &disallowed,
            Some(&byte_to_class),
            None,
            None,
            None,
        );
        let (factored, profile) = first_byte_factored_vocab_classes(
            &view,
            &tokens,
            &states,
            &disallowed,
            &byte_to_class,
        )
        .expect("test vocabulary should admit a nontrivial first-byte prepartition");

        assert!(profile.preliminary_classes < tokens.len());
        assert_eq!(factored, full);
    }

    #[test]
    fn one_byte_common_prefix_factorization_matches_full_token_state_equivalence() {
        let view = synthetic_view();
        let tokens: Vec<&[u8]> = vec![b"aa", b"ab", b"aaa", b"abb"];
        let suffix_tokens: Vec<&[u8]> = tokens.iter().map(|token| &token[1..]).collect();
        let states: Vec<usize> = (0..view.dfa().states.len()).collect();
        let normalized = normalize_disallowed_follows(2, &BTreeMap::new());

        let full = state_equivalence_analysis::find_state_equivalence_classes_with_disallowed(
            &view,
            &tokens,
            &states,
            &normalized,
        );

        let mut prefix_targets = BTreeSet::<usize>::new();
        let targets = states
            .iter()
            .map(|&source| view.dfa().trans(source, b'a' as usize))
            .collect::<Vec<_>>();
        for &target in &targets {
            if target != u32::MAX {
                prefix_targets.insert(target as usize);
            }
        }
        let prefix_targets = prefix_targets.into_iter().collect::<Vec<_>>();
        let target_representatives = state_equivalence_analysis::
            find_state_equivalence_classes_with_disallowed_and_shared_base_with_initial_finalizers(
                &view,
                &suffix_tokens,
                &prefix_targets,
                &normalized,
                None,
            );
        let behavior_for_target = prefix_targets
            .iter()
            .copied()
            .zip(target_representatives)
            .collect::<BTreeMap<_, _>>();
        let behaviors = targets
            .iter()
            .map(|&target| {
                (target != u32::MAX).then(|| behavior_for_target[&(target as usize)])
            })
            .collect::<Vec<_>>();
        let mut representative_for_behavior = BTreeMap::<Option<usize>, usize>::new();
        for (&source, &behavior) in states.iter().zip(&behaviors) {
            representative_for_behavior.entry(behavior).or_insert(source);
        }
        let factored = behaviors
            .into_iter()
            .map(|behavior| representative_for_behavior[&behavior])
            .collect::<Vec<_>>();

        assert_eq!(
            partition_from_representatives(&states, &full),
            partition_from_representatives(&states, &factored),
        );
    }

    #[test]
    fn directional_quotient_composition_retains_preclass_representative() {
        let directional = ManyToOneIdMap {
            original_to_internal: vec![0, 1, 0, 1],
            internal_to_originals: vec![vec![0, 2], vec![1, 3]],
            representative_original_ids: vec![2, 3],
        };

        let composed = compose_raw_quotient_state_map_preserving_directional_representatives(
            &directional,
            &[1, 1],
        );

        assert_eq!(composed.original_to_internal, vec![0, 0, 0, 0]);
        assert_eq!(composed.internal_to_originals, vec![vec![0, 1, 2, 3]]);
        assert_eq!(composed.representative_original_ids, vec![3]);
    }

    #[test]
    fn bounded_vocab_pass_keeps_exact_analysis_view_representatives() {
        let preclasses = ManyToOneIdMap {
            original_to_internal: vec![0, 1, 0, 1],
            internal_to_originals: vec![vec![0, 2], vec![1, 3]],
            representative_original_ids: vec![2, 3],
        };
        let composed = compose_raw_quotient_state_map(&preclasses, &[1, 1]);
        let sparse_raw_start_to_view = [u32::MAX, u32::MAX, 10, 20];

        assert_eq!(composed.representative_original_ids, vec![0]);
        assert_eq!(
            sparse_raw_start_to_view[composed.representative_original_ids[0] as usize],
            u32::MAX,
        );
        assert_eq!(exact_analysis_view_representatives(&[20, 20]), vec![20]);
    }

    #[test]
    fn lexical_dedup_reordering_preserves_representatives_and_remaps_inputs() {
        let tokens: Vec<&[u8]> = vec![b"z", b"ab", b"aa", b"ac"];
        let mut byte_to_class: [u8; 256] = std::array::from_fn(|byte| byte as u8);
        byte_to_class[b'c' as usize] = b'b';

        let ordinary = deduplicate_tokens_by_byte_class(&tokens, &byte_to_class, None);
        assert_eq!(ordinary.representative_original_indices, vec![0, 1, 2]);
        assert_eq!(ordinary.original_to_repr, vec![0, 1, 2, 1]);

        // Raw lexical order is aa, ab, ac, z. `ac` is projected-equivalent to
        // `ab`, so only the already-chosen representative for that class moves.
        let lexical = deduplicate_tokens_by_byte_class(
            &tokens,
            &byte_to_class,
            Some(&[2, 1, 3, 0]),
        );
        assert_eq!(lexical.representative_token_bytes, vec![b"aa".as_slice(), b"ab".as_slice(), b"z".as_slice()]);
        assert_eq!(lexical.representative_original_indices, vec![2, 1, 0]);
        assert_eq!(lexical.original_to_repr, vec![2, 1, 0, 1]);
        for left in 0..tokens.len() {
            for right in 0..tokens.len() {
                assert_eq!(
                    ordinary.original_to_repr[left] == ordinary.original_to_repr[right],
                    lexical.original_to_repr[left] == lexical.original_to_repr[right],
                    "dedup relation changed for ({left}, {right})",
                );
            }
        }
    }
    #[test]
    fn selects_direct_refinement_when_byte_bounded_prepass_cannot_amortize() {
        // Small vocabulary with many relevant bytes: direct token walks are
        // cheaper than a full k-bounded byte refinement per lexer state.
        assert!(direct_refinement_work_is_no_larger(100, 10, 41));
        // Larger vocabulary over a smaller byte alphabet amortizes the exact
        // prepass and should retain it.
        assert!(!direct_refinement_work_is_no_larger(900, 14, 19));
    }
}
