//! Vocab and terminal classification utilities.

use crate::automata::lexer::compile::{build_regex, build_regex_local_small_product};
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::Lexer;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::Vocab;

use super::l2p::equivalence_analysis::compat::FlatDfa;
use super::l2p::equivalence_analysis::state_equivalence::nfa::{
    TokenBoundedAnalysisTrie, build_bounded_analysis_view_from_combined_starts_with_trie,
    build_token_bounded_analysis_view_from_combined_starts,
};
use super::types::TerminalPathLength;

/// DFA-derived byte sets for terminal classification, identical across partitions.
///
/// `classify_terminal_path_lengths` scans the full DFA to compute per-terminal
/// byte sets. Since all partitions share the same tokenizer and terminal count,
/// this scan is redundant after the first call. Caching these byte sets via
/// `OnceLock` eliminates ~35ms of repeated DFA scanning per extra partition.
pub struct SharedClassifyBytesets {
    reachable_bytes: Vec<U8Set>,
    first_bytes: Vec<U8Set>,
    last_bytes: Vec<U8Set>,
    transitions_by_byte: Vec<u32>,
    sparse_transitions_by_byte: Vec<Vec<(u32, u32)>>,
    reverse_transitions_by_byte: Vec<ReverseByteTransitions>,
    matched_terminals_by_state: Arc<[Box<[u32]>]>,
    future_terminals_by_state: Arc<[Box<[u32]>]>,
    matched_states_by_terminal: Arc<[Vec<u32>]>,
    future_states_by_terminal: Arc<[Vec<u32>]>,
    has_matched_terminal_by_state: Vec<u8>,
    future_by_state_words: Vec<u64>,
    representative_future_terminal_by_state: Vec<u32>,
    words_per_terminal_set: usize,
    active_route_setup_cache: Mutex<HashMap<(BitSet, usize), Arc<ActiveL2pRouteSetup>>>,
}

impl SharedClassifyBytesets {
    pub fn ti_output_index(
        &self,
    ) -> Option<(Arc<[Box<[u32]>]>, Arc<[Box<[u32]>]>, Arc<[Vec<u32>]>, Arc<[Vec<u32>]>)> {
        let state_count = self.future_by_state_words.len() / self.words_per_terminal_set.max(1);
        (self.matched_terminals_by_state.len() == state_count
            && self.future_terminals_by_state.len() == state_count
            && self.future_states_by_terminal.len() == self.matched_states_by_terminal.len())
            .then(|| {
                (
                    Arc::clone(&self.matched_terminals_by_state),
                    Arc::clone(&self.future_terminals_by_state),
                    Arc::clone(&self.matched_states_by_terminal),
                    Arc::clone(&self.future_states_by_terminal),
                )
            })
    }
}

struct ActiveL2pRouteSetup {
    // The cache uses Arc identity in its key. Keep that object alive for as
    // long as the entry exists; otherwise a later query can recycle its
    // address with a different follow relation and hit stale cached analysis.
    _follow_owner: Arc<BTreeMap<u32, BitSet>>,
    active_start_states: Arc<[u32]>,
    allowed_boundary_pairs: Box<[U8Set; 256]>,
    allowed_boundary_pair_words: Box<[u64; 1024]>,
    active_reachable_by_byte: Box<[u8; 256]>,
    active_suffix_start_by_byte: Box<[u8; 256]>,
}

#[derive(Default)]
struct ReverseByteTransitions {
    targets: Vec<u32>,
    source_offsets: Vec<u32>,
    sources: Vec<u32>,
}

fn build_reverse_transitions_by_byte(
    sparse_transitions_by_byte: &[Vec<(u32, u32)>],
    num_states: usize,
) -> Vec<ReverseByteTransitions> {
    let mut target_seen = vec![0u32; num_states];
    let mut target_index = vec![0u32; num_states];
    let mut stamp = 0u32;

    sparse_transitions_by_byte
        .iter()
        .map(|transitions| {
            if transitions.is_empty() {
                return ReverseByteTransitions::default();
            }
            stamp = stamp.wrapping_add(1);
            if stamp == 0 {
                target_seen.fill(0);
                stamp = 1;
            }

            let mut targets = Vec::new();
            let mut counts = Vec::<u32>::new();
            for &(_, target) in transitions {
                let target = target as usize;
                if target_seen[target] != stamp {
                    target_seen[target] = stamp;
                    target_index[target] = targets.len() as u32;
                    targets.push(target as u32);
                    counts.push(0);
                }
                counts[target_index[target] as usize] += 1;
            }

            let mut source_offsets = Vec::with_capacity(targets.len() + 1);
            source_offsets.push(0);
            for &count in &counts {
                source_offsets.push(source_offsets.last().copied().unwrap() + count);
            }
            let mut next_source_offsets = source_offsets[..targets.len()].to_vec();
            let mut sources = vec![0u32; transitions.len()];
            for &(source, target) in transitions {
                let group = target_index[target as usize] as usize;
                let offset = &mut next_source_offsets[group];
                sources[*offset as usize] = source;
                *offset += 1;
            }

            ReverseByteTransitions {
                targets,
                source_offsets,
                sources,
            }
        })
        .collect()
}

/// Cache type for lazy `SharedClassifyBytesets` initialization across partitions.
pub type SharedClassifyCache = std::sync::OnceLock<SharedClassifyBytesets>;

pub fn prewarm_shared_classify_cache(
    tokenizer: &Tokenizer,
    num_terminals: u32,
    cache: &SharedClassifyCache,
) {
    cache.get_or_init(|| SharedClassifyBytesets::build(tokenizer, num_terminals));
}

pub struct L2pVocabBoundarySplit {
    boundary_token_ids: Vec<u32>,
    single_token_ids: Vec<u32>,
    pub adjacent_tokens: usize,
    pub boundary_tokens: usize,
    pub single_tokens: usize,
    pub irrelevant_tokens: usize,
}

impl L2pVocabBoundarySplit {
    #[inline]
    pub fn boundary_token_ids(&self) -> &[u32] {
        &self.boundary_token_ids
    }

    fn materialize_vocab(vocab: &Vocab, token_ids: &[u32]) -> Vocab {
        let mut entries = Vec::with_capacity(token_ids.len());
        let mut parent_entry_indices = Vec::with_capacity(token_ids.len());
        let mut token_ids = token_ids.iter().copied().peekable();
        for (parent_entry_index, (&token_id, bytes)) in vocab.entries_map().iter().enumerate() {
            while token_ids.peek().is_some_and(|candidate| *candidate < token_id) {
                token_ids.next();
            }
            if token_ids.peek().is_some_and(|candidate| *candidate == token_id) {
                entries.push((token_id, bytes.clone()));
                parent_entry_indices.push(parent_entry_index);
                token_ids.next();
            }
            if token_ids.peek().is_none() {
                break;
            }
        }
        let child = Vocab::new(entries);
        super::l2p::equivalence_analysis::vocab::common_atom::inherit_vocab_suffix_index(
            vocab,
            &child,
            parent_entry_indices,
        );
        child
    }

    pub fn boundary_vocab(&self, vocab: &Vocab) -> Vocab {
        let child = Self::materialize_vocab(vocab, &self.boundary_token_ids);
        super::l1::inherit_l2p_lexical_entry_order(vocab, &child, &self.boundary_token_ids);
        child
    }

    pub fn single_vocab(&self, vocab: &Vocab) -> Vocab {
        // The single side is consumed only by the projected-L1 builder. It
        // does not need the L2P common-atom suffix provenance propagated by
        // `materialize_vocab`. For the usual sparse split, direct token-ID
        // lookups avoid walking the entire parent partition merely to extract
        // a few dozen entries.
        Vocab::new(
            self.single_token_ids
                .iter()
                .filter_map(|&token_id| {
                    vocab
                        .entries_map()
                        .get(&token_id)
                        .map(|bytes| (token_id, bytes.clone()))
                })
                .collect(),
        )
    }
}

fn merge_sorted_token_ids(mut left: Vec<u32>, right: Vec<u32>) -> Vec<u32> {
    if right.is_empty() {
        return left;
    }
    if left.is_empty() {
        return right;
    }

    let mut merged = Vec::with_capacity(left.len() + right.len());
    let mut right = right.into_iter().peekable();
    for token_id in left.drain(..) {
        while right.peek().is_some_and(|candidate| *candidate < token_id) {
            merged.push(right.next().unwrap());
        }
        merged.push(token_id);
    }
    merged.extend(right);
    merged
}

#[derive(Debug)]
pub struct VocabClassificationFacts {
    bytes: U8Set,
    observed_follow_bytes: [U8Set; 256],
}

impl crate::vocab::VocabDerivedArtifact for VocabClassificationFacts {}

#[derive(Debug)]
struct VocabAdjacentPairIndex {
    pair_offsets: Box<[u32]>,
    entry_token_ids: Box<[u32]>,
    /// Vocabulary entry bytes in the same entry order used by `occurrences`.
    /// Kept flat so hot per-grammar classifiers can address one entry by index
    /// without rebuilding an 80k-element pointer vector from the ordered map.
    entry_byte_offsets: Box<[u32]>,
    entry_bytes: Box<[u8]>,
    /// `(entry_index << 32) | split_after`, grouped by adjacent byte pair.
    /// Entry indices let hot classification loops address a precollected slice
    /// directly instead of performing one ordered-map lookup per occurrence.
    occurrences: Box<[u64]>,
    entry_split_offsets: Box<[usize]>,
    total_splits: usize,
}

impl crate::vocab::VocabDerivedArtifact for VocabAdjacentPairIndex {}

#[derive(Debug)]
struct VocabSuffixTrie {
    /// Parent node for every trie node; node zero is the root and points to itself.
    parents: Box<[u32]>,
    /// Incoming byte for every node. The root byte is unused.
    incoming_bytes: Box<[u8]>,
    /// Trie node representing the complete suffix for each vocabulary split.
    /// Indices use the same flattened split coordinate as `entry_split_offsets`.
    split_nodes: Box<[u32]>,
}

impl crate::vocab::VocabDerivedArtifact for VocabSuffixTrie {}

#[derive(Debug)]
struct VocabPrefixTrie {
    /// Parent node for every trie node; node zero is the root and points to itself.
    parents: Box<[u32]>,
    /// Incoming byte for every node. The root byte is unused.
    incoming_bytes: Box<[u8]>,
    /// Trie node after consuming bytes `[..=split_after]` for each vocabulary split.
    /// Indices use the same flattened split coordinate as `entry_split_offsets`.
    split_nodes: Box<[u32]>,
}

impl crate::vocab::VocabDerivedArtifact for VocabPrefixTrie {}

const PREPARED_SUFFIX_TRIE_MIN_SPLITS: usize = 32_768;

impl VocabAdjacentPairIndex {
    #[inline]
    fn occurrences_for_pair(&self, left: u8, right: u8) -> &[u64] {
        let pair = ((left as usize) << 8) | right as usize;
        let start = self.pair_offsets[pair] as usize;
        let end = self.pair_offsets[pair + 1] as usize;
        &self.occurrences[start..end]
    }

    #[inline]
    fn bytes_for_entry(&self, entry_index: usize) -> &[u8] {
        let start = self.entry_byte_offsets[entry_index] as usize;
        let end = self.entry_byte_offsets[entry_index + 1] as usize;
        &self.entry_bytes[start..end]
    }
}

fn vocab_adjacent_pair_index(vocab: &Vocab) -> Arc<VocabAdjacentPairIndex> {
    if let Some(cached) = vocab.vocab_derived_cache_get::<VocabAdjacentPairIndex>() {
        return cached;
    }

    let mut counts = vec![0u32; 1 << 16];
    let mut entry_split_offsets = Vec::with_capacity(vocab.len() + 1);
    let mut entry_byte_offsets = Vec::with_capacity(vocab.len() + 1);
    let mut entry_bytes = Vec::<u8>::new();
    let mut total_splits = 0usize;
    entry_split_offsets.push(0);
    entry_byte_offsets.push(0);
    for bytes in vocab.entries_map().values() {
        assert!(total_splits <= u32::MAX as usize);
        assert!(entry_bytes.len() <= u32::MAX as usize);
        for pair in bytes.windows(2) {
            counts[((pair[0] as usize) << 8) | pair[1] as usize] += 1;
        }
        total_splits += bytes.len().saturating_sub(1);
        entry_split_offsets.push(total_splits);
        entry_bytes.extend_from_slice(bytes);
        assert!(entry_bytes.len() <= u32::MAX as usize);
        entry_byte_offsets.push(entry_bytes.len() as u32);
    }

    let mut pair_offsets = vec![0u32; (1 << 16) + 1];
    for pair in 0..(1 << 16) {
        pair_offsets[pair + 1] = pair_offsets[pair] + counts[pair];
    }
    let mut next = pair_offsets[..1 << 16].to_vec();
    let mut occurrences = vec![0u64; total_splits];
    for (entry_index, bytes) in vocab.entries_map().values().enumerate() {
        assert!(entry_index <= u32::MAX as usize);
        for (split_after, pair) in bytes.windows(2).enumerate() {
            let pair_id = ((pair[0] as usize) << 8) | pair[1] as usize;
            let slot = &mut next[pair_id];
            occurrences[*slot as usize] = ((entry_index as u64) << 32) | split_after as u64;
            *slot += 1;
        }
    }

    let index = Arc::new(VocabAdjacentPairIndex {
        pair_offsets: pair_offsets.into_boxed_slice(),
        entry_token_ids: vocab.entries_map().keys().copied().collect::<Vec<_>>().into_boxed_slice(),
        entry_byte_offsets: entry_byte_offsets.into_boxed_slice(),
        entry_bytes: entry_bytes.into_boxed_slice(),
        occurrences: occurrences.into_boxed_slice(),
        entry_split_offsets: entry_split_offsets.into_boxed_slice(),
        total_splits,
    });
    vocab.vocab_derived_cache_set(Arc::clone(&index));
    index
}

fn prepared_suffix_trie_enabled() -> bool {
    std::env::var_os("GLRMASK_DISABLE_CLASSIFY_PREPARED_SUFFIX_TRIE").is_none()
}

fn vocab_suffix_trie(vocab: &Vocab) -> Arc<VocabSuffixTrie> {
    if let Some(cached) = vocab.vocab_derived_cache_get::<VocabSuffixTrie>() {
        return cached;
    }
    let split_index = vocab_adjacent_pair_index(vocab);
    let mut parents = vec![0u32];
    let mut incoming_bytes = vec![0u8];
    let mut split_nodes = vec![0u32; split_index.total_splits];
    let mut child_by_edge = FxHashMap::<(u32, u8), u32>::default();

    for (entry_index, bytes) in vocab.entries_map().values().enumerate() {
        let split_base = split_index.entry_split_offsets[entry_index];
        for split_after in 0..bytes.len().saturating_sub(1) {
            let mut node = 0u32;
            for &byte in &bytes[split_after + 1..] {
                let key = (node, byte);
                node = if let Some(&existing) = child_by_edge.get(&key) {
                    existing
                } else {
                    let child = u32::try_from(parents.len())
                        .expect("vocabulary suffix trie exceeds u32 node space");
                    parents.push(node);
                    incoming_bytes.push(byte);
                    child_by_edge.insert(key, child);
                    child
                };
            }
            split_nodes[split_base + split_after] = node;
        }
    }

    let trie = Arc::new(VocabSuffixTrie {
        parents: parents.into_boxed_slice(),
        incoming_bytes: incoming_bytes.into_boxed_slice(),
        split_nodes: split_nodes.into_boxed_slice(),
    });
    vocab.vocab_derived_cache_set(Arc::clone(&trie));
    trie
}

fn vocab_prefix_trie(vocab: &Vocab) -> Arc<VocabPrefixTrie> {
    if let Some(cached) = vocab.vocab_derived_cache_get::<VocabPrefixTrie>() {
        return cached;
    }
    let split_index = vocab_adjacent_pair_index(vocab);
    let mut parents = vec![0u32];
    let mut incoming_bytes = vec![0u8];
    let mut split_nodes = vec![0u32; split_index.total_splits];
    let mut child_by_edge = FxHashMap::<(u32, u8), u32>::default();

    for (entry_index, bytes) in vocab.entries_map().values().enumerate() {
        let split_base = split_index.entry_split_offsets[entry_index];
        let mut node = 0u32;
        for split_after in 0..bytes.len().saturating_sub(1) {
            let byte = bytes[split_after];
            let key = (node, byte);
            node = if let Some(&existing) = child_by_edge.get(&key) {
                existing
            } else {
                let child = u32::try_from(parents.len())
                    .expect("vocabulary prefix trie exceeds u32 node space");
                parents.push(node);
                incoming_bytes.push(byte);
                child_by_edge.insert(key, child);
                child
            };
            split_nodes[split_base + split_after] = node;
        }
    }

    let trie = Arc::new(VocabPrefixTrie {
        parents: parents.into_boxed_slice(),
        incoming_bytes: incoming_bytes.into_boxed_slice(),
        split_nodes: split_nodes.into_boxed_slice(),
    });
    vocab.vocab_derived_cache_set(Arc::clone(&trie));
    trie
}

fn vocab_classification_facts(vocab: &Vocab) -> Arc<VocabClassificationFacts> {
    if let Some(cached) = vocab.vocab_derived_cache_get::<VocabClassificationFacts>() {
        return cached;
    }

    let mut byteset = U8Set::empty();
    let mut observed_follow_bytes = [U8Set::empty(); 256];
    for bytes in vocab.entries_map().values() {
        for &byte in bytes {
            byteset.insert(byte);
        }
        for pair in bytes.windows(2) {
            observed_follow_bytes[pair[0] as usize].insert(pair[1]);
        }
    }
    let facts = Arc::new(VocabClassificationFacts {
        bytes: byteset,
        observed_follow_bytes,
    });
    vocab.vocab_derived_cache_set(Arc::clone(&facts));
    facts
}

pub fn cache_vocab_classification_facts(
    vocab: &Vocab,
    bytes: U8Set,
    observed_follow_bytes: [U8Set; 256],
) {
    vocab.vocab_derived_cache_set(Arc::new(VocabClassificationFacts {
        bytes,
        observed_follow_bytes,
    }));
}

fn vocab_byte_set(vocab: &Vocab) -> U8Set {
    vocab_classification_facts(vocab).bytes
}

pub fn vocab_tokens_with_adjacent_pairs(
    vocab: &Vocab,
    allowed_pairs: &[U8Set; 256],
) -> Vec<u32> {
    let index = vocab_adjacent_pair_index(vocab);
    let mut total_occurrences = 0usize;
    for (left, rights) in allowed_pairs.iter().enumerate() {
        for right in rights.iter() {
            total_occurrences += index.occurrences_for_pair(left as u8, right).len();
        }
    }

    // Sparse boundary queries usually touch a tiny fraction of the vocabulary.
    // Avoid allocating/zeroing a full-vocabulary bitmap and scanning every
    // entry merely to recover those few token ids. Entry indices sort in the
    // same order as `entry_token_ids`, so this preserves deterministic output.
    if total_occurrences.saturating_mul(2) <= index.entry_token_ids.len() {
        let mut selected_entries = Vec::with_capacity(total_occurrences);
        for (left, rights) in allowed_pairs.iter().enumerate() {
            for right in rights.iter() {
                selected_entries.extend(
                    index
                        .occurrences_for_pair(left as u8, right)
                        .iter()
                        .map(|&occurrence| (occurrence >> 32) as u32),
                );
            }
        }
        selected_entries.sort_unstable();
        selected_entries.dedup();
        return selected_entries
            .into_iter()
            .map(|entry| index.entry_token_ids[entry as usize])
            .collect();
    }

    let mut selected = vec![false; index.entry_token_ids.len()];
    for (left, rights) in allowed_pairs.iter().enumerate() {
        for right in rights.iter() {
            for &occurrence in index.occurrences_for_pair(left as u8, right) {
                selected[(occurrence >> 32) as usize] = true;
            }
        }
    }
    selected
        .into_iter()
        .enumerate()
        .filter_map(|(entry, selected)| selected.then_some(index.entry_token_ids[entry]))
        .collect()
}

pub fn prepare_vocab_for_terminal_classification(vocab: &Vocab) {
    let _ = vocab_classification_facts(vocab);
    let prepare_common_atom_suffix_index =
        std::env::var_os("GLRMASK_DISABLE_PREPARE_COMMON_ATOM_SUFFIX_INDEX").is_none()
            && (std::env::var_os("GLRMASK_PREPARE_COMMON_ATOM_SUFFIX_INDEX").is_some()
                || (16_384..=100_000).contains(&vocab.len()));
    if prepare_common_atom_suffix_index {
        super::l2p::equivalence_analysis::vocab::common_atom::prepare_vocab_suffix_index(vocab);
    }
    let adjacent = vocab_adjacent_pair_index(vocab);
    // Large terminal-boundary classifiers can amortize a suffix trie across
    // every grammar compiled against this fixed vocabulary. Build it during
    // explicit vocabulary preparation, never on the schema critical path.
    if prepared_suffix_trie_enabled() && adjacent.total_splits >= PREPARED_SUFFIX_TRIE_MIN_SPLITS {
        let _ = vocab_suffix_trie(vocab);
        let _ = vocab_prefix_trie(vocab);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2pPartitionCostFn {
    Size,
    SizeLog,
    LogLog,
    UnionSize,
}

impl L2pPartitionCostFn {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Size => "size",
            Self::SizeLog => "size_log",
            Self::LogLog => "log_log",
            Self::UnionSize => "union_size",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2pPartitionObjective {
    Max,
    Sum,
}

impl L2pPartitionObjective {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Max => "max",
            Self::Sum => "sum",
        }
    }
}

pub struct L2pCostPartitioning {
    pub partitions: Vec<Vec<u32>>,
    pub estimated_partition_costs: Vec<f64>,
    pub estimated_l2p_terminals: Vec<usize>,
    pub objective_score: f64,
}

#[derive(Clone)]
struct L2pTokenGroup {
    l2p_terminals: BitSet,
    token_ids: Vec<u32>,
}

#[derive(Clone)]
struct L2pPartitionBucket {
    l2p_intersection: Option<BitSet>,
    l2p_union: Option<BitSet>,
    token_ids: Vec<u32>,
}

impl L2pPartitionBucket {
    fn new() -> Self {
        Self {
            l2p_intersection: None,
            l2p_union: None,
            token_ids: Vec::new(),
        }
    }

    fn size(&self) -> usize {
        self.token_ids.len()
    }

    fn l2p_count(&self) -> usize {
        self.l2p_intersection.as_ref().map_or(0, BitSet::count_ones)
    }
}

impl SharedClassifyBytesets {
    /// Union of bytes that can make progress toward any terminal.
    ///
    /// A vocabulary partition whose observed bytes are disjoint from this set
    /// classifies every terminal as `TerminalPathLength::Zero`, so callers may
    /// skip materializing that partition's token-byte map entirely.
    pub fn reachable_bytes_union(&self) -> U8Set {
        self.reachable_bytes
            .iter()
            .fold(U8Set::empty(), |acc, bytes| acc.union(bytes))
    }

    #[inline]
    pub fn transitions_by_byte(&self) -> &[u32] {
        &self.transitions_by_byte
    }

    /// Scan the DFA to compute per-terminal byte sets.
    pub fn build(tokenizer: &Tokenizer, num_terminals: u32) -> Self {
        let nt = num_terminals as usize;
        // The old implementation expanded target terminal bitsets once per
        // transition. On a large tokenizer this is catastrophic: many
        // transitions reach states whose possible-future bitsets contain most
        // terminals. The result only depends on the transition byte, so first
        // union terminal bitsets into 256 byte buckets, then transpose those
        // buckets once into the terminal -> byte sets required by callers.
        let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
        let started_at = std::time::Instant::now();
        let words_per_terminal_set = nt.div_ceil(64);
        let mut reachable_by_byte = vec![0u64; 256 * words_per_terminal_set];
        let mut last_by_byte = vec![0u64; 256 * words_per_terminal_set];
        let num_states = tokenizer.num_states() as usize;
        let mut transitions_by_byte = vec![u32::MAX; 256 * num_states];
        let mut sparse_transitions_by_byte = vec![Vec::<(u32, u32)>::new(); 256];
        let build_ti_output_index =
            std::env::var_os("GLRMASK_DISABLE_CLASSIFY_TI_OUTPUT_INDEX").is_none();
        let mut matched_terminals_by_state =
            Vec::with_capacity(if build_ti_output_index { num_states } else { 0 });
        let mut future_terminals_by_state =
            Vec::with_capacity(if build_ti_output_index { num_states } else { 0 });
        let mut matched_states_by_terminal = vec![Vec::<u32>::new(); nt];
        let mut future_states_by_terminal = vec![Vec::<u32>::new(); nt];
        let mut has_matched_terminal_by_state = vec![0u8; num_states];
        let mut future_by_state_words = vec![0u64; num_states * words_per_terminal_set];
        let mut representative_future_terminal_by_state = vec![u32::MAX; num_states];
        let mut transition_count = 0usize;

        // Every transition used to recompute the empty-input epsilon closure of
        // its target and re-OR the same terminal observations. Large synthesized
        // tokenizers have millions of transitions but only O(100k) target
        // states. Precompute the exact closure observation once per state.
        let singleton_closures = tokenizer.all_singleton_epsilon_closures();
        let closure_output_started_at = std::time::Instant::now();
        let mut closure_reachable_words = vec![0u64; num_states * words_per_terminal_set];
        let mut closure_last_words = vec![0u64; num_states * words_per_terminal_set];
        for state in 0..num_states {
            let base = state * words_per_terminal_set;
            for &closure_state in &singleton_closures[state] {
                let matched_words = tokenizer.matched_terminal_bitset(closure_state).words();
                let future_words = tokenizer.possible_future_terminals(closure_state).words();
                // BitSet storage is allowed to omit trailing zero words.  In
                // particular, loaded/composed tokenizers can have a large global
                // terminal universe while a given state mentions only low IDs.
                // Semantically those absent words are zero; do not require the
                // backing vectors to be padded to `num_terminals`.
                for word_index in 0..words_per_terminal_set {
                    let matched_word = matched_words.get(word_index).copied().unwrap_or(0);
                    let future_word = future_words.get(word_index).copied().unwrap_or(0);
                    closure_last_words[base + word_index] |= matched_word;
                    closure_reachable_words[base + word_index] |= matched_word | future_word;
                }
            }
        }
        let closure_output_ms = closure_output_started_at.elapsed().as_secs_f64() * 1000.0;

        for state in 0..tokenizer.num_states() {
            let matched = tokenizer
                .matched_terminals(state)
                .into_iter()
                .filter(|terminal| (*terminal as usize) < nt)
                .collect::<Vec<_>>();
            for &terminal in &matched {
                matched_states_by_terminal[terminal as usize].push(state);
                has_matched_terminal_by_state[state as usize] = 1;
            }
            if build_ti_output_index {
                matched_terminals_by_state.push(matched.into_boxed_slice());
            }

            let future_words = tokenizer.possible_future_terminals(state).words();
            let future_dst = &mut future_by_state_words[state as usize * words_per_terminal_set
                ..(state as usize + 1) * words_per_terminal_set];
            let copy_words = future_words.len().min(future_dst.len());
            future_dst[..copy_words].copy_from_slice(&future_words[..copy_words]);
            if build_ti_output_index {
                let future = tokenizer
                    .possible_future_terminals_iter(state)
                    .filter(|terminal| (*terminal as usize) < nt)
                    .collect::<Vec<_>>();
                for &terminal in &future {
                    future_states_by_terminal[terminal as usize].push(state);
                }
                future_terminals_by_state.push(future.into_boxed_slice());
            }
            if let Some((word_index, &word)) = future_words
                .iter()
                .take(words_per_terminal_set)
                .enumerate()
                .find(|(_, word)| **word != 0)
            {
                representative_future_terminal_by_state[state as usize] =
                    (word_index * 64 + word.trailing_zeros() as usize) as u32;
            }
            for (byte, target) in tokenizer.transitions_from(state) {
                transition_count += 1;
                transitions_by_byte[byte as usize * num_states + state as usize] = target;
                sparse_transitions_by_byte[byte as usize].push((state, target));
                let bucket_offset = byte as usize * words_per_terminal_set;
                let target_offset = target as usize * words_per_terminal_set;
                for word_index in 0..words_per_terminal_set {
                    reachable_by_byte[bucket_offset + word_index] |=
                        closure_reachable_words[target_offset + word_index];
                    last_by_byte[bucket_offset + word_index] |=
                        closure_last_words[target_offset + word_index];
                }
            }
        }
        let reverse_transitions_by_byte = if tokenizer.has_epsilon_transitions() {
            // Only the deterministic exact-boundary path consumes this index.
            // Epsilon tokenizers use their bounded/powerset analysis view, so a
            // reverse copy of every raw transition is unreachable work.
            Vec::new()
        } else {
            build_reverse_transitions_by_byte(&sparse_transitions_by_byte, num_states)
        };
        let scan_ms = started_at.elapsed().as_secs_f64() * 1000.0;

        let mut reachable_bytes = vec![U8Set::empty(); nt];
        let mut last_bytes = vec![U8Set::empty(); nt];
        for byte in 0u8..=u8::MAX {
            let bucket_offset = byte as usize * words_per_terminal_set;
            for word_index in 0..words_per_terminal_set {
                let base_terminal = word_index * 64;
                let mut reachable_word = reachable_by_byte[bucket_offset + word_index];
                while reachable_word != 0 {
                    let terminal = base_terminal + reachable_word.trailing_zeros() as usize;
                    if terminal < nt {
                        reachable_bytes[terminal].insert(byte);
                    }
                    reachable_word &= reachable_word - 1;
                }

                let mut last_word = last_by_byte[bucket_offset + word_index];
                while last_word != 0 {
                    let terminal = base_terminal + last_word.trailing_zeros() as usize;
                    if terminal < nt {
                        last_bytes[terminal].insert(byte);
                    }
                    last_word &= last_word - 1;
                }
            }
        }

        // first_bytes: one exact byte step from the epsilon-closed reset
        // configuration. The ordinary DFA case naturally has one state.
        let mut first_bytes = vec![U8Set::empty(); nt];
        let reset_states = tokenizer.execute_from_state_end_only(&[], tokenizer.initial_state_id());
        for byte in 0u8..=u8::MAX {
            let targets = tokenizer.step_all(&reset_states, byte);
            for target in targets {
                for terminal in tokenizer.matched_terminal_bitset(target).iter() {
                    if terminal < nt {
                        first_bytes[terminal].insert(byte);
                    }
                }
                for terminal in tokenizer.possible_future_terminals(target).iter() {
                    let t = terminal;
                    if t < nt {
                        first_bytes[t].insert(byte);
                    }
                }
            }
        }

        if profile_enabled {
            eprintln!(
                "[glrmask/profile][classify_bytesets] terminals={} states={} transitions={} words_per_terminal_set={} closure_output_ms={:.3} scan_ms={:.3} transpose_and_first_ms={:.3} total_ms={:.3}",
                nt,
                tokenizer.num_states(),
                transition_count,
                words_per_terminal_set,
                closure_output_ms,
                scan_ms,
                started_at.elapsed().as_secs_f64() * 1000.0 - scan_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        SharedClassifyBytesets {
            reachable_bytes,
            first_bytes,
            last_bytes,
            transitions_by_byte,
            sparse_transitions_by_byte,
            reverse_transitions_by_byte,
            matched_terminals_by_state: matched_terminals_by_state.into(),
            future_terminals_by_state: future_terminals_by_state.into(),
            matched_states_by_terminal: matched_states_by_terminal.into(),
            future_states_by_terminal: future_states_by_terminal.into(),
            has_matched_terminal_by_state,
            future_by_state_words,
            representative_future_terminal_by_state,
            words_per_terminal_set,
            active_route_setup_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }
}

/// JSON structural characters used to keep tokens in the core non-alnum
/// partition (P0) rather than splitting them into the auxiliary P5.
const JSON_STRUCTURAL: &[u8] = b"\":[]{},";

/// `_` belongs with alphabetic bytes for vocabulary partitioning. This is a
/// routing convention only: it does not change lexer or grammar semantics.
fn is_partition_ascii_alpha(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

/// Characters whose sole repetition qualifies a non-alnum token for the
/// auxiliary P5 partition even if the token contains a structural byte.
const P5_REPEATED_CHARS: &[u8] = b"\n:{ ,";

/// Classifies a token's bytes by character type for vocab partitioning.
///
/// Returns:
/// - 0: non-alnum with JSON structural chars (multi-byte, not single-repeated)
/// - 1: mixed (contains both alnum and non-alnum)
/// - 2: ASCII word token with ≥1 alpha or `_`, optionally with leading space
/// - 3: pure digit, optionally with leading space
/// - 4: Unicode-only alpha (non-ASCII alphanumeric, e.g. CJK, Cyrillic,
///       Arabic, Hangul), optionally with leading space
/// - 5: non-alnum auxiliary short (no JSON structural, or single-char repeated,
///       or length 1; ≤ 8 bytes)
/// - 6: non-alnum auxiliary long (same criteria as 5, but > 8 bytes)
/// - 7: JSON literal-boundary tokens requiring structural treatment (leading-
///       space collisions, bracketed forms, and the special ` -` token)
/// - 8: quoted ASCII identifier-start tokens
///
/// Uses Unicode-aware classification so that non-Latin scripts are separated
/// into their own partition (4) instead of being lumped with ASCII punctuation (0)
/// or bloating the ASCII alpha partition (2).
///
/// P0/P5 split: non-alnum tokens containing JSON structural characters
/// (`":[]{},`) stay in P0 for efficient L2+ terminal processing, while
/// tokens without structural chars (or trivial single-char tokens) go to P5.
pub(crate) fn fast_eval_char_type_regular_partition(bytes: &[u8]) -> u8 {
    if bytes.is_empty() {
        return 5;
    }
    let leading_space = bytes[0] == b' ';
    let content = if leading_space { &bytes[1..] } else { bytes };
    let literal_collision = is_json_literal_collision(content);
    // Bare ASCII word pieces that overlap a JSON literal spelling are ordinary
    // P2 material. Only their leading-space variants need to stay isolated at
    // the structural boundary.
    if !leading_space && literal_collision {
        return 2;
    }
    if is_quoted_identifier_boundary_token(bytes) {
        return 8;
    }
    if structural_boundary_lexical_partition_enabled()
        && (literal_collision
            || bytes == b" -"
            || (bytes.starts_with(b"[") && is_json_literal_collision(&bytes[1..])))
    {
        return 7;
    }
    // Strip optional leading ASCII space (GPT-2 BPE decodes Ġ → 0x20 before we see it)
    if content.is_empty() {
        return 5; // Just a space marker → auxiliary non-alnum
    }
    if content.len() == 1 && matches!(content[0], b'+' | b'-') {
        return 1;
    }
    // Most model-vocabulary entries are ASCII. Byte predicates are exactly
    // equivalent to the Unicode predicates below for ASCII input, while
    // avoiding UTF-8 validation and repeated `chars()` walks on this hot path.
    if content.is_ascii() {
        let all_word = content
            .iter()
            .copied()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
        if all_word {
            if content.iter().copied().any(is_partition_ascii_alpha) {
                return 2;
            }
            return 3;
        }
        if bytes
            .iter()
            .copied()
            .all(|byte| !byte.is_ascii_alphanumeric() && byte != b'_')
        {
            return classify_nonalnum(bytes);
        }
        return 1;
    }
    // Try to decode as UTF-8 for Unicode-aware classification.
    if let Ok(s) = std::str::from_utf8(content) {
        let all_word = s.chars().all(|c| c.is_alphanumeric() || c == '_');
        if all_word {
            let has_alpha = s.chars().any(|c| c.is_alphabetic() || c == '_');
            if has_alpha {
                let has_ascii_alpha = content.iter().copied().any(is_partition_ascii_alpha);
                if has_ascii_alpha {
                    return 2; // ASCII word token (may also contain non-ASCII alpha)
                }
                return 4; // Unicode-only alpha (CJK, Cyrillic, Arabic, etc.)
            }
            return 3; // Pure digit
        }
        // Check non-alphanumeric.
        if let Ok(full) = std::str::from_utf8(bytes) {
            if !full
                .chars()
                .any(|c| c.is_alphanumeric() || c == '_')
            {
                return classify_nonalnum(bytes);
            }
        }
        return 1; // Mixed
    }
    // Fallback: byte-level ASCII checks for invalid UTF-8.
    if content
        .iter()
        .copied()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        if content.iter().copied().any(is_partition_ascii_alpha) {
            return 2;
        }
        return 3;
    }
    if bytes
        .iter()
        .copied()
        .all(|byte| !byte.is_ascii_alphanumeric() && byte != b'_')
    {
        return classify_nonalnum(bytes);
    }
    1 // Mixed
}

#[cfg(test)]
fn reference_char_type_regular_partition(bytes: &[u8]) -> u8 {
    if bytes.is_empty() {
        return 5;
    }
    if !bytes.starts_with(b" ") && reference_json_literal_collision(bytes) {
        return 2;
    }
    if is_quoted_identifier_boundary_token(bytes) {
        return 8;
    }
    if structural_boundary_lexical_partition_enabled() {
        let content = bytes.strip_prefix(b" ").unwrap_or(bytes);
        if reference_json_literal_collision(content)
            || bytes == b" -"
            || (bytes.starts_with(b"[") && reference_json_literal_collision(&bytes[1..]))
        {
            return 7;
        }
    }
    let content = if bytes[0] == b' ' {
        &bytes[1..]
    } else {
        bytes
    };
    if content.is_empty() {
        return 5;
    }
    if content.len() == 1 && matches!(content[0], b'+' | b'-') {
        return 1;
    }
    if let Ok(s) = std::str::from_utf8(content) {
        let all_word = s.chars().all(|c| c.is_alphanumeric() || c == '_');
        if all_word {
            let has_alpha = s.chars().any(|c| c.is_alphabetic() || c == '_');
            if has_alpha {
                let has_ascii_alpha = content.iter().copied().any(is_partition_ascii_alpha);
                if has_ascii_alpha {
                    return 2;
                }
                return 4;
            }
            return 3;
        }
        if let Ok(full) = std::str::from_utf8(bytes)
            && !full.chars().any(|c| c.is_alphanumeric() || c == '_')
        {
            return classify_nonalnum(bytes);
        }
        return 1;
    }
    if content
        .iter()
        .copied()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        if content.iter().copied().any(is_partition_ascii_alpha) {
            return 2;
        }
        return 3;
    }
    if bytes
        .iter()
        .copied()
        .all(|byte| !byte.is_ascii_alphanumeric() && byte != b'_')
    {
        return classify_nonalnum(bytes);
    }
    1
}

#[cfg(test)]
fn reference_json_literal_collision(content: &[u8]) -> bool {
    if content.is_empty() || !content.iter().all(|byte| byte.is_ascii_alphanumeric()) {
        return false;
    }
    [b"true".as_slice(), b"false".as_slice(), b"null".as_slice()]
        .iter()
        .any(|literal| literal.starts_with(content) || content.starts_with(literal))
}

fn is_json_literal_collision(content: &[u8]) -> bool {
    if content.is_empty() {
        return false;
    }
    for literal in [b"true".as_slice(), b"false".as_slice(), b"null".as_slice()] {
        // A shorter matching prefix consists entirely of literal ASCII bytes,
        // so no separate alphanumeric validation is needed. For a token that
        // extends a complete literal, retain the historical requirement that
        // the extension itself remain alphanumeric.
        if literal.starts_with(content)
            || (content.starts_with(literal)
                && content.iter().all(|byte| byte.is_ascii_alphanumeric()))
        {
            return true;
        }
    }
    false
}

fn is_quoted_identifier_boundary_token(bytes: &[u8]) -> bool {
    structural_boundary_lexical_partition_enabled()
        && bytes
        .strip_prefix(b"\"")
        .is_some_and(|suffix| suffix.first().copied().is_some_and(is_partition_ascii_alpha))
}

pub(crate) fn structural_boundary_lexical_partition_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_STRUCTURAL_BOUNDARY_LEXICAL_PARTITION")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || trimmed == "1" || trimmed.eq_ignore_ascii_case("true")
            })
            .unwrap_or(true)
    })
}

/// Classify one token using the canonical regular-language partition by default.
/// A process-level custom regex partition remains available for experiments.
pub fn classify_vocab_char_type(bytes: &[u8]) -> u8 {
    if vocab_partition_is_custom() {
        let set = vocab_partition_set();
        classify_with_partition_set(set, bytes).unwrap_or_else(|| {
            panic!(
                "regex vocabulary partitions are not exhaustive for token bytes {:?}",
                bytes
            )
        })
    } else {
        fast_eval_char_type_regular_partition(bytes)
    }
}


/// Sub-classify a non-alphanumeric token into P0 (structural), P5 (short auxiliary),
/// or P6 (long auxiliary).
///
/// P5/P6 if: (a) no JSON structural char, (b) single repeated char from
/// `\n:{ ,`, or (c) length 1. Within that group, tokens > 8 bytes go to P6.
fn classify_nonalnum(bytes: &[u8]) -> u8 {
    // Length 1 → P5
    if bytes.len() <= 1 {
        return 5;
    }
    // Single repeated char from P5_REPEATED_CHARS → P5/P6
    if bytes.iter().all(|b| *b == bytes[0]) && P5_REPEATED_CHARS.contains(&bytes[0]) {
        return if bytes.len() > 8 { 6 } else { 5 };
    }
    // No JSON structural char → P5/P6
    if !bytes.iter().any(|b| JSON_STRUCTURAL.contains(b)) {
        return if bytes.len() > 8 { 6 } else { 5 };
    }
    0 // Structural non-alnum → P0
}


/// Environment override for the vocabulary partition languages.
///
/// The value may be either a JSON array or a newline-separated list. JSON
/// entries may be strings or `{ "name": "...", "regex": "..." }` objects.
/// Rules are ordered: if they overlap, partition i denotes its regex minus the
/// union of every earlier regex. An implicit final `rest` partition is added
/// for custom configurations, so the resulting languages are exhaustive.
pub const VOCAB_PARTITION_REGEX_ENV: &str = "GLRMASK_VOCAB_PARTITION_REGEXES";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabPartitionLanguage {
    pub name: String,
    pub include_regex: String,
    pub excluded_regexes: Vec<String>,
}

impl VocabPartitionLanguage {
    /// Exact effective language of this ordered partition.
    ///
    /// `include_regex` alone is not necessarily the partition language when
    /// earlier rules overlap it. The effective language is
    /// `include & !(excluded_0 | ... | excluded_n)`.
    pub fn effective_ast(&self) -> derivre::RegexAst {
        use derivre::RegexAst;
        let include = RegexAst::Regex(self.include_regex.clone());
        if self.excluded_regexes.is_empty() {
            include
        } else {
            RegexAst::And(vec![
                include,
                RegexAst::Not(Box::new(RegexAst::Or(
                    self.excluded_regexes
                        .iter()
                        .cloned()
                        .map(RegexAst::Regex)
                        .collect(),
                ))),
            ])
        }
    }

    pub fn compile_effective_regex(&self) -> Result<derivre::Regex, String> {
        let mut builder = derivre::RegexBuilder::new();
        builder.unicode(false).utf8(false);
        let expr = builder
            .mk(&self.effective_ast())
            .map_err(|error| format!("invalid partition regex {}: {error}", self.name))?;
        builder
            .into_regex_limited(expr, 1_000_000)
            .map_err(|error| format!("failed to compile partition regex {}: {error}", self.name))
    }

    pub fn compile_effective_dfa(&self) -> Result<VocabPartitionDfa, String> {
        VocabPartitionDfa::compile_regex(&self.name, self.compile_effective_regex()?)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct VocabPartitionDfa {
    #[serde(with = "u8_256_serde")]
    byte_to_class: [u8; 256],
    class_count: usize,
    transitions: Vec<u32>,
    accepting: Vec<bool>,
    can_reach_accepting: Vec<bool>,
}

mod u8_256_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &[u8; 256], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value.as_slice().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 256], D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Vec::<u8>::deserialize(deserializer)?;
        value.try_into().map_err(|value: Vec<u8>| {
            serde::de::Error::custom(format!("expected 256 byte classes, got {}", value.len()))
        })
    }
}

impl VocabPartitionDfa {
    #[inline]
    pub fn state_count(&self) -> usize {
        self.accepting.len()
    }

    pub fn compile_byte_regex(name: &str, pattern: &str) -> Result<Self, String> {
        let mut builder = derivre::RegexBuilder::new();
        builder.unicode(false).utf8(false);
        let expr = builder
            .mk_regex(pattern)
            .map_err(|error| format!("invalid partition regex {name}: {error}"))?;
        let regex = builder
            .into_regex_limited(expr, 1_000_000)
            .map_err(|error| format!("failed to compile partition regex {name}: {error}"))?;
        Self::compile_regex(name, regex)
    }

    /// Compile a Unicode/UTF-8 regex to the same immutable byte-DFA form.
    /// This is used by llguidance-compatible overlapping slicer languages,
    /// whose character classes are defined over Unicode scalar values even
    /// though execution ultimately consumes tokenizer bytes.
    pub fn compile_utf8_regex(name: &str, pattern: &str) -> Result<Self, String> {
        let regex = derivre::Regex::new(pattern)
            .map_err(|error| format!("invalid UTF-8 slice regex {name}: {error}"))?;
        Self::compile_regex(name, regex)
    }

    fn compile_regex(name: &str, mut regex: derivre::Regex) -> Result<Self, String> {
        use derivre::StateID;
        // `derivre` already computes an exact byte-equivalence alphabet while
        // parsing the regex. Build only one transition per equivalence class,
        // rather than expanding every state back to all 256 raw bytes.
        let class_count = regex.alpha().len();
        if class_count == 0 || class_count > 256 {
            return Err(format!(
                "partition regex {name} has invalid alphabet size {class_count}"
            ));
        }
        let mut byte_to_class = [0u8; 256];
        let mut representatives = vec![None; class_count];
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let class = regex.alpha().map(byte);
            byte_to_class[byte as usize] = class as u8;
            representatives[class].get_or_insert(byte);
        }
        let representatives = representatives
            .into_iter()
            .enumerate()
            .map(|(class, representative)| {
                representative.ok_or_else(|| {
                    format!("partition regex {name} has empty byte class {class}")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let start = regex.initial_state();
        let mut state_to_index = FxHashMap::<StateID, u32>::default();
        let mut states = Vec::<StateID>::new();
        state_to_index.insert(start, 0);
        states.push(start);
        let mut transitions = Vec::<u32>::new();
        let mut accepting = Vec::<bool>::new();
        let mut cursor = 0usize;
        while cursor < states.len() {
            let state = states[cursor];
            accepting.push(regex.is_accepting(state));
            let row_start = transitions.len();
            transitions.resize(row_start + class_count, 0);
            for (class, &byte) in representatives.iter().enumerate() {
                let target = regex.transition(state, byte);
                let target_index = if let Some(&index) = state_to_index.get(&target) {
                    index
                } else {
                    let index = states.len() as u32;
                    state_to_index.insert(target, index);
                    states.push(target);
                    index
                };
                transitions[row_start + class] = target_index;
            }
            cursor += 1;
            if states.len() > 65_536 {
                return Err(format!(
                    "partition regex {name} expanded beyond 65536 DFA states"
                ));
            }
        }
        let mut reverse = vec![Vec::<u32>::new(); accepting.len()];
        for (source, row) in transitions.chunks_exact(class_count).enumerate() {
            for &target in row {
                let target = target as usize;
                if target < reverse.len() {
                    reverse[target].push(source as u32);
                }
            }
        }
        let mut can_reach_accepting = accepting.clone();
        let mut queue = std::collections::VecDeque::<u32>::new();
        for (state, &is_accepting) in accepting.iter().enumerate() {
            if is_accepting {
                queue.push_back(state as u32);
            }
        }
        while let Some(target) = queue.pop_front() {
            for &source in &reverse[target as usize] {
                if !can_reach_accepting[source as usize] {
                    can_reach_accepting[source as usize] = true;
                    queue.push_back(source);
                }
            }
        }
        Ok(Self {
            byte_to_class,
            class_count,
            transitions,
            accepting,
            can_reach_accepting,
        })
    }

    #[inline]
    pub fn is_match(&self, bytes: &[u8]) -> bool {
        let mut state = 0usize;
        for &byte in bytes {
            state = self.step(state as u32, byte) as usize;
        }
        self.is_accepting(state as u32)
    }

    #[inline]
    pub fn start_state(&self) -> u32 {
        0
    }

    #[inline]
    pub fn byte_class(&self, byte: u8) -> u8 {
        self.byte_to_class[byte as usize]
    }

    #[inline]
    pub fn class_count(&self) -> usize {
        self.class_count
    }

    #[inline]
    pub fn byte_to_class_map(&self) -> &[u8; 256] {
        &self.byte_to_class
    }

    #[inline]
    pub fn transition_table(&self) -> &[u32] {
        &self.transitions
    }

    #[inline]
    pub fn can_reach_accepting_map(&self) -> &[bool] {
        &self.can_reach_accepting
    }

    #[inline]
    pub fn accepting_map(&self) -> &[bool] {
        &self.accepting
    }

    #[inline]
    pub fn step(&self, state: u32, byte: u8) -> u32 {
        let class = self.byte_class(byte) as usize;
        self.transitions[state as usize * self.class_count + class]
    }

    #[inline]
    pub fn is_accepting(&self, state: u32) -> bool {
        self.accepting.get(state as usize).copied().unwrap_or(false)
    }

    #[inline]
    pub fn can_reach_accepting(&self, state: u32) -> bool {
        self.can_reach_accepting
            .get(state as usize)
            .copied()
            .unwrap_or(false)
    }

    /// A regular language is infinite iff the reachable subgraph that can
    /// still reach acceptance contains a cycle.
    pub fn has_finite_language(&self) -> bool {
        let n = self.state_count();
        if n == 0 || !self.can_reach_accepting(0) {
            return true;
        }
        let mut reachable = vec![false; n];
        let mut stack = vec![0u32];
        reachable[0] = true;
        while let Some(state) = stack.pop() {
            let row = state as usize * self.class_count;
            for class in 0..self.class_count {
                let target = self.transitions[row + class];
                if self.can_reach_accepting(target) && !reachable[target as usize] {
                    reachable[target as usize] = true;
                    stack.push(target);
                }
            }
        }
        let mut indegree = vec![0u32; n];
        let mut live_count = 0usize;
        for state in 0..n as u32 {
            if !reachable[state as usize] {
                continue;
            }
            live_count += 1;
            let row = state as usize * self.class_count;
            for class in 0..self.class_count {
                let target = self.transitions[row + class];
                if reachable[target as usize] {
                    indegree[target as usize] = indegree[target as usize].saturating_add(1);
                }
            }
        }
        let mut queue = std::collections::VecDeque::new();
        for state in 0..n as u32 {
            if reachable[state as usize] && indegree[state as usize] == 0 {
                queue.push_back(state);
            }
        }
        let mut removed = 0usize;
        while let Some(state) = queue.pop_front() {
            removed += 1;
            let row = state as usize * self.class_count;
            for class in 0..self.class_count {
                let target = self.transitions[row + class];
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
}


#[derive(Debug)]
struct VocabPartitionSet {
    /// Exact effective languages indexed by stable partition id.
    languages: Vec<VocabPartitionLanguage>,
    /// Matchers in priority order. The u8 is the stable partition id returned
    /// by `classify_vocab_char_type`.
    matchers: Vec<(u8, VocabPartitionDfa)>,
    effective_dfas: Vec<std::sync::OnceLock<Arc<VocabPartitionDfa>>>,
    custom: bool,
}

#[derive(Debug, Clone)]
struct VocabPartitionRule {
    partition: u8,
    name: String,
    regex: String,
}

fn default_vocab_partition_rules() -> Vec<VocabPartitionRule> {
    // Byte-regex equivalents of the historical broad partition shapes. The
    // language is defined entirely by these expressions; no semantic callback
    // participates in membership. High bytes are kept together as a separate
    // word-like family so arbitrary byte vocabularies remain supported.
    let collision = r"(?:t|tr|tru|true[A-Za-z0-9]*|f|fa|fal|fals|false[A-Za-z0-9]*|n|nu|nul|null[A-Za-z0-9]*)";
    vec![
        VocabPartitionRule {
            partition: 8,
            name: "p8".into(),
            regex: r#"\"[A-Za-z_][\x00-\xFF]*"#.into(),
        },
        VocabPartitionRule {
            partition: 7,
            name: "p7".into(),
            regex: format!(r"(?: {collision}|\[{collision}| -)"),
        },
        VocabPartitionRule {
            partition: 2,
            name: "p2".into(),
            regex: r" ?[A-Za-z0-9_\x80-\xFF]*[A-Za-z_][A-Za-z0-9_\x80-\xFF]*".into(),
        },
        VocabPartitionRule {
            partition: 3,
            name: "p3".into(),
            regex: r" ?[0-9]+".into(),
        },
        VocabPartitionRule {
            partition: 4,
            name: "p4".into(),
            regex: r" ?[0-9\x80-\xFF]+".into(),
        },
        VocabPartitionRule {
            partition: 6,
            name: "p6".into(),
            regex: r#"(?:[^\"\:\[\]\{\},A-Za-z0-9_\x80-\xFF]{9,}|\n{9,}|:{9,}|\{{9,}| {9,}|,{9,}|\+{9,}|-{9,})"#.into(),
        },
        VocabPartitionRule {
            partition: 5,
            name: "p5".into(),
            regex: r#"(?:|[^\"\:\[\]\{\},A-Za-z0-9_\x80-\xFF]{1,8}|\n{1,8}|:{1,8}|\{{1,8}| {1,8}|,{1,8}|\+{2,8}|-{2,8}|[^A-Za-z0-9_+\-\x80-\xFF])"#.into(),
        },
        VocabPartitionRule {
            partition: 0,
            name: "p0".into(),
            regex: r"[^A-Za-z0-9_+\-\x80-\xFF]+".into(),
        },
        VocabPartitionRule {
            partition: 1,
            name: "p1".into(),
            regex: r"[\x00-\xFF]*".into(),
        },
    ]
}

fn custom_vocab_partition_rules(value: &str) -> Result<Vec<VocabPartitionRule>, String> {
    let trimmed = value.trim();
    let after_open_bracket = trimmed
        .strip_prefix('[')
        .map(str::trim_start)
        .unwrap_or_default();
    let looks_like_json_array = trimmed.starts_with('[')
        && matches!(after_open_bracket.as_bytes().first(), Some(b'"' | b'{' | b']'));
    let mut rules = if looks_like_json_array {
        let parsed: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|error| format!("invalid {VOCAB_PARTITION_REGEX_ENV} JSON: {error}"))?;
        let items = parsed
            .as_array()
            .ok_or_else(|| format!("{VOCAB_PARTITION_REGEX_ENV} must be a JSON array"))?;
        let mut rules = Vec::with_capacity(items.len() + 1);
        for (index, item) in items.iter().enumerate() {
            match item {
                serde_json::Value::String(regex) => {
                    rules.push(VocabPartitionRule { partition: index as u8, name: format!("r{index}"), regex: regex.clone() });
                }
                serde_json::Value::Object(object) => {
                    let regex = object
                        .get("regex")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| {
                            format!(
                                "{VOCAB_PARTITION_REGEX_ENV}[{index}] object requires string field `regex`"
                            )
                        })?;
                    let name = object
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("r{index}"));
                    rules.push(VocabPartitionRule { partition: index as u8, name, regex: regex.to_owned() });
                }
                _ => {
                    return Err(format!(
                        "{VOCAB_PARTITION_REGEX_ENV}[{index}] must be a string or object"
                    ));
                }
            }
        }
        rules
    } else {
        trimmed
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .enumerate()
            .map(|(index, regex)| VocabPartitionRule { partition: index as u8, name: format!("r{index}"), regex: regex.to_owned() })
            .collect::<Vec<_>>()
    };
    if rules.len() >= 32 {
        return Err(format!(
            "{VOCAB_PARTITION_REGEX_ENV} supports at most 31 explicit rules; one slot is reserved for the implicit rest partition"
        ));
    }
    rules.push(VocabPartitionRule { partition: rules.len() as u8, name: "rest".into(), regex: r"(?s:.)*".into() });
    Ok(rules)
}

fn compile_vocab_partition_set(
    raw_rules: Vec<VocabPartitionRule>,
    custom: bool,
) -> Result<VocabPartitionSet, String> {
    if raw_rules.is_empty() || raw_rules.len() > 32 {
        return Err(format!(
            "vocabulary partition rule count must be in 1..=32, got {}",
            raw_rules.len()
        ));
    }

    let partition_count = raw_rules
        .iter()
        .map(|rule| rule.partition as usize)
        .max()
        .map_or(0, |max| max + 1);
    if partition_count != raw_rules.len() {
        return Err("vocabulary partition ids must be dense from zero".to_owned());
    }
    let mut languages = vec![None; partition_count];
    let mut matchers = Vec::with_capacity(raw_rules.len());
    let mut earlier = Vec::<String>::new();
    for rule in raw_rules {
        let language = VocabPartitionLanguage {
            name: rule.name,
            include_regex: rule.regex.clone(),
            excluded_regexes: earlier.clone(),
        };
        let dfa = VocabPartitionDfa::compile_byte_regex(&language.name, &language.include_regex)?;
        earlier.push(rule.regex);
        if languages[rule.partition as usize].replace(language).is_some() {
            return Err(format!("duplicate vocabulary partition id {}", rule.partition));
        }
        matchers.push((rule.partition, dfa));
    }
    let languages = languages
        .into_iter()
        .enumerate()
        .map(|(index, language)| language.ok_or_else(|| format!("missing vocabulary partition id {index}")))
        .collect::<Result<Vec<_>, _>>()?;
    let effective_dfas = (0..languages.len())
        .map(|_| std::sync::OnceLock::new())
        .collect();
    Ok(VocabPartitionSet {
        languages,
        matchers,
        effective_dfas,
        custom,
    })
}

fn build_vocab_partition_set() -> Result<VocabPartitionSet, String> {
    let custom_value = std::env::var(VOCAB_PARTITION_REGEX_ENV).ok();
    let custom = custom_value.is_some();
    let raw_rules = match custom_value {
        Some(value) => custom_vocab_partition_rules(&value)?,
        None => default_vocab_partition_rules(),
    };
    compile_vocab_partition_set(raw_rules, custom)
}

#[inline]
fn classify_with_partition_set(set: &VocabPartitionSet, bytes: &[u8]) -> Option<u8> {
    set.matchers
        .iter()
        .find_map(|(partition, dfa)| dfa.is_match(bytes).then_some(*partition))
}

fn vocab_partition_set() -> &'static VocabPartitionSet {
    static SET: std::sync::OnceLock<VocabPartitionSet> = std::sync::OnceLock::new();
    SET.get_or_init(|| {
        build_vocab_partition_set().unwrap_or_else(|error| {
            panic!("failed to initialize regex vocabulary partitions: {error}")
        })
    })
}

pub fn vocab_partition_count() -> usize {
    vocab_partition_set().languages.len()
}

pub fn vocab_partition_is_custom() -> bool {
    vocab_partition_set().custom
}

pub fn vocab_partition_language(index: usize) -> Option<VocabPartitionLanguage> {
    vocab_partition_set().languages.get(index).cloned()
}

pub fn vocab_partition_effective_dfa(index: usize) -> Option<Arc<VocabPartitionDfa>> {
    let set = vocab_partition_set();
    let language = set.languages.get(index)?;
    let cell = set.effective_dfas.get(index)?;
    Some(Arc::clone(cell.get_or_init(|| {
        Arc::new(language.compile_effective_dfa().unwrap_or_else(|error| {
            panic!("failed to compile effective vocabulary partition {}: {error}", language.name)
        }))
    })))
}

pub fn vocab_partition_label(index: usize) -> String {
    let set = vocab_partition_set();
    if set.custom {
        let name = set
            .languages
            .get(index)
            .map(|language| language.name.as_str())
            .unwrap_or("unknown");
        format!("rx{index}:{name}")
    } else {
        set.languages
            .get(index)
            .map(|language| language.name.clone())
            .unwrap_or_else(|| format!("p{index}"))
    }
}


/// Classifies each terminal by the longest token-path length it can participate in.
///
/// - **Length 0**: No vocab byte from any tokenizer state can lead towards
///   matching this terminal.  The terminal is ignorable.
/// - **Length 1**: The terminal is matchable but never co-occurs with another
///   terminal inside a single vocab token.
/// - **Length 2+**: Some vocabulary token has an exact split where this terminal
///   is one side of an allowed `(t1, t2)` boundary.  The prefix may begin at any
///   live residual lexer state for `t1`; the suffix is scanned from lexer reset
///   for `t2` and may either match within the token or remain completable at the
///   token end.
///
/// A cheap last-byte/first-byte overlap test is used only as a sound candidate
/// prefilter.  Exact boundary witnesses determine the final L1/L2P split.
pub fn classify_terminal_path_lengths(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
    shared_classify_cache: Option<&SharedClassifyCache>,
) -> Vec<TerminalPathLength> {
    classify_terminal_path_lengths_with_probe(
        partition_label,
        tokenizer,
        vocab,
        disallowed_follows,
        num_terminals,
        shared_classify_cache,
        None,
    )
}

fn heuristic_terminal_path_two_plus(
    vocab_facts: &VocabClassificationFacts,
    bytesets: &SharedClassifyBytesets,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
) -> BitSet {
    let nt = num_terminals as usize;
    let vocab_bytes = vocab_facts.bytes;
    let mut heuristic_two_plus = BitSet::new(nt);
    for t1 in 0..nt {
        if bytesets.last_bytes[t1].is_disjoint(&vocab_bytes) {
            continue;
        }
        let mut observed_after_t1 = U8Set::empty();
        for last_byte in bytesets.last_bytes[t1].iter() {
            observed_after_t1 = observed_after_t1
                .union(&vocab_facts.observed_follow_bytes[last_byte as usize]);
        }
        if observed_after_t1.is_empty() {
            continue;
        }
        let disallowed = disallowed_follows.get(&(t1 as u32));
        for t2 in 0..nt {
            if bytesets.first_bytes[t2].is_disjoint(&observed_after_t1) {
                continue;
            }
            if disallowed.is_some_and(|d| d.contains(t2)) {
                continue;
            }
            heuristic_two_plus.set(t1);
            heuristic_two_plus.set(t2);
        }
    }
    heuristic_two_plus
}

/// Cheap sound superset of terminals that may participate in a within-token
/// terminal boundary. This performs only vocabulary byte-pair filtering; it
/// deliberately does not run the exact witness classifier.
pub fn candidate_terminal_path_two_plus(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
    shared_classify_cache: Option<&SharedClassifyCache>,
) -> Vec<bool> {
    let vocab_facts = vocab_classification_facts(vocab);
    let owned_bytesets: Option<SharedClassifyBytesets>;
    let bytesets: &SharedClassifyBytesets = if let Some(cache) = shared_classify_cache {
        cache.get_or_init(|| SharedClassifyBytesets::build(tokenizer, num_terminals))
    } else {
        owned_bytesets = Some(SharedClassifyBytesets::build(tokenizer, num_terminals));
        owned_bytesets.as_ref().unwrap()
    };
    let candidates = heuristic_terminal_path_two_plus(
        &vocab_facts,
        bytesets,
        disallowed_follows,
        num_terminals,
    );
    (0..num_terminals as usize)
        .map(|terminal| candidates.contains(terminal))
        .collect()
}

pub(crate) fn classify_terminal_path_lengths_with_probe(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
    shared_classify_cache: Option<&SharedClassifyCache>,
    witness_probe_callback: Option<&dyn Fn(&BitSet)>,
) -> Vec<TerminalPathLength> {
    let nt = num_terminals as usize;
    // 1. Vocab byte bitset: all bytes appearing in any vocab token.
    let vocab_facts = vocab_classification_facts(vocab);
    let vocab_bytes = vocab_facts.bytes;

    // 2. Byte bitsets per terminal — use cache if available.
    let owned_bytesets: Option<SharedClassifyBytesets>;
    let bytesets: &SharedClassifyBytesets = if let Some(cache) = shared_classify_cache {
        cache.get_or_init(|| SharedClassifyBytesets::build(tokenizer, num_terminals))
    } else {
        owned_bytesets = Some(SharedClassifyBytesets::build(tokenizer, num_terminals));
        owned_bytesets.as_ref().unwrap()
    };
    let reachable_bytes = &bytesets.reachable_bytes;
    // 3. Retain the former byte-set result as a diagnostic upper bound, then
    // classify length >= 2 from exact terminal-boundary witnesses. A terminal
    // participates in an L2P path iff it is either side of some allowed
    // terminal pair split inside one vocabulary token.
    // A real within-token terminal boundary necessarily consumes two adjacent
    // token bytes. Keep the cheap byte-pair relation as a sound candidate set;
    // exact boundary witnesses below determine the final classification.
    let heuristic_two_plus = heuristic_terminal_path_two_plus(
        &vocab_facts,
        bytesets,
        disallowed_follows,
        num_terminals,
    );
    let exact = exact_terminal_path_two_plus_finite_literals(
        tokenizer,
        vocab,
        disallowed_follows,
        &heuristic_two_plus,
    )
    .or_else(|| {
        let enabled = std::env::var("GLRMASK_CLASSIFY_FINITE_FOLLOWER_EMPTY")
            .map(|value| {
                let value = value.trim();
                value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(false);
        (enabled
            && finite_follower_suffix_empty_certificate(
                tokenizer,
                vocab,
                disallowed_follows,
                &heuristic_two_plus,
                bytesets,
            ) == Some(true))
        .then(|| ExactTerminalPathTwoPlus {
            two_plus: BitSet::new(heuristic_two_plus.len()),
            witnesses: vec![None; heuristic_two_plus.len()],
        })
    })
    .unwrap_or_else(|| {
        // This path propagates reset-state continuations after every discovered
        // terminal boundary. The older direct prefix/suffix scanners only saw
        // boundaries whose left terminal began at token offset zero, so they
        // could miss later terminals in paths such as `U U U ... IDENTIFIER`.
        exact_terminal_path_two_plus_candidate_dfa(
            tokenizer,
            vocab,
            disallowed_follows,
            &heuristic_two_plus,
            bytesets,
            witness_probe_callback,
        )
    });
    if std::env::var_os("GLRMASK_TERMINAL_PATH_STRICT_REFERENCE").is_some() {
        let reference = exact_terminal_path_two_plus(
            tokenizer,
            vocab,
            disallowed_follows,
            bytesets,
            &heuristic_two_plus,
        );
        assert!(
            reference.two_plus.is_subset(&exact.two_plus),
            "candidate-DFA terminal-path classification missed a direct boundary from the generic exact reference in partition {partition_label}: exact={:?} reference={:?}",
            exact.two_plus,
            reference.two_plus,
        );
    }
    if std::env::var("GLRMASK_DUMP_TERMINAL_PATH_WITNESSES")
        .ok()
        .is_some_and(|requested| requested == "1" || requested == partition_label)
    {
        for terminal in heuristic_two_plus.iter() {
            let witness = exact.witnesses[terminal].as_ref();
            eprintln!(
                "[glrmask/dump][terminal_path_witness] partition={} terminal={} expr={:?} heuristic_two_plus=1 exact_two_plus={} witness={:?}",
                partition_label,
                terminal,
                tokenizer.terminal_expr(terminal as u32),
                usize::from(exact.two_plus.contains(terminal)),
                witness,
            );
        }
    }

    // 4. Final classification.
    let mut result = vec![TerminalPathLength::Zero; nt];
    for t in 0..nt {
        if reachable_bytes[t].is_disjoint(&vocab_bytes) {
            result[t] = TerminalPathLength::Zero;
        } else if exact.two_plus.contains(t) {
            result[t] = TerminalPathLength::TwoPlus;
        } else {
            result[t] = TerminalPathLength::One;
        }
    }

    result
}

fn build_byte_terminal_reverse_index(
    bytesets: &SharedClassifyBytesets,
    num_terminals: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut byte_to_last: Vec<Vec<usize>> = vec![Vec::new(); 256];
    let mut byte_to_first: Vec<Vec<usize>> = vec![Vec::new(); 256];
    for terminal in 0..num_terminals {
        for byte in 0u8..=255 {
            if bytesets.last_bytes[terminal].contains(byte) {
                byte_to_last[byte as usize].push(terminal);
            }
            if bytesets.first_bytes[terminal].contains(byte) {
                byte_to_first[byte as usize].push(terminal);
            }
        }
    }
    (byte_to_last, byte_to_first)
}

fn token_l2p_terminals(
    bytes: &[u8],
    byte_to_last: &[Vec<usize>],
    byte_to_first: &[Vec<usize>],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: usize,
) -> BitSet {
    let mut seen = [false; 256];
    let mut last_set = BitSet::new(num_terminals);
    let mut first_set = BitSet::new(num_terminals);

    for &byte in bytes {
        if !seen[byte as usize] {
            seen[byte as usize] = true;
            for &terminal in &byte_to_last[byte as usize] {
                last_set.set(terminal);
            }
            for &terminal in &byte_to_first[byte as usize] {
                first_set.set(terminal);
            }
        }
    }

    let mut l2p_set = BitSet::new(num_terminals);
    for terminal_1 in last_set.iter() {
        let disallowed = disallowed_follows.get(&(terminal_1 as u32));
        for terminal_2 in first_set.iter() {
            if !disallowed.map_or(false, |blocked| blocked.contains(terminal_2)) {
                l2p_set.set(terminal_1);
                l2p_set.set(terminal_2);
            }
        }
    }

    l2p_set
}

fn token_has_active_l2p_boundary(bytes: &[u8], allowed_boundary_pairs: &[U8Set; 256]) -> bool {
    bytes
        .windows(2)
        .any(|pair| allowed_boundary_pairs[pair[0] as usize].contains(pair[1]))
}

fn token_has_active_l2p_boundary_words(
    bytes: &[u8],
    allowed_boundary_pair_words: &[u64; 1024],
) -> bool {
    bytes.windows(2).any(|pair| {
        let pair_index = ((pair[0] as usize) << 8) | pair[1] as usize;
        (allowed_boundary_pair_words[pair_index >> 6] & (1u64 << (pair_index & 63))) != 0
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TokenL2pRouteHint {
    Adjacent,
    Single,
    Irrelevant,
}

/// Classify the cheap, byte-level L2P route hint in one token pass.
///
/// The route scan formerly searched every token once for an allowed adjacent
/// pair and, when none was found, searched it again for an active reachable
/// byte. The adjacency result has precedence, so retaining the first byte while
/// tracking reachability produces exactly the same split with one traversal.
#[inline]
fn token_l2p_route_hint(
    bytes: &[u8],
    allowed_boundary_pair_words: &[u64; 1024],
    active_reachable_by_byte: &[u8; 256],
) -> TokenL2pRouteHint {
    let Some((&first, rest)) = bytes.split_first() else {
        return TokenL2pRouteHint::Irrelevant;
    };
    let mut previous = first;
    let mut reaches_active = active_reachable_by_byte[first as usize] != 0;
    for &byte in rest {
        let pair_index = ((previous as usize) << 8) | byte as usize;
        if (allowed_boundary_pair_words[pair_index >> 6] & (1u64 << (pair_index & 63)))
            != 0
        {
            return TokenL2pRouteHint::Adjacent;
        }
        reaches_active |= active_reachable_by_byte[byte as usize] != 0;
        previous = byte;
    }
    if reaches_active {
        TokenL2pRouteHint::Single
    } else {
        TokenL2pRouteHint::Irrelevant
    }
}

#[inline]
fn bitset_intersects_words(bitset: &BitSet, words: &[u64]) -> bool {
    bitset
        .words()
        .iter()
        .zip(words)
        .any(|(lhs, rhs)| (*lhs & *rhs) != 0)
}

#[inline]
fn state_future_intersects_words(
    bytesets: &SharedClassifyBytesets,
    state: u32,
    active_words: &[u64],
) -> bool {
    let offset = state as usize * bytesets.words_per_terminal_set;
    bytesets.future_by_state_words[offset..offset + bytesets.words_per_terminal_set]
        .iter()
        .zip(active_words)
        .any(|(state_word, active_word)| (*state_word & *active_word) != 0)
}

fn build_active_matched_by_state(
    bytesets: &SharedClassifyBytesets,
    active_bitset: &BitSet,
) -> Box<[u8]> {
    let mut matched = vec![
        0u8;
        bytesets.future_by_state_words.len() / bytesets.words_per_terminal_set
    ];
    for terminal in active_bitset.iter() {
        for &state in &bytesets.matched_states_by_terminal[terminal] {
            matched[state as usize] = 1;
        }
    }
    matched.into()
}

fn build_active_suffix_start_by_byte(
    tokenizer: &Tokenizer,
    bytesets: &SharedClassifyBytesets,
    active_words: &[u64],
) -> Box<[u8; 256]> {
    let mut can_start = Box::new([0u8; 256]);
    let initial = tokenizer.initial_state_id();
    if !state_future_intersects_words(bytesets, initial, active_words) {
        return can_start;
    }
    for byte in 0u8..=u8::MAX {
        let next = bytesets.transitions_by_byte
            [byte as usize * tokenizer.num_states() as usize + initial as usize];
        if next == u32::MAX {
            continue;
        }
        if state_future_intersects_words(bytesets, next, active_words)
            || bitset_intersects_words(tokenizer.matched_terminal_bitset(next), active_words)
        {
            can_start[byte as usize] = 1;
        }
    }
    can_start
}

#[inline]
fn suffix_can_reach_active_terminal(
    tokenizer: &Tokenizer,
    bytesets: &SharedClassifyBytesets,
    flat_trans: &[u32],
    suffix: &[u8],
    active_words: &[u64],
    active_matched_by_state: Option<&[u8]>,
) -> bool {
    let mut state = tokenizer.initial_state_id();
    for &byte in suffix {
        if !state_future_intersects_words(bytesets, state, active_words) {
            return false;
        }
        let next = flat_trans[state as usize * 256 + byte as usize];
        if next == u32::MAX {
            return false;
        }
        state = next;
        if active_matched_by_state.map_or_else(
            || bitset_intersects_words(tokenizer.matched_terminal_bitset(state), active_words),
            |matched| matched[state as usize] != 0,
        ) {
            return true;
        }
    }
    state_future_intersects_words(bytesets, state, active_words)
}

fn token_has_active_terminal_suffix(
    tokenizer: &Tokenizer,
    bytesets: &SharedClassifyBytesets,
    flat_trans: &[u32],
    bytes: &[u8],
    active_words: &[u64],
    active_matched_by_state: Option<&[u8]>,
    active_suffix_start_by_byte: &[u8; 256],
) -> bool {
    (1..bytes.len()).any(|suffix_start| {
        active_suffix_start_by_byte[bytes[suffix_start] as usize] != 0
            &&
        suffix_can_reach_active_terminal(
            tokenizer,
            bytesets,
            flat_trans,
            &bytes[suffix_start..],
            active_words,
            active_matched_by_state,
        )
    })
}

fn active_l2p_route_setup(
    tokenizer: &Tokenizer,
    bytesets: &SharedClassifyBytesets,
    active_bitset: &BitSet,
    disallowed_follows: &Arc<BTreeMap<u32, BitSet>>,
) -> Arc<ActiveL2pRouteSetup> {
    let cache_key = (
        active_bitset.clone(),
        Arc::as_ptr(disallowed_follows) as usize,
    );
    if let Some(cached) = bytesets
        .active_route_setup_cache
        .lock()
        .unwrap()
        .get(&cache_key)
        .cloned()
    {
        return cached;
    }

    let setup_started_at = std::time::Instant::now();
    let active_words = active_bitset.words();
    let active_count = active_bitset.count_ones();
    let use_representative_fast_path = active_count >= active_bitset.len().div_ceil(2);
    let mut active_start_states = Vec::new();
    let mut start_full_checks = 0usize;
    for state in 0..tokenizer.num_states() {
        if use_representative_fast_path {
            let representative = bytesets.representative_future_terminal_by_state[state as usize];
            if representative != u32::MAX && active_bitset.contains(representative as usize) {
                active_start_states.push(state);
                continue;
            }
        }
        {
            start_full_checks += 1;
            if state_future_intersects_words(bytesets, state, active_words) {
                active_start_states.push(state);
            }
        }
    }
    let start_states_ms = setup_started_at.elapsed().as_secs_f64() * 1000.0;
    let boundary_pairs_started_at = std::time::Instant::now();
    let mut allowed_boundary_pairs = Box::new([U8Set::empty(); 256]);
    let mut active_reachable = U8Set::empty();
    let mut active_first_bytes = U8Set::empty();
    let mut unrestricted_last_bytes = U8Set::empty();
    let mut allowed_first_bytes_cache = HashMap::<&BitSet, U8Set>::new();

    for terminal in active_bitset.iter() {
        active_reachable = active_reachable.union(&bytesets.reachable_bytes[terminal]);
        active_first_bytes = active_first_bytes.union(&bytesets.first_bytes[terminal]);
    }
    let mut active_reachable_by_byte = Box::new([0u8; 256]);
    for byte in active_reachable.iter() {
        active_reachable_by_byte[byte as usize] = 1;
    }
    let active_suffix_start_by_byte =
        build_active_suffix_start_by_byte(tokenizer, bytesets, active_words);
    for terminal_1 in active_bitset.iter() {
        let last_bytes = bytesets.last_bytes[terminal_1];
        if last_bytes.is_empty() {
            continue;
        }
        let Some(blocked) = disallowed_follows
            .get(&(terminal_1 as u32))
            .filter(|blocked| !blocked.is_empty())
        else {
            unrestricted_last_bytes = unrestricted_last_bytes.union(&last_bytes);
            continue;
        };
        let allowed_first_bytes = *allowed_first_bytes_cache.entry(blocked).or_insert_with(|| {
            let mut allowed = U8Set::empty();
            for terminal_2 in active_bitset.iter() {
                if !blocked.contains(terminal_2) {
                    allowed = allowed.union(&bytesets.first_bytes[terminal_2]);
                }
            }
            allowed
        });
        for last_byte in last_bytes.iter() {
            allowed_boundary_pairs[last_byte as usize] =
                allowed_boundary_pairs[last_byte as usize].union(&allowed_first_bytes);
        }
    }
    for last_byte in unrestricted_last_bytes.iter() {
        allowed_boundary_pairs[last_byte as usize] =
            allowed_boundary_pairs[last_byte as usize].union(&active_first_bytes);
    }
    let mut allowed_boundary_pair_words = Box::new([0u64; 1024]);
    for (last_byte, first_bytes) in allowed_boundary_pairs.iter().enumerate() {
        for first_byte in first_bytes.iter() {
            let pair_index = (last_byte << 8) | first_byte as usize;
            allowed_boundary_pair_words[pair_index >> 6] |= 1u64 << (pair_index & 63);
        }
    }

    let setup = Arc::new(ActiveL2pRouteSetup {
        _follow_owner: Arc::clone(disallowed_follows),
        active_start_states: active_start_states.into(),
        allowed_boundary_pairs,
        allowed_boundary_pair_words,
        active_reachable_by_byte,
        active_suffix_start_by_byte,
    });
    if super::types::compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l2p_route_setup] active={} starts={} start_full_checks={} start_states_ms={:.3} boundary_pairs_ms={:.3} total_ms={:.3}",
            active_bitset.count_ones(),
            setup.active_start_states.len(),
            start_full_checks,
            start_states_ms,
            boundary_pairs_started_at.elapsed().as_secs_f64() * 1000.0,
            setup_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let mut cache = bytesets.active_route_setup_cache.lock().unwrap();
    Arc::clone(cache.entry(cache_key).or_insert(setup))
}

#[derive(Clone, Copy, Debug)]
enum ExactL2pBoundaryFilterMode {
    Auto,
    Force(bool),
}

fn parse_exact_l2p_boundary_filter_mode(value: &str) -> ExactL2pBoundaryFilterMode {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => ExactL2pBoundaryFilterMode::Auto,
        "1" | "true" | "yes" | "on" => ExactL2pBoundaryFilterMode::Force(true),
        "0" | "false" | "no" | "off" => ExactL2pBoundaryFilterMode::Force(false),
        other => panic!(
            "invalid GLRMASK_EXACT_L2P_BOUNDARY_FILTER value `{other}`; expected auto, 1/true/on, or 0/false/off"
        ),
    }
}

fn exact_l2p_boundary_filter_mode() -> ExactL2pBoundaryFilterMode {
    static MODE: std::sync::OnceLock<ExactL2pBoundaryFilterMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        std::env::var("GLRMASK_EXACT_L2P_BOUNDARY_FILTER")
            .ok()
            .map(|value| parse_exact_l2p_boundary_filter_mode(&value))
            .unwrap_or(ExactL2pBoundaryFilterMode::Auto)
    })
}

fn exact_l2p_boundary_filter_work_limit() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("GLRMASK_EXACT_L2P_BOUNDARY_FILTER_WORK_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(5_000_000)
    })
}

fn exact_l2p_boundary_filter_min_candidates() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("GLRMASK_EXACT_L2P_BOUNDARY_FILTER_MIN_CANDIDATES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(5_000)
    })
}

fn suffix_has_allowed_l2p_follow_from_reset(
    tokenizer: &Tokenizer,
    suffix: &[u8],
    allowed_after: &BitSet,
) -> bool {
    if allowed_after.is_empty() {
        return false;
    }

    let mut states =
        tokenizer.execute_from_state_end_only(&[], tokenizer.initial_state_id());
    for &byte in suffix {
        if states.iter().all(|&state| {
            tokenizer
                .possible_future_terminals(state)
                .is_disjoint(allowed_after)
        }) {
            return false;
        }
        states = tokenizer.step_all(&states, byte);
        if states.is_empty() {
            return false;
        }
        if states.iter().any(|&state| {
            tokenizer
                .matched_terminals_iter(state)
                .any(|terminal| allowed_after.contains(terminal as usize))
        }) {
            return true;
        }
    }

    states.iter().any(|&state| {
        !tokenizer
            .possible_future_terminals(state)
            .is_disjoint(allowed_after)
    })
}

fn suffix_has_allowed_l2p_follow(
    tokenizer: &Tokenizer,
    terminal_1: usize,
    suffix: &[u8],
    active_bitset: &BitSet,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> bool {
    let allowed_after = disallowed_follows
        .get(&(terminal_1 as u32))
        .map_or_else(|| active_bitset.clone(), |blocked| active_bitset.difference(blocked));
    suffix_has_allowed_l2p_follow_from_reset(tokenizer, suffix, &allowed_after)
}

fn token_has_exact_active_l2p_boundary(
    tokenizer: &Tokenizer,
    bytes: &[u8],
    active_bitset: &BitSet,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    active_start_states: &[u32],
) -> bool {
    if bytes.len() < 2 {
        return false;
    }

    let mut current_states = active_start_states
        .iter()
        .flat_map(|&state| tokenizer.execute_from_state_end_only(&[], state))
        .collect::<Vec<_>>();
    current_states.sort_unstable();
    current_states.dedup();
    let mut suffix_cache = HashMap::<(usize, usize), bool>::new();

    for split_after in 0..bytes.len() - 1 {
        let next_states = tokenizer.step_all(&current_states, bytes[split_after]);

        if next_states.is_empty() {
            return false;
        }

        for &state in &next_states {
            for terminal_1 in tokenizer.matched_terminals_iter(state) {
                let terminal_1 = terminal_1 as usize;
                if !active_bitset.contains(terminal_1) {
                    continue;
                }
                let suffix_start = split_after + 1;
                let has_follow = *suffix_cache.entry((terminal_1, suffix_start)).or_insert_with(|| {
                    suffix_has_allowed_l2p_follow(
                        tokenizer,
                        terminal_1,
                        &bytes[suffix_start..],
                        active_bitset,
                        disallowed_follows,
                    )
                });
                if has_follow {
                    return true;
                }
            }
        }

        current_states = next_states.to_vec();
    }

    false
}

trait ExactBoundaryDeterministicScanner {
    fn num_states(&self) -> usize;
    fn start_state(&self) -> u32;
    fn transition(&self, state: u32, byte: u8) -> u32;
    fn has_finalizers(&self, state: u32) -> bool;
    fn for_each_finalizer(&self, state: u32, visit: impl FnMut(usize));
    fn for_each_future(&self, state: u32, visit: impl FnMut(usize));
    fn finalizers_intersect(&self, state: u32, terminals: &BitSet) -> bool;
    fn futures_intersect(&self, state: u32, terminals: &BitSet) -> bool;
}

struct RawExactBoundaryScanner<'a> {
    tokenizer: &'a Tokenizer,
    flat_trans: &'a [u32],
}

impl ExactBoundaryDeterministicScanner for RawExactBoundaryScanner<'_> {
    #[inline]
    fn num_states(&self) -> usize {
        self.tokenizer.num_states() as usize
    }

    #[inline]
    fn start_state(&self) -> u32 {
        self.tokenizer.initial_state_id()
    }

    #[inline]
    fn transition(&self, state: u32, byte: u8) -> u32 {
        self.flat_trans[state as usize * 256 + byte as usize]
    }

    #[inline]
    fn has_finalizers(&self, state: u32) -> bool {
        !self.tokenizer.matched_terminal_bitset(state).is_empty()
    }

    #[inline]
    fn for_each_finalizer(&self, state: u32, mut visit: impl FnMut(usize)) {
        for terminal in self.tokenizer.matched_terminals_iter(state) {
            visit(terminal as usize);
        }
    }

    #[inline]
    fn for_each_future(&self, state: u32, mut visit: impl FnMut(usize)) {
        for terminal in self.tokenizer.possible_future_terminals_iter(state) {
            visit(terminal as usize);
        }
    }

    #[inline]
    fn finalizers_intersect(&self, state: u32, terminals: &BitSet) -> bool {
        !self.tokenizer.matched_terminal_bitset(state).is_disjoint(terminals)
    }

    #[inline]
    fn futures_intersect(&self, state: u32, terminals: &BitSet) -> bool {
        !self.tokenizer.possible_future_terminals(state).is_disjoint(terminals)
    }
}

struct FlatExactBoundaryScanner<'a> {
    dfa: &'a FlatDfa,
}

impl ExactBoundaryDeterministicScanner for FlatExactBoundaryScanner<'_> {
    #[inline]
    fn num_states(&self) -> usize {
        self.dfa.states.len()
    }

    #[inline]
    fn start_state(&self) -> u32 {
        self.dfa.start_state as u32
    }

    #[inline]
    fn transition(&self, state: u32, byte: u8) -> u32 {
        self.dfa.trans(state as usize, byte as usize)
    }

    #[inline]
    fn has_finalizers(&self, state: u32) -> bool {
        !self.dfa.states[state as usize].finalizers.is_empty()
    }

    #[inline]
    fn for_each_finalizer(&self, state: u32, mut visit: impl FnMut(usize)) {
        for &terminal in &self.dfa.states[state as usize].finalizers {
            visit(terminal);
        }
    }

    #[inline]
    fn for_each_future(&self, state: u32, mut visit: impl FnMut(usize)) {
        for &terminal in &self.dfa.states[state as usize].possible_future_group_ids {
            visit(terminal);
        }
    }

    #[inline]
    fn finalizers_intersect(&self, state: u32, terminals: &BitSet) -> bool {
        self.dfa.states[state as usize]
            .finalizers
            .iter()
            .any(|&terminal| terminals.contains(terminal))
    }

    #[inline]
    fn futures_intersect(&self, state: u32, terminals: &BitSet) -> bool {
        self.dfa.states[state as usize]
            .possible_future_group_ids
            .iter()
            .any(|&terminal| terminals.contains(terminal))
    }
}

fn build_exact_boundary_transition_index(
    state_major_transitions: &[u32],
    num_states: usize,
) -> (Vec<u32>, Vec<Vec<(u32, u32)>>, Vec<ReverseByteTransitions>) {
    assert_eq!(state_major_transitions.len(), num_states * 256);
    let mut transitions_by_byte = vec![u32::MAX; num_states * 256];
    let mut sparse_transitions_by_byte = vec![Vec::<(u32, u32)>::new(); 256];
    for state in 0..num_states {
        let state_base = state * 256;
        for byte in 0..256usize {
            let target = state_major_transitions[state_base + byte];
            if target == u32::MAX {
                continue;
            }
            transitions_by_byte[byte * num_states + state] = target;
            sparse_transitions_by_byte[byte].push((state as u32, target));
        }
    }
    let reverse_transitions_by_byte =
        build_reverse_transitions_by_byte(&sparse_transitions_by_byte, num_states);
    (
        transitions_by_byte,
        sparse_transitions_by_byte,
        reverse_transitions_by_byte,
    )
}

#[derive(Default)]
struct ExactBoundaryPrefixNode {
    children: Vec<(u8, usize)>,
    allowed_follow_terminals: Option<BitSet>,
    matched_terminals: Vec<usize>,
}

fn insert_exact_boundary_prefix(
    nodes: &mut Vec<ExactBoundaryPrefixNode>,
    parent: usize,
    byte: u8,
) -> usize {
    if let Some((_, child)) = nodes[parent]
        .children
        .iter()
        .find(|(candidate, _)| *candidate == byte)
    {
        return *child;
    }

    let child = nodes.len();
    nodes.push(ExactBoundaryPrefixNode::default());
    nodes[parent].children.push((byte, child));
    child
}

fn populate_exact_boundary_prefixes<S: ExactBoundaryDeterministicScanner>(
    scanner: &S,
    transitions_by_byte: &[u32],
    sparse_transitions_by_byte: &[Vec<(u32, u32)>],
    reverse_transitions_by_byte: &[ReverseByteTransitions],
    active_bitset: &BitSet,
    nodes: &mut [ExactBoundaryPrefixNode],
    node: usize,
    current_states: &[u32],
    depth: usize,
    state_seen: &mut [u32],
    state_stamp: &mut u32,
    terminal_seen: &mut [u32],
    terminal_stamp: &mut u32,
    class_seen: &mut [u32],
    class_stamp: &mut u32,
    states_scanned: &mut usize,
    reached_states: &mut usize,
    finalizer_terminals_scanned: &mut usize,
    allowed_class_by_terminal: &[Option<usize>],
    allowed_follow_classes: &[BitSet],
    matched_classes: &mut Vec<usize>,
    state_buffers: &mut Vec<Vec<u32>>,
    source_seen_by_depth: &mut Vec<Vec<u32>>,
    source_stamps: &mut Vec<u32>,
) {
    let num_states = scanner.num_states();
    let use_state_major_transitions = transitions_by_byte.is_empty();
    let child_count = nodes[node].children.len();
    let frontier_is_dense = current_states.len() * 4 >= num_states;
    let source_stamp = if !use_state_major_transitions
        && nodes[node]
        .children
        .iter()
        .any(|(byte, _)| {
            let sparse = &sparse_transitions_by_byte[*byte as usize];
            let reverse = &reverse_transitions_by_byte[*byte as usize];
            sparse.len() < current_states.len()
                || (frontier_is_dense && reverse.targets.len() * 2 < sparse.len())
        })
    {
        if source_seen_by_depth.len() <= depth {
            source_seen_by_depth.resize_with(depth + 1, || vec![0u32; num_states]);
            source_stamps.resize(depth + 1, 0);
        }
        let source_seen = &mut source_seen_by_depth[depth];
        let source_stamp = &mut source_stamps[depth];
        *source_stamp = source_stamp.wrapping_add(1);
        if *source_stamp == 0 {
            source_seen.fill(0);
            *source_stamp = 1;
        }
        for &state in current_states {
            source_seen[state as usize] = *source_stamp;
        }
        Some(*source_stamp)
    } else {
        None
    };
    for child_index in 0..child_count {
        let (byte, child) = nodes[node].children[child_index];
        if state_buffers.len() <= depth {
            state_buffers.resize_with(depth + 1, Vec::new);
        }
        let mut next_states = std::mem::take(&mut state_buffers[depth]);
        next_states.clear();
        if next_states.capacity() < current_states.len() {
            next_states.reserve(current_states.len() - next_states.capacity());
        }
        if use_state_major_transitions {
            *state_stamp = state_stamp.wrapping_add(1);
            if *state_stamp == 0 {
                state_seen.fill(0);
                *state_stamp = 1;
            }
            *states_scanned += current_states.len();
            for &state in current_states {
                let next = scanner.transition(state, byte);
                if next == u32::MAX {
                    continue;
                }
                let slot = &mut state_seen[next as usize];
                if *slot == *state_stamp {
                    continue;
                }
                *slot = *state_stamp;
                next_states.push(next);
            }
        } else {
        let sparse_transitions = &sparse_transitions_by_byte[byte as usize];
        let reverse_transitions = &reverse_transitions_by_byte[byte as usize];
        let use_reverse_transitions = source_stamp.is_some()
            && frontier_is_dense
            && reverse_transitions.targets.len() * 2 < sparse_transitions.len();
        if use_reverse_transitions {
            let source_stamp = source_stamp.unwrap();
            let source_seen = &source_seen_by_depth[depth];
            for (target_index, &target) in reverse_transitions.targets.iter().enumerate() {
                let source_start = reverse_transitions.source_offsets[target_index] as usize;
                let source_end = reverse_transitions.source_offsets[target_index + 1] as usize;
                for &source in &reverse_transitions.sources[source_start..source_end] {
                    *states_scanned += 1;
                    if source_seen[source as usize] == source_stamp {
                        next_states.push(target);
                        break;
                    }
                }
            }
        } else {
            *state_stamp = state_stamp.wrapping_add(1);
            if *state_stamp == 0 {
                state_seen.fill(0);
                *state_stamp = 1;
            }
            if source_stamp.is_some() && sparse_transitions.len() < current_states.len() {
            *states_scanned += sparse_transitions.len();
            let source_stamp = source_stamp.unwrap();
            let source_seen = &source_seen_by_depth[depth];
            for &(source, next) in sparse_transitions {
                if source_seen[source as usize] != source_stamp {
                    continue;
                }
                let slot = &mut state_seen[next as usize];
                if *slot == *state_stamp {
                    continue;
                }
                *slot = *state_stamp;
                next_states.push(next);
            }
            } else {
            *states_scanned += current_states.len();
            let transition_column =
                &transitions_by_byte[byte as usize * num_states..(byte as usize + 1) * num_states];
            for &state in current_states {
                let next = transition_column[state as usize];
                if next == u32::MAX {
                    continue;
                }
                let slot = &mut state_seen[next as usize];
                if *slot == *state_stamp {
                    continue;
                }
                *slot = *state_stamp;
                next_states.push(next);
                }
            }
        }
        }
        *reached_states += next_states.len();

        *terminal_stamp = terminal_stamp.wrapping_add(1);
        if *terminal_stamp == 0 {
            terminal_seen.fill(0);
            *terminal_stamp = 1;
        }
        *class_stamp = class_stamp.wrapping_add(1);
        if *class_stamp == 0 {
            class_seen.fill(0);
            *class_stamp = 1;
        }
        matched_classes.clear();
        for &state in &next_states {
            if !scanner.has_finalizers(state) {
                continue;
            }
            scanner.for_each_finalizer(state, |terminal| {
                *finalizer_terminals_scanned += 1;
                if !active_bitset.contains(terminal)
                    || terminal_seen[terminal] == *terminal_stamp
                {
                    return;
                }
                terminal_seen[terminal] = *terminal_stamp;
                nodes[child].matched_terminals.push(terminal);
                if let Some(class) = allowed_class_by_terminal[terminal] {
                    if class_seen[class] != *class_stamp {
                        class_seen[class] = *class_stamp;
                        matched_classes.push(class);
                    }
                }
            });
        }
        let mut allowed_follow_terminals = None;
        for &class in matched_classes.iter() {
            allowed_follow_terminals
                .get_or_insert_with(|| BitSet::new(active_bitset.len()))
                .union_with(&allowed_follow_classes[class]);
        }
        nodes[child].allowed_follow_terminals = allowed_follow_terminals;

        if !next_states.is_empty() {
            populate_exact_boundary_prefixes(
                scanner,
                transitions_by_byte,
                sparse_transitions_by_byte,
                reverse_transitions_by_byte,
                active_bitset,
                nodes,
                child,
                &next_states,
                depth + 1,
                state_seen,
                state_stamp,
                terminal_seen,
                terminal_stamp,
                class_seen,
                class_stamp,
                states_scanned,
                reached_states,
                finalizer_terminals_scanned,
                allowed_class_by_terminal,
                allowed_follow_classes,
                matched_classes,
                state_buffers,
                source_seen_by_depth,
                source_stamps,
            );
        }
        next_states.clear();
        state_buffers[depth] = next_states;
    }
}

fn suffix_has_allowed_l2p_follow_deterministic<S: ExactBoundaryDeterministicScanner>(
    scanner: &S,
    suffix: &[u8],
    allowed: &BitSet,
) -> bool {
    let mut state = scanner.start_state();
    let mut consumed_suffix = true;
    for &byte in suffix {
        if !scanner.futures_intersect(state, allowed) {
            consumed_suffix = false;
            break;
        }
        let next = scanner.transition(state, byte);
        if next == u32::MAX {
            consumed_suffix = false;
            break;
        }
        state = next;
        if scanner.finalizers_intersect(state, allowed) {
            return true;
        }
    }
    consumed_suffix && scanner.futures_intersect(state, allowed)
}

fn tokens_have_exact_active_l2p_boundary_deterministic<
    S: ExactBoundaryDeterministicScanner,
>(
    scanner: &S,
    transitions_by_byte: &[u32],
    sparse_transitions_by_byte: &[Vec<(u32, u32)>],
    reverse_transitions_by_byte: &[ReverseByteTransitions],
    tokens: &[&[u8]],
    active_bitset: &BitSet,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    active_start_states: &[u32],
) -> Vec<bool> {
    let total_started_at = std::time::Instant::now();
    let trie_started_at = std::time::Instant::now();
    let mut nodes = vec![ExactBoundaryPrefixNode::default()];
    let total_token_bytes = tokens
        .iter()
        .map(|bytes| bytes.len().saturating_sub(1))
        .sum();
    let mut token_path_offsets = Vec::with_capacity(tokens.len() + 1);
    let mut token_paths = Vec::with_capacity(total_token_bytes);
    token_path_offsets.push(0);
    for &bytes in tokens {
        let mut node = 0usize;
        for &byte in bytes.iter().take(bytes.len().saturating_sub(1)) {
            node = insert_exact_boundary_prefix(&mut nodes, node, byte);
            token_paths.push(node);
        }
        token_path_offsets.push(token_paths.len());
    }
    let trie_ms = trie_started_at.elapsed().as_secs_f64() * 1000.0;

    let populate_started_at = std::time::Instant::now();
    let mut state_seen = vec![0u32; scanner.num_states()];
    let mut terminal_seen = vec![0u32; active_bitset.len()];
    let mut state_stamp = 0u32;
    let mut terminal_stamp = 0u32;
    let mut class_stamp = 0u32;
    let mut states_scanned = 0usize;
    let mut reached_states = 0usize;
    let mut finalizer_terminals_scanned = 0usize;
    let mut allowed_follow_classes = vec![active_bitset.clone()];
    let mut blocked_class_cache = HashMap::<&BitSet, usize>::new();
    let allowed_class_by_terminal = (0..active_bitset.len())
        .map(|terminal| {
            if !active_bitset.contains(terminal) {
                return None;
            }
            let Some(blocked) = disallowed_follows
                .get(&(terminal as u32))
                .filter(|blocked| !blocked.is_empty())
            else {
                return Some(0);
            };
            Some(*blocked_class_cache.entry(blocked).or_insert_with(|| {
                let class = allowed_follow_classes.len();
                allowed_follow_classes.push(active_bitset.difference(blocked));
                class
            }))
        })
        .collect::<Vec<_>>();
    let mut class_seen = vec![0u32; allowed_follow_classes.len()];
    let mut matched_classes = Vec::with_capacity(allowed_follow_classes.len());
    let mut state_buffers = Vec::<Vec<u32>>::new();
    let mut source_seen_by_depth = Vec::<Vec<u32>>::new();
    let mut source_stamps = Vec::<u32>::new();
    populate_exact_boundary_prefixes(
        scanner,
        transitions_by_byte,
        sparse_transitions_by_byte,
        reverse_transitions_by_byte,
        active_bitset,
        &mut nodes,
        0,
        active_start_states,
        0,
        &mut state_seen,
        &mut state_stamp,
        &mut terminal_seen,
        &mut terminal_stamp,
        &mut class_seen,
        &mut class_stamp,
        &mut states_scanned,
        &mut reached_states,
        &mut finalizer_terminals_scanned,
        &allowed_class_by_terminal,
        &allowed_follow_classes,
        &mut matched_classes,
        &mut state_buffers,
        &mut source_seen_by_depth,
        &mut source_stamps,
    );
    let populate_ms = populate_started_at.elapsed().as_secs_f64() * 1000.0;

    let evaluate_started_at = std::time::Instant::now();
    let mut prefix_allowed_checks = 0usize;
    let mut suffixes_evaluated = 0usize;
    let mut suffixes_with_terminals = 0usize;
    let result = tokens
        .iter()
        .enumerate()
        .map(|(token_index, &bytes)| {
            token_paths[token_path_offsets[token_index]..token_path_offsets[token_index + 1]]
                .iter()
                .copied()
                .enumerate()
                .any(|(split_after, node)| {
                    let Some(allowed) = nodes[node].allowed_follow_terminals.as_ref() else {
                        return false;
                    };
                    prefix_allowed_checks += 1;
                    suffixes_evaluated += 1;
                    let suffix_start = split_after + 1;
                    let suffix_has_allowed_terminal =
                        suffix_has_allowed_l2p_follow_deterministic(
                            scanner,
                            &bytes[suffix_start..],
                            allowed,
                        );
                    suffixes_with_terminals += usize::from(suffix_has_allowed_terminal);
                    suffix_has_allowed_terminal
                })
        })
        .collect();
    let evaluate_ms = evaluate_started_at.elapsed().as_secs_f64() * 1000.0;
    if super::types::compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][exact_boundary_batch] tokens={} scanner_states={} start_states={} nodes={} follow_classes={} states_scanned={} reached_states={} finalizer_terminals_scanned={} prefix_allowed_checks={} suffixes_evaluated={} suffixes_with_terminals={} trie_ms={:.3} populate_ms={:.3} evaluate_ms={:.3} total_ms={:.3}",
            tokens.len(),
            scanner.num_states(),
            active_start_states.len(),
            nodes.len(),
            allowed_follow_classes.len(),
            states_scanned,
            reached_states,
            finalizer_terminals_scanned,
            prefix_allowed_checks,
            suffixes_evaluated,
            suffixes_with_terminals,
            trie_ms,
            populate_ms,
            evaluate_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    result
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TerminalBoundaryWitness {
    token_id: u32,
    token_bytes: Vec<u8>,
    split_position: usize,
    terminal_1: usize,
    terminal_2: usize,
}

struct ExactTerminalPathTwoPlus {
    two_plus: BitSet,
    witnesses: Vec<Option<TerminalBoundaryWitness>>,
}

const FINITE_LITERAL_CLASSIFY_MAX_CANDIDATES: usize = 64;
const FINITE_LITERAL_CLASSIFY_MAX_VARIANTS: usize = 256;
const FINITE_LITERAL_CLASSIFY_MAX_TOTAL_BYTES: usize = 16 * 1024;

fn collect_finite_literal_strings(expr: &Expr) -> Option<Vec<Vec<u8>>> {
    fn collect(expr: &Expr, variants: &mut usize, total_bytes: &mut usize) -> Option<Vec<Vec<u8>>> {
        let mut account = |values: &Vec<Vec<u8>>| -> Option<()> {
            *variants = variants.checked_add(values.len())?;
            *total_bytes = total_bytes.checked_add(values.iter().map(Vec::len).sum::<usize>())?;
            (*variants <= FINITE_LITERAL_CLASSIFY_MAX_VARIANTS
                && *total_bytes <= FINITE_LITERAL_CLASSIFY_MAX_TOTAL_BYTES)
                .then_some(())
        };
        match expr {
            Expr::U8Seq(bytes) => {
                let values = vec![bytes.clone()];
                account(&values)?;
                Some(values)
            }
            Expr::Epsilon => {
                let values = vec![Vec::new()];
                account(&values)?;
                Some(values)
            }
            Expr::Shared(inner) => collect(inner, variants, total_bytes),
            Expr::Choice(options) => {
                let mut result = Vec::new();
                for option in options {
                    result.extend(collect(option, variants, total_bytes)?);
                    if result.len() > FINITE_LITERAL_CLASSIFY_MAX_VARIANTS {
                        return None;
                    }
                }
                result.sort_unstable();
                result.dedup();
                Some(result)
            }
            Expr::Seq(parts) => {
                let mut result = vec![Vec::new()];
                for part in parts {
                    let suffixes = collect(part, variants, total_bytes)?;
                    let product_len = result.len().checked_mul(suffixes.len())?;
                    if product_len > FINITE_LITERAL_CLASSIFY_MAX_VARIANTS {
                        return None;
                    }
                    let mut next = Vec::with_capacity(product_len);
                    for prefix in &result {
                        for suffix in &suffixes {
                            let mut bytes = Vec::with_capacity(prefix.len() + suffix.len());
                            bytes.extend_from_slice(prefix);
                            bytes.extend_from_slice(suffix);
                            next.push(bytes);
                        }
                    }
                    result = next;
                }
                result.sort_unstable();
                result.dedup();
                Some(result)
            }
            _ => None,
        }
    }

    let mut variants = 0usize;
    let mut total_bytes = 0usize;
    let mut values = collect(expr, &mut variants, &mut total_bytes)?;
    values.sort_unstable();
    values.dedup();
    (!values.is_empty() && values.iter().all(|value| !value.is_empty())).then_some(values)
}

/// Prove that no candidate terminal boundary can have a viable reset-side
/// follower using only finite literal follower languages.
///
/// This is deliberately an emptiness certificate, not a replacement for the
/// exact boundary classifier.  We over-approximate possible left terminals by
/// adjacent byte pairs, but require every directionally possible follower to
/// have a finite literal language.  For a finite literal `L`, a token suffix is
/// viable from lexer reset iff the suffix and some literal in `L` are
/// prefix-comparable: either the literal finishes inside the suffix or the
/// suffix finishes while the literal remains completable.  If no indexed
/// feasible split has such a follower, no exact within-token boundary exists.
fn finite_follower_suffix_empty_certificate(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    candidates: &BitSet,
    bytesets: &SharedClassifyBytesets,
) -> Option<bool> {
    let started_at = std::time::Instant::now();
    let candidate_ids = candidates.iter().collect::<Vec<_>>();
    if candidate_ids.is_empty() || candidate_ids.len() > 64 {
        return None;
    }

    let index = vocab_adjacent_pair_index(vocab);
    let mut follower_masks_by_pair = vec![0u64; 1 << 16];
    let mut follower_mask = 0u64;
    for (local_1, &terminal_1) in candidate_ids.iter().enumerate() {
        let blocked = disallowed_follows.get(&(terminal_1 as u32));
        for (local_2, &terminal_2) in candidate_ids.iter().enumerate() {
            if blocked.is_some_and(|blocked| blocked.contains(terminal_2)) {
                continue;
            }
            let follower_bit = 1u64 << local_2;
            for left in bytesets.last_bytes[terminal_1].iter() {
                for right in bytesets.first_bytes[terminal_2].iter() {
                    if index.occurrences_for_pair(left, right).is_empty() {
                        continue;
                    }
                    follower_masks_by_pair[((left as usize) << 8) | right as usize] |= follower_bit;
                    follower_mask |= follower_bit;
                }
            }
        }
        let _ = local_1;
    }
    if follower_mask == 0 {
        return Some(true);
    }

    let mut literals_by_local = vec![None::<Vec<Vec<u8>>>; candidate_ids.len()];
    let mut pending = follower_mask;
    while pending != 0 {
        let local = pending.trailing_zeros() as usize;
        pending &= pending - 1;
        let terminal = candidate_ids[local];
        let literals = collect_finite_literal_strings(tokenizer.terminal_expr(terminal as u32)?)?;
        literals_by_local[local] = Some(literals);
    }

    let mut checked_splits = 0usize;
    let mut checked_literal_pairs = 0usize;
    for pair in 0..(1 << 16) {
        let followers = follower_masks_by_pair[pair];
        if followers == 0 {
            continue;
        }
        let left = (pair >> 8) as u8;
        let right = pair as u8;
        for &packed in index.occurrences_for_pair(left, right) {
            checked_splits += 1;
            let entry_index = (packed >> 32) as usize;
            let split_after = packed as u32 as usize;
            let token_id = index.entry_token_ids[entry_index];
            let bytes = vocab.get(token_id)?;
            let suffix = &bytes[split_after + 1..];
            let mut possible = followers;
            while possible != 0 {
                let local = possible.trailing_zeros() as usize;
                possible &= possible - 1;
                let literals = literals_by_local[local].as_ref()?;
                for literal in literals {
                    checked_literal_pairs += 1;
                    let viable = if suffix.len() <= literal.len() {
                        literal.starts_with(suffix)
                    } else {
                        suffix.starts_with(literal)
                    };
                    if viable {
                        if super::types::compile_profile_enabled() {
                            eprintln!(
                                "[glrmask/profile][terminal_path_finite_follower_empty] candidates={} followers={} checked_splits={} checked_literal_pairs={} certified=false ms={:.3}",
                                candidate_ids.len(),
                                follower_mask.count_ones(),
                                checked_splits,
                                checked_literal_pairs,
                                started_at.elapsed().as_secs_f64() * 1000.0,
                            );
                        }
                        return Some(false);
                    }
                }
            }
        }
    }

    if super::types::compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][terminal_path_finite_follower_empty] candidates={} followers={} checked_splits={} checked_literal_pairs={} certified=true ms={:.3}",
            candidate_ids.len(),
            follower_mask.count_ones(),
            checked_splits,
            checked_literal_pairs,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(true)
}

/// Exact terminal-path classification for a small family of finite literal
/// languages.  Starting the prefix scanner from every live residual of a
/// literal means terminal `t` is final at a split exactly when the consumed
/// token prefix is a non-empty suffix of one of `t`'s literals.  Starting the
/// suffix scanner from reset means `t` is viable exactly when the remaining
/// token suffix and one of `t`'s literals are prefix-comparable.
fn exact_terminal_path_two_plus_finite_literals(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    candidates: &BitSet,
) -> Option<ExactTerminalPathTwoPlus> {
    let candidate_ids = candidates.iter().collect::<Vec<_>>();
    if candidate_ids.is_empty() || candidate_ids.len() > FINITE_LITERAL_CLASSIFY_MAX_CANDIDATES {
        return None;
    }
    let literals_by_local = candidate_ids
        .iter()
        .map(|&terminal| {
            collect_finite_literal_strings(tokenizer.terminal_expr(terminal as u32)?)
        })
        .collect::<Option<Vec<_>>>()?;

    // Dense trie over every possible non-empty literal completion.  The small
    // specialization cap keeps this comfortably tiny while making token-prefix
    // scanning branch-light and allocation-free.
    const NONE: u16 = u16::MAX;
    let mut trie = vec![Box::new([NONE; 256])];
    let mut matched_masks = vec![0u64];
    let mut reset_trie = vec![Box::new([NONE; 256])];
    let mut reset_accepting_masks = vec![0u64];
    for (local, literals) in literals_by_local.iter().enumerate() {
        let bit = 1u64 << local;
        for literal in literals {
            let mut reset_node = 0usize;
            for &byte in literal {
                let next = reset_trie[reset_node][byte as usize];
                reset_node = if next == NONE {
                    if reset_trie.len() >= usize::from(NONE) {
                        return None;
                    }
                    let child = reset_trie.len();
                    reset_trie.push(Box::new([NONE; 256]));
                    reset_accepting_masks.push(0);
                    reset_trie[reset_node][byte as usize] = child as u16;
                    child
                } else {
                    next as usize
                };
            }
            reset_accepting_masks[reset_node] |= bit;

            for start in 0..literal.len() {
                let mut node = 0usize;
                for &byte in &literal[start..] {
                    let next = trie[node][byte as usize];
                    node = if next == NONE {
                        if trie.len() >= usize::from(NONE) {
                            return None;
                        }
                        let child = trie.len();
                        trie.push(Box::new([NONE; 256]));
                        matched_masks.push(0);
                        trie[node][byte as usize] = child as u16;
                        child
                    } else {
                        next as usize
                    };
                }
                matched_masks[node] |= bit;
            }
        }
    }
    let mut reset_subtree_masks = reset_accepting_masks.clone();
    let mut reset_future_masks = vec![0u64; reset_trie.len()];
    for node in (0..reset_trie.len()).rev() {
        let mut subtree = reset_subtree_masks[node];
        let mut future = 0u64;
        for &child in reset_trie[node].iter() {
            if child != NONE {
                let child_subtree = reset_subtree_masks[child as usize];
                subtree |= child_subtree;
                future |= child_subtree;
            }
        }
        reset_subtree_masks[node] = subtree;
        reset_future_masks[node] = future;
    }

    let all_local = if candidate_ids.len() == 64 {
        u64::MAX
    } else {
        (1u64 << candidate_ids.len()) - 1
    };
    let mut allowed_after = vec![all_local; candidate_ids.len()];
    for (local_1, &terminal_1) in candidate_ids.iter().enumerate() {
        if let Some(blocked) = disallowed_follows.get(&(terminal_1 as u32)) {
            let mut allowed = all_local;
            for (local_2, &terminal_2) in candidate_ids.iter().enumerate() {
                if blocked.contains(terminal_2) {
                    allowed &= !(1u64 << local_2);
                }
            }
            allowed_after[local_1] = allowed;
        }
    }

    // Every exact within-token boundary between two finite literal terminals
    // necessarily contains the final byte of the left literal immediately
    // followed by the first byte of the right literal.  Use the immutable
    // per-vocabulary adjacent-pair index to restrict the expensive witness
    // scan to tokens that contain at least one such *allowed* pair.  This is a
    // strict necessary-condition filter only; the unchanged trie walk below
    // remains the authority for the actual terminal/path relation.
    let mut allowed_boundary_pairs = [U8Set::empty(); 256];
    for (local_1, left_literals) in literals_by_local.iter().enumerate() {
        let allowed = allowed_after[local_1];
        if allowed == 0 {
            continue;
        }
        for left in left_literals {
            let Some(&left_last) = left.last() else {
                continue;
            };
            let mut followers = allowed;
            while followers != 0 {
                let local_2 = followers.trailing_zeros() as usize;
                followers &= followers - 1;
                for right in &literals_by_local[local_2] {
                    if let Some(&right_first) = right.first() {
                        allowed_boundary_pairs[left_last as usize].insert(right_first);
                    }
                }
            }
        }
    }
    let candidate_token_ids = vocab_tokens_with_adjacent_pairs(vocab, &allowed_boundary_pairs);

    let started_at = std::time::Instant::now();
    let mut found = 0u64;
    let mut witnesses = vec![None; candidates.len()];
    let mut checked_prefix_matches = 0usize;
    'tokens: for token_id in candidate_token_ids.iter().copied() {
        let bytes = vocab
            .entries_map()
            .get(&token_id)
            .expect("prepared adjacent-pair index returned a token absent from its vocabulary");
        if bytes.len() < 2 {
            continue;
        }

        // `continuations` represents terminal matches that began at an earlier
        // within-token boundary. Each active mask records which candidate
        // terminals are allowed to complete along that reset-trie state.
        let mut continuations = Vec::<(usize, u64)>::new();
        let mut next_continuations = Vec::<(usize, u64)>::new();
        let mut initial_node = Some(0usize);

        for split_position in 1..bytes.len() {
            let byte = bytes[split_position - 1];
            let mut matched = 0u64;

            if let Some(node) = initial_node {
                let next = trie[node][byte as usize];
                if next == NONE {
                    initial_node = None;
                } else {
                    let next = next as usize;
                    initial_node = Some(next);
                    matched |= matched_masks[next];
                }
            }

            next_continuations.clear();
            for &(node, active) in &continuations {
                let next = reset_trie[node][byte as usize];
                if next == NONE {
                    continue;
                }
                let next = next as usize;
                matched |= reset_accepting_masks[next] & active;
                let live = reset_future_masks[next] & active;
                if live == 0 {
                    continue;
                }
                if let Some((_, existing)) = next_continuations
                    .iter_mut()
                    .find(|(candidate, _)| *candidate == next)
                {
                    *existing |= live;
                } else {
                    next_continuations.push((next, live));
                }
            }

            if matched == 0 {
                std::mem::swap(&mut continuations, &mut next_continuations);
                continue;
            }
            checked_prefix_matches += 1;

            let suffix = &bytes[split_position..];
            let mut suffix_viable = 0u64;
            let mut suffix_node = 0usize;
            let mut consumed_suffix = true;
            for &suffix_byte in suffix {
                let next = reset_trie[suffix_node][suffix_byte as usize];
                if next == NONE {
                    consumed_suffix = false;
                    break;
                }
                suffix_node = next as usize;
                // A literal ending before the suffix does is already a viable
                // next terminal; a longer literal may remain completable.
                suffix_viable |= reset_accepting_masks[suffix_node];
            }
            if consumed_suffix {
                suffix_viable |= reset_subtree_masks[suffix_node];
            }

            let mut followers_union = 0u64;
            let mut prefix_terminals = matched;
            while prefix_terminals != 0 {
                let local_1 = prefix_terminals.trailing_zeros() as usize;
                prefix_terminals &= prefix_terminals - 1;
                let followers = suffix_viable & allowed_after[local_1];
                if followers == 0 {
                    continue;
                }
                followers_union |= followers;
                found |= 1u64 << local_1;
                found |= followers;
                let local_2 = followers.trailing_zeros() as usize;
                let terminal_1 = candidate_ids[local_1];
                let terminal_2 = candidate_ids[local_2];
                if witnesses[terminal_1].is_none() || witnesses[terminal_2].is_none() {
                    let witness = TerminalBoundaryWitness {
                        token_id,
                        token_bytes: bytes.clone(),
                        split_position,
                        terminal_1,
                        terminal_2,
                    };
                    if witnesses[terminal_1].is_none() {
                        witnesses[terminal_1] = Some(witness.clone());
                    }
                    if witnesses[terminal_2].is_none() {
                        witnesses[terminal_2] = Some(witness);
                    }
                }
            }

            if followers_union != 0 {
                if let Some((_, existing)) = next_continuations
                    .iter_mut()
                    .find(|(candidate, _)| *candidate == 0)
                {
                    *existing |= followers_union;
                } else {
                    next_continuations.push((0, followers_union));
                }
            }
            std::mem::swap(&mut continuations, &mut next_continuations);

            if found == all_local {
                break 'tokens;
            }
        }
    }

    let mut two_plus = BitSet::new(candidates.len());
    for local in 0..candidate_ids.len() {
        if found & (1u64 << local) != 0 {
            two_plus.set(candidate_ids[local]);
        }
    }
    if super::types::compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][terminal_path_finite_literals] tokens={} candidate_tokens={} candidates={} trie_nodes={} checked_prefix_matches={} two_plus={} total_ms={:.3}",
            vocab.len(),
            candidate_token_ids.len(),
            candidate_ids.len(),
            trie.len(),
            checked_prefix_matches,
            two_plus.count_ones(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(ExactTerminalPathTwoPlus { two_plus, witnesses })
}

struct CandidatePrefixPowerset<'a> {
    tokenizer: &'a Tokenizer,
    words_per_mask: usize,
    terminal_to_local: Option<Box<[u32]>>,
    sparse_transitions_by_byte: Option<&'a [Vec<(u32, u32)>]>,
    singleton_epsilon_closures: Option<Vec<Option<Box<[u32]>>>>,
    closure_marks: Vec<u32>,
    closure_generation: u32,
    configs: Vec<Box<[u32]>>,
    config_membership_words: Vec<Box<[u64]>>,
    config_ids: FxHashMap<Vec<u32>, u32>,
    transitions: Vec<[u32; 256]>,
    matched_masks: Vec<u64>,
    future_masks: Vec<u64>,
}

/// Exact powerset scanner for the original epsilon-NFA, restricted to one
/// active terminal language.  Unlike the bounded-view builder this retains
/// only the configuration IDs and matched-terminal masks needed by terminal
/// path classification, so prefix propagation and boundary discovery can be
/// fused in one token scan.
struct ActivePrefixPowerset<'a> {
    tokenizer: &'a Tokenizer,
    active: &'a BitSet,
    active_language: Vec<bool>,
    words_per_mask: usize,
    configs: Vec<Box<[u32]>>,
    config_ids: FxHashMap<Vec<u32>, u32>,
    transitions: Vec<[u32; 256]>,
    matched_masks: Vec<u64>,
}

impl<'a> ActivePrefixPowerset<'a> {
    fn new(tokenizer: &'a Tokenizer, active: &'a BitSet, raw_start_states: &[usize]) -> (Self, u32) {
        let words_per_mask = active.len().div_ceil(64);
        let active_language = (0..tokenizer.num_states())
            .map(|state| {
                !tokenizer.matched_terminal_bitset(state).is_disjoint(active)
                    || !tokenizer.possible_future_terminals(state).is_disjoint(active)
            })
            .collect::<Vec<_>>();
        let singleton_closures = tokenizer.all_singleton_epsilon_closures();
        let mut start_states = Vec::<u32>::new();
        for &raw_state in raw_start_states {
            start_states.extend(
                singleton_closures[raw_state]
                    .iter()
                    .copied()
                    .filter(|&state| active_language[state as usize]),
            );
        }
        let mut scanner = Self {
            tokenizer,
            active,
            active_language,
            words_per_mask,
            configs: Vec::new(),
            config_ids: FxHashMap::default(),
            transitions: Vec::new(),
            matched_masks: Vec::new(),
        };
        let start = scanner.intern(start_states);
        (scanner, start)
    }

    fn intern(&mut self, mut states: Vec<u32>) -> u32 {
        if states.is_empty() {
            return PREFIX_DEAD_STATE;
        }
        states.sort_unstable();
        states.dedup();
        if let Some(&state) = self.config_ids.get(&states) {
            return state;
        }
        let state = self.configs.len() as u32;
        self.config_ids.insert(states.clone(), state);
        let mask_base = self.matched_masks.len();
        self.matched_masks.resize(mask_base + self.words_per_mask, 0);
        for &raw_state in &states {
            for terminal in self.tokenizer.matched_terminal_bitset(raw_state).iter() {
                if self.active.contains(terminal) {
                    self.matched_masks[mask_base + (terminal >> 6)] |=
                        1u64 << (terminal & 63);
                }
            }
        }
        self.configs.push(states.into_boxed_slice());
        self.transitions.push([PREFIX_UNKNOWN_STATE; 256]);
        state
    }

    #[inline]
    fn step(&mut self, state: u32, byte: u8) -> u32 {
        if state == PREFIX_DEAD_STATE {
            return PREFIX_DEAD_STATE;
        }
        let cached = self.transitions[state as usize][byte as usize];
        if cached != PREFIX_UNKNOWN_STATE {
            return cached;
        }
        let mut targets = self
            .tokenizer
            .step_all(self.configs[state as usize].as_ref(), byte)
            .into_iter()
            .filter(|&target| self.active_language[target as usize])
            .collect::<Vec<_>>();
        targets.sort_unstable();
        targets.dedup();
        let target = self.intern(targets);
        self.transitions[state as usize][byte as usize] = target;
        target
    }

    #[inline]
    fn matched_mask(&self, state: u32) -> &[u64] {
        let base = state as usize * self.words_per_mask;
        &self.matched_masks[base..base + self.words_per_mask]
    }
}

impl<'a> CandidatePrefixPowerset<'a> {
    fn new(
        tokenizer: &'a Tokenizer,
        words_per_mask: usize,
        terminal_to_local: Option<Box<[u32]>>,
        sparse_transitions_by_byte: Option<&'a [Vec<(u32, u32)>]>,
    ) -> (Self, u32, u32) {
        let mut scanner = Self {
            tokenizer,
            words_per_mask,
            terminal_to_local,
            sparse_transitions_by_byte,
            singleton_epsilon_closures: tokenizer
                .has_epsilon_transitions()
                .then(|| vec![None; tokenizer.num_states() as usize]),
            closure_marks: if tokenizer.has_epsilon_transitions() {
                vec![0; tokenizer.num_states() as usize]
            } else {
                Vec::new()
            },
            closure_generation: 0,
            configs: Vec::new(),
            config_membership_words: Vec::new(),
            config_ids: FxHashMap::default(),
            transitions: Vec::new(),
            matched_masks: Vec::new(),
            future_masks: Vec::new(),
        };
        let start_states = (0..tokenizer.num_states())
            .filter(|&state| scanner.state_has_candidate_future(state))
            .collect::<Vec<_>>();
        let start = scanner.intern(start_states);
        let reset = scanner.intern(
            tokenizer
                .execute_from_state_end_only(&[], tokenizer.initial_state_id())
                .to_vec(),
        );
        (scanner, start, reset)
    }

    #[inline]
    fn local_terminal(&self, terminal: usize) -> Option<usize> {
        self.terminal_to_local.as_ref().map_or(Some(terminal), |map| {
            let local = map[terminal];
            (local != u32::MAX).then_some(local as usize)
        })
    }

    #[inline]
    fn state_has_candidate_future(&self, state: u32) -> bool {
        self.tokenizer
            .possible_future_terminals(state)
            .iter()
            .any(|terminal| self.local_terminal(terminal).is_some())
    }

    fn intern(&mut self, mut states: Vec<u32>) -> u32 {
        if states.is_empty() {
            return PREFIX_DEAD_STATE;
        }
        states.sort_unstable();
        states.dedup();
        if let Some(&state) = self.config_ids.get(&states) {
            return state;
        }
        let state = self.configs.len() as u32;
        self.config_ids.insert(states.clone(), state);
        let mask_base = self.matched_masks.len();
        self.matched_masks
            .resize(mask_base + self.words_per_mask, 0);
        self.future_masks
            .resize(mask_base + self.words_per_mask, 0);
        for &raw_state in &states {
            for terminal in self.tokenizer.matched_terminal_bitset(raw_state).iter() {
                if let Some(local) = self.local_terminal(terminal) {
                    self.matched_masks[mask_base + (local >> 6)] |= 1u64 << (local & 63);
                }
            }
            for terminal in self.tokenizer.possible_future_terminals(raw_state).iter() {
                if let Some(local) = self.local_terminal(terminal) {
                    self.future_masks[mask_base + (local >> 6)] |= 1u64 << (local & 63);
                }
            }
        }
        if self.sparse_transitions_by_byte.is_some() {
            let mut membership = vec![0u64; (self.tokenizer.num_states() as usize).div_ceil(64)];
            for &raw_state in &states {
                membership[raw_state as usize >> 6] |= 1u64 << (raw_state as usize & 63);
            }
            self.config_membership_words
                .push(membership.into_boxed_slice());
        } else {
            self.config_membership_words.push(Box::new([]));
        }
        self.configs.push(states.into_boxed_slice());
        self.transitions.push([PREFIX_UNKNOWN_STATE; 256]);
        state
    }

    #[inline]
    fn step(&mut self, state: u32, byte: u8) -> u32 {
        if state == PREFIX_DEAD_STATE {
            return PREFIX_DEAD_STATE;
        }
        let cached = self.transitions[state as usize][byte as usize];
        if cached != PREFIX_UNKNOWN_STATE {
            return cached;
        }
        let direct_targets = if let Some(sparse_transitions_by_byte) =
            self.sparse_transitions_by_byte
        {
            let sparse = &sparse_transitions_by_byte[byte as usize];
            let config = self.configs[state as usize].as_ref();
            if sparse.len() < config.len() {
                let membership = self.config_membership_words[state as usize].as_ref();
                sparse
                    .iter()
                    .filter_map(|&(source, target)| {
                        ((membership[source as usize >> 6]
                            & (1u64 << (source as usize & 63)))
                            != 0)
                            .then_some(target)
                    })
                    .collect::<Vec<_>>()
            } else {
                config
                    .iter()
                    .filter_map(|&raw_state| {
                        let target = self.tokenizer.get_transition(raw_state, byte);
                        (target != u32::MAX).then_some(target)
                    })
                    .collect::<Vec<_>>()
            }
        } else {
            self.configs[state as usize]
                .iter()
                .filter_map(|&raw_state| {
                    let target = self.tokenizer.get_transition(raw_state, byte);
                    (target != u32::MAX).then_some(target)
                })
                .collect::<Vec<_>>()
        };
        let mut targets = if let Some(singleton_closures) =
            self.singleton_epsilon_closures.as_mut()
        {
            // Every interned configuration is epsilon-closed. Follow visible
            // byte edges once, then union cached singleton closures without
            // rebuilding source closures or allocating a fresh seen vector.
            self.closure_generation = self.closure_generation.wrapping_add(1);
            if self.closure_generation == 0 {
                self.closure_marks.fill(0);
                self.closure_generation = 1;
            }
            let generation = self.closure_generation;
            let mut targets = Vec::new();
            for target in direct_targets {
                let closure = singleton_closures[target as usize]
                    .get_or_insert_with(|| self.tokenizer.singleton_epsilon_closure(target));
                for &closed in closure.iter() {
                    let mark = &mut self.closure_marks[closed as usize];
                    if *mark != generation {
                        *mark = generation;
                        targets.push(closed);
                    }
                }
            }
            targets
        } else {
            direct_targets
        };
        targets.sort_unstable();
        targets.dedup();
        let target = self.intern(targets);
        self.transitions[state as usize][byte as usize] = target;
        target
    }

    #[inline]
    fn matched_mask(&self, state: u32) -> &[u64] {
        let base = state as usize * self.words_per_mask;
        &self.matched_masks[base..base + self.words_per_mask]
    }

    #[inline]
    fn future_mask(&self, state: u32) -> &[u64] {
        let base = state as usize * self.words_per_mask;
        &self.future_masks[base..base + self.words_per_mask]
    }
}

fn accumulate_terminal_path_boundaries(
    matched: &[u64],
    suffix: &[u64],
    allowed_after: &[u64],
    words_per_mask: usize,
    candidate_count: usize,
    two_plus: &mut [u64],
    followers: &mut [u64],
    witness_context: Option<(
        &mut [Option<TerminalBoundaryWitness>],
        u32,
        &[u8],
        usize,
    )>,
) -> usize {
    followers.fill(0);
    let mut witness_context = witness_context;
    let mut allowed_pairs = 0usize;
    for (matched_word_index, &matched_word) in matched.iter().enumerate() {
        let mut pending_matched = matched_word;
        while pending_matched != 0 {
            let terminal_1 =
                matched_word_index * 64 + pending_matched.trailing_zeros() as usize;
            pending_matched &= pending_matched - 1;
            if terminal_1 >= candidate_count {
                continue;
            }
            let allowed_base = terminal_1 * words_per_mask;
            let mut terminal_1_has_follow = false;
            for word_index in 0..words_per_mask {
                let accepted = suffix[word_index] & allowed_after[allowed_base + word_index];
                if accepted == 0 {
                    continue;
                }
                terminal_1_has_follow = true;
                allowed_pairs += accepted.count_ones() as usize;
                followers[word_index] |= accepted;
                two_plus[word_index] |= accepted;
                if let Some((witnesses, token_id, token_bytes, split_position)) =
                    witness_context.as_mut()
                {
                    let mut pending = accepted;
                    while pending != 0 {
                        let terminal_2 =
                            word_index * 64 + pending.trailing_zeros() as usize;
                        pending &= pending - 1;
                        if terminal_2 >= candidate_count {
                            continue;
                        }
                        if witnesses[terminal_1].is_none() || witnesses[terminal_2].is_none() {
                            let witness = TerminalBoundaryWitness {
                                token_id: *token_id,
                                token_bytes: token_bytes.to_vec(),
                                split_position: *split_position,
                                terminal_1,
                                terminal_2,
                            };
                            if witnesses[terminal_1].is_none() {
                                witnesses[terminal_1] = Some(witness.clone());
                            }
                            if witnesses[terminal_2].is_none() {
                                witnesses[terminal_2] = Some(witness);
                            }
                        }
                    }
                }
            }
            if terminal_1_has_follow {
                two_plus[terminal_1 >> 6] |= 1u64 << (terminal_1 & 63);
            }
        }
    }
    allowed_pairs
}

#[inline]
fn merge_word_continuation(continuations: &mut Vec<(u32, u64)>, state: u32, active: u64) {
    if active == 0 {
        return;
    }
    if let Some((_, existing)) = continuations.iter_mut().find(|(candidate, _)| *candidate == state)
    {
        *existing |= active;
    } else {
        continuations.push((state, active));
    }
}

struct DenseWordContinuations {
    masks: Vec<u64>,
    touched: Vec<u32>,
}

impl DenseWordContinuations {
    fn new(num_states: usize) -> Self {
        Self {
            masks: vec![0; num_states],
            touched: Vec::new(),
        }
    }

    #[inline]
    fn merge(&mut self, state: u32, active: u64) {
        if active == 0 {
            return;
        }
        let slot = &mut self.masks[state as usize];
        if *slot == 0 {
            self.touched.push(state);
        }
        *slot |= active;
    }

    fn clear(&mut self) {
        for &state in &self.touched {
            self.masks[state as usize] = 0;
        }
        self.touched.clear();
    }
}

fn merge_mask_continuation(
    continuations: &mut Vec<(u32, Box<[u64]>)>,
    state: u32,
    active: &[u64],
) {
    if active.iter().all(|&word| word == 0) {
        return;
    }
    if let Some((_, existing)) = continuations.iter_mut().find(|(candidate, _)| *candidate == state)
    {
        for (existing_word, active_word) in existing.iter_mut().zip(active) {
            *existing_word |= *active_word;
        }
    } else {
        continuations.push((state, active.to_vec().into_boxed_slice()));
    }
}

fn execute_prepared_dense_suffix_trie(
    trie: &VocabSuffixTrie,
    continuation_reset_state: u32,
    dense_flat_trans: &[u32],
    dense_finalizer_masks: &[u64],
    dense_future_masks: &[u64],
) -> (Vec<u32>, Vec<u64>) {
    let mut state_by_node = vec![u32::MAX; trie.parents.len()];
    let mut matched_by_node = vec![0u64; trie.parents.len()];
    state_by_node[0] = continuation_reset_state;
    for node in 1..trie.parents.len() {
        let parent = trie.parents[node] as usize;
        let parent_state = state_by_node[parent];
        let parent_matched = matched_by_node[parent];
        matched_by_node[node] = parent_matched;
        if parent_state == u32::MAX || dense_future_masks[parent_state as usize] == 0 {
            continue;
        }
        let state = dense_flat_trans
            [parent_state as usize * 256 + trie.incoming_bytes[node] as usize];
        if state == u32::MAX {
            continue;
        }
        state_by_node[node] = state;
        matched_by_node[node] |= dense_finalizer_masks[state as usize];
    }
    (state_by_node, matched_by_node)
}

fn execute_prepared_prefix_trie(
    trie: &VocabPrefixTrie,
    scanner: &mut CandidatePrefixPowerset<'_>,
    prefix_start: u32,
) -> Vec<u32> {
    let mut state_by_node = vec![PREFIX_DEAD_STATE; trie.parents.len()];
    state_by_node[0] = prefix_start;
    for node in 1..trie.parents.len() {
        let parent = trie.parents[node] as usize;
        state_by_node[node] = scanner.step(state_by_node[parent], trie.incoming_bytes[node]);
    }
    state_by_node
}

fn exact_terminal_path_two_plus_candidate_dfa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    candidates: &BitSet,
    bytesets: &SharedClassifyBytesets,
    witness_probe_callback: Option<&dyn Fn(&BitSet)>,
) -> ExactTerminalPathTwoPlus {
    let total_started_at = std::time::Instant::now();
    let candidate_ids = candidates.iter().collect::<Vec<_>>();
    if candidate_ids.is_empty() {
        return ExactTerminalPathTwoPlus {
            two_plus: BitSet::new(candidates.len()),
            witnesses: vec![None; candidates.len()],
        };
    }
    let candidate_count = candidate_ids.len();
    let words_per_mask = candidate_count.div_ceil(64);
    let use_pair_index = std::env::var("GLRMASK_CLASSIFY_PAIR_INDEX")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);

    let mut local_disallowed = BTreeMap::<u32, BitSet>::new();
    for (local_1, &terminal_1) in candidate_ids.iter().enumerate() {
        let Some(blocked) = disallowed_follows.get(&(terminal_1 as u32)) else {
            continue;
        };
        let mut local_blocked = BitSet::new(candidate_count);
        for (local_2, &terminal_2) in candidate_ids.iter().enumerate() {
            if blocked.contains(terminal_2) {
                local_blocked.set(local_2);
            }
        }
        if !local_blocked.is_empty() {
            local_disallowed.insert(local_1 as u32, local_blocked);
        }
    }
    // Any exact within-token boundary must place a terminal-1 last byte
    // immediately before a terminal-2 first byte. Apply that same sound byte
    // condition at each split, rather than analyzing every suffix admitted by
    // the partition-wide candidate upper bound.
    let mut feasible_follow_bytes_by_last_byte = [U8Set::empty(); 256];
    for (local_1, &terminal_1) in candidate_ids.iter().enumerate() {
        let blocked = local_disallowed.get(&(local_1 as u32));
        for (local_2, &terminal_2) in candidate_ids.iter().enumerate() {
            if blocked.is_some_and(|blocked| blocked.contains(local_2)) {
                continue;
            }
            let first_bytes = bytesets.first_bytes[terminal_2];
            for last_byte in bytesets.last_bytes[terminal_1].iter() {
                feasible_follow_bytes_by_last_byte[last_byte as usize] |= first_bytes;
            }
        }
    }
    let split_may_host_boundary = |bytes: &[u8], split_after: usize| {
        feasible_follow_bytes_by_last_byte[bytes[split_after] as usize]
            .contains(bytes[split_after + 1])
    };
    let adjacent_pair_index = use_pair_index.then(|| vocab_adjacent_pair_index(vocab));
    let feasible_split_work = if candidate_count < 64 {
        adjacent_pair_index.as_ref().map_or(0usize, |index| {
            (0u8..=u8::MAX)
                .map(|left| {
                    feasible_follow_bytes_by_last_byte[left as usize]
                        .iter()
                        .map(|right| index.occurrences_for_pair(left, right).len())
                        .sum::<usize>()
                })
                .sum()
        })
    } else {
        0
    };
    let prepared_suffix_trie = (words_per_mask == 1
        && adjacent_pair_index
            .as_ref()
            .is_some_and(|index| index.total_splits >= PREPARED_SUFFIX_TRIE_MIN_SPLITS)
        && prepared_suffix_trie_enabled())
    .then(|| vocab.vocab_derived_cache_get::<VocabSuffixTrie>())
    .flatten();

    let prepared_prefix_trie = (words_per_mask == 1
        && adjacent_pair_index
            .as_ref()
            .is_some_and(|index| index.total_splits >= PREPARED_SUFFIX_TRIE_MIN_SPLITS)
        && std::env::var_os("GLRMASK_CLASSIFY_PREPARED_PREFIX_TRIE").is_some())
    .then(|| vocab.vocab_derived_cache_get::<VocabPrefixTrie>())
    .flatten();

    let compile_started_at = std::time::Instant::now();
    // Reuse the original tokenizer for broad candidate sets and for every
    // epsilon tokenizer. Recompiling an epsilon lexer from a small terminal
    // subset can destroy sharing and produce a much larger determinized scanner
    // than the original combined lexer. The powerset scanner filters matched
    // and future masks through `terminal_to_local`, so scanning the original
    // tokenizer remains exact while avoiding that subset-compilation cliff.
    let force_subset_tokenizer = std::env::var("GLRMASK_CLASSIFY_FORCE_SUBSET_TOKENIZER")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(false);
    let subset_min_split_work = std::env::var("GLRMASK_CLASSIFY_SUBSET_MIN_SPLIT_WORK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100_000);
    let automatic_subset_attempt = tokenizer.has_epsilon_transitions()
        && candidate_count < 64
        && feasible_split_work >= subset_min_split_work;
    let normally_requires_original =
        tokenizer.has_epsilon_transitions() || candidate_count >= 64;
    let attempt_subset = force_subset_tokenizer
        || automatic_subset_attempt
        || !normally_requires_original;
    let compiled_subset_tokenizer = attempt_subset.then(|| {
        let exprs = Arc::<[crate::automata::lexer::ast::Expr]>::from(
            candidate_ids
                .iter()
                .map(|&terminal| {
                    tokenizer
                        .terminal_expr(terminal as u32)
                        .unwrap_or_else(|| panic!("missing terminal expression for terminal {terminal}"))
                        .clone()
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        let regex = if automatic_subset_attempt && prepared_suffix_trie.is_some() {
            build_regex_local_small_product(exprs.as_ref())
        } else {
            build_regex(exprs.as_ref())
        };
        Arc::new(regex.into_tokenizer(
            candidate_ids.len() as u32,
            Some(Arc::clone(&exprs)),
        ))
    });
    // Subset compilation is exact, but epsilon sharing can make its DFA larger
    // than the original combined scanner. Automatic selection therefore
    // remains fail-closed: pay the small probe cost, then retain the original
    // representation unless the candidate DFA is strictly smaller.
    let automatic_subset_tokenizer = automatic_subset_attempt
        && compiled_subset_tokenizer
            .as_ref()
            .is_some_and(|subset| subset.num_states() < tokenizer.num_states());
    let uses_original_tokenizer = normally_requires_original
        && !(force_subset_tokenizer || automatic_subset_tokenizer);
    let terminal_to_local = uses_original_tokenizer.then(|| {
        let mut map = vec![u32::MAX; tokenizer.num_terminals() as usize];
        for (local, &terminal) in candidate_ids.iter().enumerate() {
            map[terminal] = local as u32;
        }
        map.into_boxed_slice()
    });
    let owned_candidate_tokenizer = (!uses_original_tokenizer)
        .then(|| compiled_subset_tokenizer.expect("selected subset tokenizer was compiled"));
    let candidate_tokenizer = owned_candidate_tokenizer.as_deref().unwrap_or(tokenizer);
    let compile_ms = compile_started_at.elapsed().as_secs_f64() * 1000.0;

    let analyze_started_at = std::time::Instant::now();
    let total_splits = adjacent_pair_index.as_ref().map_or_else(
        || {
            vocab
                .entries_map()
                .values()
                .map(|bytes| bytes.len().saturating_sub(1))
                .sum::<usize>()
        },
        |index| index.total_splits,
    );

    // The scanner already interns exact epsilon-closed prefix configurations and
    // memoizes every (configuration, byte) transition. Store that compact state
    // ID at each token split directly. A global byte trie duplicates the same
    // prefixes as nodes and masks, while the downstream pass still visits every
    // token split in vocabulary order.
    let (mut prefix_scanner, prefix_start, reset_state) = CandidatePrefixPowerset::new(
        candidate_tokenizer,
        words_per_mask,
        terminal_to_local,
        uses_original_tokenizer.then_some(bytesets.sparse_transitions_by_byte.as_slice()),
    );
    let owned_split_offsets;
    let split_offsets: &[usize] = if let Some(index) = adjacent_pair_index.as_ref() {
        &index.entry_split_offsets
    } else {
        owned_split_offsets = {
            let mut offsets = Vec::with_capacity(vocab.entries_map().len() + 1);
            offsets.push(0usize);
            for bytes in vocab.entries_map().values() {
                offsets.push(
                    offsets
                        .last()
                        .copied()
                        .unwrap_or(0usize)
                        .saturating_add(bytes.len().saturating_sub(1)),
                );
            }
            offsets
        };
        &owned_split_offsets
    };
    let prefix_started_at = std::time::Instant::now();
    let prepared_prefix_states = prepared_prefix_trie.as_ref().map(|trie| {
        execute_prepared_prefix_trie(trie, &mut prefix_scanner, prefix_start)
    });
    let prefix_ms = prefix_started_at.elapsed().as_secs_f64() * 1000.0;
    let trie_ms = 0.0;
    if super::types::compile_profile_enabled() {
        if let (Some(trie), Some(_)) = (prepared_prefix_trie.as_ref(), prepared_prefix_states.as_ref()) {
            eprintln!(
                "[glrmask/profile][terminal_path_prefix_trie] nodes={} splits={} execute_ms={:.3}",
                trie.parents.len(),
                trie.split_nodes.len(),
                prefix_ms,
            );
        }
    }

    let suffix_started_at = std::time::Instant::now();
    let mut dense_flat_trans = Vec::new();
    let mut dense_finalizer_masks = Vec::new();
    let mut dense_future_masks = Vec::new();
    let continuation_reset_state = if uses_original_tokenizer {
        reset_state
    } else {
        let state_count = candidate_tokenizer.num_states() as usize;
        dense_flat_trans = super::l1::build_flat_transition_table(candidate_tokenizer);
        dense_finalizer_masks = vec![0u64; state_count * words_per_mask];
        dense_future_masks = vec![0u64; state_count * words_per_mask];
        for state in 0..state_count {
            let base = state * words_per_mask;
            dense_finalizer_masks[base..base + words_per_mask].copy_from_slice(
                &candidate_tokenizer.matched_terminal_bitset(state as u32).words()
                    [..words_per_mask],
            );
            dense_future_masks[base..base + words_per_mask].copy_from_slice(
                &candidate_tokenizer.possible_future_terminals(state as u32).words()
                    [..words_per_mask],
            );
        }
        candidate_tokenizer.initial_state_id()
    };
    // Before materializing suffix viability for every feasible split, try a
    // small exact witness probe.  Terminal-path classification asks only whether
    // each candidate has *some* two-terminal witness.  A witness found here is
    // therefore a complete positive certificate; if every candidate is
    // witnessed we can skip the full split table entirely.  Failure to cover
    // every candidate simply falls through to the unchanged exhaustive proof.
    let witness_probe_started_at = std::time::Instant::now();
    let mut witness_probe_checks = 0usize;
    let mut witness_probe_found = 0u64;
    let witness_probe_limit = std::env::var("GLRMASK_CLASSIFY_WITNESS_PROBE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1024);
    let emit_witness_probe = |found: u64| {
        if let Some(callback) = witness_probe_callback {
            let mut global = BitSet::new(candidates.len());
            let mut pending = found;
            while pending != 0 {
                let local = pending.trailing_zeros() as usize;
                pending &= pending - 1;
                if local < candidate_ids.len() {
                    global.set(candidate_ids[local]);
                }
            }
            callback(&global);
        }
    };
    let mut witness_probe_emitted = false;
    if words_per_mask == 1
        && candidate_count < 64
        && feasible_split_work >= 10_000
        && witness_probe_limit > 0
        && std::env::var_os("GLRMASK_DUMP_TERMINAL_PATH_WITNESSES").is_none()
    {
        let all_local = (1u64 << candidate_count) - 1;
        if let Some(index) = adjacent_pair_index.as_ref() {
            let vocab_entries = vocab.entries_map().iter().collect::<Vec<_>>();
            'probe: for left in 0u8..=u8::MAX {
                for right in feasible_follow_bytes_by_last_byte[left as usize].iter() {
                    // Sampling a few occurrences of every semantically feasible
                    // byte pair gives broad witness diversity without scanning
                    // the potentially hundreds of thousands of occurrences.
                    for &packed in index.occurrences_for_pair(left, right).iter().take(4) {
                        let entry_index = (packed >> 32) as usize;
                        let split_after = packed as u32 as usize;
                        let (&_token_id, bytes) = vocab_entries[entry_index];
                        let mut prefix_state = prefix_start;
                        for &byte in &bytes[..=split_after] {
                            prefix_state = prefix_scanner.step(prefix_state, byte);
                            if prefix_state == PREFIX_DEAD_STATE {
                                break;
                            }
                        }
                        if prefix_state == PREFIX_DEAD_STATE {
                            continue;
                        }
                        let prefix_matched = prefix_scanner.matched_mask(prefix_state)[0];
                        if prefix_matched == 0 {
                            continue;
                        }

                        let mut suffix_state = continuation_reset_state;
                        let mut suffix_matched = 0u64;
                        let mut consumed_suffix = true;
                        for &byte in &bytes[split_after + 1..] {
                            if uses_original_tokenizer {
                                if prefix_scanner.future_mask(suffix_state)[0] == 0 {
                                    consumed_suffix = false;
                                    break;
                                }
                                suffix_state = prefix_scanner.step(suffix_state, byte);
                                if suffix_state == PREFIX_DEAD_STATE {
                                    consumed_suffix = false;
                                    break;
                                }
                                suffix_matched |= prefix_scanner.matched_mask(suffix_state)[0];
                            } else {
                                if dense_future_masks[suffix_state as usize] == 0 {
                                    consumed_suffix = false;
                                    break;
                                }
                                suffix_state = dense_flat_trans
                                    [suffix_state as usize * 256 + byte as usize];
                                if suffix_state == u32::MAX {
                                    consumed_suffix = false;
                                    break;
                                }
                                suffix_matched |= dense_finalizer_masks[suffix_state as usize];
                            }
                        }
                        if consumed_suffix {
                            suffix_matched |= if uses_original_tokenizer {
                                prefix_scanner.future_mask(suffix_state)[0]
                            } else {
                                dense_future_masks[suffix_state as usize]
                            };
                        }
                        if suffix_matched == 0 {
                            continue;
                        }
                        witness_probe_checks += 1;
                        let mut pending = prefix_matched;
                        while pending != 0 {
                            let terminal_1 = pending.trailing_zeros() as usize;
                            pending &= pending - 1;
                            if terminal_1 >= candidate_count {
                                continue;
                            }
                            let blocked = local_disallowed.get(&(terminal_1 as u32));
                            let mut accepted = suffix_matched;
                            if let Some(blocked) = blocked {
                                accepted &= !blocked.words().first().copied().unwrap_or(0);
                            }
                            accepted &= all_local;
                            if accepted != 0 {
                                witness_probe_found |= (1u64 << terminal_1) | accepted;
                            }
                        }
                        if witness_probe_found == all_local {
                            emit_witness_probe(witness_probe_found);
                            if super::types::compile_profile_enabled() {
                                eprintln!(
                                    "[glrmask/profile][terminal_path_witness_probe] candidates={} checks={} found={} selected=true ms={:.3}",
                                    candidate_count,
                                    witness_probe_checks,
                                    witness_probe_found.count_ones(),
                                    witness_probe_started_at.elapsed().as_secs_f64() * 1000.0,
                                );
                            }
                            let mut two_plus = BitSet::new(candidates.len());
                            for local in 0..candidate_count {
                                two_plus.set(candidate_ids[local]);
                            }
                            return ExactTerminalPathTwoPlus {
                                two_plus,
                                witnesses: vec![None; candidates.len()],
                            };
                        }
                        if witness_probe_checks >= witness_probe_limit {
                            break 'probe;
                        }
                    }
                }
            }
        }
    }
    if !witness_probe_emitted && witness_probe_checks != 0 {
        emit_witness_probe(witness_probe_found);
        witness_probe_emitted = true;
    }
    let _ = witness_probe_emitted;
    if super::types::compile_profile_enabled() && witness_probe_checks != 0 {
        eprintln!(
            "[glrmask/profile][terminal_path_witness_probe] candidates={} checks={} found={} selected=false ms={:.3}",
            candidate_count,
            witness_probe_checks,
            witness_probe_found.count_ones(),
            witness_probe_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let early_empty_suffix_enabled = std::env::var("GLRMASK_CLASSIFY_EARLY_EMPTY_SUFFIX")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true);

    // For the common text-vocabulary case the adjacent-byte proof leaves only
    // a tiny fraction of token splits as possible boundaries.  The historical
    // path nevertheless allocates a dense mask for *every* token split before
    // discovering that none of those few feasible suffixes can start/continue
    // a candidate terminal.  Probe the sparse feasible set first.  A miss is an
    // exact proof of an empty L2+ relation and avoids both the dense allocation
    // and the subsequent all-split combine pass.  A hit simply falls through to
    // the existing exact implementation.
    let sparse_empty_probe = early_empty_suffix_enabled
        && words_per_mask == 1
        && uses_original_tokenizer
        && adjacent_pair_index.is_some()
        && std::env::var_os("GLRMASK_DISABLE_SPARSE_EMPTY_SUFFIX_PROBE").is_none()
        && feasible_split_work.saturating_mul(16) < total_splits;
    if sparse_empty_probe {
        let sparse_started_at = std::time::Instant::now();
        let index = adjacent_pair_index.as_ref().expect("checked above");
        let indexed_entry_bytes = std::env::var("GLRMASK_CLASSIFY_INDEXED_ENTRY_BYTES")
            .map(|value| {
                let value = value.trim();
                value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
            })
            // The adjacent-pair index is prepared once per vocabulary. Reuse
            // its compact entry-byte view instead of rebuilding an indexable
            // Vec by walking every BTreeMap entry on each grammar compile.
            .unwrap_or(true);
        let vocab_entries = (!indexed_entry_bytes)
            .then(|| vocab.entries_map().values().collect::<Vec<_>>());
        let mut any_viable = false;
        'pairs: for left in 0u8..=u8::MAX {
            for right in feasible_follow_bytes_by_last_byte[left as usize].iter() {
                for &packed in index.occurrences_for_pair(left, right) {
                    let entry_index = (packed >> 32) as usize;
                    let split_after = packed as u32 as usize;
                    let bytes = if indexed_entry_bytes {
                        index.bytes_for_entry(entry_index)
                    } else {
                        vocab_entries
                            .as_ref()
                            .expect("non-indexed sparse probe requires entry view")[entry_index]
                            .as_slice()
                    };
                    let mut state = continuation_reset_state;
                    let mut matched = 0u64;
                    let mut consumed_suffix = true;
                    for &byte in &bytes[split_after + 1..] {
                        if prefix_scanner.future_mask(state)[0] == 0 {
                            consumed_suffix = false;
                            break;
                        }
                        state = prefix_scanner.step(state, byte);
                        if state == PREFIX_DEAD_STATE {
                            consumed_suffix = false;
                            break;
                        }
                        matched |= prefix_scanner.matched_mask(state)[0];
                    }
                    if consumed_suffix {
                        matched |= prefix_scanner.future_mask(state)[0];
                    }
                    if matched != 0 {
                        any_viable = true;
                        break 'pairs;
                    }
                }
            }
        }
        if !any_viable {
            if super::types::compile_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][terminal_path_sparse_empty_suffix] tokens={} candidates={} total_splits={} feasible_splits={} ms={:.3}",
                    vocab.len(),
                    candidate_count,
                    total_splits,
                    feasible_split_work,
                    sparse_started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return ExactTerminalPathTwoPlus {
                two_plus: BitSet::new(candidates.len()),
                witnesses: vec![None; candidates.len()],
            };
        }
    }

    let sparse_suffix_direct = words_per_mask == 1
        && adjacent_pair_index.is_some()
        && feasible_split_work != 0
        && feasible_split_work.saturating_mul(32) < total_splits
        && std::env::var("GLRMASK_CLASSIFY_DIRECT_SPARSE_SUFFIX")
            .map(|value| {
                let value = value.trim();
                value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
    let mut suffix_viable_masks = if sparse_suffix_direct {
        Vec::new()
    } else {
        vec![0u64; total_splits * words_per_mask]
    };
    let mut direct_sparse_viable_splits = None::<Vec<(usize, usize, u64)>>;
    let mut candidate_splits = 0usize;
    let mut any_viable_suffix = false;
    let mut internal_reset_completion_splits = 0usize;
    let mut internal_reset_completion_mask = 0u64;
    let mut per_token_continuations_needed: Option<Vec<bool>> = None;
    let use_prepared_suffix_trie = prepared_suffix_trie.is_some()
        && !uses_original_tokenizer
        && total_splits >= PREPARED_SUFFIX_TRIE_MIN_SPLITS;
    let parallel_dense_suffix = !use_prepared_suffix_trie
        && words_per_mask == 1
        && adjacent_pair_index.is_some()
        && !uses_original_tokenizer
        && rayon::current_num_threads() > 1
        && total_splits >= PREPARED_SUFFIX_TRIE_MIN_SPLITS
        && std::env::var("GLRMASK_PARALLEL_CLASSIFY_SUFFIX")
            .map(|value| {
                let value = value.trim();
                value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
    if sparse_suffix_direct {
        let index = adjacent_pair_index.as_ref().expect("checked above");
        let vocab_entries = vocab.entries_map().iter().collect::<Vec<_>>();
        let mut splits = Vec::<(usize, usize, u64)>::with_capacity(feasible_split_work);
        for left in 0u8..=u8::MAX {
            for right in feasible_follow_bytes_by_last_byte[left as usize].iter() {
                for &packed in index.occurrences_for_pair(left, right) {
                    let entry_index = (packed >> 32) as usize;
                    let split_after = packed as u32 as usize;
                    let (&_token_id, bytes) = vocab_entries[entry_index];
                    candidate_splits += 1;
                    let mut state = continuation_reset_state;
                    let mut matched = 0u64;
                    let mut consumed_suffix = true;
                    for &byte in &bytes[split_after + 1..] {
                        if uses_original_tokenizer {
                            if prefix_scanner.future_mask(state)[0] == 0 {
                                consumed_suffix = false;
                                break;
                            }
                            state = prefix_scanner.step(state, byte);
                            if state == PREFIX_DEAD_STATE {
                                consumed_suffix = false;
                                break;
                            }
                            matched |= prefix_scanner.matched_mask(state)[0];
                        } else {
                            if dense_future_masks[state as usize] == 0 {
                                consumed_suffix = false;
                                break;
                            }
                            state = dense_flat_trans[state as usize * 256 + byte as usize];
                            if state == u32::MAX {
                                consumed_suffix = false;
                                break;
                            }
                            matched |= dense_finalizer_masks[state as usize];
                        }
                    }
                    if consumed_suffix {
                        matched |= if uses_original_tokenizer {
                            prefix_scanner.future_mask(state)[0]
                        } else {
                            dense_future_masks[state as usize]
                        };
                    }
                    if matched != 0 {
                        any_viable_suffix = true;
                        splits.push((entry_index, split_after, matched));
                    }
                }
            }
        }
        splits.sort_unstable_by_key(|&(entry_index, split_after, _)| (entry_index, split_after));
        direct_sparse_viable_splits = Some(splits);
    } else if use_prepared_suffix_trie {
        let index = adjacent_pair_index.as_ref().expect("checked above");
        let trie = prepared_suffix_trie
            .as_ref()
            .expect("prepared suffix trie route requires a cached artifact");
        debug_assert_eq!(trie.parents.len(), trie.incoming_bytes.len());
        debug_assert_eq!(trie.split_nodes.len(), index.total_splits);

        // Node IDs are created only after their parent, so one forward pass is
        // an exact shared execution of every suffix prefix from the reset state.
        // `matched_by_node` intentionally survives a dead/no-future edge: the
        // historical scalar loop also retains terminals that finalized before
        // discovering that the remaining bytes cannot continue.
        let (state_by_node, matched_by_node) = execute_prepared_dense_suffix_trie(
            trie,
            continuation_reset_state,
            &dense_flat_trans,
            &dense_finalizer_masks,
            &dense_future_masks,
        );

        let token_continuations_needed =
            per_token_continuations_needed.get_or_insert_with(|| vec![false; vocab.len()]);
        for left in 0u8..=u8::MAX {
            for right in feasible_follow_bytes_by_last_byte[left as usize].iter() {
                for &packed in index.occurrences_for_pair(left, right) {
                    let entry_index = (packed >> 32) as usize;
                    let split_after = packed as u32 as usize;
                    let split_index = index.entry_split_offsets[entry_index] + split_after;
                    let node = trie.split_nodes[split_index] as usize;
                    if node != 0 {
                        let parent = trie.parents[node] as usize;
                        let prior_matched = matched_by_node[parent];
                        if prior_matched != 0 {
                            internal_reset_completion_splits += 1;
                            internal_reset_completion_mask |= prior_matched;
                            token_continuations_needed[entry_index] = true;
                        }
                    }
                    let mut matched = matched_by_node[node];
                    let state = state_by_node[node];
                    if state != u32::MAX {
                        matched |= dense_future_masks[state as usize];
                    }
                    suffix_viable_masks[split_index] = matched;
                    candidate_splits += 1;
                    any_viable_suffix |= matched != 0;
                }
            }
        }
        if super::types::compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][terminal_path_suffix_trie] nodes={} splits={} candidate_splits={} internal_reset_completion_splits={} internal_reset_completion_mask=0x{:x} continuation_tokens={}",
                trie.parents.len(),
                trie.split_nodes.len(),
                candidate_splits,
                internal_reset_completion_splits,
                internal_reset_completion_mask,
                token_continuations_needed.iter().filter(|&&needed| needed).count(),
            );
        }
    } else if parallel_dense_suffix {
        let vocab_entries = vocab.entries_map().iter().collect::<Vec<_>>();
        // Each token owns one disjoint interval of `suffix_viable_masks`, so the
        // suffix proof is embarrassingly parallel.  Compute several token
        // intervals per Rayon worker into private buffers, then copy them back
        // in lexical-vocabulary order.  This is exactly the serial computation:
        // the only cross-token aggregate is OR (`any_viable_suffix`) and the
        // count of inspected feasible splits.
        let workers = rayon::current_num_threads().max(1);
        let token_chunk_size = vocab_entries
            .len()
            .div_ceil(workers.saturating_mul(4))
            .max(64);
        let chunk_results = vocab_entries
            .par_chunks(token_chunk_size)
            .enumerate()
            .map(|(chunk_index, entries)| {
                let first_token = chunk_index * token_chunk_size;
                let split_base = split_offsets[first_token];
                let split_end = split_offsets[first_token + entries.len()];
                let mut local_masks = vec![0u64; split_end - split_base];
                let mut local_candidate_splits = 0usize;
                let mut local_any_viable_suffix = false;
                for (entry_offset, (_token_id, bytes)) in entries.iter().enumerate() {
                    let token_index = first_token + entry_offset;
                    let token_split_start = split_offsets[token_index];
                    for split_after in 0..bytes.len().saturating_sub(1) {
                        if !split_may_host_boundary(bytes, split_after) {
                            continue;
                        }
                        local_candidate_splits += 1;
                        let mut state = continuation_reset_state;
                        let mut matched = 0u64;
                        let mut consumed_suffix = true;
                        for &byte in &bytes[split_after + 1..] {
                            if dense_future_masks[state as usize] == 0 {
                                consumed_suffix = false;
                                break;
                            }
                            state = dense_flat_trans[state as usize * 256 + byte as usize];
                            if state == u32::MAX {
                                consumed_suffix = false;
                                break;
                            }
                            matched |= dense_finalizer_masks[state as usize];
                        }
                        if consumed_suffix {
                            matched |= dense_future_masks[state as usize];
                        }
                        local_any_viable_suffix |= matched != 0;
                        local_masks[token_split_start - split_base + split_after] = matched;
                    }
                }
                (
                    split_base,
                    local_masks,
                    local_candidate_splits,
                    local_any_viable_suffix,
                )
            })
            .collect::<Vec<_>>();
        for (split_base, local_masks, local_candidate_splits, local_any) in chunk_results {
            suffix_viable_masks[split_base..split_base + local_masks.len()]
                .copy_from_slice(&local_masks);
            candidate_splits += local_candidate_splits;
            any_viable_suffix |= local_any;
        }
    } else if words_per_mask == 1 && adjacent_pair_index.is_some() {
        let index = adjacent_pair_index.as_ref().expect("checked above");
        let vocab_entries = vocab.entries_map().iter().collect::<Vec<_>>();
        for left in 0u8..=u8::MAX {
            for right in feasible_follow_bytes_by_last_byte[left as usize].iter() {
                for &packed in index.occurrences_for_pair(left, right) {
                    let entry_index = (packed >> 32) as usize;
                    let split_after = packed as u32 as usize;
                    let (&_token_id, bytes) = vocab_entries[entry_index];
                    candidate_splits += 1;
                    let mut state = continuation_reset_state;
                    let mut matched = 0u64;
                    let mut consumed_suffix = true;
                    for &byte in &bytes[split_after + 1..] {
                        if uses_original_tokenizer {
                            if prefix_scanner.future_mask(state)[0] == 0 {
                                consumed_suffix = false;
                                break;
                            }
                            state = prefix_scanner.step(state, byte);
                            if state == PREFIX_DEAD_STATE {
                                consumed_suffix = false;
                                break;
                            }
                            matched |= prefix_scanner.matched_mask(state)[0];
                        } else {
                            if dense_future_masks[state as usize] == 0 {
                                consumed_suffix = false;
                                break;
                            }
                            state = dense_flat_trans[state as usize * 256 + byte as usize];
                            if state == u32::MAX {
                                consumed_suffix = false;
                                break;
                            }
                            matched |= dense_finalizer_masks[state as usize];
                        }
                    }
                    if consumed_suffix {
                        matched |= if uses_original_tokenizer {
                            prefix_scanner.future_mask(state)[0]
                        } else {
                            dense_future_masks[state as usize]
                        };
                    }
                    any_viable_suffix |= matched != 0;
                    let split_index = index.entry_split_offsets[entry_index] + split_after;
                    suffix_viable_masks[split_index] = matched;
                }
            }
        }
    } else if words_per_mask == 1 {
        for (token_index, bytes) in vocab.entries_map().values().enumerate() {
            let split_start = split_offsets[token_index];
            for split_after in 0..bytes.len().saturating_sub(1) {
                if !split_may_host_boundary(bytes, split_after) {
                    continue;
                }
                candidate_splits += 1;
                let mut state = continuation_reset_state;
                let mut matched = 0u64;
                let mut consumed_suffix = true;
                for &byte in &bytes[split_after + 1..] {
                    if uses_original_tokenizer {
                        if prefix_scanner.future_mask(state)[0] == 0 {
                            consumed_suffix = false;
                            break;
                        }
                        state = prefix_scanner.step(state, byte);
                        if state == PREFIX_DEAD_STATE {
                            consumed_suffix = false;
                            break;
                        }
                        matched |= prefix_scanner.matched_mask(state)[0];
                    } else {
                        if dense_future_masks[state as usize] == 0 {
                            consumed_suffix = false;
                            break;
                        }
                        state = dense_flat_trans[state as usize * 256 + byte as usize];
                        if state == u32::MAX {
                            consumed_suffix = false;
                            break;
                        }
                        matched |= dense_finalizer_masks[state as usize];
                    }
                }
                if consumed_suffix {
                    matched |= if uses_original_tokenizer {
                        prefix_scanner.future_mask(state)[0]
                    } else {
                        dense_future_masks[state as usize]
                    };
                }
                any_viable_suffix |= matched != 0;
                suffix_viable_masks[split_start + split_after] = matched;
            }
        }
    } else {
        let mut matched = vec![0u64; words_per_mask];
        for (token_index, bytes) in vocab.entries_map().values().enumerate() {
            let split_start = split_offsets[token_index];
            for split_after in 0..bytes.len().saturating_sub(1) {
                if !split_may_host_boundary(bytes, split_after) {
                    continue;
                }
                candidate_splits += 1;
                matched.fill(0);
                let mut state = continuation_reset_state;
                let mut consumed_suffix = true;
                for &byte in &bytes[split_after + 1..] {
                    if uses_original_tokenizer {
                        if prefix_scanner
                            .future_mask(state)
                            .iter()
                            .all(|&word| word == 0)
                        {
                            consumed_suffix = false;
                            break;
                        }
                        state = prefix_scanner.step(state, byte);
                        if state == PREFIX_DEAD_STATE {
                            consumed_suffix = false;
                            break;
                        }
                        for word_index in 0..words_per_mask {
                            matched[word_index] |= prefix_scanner.matched_mask(state)[word_index];
                        }
                    } else {
                        let state_base = state as usize * words_per_mask;
                        if dense_future_masks[state_base..state_base + words_per_mask]
                            .iter()
                            .all(|&word| word == 0)
                        {
                            consumed_suffix = false;
                            break;
                        }
                        state = dense_flat_trans[state as usize * 256 + byte as usize];
                        if state == u32::MAX {
                            consumed_suffix = false;
                            break;
                        }
                        let state_base = state as usize * words_per_mask;
                        for word_index in 0..words_per_mask {
                            matched[word_index] |= dense_finalizer_masks[state_base + word_index];
                        }
                    }
                }
                if consumed_suffix {
                    if uses_original_tokenizer {
                        for word_index in 0..words_per_mask {
                            matched[word_index] |= prefix_scanner.future_mask(state)[word_index];
                        }
                    } else {
                        let state_base = state as usize * words_per_mask;
                        for word_index in 0..words_per_mask {
                            matched[word_index] |= dense_future_masks[state_base + word_index];
                        }
                    }
                }
                any_viable_suffix |= matched.iter().any(|&word| word != 0);
                let suffix_base = (split_start + split_after) * words_per_mask;
                suffix_viable_masks[suffix_base..suffix_base + words_per_mask]
                    .copy_from_slice(&matched);
            }
        }
    }
    let suffix_ms = suffix_started_at.elapsed().as_secs_f64() * 1000.0;
    if early_empty_suffix_enabled && !any_viable_suffix {
        if super::types::compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][terminal_path_candidate_dfa] tokens={} candidates={} states={} transitions={} prefix_entries={} suffix_splits={} candidate_splits={} prefix_configs={} split_checks=0 allowed_pairs=0 two_plus=0 compile_ms={:.3} trie_ms={:.3} prefix_ms={:.3} suffix_ms={:.3} combine_ms=0.000 analyze_ms={:.3} total_ms={:.3} early_empty_suffix=true",
                vocab.len(),
                candidate_ids.len(),
                candidate_tokenizer.num_states(),
                candidate_tokenizer.transition_count(),
                split_offsets.last().copied().unwrap_or(0),
                total_splits,
                candidate_splits,
                prefix_scanner.configs.len(),
                compile_ms,
                trie_ms,
                prefix_ms,
                suffix_ms,
                analyze_started_at.elapsed().as_secs_f64() * 1000.0,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return ExactTerminalPathTwoPlus {
            two_plus: BitSet::new(candidates.len()),
            witnesses: vec![None; candidates.len()],
        };
    }

    let combine_started_at = std::time::Instant::now();
    let per_token_boundary_certificate = per_token_continuations_needed.is_some()
        && std::env::var_os("GLRMASK_CLASSIFY_PER_TOKEN_BOUNDARY_CERTIFICATE").is_some();
    let single_boundary_certified = use_prepared_suffix_trie
        && internal_reset_completion_splits == 0
        && std::env::var_os("GLRMASK_CLASSIFY_SINGLE_BOUNDARY_CERTIFICATE").is_some();
    if super::types::compile_profile_enabled() && single_boundary_certified {
        eprintln!(
            "[glrmask/profile][terminal_path_single_boundary_certificate] certified=true"
        );
    }
    let mut allowed_after = vec![0u64; candidate_count * words_per_mask];
    for terminal_1 in 0..candidate_count {
        let blocked = local_disallowed.get(&(terminal_1 as u32));
        for terminal_2 in 0..candidate_count {
            if blocked.is_none_or(|blocked| !blocked.contains(terminal_2)) {
                allowed_after[terminal_1 * words_per_mask + (terminal_2 >> 6)] |=
                    1u64 << (terminal_2 & 63);
            }
        }
    }
    let small_mask_lut = (words_per_mask == 1
        && candidate_count <= 16
        && std::env::var_os("GLRMASK_CLASSIFY_DISABLE_SMALL_MASK_LUT").is_none())
    .then(|| {
        let mask_count = 1usize << candidate_count;
        let valid_mask = mask_count as u64 - 1;
        let mut allowed_union_by_matched = vec![0u64; mask_count];
        for mask in 1..mask_count {
            let bit = mask & mask.wrapping_neg();
            let terminal = bit.trailing_zeros() as usize;
            allowed_union_by_matched[mask] =
                allowed_union_by_matched[mask ^ bit] | allowed_after[terminal];
        }
        let mut left_with_follow_by_suffix = vec![0u64; mask_count];
        for suffix in 1..mask_count {
            let suffix_word = suffix as u64;
            let mut left = 0u64;
            for terminal in 0..candidate_count {
                if suffix_word & allowed_after[terminal] != 0 {
                    left |= 1u64 << terminal;
                }
            }
            left_with_follow_by_suffix[suffix] = left;
        }
        (
            allowed_union_by_matched,
            left_with_follow_by_suffix,
            valid_mask,
        )
    });

    let collect_witnesses = std::env::var_os("GLRMASK_DUMP_TERMINAL_PATH_WITNESSES").is_some();
    let mut local_two_plus_words = vec![0u64; words_per_mask];
    let mut local_witnesses = vec![None; candidate_count];
    let mut split_checks = 0usize;
    let mut allowed_pairs = 0usize;
    if words_per_mask == 1 {
        // When the adjacent-byte filter leaves only a tiny number of viable
        // suffixes, do not walk every token merely to rediscover those split
        // positions. Collect the non-empty feasible suffixes directly from the
        // pair index, group them by token, then run the ordinary exact prefix /
        // continuation machine only across those selected tokens. Continuation
        // state still advances over every byte up to the last selected split,
        // so multiple boundaries inside one token are represented exactly.
        let sparse_viable_splits = if let Some(splits) = direct_sparse_viable_splits.take() {
            Some(splits)
        } else {
            adjacent_pair_index.as_ref().and_then(|index| {
            let enabled = std::env::var("GLRMASK_CLASSIFY_SPARSE_COMBINE")
                .map(|value| {
                    let value = value.trim();
                    value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
                })
                .unwrap_or(true);
            if !enabled || candidate_splits == 0 || candidate_splits.saturating_mul(32) >= total_splits {
                return None;
            }
            let mut splits = Vec::<(usize, usize, u64)>::new();
            for left in 0u8..=u8::MAX {
                for right in feasible_follow_bytes_by_last_byte[left as usize].iter() {
                    for &packed in index.occurrences_for_pair(left, right) {
                        let entry_index = (packed >> 32) as usize;
                        let split_after = packed as u32 as usize;
                        let split_index = index.entry_split_offsets[entry_index] + split_after;
                        let suffix_word = suffix_viable_masks[split_index];
                        if suffix_word != 0 {
                            splits.push((entry_index, split_after, suffix_word));
                        }
                    }
                }
            }
            if splits.is_empty() {
                return None;
            }
            splits.sort_unstable_by_key(|&(entry_index, split_after, _)| (entry_index, split_after));
            Some(splits)
            })
        };
        let parallel_combine = !collect_witnesses
            && !uses_original_tokenizer
            && std::env::var_os("GLRMASK_CLASSIFY_PARALLEL_COMBINE").is_some()
            && rayon::current_num_threads() > 1
            && vocab.len() >= 256;
        if let Some(sparse_splits) = sparse_viable_splits.as_ref() {
            let sparse_started_at = std::time::Instant::now();
            let token_entries = vocab.entries_map().iter().collect::<Vec<_>>();
            let mut continuations = Vec::<(u32, u64)>::new();
            let mut next_continuations = Vec::<(u32, u64)>::new();
            let mut cursor = 0usize;
            while cursor < sparse_splits.len() {
                let entry_index = sparse_splits[cursor].0;
                let begin = cursor;
                cursor += 1;
                while cursor < sparse_splits.len() && sparse_splits[cursor].0 == entry_index {
                    cursor += 1;
                }
                let selected = &sparse_splits[begin..cursor];
                let (&token_id, bytes) = token_entries[entry_index];
                let last_split = selected.last().expect("non-empty sparse split group").1;
                let token_needs_continuations = !per_token_boundary_certificate
                    || per_token_continuations_needed
                        .as_ref()
                        .expect("per-token certificate requires prepared suffix metadata")[entry_index];
                continuations.clear();
                next_continuations.clear();
                let mut prefix_state = prefix_start;
                let mut selected_cursor = 0usize;
                for split_after in 0..=last_split {
                    let byte = bytes[split_after];
                    prefix_state = if let (Some(trie), Some(states)) =
                        (prepared_prefix_trie.as_ref(), prepared_prefix_states.as_ref())
                    {
                        let split_index = split_offsets[entry_index] + split_after;
                        states[trie.split_nodes[split_index] as usize]
                    } else {
                        prefix_scanner.step(prefix_state, byte)
                    };
                    let mut matched = if prefix_state == PREFIX_DEAD_STATE {
                        0
                    } else {
                        prefix_scanner.matched_mask(prefix_state)[0]
                    };
                    next_continuations.clear();
                    if token_needs_continuations {
                        for &(state, active) in &continuations {
                            let target = if uses_original_tokenizer {
                                prefix_scanner.step(state, byte)
                            } else {
                                dense_flat_trans[state as usize * 256 + byte as usize]
                            };
                            if target == PREFIX_DEAD_STATE || target == u32::MAX {
                                continue;
                            }
                            let (target_matched, target_future) = if uses_original_tokenizer {
                                (
                                    prefix_scanner.matched_mask(target)[0],
                                    prefix_scanner.future_mask(target)[0],
                                )
                            } else {
                                (
                                    dense_finalizer_masks[target as usize],
                                    dense_future_masks[target as usize],
                                )
                            };
                            matched |= target_matched & active;
                            merge_word_continuation(
                                &mut next_continuations,
                                target,
                                target_future & active,
                            );
                        }
                    }

                    let mut followers = 0u64;
                    if selected_cursor < selected.len() && selected[selected_cursor].1 == split_after {
                        let suffix_word = selected[selected_cursor].2;
                        selected_cursor += 1;
                        if matched != 0 {
                            split_checks += 1;
                            if collect_witnesses {
                                let matched_mask = [matched];
                                let suffix = [suffix_word];
                                let mut follower_mask = [0u64; 1];
                                allowed_pairs += accumulate_terminal_path_boundaries(
                                    &matched_mask,
                                    &suffix,
                                    &allowed_after,
                                    words_per_mask,
                                    candidate_count,
                                    &mut local_two_plus_words,
                                    &mut follower_mask,
                                    Some((
                                        local_witnesses.as_mut_slice(),
                                        token_id,
                                        bytes.as_slice(),
                                        split_after + 1,
                                    )),
                                );
                                followers = follower_mask[0];
                            } else if let Some((allowed_union, left_with_follow, valid_mask)) =
                                small_mask_lut.as_ref()
                            {
                                let matched_word = matched & *valid_mask;
                                let suffix_word = suffix_word & *valid_mask;
                                followers = suffix_word & allowed_union[matched_word as usize];
                                local_two_plus_words[0] |= followers
                                    | (matched_word & left_with_follow[suffix_word as usize]);
                                allowed_pairs += followers.count_ones() as usize;
                            } else {
                                let mut pending_matched = matched;
                                while pending_matched != 0 {
                                    let terminal_1 = pending_matched.trailing_zeros() as usize;
                                    pending_matched &= pending_matched - 1;
                                    if terminal_1 >= candidate_count {
                                        continue;
                                    }
                                    let accepted = suffix_word & allowed_after[terminal_1];
                                    if accepted == 0 {
                                        continue;
                                    }
                                    allowed_pairs += accepted.count_ones() as usize;
                                    followers |= accepted;
                                    local_two_plus_words[0] |= accepted | (1u64 << terminal_1);
                                }
                            }
                        }
                    }
                    if token_needs_continuations {
                        merge_word_continuation(
                            &mut next_continuations,
                            continuation_reset_state,
                            followers,
                        );
                        std::mem::swap(&mut continuations, &mut next_continuations);
                    }
                }
            }
            if super::types::compile_profile_enabled() {
                let sparse_tokens = sparse_splits
                    .iter()
                    .map(|&(entry, _, _)| entry)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
                eprintln!(
                    "[glrmask/profile][terminal_path_sparse_combine] viable_splits={} tokens={} split_checks={} allowed_pairs={} ms={:.3}",
                    sparse_splits.len(),
                    sparse_tokens,
                    split_checks,
                    allowed_pairs,
                    sparse_started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
        } else if parallel_combine {
            let token_entries = vocab.entries_map().values().collect::<Vec<_>>();

            // The prefix powerset is the only mutable automaton in this hot
            // path. Materialize its matched-terminal mask once per candidate
            // split, serially, then let all token-local continuation work use
            // immutable dense DFA tables in parallel.
            let prefix_precompute_started_at = std::time::Instant::now();
            let mut prefix_matched_at_split = vec![0u64; total_splits];
            for (token_index, bytes) in token_entries.iter().enumerate() {
                let split_start = split_offsets[token_index];
                let Some(last_candidate_split) = (0..bytes.len().saturating_sub(1))
                    .rfind(|&split_after| split_may_host_boundary(bytes, split_after))
                else {
                    continue;
                };
                let mut prefix_state = prefix_start;
                for split_after in 0..=last_candidate_split {
                    prefix_state = prefix_scanner.step(prefix_state, bytes[split_after]);
                    if prefix_state == PREFIX_DEAD_STATE {
                        break;
                    }
                    prefix_matched_at_split[split_start + split_after] =
                        prefix_scanner.matched_mask(prefix_state)[0];
                }
            }
            let _prefix_precompute_ms =
                prefix_precompute_started_at.elapsed().as_secs_f64() * 1000.0;

            let (two_plus_word, total_split_checks, total_allowed_pairs) = token_entries
                .par_iter()
                .enumerate()
                .map_init(
                    || {
                        (
                            DenseWordContinuations::new(candidate_tokenizer.num_states() as usize),
                            DenseWordContinuations::new(candidate_tokenizer.num_states() as usize),
                        )
                    },
                    |(continuations, next_continuations), (token_index, bytes)| {
                        continuations.clear();
                        next_continuations.clear();
                        let token_needs_continuations = !per_token_boundary_certificate
                            || per_token_continuations_needed
                                .as_ref()
                                .expect("per-token certificate requires prepared suffix metadata")[token_index];
                        let split_start = split_offsets[token_index];
                        let Some(last_candidate_split) = (0..bytes.len().saturating_sub(1))
                            .rfind(|&split_after| split_may_host_boundary(bytes, split_after))
                        else {
                            return (0u64, 0usize, 0usize);
                        };
                        let split_end = split_start + last_candidate_split + 1;
                        let mut local_two_plus = 0u64;
                        let mut local_split_checks = 0usize;
                        let mut local_allowed_pairs = 0usize;
                        for (split_after, split_index) in (split_start..split_end).enumerate() {
                            let byte = bytes[split_after];
                            let mut matched = prefix_matched_at_split[split_index];
                            next_continuations.clear();
                            if token_needs_continuations {
                                for &state in &continuations.touched {
                                    let active = continuations.masks[state as usize];
                                    let target =
                                        dense_flat_trans[state as usize * 256 + byte as usize];
                                    if target == u32::MAX {
                                        continue;
                                    }
                                    matched |= dense_finalizer_masks[target as usize] & active;
                                    let live = dense_future_masks[target as usize] & active;
                                    next_continuations.merge(target, live);
                                }
                            }

                            let mut followers = 0u64;
                            let suffix_word = suffix_viable_masks[split_index];
                            if matched != 0 && suffix_word != 0 {
                                local_split_checks += 1;
                                if let Some((allowed_union, left_with_follow, valid_mask)) =
                                    small_mask_lut.as_ref()
                                {
                                    let matched_word = matched & *valid_mask;
                                    let suffix_word = suffix_word & *valid_mask;
                                    followers = suffix_word & allowed_union[matched_word as usize];
                                    local_two_plus |= followers
                                        | (matched_word & left_with_follow[suffix_word as usize]);
                                    local_allowed_pairs += followers.count_ones() as usize;
                                } else {
                                    let mut pending_matched = matched;
                                    while pending_matched != 0 {
                                        let terminal_1 = pending_matched.trailing_zeros() as usize;
                                        pending_matched &= pending_matched - 1;
                                        if terminal_1 >= candidate_count {
                                            continue;
                                        }
                                        let accepted = suffix_word & allowed_after[terminal_1];
                                        if accepted == 0 {
                                            continue;
                                        }
                                        local_allowed_pairs += accepted.count_ones() as usize;
                                        followers |= accepted;
                                        local_two_plus |= accepted | (1u64 << terminal_1);
                                    }
                                }
                            }
                            if token_needs_continuations {
                                next_continuations.merge(continuation_reset_state, followers);
                                std::mem::swap(continuations, next_continuations);
                            }
                        }
                        (local_two_plus, local_split_checks, local_allowed_pairs)
                    },
                )
                .reduce(
                    || (0u64, 0usize, 0usize),
                    |left, right| (left.0 | right.0, left.1 + right.1, left.2 + right.2),
                );
            local_two_plus_words[0] = two_plus_word;
            split_checks = total_split_checks;
            allowed_pairs = total_allowed_pairs;
        } else if !collect_witnesses && !uses_original_tokenizer && single_boundary_certified {
            let mut matched = [0u64; 1];
            let mut followers = [0u64; 1];
            for (token_index, bytes) in vocab.entries_map().values().enumerate() {
                let split_start = split_offsets[token_index];
                let Some(last_candidate_split) = (0..bytes.len().saturating_sub(1))
                    .rfind(|&split_after| split_may_host_boundary(bytes, split_after))
                else {
                    continue;
                };
                let split_end = split_start + last_candidate_split + 1;
                let mut prefix_state = prefix_start;
                for (_split_after, split_index) in (split_start..split_end).enumerate() {
                    let byte = bytes[split_index - split_start];
                    prefix_state = if let (Some(trie), Some(states)) =
                        (prepared_prefix_trie.as_ref(), prepared_prefix_states.as_ref())
                    {
                        states[trie.split_nodes[split_index] as usize]
                    } else {
                        prefix_scanner.step(prefix_state, byte)
                    };
                    matched[0] = if prefix_state == PREFIX_DEAD_STATE {
                        0
                    } else {
                        prefix_scanner.matched_mask(prefix_state)[0]
                    };
                    followers[0] = 0;
                    let suffix_word = suffix_viable_masks[split_index];
                    if matched[0] != 0 && suffix_word != 0 {
                        split_checks += 1;
                        if let Some((allowed_union, left_with_follow, valid_mask)) =
                            small_mask_lut.as_ref()
                        {
                            let matched_word = matched[0] & *valid_mask;
                            let suffix_word = suffix_word & *valid_mask;
                            followers[0] = suffix_word & allowed_union[matched_word as usize];
                            local_two_plus_words[0] |= followers[0]
                                | (matched_word & left_with_follow[suffix_word as usize]);
                            allowed_pairs += followers[0].count_ones() as usize;
                        } else {
                            let mut pending_matched = matched[0];
                            while pending_matched != 0 {
                                let terminal_1 = pending_matched.trailing_zeros() as usize;
                                pending_matched &= pending_matched - 1;
                                if terminal_1 >= candidate_count {
                                    continue;
                                }
                                let accepted = suffix_word & allowed_after[terminal_1];
                                if accepted == 0 {
                                    continue;
                                }
                                allowed_pairs += accepted.count_ones() as usize;
                                followers[0] |= accepted;
                                local_two_plus_words[0] |= accepted | (1u64 << terminal_1);
                            }
                        }
                    }
                }
            }
        } else if !uses_original_tokenizer
            && std::env::var_os("GLRMASK_CLASSIFY_DENSE_WORD_CONTINUATIONS").is_some()
        {
            let num_dense_states = candidate_tokenizer.num_states() as usize;
            let mut continuations = DenseWordContinuations::new(num_dense_states);
            let mut next_continuations = DenseWordContinuations::new(num_dense_states);
            let mut matched = [0u64; 1];
            let mut followers = [0u64; 1];
            let mut continuation_entries = 0usize;
            let mut continuation_max = 0usize;
            let mut continuation_match_events = 0usize;
            let mut continuation_match_bits = 0usize;
            for (token_index, (&token_id, bytes)) in vocab.entries_map().iter().enumerate() {
                continuations.clear();
                next_continuations.clear();
                let split_start = split_offsets[token_index];
                let Some(last_candidate_split) = (0..bytes.len().saturating_sub(1))
                    .rfind(|&split_after| split_may_host_boundary(bytes, split_after))
                else {
                    continue;
                };
                let split_end = split_start + last_candidate_split + 1;
                let mut prefix_state = prefix_start;
                for (split_after, split_index) in (split_start..split_end).enumerate() {
                    let byte = bytes[split_after];
                    prefix_state = if let (Some(trie), Some(states)) =
                        (prepared_prefix_trie.as_ref(), prepared_prefix_states.as_ref())
                    {
                        states[trie.split_nodes[split_index] as usize]
                    } else {
                        prefix_scanner.step(prefix_state, byte)
                    };
                    matched[0] = if prefix_state == PREFIX_DEAD_STATE {
                        0
                    } else {
                        prefix_scanner.matched_mask(prefix_state)[0]
                    };
                    next_continuations.clear();
                    continuation_entries += continuations.touched.len();
                    continuation_max = continuation_max.max(continuations.touched.len());
                    for &state in &continuations.touched {
                        let active = continuations.masks[state as usize];
                        let target = dense_flat_trans[state as usize * 256 + byte as usize];
                        if target == u32::MAX {
                            continue;
                        }
                        let continuation_matched = dense_finalizer_masks[target as usize] & active;
                        continuation_match_events += usize::from(continuation_matched != 0);
                        continuation_match_bits += continuation_matched.count_ones() as usize;
                        matched[0] |= continuation_matched;
                        let live = dense_future_masks[target as usize] & active;
                        next_continuations.merge(target, live);
                    }

                    followers[0] = 0;
                    let suffix_word = suffix_viable_masks[split_index];
                    if matched[0] != 0 && suffix_word != 0 {
                        split_checks += 1;
                        if collect_witnesses {
                            let suffix = [suffix_word];
                            allowed_pairs += accumulate_terminal_path_boundaries(
                                &matched,
                                &suffix,
                                &allowed_after,
                                words_per_mask,
                                candidate_count,
                                &mut local_two_plus_words,
                                &mut followers,
                                Some((
                                    local_witnesses.as_mut_slice(),
                                    token_id,
                                    bytes.as_slice(),
                                    split_after + 1,
                                )),
                            );
                        } else if let Some((allowed_union, left_with_follow, valid_mask)) =
                            small_mask_lut.as_ref()
                        {
                            let matched_word = matched[0] & *valid_mask;
                            let suffix_word = suffix_word & *valid_mask;
                            followers[0] = suffix_word & allowed_union[matched_word as usize];
                            local_two_plus_words[0] |= followers[0]
                                | (matched_word & left_with_follow[suffix_word as usize]);
                            allowed_pairs += followers[0].count_ones() as usize;
                        } else {
                            let mut pending_matched = matched[0];
                            while pending_matched != 0 {
                                let terminal_1 = pending_matched.trailing_zeros() as usize;
                                pending_matched &= pending_matched - 1;
                                if terminal_1 >= candidate_count {
                                    continue;
                                }
                                let accepted = suffix_word & allowed_after[terminal_1];
                                if accepted == 0 {
                                    continue;
                                }
                                allowed_pairs += accepted.count_ones() as usize;
                                followers[0] |= accepted;
                                local_two_plus_words[0] |= accepted | (1u64 << terminal_1);
                            }
                        }
                    }
                    next_continuations.merge(continuation_reset_state, followers[0]);
                    std::mem::swap(&mut continuations, &mut next_continuations);
                }
            }
            if super::types::compile_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][terminal_path_dense_continuations] states={} entries={} max={} match_events={} match_bits={}",
                    num_dense_states,
                    continuation_entries,
                    continuation_max,
                    continuation_match_events,
                    continuation_match_bits,
                );
            }
        } else {
            let mut continuations = Vec::<(u32, u64)>::new();
            let mut next_continuations = Vec::<(u32, u64)>::new();
            let mut matched = [0u64; 1];
            let mut followers = [0u64; 1];
            for (token_index, (&token_id, bytes)) in vocab.entries_map().iter().enumerate() {
                let token_needs_continuations = !per_token_boundary_certificate
                    || per_token_continuations_needed
                        .as_ref()
                        .expect("per-token certificate requires prepared suffix metadata")[token_index];
                continuations.clear();
                let split_start = split_offsets[token_index];
                let Some(last_candidate_split) = (0..bytes.len().saturating_sub(1))
                    .rfind(|&split_after| split_may_host_boundary(bytes, split_after))
                else {
                    continue;
                };
                let split_end = split_start + last_candidate_split + 1;
                let mut prefix_state = prefix_start;
                for (split_after, split_index) in (split_start..split_end).enumerate() {
                    let byte = bytes[split_after];
                    prefix_state = if let (Some(trie), Some(states)) =
                        (prepared_prefix_trie.as_ref(), prepared_prefix_states.as_ref())
                    {
                        states[trie.split_nodes[split_index] as usize]
                    } else {
                        prefix_scanner.step(prefix_state, byte)
                    };
                    matched[0] = if prefix_state == PREFIX_DEAD_STATE {
                        0
                    } else {
                        prefix_scanner.matched_mask(prefix_state)[0]
                    };
                    next_continuations.clear();
                    if token_needs_continuations {
                    for &(state, active) in &continuations {
                        let target = if uses_original_tokenizer {
                            prefix_scanner.step(state, byte)
                        } else {
                            dense_flat_trans[state as usize * 256 + byte as usize]
                        };
                        if target == PREFIX_DEAD_STATE || target == u32::MAX {
                            continue;
                        }
                        let (target_matched, target_future) = if uses_original_tokenizer {
                            (
                                prefix_scanner.matched_mask(target)[0],
                                prefix_scanner.future_mask(target)[0],
                            )
                        } else {
                            (
                                dense_finalizer_masks[target as usize],
                                dense_future_masks[target as usize],
                            )
                        };
                        matched[0] |= target_matched & active;
                        let live = target_future & active;
                        merge_word_continuation(&mut next_continuations, target, live);
                    }
                    }

                    followers[0] = 0;
                    let suffix_word = suffix_viable_masks[split_index];
                    if matched[0] != 0 && suffix_word != 0 {
                        split_checks += 1;
                        if collect_witnesses {
                            let suffix = [suffix_word];
                            allowed_pairs += accumulate_terminal_path_boundaries(
                                &matched,
                                &suffix,
                                &allowed_after,
                                words_per_mask,
                                candidate_count,
                                &mut local_two_plus_words,
                                &mut followers,
                                Some((
                                    local_witnesses.as_mut_slice(),
                                    token_id,
                                    bytes.as_slice(),
                                    split_after + 1,
                                )),
                            );
                        } else if let Some((allowed_union, left_with_follow, valid_mask)) =
                            small_mask_lut.as_ref()
                        {
                            let matched_word = matched[0] & *valid_mask;
                            let suffix_word = suffix_word & *valid_mask;
                            followers[0] = suffix_word & allowed_union[matched_word as usize];
                            local_two_plus_words[0] |= followers[0]
                                | (matched_word & left_with_follow[suffix_word as usize]);
                            // Diagnostic count only; correctness depends on the masks above.
                            allowed_pairs += followers[0].count_ones() as usize;
                        } else {
                            let mut pending_matched = matched[0];
                            while pending_matched != 0 {
                                let terminal_1 = pending_matched.trailing_zeros() as usize;
                                pending_matched &= pending_matched - 1;
                                if terminal_1 >= candidate_count {
                                    continue;
                                }
                                let accepted = suffix_word & allowed_after[terminal_1];
                                if accepted == 0 {
                                    continue;
                                }
                                allowed_pairs += accepted.count_ones() as usize;
                                followers[0] |= accepted;
                                local_two_plus_words[0] |= accepted | (1u64 << terminal_1);
                            }
                        }
                    }
                    if token_needs_continuations {
                        merge_word_continuation(
                            &mut next_continuations,
                            continuation_reset_state,
                            followers[0],
                        );
                        std::mem::swap(&mut continuations, &mut next_continuations);
                    }
                }
            }
        }
    } else {
        let mut continuations = Vec::<(u32, Box<[u64]>)>::new();
        let mut next_continuations = Vec::<(u32, Box<[u64]>)>::new();
        let mut matched = vec![0u64; words_per_mask];
        let mut live = vec![0u64; words_per_mask];
        let mut followers = vec![0u64; words_per_mask];
        for (token_index, (&token_id, bytes)) in vocab.entries_map().iter().enumerate() {
            continuations.clear();
            let split_start = split_offsets[token_index];
            let Some(last_candidate_split) = (0..bytes.len().saturating_sub(1))
                .rfind(|&split_after| split_may_host_boundary(bytes, split_after))
            else {
                continue;
            };
            let split_end = split_start + last_candidate_split + 1;
            let mut prefix_state = prefix_start;
            for (split_after, split_index) in (split_start..split_end).enumerate() {
                let byte = bytes[split_after];
                prefix_state = prefix_scanner.step(prefix_state, byte);
                if prefix_state == PREFIX_DEAD_STATE {
                    matched.fill(0);
                } else {
                    matched.copy_from_slice(prefix_scanner.matched_mask(prefix_state));
                }
                next_continuations.clear();
                for (state, active) in &continuations {
                    let target = if uses_original_tokenizer {
                        prefix_scanner.step(*state, byte)
                    } else {
                        dense_flat_trans[*state as usize * 256 + byte as usize]
                    };
                    if target == PREFIX_DEAD_STATE || target == u32::MAX {
                        continue;
                    }
                    for word_index in 0..words_per_mask {
                        let target_matched = if uses_original_tokenizer {
                            prefix_scanner.matched_mask(target)[word_index]
                        } else {
                            dense_finalizer_masks[target as usize * words_per_mask + word_index]
                        };
                        let target_future = if uses_original_tokenizer {
                            prefix_scanner.future_mask(target)[word_index]
                        } else {
                            dense_future_masks[target as usize * words_per_mask + word_index]
                        };
                        matched[word_index] |= target_matched & active[word_index];
                        live[word_index] = target_future & active[word_index];
                    }
                    merge_mask_continuation(&mut next_continuations, target, &live);
                }

                followers.fill(0);
                let suffix_base = split_index * words_per_mask;
                let suffix =
                    &suffix_viable_masks[suffix_base..suffix_base + words_per_mask];
                if matched.iter().any(|&word| word != 0)
                    && suffix.iter().any(|&word| word != 0)
                {
                    split_checks += 1;
                    let witness_context = collect_witnesses.then_some((
                        local_witnesses.as_mut_slice(),
                        token_id,
                        bytes.as_slice(),
                        split_after + 1,
                    ));
                    allowed_pairs += accumulate_terminal_path_boundaries(
                        &matched,
                        suffix,
                        &allowed_after,
                        words_per_mask,
                        candidate_count,
                        &mut local_two_plus_words,
                        &mut followers,
                        witness_context,
                    );
                }
                merge_mask_continuation(
                    &mut next_continuations,
                    continuation_reset_state,
                    &followers,
                );
                std::mem::swap(&mut continuations, &mut next_continuations);
            }
        }
    }
    let mut local_two_plus = BitSet::new(candidate_count);
    for (word_index, &word) in local_two_plus_words.iter().enumerate() {
        let mut pending = word;
        while pending != 0 {
            let terminal = word_index * 64 + pending.trailing_zeros() as usize;
            pending &= pending - 1;
            if terminal < candidate_count {
                local_two_plus.set(terminal);
            }
        }
    }
    let local = ExactTerminalPathTwoPlus {
        two_plus: local_two_plus,
        witnesses: local_witnesses,
    };
    let combine_ms = combine_started_at.elapsed().as_secs_f64() * 1000.0;
    let analyze_ms = analyze_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut two_plus = BitSet::new(candidates.len());
    let mut witnesses = vec![None; candidates.len()];
    for local_terminal in local.two_plus.iter() {
        two_plus.set(candidate_ids[local_terminal]);
    }
    for (local_terminal, witness) in local.witnesses.into_iter().enumerate() {
        let Some(mut witness) = witness else {
            continue;
        };
        witness.terminal_1 = candidate_ids[witness.terminal_1];
        witness.terminal_2 = candidate_ids[witness.terminal_2];
        witnesses[candidate_ids[local_terminal]] = Some(witness);
    }
    if super::types::compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][terminal_path_candidate_dfa] tokens={} candidates={} states={} transitions={} uses_original_tokenizer={} automatic_subset_tokenizer={} feasible_split_work={} prefix_entries={} suffix_splits={} candidate_splits={} prefix_configs={} split_checks={} allowed_pairs={} two_plus={} compile_ms={:.3} trie_ms={:.3} prefix_ms={:.3} suffix_ms={:.3} combine_ms={:.3} analyze_ms={:.3} total_ms={:.3}",
            vocab.entries_map().len(),
            candidate_ids.len(),
            candidate_tokenizer.num_states(),
            candidate_tokenizer.transition_count(),
            uses_original_tokenizer,
            automatic_subset_tokenizer,
            feasible_split_work,
            total_splits,
            total_splits,
            candidate_splits,
            prefix_scanner.configs.len(),
            split_checks,
            allowed_pairs,
            two_plus.count_ones(),
            compile_ms,
            trie_ms,
            prefix_ms,
            suffix_ms,
            combine_ms,
            analyze_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    ExactTerminalPathTwoPlus { two_plus, witnesses }
}

fn exact_terminal_path_two_plus_deterministic<S: ExactBoundaryDeterministicScanner>(
    scanner: &S,
    transitions_by_byte: &[u32],
    sparse_transitions_by_byte: &[Vec<(u32, u32)>],
    reverse_transitions_by_byte: &[ReverseByteTransitions],
    token_entries: &[(u32, &[u8])],
    active: &BitSet,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    active_start_states: &[u32],
) -> ExactTerminalPathTwoPlus {
    exact_terminal_path_two_plus_with_suffix(
        scanner,
        transitions_by_byte,
        sparse_transitions_by_byte,
        reverse_transitions_by_byte,
        token_entries,
        active,
        disallowed_follows,
        active_start_states,
        |suffix, active, candidates| {
            suffix_terminal_candidates_deterministic(scanner, suffix, active, candidates);
        },
    )
}

const PREFIX_UNKNOWN_STATE: u32 = u32::MAX - 1;
const PREFIX_DEAD_STATE: u32 = u32::MAX;

#[derive(Default)]
struct TerminalPathTrieNode {
    children: Vec<(u8, usize)>,
}

struct TerminalPathTrieBuilder {
    nodes: Vec<TerminalPathTrieNode>,
    edges: FxHashMap<u64, usize>,
}

impl TerminalPathTrieBuilder {
    fn with_capacity(edge_capacity: usize) -> Self {
        let mut edges = FxHashMap::default();
        edges.reserve(edge_capacity);
        Self {
            nodes: vec![TerminalPathTrieNode::default()],
            edges,
        }
    }

    #[inline]
    fn insert_byte(&mut self, parent: usize, byte: u8) -> usize {
        let key = ((parent as u64) << 8) | byte as u64;
        if let Some(&child) = self.edges.get(&key) {
            return child;
        }
        let child = self.nodes.len();
        self.nodes.push(TerminalPathTrieNode::default());
        self.nodes[parent].children.push((byte, child));
        self.edges.insert(key, child);
        child
    }

    fn finish(self) -> Vec<TerminalPathTrieNode> {
        self.nodes
    }
}

fn suffix_terminal_candidates_deterministic<S: ExactBoundaryDeterministicScanner>(
    scanner: &S,
    suffix: &[u8],
    active: &BitSet,
    candidates: &mut BitSet,
) {
    candidates.clear_all();
    let mut state = scanner.start_state();
    let mut consumed_suffix = true;
    for &byte in suffix {
        if !scanner.futures_intersect(state, active) {
            consumed_suffix = false;
            break;
        }
        let next = scanner.transition(state, byte);
        if next == u32::MAX {
            consumed_suffix = false;
            break;
        }
        state = next;
        scanner.for_each_finalizer(state, |terminal| {
            if active.contains(terminal) {
                candidates.set(terminal);
            }
        });
    }
    if consumed_suffix {
        scanner.for_each_future(state, |terminal| {
            if active.contains(terminal) {
                candidates.set(terminal);
            }
        });
    }
}

fn exact_terminal_path_two_plus_with_suffix<S, F>(
    scanner: &S,
    transitions_by_byte: &[u32],
    sparse_transitions_by_byte: &[Vec<(u32, u32)>],
    reverse_transitions_by_byte: &[ReverseByteTransitions],
    token_entries: &[(u32, &[u8])],
    active: &BitSet,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    active_start_states: &[u32],
    mut fill_suffix_candidates: F,
) -> ExactTerminalPathTwoPlus
where
    S: ExactBoundaryDeterministicScanner,
    F: FnMut(&[u8], &BitSet, &mut BitSet),
{
    let total_started_at = std::time::Instant::now();
    let mut nodes = vec![ExactBoundaryPrefixNode::default()];
    let total_token_bytes = token_entries
        .iter()
        .map(|(_, bytes)| bytes.len().saturating_sub(1))
        .sum();
    let mut token_path_offsets = Vec::with_capacity(token_entries.len() + 1);
    let mut token_paths = Vec::with_capacity(total_token_bytes);
    token_path_offsets.push(0);
    for &(_, bytes) in token_entries {
        let mut node = 0usize;
        for &byte in bytes.iter().take(bytes.len().saturating_sub(1)) {
            node = insert_exact_boundary_prefix(&mut nodes, node, byte);
            token_paths.push(node);
        }
        token_path_offsets.push(token_paths.len());
    }
    let trie_ms = total_started_at.elapsed().as_secs_f64() * 1000.0;

    let populate_started_at = std::time::Instant::now();
    let mut state_seen = vec![0u32; scanner.num_states()];
    let mut terminal_seen = vec![0u32; active.len()];
    let mut state_stamp = 0u32;
    let mut terminal_stamp = 0u32;
    let mut class_stamp = 0u32;
    let mut states_scanned = 0usize;
    let mut reached_states = 0usize;
    let mut finalizer_terminals_scanned = 0usize;
    let mut allowed_follow_classes = vec![active.clone()];
    let mut blocked_class_cache = HashMap::<&BitSet, usize>::new();
    let allowed_class_by_terminal = (0..active.len())
        .map(|terminal| {
            if !active.contains(terminal) {
                return None;
            }
            let Some(blocked) = disallowed_follows
                .get(&(terminal as u32))
                .filter(|blocked| !blocked.is_empty())
            else {
                return Some(0);
            };
            Some(*blocked_class_cache.entry(blocked).or_insert_with(|| {
                let class = allowed_follow_classes.len();
                allowed_follow_classes.push(active.difference(blocked));
                class
            }))
        })
        .collect::<Vec<_>>();
    let mut class_seen = vec![0u32; allowed_follow_classes.len()];
    let mut matched_classes = Vec::with_capacity(allowed_follow_classes.len());
    let mut state_buffers = Vec::<Vec<u32>>::new();
    let mut source_seen_by_depth = Vec::<Vec<u32>>::new();
    let mut source_stamps = Vec::<u32>::new();
    populate_exact_boundary_prefixes(
        scanner,
        transitions_by_byte,
        sparse_transitions_by_byte,
        reverse_transitions_by_byte,
        active,
        &mut nodes,
        0,
        active_start_states,
        0,
        &mut state_seen,
        &mut state_stamp,
        &mut terminal_seen,
        &mut terminal_stamp,
        &mut class_seen,
        &mut class_stamp,
        &mut states_scanned,
        &mut reached_states,
        &mut finalizer_terminals_scanned,
        &allowed_class_by_terminal,
        &allowed_follow_classes,
        &mut matched_classes,
        &mut state_buffers,
        &mut source_seen_by_depth,
        &mut source_stamps,
    );
    let populate_ms = populate_started_at.elapsed().as_secs_f64() * 1000.0;

    let evaluate_started_at = std::time::Instant::now();
    let mut two_plus = BitSet::new(active.len());
    let mut witnesses = vec![None; active.len()];
    let mut suffix_candidates = BitSet::new(active.len());
    let mut suffixes_evaluated = 0usize;
    let mut allowed_pairs = 0usize;
    for (token_index, &(token_id, bytes)) in token_entries.iter().enumerate() {
        for (split_after, &node) in token_paths
            [token_path_offsets[token_index]..token_path_offsets[token_index + 1]]
            .iter()
            .enumerate()
        {
            if nodes[node].matched_terminals.is_empty() {
                continue;
            }
            suffixes_evaluated += 1;
            fill_suffix_candidates(
                &bytes[split_after + 1..],
                active,
                &mut suffix_candidates,
            );
            if suffix_candidates.is_empty() {
                continue;
            }
            for &terminal_1 in &nodes[node].matched_terminals {
                let blocked = disallowed_follows.get(&(terminal_1 as u32));
                for terminal_2 in suffix_candidates.iter() {
                    if blocked.is_some_and(|blocked| blocked.contains(terminal_2)) {
                        continue;
                    }
                    allowed_pairs += 1;
                    two_plus.set(terminal_1);
                    two_plus.set(terminal_2);
                    let witness = || TerminalBoundaryWitness {
                        token_id,
                        token_bytes: bytes.to_vec(),
                        split_position: split_after + 1,
                        terminal_1,
                        terminal_2,
                    };
                    if witnesses[terminal_1].is_none() {
                        witnesses[terminal_1] = Some(witness());
                    }
                    if witnesses[terminal_2].is_none() {
                        witnesses[terminal_2] = Some(witness());
                    }
                }
            }
        }
    }
    let evaluate_ms = evaluate_started_at.elapsed().as_secs_f64() * 1000.0;
    if super::types::compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][terminal_path_exact] tokens={} active={} scanner_states={} start_states={} nodes={} states_scanned={} reached_states={} finalizer_terminals_scanned={} suffixes_evaluated={} allowed_pairs={} two_plus={} trie_ms={:.3} populate_ms={:.3} evaluate_ms={:.3} total_ms={:.3}",
            token_entries.len(),
            active.count_ones(),
            scanner.num_states(),
            active_start_states.len(),
            nodes.len(),
            states_scanned,
            reached_states,
            finalizer_terminals_scanned,
            suffixes_evaluated,
            allowed_pairs,
            two_plus.count_ones(),
            trie_ms,
            populate_ms,
            evaluate_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    ExactTerminalPathTwoPlus {
        two_plus,
        witnesses,
    }
}

fn suffix_terminal_candidates_nfa(
    tokenizer: &Tokenizer,
    suffix: &[u8],
    active: &BitSet,
    candidates: &mut BitSet,
) {
    candidates.clear_all();
    let mut states = tokenizer
        .execute_from_state_end_only(&[], tokenizer.initial_state_id());
    let mut consumed_suffix = true;
    for &byte in suffix {
        if states.iter().all(|&state| {
            tokenizer
                .possible_future_terminals(state)
                .is_disjoint(active)
        }) {
            consumed_suffix = false;
            break;
        }
        states = tokenizer.step_all(&states, byte);
        if states.is_empty() {
            consumed_suffix = false;
            break;
        }
        for &state in &states {
            for terminal in tokenizer.matched_terminals_iter(state) {
                let terminal = terminal as usize;
                if active.contains(terminal) {
                    candidates.set(terminal);
                }
            }
        }
    }
    if consumed_suffix {
        for &state in &states {
            for terminal in tokenizer.possible_future_terminals_iter(state) {
                let terminal = terminal as usize;
                if active.contains(terminal) {
                    candidates.set(terminal);
                }
            }
        }
    }
}

fn exact_terminal_path_two_plus_fused_nfa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    active: &BitSet,
    raw_start_states: &[usize],
) -> ExactTerminalPathTwoPlus {
    let started_at = std::time::Instant::now();
    let words_per_mask = active.len().div_ceil(64);
    let (mut prefix_scanner, prefix_start) =
        ActivePrefixPowerset::new(tokenizer, active, raw_start_states);

    let mut allowed_after = vec![0u64; active.len() * words_per_mask];
    for terminal_1 in active.iter() {
        let base = terminal_1 * words_per_mask;
        for word_index in 0..words_per_mask {
            let mut allowed = active.words().get(word_index).copied().unwrap_or(0);
            if let Some(blocked) = disallowed_follows.get(&(terminal_1 as u32)) {
                allowed &= !blocked.words().get(word_index).copied().unwrap_or(0);
            }
            allowed_after[base + word_index] = allowed;
        }
    }

    let mut two_plus_words = vec![0u64; words_per_mask];
    let mut followers = vec![0u64; words_per_mask];
    let mut suffix_candidates = BitSet::new(active.len());
    let mut witnesses = vec![None; active.len()];
    let mut split_checks = 0usize;
    let mut allowed_pairs = 0usize;

    for (&token_id, bytes) in vocab.entries_map().iter() {
        if bytes.len() < 2 {
            continue;
        }
        let mut prefix_state = prefix_start;
        for split_position in 1..bytes.len() {
            prefix_state = prefix_scanner.step(prefix_state, bytes[split_position - 1]);
            if prefix_state == PREFIX_DEAD_STATE {
                break;
            }
            let matched = prefix_scanner.matched_mask(prefix_state);
            if matched.iter().all(|&word| word == 0) {
                continue;
            }
            split_checks += 1;
            suffix_terminal_candidates_nfa(
                tokenizer,
                &bytes[split_position..],
                active,
                &mut suffix_candidates,
            );
            if suffix_candidates.is_empty() {
                continue;
            }
            allowed_pairs += accumulate_terminal_path_boundaries(
                matched,
                suffix_candidates.words(),
                &allowed_after,
                words_per_mask,
                active.len(),
                &mut two_plus_words,
                &mut followers,
                Some((&mut witnesses, token_id, bytes, split_position)),
            );
        }
    }

    let mut two_plus = BitSet::new(active.len());
    for (word_index, &word) in two_plus_words.iter().enumerate() {
        let mut pending = word;
        while pending != 0 {
            let terminal = word_index * 64 + pending.trailing_zeros() as usize;
            pending &= pending - 1;
            if terminal < active.len() {
                two_plus.set(terminal);
            }
        }
    }
    if super::types::compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][terminal_path_fused_nfa] tokens={} active={} raw_start_states={} configs={} split_checks={} allowed_pairs={} two_plus={} total_ms={:.3}",
            vocab.len(),
            active.count_ones(),
            raw_start_states.len(),
            prefix_scanner.configs.len(),
            split_checks,
            allowed_pairs,
            two_plus.count_ones(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    ExactTerminalPathTwoPlus { two_plus, witnesses }
}

fn exact_terminal_path_two_plus_bounded_view_from_starts(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    active: &BitSet,
    raw_start_states: &[usize],
) -> ExactTerminalPathTwoPlus {
    let token_entries = vocab
        .entries_map()
        .iter()
        .map(|(&token_id, bytes)| (token_id, bytes.as_slice()))
        .collect::<Vec<_>>();
    let tokens = token_entries
        .iter()
        .map(|(_, bytes)| *bytes)
        .collect::<Vec<_>>();
    let active_groups = (0..active.len())
        .map(|terminal| active.contains(terminal))
        .collect::<Vec<_>>();
    let view_started_at = std::time::Instant::now();
    let bounded = build_token_bounded_analysis_view_from_combined_starts(
        tokenizer,
        raw_start_states,
        &tokens,
        Some(&active_groups),
    );
    let view_build_ms = view_started_at.elapsed().as_secs_f64() * 1000.0;
    let scanner = FlatExactBoundaryScanner {
        dfa: bounded.tokenizer_view.dfa(),
    };
    let mut view_start_states = raw_start_states
        .iter()
        .map(|&raw_state| bounded.view_state_for_raw_start(raw_state) as u32)
        .collect::<Vec<_>>();
    view_start_states.sort_unstable();
    view_start_states.dedup();
    let result = exact_terminal_path_two_plus_with_suffix(
        &scanner,
        &[],
        &[],
        &[],
        &token_entries,
        active,
        disallowed_follows,
        &view_start_states,
        |suffix, active, candidates| {
            suffix_terminal_candidates_nfa(tokenizer, suffix, active, candidates);
        },
    );
    if super::types::compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][terminal_path_exact_view] raw_states={} raw_start_states={} view_states={} view_start_states={} view_build_ms={:.3} total_ms={:.3}",
            tokenizer.num_states(),
            raw_start_states.len(),
            scanner.num_states(),
            view_start_states.len(),
            view_build_ms,
            view_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    result
}

fn exact_terminal_path_two_plus(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    bytesets: &SharedClassifyBytesets,
    active: &BitSet,
) -> ExactTerminalPathTwoPlus {
    let empty = || ExactTerminalPathTwoPlus {
        two_plus: BitSet::new(active.len()),
        witnesses: vec![None; active.len()],
    };
    if active.is_empty() || vocab.is_empty() {
        return empty();
    }
    let active_words = active.words();
    let raw_start_states = (0..tokenizer.num_states())
        .filter(|&state| state_future_intersects_words(bytesets, state, active_words))
        .map(|state| state as usize)
        .collect::<Vec<_>>();
    if raw_start_states.is_empty() {
        return empty();
    }
    if active.count_ones() <= 256 && vocab.len() <= 4_096 {
        let fused = exact_terminal_path_two_plus_fused_nfa(
            tokenizer,
            vocab,
            disallowed_follows,
            active,
            &raw_start_states,
        );
        if std::env::var_os("GLRMASK_TERMINAL_PATH_FUSED_NFA_STRICT_REFERENCE").is_some() {
            let reference = exact_terminal_path_two_plus_bounded_view_from_starts(
                tokenizer,
                vocab,
                disallowed_follows,
                active,
                &raw_start_states,
            );
            assert_eq!(
                fused.two_plus, reference.two_plus,
                "fused NFA terminal-path classification differed from bounded-view reference"
            );
        }
        return fused;
    }
    exact_terminal_path_two_plus_bounded_view_from_starts(
        tokenizer,
        vocab,
        disallowed_follows,
        active,
        &raw_start_states,
    )
}

fn assert_exact_boundary_batch_matches_scalar_reference(
    tokenizer: &Tokenizer,
    tokens: &[&[u8]],
    active_bitset: &BitSet,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    active_start_states: &[u32],
    result: &[bool],
) {
    if std::env::var_os("GLRMASK_EXACT_L2P_BOUNDARY_STRICT_REFERENCE").is_none() {
        return;
    }
    let reference = tokens
        .iter()
        .map(|bytes| {
            token_has_exact_active_l2p_boundary(
                tokenizer,
                bytes,
                active_bitset,
                disallowed_follows,
                active_start_states,
            )
        })
        .collect::<Vec<_>>();
    if let Some(index) = result
        .iter()
        .zip(&reference)
        .position(|(fast, reference)| fast != reference)
    {
        panic!(
            "exact L2P boundary batch differed from scalar reference at token index {index}: bytes={:?} fast={} reference={}",
            tokens[index], result[index], reference[index]
        );
    }
}

fn tokens_have_exact_active_l2p_boundary(
    tokenizer: &Tokenizer,
    bytesets: &SharedClassifyBytesets,
    flat_trans: &[u32],
    transitions_by_byte: &[u32],
    tokens: &[&[u8]],
    active_bitset: &BitSet,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    active_start_states: &[u32],
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> Vec<bool> {
    if !tokenizer.has_epsilon_transitions() {
        let scanner = RawExactBoundaryScanner { tokenizer, flat_trans };
        let result = tokens_have_exact_active_l2p_boundary_deterministic(
            &scanner,
            transitions_by_byte,
            &bytesets.sparse_transitions_by_byte,
            &bytesets.reverse_transitions_by_byte,
            tokens,
            active_bitset,
            disallowed_follows,
            active_start_states,
        );
        assert_exact_boundary_batch_matches_scalar_reference(
            tokenizer,
            tokens,
            active_bitset,
            disallowed_follows,
            active_start_states,
            &result,
        );
        return result;
    }

    let view_started_at = std::time::Instant::now();
    let raw_start_states = active_start_states
        .iter()
        .map(|&state| state as usize)
        .collect::<Vec<_>>();
    let active_groups = (0..active_bitset.len())
        .map(|terminal| active_bitset.contains(terminal))
        .collect::<Vec<_>>();
    let bounded = build_bounded_analysis_view_from_combined_starts_with_trie(
        tokenizer,
        &raw_start_states,
        tokens,
        Some(&active_groups),
        prebuilt_token_trie.filter(|trie| trie.is_reasonable_superset_for(tokens.len())),
    );
    let view_build_ms = view_started_at.elapsed().as_secs_f64() * 1000.0;
    let dfa = bounded.tokenizer_view.dfa();
    let scanner = FlatExactBoundaryScanner { dfa };
    let mut view_start_states = raw_start_states
        .iter()
        .map(|&raw_state| bounded.view_state_for_raw_start(raw_state) as u32)
        .collect::<Vec<_>>();
    view_start_states.sort_unstable();
    view_start_states.dedup();

    let index_ms = 0.0;
    let batch_started_at = std::time::Instant::now();
    let result = tokens_have_exact_active_l2p_boundary_deterministic(
        &scanner,
        &[],
        &[],
        &[],
        tokens,
        active_bitset,
        disallowed_follows,
        &view_start_states,
    );
    let batch_ms = batch_started_at.elapsed().as_secs_f64() * 1000.0;

    assert_exact_boundary_batch_matches_scalar_reference(
        tokenizer,
        tokens,
        active_bitset,
        disallowed_follows,
        active_start_states,
        &result,
    );

    if super::types::compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][exact_boundary_nfa_batch] raw_states={} raw_start_states={} view_states={} view_start_states={} view_build_ms={:.3} transition_index_ms={:.3} batch_ms={:.3} total_ms={:.3}",
            tokenizer.num_states(),
            active_start_states.len(),
            dfa.states.len(),
            view_start_states.len(),
            view_build_ms,
            index_ms,
            batch_ms,
            view_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    result
}

pub fn split_vocab_for_active_l2p_terminals(
    tokenizer: &Tokenizer,
    flat_trans: &[u32],
    vocab: &Vocab,
    disallowed_follows: &Arc<BTreeMap<u32, BitSet>>,
    num_terminals: u32,
    active_terminals: &[bool],
    shared_classify_cache: Option<&SharedClassifyCache>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> L2pVocabBoundarySplit {
    let prebuilt_token_trie = std::env::var("GLRMASK_USE_PREBUILT_L2P_ROUTE_TRIE")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true)
        .then_some(prebuilt_token_trie)
        .flatten();
    let split_started_at = std::time::Instant::now();
    let owned_bytesets: Option<SharedClassifyBytesets>;
    let bytesets = if let Some(cache) = shared_classify_cache {
        cache.get_or_init(|| SharedClassifyBytesets::build(tokenizer, num_terminals))
    } else {
        owned_bytesets = Some(SharedClassifyBytesets::build(tokenizer, num_terminals));
        owned_bytesets.as_ref().unwrap()
    };

    let active: Vec<usize> = active_terminals
        .iter()
        .enumerate()
        .filter_map(|(terminal, active)| active.then_some(terminal))
        .collect();
    let mut active_bitset = BitSet::new(num_terminals as usize);
    for &terminal in &active {
        active_bitset.set(terminal);
    }
    let active_setup_started_at = std::time::Instant::now();
    let route_setup =
        active_l2p_route_setup(tokenizer, bytesets, &active_bitset, disallowed_follows);
    let active_setup_ms = active_setup_started_at.elapsed().as_secs_f64() * 1000.0;
    let boundary_pairs_ms = 0.0;

    let vocab_scan_started_at = std::time::Instant::now();
    let mut boundary_token_ids = Vec::<u32>::new();
    let mut single_token_ids = Vec::<u32>::with_capacity(vocab.entries_map().len());
    let mut irrelevant_tokens = 0usize;
    let mut adjacent_entries = Vec::new();
    for (&token_id, bytes) in vocab.entries_map().iter() {
        match token_l2p_route_hint(
            bytes,
            &route_setup.allowed_boundary_pair_words,
            &route_setup.active_reachable_by_byte,
        ) {
            TokenL2pRouteHint::Adjacent => adjacent_entries.push((token_id, bytes.as_slice())),
            TokenL2pRouteHint::Single => single_token_ids.push(token_id),
            TokenL2pRouteHint::Irrelevant => irrelevant_tokens += 1,
        }
    }
    let vocab_scan_ms = vocab_scan_started_at.elapsed().as_secs_f64() * 1000.0;
    let adjacent_candidate_count = adjacent_entries.len();
    let estimated_exact_work = route_setup
        .active_start_states
        .len()
        .saturating_mul(adjacent_candidate_count);
    // The exact filter follows both prefix and suffix state sets for a
    // deterministic-dispatch tokenizer, so the same validation-gated path is
    // sound for ordinary and partitioned lexer representations.
    let use_exact_boundary_filter = match exact_l2p_boundary_filter_mode() {
            ExactL2pBoundaryFilterMode::Force(enabled) => enabled,
            ExactL2pBoundaryFilterMode::Auto => {
                adjacent_candidate_count >= exact_l2p_boundary_filter_min_candidates()
                    && estimated_exact_work <= exact_l2p_boundary_filter_work_limit()
            }
        };

    if std::env::var_os("GLRMASK_PROFILE_EXACT_L2P_BOUNDARY_FILTER").is_some() {
        eprintln!(
            "[glrmask/profile][exact_l2p_boundary_filter] vocab_tokens={} active_terminals={} active_start_states={} adjacent_candidates={} estimated_work={} enabled={}",
            vocab.entries_map().len(),
            active.len(),
            route_setup.active_start_states.len(),
            adjacent_candidate_count,
            estimated_exact_work,
            use_exact_boundary_filter,
        );
    }

    let exact_started_at = std::time::Instant::now();
    let mut exact_prefilter_ms = 0.0;
    let mut exact_batch_ms = 0.0;
    let mut exact_viable_tokens = 0usize;
    let exact_boundary_matches = use_exact_boundary_filter.then(|| {
        let prefilter_started_at = std::time::Instant::now();
        let active_words = active_bitset.words();
        let active_matched_by_state = (adjacent_entries.len()
            > (tokenizer.num_states() as usize).div_ceil(16))
        .then(|| build_active_matched_by_state(bytesets, &active_bitset));
        let viable_indices = if tokenizer.has_epsilon_transitions() {
            // The scalar suffix prefilter cannot represent a state-set scanner
            // configuration. Let the exact NFA checker classify every
            // byte-adjacent token.
            (0..adjacent_entries.len()).collect::<Vec<_>>()
        } else {
            adjacent_entries
                .iter()
                .enumerate()
                .filter_map(|(index, (_, bytes))| {
                    token_has_active_terminal_suffix(
                        tokenizer,
                        bytesets,
                        flat_trans,
                        bytes,
                        active_words,
                        active_matched_by_state.as_deref(),
                        &route_setup.active_suffix_start_by_byte,
                    )
                    .then_some(index)
                })
                .collect::<Vec<_>>()
        };
        exact_prefilter_ms = prefilter_started_at.elapsed().as_secs_f64() * 1000.0;
        exact_viable_tokens = viable_indices.len();
        let tokens = viable_indices
            .iter()
            .map(|&index| adjacent_entries[index].1)
            .collect::<Vec<_>>();
        let batch_started_at = std::time::Instant::now();
        let viable_matches = tokens_have_exact_active_l2p_boundary(
            tokenizer,
            bytesets,
            flat_trans,
            &bytesets.transitions_by_byte,
            &tokens,
            &active_bitset,
            disallowed_follows.as_ref(),
            &route_setup.active_start_states,
            prebuilt_token_trie,
        );
        exact_batch_ms = batch_started_at.elapsed().as_secs_f64() * 1000.0;
        let mut matches = vec![false; adjacent_entries.len()];
        for (index, exact_match) in viable_indices.into_iter().zip(viable_matches) {
            matches[index] = exact_match;
        }
        matches
    });
    let exact_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
    let finalize_started_at = std::time::Instant::now();
    let mut adjacent_single_token_ids = Vec::<u32>::new();
    for (adjacent_index, &(token_id, bytes)) in adjacent_entries.iter().enumerate() {
        let exact_match = exact_boundary_matches
            .as_ref()
            .map_or(true, |matches| matches[adjacent_index]);
        if exact_match {
            boundary_token_ids.push(token_id);
        } else if bytes
            .iter()
            .any(|&byte| route_setup.active_reachable_by_byte[byte as usize] != 0)
        {
            adjacent_single_token_ids.push(token_id);
        } else {
            irrelevant_tokens += 1;
        }
    }

    single_token_ids = merge_sorted_token_ids(single_token_ids, adjacent_single_token_ids);
    let adjacent_tokens = adjacent_entries.len();
    let boundary_tokens = boundary_token_ids.len();
    let single_tokens = single_token_ids.len();
    let finalize_ms = finalize_started_at.elapsed().as_secs_f64() * 1000.0;
    if super::types::compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l2p_vocab_route] tokens={} active={} starts={} adjacent={} exact_viable={} setup_ms={:.3} pairs_ms={:.3} scan_ms={:.3} exact_prefilter_ms={:.3} exact_batch_ms={:.3} exact_ms={:.3} finalize_ms={:.3} total_ms={:.3}",
            vocab.entries_map().len(),
            active.len(),
            route_setup.active_start_states.len(),
            adjacent_entries.len(),
            exact_viable_tokens,
            active_setup_ms,
            boundary_pairs_ms,
            vocab_scan_ms,
            exact_prefilter_ms,
            exact_batch_ms,
            exact_ms,
            finalize_ms,
            split_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    L2pVocabBoundarySplit {
        boundary_token_ids,
        single_token_ids,
        adjacent_tokens,
        boundary_tokens,
        single_tokens,
        irrelevant_tokens,
    }
}

fn compute_partition_cost(
    cost_fn: L2pPartitionCostFn,
    l2p_terminals: usize,
    partition_size: usize,
) -> f64 {
    if l2p_terminals == 0 || partition_size == 0 {
        return 0.0;
    }

    let num_l2p = l2p_terminals as f64;
    let size = partition_size as f64;
    match cost_fn {
        L2pPartitionCostFn::Size => num_l2p * size,
        L2pPartitionCostFn::SizeLog => num_l2p * size.ln(),
        L2pPartitionCostFn::LogLog => num_l2p.ln() * size.ln(),
        L2pPartitionCostFn::UnionSize => num_l2p * size,
    }
}

fn partition_metric_count(
    cost_fn: L2pPartitionCostFn,
    intersection_count: usize,
    union_count: usize,
) -> usize {
    match cost_fn {
        L2pPartitionCostFn::UnionSize => union_count,
        L2pPartitionCostFn::Size
        | L2pPartitionCostFn::SizeLog
        | L2pPartitionCostFn::LogLog => intersection_count,
    }
}

fn objective_score(objective: L2pPartitionObjective, costs: &[f64]) -> f64 {
    match objective {
        L2pPartitionObjective::Max => costs.iter().copied().fold(0.0, f64::max),
        L2pPartitionObjective::Sum => costs.iter().sum(),
    }
}

fn compute_token_l2p_map(
    vocab: &Vocab,
    bytesets: &SharedClassifyBytesets,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
) -> BTreeMap<u32, BitSet> {
    let num_terminals = num_terminals as usize;
    let (byte_to_last, byte_to_first) =
        build_byte_terminal_reverse_index(bytesets, num_terminals);

    let mut token_l2p_map = BTreeMap::<u32, BitSet>::new();
    for (&token_id, bytes) in vocab.entries_map().iter() {
        token_l2p_map.insert(
            token_id,
            token_l2p_terminals(
                bytes,
                &byte_to_last,
                &byte_to_first,
                disallowed_follows,
                num_terminals,
            ),
        );
    }
    token_l2p_map
}

pub fn partition_vocab_char_type_tokens(
    vocab: &Vocab,
    automatic_bounded_synthesis_overflow: bool,
    automatic_p2_overflow_threshold: Option<usize>,
) -> Vec<Vec<u32>> {
    let threshold = |name: &str, automatic: usize| match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|&threshold| threshold > 0),
        Err(_) => automatic_bounded_synthesis_overflow.then_some(automatic),
    };
    let p0_overflow_threshold = threshold("GLRMASK_P0_LONG_TOKEN_OVERFLOW_THRESHOLD", 16);
    let p1_overflow_threshold = threshold("GLRMASK_P1_LONG_TOKEN_OVERFLOW_THRESHOLD", 20);
    let p2_overflow_threshold = match std::env::var("GLRMASK_P2_LONG_TOKEN_OVERFLOW_THRESHOLD") {
        Ok(value) => value.parse::<usize>().ok().filter(|&threshold| threshold > 0),
        Err(_) => automatic_p2_overflow_threshold,
    };
    let p4_overflow_threshold = threshold("GLRMASK_P4_LONG_TOKEN_OVERFLOW_THRESHOLD", 32);

    let spec = super::regular_partition::char_type_regular_partition(
        super::regular_partition::CharTypeRegularPartitionKey {
            p0_overflow_threshold,
            p1_overflow_threshold,
            p2_overflow_threshold,
            p4_overflow_threshold,
            structural_boundary_enabled: structural_boundary_lexical_partition_enabled(),
        },
    );
    let partition_count = spec.cells().len();

    let mut partitions: Vec<Vec<u32>> =
        (0..partition_count).map(|_| Vec::new()).collect();
    for (&token_id, bytes) in vocab.entries_map().iter() {
        let idx = spec.partition_index(bytes);
        partitions[idx].push(token_id);
    }
    partitions
}

pub fn estimate_l2p_objective_for_token_partitions(
    token_partitions: &[Vec<u32>],
    token_l2p_map: &BTreeMap<u32, BitSet>,
    cost_fn: L2pPartitionCostFn,
    objective: L2pPartitionObjective,
) -> (Vec<f64>, Vec<usize>, f64) {
    let mut costs = Vec::with_capacity(token_partitions.len());
    let mut l2p_counts = Vec::with_capacity(token_partitions.len());

    for token_ids in token_partitions {
        if token_ids.is_empty() {
            costs.push(0.0);
            l2p_counts.push(0);
            continue;
        }

        let mut intersection: Option<BitSet> = None;
        let mut union: Option<BitSet> = None;
        for &token_id in token_ids {
            if let Some(token_l2p) = token_l2p_map.get(&token_id) {
                if let Some(current) = &mut intersection {
                    current.intersect_with(token_l2p);
                } else {
                    intersection = Some(token_l2p.clone());
                }
                if let Some(current) = &mut union {
                    current.union_with(token_l2p);
                } else {
                    union = Some(token_l2p.clone());
                }
            }
        }

        let l2p_count = intersection.as_ref().map_or(0, BitSet::count_ones);
        let union_count = union.as_ref().map_or(0, BitSet::count_ones);
        l2p_counts.push(l2p_count);
        costs.push(compute_partition_cost(
            cost_fn,
            partition_metric_count(cost_fn, l2p_count, union_count),
            token_ids.len(),
        ));
    }

    let score = objective_score(objective, &costs);
    (costs, l2p_counts, score)
}

fn partition_token_l2p_map_by_cost(
    token_l2p_map: &BTreeMap<u32, BitSet>,
    num_partitions: usize,
    cost_fn: L2pPartitionCostFn,
    objective: L2pPartitionObjective,
) -> L2pCostPartitioning {
    let mut grouped_index = BTreeMap::<Vec<u64>, usize>::new();
    let mut groups: Vec<L2pTokenGroup> = Vec::new();

    for (&token_id, l2p_terminals) in token_l2p_map {
        let key = l2p_terminals.words().to_vec();
        if let Some(&group_idx) = grouped_index.get(&key) {
            groups[group_idx].token_ids.push(token_id);
        } else {
            let group_idx = groups.len();
            grouped_index.insert(key, group_idx);
            groups.push(L2pTokenGroup {
                l2p_terminals: l2p_terminals.clone(),
                token_ids: vec![token_id],
            });
        }
    }

    groups.sort_by(|left, right| {
        let left_weight = left.l2p_terminals.count_ones() * left.token_ids.len();
        let right_weight = right.l2p_terminals.count_ones() * right.token_ids.len();
        right_weight
            .cmp(&left_weight)
            .then_with(|| right.l2p_terminals.count_ones().cmp(&left.l2p_terminals.count_ones()))
            .then_with(|| right.token_ids.len().cmp(&left.token_ids.len()))
    });

    let mut buckets = vec![L2pPartitionBucket::new(); num_partitions.max(1)];
    let mut current_costs = vec![0.0; buckets.len()];

    for group in groups {
        let mut best_idx = 0usize;
        let mut best_score = f64::INFINITY;
        let mut best_cost = f64::INFINITY;
        let mut best_l2p_count = usize::MAX;
        let mut best_size = usize::MAX;

        for (idx, bucket) in buckets.iter().enumerate() {
            let candidate_intersection = if let Some(current) = &bucket.l2p_intersection {
                current.intersection(&group.l2p_terminals)
            } else {
                group.l2p_terminals.clone()
            };
            let candidate_union = if let Some(current) = &bucket.l2p_union {
                current.union(&group.l2p_terminals)
            } else {
                group.l2p_terminals.clone()
            };
            let candidate_l2p_count = candidate_intersection.count_ones();
            let candidate_union_count = candidate_union.count_ones();
            let candidate_size = bucket.size() + group.token_ids.len();
            let candidate_cost = compute_partition_cost(
                cost_fn,
                partition_metric_count(cost_fn, candidate_l2p_count, candidate_union_count),
                candidate_size,
            );

            let mut candidate_costs = current_costs.clone();
            candidate_costs[idx] = candidate_cost;
            let score = objective_score(objective, &candidate_costs);

            let better = score < best_score
                || (score == best_score
                    && (candidate_cost < best_cost
                        || (candidate_cost == best_cost
                            && (candidate_l2p_count < best_l2p_count
                                || (candidate_l2p_count == best_l2p_count
                                    && candidate_size < best_size)))));
            if better {
                best_idx = idx;
                best_score = score;
                best_cost = candidate_cost;
                best_l2p_count = candidate_l2p_count;
                best_size = candidate_size;
            }
        }

        let bucket = &mut buckets[best_idx];
        if let Some(current) = &mut bucket.l2p_intersection {
            current.intersect_with(&group.l2p_terminals);
        } else {
            bucket.l2p_intersection = Some(group.l2p_terminals.clone());
        }
        if let Some(current) = &mut bucket.l2p_union {
            current.union_with(&group.l2p_terminals);
        } else {
            bucket.l2p_union = Some(group.l2p_terminals.clone());
        }
        bucket.token_ids.extend(group.token_ids);
        current_costs[best_idx] = best_cost;
    }

    buckets.sort_by(|left, right| right.size().cmp(&left.size()));

    let estimated_partition_costs = buckets
        .iter()
        .map(|bucket| {
            let union_count = bucket.l2p_union.as_ref().map_or(0, BitSet::count_ones);
            compute_partition_cost(
                cost_fn,
                partition_metric_count(cost_fn, bucket.l2p_count(), union_count),
                bucket.size(),
            )
        })
        .collect::<Vec<_>>();
    let estimated_l2p_terminals = buckets
        .iter()
        .map(L2pPartitionBucket::l2p_count)
        .collect::<Vec<_>>();
    let partitions = buckets
        .into_iter()
        .map(|bucket| bucket.token_ids)
        .collect::<Vec<_>>();

    L2pCostPartitioning {
        objective_score: objective_score(objective, &estimated_partition_costs),
        partitions,
        estimated_partition_costs,
        estimated_l2p_terminals,
    }
}

pub fn partition_vocab_by_l2p_cost_with_token_map(
    vocab: &Vocab,
    bytesets: &SharedClassifyBytesets,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
    num_partitions: usize,
    cost_fn: L2pPartitionCostFn,
    objective: L2pPartitionObjective,
) -> (L2pCostPartitioning, BTreeMap<u32, BitSet>) {
    let token_l2p_map = compute_token_l2p_map(vocab, bytesets, disallowed_follows, num_terminals);
    let partitioning =
        partition_token_l2p_map_by_cost(&token_l2p_map, num_partitions, cost_fn, objective);
    (partitioning, token_l2p_map)
}

pub fn partition_vocab_by_l2p_cost(
    vocab: &Vocab,
    bytesets: &SharedClassifyBytesets,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
    num_partitions: usize,
    cost_fn: L2pPartitionCostFn,
    objective: L2pPartitionObjective,
) -> L2pCostPartitioning {
    partition_vocab_by_l2p_cost_with_token_map(
        vocab,
        bytesets,
        disallowed_follows,
        num_terminals,
        num_partitions,
        cost_fn,
        objective,
    )
    .0
}


#[cfg(test)]
mod tests ;

#[cfg(test)]
#[test]
fn shared_route_cache_retains_follow_identity_and_separates_policies() {
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex_monolithic;
    let tokenizer = build_regex_monolithic(&[
        Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"b".to_vec()),
    ]).into_tokenizer(2, None);
    let bytesets = SharedClassifyBytesets::build(&tokenizer, 2);
    let mut active = BitSet::new(2); active.set(0); active.set(1);
    let follows = Arc::new(BTreeMap::new());
    let weak = Arc::downgrade(&follows);
    let first = active_l2p_route_setup(&tokenizer, &bytesets, &active, &follows);
    assert!(first.allowed_boundary_pairs[b'a' as usize].contains(b'b'));
    let repeated = active_l2p_route_setup(&tokenizer, &bytesets, &active, &follows);
    assert!(Arc::ptr_eq(&first, &repeated));
    drop(first); drop(repeated); drop(follows);
    assert!(weak.upgrade().is_some(), "cache identity must remain owned");
    let mut blocked = BitSet::new(2); blocked.set(1);
    let follows = Arc::new(BTreeMap::from([(0, blocked)]));
    let other = active_l2p_route_setup(&tokenizer, &bytesets, &active, &follows);
    assert!(!other.allowed_boundary_pairs[b'a' as usize].contains(b'b'));
    drop(other); drop(bytesets);
    assert!(weak.upgrade().is_none(), "dropping cache releases old identity");
}
