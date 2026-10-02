use crate::automata::lexer::Lexer;
use std::collections::BTreeMap;
use std::hash::Hasher;
use std::sync::Mutex;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::constraint_possible_matches::collector::{
    IntervalPossibleMatchMap, TerminalRangeGroup, TrieClassBuildResult,
};
use crate::compiler::pm_profile::elapsed_ms;
use crate::compiler::possible_matches::PossibleMatchesComputer;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap, MappedArtifact};
use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::compat::{
    FlatDfa, FlatDfaState, TokenizerView,
};
use crate::compiler::stages::id_map_and_terminal_dwa::classify::{
    classify_vocab_char_type, VocabPartitionDfa,
};
use crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalDwaFamilies;
use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::vocab::fast as vocab_equivalence_analysis;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::ds::weight::{shared_rangeset, Weight};
use crate::grammar::flat::TerminalID;
use crate::runtime::{dynamic_mask_vocab_layout_class, DynamicMaskTrie, DynamicMaskVocab};
use crate::runtime::{
    dynamic_mask_llg_master_layout_class,
    dynamic_mask_llg_master_safe_chars, DYNAMIC_MASK_LLG_MASTER_CACHE_ID,
};
use crate::vocab::VocabDerivedArtifact;
use crate::Vocab;

pub(crate) mod collector;

pub(crate) type RuntimePossibleMatchesByTerminal = BTreeMap<TerminalID, Weight>;
pub(crate) type SignatureClassId = u32;
type StateTerminalLabel = (u32, TerminalID);

#[derive(Debug, Clone)]
pub(crate) struct PossibleMatchVocabMap {
    pub(crate) original_to_internal: Vec<u32>,
    pub(crate) internal_to_originals: Vec<Vec<u32>>,
}

// WARNING: terminal-DWA equivalence maps must never be reused for possible
// matches. Terminal-DWA equivalence does not imply possible-matches
// equivalence. This warning must never be removed under any circumstances.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConstraintPossibleMatchesConfig {
    defer_to_dynamic_mask: bool,
}

impl ConstraintPossibleMatchesConfig {
    /// Materialize the full possible-match table during compilation.
    pub(crate) const EAGER: Self = Self {
        defer_to_dynamic_mask: false,
    };
    /// Keep compile-time PM empty. Runtime static masking remains exact until
    /// token-start terminal exclusions appear, then falls back to the exact
    /// dynamic masker for that mask call.
    pub(crate) const DEFER_TO_DYNAMIC_MASK: Self = Self {
        defer_to_dynamic_mask: true,
    };

    #[inline]
    pub(crate) const fn is_complete(self) -> bool {
        !self.defer_to_dynamic_mask
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ConstraintPossibleMatchesProfile {
    pub(crate) vocab_equiv_ms: f64,
    pub(crate) possible_matches_collect_ms: f64,
    pub(crate) possible_match_vocab_ms: f64,
}

#[derive(Debug)]
pub(crate) struct ConstraintPossibleMatchesComputation {
    pub(crate) mapped_possible_matches: MappedArtifact<RuntimePossibleMatchesByTerminal>,
    pub(crate) runtime_dynamic_vocab: RuntimeDynamicMaskVocabArtifacts,
    pub(crate) complete: bool,
    pub(crate) profile: ConstraintPossibleMatchesProfile,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeDynamicMaskVocabArtifacts {
    pub(crate) vocab: DynamicMaskVocab,
}

/// Complete the possible-match artifact from the sole global-L1 transition.
///
/// This is intentionally much narrower than reusing terminal-DWA equivalence
/// for possible matches in general, which is unsound. The caller proves that
/// the parser language consumes one terminal at most once and that there is no
/// ignore terminal. In that shape, the one-terminal lexer keeps acceptance
/// monotone while the terminal remains live, so a token matches the delayed
/// terminal at some prefix exactly when the global L1 scan ends in that
/// terminal's accepting signature. The L1 transition weight is therefore the
/// exact possible-match relation, already expressed in its compact TSID/token
/// coordinates.
pub(crate) fn complete_single_use_terminal_possible_matches_from_l1(
    families: &TerminalDwaFamilies,
    mut deferred: ConstraintPossibleMatchesComputation,
) -> Option<ConstraintPossibleMatchesComputation> {
    if deferred.complete || families.l2p.is_some() || families.special.is_some() {
        return None;
    }
    let l1 = families.l1.as_ref()?;
    let TerminalAutomaton::Dwa(dwa) = l1.artifact() else {
        return None;
    };
    let start = dwa.states().get(dwa.start_state() as usize)?;
    if start.transitions.len() != 1 {
        return None;
    }
    let (&label, (_, weight)) = start.transitions.first_key_value()?;
    let terminal = TerminalID::try_from(label).ok()?;

    let mut possible_matches = RuntimePossibleMatchesByTerminal::new();
    possible_matches.insert(terminal, weight.clone());
    deferred.mapped_possible_matches =
        MappedArtifact::new(possible_matches, l1.id_map().clone());
    deferred.complete = true;
    deferred.profile = ConstraintPossibleMatchesProfile::default();
    Some(deferred)
}

#[derive(Debug, Clone)]
struct OrderedVocab {
    original_slot_count: usize,
    ordered_to_originals: Arc<Vec<Vec<u32>>>,
    ordered_token_bytes: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct OrderedVocabTrieArtifacts {
    ordered_vocab: Arc<OrderedVocab>,
    trie: Arc<VocabPrefixTree>,
    /// Fully materialized vocabulary-only dynamic-mask template. Constraint
    /// builds clone its immutable indexes into fresh runtime-local caches.
    runtime_dynamic_vocab: Arc<OnceLock<Arc<DynamicMaskVocab>>>,
    /// Same vocabulary-only template with the llguidance slice residual tries
    /// materialized. Kept separate so non-slicer users never pay this one-time
    /// preparation cost, while every slicer-enabled constraint Arc-shares it.
    runtime_dynamic_vocab_llg: Arc<OnceLock<Arc<DynamicMaskVocab>>>,
    /// Vocabulary-global byte support of whole tokens admitted by the exact
    /// llguidance safe-string language. Producer-side proof preparation needs
    /// only this tiny set; caching it avoids materializing the full runtime
    /// dynamic vocabulary just to select proof candidates.
    llg_safe_slice_token_bytes: Arc<OnceLock<U8Set>>,
    /// Largest Unicode-scalar length of any whole token in the exact safe
    /// language. Bounded master proofs only need radii up to this value.
    llg_master_max_safe_chars: Arc<OnceLock<u16>>,
}

impl VocabDerivedArtifact for OrderedVocabTrieArtifacts {}

impl OrderedVocabTrieArtifacts {
    fn new(ordered_vocab: Arc<OrderedVocab>, trie: Arc<VocabPrefixTree>) -> Self {
        Self {
            ordered_vocab,
            trie,
            runtime_dynamic_vocab: Arc::new(OnceLock::new()),
            runtime_dynamic_vocab_llg: Arc::new(OnceLock::new()),
            llg_safe_slice_token_bytes: Arc::new(OnceLock::new()),
            llg_master_max_safe_chars: Arc::new(OnceLock::new()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OrderedVocabCacheFingerprint {
    token_count: usize,
    max_token_id: u32,
    total_bytes: usize,
    hash: u64,
}

#[derive(Debug, Clone)]
struct OrderedVocabCacheEntry {
    fingerprint: OrderedVocabCacheFingerprint,
    source_original_to_ordered: Arc<[u32]>,
    artifacts: OrderedVocabTrieArtifacts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrderedVocabCacheStatus {
    Disabled,
    Hit,
    Miss,
}

impl OrderedVocabCacheStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Hit => "hit",
            Self::Miss => "miss",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct OrderedVocabCacheProfile {
    status: OrderedVocabCacheStatus,
    probe_ns: u128,
    verify_ns: u128,
    ordered_vocab_build_ns: u128,
    trie_build_ns: u128,
    cache_entries: usize,
    capacity: usize,
}

#[derive(Debug, Clone, Copy)]
struct SweepEvent {
    add: bool,
    group_id: u32,
}

#[derive(Debug, Clone)]
struct SweepGroup {
    label_ids: Box<[u32]>,
}

#[derive(Debug, Default, Clone, Copy)]
struct SweepBuildStats {
    used_state_classes: usize,
    terminal_groups: usize,
    terminal_labels: usize,
    group_label_refs: usize,
    total_intervals: usize,
    total_events: usize,
}

pub(crate) fn build_internal_token_bytes_from_groups(
    vocab: &Vocab,
    internal_to_originals: &[Vec<u32>],
) -> BTreeMap<u32, Vec<u8>> {
    internal_to_originals.iter().enumerate().filter_map(|(internal_token_id, originals)| {
        let bytes = originals.iter().find_map(|original| vocab.entries_map().get(original))?.clone();
        Some((internal_token_id as u32, bytes))
    }).collect()
}

fn build_ordered_vocab(token_bytes: &BTreeMap<u32, Vec<u8>>) -> OrderedVocab {
    let original_slot_count = token_bytes.keys().next_back().map(|token_id| *token_id as usize + 1).unwrap_or(0);
    let mut entries: Vec<(u32, &[u8])> = token_bytes
        .iter()
        .map(|(&token_id, bytes)| (token_id, bytes.as_slice()))
        .collect();
    entries.sort_unstable_by(|left, right| left.1.cmp(right.1).then_with(|| left.0.cmp(&right.0)));

    let mut ordered_to_originals = Vec::new();
    let mut ordered_token_bytes = Vec::new();
    let mut index = 0usize;
    while index < entries.len() {
        let bytes = entries[index].1;
        let mut originals = Vec::new();
        while index < entries.len() && entries[index].1 == bytes {
            originals.push(entries[index].0);
            index += 1;
        }
        originals.sort_unstable();
        originals.dedup();
        ordered_token_bytes.push(bytes.to_vec());
        ordered_to_originals.push(originals);
    }

    OrderedVocab {
        original_slot_count,
        ordered_to_originals: Arc::new(ordered_to_originals),
        ordered_token_bytes,
    }
}

fn build_ordered_vocab_prefix_tree(ordered_vocab: &OrderedVocab) -> VocabPrefixTree {
    let entries: Vec<(usize, &[u8])> = ordered_vocab.ordered_token_bytes.iter().enumerate().map(|(ordered_id, bytes)| (ordered_id, bytes.as_slice())).collect();
    VocabPrefixTree::build_presorted(&entries)
}

fn ordered_vocab_cache_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_PM_ORDERED_VOCAB_CACHE")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true)
    })
}

fn ordered_vocab_cache_capacity() -> usize {
    static CAPACITY: OnceLock<usize> = OnceLock::new();
    *CAPACITY.get_or_init(|| {
        std::env::var("GLRMASK_PM_ORDERED_VOCAB_CACHE_CAPACITY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(4)
    })
}

fn ordered_vocab_cache() -> &'static Mutex<Vec<OrderedVocabCacheEntry>> {
    static CACHE: OnceLock<Mutex<Vec<OrderedVocabCacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn ordered_vocab_cache_fingerprint(
    token_bytes: &BTreeMap<u32, Vec<u8>>,
) -> OrderedVocabCacheFingerprint {
    let mut hasher = rustc_hash::FxHasher::default();
    let mut token_count = 0usize;
    let mut max_token_id = 0u32;
    let mut total_bytes = 0usize;
    for (&token_id, bytes) in token_bytes {
        hasher.write_u32(token_id);
        hasher.write_usize(bytes.len());
        hasher.write(bytes);
        token_count += 1;
        max_token_id = token_id;
        total_bytes += bytes.len();
    }
    OrderedVocabCacheFingerprint {
        token_count,
        max_token_id,
        total_bytes,
        hash: hasher.finish(),
    }
}

fn ordered_vocab_cache_source_matches(
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    source_original_to_ordered: &[u32],
    ordered_vocab: &OrderedVocab,
) -> bool {
    if ordered_vocab.ordered_token_bytes.len() != ordered_vocab.ordered_to_originals.len() {
        return false;
    }

    let cached_token_count: usize = ordered_vocab
        .ordered_to_originals
        .iter()
        .map(|originals| originals.len())
        .sum();
    if token_bytes.len() != cached_token_count {
        return false;
    }

    let actual_slot_count = token_bytes
        .keys()
        .next_back()
        .map(|token_id| *token_id as usize + 1)
        .unwrap_or(0);
    if actual_slot_count != ordered_vocab.original_slot_count {
        return false;
    }

    if source_original_to_ordered.len() != ordered_vocab.original_slot_count {
        return false;
    }

    for (&original_id, actual_bytes) in token_bytes {
        let Some(&ordered_id) = source_original_to_ordered.get(original_id as usize) else {
            return false;
        };
        let Some(cached_bytes) = ordered_vocab.ordered_token_bytes.get(ordered_id as usize) else {
            return false;
        };
        if actual_bytes != cached_bytes {
            return false;
        }
    }

    true
}

fn ordered_vocab_cache_source_original_to_ordered(
    ordered_vocab: &OrderedVocab,
) -> Arc<[u32]> {
    let mut original_to_ordered = vec![u32::MAX; ordered_vocab.original_slot_count];
    for (ordered_id, originals) in ordered_vocab.ordered_to_originals.iter().enumerate() {
        for &original_id in originals {
            let slot = &mut original_to_ordered[original_id as usize];
            debug_assert_eq!(*slot, u32::MAX);
            *slot = ordered_id as u32;
        }
    }
    original_to_ordered.into()
}

fn compile_profile_requested() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

fn emit_ordered_vocab_cache_profile(profile: OrderedVocabCacheProfile) {
    if !compile_profile_requested() {
        return;
    }
    eprintln!(
        "[glrmask/profile][ordered_vocab_cache] status={} probe_ms={:.3} verify_ms={:.3} ordered_vocab_ms={:.3} vocab_prefix_tree_ms={:.3} cache_entries={} capacity={}",
        profile.status.as_str(),
        profile.probe_ns as f64 / 1_000_000.0,
        profile.verify_ns as f64 / 1_000_000.0,
        profile.ordered_vocab_build_ns as f64 / 1_000_000.0,
        profile.trie_build_ns as f64 / 1_000_000.0,
        profile.cache_entries,
        profile.capacity,
    );
}

fn get_ordered_vocab_trie_artifacts(
    token_bytes: &BTreeMap<u32, Vec<u8>>,
) -> (OrderedVocabTrieArtifacts, OrderedVocabCacheProfile) {
    let capacity = ordered_vocab_cache_capacity();
    if !ordered_vocab_cache_enabled() || capacity == 0 {
        let ordered_vocab_started_at = Instant::now();
        let ordered_vocab = Arc::new(build_ordered_vocab(token_bytes));
        let ordered_vocab_build_ns = ordered_vocab_started_at.elapsed().as_nanos();
        let trie_started_at = Instant::now();
        let trie = Arc::new(build_ordered_vocab_prefix_tree(ordered_vocab.as_ref()));
        let trie_build_ns = trie_started_at.elapsed().as_nanos();
        return (
            OrderedVocabTrieArtifacts::new(ordered_vocab, trie),
            OrderedVocabCacheProfile {
                status: OrderedVocabCacheStatus::Disabled,
                probe_ns: 0,
                verify_ns: 0,
                ordered_vocab_build_ns,
                trie_build_ns,
                cache_entries: 0,
                capacity,
            },
        );
    }

    let probe_started_at = Instant::now();
    let fingerprint = ordered_vocab_cache_fingerprint(token_bytes);
    let mut verify_ns = 0u128;

    {
        let mut cache = ordered_vocab_cache().lock().unwrap();
        let mut hit_index = None;
        for (index, entry) in cache.iter().enumerate() {
            if entry.fingerprint != fingerprint {
                continue;
            }
            let verify_started_at = Instant::now();
            let is_match = ordered_vocab_cache_source_matches(
                token_bytes,
                entry.source_original_to_ordered.as_ref(),
                entry.artifacts.ordered_vocab.as_ref(),
            );
            verify_ns += verify_started_at.elapsed().as_nanos();
            if is_match {
                hit_index = Some(index);
                break;
            }
        }

        if let Some(index) = hit_index {
            let entry = cache.remove(index);
            let artifacts = entry.artifacts.clone();
            cache.push(entry);
            let cache_entries = cache.len();
            return (
                artifacts,
                OrderedVocabCacheProfile {
                    status: OrderedVocabCacheStatus::Hit,
                    probe_ns: probe_started_at.elapsed().as_nanos(),
                    verify_ns,
                    ordered_vocab_build_ns: 0,
                    trie_build_ns: 0,
                    cache_entries,
                    capacity,
                },
            );
        }
    }

    let ordered_vocab_started_at = Instant::now();
    let ordered_vocab = Arc::new(build_ordered_vocab(token_bytes));
    let ordered_vocab_build_ns = ordered_vocab_started_at.elapsed().as_nanos();
    let trie_started_at = Instant::now();
    let trie = Arc::new(build_ordered_vocab_prefix_tree(ordered_vocab.as_ref()));
    let trie_build_ns = trie_started_at.elapsed().as_nanos();
    let source_original_to_ordered = ordered_vocab_cache_source_original_to_ordered(ordered_vocab.as_ref());
    let entry = OrderedVocabCacheEntry {
        fingerprint,
        source_original_to_ordered,
        artifacts: OrderedVocabTrieArtifacts::new(
            Arc::clone(&ordered_vocab),
            Arc::clone(&trie),
        ),
    };

    let cache_entries = {
        let mut cache = ordered_vocab_cache().lock().unwrap();
        if cache.len() >= capacity {
            cache.remove(0);
        }
        cache.push(entry);
        cache.len()
    };

    (
        OrderedVocabTrieArtifacts::new(ordered_vocab, trie),
        OrderedVocabCacheProfile {
            status: OrderedVocabCacheStatus::Miss,
            probe_ns: probe_started_at.elapsed().as_nanos(),
            verify_ns,
            ordered_vocab_build_ns,
            trie_build_ns,
            cache_entries,
            capacity,
        },
    )
}

fn get_ordered_vocab_trie_artifacts_for_vocab(
    vocab: &Vocab,
) -> (OrderedVocabTrieArtifacts, OrderedVocabCacheProfile) {
    let capacity = ordered_vocab_cache_capacity();
    if !ordered_vocab_cache_enabled() || capacity == 0 {
        return get_ordered_vocab_trie_artifacts(vocab.entries_map());
    }

    let probe_started_at = Instant::now();
    if let Some(artifacts) = vocab.vocab_derived_cache_get::<OrderedVocabTrieArtifacts>() {
        return (
            artifacts.as_ref().clone(),
            OrderedVocabCacheProfile {
                status: OrderedVocabCacheStatus::Hit,
                probe_ns: probe_started_at.elapsed().as_nanos(),
                verify_ns: 0,
                ordered_vocab_build_ns: 0,
                trie_build_ns: 0,
                cache_entries: 1,
                capacity,
            },
        );
    }

    let ordered_vocab_started_at = Instant::now();
    let ordered_vocab = Arc::new(build_ordered_vocab(vocab.entries_map()));
    let ordered_vocab_build_ns = ordered_vocab_started_at.elapsed().as_nanos();
    let trie_started_at = Instant::now();
    let trie = Arc::new(build_ordered_vocab_prefix_tree(ordered_vocab.as_ref()));
    let trie_build_ns = trie_started_at.elapsed().as_nanos();
    let artifacts = OrderedVocabTrieArtifacts::new(ordered_vocab, trie);
    vocab.vocab_derived_cache_set(Arc::new(artifacts.clone()));

    (
        artifacts,
        OrderedVocabCacheProfile {
            status: OrderedVocabCacheStatus::Miss,
            probe_ns: probe_started_at.elapsed().as_nanos(),
            verify_ns: 0,
            ordered_vocab_build_ns,
            trie_build_ns,
            cache_entries: 1,
            capacity,
        },
    )
}

#[allow(dead_code)]
pub(crate) fn dense_word_count(token_slots: u32) -> usize { (token_slots as usize + 63) / 64 }

#[allow(dead_code)]
pub(crate) fn max_original_token_slot(token_bytes: &BTreeMap<u32, Vec<u8>>) -> u32 {
    token_bytes.keys().next_back().map(|token_id| token_id.saturating_add(1)).unwrap_or(0)
}

fn range_set_from_sorted_ids(ids: &[u32]) -> RangeSetBlaze<u32> {
    let Some((&first, rest)) = ids.split_first() else { return RangeSetBlaze::new(); };
    let mut ranges = Vec::new();
    let mut start = first;
    let mut end = first;
    for &id in rest {
        if id == end + 1 { end = id; }
        else { ranges.push(start..=end); start = id; end = id; }
    }
    ranges.push(start..=end);
    RangeSetBlaze::from_iter(ranges)
}

fn range_set_from_u128_mask(mask: u128) -> RangeSetBlaze<u32> {
    if mask == 0 {
        return RangeSetBlaze::new();
    }

    let mut ranges = Vec::new();
    let mut bits = mask;
    while bits != 0 {
        let start = bits.trailing_zeros();
        let mut end = start;
        bits &= !(1u128 << start);
        while bits != 0 {
            let next = bits.trailing_zeros();
            if next != end + 1 {
                break;
            }
            end = next;
            bits &= !(1u128 << next);
        }
        ranges.push(start..=end);
    }

    RangeSetBlaze::from_iter(ranges)
}

#[inline]
fn pm_vocab_equiv_enabled() -> bool {
    std::env::var("GLRMASK_PM_VOCAB_EQUIV")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(false)
}

#[inline]
fn pm_vocab_equiv_supported(tokenizer: &Tokenizer) -> bool {
    let _ = tokenizer;
    true
}

#[derive(Clone, Copy)]
struct PmTokenOutcome {
    terminals: u128,
    end_state: u32,
}

#[derive(Clone, Copy)]
struct NfaPmTokenOutcome {
    terminal_set: u32,
    end_config: u32,
}

#[derive(Default)]
struct NfaPmAnalysis<'a> {
    tokenizer: Option<&'a Tokenizer>,
    config_ids: FxHashMap<Vec<u32>, u32>,
    configs: Vec<Box<[u32]>>,
    transitions: FxHashMap<(u32, u8), u32>,
    terminal_set_ids: FxHashMap<Vec<u64>, u32>,
    terminal_sets: Vec<Box<[u64]>>,
    union_cache: FxHashMap<(u32, u32), u32>,
}

impl<'a> NfaPmAnalysis<'a> {
    fn new(tokenizer: &'a Tokenizer) -> Self {
        let mut analysis = Self {
            tokenizer: Some(tokenizer),
            ..Self::default()
        };
        analysis.intern_terminal_set(Vec::new());
        analysis
    }

    #[inline]
    fn tokenizer(&self) -> &'a Tokenizer {
        self.tokenizer.expect("NFA PM tokenizer missing")
    }

    fn intern_config(&mut self, states: Vec<u32>) -> u32 {
        if let Some(&id) = self.config_ids.get(&states) {
            return id;
        }
        let id = self.configs.len() as u32;
        self.config_ids.insert(states.clone(), id);
        self.configs.push(states.into_boxed_slice());
        id
    }

    fn config_for_raw_state(&mut self, raw_state: u32) -> u32 {
        let closure = self.tokenizer().execute_from_state_end_only(&[], raw_state);
        self.intern_config(closure.to_vec())
    }

    fn step_config(&mut self, config: u32, byte: u8) -> u32 {
        if let Some(&target) = self.transitions.get(&(config, byte)) {
            return target;
        }
        let states = self.configs[config as usize].to_vec();
        let targets = self.tokenizer().step_all(&states, byte);
        let target = if targets.is_empty() {
            u32::MAX
        } else {
            self.intern_config(targets.to_vec())
        };
        self.transitions.insert((config, byte), target);
        target
    }

    fn intern_terminal_set(&mut self, words: Vec<u64>) -> u32 {
        if let Some(&id) = self.terminal_set_ids.get(&words) {
            return id;
        }
        let id = self.terminal_sets.len() as u32;
        self.terminal_set_ids.insert(words.clone(), id);
        self.terminal_sets.push(words.into_boxed_slice());
        id
    }

    fn matched_terminal_set_for_config(&mut self, config: u32) -> u32 {
        let word_count = (self.tokenizer().num_terminals() as usize).div_ceil(64);
        let mut words = vec![0u64; word_count];
        for &state in self.configs[config as usize].iter() {
            for terminal in self.tokenizer().matched_terminals_iter(state) {
                let terminal = terminal as usize;
                words[terminal >> 6] |= 1u64 << (terminal & 63);
            }
        }
        self.intern_terminal_set(words)
    }

    fn union_terminal_sets(&mut self, left: u32, right: u32) -> u32 {
        if left == 0 {
            return right;
        }
        if right == 0 || left == right {
            return left;
        }
        let key = if left < right { (left, right) } else { (right, left) };
        if let Some(&id) = self.union_cache.get(&key) {
            return id;
        }
        let left_words = self.terminal_sets[left as usize].to_vec();
        let right_words = &self.terminal_sets[right as usize];
        let mut words = left_words;
        if words.len() < right_words.len() {
            words.resize(right_words.len(), 0);
        }
        for (word, &right_word) in words.iter_mut().zip(right_words.iter()) {
            *word |= right_word;
        }
        let id = self.intern_terminal_set(words);
        self.union_cache.insert(key, id);
        id
    }

    fn advance_outcomes(
        &mut self,
        parent: &[NfaPmTokenOutcome],
        segment: &[u8],
    ) -> Vec<NfaPmTokenOutcome> {
        let mut child = Vec::with_capacity(parent.len());
        for &outcome in parent {
            let mut terminal_set = outcome.terminal_set;
            let mut current_config = outcome.end_config;
            if current_config != u32::MAX {
                for &byte in segment {
                    current_config = self.step_config(current_config, byte);
                    if current_config == u32::MAX {
                        break;
                    }
                    let matched = self.matched_terminal_set_for_config(current_config);
                    terminal_set = self.union_terminal_sets(terminal_set, matched);
                }
            }
            child.push(NfaPmTokenOutcome {
                terminal_set,
                end_config: current_config,
            });
        }
        child
    }
}

struct NfaPmVocabEquivBuilder<'a, 'b> {
    ordered_vocab: &'a OrderedVocab,
    analysis: &'b mut NfaPmAnalysis<'a>,
    signature_buckets: FxHashMap<u64, Vec<u32>>,
    signatures: Vec<Vec<u32>>,
    original_to_internal: Vec<u32>,
    internal_to_originals: Vec<Vec<u32>>,
    representative_original_ids: Vec<u32>,
}

impl<'a, 'b> NfaPmVocabEquivBuilder<'a, 'b> {
    fn new(ordered_vocab: &'a OrderedVocab, analysis: &'b mut NfaPmAnalysis<'a>) -> Self {
        Self {
            ordered_vocab,
            analysis,
            signature_buckets: FxHashMap::default(),
            signatures: Vec::new(),
            original_to_internal: vec![u32::MAX; ordered_vocab.original_slot_count],
            internal_to_originals: Vec::new(),
            representative_original_ids: Vec::new(),
        }
    }

    fn signature_hash(outcomes: &[NfaPmTokenOutcome]) -> u64 {
        outcomes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, outcome| {
            mix_pm_signature_word(hash, outcome.terminal_set as u64)
        })
    }

    fn intern_signature(&mut self, outcomes: &[NfaPmTokenOutcome]) -> u32 {
        let hash = Self::signature_hash(outcomes);
        if let Some(bucket) = self.signature_buckets.get(&hash) {
            for &signature_id in bucket {
                let signature = &self.signatures[signature_id as usize];
                if signature.len() == outcomes.len()
                    && signature
                        .iter()
                        .zip(outcomes.iter())
                        .all(|(&left, right)| left == right.terminal_set)
                {
                    return signature_id;
                }
            }
        }
        let signature_id = self.signatures.len() as u32;
        self.signatures.push(outcomes.iter().map(|outcome| outcome.terminal_set).collect());
        self.signature_buckets.entry(hash).or_default().push(signature_id);
        signature_id
    }

    fn record_token(&mut self, ordered_token_id: usize, outcomes: &[NfaPmTokenOutcome]) {
        let class_id = self.intern_signature(outcomes);
        let class_idx = class_id as usize;
        while self.internal_to_originals.len() <= class_idx {
            self.internal_to_originals.push(Vec::new());
            self.representative_original_ids.push(u32::MAX);
        }
        let Some(originals) = self.ordered_vocab.ordered_to_originals.get(ordered_token_id) else {
            return;
        };
        for &original in originals {
            if let Some(slot) = self.original_to_internal.get_mut(original as usize) {
                *slot = class_id;
            }
            if self.representative_original_ids[class_idx] == u32::MAX {
                self.representative_original_ids[class_idx] = original;
            }
            self.internal_to_originals[class_idx].push(original);
        }
    }

    fn visit(&mut self, node: &VocabPrefixTreeNode, outcomes: &[NfaPmTokenOutcome]) {
        if node.has_token() {
            self.record_token(node.token_id(), outcomes);
        }
        for (segment, child) in node.iter_children() {
            let child_outcomes = self.analysis.advance_outcomes(outcomes, segment);
            self.visit(child, &child_outcomes);
        }
    }

    fn finish(mut self) -> ManyToOneIdMap {
        for originals in &mut self.internal_to_originals {
            originals.sort_unstable();
            originals.dedup();
        }
        ManyToOneIdMap {
            original_to_internal: self.original_to_internal,
            internal_to_originals: self.internal_to_originals,
            representative_original_ids: self.representative_original_ids,
        }
    }
}

#[inline]
fn mix_pm_signature_word(hash: u64, word: u64) -> u64 {
    let mixed = word.wrapping_add(0x9e37_79b9_7f4a_7c15);
    hash ^ mixed
        .wrapping_add(hash << 6)
        .wrapping_add(hash >> 2)
        .wrapping_mul(0x517c_c1b7_2722_0a95)
}

fn pm_signature_hash(outcomes: &[PmTokenOutcome]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for outcome in outcomes {
        hash = mix_pm_signature_word(hash, outcome.terminals as u64);
        hash = mix_pm_signature_word(hash, (outcome.terminals >> 64) as u64);
    }
    hash
}

fn pm_signature_matches(signature: &[u128], outcomes: &[PmTokenOutcome]) -> bool {
    signature.len() == outcomes.len()
        && signature
            .iter()
            .zip(outcomes.iter())
            .all(|(&left, right)| left == right.terminals)
}

fn intern_pm_token_signature(
    outcomes: &[PmTokenOutcome],
    buckets: &mut FxHashMap<u64, Vec<u32>>,
    signatures: &mut Vec<Vec<u128>>,
) -> u32 {
    let hash = pm_signature_hash(outcomes);
    if let Some(bucket) = buckets.get(&hash) {
        for &signature_id in bucket {
            if pm_signature_matches(&signatures[signature_id as usize], outcomes) {
                return signature_id;
            }
        }
    }

    let signature_id = signatures.len() as u32;
    let signature = outcomes
        .iter()
        .map(|outcome| outcome.terminals)
        .collect::<Vec<_>>();
    signatures.push(signature);
    buckets.entry(hash).or_default().push(signature_id);
    signature_id
}

fn advance_pm_token_outcomes(
    parent: &[PmTokenOutcome],
    segment: &[u8],
    byte_transitions: &[Vec<u32>],
    matched_terminal_masks: &[u128],
) -> Vec<PmTokenOutcome> {
    let mut child = Vec::with_capacity(parent.len());
    for &outcome in parent {
        let mut terminals = outcome.terminals;
        let mut current_state = outcome.end_state;
        if current_state != u32::MAX {
            for &byte in segment {
                let next_state = byte_transitions[byte as usize][current_state as usize];
                if next_state == u32::MAX {
                    current_state = u32::MAX;
                    break;
                }
                current_state = next_state;
                terminals |= matched_terminal_masks[current_state as usize];
            }
        }
        child.push(PmTokenOutcome {
            terminals,
            end_state: current_state,
        });
    }
    child
}

struct PmVocabEquivBuilder<'a> {
    ordered_vocab: &'a OrderedVocab,
    byte_transitions: &'a [Vec<u32>],
    matched_terminal_masks: &'a [u128],
    signature_buckets: FxHashMap<u64, Vec<u32>>,
    signatures: Vec<Vec<u128>>,
    original_to_internal: Vec<u32>,
    internal_to_originals: Vec<Vec<u32>>,
    representative_original_ids: Vec<u32>,
}

impl<'a> PmVocabEquivBuilder<'a> {
    fn new(
        ordered_vocab: &'a OrderedVocab,
        byte_transitions: &'a [Vec<u32>],
        matched_terminal_masks: &'a [u128],
    ) -> Self {
        Self {
            ordered_vocab,
            byte_transitions,
            matched_terminal_masks,
            signature_buckets: FxHashMap::default(),
            signatures: Vec::new(),
            original_to_internal: vec![u32::MAX; ordered_vocab.original_slot_count],
            internal_to_originals: Vec::new(),
            representative_original_ids: Vec::new(),
        }
    }

    fn record_token(&mut self, ordered_token_id: usize, outcomes: &[PmTokenOutcome]) {
        let class_id = intern_pm_token_signature(
            outcomes,
            &mut self.signature_buckets,
            &mut self.signatures,
        );
        let class_idx = class_id as usize;
        while self.internal_to_originals.len() <= class_idx {
            self.internal_to_originals.push(Vec::new());
            self.representative_original_ids.push(u32::MAX);
        }
        let Some(originals) = self.ordered_vocab.ordered_to_originals.get(ordered_token_id) else {
            return;
        };
        for &original in originals {
            if let Some(slot) = self.original_to_internal.get_mut(original as usize) {
                *slot = class_id;
            }
            if self.representative_original_ids[class_idx] == u32::MAX {
                self.representative_original_ids[class_idx] = original;
            }
            self.internal_to_originals[class_idx].push(original);
        }
    }

    fn visit(&mut self, node: &VocabPrefixTreeNode, outcomes: &[PmTokenOutcome]) {
        if node.has_token() {
            self.record_token(node.token_id(), outcomes);
        }
        for (segment, child) in node.iter_children() {
            let child_outcomes = advance_pm_token_outcomes(
                outcomes,
                segment,
                self.byte_transitions,
                self.matched_terminal_masks,
            );
            self.visit(child, &child_outcomes);
        }
    }

    fn finish(mut self) -> ManyToOneIdMap {
        for originals in &mut self.internal_to_originals {
            originals.sort_unstable();
            originals.dedup();
        }
        ManyToOneIdMap {
            original_to_internal: self.original_to_internal,
            internal_to_originals: self.internal_to_originals,
            representative_original_ids: self.representative_original_ids,
        }
    }
}

fn compute_pm_vocab_equivalence_map(
    tokenizer: &Tokenizer,
    ordered_vocab: &OrderedVocab,
    trie: &VocabPrefixTree,
) -> ManyToOneIdMap {
    if tokenizer.has_epsilon_transitions() {
        let mut analysis = NfaPmAnalysis::new(tokenizer);
        let root_outcomes = (0..tokenizer.num_states())
            .map(|state| NfaPmTokenOutcome {
                terminal_set: 0,
                end_config: analysis.config_for_raw_state(state),
            })
            .collect::<Vec<_>>();
        let mut builder = NfaPmVocabEquivBuilder::new(ordered_vocab, &mut analysis);
        builder.visit(&trie.root, &root_outcomes);
        return builder.finish();
    }
    let num_states = tokenizer.num_states() as usize;
    let mut byte_transitions = vec![vec![u32::MAX; num_states]; 256];
    for state_idx in 0..num_states {
        for (byte, target) in tokenizer.transitions_from(state_idx as u32) {
            byte_transitions[byte as usize][state_idx] = target;
        }
    }

    let mut matched_terminal_masks = Vec::with_capacity(num_states);
    for state in 0..tokenizer.num_states() {
        let mut mask = 0u128;
        for terminal in tokenizer.matched_terminals_iter(state) {
            if terminal < 128 {
                mask |= 1u128 << terminal;
            }
        }
        matched_terminal_masks.push(mask);
    }

    // For a deterministic-component dispatch, the synthetic reset state's PM
    // behavior is the union of the component-root behaviors.  Equality at all
    // physical component states therefore implies equality at the dispatch
    // state, so including it as a dead scalar row would be both unnecessary
    // and incorrect.
    let dispatch_start = tokenizer
        .has_deterministic_dispatch()
        .then(|| tokenizer.start_state());
    let root_outcomes = (0..tokenizer.num_states())
        .filter(|state| Some(*state) != dispatch_start)
        .map(|state| PmTokenOutcome {
            terminals: 0,
            end_state: state,
        })
        .collect::<Vec<_>>();
    let mut builder = PmVocabEquivBuilder::new(
        ordered_vocab,
        &byte_transitions,
        &matched_terminal_masks,
    );
    builder.visit(&trie.root, &root_outcomes);
    builder.finish()
}

fn compute_pm_vocab_equivalence_map_fast(
    tokenizer: &Tokenizer,
    ordered_vocab: &OrderedVocab,
) -> ManyToOneIdMap {
    let num_states = tokenizer.num_states() as usize;
    let mut transitions = vec![u32::MAX; num_states * 256];
    let states = (0..num_states)
        .map(|state_idx| {
            let base = state_idx * 256;
            for (byte, target) in tokenizer.transitions_from(state_idx as u32) {
                transitions[base + byte as usize] = target;
            }
            FlatDfaState {
                finalizers: tokenizer
                    .matched_terminals_iter(state_idx as u32)
                    .map(|terminal| terminal as usize)
                    .collect(),
                // The equivalence signature is based on terminals actually
                // reached while consuming a token, but the fast walker also
                // uses future groups to identify genuinely dead states.  An
                // empty vector here marks every non-final state dead and can
                // collapse the entire vocabulary into one bogus class.
                possible_future_group_ids: tokenizer
                    .possible_future_terminals_iter(state_idx as u32)
                    .map(|terminal| terminal as usize)
                    .collect(),
            }
        })
        .collect::<Vec<_>>();
    let tokenizer_view = TokenizerView {
        flat_dfa: FlatDfa {
            states,
            start_state: tokenizer.start_state() as usize,
            transitions: Arc::from(transitions),
        },
    };
    let strings = ordered_vocab
        .ordered_token_bytes
        .iter()
        .map(|bytes| bytes.as_slice())
        .collect::<Vec<_>>();
    let dispatch_start = tokenizer
        .has_deterministic_dispatch()
        .then(|| tokenizer.start_state() as usize);
    let initial_states = (0..tokenizer.num_states() as usize)
        .filter(|state| Some(*state) != dispatch_start)
        .collect::<Vec<_>>();
    let disallowed_follows = BTreeMap::<u32, BitSet>::new();
    let classes = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
        &tokenizer_view,
        &strings,
        &initial_states,
        &disallowed_follows,
        None,
        None,
        None,
        None,
    );

    let mut original_to_internal = vec![u32::MAX; ordered_vocab.original_slot_count];
    let mut internal_to_originals = Vec::new();
    let mut representative_original_ids = Vec::new();
    for class in classes {
        let internal = internal_to_originals.len() as u32;
        let mut originals = Vec::new();
        for ordered_id in class {
            if let Some(ordered_originals) = ordered_vocab.ordered_to_originals.get(ordered_id) {
                for &original in ordered_originals {
                    if let Some(slot) = original_to_internal.get_mut(original as usize) {
                        *slot = internal;
                    }
                    originals.push(original);
                }
            }
        }
        originals.sort_unstable();
        originals.dedup();
        let representative = originals.first().copied().unwrap_or(u32::MAX);
        internal_to_originals.push(originals);
        representative_original_ids.push(representative);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn used_state_class_ids(state_classes: &[u32]) -> Vec<u32> {
    let mut ids: Vec<u32> = state_classes.iter().copied().filter(|&class_id| class_id != u32::MAX).collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn next_nonzero_stamp(generation: &mut u32, stamps: &mut [u32]) -> u32 {
    *generation = generation.wrapping_add(1);
    if *generation == 0 {
        stamps.fill(0);
        *generation = 1;
    }
    *generation
}

fn push_sweep_event(events: &mut [Vec<SweepEvent>], event_positions: &mut Vec<u32>, position: u32, event: SweepEvent) {
    let Some(bucket) = events.get_mut(position as usize) else { return; };
    if bucket.is_empty() { event_positions.push(position); }
    bucket.push(event);
}

fn intern_state_terminal_label(
    label_ids: &mut FxHashMap<StateTerminalLabel, u32>,
    labels_by_id: &mut Vec<StateTerminalLabel>,
    label: StateTerminalLabel,
) -> u32 {
    if let Some(&label_id) = label_ids.get(&label) {
        label_id
    } else {
        let label_id = labels_by_id.len() as u32;
        labels_by_id.push(label);
        label_ids.insert(label, label_id);
        label_id
    }
}

fn build_sweep_events(
    class_maps: &[Arc<IntervalPossibleMatchMap>],
    state_classes: &[u32],
    num_ordered_tokens: usize,
) -> (Vec<Vec<SweepEvent>>, Vec<u32>, Vec<SweepGroup>, Vec<StateTerminalLabel>, SweepBuildStats) {
    let mut events = vec![Vec::new(); num_ordered_tokens + 1];
    let mut event_positions = Vec::new();
    let mut groups = Vec::<SweepGroup>::new();
    let mut labels_by_id = Vec::<StateTerminalLabel>::new();
    let mut label_ids = FxHashMap::<StateTerminalLabel, u32>::default();
    let mut stats = SweepBuildStats::default();

    let used_state_classes = used_state_class_ids(state_classes);
    stats.used_state_classes = used_state_classes.len();

    for class_id in used_state_classes {
        let Some(class_map) = class_maps.get(class_id as usize) else { continue; };
        for entry in class_map.iter() {
            if entry.terminals.is_empty() || entry.ranges.is_empty() { continue; }

            let mut group_label_ids = Vec::with_capacity(entry.terminals.len());
            for &terminal_id in entry.terminals.iter() {
                group_label_ids.push(intern_state_terminal_label(&mut label_ids, &mut labels_by_id, (class_id, terminal_id)));
            }
            group_label_ids.sort_unstable();
            group_label_ids.dedup();
            if group_label_ids.is_empty() { continue; }

            let group_id = groups.len() as u32;
            stats.group_label_refs += group_label_ids.len();
            groups.push(SweepGroup { label_ids: group_label_ids.into_boxed_slice() });

            for &(lo, mut hi) in entry.ranges.iter() {
                if num_ordered_tokens == 0 { continue; }
                let max_token = num_ordered_tokens as u32 - 1;
                if lo > max_token { continue; }
                hi = hi.min(max_token);
                if lo > hi { continue; }
                stats.total_intervals += 1;
                push_sweep_event(&mut events, &mut event_positions, lo, SweepEvent { add: true, group_id });
                stats.total_events += 1;
                let after = hi.saturating_add(1);
                if after <= num_ordered_tokens as u32 {
                    push_sweep_event(&mut events, &mut event_positions, after, SweepEvent { add: false, group_id });
                    stats.total_events += 1;
                }
            }
        }
    }

    event_positions.sort_unstable();
    event_positions.dedup();
    stats.terminal_groups = groups.len();
    stats.terminal_labels = labels_by_id.len();
    (events, event_positions, groups, labels_by_id, stats)
}

#[inline]
fn active_group_hash(group_id: u32) -> u64 {
    let mut value = (group_id as u64).wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn insert_active_group_id(
    active_group_ids: &mut Vec<u32>,
    active_group_positions: &mut [u32],
    active_group_fingerprint: &mut u64,
    group_id: u32,
) {
    let slot = &mut active_group_positions[group_id as usize];
    if *slot != u32::MAX {
        return;
    }
    *slot = active_group_ids.len() as u32;
    active_group_ids.push(group_id);
    *active_group_fingerprint ^= active_group_hash(group_id);
}

fn remove_active_group_id(
    active_group_ids: &mut Vec<u32>,
    active_group_positions: &mut [u32],
    active_group_fingerprint: &mut u64,
    group_id: u32,
) {
    let remove_index = active_group_positions[group_id as usize] as usize;
    debug_assert!(remove_index < active_group_ids.len());
    let removed_group_id = active_group_ids.swap_remove(remove_index);
    debug_assert_eq!(removed_group_id, group_id);
    if remove_index < active_group_ids.len() {
        let moved_group_id = active_group_ids[remove_index];
        active_group_positions[moved_group_id as usize] = remove_index as u32;
    }
    active_group_positions[group_id as usize] = u32::MAX;
    *active_group_fingerprint ^= active_group_hash(group_id);
}

fn apply_sweep_events(
    active_group_counts: &mut [u32],
    events: &[SweepEvent],
    active_group_ids: &mut Vec<u32>,
    active_group_positions: &mut [u32],
    active_group_fingerprint: &mut u64,
) {
    for event in events.iter().filter(|event| !event.add) {
        let count = &mut active_group_counts[event.group_id as usize];
        assert!(*count > 0, "pmv sweep removal underflow for group_id={}", event.group_id);
        if *count == 1 {
            remove_active_group_id(active_group_ids, active_group_positions, active_group_fingerprint, event.group_id);
        }
        *count -= 1;
    }
    for event in events.iter().filter(|event| event.add) {
        let count = &mut active_group_counts[event.group_id as usize];
        if *count == 0 {
            insert_active_group_id(active_group_ids, active_group_positions, active_group_fingerprint, event.group_id);
        }
        *count += 1;
    }
}

fn active_group_key_matches(
    active_group_counts: &[u32],
    active_group_ids: &[u32],
    sorted_key: &[u32],
) -> bool {
    if active_group_ids.len() != sorted_key.len() {
        return false;
    }
    sorted_key.iter().all(|&group_id| active_group_counts[group_id as usize] > 0)
}

fn build_signature_from_active_groups(
    active_group_counts: &[u32],
    active_group_count: usize,
    groups: &[SweepGroup],
    labels_by_id: &[StateTerminalLabel],
    label_stamps: &mut [u32],
    stamp_generation: &mut u32,
) -> Vec<StateTerminalLabel> {
    if active_group_count == 0 { return Vec::new(); }
    let stamp = next_nonzero_stamp(stamp_generation, label_stamps);
    let mut signature = Vec::new();
    for (group_id, group) in groups.iter().enumerate() {
        if active_group_counts[group_id] == 0 { continue; }
        for &label_id in group.label_ids.iter() {
            let stamp_slot = &mut label_stamps[label_id as usize];
            if *stamp_slot != stamp {
                *stamp_slot = stamp;
                signature.push(labels_by_id[label_id as usize]);
            }
        }
    }
    signature.sort_unstable();
    signature
}

fn build_signature_from_active_group_ids(
    active_group_ids: &[u32],
    groups: &[SweepGroup],
    labels_by_id: &[StateTerminalLabel],
    label_stamps: &mut [u32],
    stamp_generation: &mut u32,
) -> Vec<StateTerminalLabel> {
    if active_group_ids.is_empty() { return Vec::new(); }

    let stamp = next_nonzero_stamp(stamp_generation, label_stamps);
    let mut signature = Vec::new();
    for &group_id in active_group_ids {
        let Some(group) = groups.get(group_id as usize) else { continue; };
        for &label_id in group.label_ids.iter() {
            let stamp_slot = &mut label_stamps[label_id as usize];
            if *stamp_slot != stamp {
                *stamp_slot = stamp;
                signature.push(labels_by_id[label_id as usize]);
            }
        }
    }
    signature.sort_unstable();
    signature
}

fn build_possible_match_vocab_and_weights_from_interval_maps(
    class_maps: &[Arc<IntervalPossibleMatchMap>],
    state_classes: &[u32],
    ordered_vocab: &OrderedVocab,
) -> (PossibleMatchVocabMap, RuntimePossibleMatchesByTerminal) {
    let num_ordered_tokens = ordered_vocab.ordered_to_originals.len();
    let pmv_detail_enabled = std::env::var("GLRMASK_PROFILE_PMV_DETAIL")
        .map(|value| value == "1")
        .unwrap_or(false);

    if group_pmv_legacy_enabled() {
        if pmv_detail_enabled {
            eprintln!("[glrmask/profile][pmv_detail] stage=legacy_expanded enabled=1");
        }
        return build_legacy_possible_match_vocab_and_weights_from_interval_maps(class_maps, state_classes, ordered_vocab);
    }

    let sweep_events_started_at = Instant::now();
    let (events, event_positions, groups, labels_by_id, sweep_build_stats) =
        build_sweep_events(class_maps, state_classes, num_ordered_tokens);
    let sweep_events_ms = elapsed_ms(sweep_events_started_at);

    let mut signature_to_id: FxHashMap<Vec<StateTerminalLabel>, SignatureClassId> = FxHashMap::default();
    let mut active_group_signature_to_signature_id: FxHashMap<u64, Vec<(Vec<u32>, SignatureClassId)>> = FxHashMap::default();
    let mut signature_labels: Vec<Vec<StateTerminalLabel>> = Vec::new();
    let mut original_to_internal = vec![u32::MAX; ordered_vocab.original_slot_count];
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut active_group_counts = vec![0u32; groups.len()];
    let mut active_group_ids = Vec::<u32>::new();
    let mut active_group_positions = vec![u32::MAX; groups.len()];
    let mut active_group_fingerprint = 0u64;
    let mut label_stamps = vec![0u32; labels_by_id.len()];
    let mut stamp_generation = 0u32;

    let sweep_started_at = Instant::now();
    let mut signature_build_ms = 0.0;
    let mut signature_lookup_ms = 0.0;
    let mut assignment_ms = 0.0;
    let mut sweep_segments = 0usize;
    let mut active_group_signature_cache_hits = 0usize;
    let mut active_group_signature_cache_misses = 0usize;
    let mut active_group_signature_build_ms = 0.0;
    let mut label_signature_build_ms = 0.0;
    let mut total_active_signature_len = 0usize;
    let mut max_active_signature_len = 0usize;
    let mut total_active_group_len = 0usize;
    let mut max_active_group_len = 0usize;

    let mut event_index = 0usize;
    let mut position = 0usize;
    while position < num_ordered_tokens {
        while event_index < event_positions.len() && event_positions[event_index] as usize == position {
            apply_sweep_events(
                &mut active_group_counts,
                &events[position],
                &mut active_group_ids,
                &mut active_group_positions,
                &mut active_group_fingerprint,
            );
            event_index += 1;
        }

        let next_position = event_positions.get(event_index).map(|&next| (next as usize).min(num_ordered_tokens)).unwrap_or(num_ordered_tokens);
        let active_group_signature_started_at = Instant::now();
        sweep_segments += 1;
        total_active_group_len += active_group_ids.len();
        max_active_group_len = max_active_group_len.max(active_group_ids.len());
        let cached_signature_id = active_group_signature_to_signature_id
            .get(&active_group_fingerprint)
            .and_then(|bucket| {
                bucket.iter().find_map(|(sorted_key, signature_id)| {
                    if active_group_key_matches(&active_group_counts, &active_group_ids, sorted_key) {
                        Some(*signature_id)
                    } else {
                        None
                    }
                })
            });
        active_group_signature_build_ms += elapsed_ms(active_group_signature_started_at);

        let signature_lookup_started_at = Instant::now();
        let signature_id = if let Some(existing) = cached_signature_id {
            active_group_signature_cache_hits += 1;
            existing
        } else {
            active_group_signature_cache_misses += 1;
            let label_signature_started_at = Instant::now();
            let signature = build_signature_from_active_group_ids(
                &active_group_ids,
                &groups,
                &labels_by_id,
                &mut label_stamps,
                &mut stamp_generation,
            );
            label_signature_build_ms += elapsed_ms(label_signature_started_at);

            let signature_id = if let Some(&existing) = signature_to_id.get(&signature) {
                existing
            } else {
                let new_id = signature_labels.len() as SignatureClassId;
                signature_to_id.insert(signature.clone(), new_id);
                signature_labels.push(signature);
                internal_to_originals.push(Vec::new());
                new_id
            };
            let active_group_key_started_at = Instant::now();
            let mut active_group_key = active_group_ids.clone();
            active_group_key.sort_unstable();
            active_group_signature_build_ms += elapsed_ms(active_group_key_started_at);
            active_group_signature_to_signature_id
                .entry(active_group_fingerprint)
                .or_default()
                .push((active_group_key, signature_id));
            signature_id
        };
        signature_lookup_ms += elapsed_ms(signature_lookup_started_at);
        signature_build_ms = active_group_signature_build_ms + label_signature_build_ms;

        let signature_len = signature_labels
            .get(signature_id as usize)
            .map(|labels| labels.len())
            .unwrap_or(0);
        total_active_signature_len += signature_len;
        max_active_signature_len = max_active_signature_len.max(signature_len);

        let assignment_started_at = Instant::now();
        for ordered_id in position..next_position {
            for &original in &ordered_vocab.ordered_to_originals[ordered_id] {
                if let Some(slot) = original_to_internal.get_mut(original as usize) { *slot = signature_id; }
            }
        }
        assignment_ms += elapsed_ms(assignment_started_at);
        position = next_position;
    }
    let sweep_ms = elapsed_ms(sweep_started_at);

    let internal_to_originals_started_at = Instant::now();
    for (original, &signature_id) in original_to_internal.iter().enumerate() {
        if signature_id != u32::MAX {
            internal_to_originals[signature_id as usize].push(original as u32);
        }
    }
    let sort_dedup_ms = elapsed_ms(internal_to_originals_started_at);

    let ids_by_label_started_at = Instant::now();
    let use_bitmask_ids_by_label = signature_labels.len() <= u128::BITS as usize;
    let mut label_entries = 0usize;
    let mut ids_by_label: BTreeMap<TerminalID, BTreeMap<u32, Vec<u32>>> = BTreeMap::new();
    let mut pair_masks = FxHashMap::<(TerminalID, u32), u128>::default();
    if use_bitmask_ids_by_label {
        for (signature_id, labels) in signature_labels.iter().enumerate() {
            let bit = 1u128 << signature_id;
            for &(class_id, terminal_id) in labels {
                label_entries += 1;
                *pair_masks.entry((terminal_id, class_id)).or_insert(0) |= bit;
            }
        }
    } else {
        for (signature_id, labels) in signature_labels.iter().enumerate() {
            let signature_id = signature_id as u32;
            for &(class_id, terminal_id) in labels {
                label_entries += 1;
                ids_by_label.entry(terminal_id).or_default().entry(class_id).or_default().push(signature_id);
            }
        }
    }
    let ids_by_label_ms = elapsed_ms(ids_by_label_started_at);

    let weight_build_started_at = Instant::now();
    let mut state_token_sets = 0usize;
    let mut bitmask_unique_masks = 0usize;
    let mut bitmask_mask_cache_hits = 0usize;
    let mut bitmask_mask_cache_misses = 0usize;
    let possible_matches: RuntimePossibleMatchesByTerminal = if use_bitmask_ids_by_label {
        let mut by_terminal: BTreeMap<TerminalID, Vec<(u32, u128)>> = BTreeMap::new();
        for ((terminal_id, class_id), mask) in pair_masks {
            by_terminal.entry(terminal_id).or_default().push((class_id, mask));
        }
        let mut shared_token_set_by_mask = FxHashMap::<u128, std::sync::Arc<RangeSetBlaze<u32>>>::default();
        by_terminal.into_iter().map(|(terminal_id, mut by_state)| {
            by_state.sort_unstable_by_key(|(state, _)| *state);
            let mut entries = Vec::new();
            for (state, mask) in by_state {
                if mask == 0 {
                    continue;
                }
                let shared_token_set = if let Some(existing) = shared_token_set_by_mask.get(&mask) {
                    bitmask_mask_cache_hits += 1;
                    existing.clone()
                } else {
                    bitmask_mask_cache_misses += 1;
                    let token_set = shared_rangeset(range_set_from_u128_mask(mask));
                    shared_token_set_by_mask.insert(mask, token_set.clone());
                    token_set
                };
                state_token_sets += 1;
                entries.push((state, shared_token_set));
            }
            if !entries.is_empty() {
                bitmask_unique_masks = shared_token_set_by_mask.len();
            }
            (terminal_id, Weight::from_per_tsid_shared(entries.into_iter()))
        }).filter(|(_, weight)| !weight.is_empty()).collect()
    } else {
        ids_by_label.into_iter().map(|(terminal_id, by_state)| {
            let mut entries = Vec::new();
            // `ids` are appended while iterating `signature_labels` in increasing
            // `signature_id` order, and labels are deduped within each signature,
            // so each bucket is already strictly increasing and unique.
            for (state, ids) in by_state {
                let token_set = range_set_from_sorted_ids(&ids);
                if !token_set.is_empty() {
                    state_token_sets += 1;
                    entries.push((state, shared_rangeset(token_set)));
                }
            }
            (terminal_id, Weight::from_per_tsid_shared(entries.into_iter()))
        }).filter(|(_, weight)| !weight.is_empty()).collect()
    };
    let terminal_ids = possible_matches.len();
    let weight_build_ms = elapsed_ms(weight_build_started_at);

    if pmv_detail_enabled {
        let mean_active_signature_len = if sweep_segments == 0 {
            0.0
        } else {
            total_active_signature_len as f64 / sweep_segments as f64
        };
        let mean_active_group_len = if sweep_segments == 0 {
            0.0
        } else {
            total_active_group_len as f64 / sweep_segments as f64
        };
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=group_sweep_events sweep_events_ms={:.3} event_positions={} total_group_events={} used_state_classes={} total_group_intervals={} terminal_groups={} terminal_labels={} group_label_refs={}",
            sweep_events_ms,
            event_positions.len(),
            sweep_build_stats.total_events,
            sweep_build_stats.used_state_classes,
            sweep_build_stats.total_intervals,
            sweep_build_stats.terminal_groups,
            sweep_build_stats.terminal_labels,
            sweep_build_stats.group_label_refs,
        );
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=sweep sweep_ms={:.3} segments={} signature_build_ms={:.3} signature_lookup_ms={:.3} assignment_ms={:.3} active_group_signature_cache_hits={} active_group_signature_cache_misses={} active_group_signature_build_ms={:.3} label_signature_build_ms={:.3} unique_signatures={} max_active_signature_len={} mean_active_signature_len={:.3} max_active_groups={} mean_active_groups={:.3}",
            sweep_ms,
            sweep_segments,
            signature_build_ms,
            signature_lookup_ms,
            assignment_ms,
            active_group_signature_cache_hits,
            active_group_signature_cache_misses,
            active_group_signature_build_ms,
            label_signature_build_ms,
            signature_labels.len(),
            max_active_signature_len,
            mean_active_signature_len,
            max_active_group_len,
            mean_active_group_len,
        );
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=sort_dedup sort_dedup_ms={:.3} internal_signature_classes={}",
            sort_dedup_ms,
            internal_to_originals.len(),
        );
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=ids_by_label ids_by_label_ms={:.3} label_entries={} terminal_ids={} bitmask_path_used={}",
            ids_by_label_ms,
            label_entries,
            terminal_ids,
            use_bitmask_ids_by_label,
        );
        eprintln!(
            "[glrmask/profile][pmv_detail] stage=weights weights_ms={:.3} terminal_ids={} state_token_sets={} bitmask_path_used={} bitmask_unique_masks={} bitmask_mask_cache_hits={} bitmask_mask_cache_misses={}",
            weight_build_ms,
            terminal_ids,
            state_token_sets,
            use_bitmask_ids_by_label,
            bitmask_unique_masks,
            bitmask_mask_cache_hits,
            bitmask_mask_cache_misses,
        );
    }

    let possible_match_vocab = PossibleMatchVocabMap { original_to_internal, internal_to_originals };
    if group_pmv_validation_enabled() {
        validate_group_pmv_outputs(class_maps, state_classes, ordered_vocab, &possible_match_vocab, &possible_matches);
    }

    (possible_match_vocab, possible_matches)
}


type ExpandedIntervalPossibleMatchMap = BTreeMap<TerminalID, Vec<(u32, u32)>>;

#[derive(Debug, Clone, Copy)]
struct LegacySweepEvent {
    add: bool,
    label_id: u32,
}

fn normalize_token_ranges(ranges: &mut Vec<(u32, u32)>) {
    if ranges.len() <= 1 { return; }
    ranges.sort_unstable();
    let mut write = 0usize;
    for read in 1..ranges.len() {
        let (start, end) = ranges[read];
        let current = &mut ranges[write];
        if start <= current.1.saturating_add(1) {
            current.1 = current.1.max(end);
        } else {
            write += 1;
            ranges[write] = (start, end);
        }
    }
    ranges.truncate(write + 1);
}

fn append_expanded_ranges(
    map: &mut ExpandedIntervalPossibleMatchMap,
    terminal: TerminalID,
    ranges: &[(u32, u32)],
) {
    if !ranges.is_empty() {
        map.entry(terminal).or_default().extend_from_slice(ranges);
    }
}

fn normalize_expanded_interval_map(map: &mut ExpandedIntervalPossibleMatchMap) {
    map.retain(|_, ranges| {
        normalize_token_ranges(ranges);
        !ranges.is_empty()
    });
}

fn expand_interval_class_maps(
    class_maps: &[Arc<IntervalPossibleMatchMap>],
) -> Vec<Arc<ExpandedIntervalPossibleMatchMap>> {
    class_maps.iter().map(|class_map| {
        let mut expanded = ExpandedIntervalPossibleMatchMap::new();
        for entry in class_map.iter() {
            for &terminal_id in entry.terminals.iter() {
                append_expanded_ranges(&mut expanded, terminal_id, &entry.ranges);
            }
        }
        normalize_expanded_interval_map(&mut expanded);
        Arc::new(expanded)
    }).collect()
}

fn push_legacy_sweep_event(
    events: &mut [Vec<LegacySweepEvent>],
    event_positions: &mut Vec<u32>,
    position: u32,
    event: LegacySweepEvent,
) {
    let Some(bucket) = events.get_mut(position as usize) else { return; };
    if bucket.is_empty() { event_positions.push(position); }
    bucket.push(event);
}

fn build_legacy_sweep_events(
    class_maps: &[Arc<ExpandedIntervalPossibleMatchMap>],
    state_classes: &[u32],
    num_ordered_tokens: usize,
) -> (Vec<Vec<LegacySweepEvent>>, Vec<u32>, Vec<StateTerminalLabel>) {
    let mut events = vec![Vec::new(); num_ordered_tokens + 1];
    let mut event_positions = Vec::new();
    let mut labels_by_id = Vec::<StateTerminalLabel>::new();
    let mut label_ids = FxHashMap::<StateTerminalLabel, u32>::default();

    for class_id in used_state_class_ids(state_classes) {
        let Some(class_map) = class_maps.get(class_id as usize) else { continue; };
        for (&terminal_id, ranges) in class_map.iter() {
            let label_id = intern_state_terminal_label(&mut label_ids, &mut labels_by_id, (class_id, terminal_id));
            for &(lo, mut hi) in ranges.iter() {
                if num_ordered_tokens == 0 { continue; }
                let max_token = num_ordered_tokens as u32 - 1;
                if lo > max_token { continue; }
                hi = hi.min(max_token);
                if lo > hi { continue; }
                push_legacy_sweep_event(&mut events, &mut event_positions, lo, LegacySweepEvent { add: true, label_id });
                let after = hi.saturating_add(1);
                if after <= num_ordered_tokens as u32 {
                    push_legacy_sweep_event(&mut events, &mut event_positions, after, LegacySweepEvent { add: false, label_id });
                }
            }
        }
    }

    event_positions.sort_unstable();
    event_positions.dedup();
    (events, event_positions, labels_by_id)
}

fn apply_legacy_sweep_events(
    active_counts: &mut [u32],
    events: &[LegacySweepEvent],
    active_label_count: &mut usize,
) {
    for event in events.iter().filter(|event| !event.add) {
        let count = &mut active_counts[event.label_id as usize];
        assert!(*count > 0, "legacy pmv sweep removal underflow for label_id={}", event.label_id);
        if *count == 1 {
            *active_label_count -= 1;
        }
        *count -= 1;
    }
    for event in events.iter().filter(|event| event.add) {
        let count = &mut active_counts[event.label_id as usize];
        if *count == 0 {
            *active_label_count += 1;
        }
        *count += 1;
    }
}

fn build_legacy_possible_match_vocab_and_weights_from_interval_maps(
    class_maps: &[Arc<IntervalPossibleMatchMap>],
    state_classes: &[u32],
    ordered_vocab: &OrderedVocab,
) -> (PossibleMatchVocabMap, RuntimePossibleMatchesByTerminal) {
    let expanded_class_maps = expand_interval_class_maps(class_maps);
    let num_ordered_tokens = ordered_vocab.ordered_to_originals.len();
    let (events, event_positions, labels_by_id) =
        build_legacy_sweep_events(&expanded_class_maps, state_classes, num_ordered_tokens);

    let mut signature_to_id: FxHashMap<Vec<StateTerminalLabel>, SignatureClassId> = FxHashMap::default();
    let mut signature_labels: Vec<Vec<StateTerminalLabel>> = Vec::new();
    let mut original_to_internal = vec![u32::MAX; ordered_vocab.original_slot_count];
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut active_counts = vec![0u32; labels_by_id.len()];
    let mut active_label_count = 0usize;

    let mut event_index = 0usize;
    let mut position = 0usize;
    while position < num_ordered_tokens {
        while event_index < event_positions.len() && event_positions[event_index] as usize == position {
            apply_legacy_sweep_events(&mut active_counts, &events[position], &mut active_label_count);
            event_index += 1;
        }

        let next_position = event_positions.get(event_index).map(|&next| (next as usize).min(num_ordered_tokens)).unwrap_or(num_ordered_tokens);
        let mut signature = Vec::with_capacity(active_label_count);
        for (label_id, &label) in labels_by_id.iter().enumerate() {
            if active_counts[label_id] > 0 {
                signature.push(label);
            }
        }
        signature.sort_unstable();

        let signature_id = if let Some(&existing) = signature_to_id.get(&signature) { existing } else {
            let new_id = signature_labels.len() as SignatureClassId;
            signature_to_id.insert(signature.clone(), new_id);
            signature_labels.push(signature);
            internal_to_originals.push(Vec::new());
            new_id
        };

        for ordered_id in position..next_position {
            for &original in &ordered_vocab.ordered_to_originals[ordered_id] {
                if let Some(slot) = original_to_internal.get_mut(original as usize) { *slot = signature_id; }
                internal_to_originals[signature_id as usize].push(original);
            }
        }
        position = next_position;
    }

    for originals in &mut internal_to_originals { originals.sort_unstable(); originals.dedup(); }

    let mut ids_by_label: BTreeMap<TerminalID, BTreeMap<u32, Vec<u32>>> = BTreeMap::new();
    for (signature_id, labels) in signature_labels.iter().enumerate() {
        let signature_id = signature_id as u32;
        for &(class_id, terminal_id) in labels {
            ids_by_label.entry(terminal_id).or_default().entry(class_id).or_default().push(signature_id);
        }
    }

    let possible_matches = ids_by_label.into_iter().map(|(terminal_id, by_state)| {
        let mut entries = Vec::new();
        for (state, mut ids) in by_state {
            ids.sort_unstable();
            ids.dedup();
            let token_set = range_set_from_sorted_ids(&ids);
            if !token_set.is_empty() {
                entries.push((state, shared_rangeset(token_set)));
            }
        }
        (terminal_id, Weight::from_per_tsid_shared(entries.into_iter()))
    }).filter(|(_, weight)| !weight.is_empty()).collect();

    (PossibleMatchVocabMap { original_to_internal, internal_to_originals }, possible_matches)
}

fn validate_group_pmv_outputs(
    class_maps: &[Arc<IntervalPossibleMatchMap>],
    state_classes: &[u32],
    ordered_vocab: &OrderedVocab,
    actual_vocab: &PossibleMatchVocabMap,
    actual_matches: &RuntimePossibleMatchesByTerminal,
) {
    let started_at = Instant::now();
    let (expected_vocab, expected_matches) =
        build_legacy_possible_match_vocab_and_weights_from_interval_maps(class_maps, state_classes, ordered_vocab);

    if actual_vocab.original_to_internal != expected_vocab.original_to_internal {
        let mut mismatch = None;
        for idx in 0..actual_vocab.original_to_internal.len().min(expected_vocab.original_to_internal.len()) {
            let actual = actual_vocab.original_to_internal[idx];
            let expected = expected_vocab.original_to_internal[idx];
            if actual != expected {
                mismatch = Some((idx, actual, expected));
                break;
            }
        }
        panic!("group PMV validation failed: original_to_internal mismatch at {:?}", mismatch);
    }
    if actual_vocab.internal_to_originals != expected_vocab.internal_to_originals {
        let mut mismatch = None;
        for idx in 0..actual_vocab.internal_to_originals.len().min(expected_vocab.internal_to_originals.len()) {
            let actual = &actual_vocab.internal_to_originals[idx];
            let expected = &expected_vocab.internal_to_originals[idx];
            if actual != expected {
                mismatch = Some((idx, actual.clone(), expected.clone()));
                break;
            }
        }
        panic!("group PMV validation failed: internal_to_originals mismatch at {:?}; actual_len={} expected_len={}", mismatch, actual_vocab.internal_to_originals.len(), expected_vocab.internal_to_originals.len());
    }
    if actual_matches != &expected_matches {
        let mut terminal_ids: Vec<TerminalID> = actual_matches.keys().chain(expected_matches.keys()).copied().collect();
        terminal_ids.sort_unstable();
        terminal_ids.dedup();
        let mismatch = terminal_ids.into_iter().find(|terminal_id| actual_matches.get(terminal_id) != expected_matches.get(terminal_id));
        panic!("group PMV validation failed: possible match weight mismatch for terminal {:?}", mismatch);
    }

    if std::env::var_os("GLRMASK_PROFILE_PMV_DETAIL").is_some() {
        eprintln!("[glrmask/profile][pmv_validate] legacy_expand_compare_ms={:.3}", elapsed_ms(started_at));
    }
}

fn group_pmv_validation_enabled() -> bool {
    std::env::var("GLRMASK_VALIDATE_GROUP_PMV")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn group_pmv_legacy_enabled() -> bool {
    std::env::var("GLRMASK_PM_USE_LEGACY_PMV")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

const PM_NFA_POWERSET_DEFAULT_MAX_STATES: usize = 12_000;
const PM_NFA_POWERSET_NARROW_MAX_STATES: usize = 65_536;
const PM_NFA_POWERSET_NARROW_MAX_TERMINALS: usize = 256;

fn nfa_powerset_collect_default(state_count: usize, root_terminal_union: usize) -> bool {
    state_count <= PM_NFA_POWERSET_DEFAULT_MAX_STATES
        || (state_count <= PM_NFA_POWERSET_NARROW_MAX_STATES
            && root_terminal_union <= PM_NFA_POWERSET_NARROW_MAX_TERMINALS)
}

fn nfa_powerset_collect_enabled(state_count: usize, root_terminal_union: usize) -> bool {
    std::env::var("GLRMASK_PM_NFA_POWERSET_COLLECT")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or_else(|_| nfa_powerset_collect_default(state_count, root_terminal_union))
}

struct PossibleMatchPowersetView {
    num_states: usize,
    matched_terminals: Vec<Box<[TerminalID]>>,
    byte_transitions: Vec<Vec<u32>>,
    self_loop_bytes: Vec<U8Set>,
    raw_start_to_view: Vec<u32>,
    boundary_state: Vec<u32>,
    is_end: Vec<bool>,
}

struct DelayedTerminalDemand {
    terminals: BitSet,
    raw_state_relevant: Vec<bool>,
    raw_query_state: Vec<bool>,
    accepting_future_states: usize,
}

fn terminal_iter_intersects_demand(
    mut terminals: impl Iterator<Item = u32>,
    demand: &BitSet,
) -> bool {
    terminals.any(|terminal| demand.contains(terminal as usize))
}

/// Exact domain of static possible-match queries.
///
/// A terminal is inserted into `TerminalsDisallowed` only after it has matched
/// and the current tokenizer continuation still has that same terminal in its
/// strict future. Therefore a terminal can be queried by static masking iff at
/// least one tokenizer state contains it in both `finalizers` and
/// `possible_future_terminals`. Accumulator remapping retains an exclusion only
/// at states where that terminal remains in the strict future, so those states
/// are the complete query-state domain.
fn delayed_terminal_demand(tokenizer: &Tokenizer) -> DelayedTerminalDemand {
    let num_terminals = tokenizer.num_terminals() as usize;
    let mut terminals = BitSet::new(num_terminals);
    let mut accepting_future_states = 0usize;
    for state in 0..tokenizer.num_states() {
        let future = tokenizer.possible_future_terminals(state);
        let matched = tokenizer.matched_terminal_bitset(state);
        let state_relevant = terminals.union_intersection_with(matched, future);
        accepting_future_states += usize::from(state_relevant);
    }

    let mut raw_state_relevant = vec![false; tokenizer.num_states() as usize];
    let mut raw_query_state = vec![false; tokenizer.num_states() as usize];
    if !terminals.is_zero() {
        for state in 0..tokenizer.num_states() {
            let future_relevant = !tokenizer
                .possible_future_terminals(state)
                .is_disjoint(&terminals);
            let match_relevant = !tokenizer
                .matched_terminal_bitset(state)
                .is_disjoint(&terminals);
            raw_query_state[state as usize] = future_relevant;
            raw_state_relevant[state as usize] = future_relevant || match_relevant;
        }
    }
    DelayedTerminalDemand {
        terminals,
        raw_state_relevant,
        raw_query_state,
        accepting_future_states,
    }
}

fn intern_possible_match_config(
    mut config: Vec<u32>,
    is_closed: bool,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    configs: &mut Vec<Box<[u32]>>,
    config_is_closed: &mut Vec<bool>,
) -> Option<u32> {
    config.sort_unstable();
    config.dedup();
    if config.is_empty() {
        return None;
    }
    if let Some(&id) = config_ids.get(&config) {
        config_is_closed[id as usize] |= is_closed;
        return Some(id);
    }
    let id = configs.len() as u32;
    config_ids.insert(config.clone(), id);
    configs.push(config.into_boxed_slice());
    config_is_closed.push(is_closed);
    Some(id)
}

fn intern_possible_match_canonical_config(
    config: &[u32],
    is_closed: bool,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    configs: &mut Vec<Box<[u32]>>,
    config_is_closed: &mut Vec<bool>,
) -> Option<u32> {
    if config.is_empty() {
        return None;
    }
    debug_assert!(config.windows(2).all(|pair| pair[0] < pair[1]));
    if let Some(&id) = config_ids.get(config) {
        config_is_closed[id as usize] |= is_closed;
        return Some(id);
    }
    let id = configs.len() as u32;
    let owned = config.to_vec();
    config_ids.insert(owned.clone(), id);
    configs.push(owned.into_boxed_slice());
    config_is_closed.push(is_closed);
    Some(id)
}

fn build_possible_match_powerset_view(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    raw_byte_to_class: Option<&[u8; 256]>,
    demand: &DelayedTerminalDemand,
) -> PossibleMatchPowersetView {
    let singleton_closures = tokenizer.all_singleton_epsilon_closures();
    let mut config_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut configs = Vec::<Box<[u32]>>::new();
    let mut config_is_closed = Vec::<bool>::new();
    let raw_start_to_view = (0..tokenizer.num_states())
        .map(|raw_state| {
            if !demand.raw_query_state[raw_state as usize] {
                return u32::MAX;
            }
            let relevant_closure = singleton_closures[raw_state as usize]
                .iter()
                .copied()
                .filter(|&state| demand.raw_state_relevant[state as usize])
                .collect::<Vec<_>>();
            intern_possible_match_config(
                relevant_closure,
                true,
                &mut config_ids,
                &mut configs,
                &mut config_is_closed,
            )
            .expect("a PM query-state closure must retain an active terminal residual")
        })
        .collect::<Vec<_>>();

    let active_byte_classes = if let Some(raw_byte_to_class) = raw_byte_to_class {
        let mut members = vec![Vec::<u8>::new(); 256];
        for (byte, &active) in relevant_bytes.iter().enumerate() {
            if active {
                members[raw_byte_to_class[byte] as usize].push(byte as u8);
            }
        }
        members
            .into_iter()
            .filter(|members| !members.is_empty())
            .map(|members| (members[0], members))
            .collect::<Vec<_>>()
    } else {
        relevant_bytes
            .iter()
            .enumerate()
            .filter_map(|(byte, &active)| active.then_some((byte as u8, vec![byte as u8])))
            .collect::<Vec<_>>()
    };
    let mut matched_terminals = Vec::<Box<[TerminalID]>>::new();
    let mut byte_transitions = (0..256).map(|_| Vec::<u32>::new()).collect::<Vec<_>>();
    let mut self_loop_bytes = Vec::<U8Set>::new();
    let mut boundary_state = Vec::<u32>::new();
    let mut is_end = Vec::<bool>::new();
    let mut target_marks = vec![0u32; tokenizer.num_states() as usize];
    let mut target_generation = 0u32;
    let mut target_config = Vec::<u32>::new();
    let mut config_index = 0usize;
    while config_index < configs.len() {
        let config = configs[config_index].to_vec();
        let mut finalizers = config
            .iter()
            .flat_map(|&raw_state| tokenizer.matched_terminals_iter(raw_state))
            .filter(|&terminal| demand.terminals.contains(terminal as usize))
            .map(|terminal| terminal as TerminalID)
            .collect::<Vec<_>>();
        finalizers.sort_unstable();
        finalizers.dedup();
        matched_terminals.push(finalizers.into_boxed_slice());

        let live_config = config
            .iter()
            .copied()
            .filter(|&raw_state| !tokenizer.is_end(raw_state))
            .collect::<Vec<_>>();
        is_end.push(live_config.is_empty());
        boundary_state.push(
            intern_possible_match_canonical_config(
                &live_config,
                false,
                &mut config_ids,
                &mut configs,
                &mut config_is_closed,
            )
            .unwrap_or(u32::MAX),
        );

        let mut row = Box::new([u32::MAX; 256]);
        let source_is_closed = config_is_closed[config_index];
        for &(byte, ref class_members) in &active_byte_classes {
            target_generation = target_generation.wrapping_add(1);
            if target_generation == 0 {
                target_marks.fill(0);
                target_generation = 1;
            }
            target_config.clear();
            // Ordinary powerset configurations are already epsilon-closed, so
            // re-expanding every member's source closure is redundant. A
            // boundary projection may have accepting/end states removed and
            // is not necessarily closed; those configurations retain the
            // exact source-closure walk.
            for &source in &config {
                let sources = if source_is_closed {
                    std::slice::from_ref(&source)
                } else {
                    singleton_closures[source as usize].as_ref()
                };
                for &closed_source in sources {
                    let target = tokenizer.get_transition(closed_source, byte);
                    if target == u32::MAX {
                        continue;
                    }
                    for &reachable in singleton_closures[target as usize].iter() {
                        if !demand.raw_state_relevant[reachable as usize] {
                            continue;
                        }
                        let mark = &mut target_marks[reachable as usize];
                        if *mark != target_generation {
                            *mark = target_generation;
                            target_config.push(reachable);
                        }
                    }
                }
            }
            target_config.sort_unstable();
            if let Some(target) = intern_possible_match_canonical_config(
                &target_config,
                true,
                &mut config_ids,
                &mut configs,
                &mut config_is_closed,
            ) {
                for &class_byte in class_members {
                    row[class_byte as usize] = target;
                }
            }
        }
        let mut self_loops = U8Set::empty();
        for (byte, &target) in row.iter().enumerate() {
            byte_transitions[byte].push(target);
            if target == config_index as u32 {
                self_loops.insert(byte as u8);
            }
        }
        self_loop_bytes.push(self_loops);
        config_index += 1;
    }

    let num_states = configs.len();
    debug_assert_eq!(matched_terminals.len(), num_states);
    debug_assert!(byte_transitions.iter().all(|column| column.len() == num_states));
    debug_assert_eq!(self_loop_bytes.len(), num_states);
    debug_assert_eq!(config_is_closed.len(), configs.len());
    debug_assert_eq!(boundary_state.len(), num_states);
    debug_assert_eq!(is_end.len(), num_states);

    PossibleMatchPowersetView {
        num_states,
        matched_terminals,
        byte_transitions,
        self_loop_bytes,
        raw_start_to_view,
        boundary_state,
        is_end,
    }
}

fn sparse_root_collect_enabled() -> bool {
    std::env::var("GLRMASK_PM_SPARSE_ROOT_COLLECT")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

fn sparse_root_state_limit() -> usize {
    std::env::var("GLRMASK_PM_SPARSE_ROOT_MAX_STATES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128)
}

fn sparse_root_terminal_limit() -> usize {
    std::env::var("GLRMASK_PM_SPARSE_ROOT_MAX_TERMINALS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16)
}

fn batched_demand_collect_enabled(structured_dispatch: bool) -> bool {
    std::env::var("GLRMASK_PM_BATCHED_DEMAND_COLLECT")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(structured_dispatch)
}

fn root_terminal_union_count(tokenizer: &Tokenizer, states: &[u32]) -> usize {
    let mut seen = vec![false; tokenizer.num_terminals() as usize];
    let mut count = 0usize;
    for &state in states {
        for terminal in tokenizer
            .matched_terminals_iter(state)
            .chain(tokenizer.possible_future_terminals_iter(state))
        {
            let slot = terminal as usize;
            if slot < seen.len() && !seen[slot] {
                seen[slot] = true;
                count += 1;
            }
        }
    }
    count
}

fn interval_map_from_sparse_matches(
    matches: &FxHashMap<TerminalID, RangeSetBlaze<u32>>,
) -> IntervalPossibleMatchMap {
    let mut by_ranges = BTreeMap::<Vec<(u32, u32)>, Vec<TerminalID>>::new();
    for (&terminal, token_ids) in matches {
        let ranges: Vec<(u32, u32)> = token_ids
            .ranges()
            .map(|range| (*range.start(), *range.end()))
            .collect();
        if !ranges.is_empty() {
            by_ranges.entry(ranges).or_default().push(terminal);
        }
    }

    let mut map = Vec::with_capacity(by_ranges.len());
    for (ranges, mut terminals) in by_ranges {
        terminals.sort_unstable();
        terminals.dedup();
        if !terminals.is_empty() {
            map.push(TerminalRangeGroup {
                terminals: terminals.into_boxed_slice(),
                ranges,
            });
        }
    }
    map.sort_unstable_by(|left, right| {
        left.terminals
            .as_ref()
            .cmp(right.terminals.as_ref())
            .then_with(|| left.ranges.cmp(&right.ranges))
    });
    map
}

fn filter_interval_map_to_terminals(
    map: &IntervalPossibleMatchMap,
    terminals: &BitSet,
) -> IntervalPossibleMatchMap {
    let mut by_ranges = BTreeMap::<Vec<(u32, u32)>, Vec<TerminalID>>::new();
    for group in map {
        let retained = group
            .terminals
            .iter()
            .copied()
            .filter(|&terminal| terminals.contains(terminal as usize));
        by_ranges
            .entry(group.ranges.clone())
            .or_default()
            .extend(retained);
    }
    let mut filtered = by_ranges
        .into_iter()
        .filter_map(|(ranges, mut terminals)| {
            terminals.sort_unstable();
            terminals.dedup();
            (!terminals.is_empty()).then_some(TerminalRangeGroup {
                terminals: terminals.into_boxed_slice(),
                ranges,
            })
        })
        .collect::<Vec<_>>();
    filtered.sort_unstable_by(|left, right| {
        left.terminals
            .as_ref()
            .cmp(right.terminals.as_ref())
            .then_with(|| left.ranges.cmp(&right.ranges))
    });
    filtered
}

fn filter_trie_class_result_to_terminals(
    result: TrieClassBuildResult,
    terminals: &BitSet,
) -> TrieClassBuildResult {
    let mut filtered_maps = Vec::<Arc<IntervalPossibleMatchMap>>::new();
    let mut map_to_class = FxHashMap::<IntervalPossibleMatchMap, u32>::default();
    let mut old_to_new = vec![u32::MAX; result.class_maps.len()];
    for (old_class, map) in result.class_maps.iter().enumerate() {
        let filtered = filter_interval_map_to_terminals(map.as_ref(), terminals);
        let next_class = filtered_maps.len() as u32;
        let class = *map_to_class.entry(filtered.clone()).or_insert_with(|| {
            filtered_maps.push(Arc::new(filtered));
            next_class
        });
        old_to_new[old_class] = class;
    }
    let state_classes = result
        .state_classes
        .into_iter()
        .map(|class| {
            if class == u32::MAX {
                u32::MAX
            } else {
                old_to_new[class as usize]
            }
        })
        .collect();
    TrieClassBuildResult {
        state_classes,
        class_maps: filtered_maps,
    }
}

fn collect_sparse_root_possible_matches(
    tokenizer: &Tokenizer,
    root: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
    entries: &[u32],
    canonical_state: Option<&[u32]>,
) -> TrieClassBuildResult {
    let mut computer = PossibleMatchesComputer::new_with_canonical_state(tokenizer, canonical_state);
    let mut state_classes = vec![u32::MAX; tokenizer.num_states() as usize];
    let mut class_maps = Vec::<Arc<IntervalPossibleMatchMap>>::new();
    let mut map_to_class = FxHashMap::<IntervalPossibleMatchMap, u32>::default();

    for &state in entries {
        let sparse_matches = computer.possible_matches_for_node(root, state);
        let interval_map = interval_map_from_sparse_matches(sparse_matches.as_ref());
        let class_id = if let Some(&class_id) = map_to_class.get(&interval_map) {
            class_id
        } else {
            let class_id = class_maps.len() as u32;
            map_to_class.insert(interval_map.clone(), class_id);
            class_maps.push(Arc::new(interval_map));
            class_id
        };

        if let Some(slot) = state_classes.get_mut(state as usize) {
            *slot = class_id;
        }
    }

    TrieClassBuildResult {
        state_classes,
        class_maps,
    }
}

struct BatchedDemandView {
    raw_state_to_index: Vec<u32>,
    byte_transitions: Vec<Vec<u32>>,
    matched_masks: Vec<u128>,
    /// Conservative demanded-terminal reachability over byte transitions and
    /// optional token-boundary projection edges. `None` keeps the reference
    /// traversal unchanged.
    future_matched_masks: Option<Vec<u128>>,
    is_end: Vec<bool>,
    self_loop_bytes: Vec<U8Set>,
    boundary_state: Option<Vec<u32>>,
}

struct BatchedDemandGroup {
    state_index: u32,
    remaining: u128,
    origins: SmallVec<[u32; 4]>,
}

struct BatchedGroupIndex {
    mask_stride: usize,
    generation: u32,
    stamps: Vec<u32>,
    positions: Vec<u32>,
    fallback: FxHashMap<(u32, u128), usize>,
}

impl BatchedGroupIndex {
    fn new(state_count: usize, terminal_count: usize) -> Self {
        let mask_stride = if terminal_count <= 8 {
            1usize << terminal_count
        } else {
            0
        };
        let slots = state_count.saturating_mul(mask_stride);
        Self {
            mask_stride,
            generation: 0,
            stamps: vec![0; slots],
            positions: vec![0; slots],
            fallback: FxHashMap::default(),
        }
    }

    fn begin_edge(&mut self) {
        if self.mask_stride == 0 {
            self.fallback.clear();
            return;
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.stamps.fill(0);
            self.generation = 1;
        }
    }

    fn get(&self, state_index: u32, remaining: u128) -> Option<usize> {
        if self.mask_stride == 0 {
            return self.fallback.get(&(state_index, remaining)).copied();
        }
        let slot = state_index as usize * self.mask_stride + remaining as usize;
        (self.stamps[slot] == self.generation).then_some(self.positions[slot] as usize)
    }

    fn insert(&mut self, state_index: u32, remaining: u128, position: usize) {
        if self.mask_stride == 0 {
            self.fallback.insert((state_index, remaining), position);
            return;
        }
        let slot = state_index as usize * self.mask_stride + remaining as usize;
        self.stamps[slot] = self.generation;
        self.positions[slot] = position as u32;
    }
}

#[derive(Default)]
struct BatchedDemandProfile {
    nodes: usize,
    child_edges: usize,
    input_groups: usize,
    output_groups: usize,
    group_merges: usize,
    transition_steps: usize,
    matched_events: usize,
    origin_updates: usize,
    range_inserts: usize,
    future_prunes: usize,
    future_pruned_origins: usize,
    append_ms: f64,
}

fn append_batched_demand_matches(
    outputs: &mut [Vec<(u32, u32)>],
    terminal_count: usize,
    origins: &[u32],
    matched: u128,
    child: &VocabPrefixTreeNode,
    profile: &mut BatchedDemandProfile,
) {
    if matched == 0 {
        return;
    }
    let started_at = Instant::now();
    profile.matched_events += matched.count_ones() as usize;
    for bit in 0..terminal_count {
        if matched & (1u128 << bit) == 0 {
            continue;
        }
        for &origin in origins {
            profile.origin_updates += 1;
            for range in child.reachable_token_ids().ranges() {
                profile.range_inserts += 1;
                outputs[origin as usize * terminal_count + bit]
                    .push((*range.start() as u32, *range.end() as u32));
            }
        }
    }
    profile.append_ms += elapsed_ms(started_at);
}

fn batched_future_matched_masks(
    matched_masks: &[u128],
    byte_transitions: &[Vec<u32>],
    boundary_state: Option<&[u32]>,
) -> Vec<u128> {
    let num_states = matched_masks.len();
    let mut predecessors = vec![Vec::<u32>::new(); num_states];
    for column in byte_transitions {
        debug_assert_eq!(column.len(), num_states);
        for (source, &target) in column.iter().enumerate() {
            if target != u32::MAX {
                predecessors[target as usize].push(source as u32);
            }
        }
    }
    if let Some(boundary_state) = boundary_state {
        debug_assert_eq!(boundary_state.len(), num_states);
        for (source, &target) in boundary_state.iter().enumerate() {
            if target != u32::MAX {
                predecessors[target as usize].push(source as u32);
            }
        }
    }
    for incoming in &mut predecessors {
        incoming.sort_unstable();
        incoming.dedup();
    }

    let mut future = matched_masks.to_vec();
    let mut queue = std::collections::VecDeque::<u32>::new();
    let mut queued = vec![false; num_states];
    for state in 0..num_states {
        if future[state] != 0 {
            queue.push_back(state as u32);
            queued[state] = true;
        }
    }
    while let Some(target) = queue.pop_front() {
        queued[target as usize] = false;
        let target_mask = future[target as usize];
        for &source in &predecessors[target as usize] {
            let slot = &mut future[source as usize];
            let merged = *slot | target_mask;
            if merged != *slot {
                *slot = merged;
                if !queued[source as usize] {
                    queued[source as usize] = true;
                    queue.push_back(source);
                }
            }
        }
    }
    future
}

#[inline]
fn batched_group_has_future_demand(view: &BatchedDemandView, state: u32, remaining: u128) -> bool {
    view.future_matched_masks
        .as_ref()
        .is_none_or(|future| future[state as usize] & remaining != 0)
}

fn collect_batched_demand_node(
    node: &VocabPrefixTreeNode,
    groups: Vec<BatchedDemandGroup>,
    view: &BatchedDemandView,
    demanded_terminals: &[TerminalID],
    outputs: &mut [Vec<(u32, u32)>],
    profile: &mut BatchedDemandProfile,
    group_index: &mut BatchedGroupIndex,
) {
    profile.nodes += 1;
    profile.input_groups += groups.len();
    for (segment, child) in node.iter_children() {
        profile.child_edges += 1;
        let mut next_groups = Vec::<BatchedDemandGroup>::new();
        group_index.begin_edge();
        for group in &groups {
            if !batched_group_has_future_demand(view, group.state_index, group.remaining) {
                profile.future_prunes += 1;
                profile.future_pruned_origins += group.origins.len();
                continue;
            }
            let mut state_index = group.state_index;
            let mut remaining = group.remaining;
            let mut matched = 0u128;
            let mut live = true;
            for &byte in segment {
                profile.transition_steps += 1;
                let target_index = view.byte_transitions[byte as usize][state_index as usize];
                if target_index == u32::MAX {
                    live = false;
                    break;
                }
                state_index = target_index;
                let newly_matched = view.matched_masks[target_index as usize] & remaining;
                matched |= newly_matched;
                remaining &= !newly_matched;
                if remaining == 0 {
                    break;
                }
                if !batched_group_has_future_demand(view, state_index, remaining) {
                    profile.future_prunes += 1;
                    profile.future_pruned_origins += group.origins.len();
                    live = false;
                    break;
                }
            }

            append_batched_demand_matches(
                outputs,
                demanded_terminals.len(),
                &group.origins,
                matched,
                child,
                profile,
            );
            if !live || remaining == 0 {
                continue;
            }
            if view.is_end[state_index as usize]
                || U8Set::from_words(*child.subtree_bytes())
                    .is_subset(&view.self_loop_bytes[state_index as usize])
            {
                continue;
            }
            if let Some(boundary_state) = view.boundary_state.as_ref() {
                state_index = boundary_state[state_index as usize];
                if state_index == u32::MAX {
                    continue;
                }
            }
            if !batched_group_has_future_demand(view, state_index, remaining) {
                profile.future_prunes += 1;
                profile.future_pruned_origins += group.origins.len();
                continue;
            }
            if let Some(existing_group_index) = group_index.get(state_index, remaining) {
                profile.group_merges += 1;
                next_groups[existing_group_index]
                    .origins
                    .extend_from_slice(&group.origins);
            } else {
                let next_group_index = next_groups.len();
                group_index.insert(state_index, remaining, next_group_index);
                next_groups.push(BatchedDemandGroup {
                    state_index,
                    remaining,
                    origins: group.origins.clone(),
                });
            }
        }
        profile.output_groups += next_groups.len();
        if !next_groups.is_empty() {
            collect_batched_demand_node(
                child,
                next_groups,
                view,
                demanded_terminals,
                outputs,
                profile,
                group_index,
            );
        }
    }
}

fn interval_map_from_batched_ranges(
    ranges_by_terminal: &mut [Vec<(u32, u32)>],
    demanded_terminals: &[TerminalID],
) -> IntervalPossibleMatchMap {
    let mut by_ranges = BTreeMap::<Vec<(u32, u32)>, Vec<TerminalID>>::new();
    for (&terminal, ranges) in demanded_terminals.iter().zip(ranges_by_terminal) {
        normalize_token_ranges(ranges);
        if !ranges.is_empty() {
            by_ranges.entry(std::mem::take(ranges)).or_default().push(terminal);
        }
    }
    let mut map = by_ranges
        .into_iter()
        .map(|(ranges, mut terminals)| {
            terminals.sort_unstable();
            terminals.dedup();
            TerminalRangeGroup {
                terminals: terminals.into_boxed_slice(),
                ranges,
            }
        })
        .collect::<Vec<_>>();
    map.sort_unstable_by(|left, right| {
        left.terminals
            .as_ref()
            .cmp(right.terminals.as_ref())
            .then_with(|| left.ranges.cmp(&right.ranges))
    });
    map
}

fn collect_batched_demand_possible_matches(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    entries: &[u32],
    demand: &DelayedTerminalDemand,
) -> TrieClassBuildResult {
    let demanded_terminals = demand
        .terminals
        .iter()
        .map(|terminal| terminal as TerminalID)
        .collect::<Vec<_>>();
    assert!(!demanded_terminals.is_empty() && demanded_terminals.len() <= 128);
    let full_mask = if demanded_terminals.len() == 128 {
        u128::MAX
    } else {
        (1u128 << demanded_terminals.len()) - 1
    };
    let mut terminal_to_bit = vec![0u128; tokenizer.num_terminals() as usize];
    for (bit, &terminal) in demanded_terminals.iter().enumerate() {
        terminal_to_bit[terminal as usize] = 1u128 << bit;
    }

    let relevant_states = demand
        .raw_state_relevant
        .iter()
        .enumerate()
        .filter_map(|(state, &relevant)| relevant.then_some(state as u32))
        .collect::<Vec<_>>();
    let mut raw_state_to_index = vec![u32::MAX; tokenizer.num_states() as usize];
    for (index, &state) in relevant_states.iter().enumerate() {
        raw_state_to_index[state as usize] = index as u32;
    }
    let mut byte_transitions = vec![vec![u32::MAX; relevant_states.len()]; 256];
    let mut matched_masks = Vec::with_capacity(relevant_states.len());
    let mut is_end = Vec::with_capacity(relevant_states.len());
    for (state_index, &state) in relevant_states.iter().enumerate() {
        for (byte, target) in tokenizer.transitions_from(state) {
            let target_index = raw_state_to_index[target as usize];
            if target_index != u32::MAX {
                byte_transitions[byte as usize][state_index] = target_index;
            }
        }
        matched_masks.push(
            tokenizer
                .matched_terminals_iter(state)
                .fold(0u128, |mask, terminal| {
                    mask | terminal_to_bit[terminal as usize]
                }),
        );
        is_end.push(tokenizer.is_end(state));
    }
    let self_loop_bytes = relevant_states
        .iter()
        .map(|&state| tokenizer.self_loop_bytes(state))
        .collect();
    let view = BatchedDemandView {
        raw_state_to_index,
        byte_transitions,
        matched_masks,
        future_matched_masks: None,
        is_end,
        self_loop_bytes,
        boundary_state: None,
    };

    let mut outputs = (0..entries.len() * demanded_terminals.len())
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    if root.has_token() {
        let token_id = root.token_id() as u32;
        for (origin, &state) in entries.iter().enumerate() {
            let state_index = view.raw_state_to_index[state as usize] as usize;
            let matched = view.matched_masks[state_index];
            for bit in 0..demanded_terminals.len() {
                if matched & (1u128 << bit) != 0 {
                    outputs[origin * demanded_terminals.len() + bit].push((token_id, token_id));
                }
            }
        }
    }
    let groups = entries
        .iter()
        .enumerate()
        .map(|(origin, &state)| BatchedDemandGroup {
            state_index: view.raw_state_to_index[state as usize],
            remaining: full_mask,
            origins: SmallVec::from_slice(&[origin as u32]),
        })
        .collect::<Vec<_>>();
    let mut profile = BatchedDemandProfile::default();
    let mut group_index =
        BatchedGroupIndex::new(view.matched_masks.len(), demanded_terminals.len());
    let walk_started_at = Instant::now();
    collect_batched_demand_node(
        root,
        groups,
        &view,
        &demanded_terminals,
        &mut outputs,
        &mut profile,
        &mut group_index,
    );
    let walk_ms = elapsed_ms(walk_started_at);

    let class_started_at = Instant::now();
    let mut state_classes = vec![u32::MAX; tokenizer.num_states() as usize];
    let mut class_maps = Vec::<Arc<IntervalPossibleMatchMap>>::new();
    let mut map_to_class = FxHashMap::<IntervalPossibleMatchMap, u32>::default();
    for (origin, &state) in entries.iter().enumerate() {
        let start = origin * demanded_terminals.len();
        let end = start + demanded_terminals.len();
        let interval_map = interval_map_from_batched_ranges(
            &mut outputs[start..end],
            &demanded_terminals,
        );
        let class_id = if let Some(&class_id) = map_to_class.get(&interval_map) {
            class_id
        } else {
            let class_id = class_maps.len() as u32;
            map_to_class.insert(interval_map.clone(), class_id);
            class_maps.push(Arc::new(interval_map));
            class_id
        };
        state_classes[state as usize] = class_id;
    }
    let class_ms = elapsed_ms(class_started_at);
    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
    {
        eprintln!(
            "[glrmask/profile][batched_demand_detail] nodes={} child_edges={} input_groups={} output_groups={} group_merges={} transition_steps={} matched_events={} origin_updates={} range_inserts={} future_prunes={} future_pruned_origins={} append_ms={:.3} walk_ms={:.3} class_ms={:.3}",
            profile.nodes,
            profile.child_edges,
            profile.input_groups,
            profile.output_groups,
            profile.group_merges,
            profile.transition_steps,
            profile.matched_events,
            profile.origin_updates,
            profile.range_inserts,
            profile.future_prunes,
            profile.future_pruned_origins,
            profile.append_ms,
            walk_ms,
            class_ms,
        );
    }
    TrieClassBuildResult {
        state_classes,
        class_maps,
    }
}

/// Batched delayed-terminal collection over an already deterministic powerset
/// view. This is the same exact existential-prefix computation as
/// `collect_batched_demand_possible_matches`, but it avoids requiring the raw
/// epsilon-NFA tokenizer itself to be deterministic.
fn collect_batched_demand_possible_matches_precomputed(
    root: &VocabPrefixTreeNode,
    entries: &[u32],
    num_states: usize,
    num_terminals: usize,
    matched_terminals: &[Box<[TerminalID]>],
    is_end: &[bool],
    byte_transitions: &[Vec<u32>],
    self_loop_bytes: &[U8Set],
    boundary_state: &[u32],
    demand: &DelayedTerminalDemand,
) -> TrieClassBuildResult {
    let demanded_terminals = demand
        .terminals
        .iter()
        .map(|terminal| terminal as TerminalID)
        .collect::<Vec<_>>();
    assert!(!demanded_terminals.is_empty() && demanded_terminals.len() <= 128);
    let full_mask = if demanded_terminals.len() == 128 {
        u128::MAX
    } else {
        (1u128 << demanded_terminals.len()) - 1
    };
    let mut terminal_to_bit = vec![0u128; num_terminals];
    for (bit, &terminal) in demanded_terminals.iter().enumerate() {
        terminal_to_bit[terminal as usize] = 1u128 << bit;
    }
    let matched_masks = matched_terminals
        .iter()
        .map(|terminals| {
            terminals.iter().fold(0u128, |mask, &terminal| {
                mask | terminal_to_bit[terminal as usize]
            })
        })
        .collect::<Vec<_>>();
    let use_future_prune = std::env::var("GLRMASK_PM_POWERSET_FUTURE_DEMAND_PRUNE")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || value == "1" || value.eq_ignore_ascii_case("true")
        })
        .unwrap_or(true);
    let future_mask_started_at = Instant::now();
    let future_matched_masks = use_future_prune.then(|| {
        batched_future_matched_masks(&matched_masks, byte_transitions, Some(boundary_state))
    });
    let future_mask_ms = elapsed_ms(future_mask_started_at);
    let view = BatchedDemandView {
        raw_state_to_index: (0..num_states as u32).collect(),
        byte_transitions: byte_transitions.to_vec(),
        matched_masks,
        future_matched_masks,
        is_end: is_end.to_vec(),
        self_loop_bytes: self_loop_bytes.to_vec(),
        boundary_state: Some(boundary_state.to_vec()),
    };

    let mut outputs = (0..entries.len() * demanded_terminals.len())
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    if root.has_token() {
        let token_id = root.token_id() as u32;
        for (origin, &state) in entries.iter().enumerate() {
            let matched = view.matched_masks[state as usize];
            for bit in 0..demanded_terminals.len() {
                if matched & (1u128 << bit) != 0 {
                    outputs[origin * demanded_terminals.len() + bit].push((token_id, token_id));
                }
            }
        }
    }
    let groups = entries
        .iter()
        .enumerate()
        .map(|(origin, &state)| BatchedDemandGroup {
            state_index: state,
            remaining: full_mask,
            origins: SmallVec::from_slice(&[origin as u32]),
        })
        .collect::<Vec<_>>();
    let mut profile = BatchedDemandProfile::default();
    let mut group_index = BatchedGroupIndex::new(num_states, demanded_terminals.len());
    let walk_started_at = Instant::now();
    collect_batched_demand_node(
        root,
        groups,
        &view,
        &demanded_terminals,
        &mut outputs,
        &mut profile,
        &mut group_index,
    );
    let walk_ms = elapsed_ms(walk_started_at);

    let class_started_at = Instant::now();
    let mut state_classes = vec![u32::MAX; num_states];
    let mut class_maps = Vec::<Arc<IntervalPossibleMatchMap>>::new();
    let mut map_to_class = FxHashMap::<IntervalPossibleMatchMap, u32>::default();
    for (origin, &state) in entries.iter().enumerate() {
        let start = origin * demanded_terminals.len();
        let end = start + demanded_terminals.len();
        let interval_map = interval_map_from_batched_ranges(
            &mut outputs[start..end],
            &demanded_terminals,
        );
        let class_id = if let Some(&class_id) = map_to_class.get(&interval_map) {
            class_id
        } else {
            let class_id = class_maps.len() as u32;
            map_to_class.insert(interval_map.clone(), class_id);
            class_maps.push(Arc::new(interval_map));
            class_id
        };
        state_classes[state as usize] = class_id;
    }
    let class_ms = elapsed_ms(class_started_at);
    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
    {
        eprintln!(
            "[glrmask/profile][batched_demand_powerset_detail] nodes={} child_edges={} input_groups={} output_groups={} group_merges={} transition_steps={} matched_events={} origin_updates={} range_inserts={} future_prunes={} future_pruned_origins={} future_mask_ms={:.3} append_ms={:.3} walk_ms={:.3} class_ms={:.3}",
            profile.nodes,
            profile.child_edges,
            profile.input_groups,
            profile.output_groups,
            profile.group_merges,
            profile.transition_steps,
            profile.matched_events,
            profile.origin_updates,
            profile.range_inserts,
            profile.future_prunes,
            profile.future_pruned_origins,
            future_mask_ms,
            profile.append_ms,
            walk_ms,
            class_ms,
        );
    }
    TrieClassBuildResult {
        state_classes,
        class_maps,
    }
}

fn attach_structured_dispatch_possible_matches(
    tokenizer: &Tokenizer,
    result: &mut TrieClassBuildResult,
    demand: &DelayedTerminalDemand,
) {
    let Some(roots) = tokenizer.deterministic_dispatch_roots() else {
        return;
    };

    let start = tokenizer.start_state();
    let mut seen_classes = vec![false; result.class_maps.len()];
    let mut ranges_by_terminal = BTreeMap::<TerminalID, Vec<(u32, u32)>>::new();
    for &root in roots {
        let class = result.state_classes[root as usize];
        if class == u32::MAX || seen_classes[class as usize] {
            continue;
        }
        seen_classes[class as usize] = true;
        for group in result.class_maps[class as usize].iter() {
            for &terminal in group.terminals.iter() {
                ranges_by_terminal
                    .entry(terminal)
                    .or_default()
                    .extend_from_slice(&group.ranges);
            }
        }
    }
    let mut grouped_by_ranges = BTreeMap::<Vec<(u32, u32)>, Vec<TerminalID>>::new();
    for (terminal, mut ranges) in ranges_by_terminal {
        normalize_token_ranges(&mut ranges);
        if !ranges.is_empty() {
            grouped_by_ranges.entry(ranges).or_default().push(terminal);
        }
    }
    let mut dispatch_map = grouped_by_ranges
        .into_iter()
        .map(|(ranges, mut terminals)| {
            terminals.sort_unstable();
            terminals.dedup();
            TerminalRangeGroup {
                terminals: terminals.into_boxed_slice(),
                ranges,
            }
        })
        .collect::<IntervalPossibleMatchMap>();
    dispatch_map.sort_unstable_by(|left, right| {
        left.terminals
            .as_ref()
            .cmp(right.terminals.as_ref())
            .then_with(|| left.ranges.cmp(&right.ranges))
    });
    let class = result
        .class_maps
        .iter()
        .position(|existing| existing.as_ref() == &dispatch_map)
        .map(|class| class as u32)
        .unwrap_or_else(|| {
            let class = result.class_maps.len() as u32;
            result.class_maps.push(Arc::new(dispatch_map));
            class
        });
    result.state_classes[start as usize] = class;
    for &root in roots {
        if !demand.raw_query_state[root as usize] {
            result.state_classes[root as usize] = u32::MAX;
        }
    }
}

pub(crate) fn compute_constraint_possible_matches(
    tokenizer: &Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    config: ConstraintPossibleMatchesConfig,
) -> ConstraintPossibleMatchesComputation {
    let artifacts_and_profile = get_ordered_vocab_trie_artifacts(token_bytes);
    if config.defer_to_dynamic_mask {
        let (artifacts, profile) = artifacts_and_profile;
        emit_ordered_vocab_cache_profile(profile);
        let runtime_dynamic_vocab = runtime_dynamic_vocab_artifacts(&artifacts);
        return empty_possible_matches_computation(
            tokenizer,
            token_bytes.len(),
            runtime_dynamic_vocab,
        );
    }
    compute_constraint_possible_matches_with_artifacts(
        tokenizer,
        token_bytes.len(),
        artifacts_and_profile,
        None,
        None,
    )
}

fn prepared_runtime_dynamic_vocab(
    artifacts: &OrderedVocabTrieArtifacts,
) -> &Arc<DynamicMaskVocab> {
    artifacts.runtime_dynamic_vocab.get_or_init(|| {
        // The global ordered-vocabulary trie is the production runtime layout.
        // It avoids artificial root partitions that make the full vocabulary
        // walk revisit equivalent prefix structure. Keep the older partitioned
        // layout only as a diagnostic escape hatch for regression comparison.
        if std::env::var_os("GLRMASK_LEGACY_PARTITIONED_DYNAMIC_MASK_TRIE").is_none() {
            let runtime_trie = Arc::new(DynamicMaskTrie::from_vocab_prefix_tree(
                artifacts.trie.as_ref(),
            ));
            return Arc::new(DynamicMaskVocab::from_materialized_ordered(
                runtime_trie,
                Arc::clone(&artifacts.ordered_vocab.ordered_to_originals),
            ));
        }
        // Runtime trie layout refines the compiler's broad character-type
        // classes by the two properties that most often contaminate a large
        // otherwise-uniform continuation subtree: an optional leading ASCII
        // space and the presence of non-ASCII bytes. This is only physical
        // layout; token IDs and lexer/grammar semantics are untouched.
        let mut entries = artifacts
            .ordered_vocab
            .ordered_token_bytes
            .iter()
            .enumerate()
            .map(|(token_id, bytes)| {
                let layout_class =
                    dynamic_mask_vocab_layout_class(classify_vocab_char_type(bytes), bytes);
                (layout_class, token_id, bytes.as_slice())
            })
            .collect::<Vec<_>>();
        entries.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.2.cmp(right.2))
                .then_with(|| left.1.cmp(&right.1))
        });
        let runtime_trie = DynamicMaskTrie::from_partitioned_token_refs(&entries);
        let runtime_trie = Arc::new(runtime_trie);
        Arc::new(DynamicMaskVocab::from_materialized_ordered(
            runtime_trie,
            Arc::clone(&artifacts.ordered_vocab.ordered_to_originals),
        ))
    })
}

fn prepare_llg_slice_leftovers_for_ordered_vocab(
    ordered_vocab: &OrderedVocab,
    safe_token_bytes: U8Set,
    vocab: &mut DynamicMaskVocab,
) {
    if vocab.has_llg_slice_leftovers() {
        return;
    }

    // These are language definitions, not benchmark buckets. The runtime
    // derives exact bounded safe-string radii from the current residual
    // language; vocabulary structure is partitioned by exact Unicode-scalar
    // length below.
    const SAFE_PLUS_CACHE_ID: u32 = 0;
    const WHITESPACE_CACHE_ID: u32 = 3;
    const ASCII_WORD_CACHE_ID: u32 = 0x40;
    let safe_plus = Arc::new(
        VocabPartitionDfa::compile_utf8_regex(
            "llg-safe+",
            r#"[^"\\\x00-\x1F\x7F]+"#,
        )
        .expect("safe-string slice regex must compile"),
    );
    let whitespace = Arc::new(
        VocabPartitionDfa::compile_utf8_regex("llg-whitespace", r"[\x20\x0A\x0D\x09]+")
            .expect("whitespace slice regex must compile"),
    );
    let max_token_byte_len = ordered_vocab
        .ordered_token_bytes
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    // Default to the validated finite word language. This is vocabulary-only
    // preparation; every runtime use still needs an exact residual proof and
    // must beat the already-certified route by the conservative work margin.
    let ascii_slice_max_token_byte_len = std::env::var(
        "GLRMASK_EXPERIMENT_DIRECT_RESIDUAL_ASCII_WORD_SLICE_MAX_BYTES",
    )
    .ok()
    .and_then(|value| value.trim().parse::<usize>().ok())
    .filter(|&value| value != 0)
    .map_or(16.min(max_token_byte_len), |value| value.min(max_token_byte_len));
    let ascii_word = (max_token_byte_len != 0
        && std::env::var_os("GLRMASK_DISABLE_DIRECT_RESIDUAL_WORD_SLICE").is_none())
    .then(|| {
        Arc::new(
            VocabPartitionDfa::compile_utf8_regex(
                "dynamic-optional-space-nonspace-safe",
                &format!(
                    r#" ?[^\s"\\\x00-\x1F\x7F]{{1,{ascii_slice_max_token_byte_len}}}"#
                ),
            )
            .expect("bounded optional-space nonspace-safe slice regex must compile"),
        )
    });

    let word_len = vocab.all_original_token_words().len();
    let mut safe_words = vec![0u32; word_len];
    let mut whitespace_words = vec![0u32; word_len];
    let mut ascii_word_words = vec![0u32; word_len];
    let mut whitespace_token_bytes = U8Set::empty();
    let mut ascii_word_token_bytes = U8Set::empty();
    let mut safe_max_token_byte_len = 0u32;
    let mut whitespace_max_token_byte_len = 0u32;
    let mut ascii_word_max_token_byte_len = 0u32;
    let mut ascii_word_residual_entries =
        Vec::<(u16, usize, &[u8])>::with_capacity(ordered_vocab.ordered_token_bytes.len());
    let mut entries = Vec::<(u16, usize, &[u8])>::with_capacity(ordered_vocab.ordered_token_bytes.len());
    let mut max_safe_chars = 0u16;

    for (canonical, bytes) in ordered_vocab.ordered_token_bytes.iter().enumerate() {
        let bytes = bytes.as_slice();
        let is_safe = safe_plus.is_match(bytes);
        let safe_chars = if is_safe {
            let chars = std::str::from_utf8(bytes)
                .expect("safe-string UTF-8 regex matched invalid UTF-8")
                .chars()
                .count();
            u16::try_from(chars).expect("model token exceeds u16 Unicode-scalar count")
        } else {
            0
        };
        let is_whitespace = whitespace.is_match(bytes);
        let is_ascii_word = ascii_word.as_ref().is_some_and(|slice| slice.is_match(bytes));
        max_safe_chars = max_safe_chars.max(safe_chars);
        entries.push((
            dynamic_mask_llg_master_layout_class(safe_chars, is_whitespace),
            canonical,
            bytes,
        ));

        let Some(originals) = ordered_vocab.ordered_to_originals.get(canonical) else {
            continue;
        };
        if is_safe {
            safe_max_token_byte_len = safe_max_token_byte_len.max(bytes.len() as u32);
        }
        if is_whitespace {
            whitespace_max_token_byte_len = whitespace_max_token_byte_len.max(bytes.len() as u32);
            for &byte in bytes {
                whitespace_token_bytes.insert(byte);
            }
        }
        if is_ascii_word {
            ascii_word_max_token_byte_len = ascii_word_max_token_byte_len.max(bytes.len() as u32);
            for &byte in bytes {
                ascii_word_token_bytes.insert(byte);
            }
        } else {
            ascii_word_residual_entries.push((0, canonical, bytes));
        }
        for &token in originals {
            let word = token as usize / 32;
            if word >= word_len {
                continue;
            }
            let bit = 1u32 << (token % 32);
            if is_safe {
                safe_words[word] |= bit;
            }
            if is_whitespace {
                whitespace_words[word] |= bit;
            }
            if is_ascii_word {
                ascii_word_words[word] |= bit;
            }
        }
    }

    entries.sort_unstable_by(|left, right| {
        left.2
            .is_empty()
            .cmp(&right.2.is_empty())
            .reverse()
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.2.cmp(right.2))
            .then_with(|| left.1.cmp(&right.1))
    });
    let master_trie = DynamicMaskTrie::from_partitioned_token_refs(&entries);
    ascii_word_residual_entries.sort_unstable_by(|left, right| {
        left.2.cmp(right.2).then_with(|| left.1.cmp(&right.1))
    });
    let ascii_word_residual_trie =
        DynamicMaskTrie::from_partitioned_token_refs(&ascii_word_residual_entries);

    // Build cumulative exact-safe-length masks once per model vocabulary. A
    // runtime radius r then initializes all safe tokens of lengths <= r in one
    // dense copy; no fixed length thresholds are encoded here.
    let mut exact_safe_words = vec![vec![0u32; word_len]; usize::from(max_safe_chars) + 1];
    for &(class, canonical, _) in &entries {
        let safe_chars = dynamic_mask_llg_master_safe_chars(class);
        if safe_chars == 0 {
            continue;
        }
        let Some(originals) = ordered_vocab.ordered_to_originals.get(canonical) else {
            continue;
        };
        for &token in originals {
            let word = token as usize / 32;
            if word < word_len {
                exact_safe_words[usize::from(safe_chars)][word] |= 1u32 << (token % 32);
            }
        }
    }
    let mut admitted_words = Vec::<Vec<u32>>::with_capacity((usize::from(max_safe_chars) + 1) * 2);
    let mut safe_prefix = vec![0u32; word_len];
    for radius in 0..=usize::from(max_safe_chars) {
        if radius != 0 {
            for (target, &source) in safe_prefix.iter_mut().zip(&exact_safe_words[radius]) {
                *target |= source;
            }
        }
        admitted_words.push(safe_prefix.clone());
        let mut with_whitespace = safe_prefix.clone();
        for (target, &source) in with_whitespace.iter_mut().zip(&whitespace_words) {
            *target |= source;
        }
        admitted_words.push(with_whitespace);
    }
    vocab.set_llg_master_admitted_words(max_safe_chars, admitted_words);

    let mut slices = vec![
        (
            SAFE_PLUS_CACHE_ID,
            Arc::clone(&safe_plus),
            Arc::new(DynamicMaskTrie::new()),
            Arc::new(safe_words),
            safe_token_bytes,
            safe_max_token_byte_len,
        ),
        (
            WHITESPACE_CACHE_ID,
            Arc::clone(&whitespace),
            Arc::new(DynamicMaskTrie::new()),
            Arc::new(whitespace_words),
            whitespace_token_bytes,
            whitespace_max_token_byte_len,
        ),
        (
            DYNAMIC_MASK_LLG_MASTER_CACHE_ID,
            safe_plus,
            Arc::new(master_trie),
            Arc::new(Vec::new()),
            U8Set::empty(),
            0,
        ),
    ];
    if let Some(ascii_word) = ascii_word {
        slices.push((
            ASCII_WORD_CACHE_ID,
            ascii_word,
            Arc::new(ascii_word_residual_trie),
            Arc::new(ascii_word_words),
            ascii_word_token_bytes,
            ascii_word_max_token_byte_len,
        ));
    }
    vocab.set_llg_slice_leftovers(slices);
}

fn llg_safe_slice_token_bytes_for_ordered_vocab(ordered_vocab: &OrderedVocab) -> U8Set {
    let safe_plus = VocabPartitionDfa::compile_utf8_regex(
        "llg-safe+",
        r#"[^"\\\x00-\x1F\x7F]+"#,
    )
    .expect("safe-string slice regex must compile");
    let mut bytes = U8Set::empty();
    for token in &ordered_vocab.ordered_token_bytes {
        if safe_plus.is_match(token) {
            for &byte in token {
                bytes.insert(byte);
            }
        }
    }
    bytes
}

fn llg_safe_slice_token_bytes_for_artifacts(artifacts: &OrderedVocabTrieArtifacts) -> U8Set {
    *artifacts
        .llg_safe_slice_token_bytes
        .get_or_init(|| llg_safe_slice_token_bytes_for_ordered_vocab(&artifacts.ordered_vocab))
}

/// Return only the vocabulary-global byte support needed to choose exact
/// safe-slice containment candidates. This deliberately does not construct the
/// dynamic-mask trie or the full llguidance slice runtime template.
pub(crate) fn llg_safe_slice_token_bytes_for_vocab(vocab: &Vocab) -> U8Set {
    let artifacts = get_ordered_vocab_trie_artifacts_for_vocab(vocab).0;
    llg_safe_slice_token_bytes_for_artifacts(&artifacts)
}

fn llg_master_max_safe_chars_for_artifacts(artifacts: &OrderedVocabTrieArtifacts) -> u16 {
    *artifacts.llg_master_max_safe_chars.get_or_init(|| {
        let safe_plus = VocabPartitionDfa::compile_utf8_regex(
            "llg-safe+",
            r#"[^"\\\x00-\x1F\x7F]+"#,
        )
        .expect("safe-string slice regex must compile");
        artifacts
            .ordered_vocab
            .ordered_token_bytes
            .iter()
            .filter_map(|token| {
                safe_plus.is_match(token).then(|| {
                    let chars = std::str::from_utf8(token)
                        .expect("safe-string UTF-8 regex matched invalid UTF-8")
                        .chars()
                        .count();
                    u16::try_from(chars).expect("model token exceeds u16 Unicode-scalar count")
                })
            })
            .max()
            .unwrap_or(0)
    })
}

pub(crate) fn llg_master_max_safe_chars_for_vocab(vocab: &Vocab) -> u16 {
    let artifacts = get_ordered_vocab_trie_artifacts_for_vocab(vocab).0;
    llg_master_max_safe_chars_for_artifacts(&artifacts)
}

fn prepared_runtime_dynamic_vocab_with_llg(
    artifacts: &OrderedVocabTrieArtifacts,
) -> &Arc<DynamicMaskVocab> {
    artifacts.runtime_dynamic_vocab_llg.get_or_init(|| {
        let mut vocab = prepared_runtime_dynamic_vocab(artifacts).fresh_runtime_instance();
        prepare_llg_slice_leftovers_for_ordered_vocab(
            artifacts.ordered_vocab.as_ref(),
            llg_safe_slice_token_bytes_for_artifacts(artifacts),
            &mut vocab,
        );
        Arc::new(vocab)
    })
}

fn runtime_dynamic_vocab_artifacts(
    artifacts: &OrderedVocabTrieArtifacts,
) -> RuntimeDynamicMaskVocabArtifacts {
    let template = prepared_runtime_dynamic_vocab_with_llg(artifacts);
    RuntimeDynamicMaskVocabArtifacts {
        vocab: template.fresh_runtime_instance(),
    }
}

pub(crate) fn prepared_runtime_dynamic_vocab_for_vocab(vocab: &Vocab) -> Arc<DynamicMaskVocab> {
    let artifacts = get_ordered_vocab_trie_artifacts_for_vocab(vocab).0;
    Arc::clone(prepared_runtime_dynamic_vocab_with_llg(&artifacts))
}

pub(crate) fn runtime_dynamic_vocab_for_vocab(vocab: &Vocab) -> DynamicMaskVocab {
    prepared_runtime_dynamic_vocab_for_vocab(vocab).fresh_runtime_instance()
}

/// Build the runtime vocabulary used by the partition-optimized dynamic masker.
///
/// The constraint continues to own the complete original model vocabulary. This
/// value contains only one representative byte string per grammar-equivalence
/// class (modulo exact duplicate representative bytes), while each trie endpoint
/// expands directly to every original model token in the represented class.
/// Commit, composition compatibility, and boundary-trigger logic therefore stay
/// in original-token space; only the local dynamic mask walk is quotient-sized.
pub(crate) fn runtime_dynamic_vocab_for_partition(
    vocab: &Vocab,
    partition: ManyToOneIdMap,
    materialize_runtime_indexes: bool,
) -> Result<DynamicMaskVocab, String> {
    if partition.internal_to_originals.len() != partition.representative_original_ids.len() {
        return Err("vocabulary partition has inconsistent class metadata".to_owned());
    }

    let profile = crate::compiler::compile::compile_profile_enabled();
    let total_started = profile.then(Instant::now);
    let destructure_started = profile.then(Instant::now);
    let token_slots = if vocab.is_empty() {
        0
    } else {
        vocab.max_token_id() as usize + 1
    };
    let ManyToOneIdMap {
        original_to_internal: _,
        internal_to_originals,
        representative_original_ids,
    } = partition;
    let destructure_ms = destructure_started.map_or(0.0, elapsed_ms);
    let classes_started = profile.then(Instant::now);

    // Sort only class representatives. Class membership vectors are moved into
    // the runtime coordinate below instead of copying every original token ID.
    let mut classes = representative_original_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(class, representative)| {
            let bytes = vocab.get(representative).ok_or_else(|| {
                format!(
                    "vocabulary partition representative {representative} is absent from the vocabulary"
                )
            })?;
            Ok((class, representative, bytes))
        })
        .collect::<Result<Vec<_>, String>>()?;
    classes.sort_unstable_by(|left, right| left.2.cmp(right.2).then_with(|| left.0.cmp(&right.0)));
    let classes_ms = classes_started.map_or(0.0, elapsed_ms);
    let groups_started = profile.then(Instant::now);

    let mut groups = internal_to_originals
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>();
    let mut ordered_token_bytes = Vec::<Vec<u8>>::with_capacity(classes.len());
    let mut expanded_groups = Vec::<Vec<u32>>::with_capacity(classes.len());
    let mut index = 0usize;
    while index < classes.len() {
        let bytes = classes[index].2;
        let group_start = index;
        while index < classes.len() && classes[index].2 == bytes {
            index += 1;
        }

        let first_class = classes[group_start].0;
        let mut originals = groups[first_class]
            .take()
            .expect("partition class consumed twice");
        for &(class, _, _) in &classes[group_start + 1..index] {
            let mut duplicate_group = groups[class]
                .take()
                .expect("partition class consumed twice");
            originals.append(&mut duplicate_group);
        }
        if index - group_start > 1 || !originals.windows(2).all(|pair| pair[0] <= pair[1]) {
            // The grouped O2 partition keeps every class sorted by original
            // token ID.  Merging byte-identical canonical representatives can
            // concatenate multiple such classes, so restore that ordering only
            // in the uncommon duplicate-byte case.  The ordered runtime
            // constructor can then build sparse word masks linearly.
            originals.sort_unstable();
        }
        ordered_token_bytes.push(bytes.to_vec());
        expanded_groups.push(originals);
    }
    let groups_ms = groups_started.map_or(0.0, elapsed_ms);

    let ordered = OrderedVocab {
        original_slot_count: token_slots,
        ordered_to_originals: Arc::new(expanded_groups),
        ordered_token_bytes,
    };

    let prefix_started = profile.then(Instant::now);
    let prefix = build_ordered_vocab_prefix_tree(&ordered);
    let prefix_ms = prefix_started.map_or(0.0, elapsed_ms);
    let trie_started = profile.then(Instant::now);
    let runtime_trie = Arc::new(DynamicMaskTrie::from_vocab_prefix_tree(&prefix));
    let trie_ms = trie_started.map_or(0.0, elapsed_ms);
    // This mask depends only on the model vocabulary, never on the grammar
    // quotient. Reuse the prepared Vocab-global copy instead of scanning all
    // original token IDs for every O2 constraint.
    let full_vocab_template = prepared_runtime_dynamic_vocab_for_vocab(vocab);

    let prepared_all_original_token_words = full_vocab_template.all_original_token_words_arc();
    let runtime_started = profile.then(Instant::now);
    let mut runtime = if materialize_runtime_indexes {
        DynamicMaskVocab::from_materialized_ordered_with_all_original_token_words(
            runtime_trie,
            ordered.ordered_to_originals,
            Some(prepared_all_original_token_words),
        )
    } else {
        DynamicMaskVocab::from_materialized_ordered_for_transfer(
            runtime_trie,
            ordered.ordered_to_originals,
        )
    };
    let runtime_ms = runtime_started.map_or(0.0, elapsed_ms);

    // O2's main grammar quotient is already the small exact walk coordinate.
    // Reuse only the vocabulary-global safe/whitespace proof definitions; never
    // attach a separate master walk trie, which would discard the grammar
    // quotient and is unnecessary for this tier.
    runtime.inherit_llg_vocab_acceleration_without_master_from(full_vocab_template.as_ref());
    let inherit_started = profile.then(Instant::now);
    runtime.mark_grammar_quotiented();
    runtime.set_source_vocab_digest(crate::compiler::compile::vocab_content_digest(vocab));
    let inherit_ms = inherit_started.map_or(0.0, elapsed_ms);
    if let Some(total_started) = total_started {
        eprintln!(
            "[glrmask/profile][o2_quotient_materialize] materialize_runtime_indexes={materialize_runtime_indexes} destructure_ms={destructure_ms:.3} classes_ms={classes_ms:.3} groups_ms={groups_ms:.3} prefix_ms={prefix_ms:.3} trie_ms={trie_ms:.3} runtime_ms={runtime_ms:.3} inherit_ms={inherit_ms:.3} total_ms={:.3}",
            elapsed_ms(total_started),
        );
    }
    Ok(runtime)
}

pub(crate) fn prepare_vocab_for_dynamic_mask(vocab: &Vocab) {
    let artifacts = get_ordered_vocab_trie_artifacts_for_vocab(vocab).0;
    let _ = prepared_runtime_dynamic_vocab_with_llg(&artifacts);
}

/// Neutral PM artifact for the deferred mode. All dimensions are deliberately
/// unmapped so PM cannot force tokenizer-state or vocabulary splits during ID
/// reconciliation; the independently retained dynamic vocabulary is the exact
/// fallback representation.
fn empty_possible_matches_computation(
    tokenizer: &Tokenizer,
    original_token_count: usize,
    runtime_dynamic_vocab: RuntimeDynamicMaskVocabArtifacts,
) -> ConstraintPossibleMatchesComputation {
    let possible_matches_id_map = InternalIdMap {
        tokenizer_states: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            vec![u32::MAX; tokenizer.num_states() as usize],
            0,
        ),
        vocab_tokens: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            vec![u32::MAX; original_token_count],
            0,
        ),
        deferred_vocab_singleton_original_ids: None,
    };
    ConstraintPossibleMatchesComputation {
        mapped_possible_matches: MappedArtifact::new(
            RuntimePossibleMatchesByTerminal::new(),
            possible_matches_id_map,
        ),
        runtime_dynamic_vocab,
        complete: false,
        profile: ConstraintPossibleMatchesProfile::default(),
    }
}

fn complete_empty_possible_matches_computation(
    tokenizer: &Tokenizer,
    original_token_count: usize,
    runtime_dynamic_vocab: RuntimeDynamicMaskVocabArtifacts,
    possible_matches_collect_ms: f64,
) -> ConstraintPossibleMatchesComputation {
    let possible_matches_id_map = InternalIdMap {
        tokenizer_states: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            vec![u32::MAX; tokenizer.num_states() as usize],
            0,
        ),
        vocab_tokens: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            vec![u32::MAX; original_token_count],
            0,
        ),
        deferred_vocab_singleton_original_ids: None,
    };
    ConstraintPossibleMatchesComputation {
        mapped_possible_matches: MappedArtifact::new(
            RuntimePossibleMatchesByTerminal::new(),
            possible_matches_id_map,
        ),
        runtime_dynamic_vocab,
        complete: true,
        profile: ConstraintPossibleMatchesProfile {
            possible_matches_collect_ms,
            ..ConstraintPossibleMatchesProfile::default()
        },
    }
}

/// Runtime dynamic-mask state is not part of a complete static PM artifact.
///
/// Keep it unmaterialized when the PM analysis proves there are no delayed
/// terminals at all. `Constraint::dynamic_mask_vocab_for_runtime()` already
/// reconstructs this vocabulary from the retained token bytes if an unusual
/// runtime path later asks for exact dynamic masking, so constructing the full
/// byte-sorted vocabulary/trie here would only charge vocabulary-global setup
/// to the first otherwise-trivial static constraint.
fn lazy_runtime_dynamic_vocab_artifacts() -> RuntimeDynamicMaskVocabArtifacts {
    RuntimeDynamicMaskVocabArtifacts {
        vocab: DynamicMaskVocab::default(),
    }
}

fn compute_constraint_possible_matches_with_artifacts(
    tokenizer: &Tokenizer,
    original_token_count: usize,
    artifacts_and_profile: (OrderedVocabTrieArtifacts, OrderedVocabCacheProfile),
    initial_vocab_map: Option<&ManyToOneIdMap>,
    raw_byte_to_class: Option<&[u8; 256]>,
) -> ConstraintPossibleMatchesComputation {
    let pm_started_at = Instant::now();

    let (artifacts, ordered_vocab_cache_profile) = artifacts_and_profile;
    emit_ordered_vocab_cache_profile(ordered_vocab_cache_profile);
    let runtime_dynamic_vocab = runtime_dynamic_vocab_artifacts(&artifacts);
    let ordered_vocab = artifacts.ordered_vocab;
    let trie = artifacts.trie;

    let demand = delayed_terminal_demand(tokenizer);
    if demand.terminals.is_zero() {
        return complete_empty_possible_matches_computation(
            tokenizer,
            original_token_count,
            runtime_dynamic_vocab,
            elapsed_ms(pm_started_at),
        );
    }

    let structured_dispatch = tokenizer.has_deterministic_dispatch();
    let scalar_dispatch = tokenizer.has_scalar_deterministic_dispatch();
    let dispatch_start = structured_dispatch.then(|| tokenizer.start_state());
    let mut trie_build_states: Vec<u32> = (0..tokenizer.num_states())
        .filter(|state| {
            Some(*state) != dispatch_start && demand.raw_query_state[*state as usize]
        })
        .collect();
    if structured_dispatch && demand.raw_query_state[tokenizer.start_state() as usize] {
        trie_build_states.extend(
            tokenizer
                .deterministic_dispatch_roots()
                .into_iter()
                .flatten()
                .copied()
                .filter(|&state| demand.raw_state_relevant[state as usize]),
        );
        trie_build_states.sort_unstable();
        trie_build_states.dedup();
    }

    let query_state_count = demand
        .raw_query_state
        .iter()
        .filter(|&&active| active)
        .count();
    let relevant_state_count = demand
        .raw_state_relevant
        .iter()
        .filter(|&&active| active)
        .count();
    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
    {
        eprintln!(
            "[glrmask/profile][pm_delayed_terminal_demand] terminals={} accepting_future_states={} query_states={} relevant_states={} total_terminals={} total_states={}",
            demand.terminals.count_ones(),
            demand.accepting_future_states,
            query_state_count,
            relevant_state_count,
            tokenizer.num_terminals(),
            tokenizer.num_states(),
        );
    }

    let root_terminal_union = demand.terminals.count_ones();
    // The exact NFA powerset collector only retains delayed-terminal-relevant
    // states. Admission must therefore scale with that live domain, not with
    // unrelated tokenizer states. Large synthesized tokenizers can have fewer
    // than 500 relevant states; rejecting the powerset on total state count
    // falls back to a multi-second sparse trie walk for no semantic benefit.
    let use_nfa_powerset_collect = tokenizer.has_epsilon_transitions()
        && !scalar_dispatch
        && nfa_powerset_collect_enabled(relevant_state_count, root_terminal_union);
    let use_sparse_root_collect = (tokenizer.has_epsilon_transitions() && !scalar_dispatch)
        || (sparse_root_collect_enabled()
            && trie_build_states.len() <= sparse_root_state_limit()
            && root_terminal_union <= sparse_root_terminal_limit());
    let use_batched_demand_collect = batched_demand_collect_enabled(scalar_dispatch)
        && (!tokenizer.has_epsilon_transitions() || scalar_dispatch);

    let mut trie_class_result = if use_batched_demand_collect {
        let started_at = Instant::now();
        let result = collect_batched_demand_possible_matches(
            tokenizer,
            &trie.root,
            &trie_build_states,
            &demand,
        );
        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][trie_build_batched_demand] states={} terminals={} classes={} ms={:.3}",
                trie_build_states.len(),
                root_terminal_union,
                result.class_maps.len(),
                elapsed_ms(started_at),
            );
        }
        result
    } else if use_nfa_powerset_collect {
        let mut relevant_bytes = [false; 256];
        for bytes in &ordered_vocab.ordered_token_bytes {
            for &byte in bytes {
                relevant_bytes[byte as usize] = true;
            }
        }
        let view_started_at = Instant::now();
        let powerset = build_possible_match_powerset_view(
            tokenizer,
            &relevant_bytes,
            raw_byte_to_class,
            &demand,
        );
        let view_build_ms = elapsed_ms(view_started_at);
        let mut view_entries = trie_build_states
            .iter()
            .map(|&raw_state| powerset.raw_start_to_view[raw_state as usize])
            .collect::<Vec<_>>();
        view_entries.sort_unstable();
        view_entries.dedup();
        // The batched collector computes the same exact delayed-terminal relation,
        // but avoids replaying the vocabulary independently for every demanded
        // terminal. It wins broadly on small deterministic powerset views around
        // p90; larger views can regress, so keep a conservative structural cap.
        let powerset_batched_requested = std::env::var("GLRMASK_PM_POWERSET_BATCHED_DEMAND")
            .map(|value| {
                let value = value.trim();
                value.is_empty() || value == "1" || value.eq_ignore_ascii_case("true")
            })
            .unwrap_or(true);
        let powerset_batched_max_states = std::env::var("GLRMASK_PM_POWERSET_BATCHED_MAX_STATES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(512);
        let powerset_batched =
            powerset_batched_requested && view_entries.len() <= powerset_batched_max_states;
        let view_result = if powerset_batched {
            collect_batched_demand_possible_matches_precomputed(
                &trie.root,
                &view_entries,
                powerset.num_states,
                tokenizer.num_terminals() as usize,
                &powerset.matched_terminals,
                &powerset.is_end,
                &powerset.byte_transitions,
                &powerset.self_loop_bytes,
                &powerset.boundary_state,
                &demand,
            )
        } else {
            collector::collect_possible_matches_interval_trie_class_build_precomputed(
                &trie.root,
                &view_entries,
                Some(&powerset.boundary_state),
                powerset.num_states,
                tokenizer.num_terminals() as usize,
                &powerset.matched_terminals,
                &powerset.is_end,
                &powerset.byte_transitions,
                &powerset.self_loop_bytes,
            )
            .0
        };
        if powerset_batched
            && std::env::var_os("GLRMASK_PM_POWERSET_BATCHED_STRICT_REFERENCE").is_some()
        {
            let strict_started_at = Instant::now();
            let (reference, _) =
                collector::collect_possible_matches_interval_trie_class_build_precomputed(
                    &trie.root,
                    &view_entries,
                    Some(&powerset.boundary_state),
                    powerset.num_states,
                    tokenizer.num_terminals() as usize,
                    &powerset.matched_terminals,
                    &powerset.is_end,
                    &powerset.byte_transitions,
                    &powerset.self_loop_bytes,
                );
            for &state in &view_entries {
                let actual_class = view_result.state_classes[state as usize];
                let reference_class = reference.state_classes[state as usize];
                assert_ne!(actual_class, u32::MAX);
                assert_ne!(reference_class, u32::MAX);
                let actual = view_result.class_maps[actual_class as usize].as_ref();
                let expected = reference.class_maps[reference_class as usize].as_ref();
                let by_terminal = |map: &IntervalPossibleMatchMap| {
                    let mut result = BTreeMap::<TerminalID, Vec<(u32, u32)>>::new();
                    for group in map {
                        for &terminal in group.terminals.iter() {
                            result
                                .entry(terminal)
                                .or_default()
                                .extend_from_slice(&group.ranges);
                        }
                    }
                    for ranges in result.values_mut() {
                        normalize_token_ranges(ranges);
                    }
                    result
                };
                let actual_by_terminal = by_terminal(actual);
                let expected_by_terminal = by_terminal(expected);
                if actual_by_terminal != expected_by_terminal {
                    let terminals = actual_by_terminal
                        .keys()
                        .chain(expected_by_terminal.keys())
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>();
                    for terminal in terminals {
                        let actual_ranges = actual_by_terminal
                            .get(&terminal)
                            .map(Vec::as_slice)
                            .unwrap_or(&[]);
                        let expected_ranges = expected_by_terminal
                            .get(&terminal)
                            .map(Vec::as_slice)
                            .unwrap_or(&[]);
                        if actual_ranges == expected_ranges {
                            continue;
                        }
                        let contains = |ranges: &[(u32, u32)], token: u32| {
                            ranges.iter().any(|&(start, end)| start <= token && token <= end)
                        };
                        let max_token = actual_ranges
                            .iter()
                            .chain(expected_ranges.iter())
                            .map(|&(_, end)| end)
                            .max()
                            .unwrap_or(0);
                        let witness = (0..=max_token).find(|&token| {
                            contains(actual_ranges, token) != contains(expected_ranges, token)
                        });
                        if let Some(token) = witness {
                            let bytes = ordered_vocab
                                .ordered_token_bytes
                                .get(token as usize)
                                .map(Vec::as_slice)
                                .unwrap_or(&[]);
                            eprintln!(
                                "[glrmask/profile][pm_powerset_batched_mismatch] state={} terminal={} token={} actual={} expected={} bytes={:?} actual_ranges={} expected_ranges={}",
                                state,
                                terminal,
                                token,
                                contains(actual_ranges, token),
                                contains(expected_ranges, token),
                                bytes,
                                actual_ranges.len(),
                                expected_ranges.len(),
                            );
                            break;
                        }
                    }
                    panic!("powerset batched-demand possible matches differ at view state {state}");
                }
            }
            eprintln!(
                "[glrmask/profile][pm_powerset_batched_strict_reference] states={} differs=false compare_ms={:.3}",
                view_entries.len(),
                elapsed_ms(strict_started_at),
            );
        }
        let mut state_classes = vec![u32::MAX; tokenizer.num_states() as usize];
        for &raw_state in &trie_build_states {
            let view_state = powerset.raw_start_to_view[raw_state as usize] as usize;
            state_classes[raw_state as usize] = view_result.state_classes[view_state];
        }
        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][trie_build_nfa_powerset] raw_states={} view_states={} root_view_states={} classes={} collector={} view_build_ms={:.3}",
                trie_build_states.len(),
                powerset.num_states,
                view_entries.len(),
                view_result.class_maps.len(),
                if powerset_batched { "batched_demand" } else { "interval_classes" },
                view_build_ms,
            );
        }
        TrieClassBuildResult {
            state_classes,
            class_maps: view_result.class_maps,
        }
    } else if use_sparse_root_collect {
        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][trie_build_sparse_root] states={} terminals={} max_states={} max_terminals={}",
                trie_build_states.len(),
                root_terminal_union,
                sparse_root_state_limit(),
                sparse_root_terminal_limit(),
            );
        }
        collect_sparse_root_possible_matches(
            tokenizer,
            &trie.root,
            &trie_build_states,
            None,
        )
    } else {
        collector::collect_possible_matches_interval_trie_class_build_with_classes(
            tokenizer,
            &trie.root,
            &trie_build_states,
            None,
        )
        .0
    };
    if tokenizer.has_deterministic_dispatch()
        && demand.raw_query_state[tokenizer.start_state() as usize]
    {
        attach_structured_dispatch_possible_matches(tokenizer, &mut trie_class_result, &demand);
    }
    trie_class_result =
        filter_trie_class_result_to_terminals(trie_class_result, &demand.terminals);

    let possible_matches_collect_ms = elapsed_ms(pm_started_at);

    let possible_match_vocab_started_at = Instant::now();
    let (possible_match_vocab, possible_matches) = build_possible_match_vocab_and_weights_from_interval_maps(&trie_class_result.class_maps, &trie_class_result.state_classes, ordered_vocab.as_ref());

    let local_vocab_map = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
        possible_match_vocab.original_to_internal.clone(),
        possible_match_vocab.internal_to_originals.len() as u32,
    );
    let vocab_tokens = if let Some(initial_vocab_map) = initial_vocab_map {
        initial_vocab_map.compose(&local_vocab_map)
    } else {
        local_vocab_map
    };

    let possible_matches_id_map = InternalIdMap {
        tokenizer_states: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            trie_class_result.state_classes.clone(),
            trie_class_result.state_classes.iter().copied().filter(|&class_id| class_id != u32::MAX).max().map(|class_id| class_id + 1).unwrap_or(0),
        ),
        vocab_tokens,
        deferred_vocab_singleton_original_ids: None,
    };

    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some() || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
        eprintln!("[glrmask/profile][possible_match_vocab] original_tokens={} ordered_byte_tokens={} possible_match_tokens={}", original_token_count, ordered_vocab.ordered_to_originals.len(), possible_matches_id_map.vocab_tokens.internal_to_originals.len());
    }

    let possible_match_vocab_ms = elapsed_ms(possible_match_vocab_started_at);

    ConstraintPossibleMatchesComputation {
        mapped_possible_matches: MappedArtifact::new(possible_matches, possible_matches_id_map),
        runtime_dynamic_vocab,
        complete: true,
        profile: ConstraintPossibleMatchesProfile {
            vocab_equiv_ms: 0.0,
            possible_matches_collect_ms,
            possible_match_vocab_ms,
        },
    }
}

fn compute_constraint_possible_matches_for_vocab_impl(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    config: ConstraintPossibleMatchesConfig,
    raw_byte_to_class: Option<&[u8; 256]>,
) -> ConstraintPossibleMatchesComputation {
    if config.defer_to_dynamic_mask {
        let (full_artifacts, full_profile) = get_ordered_vocab_trie_artifacts_for_vocab(vocab);
        emit_ordered_vocab_cache_profile(full_profile);
        let runtime_dynamic_vocab = runtime_dynamic_vocab_artifacts(&full_artifacts);
        return empty_possible_matches_computation(
            tokenizer,
            vocab.entries_map().len(),
            runtime_dynamic_vocab,
        );
    }

    // This is a grammar/tokenizer fact, independent of the model vocabulary.
    // Check it before touching the vocabulary-global ordered-trie cache.  For
    // simple exact languages (for example the JSON schema that accepts only
    // `{}`), there is no delayed terminal whose possible-token relation needs
    // collecting, so a full Llama-sized trie build and dynamic-mask template
    // are pure first-use overhead.
    if delayed_terminal_demand(tokenizer).terminals.is_zero() {
        return complete_empty_possible_matches_computation(
            tokenizer,
            vocab.entries_map().len(),
            lazy_runtime_dynamic_vocab_artifacts(),
            0.0,
        );
    }

    if pm_vocab_equiv_enabled() && pm_vocab_equiv_supported(tokenizer) {
        let (full_artifacts, full_profile) = get_ordered_vocab_trie_artifacts_for_vocab(vocab);
        let runtime_dynamic_vocab = runtime_dynamic_vocab_artifacts(&full_artifacts);
        emit_ordered_vocab_cache_profile(full_profile);
        let vocab_equiv_started_at = Instant::now();
        let use_naive = std::env::var("GLRMASK_PM_VOCAB_EQUIV_NAIVE")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let pm_vocab_map = if use_naive || tokenizer.has_epsilon_transitions() {
            compute_pm_vocab_equivalence_map(
                tokenizer,
                full_artifacts.ordered_vocab.as_ref(),
                full_artifacts.trie.as_ref(),
            )
        } else {
            compute_pm_vocab_equivalence_map_fast(tokenizer, full_artifacts.ordered_vocab.as_ref())
        };
        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][pm_vocab_equiv] original_tokens={} pm_vocab_classes={} mode={} ms={:.3}",
                vocab.entries_map().len(),
                pm_vocab_map.internal_to_originals.len(),
                if tokenizer.has_epsilon_transitions() {
                    "nfa_exact"
                } else if use_naive {
                    "naive"
                } else {
                    "fast"
                },
                elapsed_ms(vocab_equiv_started_at),
            );
        }
        let compact_token_bytes =
            build_internal_token_bytes_from_groups(vocab, &pm_vocab_map.internal_to_originals);
        let vocab_equiv_ms = elapsed_ms(vocab_equiv_started_at);
        let mut computation = compute_constraint_possible_matches_with_artifacts(
            tokenizer,
            vocab.entries_map().len(),
            get_ordered_vocab_trie_artifacts(&compact_token_bytes),
            Some(&pm_vocab_map),
            raw_byte_to_class,
        );
        computation.runtime_dynamic_vocab = runtime_dynamic_vocab;
        computation.profile.vocab_equiv_ms = vocab_equiv_ms;
        return computation;
    }

    compute_constraint_possible_matches_with_artifacts(
        tokenizer,
        vocab.entries_map().len(),
        get_ordered_vocab_trie_artifacts_for_vocab(vocab),
        None,
        raw_byte_to_class,
    )
}

pub(crate) fn compute_constraint_possible_matches_for_vocab(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    config: ConstraintPossibleMatchesConfig,
) -> ConstraintPossibleMatchesComputation {
    compute_constraint_possible_matches_for_vocab_impl(tokenizer, vocab, config, None)
}

pub(crate) fn compute_constraint_possible_matches_for_vocab_with_raw_byte_classes(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    config: ConstraintPossibleMatchesConfig,
    raw_byte_to_class: &[u8; 256],
) -> ConstraintPossibleMatchesComputation {
    compute_constraint_possible_matches_for_vocab_impl(
        tokenizer,
        vocab,
        config,
        Some(raw_byte_to_class),
    )
}

pub(crate) fn prepare_vocab_for_possible_matches(vocab: &Vocab) {
    // Cache the byte-sorted vocabulary and its prefix tree. The packed dynamic
    // mask trie is substantially larger and is irrelevant to ordinary static
    // constraints with complete possible-match tables, so leave that second
    // representation lazy until a dynamic mask is actually requested.
    let _ = get_ordered_vocab_trie_artifacts_for_vocab(vocab);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer;
    use crate::compiler::pipeline::build_tokenizer_from_exprs_partitioned_with_adaptive;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::BTreeSet;

    #[test]
    fn possible_match_configs_report_table_completeness() {
        assert!(ConstraintPossibleMatchesConfig::EAGER.is_complete());
        assert!(!ConstraintPossibleMatchesConfig::DEFER_TO_DYNAMIC_MASK.is_complete());
    }

    fn directly_matched_terminals(
        tokenizer: &Tokenizer,
        start_state: u32,
        bytes: &[u8],
    ) -> BTreeSet<u32> {
        let mut states = tokenizer.execute_from_state_end_only(&[], start_state);
        let mut terminals = BTreeSet::new();
        if bytes.is_empty() {
            for &state in &states {
                terminals.extend(tokenizer.matched_terminals_iter(state));
            }
        }
        for &byte in bytes {
            states = tokenizer.step_all(&states, byte);
            for &state in &states {
                terminals.extend(tokenizer.matched_terminals_iter(state));
            }
            if states.is_empty() {
                break;
            }
        }
        terminals
    }

    fn assert_demanded_pm_matches_direct(
        tokenizer: &Tokenizer,
        entries: &[(u32, Vec<u8>)],
        context: &str,
    ) {
        let vocab = Vocab::new(entries.to_vec());
        let computation = compute_constraint_possible_matches_for_vocab(
            tokenizer,
            &vocab,
            ConstraintPossibleMatchesConfig::EAGER,
        );
        assert!(computation.complete);
        let mapped = &computation.mapped_possible_matches;
        let demand = delayed_terminal_demand(tokenizer);

        for terminal in 0..tokenizer.num_terminals() {
            if !demand.terminals.contains(terminal as usize) {
                assert!(
                    !mapped.artifact().contains_key(&terminal),
                    "non-demand terminal {terminal} must not force a PM row",
                );
            }
        }

        for state in 0..tokenizer.num_states() {
            let internal_state =
                mapped.id_map().tokenizer_states.original_to_internal[state as usize];
            if !demand.raw_query_state[state as usize] {
                assert_eq!(internal_state, u32::MAX, "non-query state={state}");
                continue;
            }
            assert_ne!(internal_state, u32::MAX, "query state={state}");
            for (token_id, bytes) in entries {
                let internal_token =
                    mapped.id_map().vocab_tokens.original_to_internal[*token_id as usize];
                let expected = directly_matched_terminals(tokenizer, state, bytes);
                for terminal in demand.terminals.iter().map(|terminal| terminal as u32) {
                    let actual = internal_token != u32::MAX
                        && mapped.artifact().get(&terminal).is_some_and(|weight| {
                            weight
                                .tokens_for_tsid(internal_state)
                                .contains(internal_token)
                        });
                    assert_eq!(
                        actual,
                        expected.contains(&terminal),
                        "{context} state={state} token={token_id} bytes={bytes:?} terminal={terminal}",
                    );
                }
            }
        }
    }

    #[test]
    fn demanded_possible_matches_match_direct_state_set_execution() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b" ".to_vec())),
                min: 1,
                max: None,
            },
        ];
        let tokenizer = build_tokenizer_from_exprs_partitioned_with_adaptive(
            &expressions,
            None,
            &[0, 1, 2, 2],
            false,
        );
        assert!(tokenizer.has_deterministic_dispatch());
        assert!(pm_vocab_equiv_supported(&tokenizer));

        let entries = vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"b".to_vec()),
            (3, b" a".to_vec()),
            (4, b"a ".to_vec()),
            (5, b"x".to_vec()),
            (6, b"ab".to_vec()),
        ];
        let demand = delayed_terminal_demand(&tokenizer);
        assert_eq!(demand.terminals.iter().collect::<Vec<_>>(), vec![3]);
        assert_demanded_pm_matches_direct(&tokenizer, &entries, "structured dispatch");
    }

    #[test]
    fn demanded_possible_matches_match_direct_execution_on_random_small_lexers() {
        let alphabet = [b'a', b'b', b' '];
        let mut entries = vec![(0u32, Vec::new())];
        fn add_words(
            entries: &mut Vec<(u32, Vec<u8>)>,
            alphabet: &[u8],
            prefix: &mut Vec<u8>,
            remaining: usize,
        ) {
            if remaining == 0 {
                return;
            }
            for &byte in alphabet {
                prefix.push(byte);
                entries.push((entries.len() as u32, prefix.clone()));
                add_words(entries, alphabet, prefix, remaining - 1);
                prefix.pop();
            }
        }
        add_words(&mut entries, &alphabet, &mut Vec::new(), 4);

        let mut rng = StdRng::seed_from_u64(0x504d_4445_4d41_4e44);
        for case in 0..32 {
            let terminal_count = rng.gen_range(2..=6);
            let mut expressions = Vec::with_capacity(terminal_count);
            let mut partitions = Vec::with_capacity(terminal_count);
            for _ in 0..terminal_count {
                let byte = alphabet[rng.gen_range(0..alphabet.len())];
                let other = alphabet[rng.gen_range(0..alphabet.len())];
                let expression = match rng.gen_range(0..5) {
                    0 => Expr::U8Seq(vec![byte]),
                    1 => Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(vec![byte])),
                        min: 1,
                        max: None,
                    },
                    2 => Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(vec![byte])),
                        min: 1,
                        max: Some(3),
                    },
                    3 => Expr::Seq(vec![
                        Expr::U8Seq(vec![byte]),
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(vec![other])),
                            min: 0,
                            max: None,
                        },
                    ]),
                    _ => Expr::Choice(vec![
                        Expr::U8Seq(vec![byte]),
                        Expr::U8Seq(vec![byte, other]),
                    ]),
                };
                expressions.push(expression);
                partitions.push(rng.gen_range(0..3));
            }
            let tokenizer = build_tokenizer_from_exprs_partitioned_with_adaptive(
                &expressions,
                None,
                &partitions,
                false,
            );
            assert_demanded_pm_matches_direct(
                &tokenizer,
                &entries,
                &format!("case={case} expressions={expressions:?} partitions={partitions:?}"),
            );
        }
    }

    #[test]
    fn no_delayed_terminals_produces_complete_empty_possible_matches() {
        let tokenizer = build_tokenizer_from_exprs_partitioned_with_adaptive(
            &[Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"b".to_vec())],
            None,
            &[0, 1],
            false,
        );
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let demand = delayed_terminal_demand(&tokenizer);
        assert!(demand.terminals.is_zero());

        let computation = compute_constraint_possible_matches_for_vocab(
            &tokenizer,
            &vocab,
            ConstraintPossibleMatchesConfig::EAGER,
        );
        assert!(computation.complete);
        assert!(computation.mapped_possible_matches.artifact().is_empty());
        assert!(
            !computation.runtime_dynamic_vocab.vocab.is_initialized(),
            "complete empty PM must not eagerly build the dynamic-mask vocabulary",
        );
        assert!(computation
            .mapped_possible_matches
            .id_map()
            .tokenizer_states
            .original_to_internal
            .iter()
            .all(|&state| state == u32::MAX));
        assert!(computation
            .mapped_possible_matches
            .id_map()
            .vocab_tokens
            .original_to_internal
            .iter()
            .all(|&token| token == u32::MAX));
    }

    #[test]
    fn batched_future_masks_follow_bytes_and_boundary_edges() {
        // 0 -a-> 1 -b-> 2, with 3 projecting to 1 at a token boundary.
        // Only state 2 directly matches demanded bit 0b10; every predecessor
        // that can reach it must inherit that bit, while disconnected state 4
        // must remain empty.
        let mut transitions = vec![vec![u32::MAX; 5]; 256];
        transitions[b'a' as usize][0] = 1;
        transitions[b'b' as usize][1] = 2;
        let boundary = [u32::MAX, u32::MAX, u32::MAX, 1, u32::MAX];
        let masks = batched_future_matched_masks(
            &[0, 0, 0b10, 0, 0],
            &transitions,
            Some(&boundary),
        );
        assert_eq!(masks, vec![0b10, 0b10, 0b10, 0b10, 0]);
    }

    #[test]
    fn epsilon_nfa_possible_match_collector_defaults_by_state_scale() {
        assert!(nfa_powerset_collect_default(914, 1_707));
        assert!(nfa_powerset_collect_default(8_108, 1_000));
        assert!(nfa_powerset_collect_default(10_355, 1_000));
        assert!(nfa_powerset_collect_default(
            PM_NFA_POWERSET_DEFAULT_MAX_STATES,
            usize::MAX,
        ));
        assert!(!nfa_powerset_collect_default(18_943, 1_707));
        assert!(nfa_powerset_collect_default(26_965, 192));
        assert!(!nfa_powerset_collect_default(
            PM_NFA_POWERSET_NARROW_MAX_STATES + 1,
            192,
        ));
        assert!(!nfa_powerset_collect_default(26_965, 1_707));
    }

    #[test]
    fn epsilon_powerset_interval_collector_matches_sparse_nfa_rows() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        assert!(tokenizer.has_epsilon_transitions());
        assert!(!tokenizer.has_deterministic_dispatch());

        let vocab = Vocab::new(
            vec![
                (0, b"".to_vec()),
                (1, b"a".to_vec()),
                (2, b"aa".to_vec()),
                (3, b"ab".to_vec()),
                (4, b"b".to_vec()),
                (5, b"ba".to_vec()),
                (6, b"x".to_vec()),
            ]);
        let artifacts = get_ordered_vocab_trie_artifacts_for_vocab(&vocab).0;
        let raw_states = (0..tokenizer.num_states()).collect::<Vec<_>>();
        let sparse = collect_sparse_root_possible_matches(
            &tokenizer,
            &artifacts.trie.root,
            &raw_states,
            None,
        );

        let mut relevant_bytes = [false; 256];
        for bytes in &artifacts.ordered_vocab.ordered_token_bytes {
            for &byte in bytes {
                relevant_bytes[byte as usize] = true;
            }
        }
        let demand = DelayedTerminalDemand {
            terminals: BitSet::all(tokenizer.num_terminals() as usize),
            raw_state_relevant: vec![true; tokenizer.num_states() as usize],
            raw_query_state: vec![true; tokenizer.num_states() as usize],
            accepting_future_states: tokenizer.num_states() as usize,
        };
        let powerset =
            build_possible_match_powerset_view(&tokenizer, &relevant_bytes, None, &demand);
        let mut view_entries = powerset.raw_start_to_view.clone();
        view_entries.retain(|&state| state != u32::MAX);
        view_entries.sort_unstable();
        view_entries.dedup();
        let (powerset_rows, _) =
            collector::collect_possible_matches_interval_trie_class_build_precomputed(
                &artifacts.trie.root,
                &view_entries,
                Some(&powerset.boundary_state),
                powerset.num_states,
                tokenizer.num_terminals() as usize,
                &powerset.matched_terminals,
                &powerset.is_end,
                &powerset.byte_transitions,
                &powerset.self_loop_bytes,
            );
        let sparse_expanded = expand_interval_class_maps(&sparse.class_maps);
        let powerset_expanded = expand_interval_class_maps(&powerset_rows.class_maps);

        for raw_state in raw_states {
            let sparse_class = sparse.state_classes[raw_state as usize];
            assert_ne!(sparse_class, u32::MAX, "raw_state={raw_state}");
            let view_state = powerset.raw_start_to_view[raw_state as usize] as usize;
            let powerset_class = powerset_rows.state_classes[view_state];
            assert_ne!(powerset_class, u32::MAX, "raw_state={raw_state}");
            assert_eq!(
                sparse_expanded[sparse_class as usize].as_ref(),
                powerset_expanded[powerset_class as usize].as_ref(),
                "raw_state={raw_state} view_state={view_state}",
            );
        }
    }

    #[test]
    fn epsilon_pm_vocab_equivalence_distinguishes_terminals_above_127() {
        let mut expressions = Vec::new();
        let mut partitions = Vec::new();
        for terminal in 0..130u32 {
            expressions.push(Expr::U8Seq(vec![terminal as u8]));
            partitions.push(terminal % 3);
        }
        let tokenizer = build_tokenizer_from_exprs_partitioned_with_adaptive(
            &expressions,
            None,
            &partitions,
            false,
        );
        assert!(tokenizer.has_epsilon_transitions());

        let vocab = Vocab::new(vec![(0, vec![128]), (1, vec![129])]);
        let full_artifacts = get_ordered_vocab_trie_artifacts_for_vocab(&vocab).0;
        let classes = compute_pm_vocab_equivalence_map(
            &tokenizer,
            full_artifacts.ordered_vocab.as_ref(),
            full_artifacts.trie.as_ref(),
        );

        assert_ne!(
            classes.original_to_internal[0],
            classes.original_to_internal[1],
            "PM vocab equivalence must include terminal IDs above the old u128 ceiling",
        );
    }
}
