//! Fast vocab equivalence analysis based on DFA behavior signatures.
//!
//! Each token is classified by its match positions, suffix structure, and end
//! states across all tokenizer starts.

use super::super::compat::{compute_byte_classes, FlatDfa, FlatTransitionCache, TokenizerView};
use ahash::{AHasher, RandomState};
use hashbrown::HashMap;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use super::super::disallowed_follows::normalize_disallowed_follows;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::compiler::stages::id_map_and_terminal_dwa::types::compile_profile_enabled;

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

type EdgeList = SmallVec<[(usize, usize); 4]>;

struct DagNode {
    hash: u64,
    edges: EdgeList,
    end_state: usize,
}

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE: u32 = u32::MAX;
const STATE_NONE: usize = usize::MAX;
const VOCAB_MATCH_POSITIONS_GROUP_BYTES: usize = 256 * 1024;
const VOCAB_DEFAULT_BATCH_MAX_STATES: usize = 4_000;
const VOCAB_DEFAULT_BATCH_MATCH_POSITION_BYTES: usize = 4 * 1024 * 1024;
const VOCAB_LARGE_WORK_BATCH_MATCH_POSITION_BYTES: usize = 768 * 1024;
// A one-batch trie walk is already bounded to one state slab. For modest
// token×state work, running it directly avoids nested Rayon scheduling while
// retaining full lexical-prefix sharing. Larger analyses remain chunk-parallel.
const VOCAB_SEQUENTIAL_TRIE_WORK_MAX_DEFAULT: usize = 10_000_000;
// Once the compile pool is genuinely wide, coarse 128-token trie chunks stop
// competing for scarce macro-parallel workers. Let moderate one-batch analyses
// use that path while retaining the historical conservative cutoff on smaller pools.
const VOCAB_SEQUENTIAL_TRIE_WORK_MAX_LARGE_POOL: usize = 40_000;
const VOCAB_SEQUENTIAL_TRIE_LARGE_POOL_THREADS: usize = 32;
const SELF_LOOP_ACTIVE_LEN_LIMIT: usize = 512;

#[inline]
fn state_signature_block_factor(block_len: usize) -> u64 {
    let mut factor = 1u64;
    for _ in 0..block_len {
        factor = factor.wrapping_mul(HASH_SEED1);
    }
    factor
}

#[inline]
fn append_state_signature_block(prefix: u64, block: u64, factor: u64) -> u64 {
    // A block signature is
    //   HASH_SEED3 * a^n + observations(block)
    // for a = HASH_SEED1. Appending it to an existing prefix therefore gives
    //   prefix * a^n + observations(block)
    // = block + (prefix - HASH_SEED3) * a^n.
    // This is bit-for-bit the same wrapping-u64 polynomial as one monolithic
    // state batch, not merely an equivalent vector of per-batch hashes.
    block.wrapping_add(prefix.wrapping_sub(HASH_SEED3).wrapping_mul(factor))
}

/// Flat DFA with byte-class-compressed transposed transition tables.
///
/// Byte equivalence classes group bytes that produce identical transitions across
/// all DFA states. The transposed layout `trans_by_class[class * num_states + state]`
/// gives optimal cache locality: for a given byte, all state lookups hit a single
/// contiguous array chunk that fits in L1 cache.
struct Dfa {
    start_state: usize,
    num_states: usize,
    /// Byte-to-class mapping (byte equivalence classes).
    byte_to_class: [u8; 256],
    /// Transposed transition table: `trans_by_class[class * num_states + state]`.
    /// For a given byte class, all state transitions are contiguous in memory.
    trans_by_class: Arc<[u32]>,
    finalizers: Vec<SmallVec<[usize; 4]>>,
    is_dead_end: Vec<bool>,
    num_groups: usize,
    possible_future_groups: Vec<SmallVec<[usize; 4]>>,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
    /// Per-state bitset: which bytes cause a self-loop (transition back to same state).
    self_loop_bytes: Arc<[U8Set]>,
    disallowed_follows: Vec<BitSet>,
}

/// Precomputed transition-only data that is identical across partitions.
///
/// `filter_for_terminals` only changes finalizers and possible_future_group_ids,
/// not transitions. So the compressed transition layouts, `byte_to_class`, and
/// `self_loop_bytes` can be computed once and shared across all partition
/// equivalence calls.
pub struct SharedVocabDfaBase {
    byte_to_class: [u8; 256],
    pub num_classes: usize,
    class_representatives: Arc<[u8]>,
    /// The two dense compressed layouts are independently lazy. A reduced
    /// representative analysis can use the shared byte partition without
    /// paying to materialize raw-state layouts it never walks.
    trans_by_class: OnceLock<Arc<[u32]>>,
    /// Row-major transition table: `state * num_classes + class`.
    trans_by_state_class: OnceLock<Arc<[u32]>>,
    self_loop_bytes: Arc<[U8Set]>,
    none_completion_hash: u64,
    /// Exact source table for pointer-fast and value-exact compatibility checks.
    source_transitions: Arc<[u32]>,
}

fn compute_byte_classes_and_self_loops(
    dfa: &super::super::compat::FlatDfa,
) -> ([u8; 256], Vec<U8Set>) {
    // Exact byte-class discovery already requires a full DFA-table pass. Build
    // self-loop masks in that same pass rather than paying for another scan.
    let mut column_hashes = [0u64; 256];
    let mut self_loop_bytes = Vec::with_capacity(dfa.states.len());
    for state in 0..dfa.states.len() {
        let mut self_loops = U8Set::empty();
        let base = state * 256;
        for byte in 0..256usize {
            let target = dfa.transitions[base + byte];
            column_hashes[byte] = column_hashes[byte]
                .wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(target as u64);
            if target == state as u32 {
                self_loops.insert(byte as u8);
            }
        }
        self_loop_bytes.push(self_loops);
    }

    let mut sorted_indices: [u8; 256] = std::array::from_fn(|i| i as u8);
    sorted_indices.sort_unstable_by_key(|&byte| column_hashes[byte as usize]);
    let mut byte_to_class = [0u8; 256];
    let mut next_class = 0u8;
    byte_to_class[sorted_indices[0] as usize] = 0;
    for i in 1..256 {
        let current = sorted_indices[i];
        let hash = column_hashes[current as usize];
        if hash != column_hashes[sorted_indices[i - 1] as usize] {
            next_class += 1;
            byte_to_class[current as usize] = next_class;
            continue;
        }
        let mut assigned = false;
        for j in (0..i).rev() {
            let previous = sorted_indices[j];
            if column_hashes[previous as usize] != hash {
                break;
            }
            let same = (0..dfa.states.len()).all(|state| {
                let base = state * 256;
                dfa.transitions[base + current as usize]
                    == dfa.transitions[base + previous as usize]
            });
            if same {
                byte_to_class[current as usize] = byte_to_class[previous as usize];
                assigned = true;
                break;
            }
        }
        if !assigned {
            next_class += 1;
            byte_to_class[current as usize] = next_class;
        }
    }
    (byte_to_class, self_loop_bytes)
}

/// Discover byte classes while observing only bytes used by the partition.
/// Irrelevant columns collapse into reserved class zero and are never walked by
/// these callers.
fn compute_byte_classes_and_self_loops_relevant(
    dfa: &super::super::compat::FlatDfa,
    relevant_bytes: &[bool; 256],
) -> ([u8; 256], Vec<U8Set>) {
    let relevant: Vec<usize> = (0..256usize).filter(|&byte| relevant_bytes[byte]).collect();
    if relevant.len() == 256 {
        // There is no irrelevant catch-all byte to reserve as class zero, and
        // 256 distinct relevant columns cannot be represented as classes 1..=256
        // in the u8 class table. The full exact builder already handles the
        // all-bytes case with classes 0..=255.
        return compute_byte_classes_and_self_loops(dfa);
    }
    let mut hashes = [0u64; 256];
    let mut self_loop_bytes = Vec::with_capacity(dfa.states.len());
    for state in 0..dfa.states.len() {
        let mut self_loops = U8Set::empty();
        let base = state * 256;
        for &byte in &relevant {
            let target = dfa.transitions[base + byte];
            hashes[byte] = hashes[byte]
                .wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(target as u64 + 1);
            if target == state as u32 {
                self_loops.insert(byte as u8);
            }
        }
        self_loop_bytes.push(self_loops);
    }

    let mut byte_to_class = [0u8; 256];
    let mut sorted = relevant;
    sorted.sort_unstable_by_key(|&byte| hashes[byte]);
    let mut next_class = 0u8;
    for (index, &byte) in sorted.iter().enumerate() {
        let hash = hashes[byte];
        let mut matching_class = None;
        for &previous in sorted[..index].iter().rev() {
            if hashes[previous] != hash {
                break;
            }
            let same_column = (0..dfa.states.len()).all(|state| {
                let base = state * 256;
                dfa.transitions[base + byte] == dfa.transitions[base + previous]
            });
            if same_column {
                matching_class = Some(byte_to_class[previous]);
                break;
            }
        }
        if let Some(class) = matching_class {
            byte_to_class[byte] = class;
        } else {
            next_class = next_class
                .checked_add(1)
                .expect("relevant byte classes must fit in the reserved u8 domain");
            byte_to_class[byte] = next_class;
        }
    }
    (byte_to_class, self_loop_bytes)
}

impl SharedVocabDfaBase {
    fn class_representatives(byte_to_class: &[u8; 256], num_classes: usize) -> Arc<[u8]> {
        let mut representatives = vec![0u8; num_classes];
        let mut seen = vec![false; num_classes];
        for byte in 0..=255u8 {
            let class = byte_to_class[byte as usize] as usize;
            if !seen[class] {
                seen[class] = true;
                representatives[class] = byte;
            }
        }
        Arc::from(representatives)
    }

    fn build_layout(&self, transposed: bool) -> Arc<[u32]> {
        let num_states = self.source_transitions.len() / 256;
        let mut transitions = vec![NONE; self.num_classes * num_states];
        for state in 0..num_states {
            let source_base = state * 256;
            for class in 0..self.num_classes {
                let target = self.source_transitions
                    [source_base + self.class_representatives[class] as usize];
                let destination = if transposed {
                    class * num_states + state
                } else {
                    state * self.num_classes + class
                };
                transitions[destination] = target;
            }
        }
        Arc::from(transitions)
    }

    /// Build from transition data derived from sparse lexer rows. Byte classes
    /// and self-loop bytes are shared immediately; the two dense layouts remain
    /// lazy until a caller actually needs their access pattern.
    pub fn build_from_flat_transition_cache(cache: &FlatTransitionCache) -> Self {
        let byte_to_class = cache.byte_to_class;
        let num_classes = byte_to_class
            .iter()
            .copied()
            .max()
            .map_or(0usize, |max_class| max_class as usize + 1);
        let none_completion_hash = {
            let mut h = new_hasher();
            h.write_u8(0);
            h.finish()
        };
        Self {
            byte_to_class,
            num_classes,
            class_representatives: Self::class_representatives(&byte_to_class, num_classes),
            trans_by_class: OnceLock::new(),
            trans_by_state_class: OnceLock::new(),
            self_loop_bytes: Arc::clone(&cache.self_loop_bytes),
            none_completion_hash,
            source_transitions: Arc::clone(&cache.transitions),
        }
    }

    /// Build from a FlatDfa. Called lazily via OnceLock on first use.
    pub fn build_from_dfa(dfa: &super::super::compat::FlatDfa) -> Self {
        let (byte_to_class, self_loop_bytes) = compute_byte_classes_and_self_loops(dfa);
        let num_classes = byte_to_class
            .iter()
            .copied()
            .max()
            .map_or(0usize, |max_class| max_class as usize + 1);
        let none_completion_hash = {
            let mut h = new_hasher();
            h.write_u8(0);
            h.finish()
        };
        Self {
            byte_to_class,
            num_classes,
            class_representatives: Self::class_representatives(&byte_to_class, num_classes),
            trans_by_class: OnceLock::new(),
            trans_by_state_class: OnceLock::new(),
            self_loop_bytes: Arc::from(self_loop_bytes),
            none_completion_hash,
            source_transitions: Arc::clone(&dfa.transitions),
        }
    }

    /// Build a conservative local byte layout for a finite vocabulary. Every
    /// relevant byte receives its own class; all other bytes share an unused
    /// catch-all class. Token walks only observe relevant bytes, so this is an
    /// exact replacement for full 256-column class discovery on that walk.
    pub fn build_from_dfa_for_relevant_bytes(
        dfa: &super::super::compat::FlatDfa,
        relevant_bytes: &[bool; 256],
    ) -> Option<Self> {
        let relevant_count = relevant_bytes.iter().filter(|&&relevant| relevant).count();
        if relevant_count >= 255 {
            return None;
        }
        let mut byte_to_class = [0u8; 256];
        let mut next_class = 1u8;
        for byte in 0..=255u8 {
            if relevant_bytes[byte as usize] {
                byte_to_class[byte as usize] = next_class;
                next_class += 1;
            }
        }
        let num_classes = next_class as usize;
        let mut self_loop_bytes = Vec::with_capacity(dfa.states.len());
        for state in 0..dfa.states.len() {
            let mut self_loops = U8Set::empty();
            let base = state * 256;
            for byte in 0..=255u8 {
                if relevant_bytes[byte as usize]
                    && dfa.transitions[base + byte as usize] == state as u32
                {
                    self_loops.insert(byte);
                }
            }
            self_loop_bytes.push(self_loops);
        }
        let none_completion_hash = {
            let mut h = new_hasher();
            h.write_u8(0);
            h.finish()
        };
        Some(Self {
            byte_to_class,
            num_classes,
            class_representatives: Self::class_representatives(&byte_to_class, num_classes),
            trans_by_class: OnceLock::new(),
            trans_by_state_class: OnceLock::new(),
            self_loop_bytes: Arc::from(self_loop_bytes),
            none_completion_hash,
            source_transitions: Arc::clone(&dfa.transitions),
        })
    }

    /// Build a byte-class base that only distinguishes the *relevant* bytes
    /// (those marked `true`), collapsing every other byte into a single class.
    ///
    /// This is exact for any analysis whose transition walks only ever follow
    /// relevant bytes — e.g. the C-seeded L2P state/vocab equivalence, which
    /// only steps through the partition's token bytes. Because irrelevant byte
    /// columns are never indexed, merging them is unobservable. Restricting the
    /// scan to relevant columns makes construction proportional to the number
    /// of relevant bytes instead of the full 256-wide alphabet.
    pub fn build_from_dfa_relevant(
        dfa: &super::super::compat::FlatDfa,
        relevant_bytes: &[bool; 256],
    ) -> Self {
        let (byte_to_class, self_loop_bytes) =
            compute_byte_classes_and_self_loops_relevant(dfa, relevant_bytes);
        let num_classes = byte_to_class
            .iter()
            .copied()
            .max()
            .map_or(0usize, |max_class| max_class as usize + 1);
        let none_completion_hash = {
            let mut h = new_hasher();
            h.write_u8(0);
            h.finish()
        };
        Self {
            byte_to_class,
            num_classes,
            class_representatives: Self::class_representatives(&byte_to_class, num_classes),
            trans_by_class: OnceLock::new(),
            trans_by_state_class: OnceLock::new(),
            self_loop_bytes: Arc::from(self_loop_bytes),
            none_completion_hash,
            source_transitions: Arc::clone(&dfa.transitions),
        }
    }

    /// Build a quotient's finite-vocabulary byte layout directly from raw
    /// representative rows. This avoids materializing an otherwise-unused
    /// quotient `states × 256` table: every relevant byte gets its own exact
    /// class and all remaining bytes share a catch-all class.
    pub fn build_from_raw_quotient_for_relevant_bytes(
        raw_transitions: &Arc<[u32]>,
        state_map: &ManyToOneIdMap,
        relevant_bytes: &[bool; 256],
        active_dead_classes: Option<&[bool]>,
    ) -> Option<Self> {
        let relevant_count = relevant_bytes.iter().filter(|&&relevant| relevant).count();
        if relevant_count >= 255 {
            return None;
        }
        let raw_states = state_map.original_to_internal.len();
        if raw_transitions.len() != raw_states * 256 {
            return None;
        }

        let mut byte_to_class = [0u8; 256];
        let mut next_class = 1u8;
        for byte in 0..=255u8 {
            if relevant_bytes[byte as usize] {
                byte_to_class[byte as usize] = next_class;
                next_class += 1;
            }
        }
        let num_classes = next_class as usize;
        let class_representatives = Self::class_representatives(&byte_to_class, num_classes);
        let quotient_states = state_map.internal_to_originals.len();
        if active_dead_classes.is_some_and(|dead| dead.len() != quotient_states) {
            return None;
        }
        let mut row_major = vec![NONE; quotient_states * num_classes];
        let mut self_loop_bytes = Vec::with_capacity(quotient_states);
        for (internal, &raw_representative) in state_map.representative_original_ids.iter().enumerate() {
            let raw_representative = raw_representative as usize;
            if raw_representative >= raw_states {
                return None;
            }
            let source_dead = active_dead_classes.is_some_and(|dead| dead[internal]);
            let raw_base = raw_representative * 256;
            if !source_dead {
                for class in 0..num_classes {
                    let target = raw_transitions[raw_base + class_representatives[class] as usize];
                    row_major[internal * num_classes + class] = if target == NONE {
                        NONE
                    } else {
                        let mapped = state_map.original_to_internal[target as usize];
                        if active_dead_classes.is_some_and(|dead| dead[mapped as usize]) {
                            NONE
                        } else {
                            mapped
                        }
                    };
                }
            }
            let mut self_loops = U8Set::empty();
            for byte in 0..=255u8 {
                if !relevant_bytes[byte as usize] {
                    continue;
                }
                if source_dead {
                    continue;
                }
                let target = raw_transitions[raw_base + byte as usize];
                if target != NONE {
                    let mapped = state_map.original_to_internal[target as usize];
                    if mapped == internal as u32
                        && !active_dead_classes.is_some_and(|dead| dead[mapped as usize])
                    {
                        self_loops.insert(byte);
                    }
                }
            }
            self_loop_bytes.push(self_loops);
        }
        let mut class_major = vec![NONE; quotient_states * num_classes];
        for state in 0..quotient_states {
            for class in 0..num_classes {
                class_major[class * quotient_states + state] = row_major[state * num_classes + class];
            }
        }
        let row_lock = OnceLock::new();
        let _ = row_lock.set(Arc::from(row_major));
        let class_lock = OnceLock::new();
        let _ = class_lock.set(Arc::from(class_major));
        let none_completion_hash = {
            let mut h = new_hasher();
            h.write_u8(0);
            h.finish()
        };
        Some(Self {
            byte_to_class,
            num_classes,
            class_representatives,
            trans_by_class: class_lock,
            trans_by_state_class: row_lock,
            self_loop_bytes: Arc::from(self_loop_bytes),
            none_completion_hash,
            source_transitions: Arc::from(Vec::<u32>::new()),
        })
    }

    /// Return the precomputed byte-to-class mapping.
    pub fn byte_to_class(&self) -> [u8; 256] {
        self.byte_to_class
    }

    /// Borrow the precomputed byte-to-class mapping for a compatible DFA.
    pub fn byte_to_class_ref(&self) -> &[u8; 256] {
        &self.byte_to_class
    }

    /// Borrow the class-major compressed layout for token-equivalence walks.
    pub fn transitions_by_class(&self) -> &[u32] {
        self.trans_by_class
            .get_or_init(|| self.build_layout(true))
            .as_ref()
    }

    pub fn transitions_by_class_arc(&self) -> Arc<[u32]> {
        Arc::clone(self.trans_by_class.get_or_init(|| self.build_layout(true)))
    }

    /// Borrow the row-major compressed layout for state-by-token walks.
    pub fn transitions_by_state_class(&self) -> &[u32] {
        self.trans_by_state_class
            .get_or_init(|| self.build_layout(false))
            .as_ref()
    }

    /// Check full compatibility without forcing either lazy layout.
    pub fn is_compatible_with_dfa(&self, dfa: &super::super::compat::FlatDfa) -> bool {
        let num_dfa_states = dfa.states.len();
        if self.source_transitions.is_empty() {
            return self.self_loop_bytes.len() == num_dfa_states
                && self
                    .trans_by_state_class
                    .get()
                    .is_some_and(|transitions| transitions.len() == num_dfa_states * self.num_classes);
        }
        self.self_loop_bytes.len() == num_dfa_states
            && self.source_transitions.len() == dfa.transitions.len()
            && (Arc::ptr_eq(&self.source_transitions, &dfa.transitions)
                || self.source_transitions.as_ref() == dfa.transitions.as_ref())
    }
}

/// Cache type for lazy SharedVocabDfaBase initialization across partitions.
pub type SharedVocabDfaCache = std::sync::OnceLock<SharedVocabDfaBase>;

/// Deterministic stage-local identity for one immutable vocabulary-analysis
/// DFA. The stage fixes the source tokenizer and raw transition relation; this
/// key therefore contains only the view-dependent fields materialized into the
/// DFA itself.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AnalysisDfaCacheKey {
    num_groups: usize,
    active_group_len: usize,
    active_group_words: Vec<u64>,
    normalized_disallowed_follows: Vec<BitSet>,
}

fn dfa_num_groups(dfa: &super::super::compat::FlatDfa) -> usize {
    dfa.states
        .iter()
        .flat_map(|state| {
            state
                .finalizers
                .iter()
                .copied()
                .chain(state.possible_future_group_ids.iter().copied())
        })
        .max()
        .map_or(0, |group| group + 1)
}

fn pack_active_groups(active_groups: Option<&[bool]>, default_len: usize) -> (usize, Vec<u64>) {
    let active_len = active_groups.map_or(default_len, <[bool]>::len);
    let mut words = vec![0u64; active_len.div_ceil(64)];
    for group in 0..active_len {
        if active_groups.map_or(true, |groups| groups[group]) {
            words[group / 64] |= 1u64 << (group % 64);
        }
    }
    (active_len, words)
}

impl AnalysisDfaCacheKey {
    fn for_analysis_view(
        tokenizer: &TokenizerView,
        active_groups: Option<&[bool]>,
        effective_disallowed_follows: &BTreeMap<u32, BitSet>,
    ) -> Self {
        let num_groups = dfa_num_groups(tokenizer.dfa());
        let (active_group_len, active_group_words) = pack_active_groups(active_groups, num_groups);
        Self {
            num_groups,
            active_group_len,
            active_group_words,
            normalized_disallowed_follows: normalize_disallowed_follows(
                num_groups,
                effective_disallowed_follows,
            ),
        }
    }
}

/// Stage-local cache for immutable full vocabulary-analysis DFAs.
///
/// Source identity is fixed by the surrounding terminal-DWA stage. Entries
/// are separated by the canonical active-terminal view and the normalized
/// effective disallowed-follows relation. In particular, terminal
/// interchangeability changes neither the raw transition infrastructure nor
/// this cache's source identity.
#[derive(Default)]
pub struct SharedVocabAnalysisDfaCache {
    entries: Mutex<HashMap<AnalysisDfaCacheKey, Arc<OnceLock<Arc<Dfa>>>>>,
}

impl SharedVocabAnalysisDfaCache {
    fn get_or_init(
        &self,
        key: AnalysisDfaCacheKey,
        build: impl FnOnce() -> Dfa,
    ) -> Arc<Dfa> {
        let entry = {
            let mut entries = self.entries.lock().expect("vocab analysis DFA cache poisoned");
            entries
                .entry(key)
                .or_insert_with(|| Arc::new(OnceLock::new()))
                .clone()
        };
        Arc::clone(entry.get_or_init(|| Arc::new(build())))
    }
}

impl Dfa {
    /// Get completion hash for a state (or none_completion_hash for STATE_NONE).
    #[inline]
    fn completion(&self, state: usize) -> u64 {
        if state < self.completion_hash.len() {
            self.completion_hash[state]
        } else {
            self.none_completion_hash
        }
    }

    #[inline]
    fn completion_with_disallowed(&self, state: usize, disallowed: Option<&BitSet>) -> u64 {
        let Some(disallowed) = disallowed.filter(|bits| !bits.is_zero()) else {
            return self.completion(state);
        };
        if state >= self.possible_future_groups.len() {
            return self.none_completion_hash;
        }
        let mut h = new_hasher();
        h.write_u8(2);
        h.write_u64(hash_filtered_group_list(&self.possible_future_groups[state], disallowed));
        h.finish()
    }

    /// Look up transition: given a DFA state and a byte, return the next state (u32).
    #[inline]
    fn transition(&self, state: usize, byte: u8) -> u32 {
        let class = self.byte_to_class[byte as usize] as usize;
        unsafe { *self.trans_by_class.get_unchecked(class * self.num_states + state) }
    }

    #[inline]
    fn disallowed_for(&self, gid: usize) -> &BitSet {
        &self.disallowed_follows[gid]
    }
}

/// Combined scratch space for batch DFA execution and suffix DAG construction.
struct Scratch {
    // Batch execution across initial states
    current_states: Vec<usize>,
    active_indices: Vec<usize>,
    match_positions: Vec<u32>,
    dirty_state_flags: Vec<u8>,
    /// Bitset mirroring `dirty_state_flags` for sparse dirty-token signature correction.
    dirty_state_bits: Vec<u64>,
    /// Polynomial weight of each state position in the token signature fold.
    completion_weights: Vec<u64>,
    /// Per-state multi-word dirty bitmask.  Layout: `[dirty_words * num_states]`
    /// where state `i`'s dirty mask occupies indices `[i*dirty_words .. (i+1)*dirty_words]`.
    dirty_group_masks: Vec<u64>,
    /// Number of u64 words per state in `dirty_group_masks` (= ceil(num_groups/64)).
    dirty_words: usize,
    num_groups: usize,
    /// For the active DFS path, how many initial states have their latest
    /// match for `(position, group)` at this slot.
    trie_target_group_counts: Vec<u32>,
    /// Bitset of live groups at each token position. This avoids rescanning all
    /// groups for every dirty token while the exact reference counts remain
    /// authoritative.
    trie_target_group_bits: Vec<u64>,
    trie_target_bits_enabled: bool,
    /// Number of live groups at each token position in the active DFS path.
    trie_target_position_counts: Vec<u32>,
    targets: Vec<usize>,
    target_gids: HashMap<usize, SmallVec<[usize; 16]>>,
    single_target_pos: usize,
    single_target_gids: SmallVec<[usize; 16]>,
    single_target_seen: Vec<u32>,
    single_target_seen_epoch: u32,
    // Suffix DAG
    dag_nodes: Vec<Option<DagNode>>,
    dag_queue: Vec<usize>,
    dag_disallowed: Vec<Option<BitSet>>,
    dense_dag_disallowed: Vec<BitSet>,
    dense_dag_disallowed_active: Vec<bool>,
    single_target_disallowed: BitSet,
    single_target_hash_pos: usize,
    single_target_hash: u64,
    suffix_match_positions: Vec<u32>,
    suffix_dirty_groups: SmallVec<[usize; 16]>,
}

static HASH_RANDOM_STATE: Lazy<RandomState> =
    Lazy::new(|| RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4));
static VOCAB_UNGROUPED_BATCH: Lazy<bool> =
    Lazy::new(|| env_flag_enabled("GLRMASK_VOCAB_UNGROUPED_BATCH"));
static VOCAB_ROW_CERT_DIAG: Lazy<bool> =
    Lazy::new(|| env_flag_enabled("GLRMASK_VOCAB_EQUIV_ROW_CERT_DIAG"));
static VOCAB_SPARSE_DIRTY_FINISH_DISABLED: Lazy<bool> =
    Lazy::new(|| env_flag_enabled("GLRMASK_DISABLE_VOCAB_SPARSE_DIRTY_FINISH"));
static VOCAB_TRIE_TARGET_BITS_ENABLED: Lazy<bool> =
    Lazy::new(|| !env_flag_enabled("GLRMASK_DISABLE_VOCAB_TRIE_TARGET_BITS"));
static VOCAB_DENSE_TRIE_SUFFIX_DAG_ENABLED: Lazy<bool> =
    Lazy::new(|| env_flag_override("GLRMASK_VOCAB_DENSE_TRIE_SUFFIX_DAG").unwrap_or(true));
#[inline]
fn new_hasher() -> AHasher {
    HASH_RANDOM_STATE.build_hasher()
}

#[inline]
fn intersect_disallowed(current: &mut BitSet, incoming: &BitSet) {
    if super::super::super::vocab_inplace_disallowed_intersection_enabled() {
        current.intersect_with(incoming);
    } else {
        *current = current.intersection(incoming);
    }
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let trimmed = value.trim();
            !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn env_flag_override(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|value| {
        let normalized = value.trim().to_ascii_lowercase();
        !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
    })
}

fn vocab_batch_size_override() -> Option<usize> {
    std::env::var("GLRMASK_VOCAB_EQUIV_BATCH_SIZE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
}

fn vocab_parallel_state_batch_size(num_states: usize, num_tokens: usize) -> Option<usize> {
    let _ = (num_states, num_tokens);
    // Keep state-axis decomposition opt-in. In isolation the slabs can expose
    // useful parallelism, but schema compilation already runs vocabulary
    // partitions concurrently. Automatically spawning several full-trie walks
    // inside each partition caused severe nested-parallel contention on a
    // 48-core build (notably the large p2 vocabulary). The explicit override is
    // retained for experiments and callers that know they have an isolated
    // vocabulary-proof workload.
    std::env::var("GLRMASK_VOCAB_EQUIV_PARALLEL_STATE_BATCH_SIZE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&batch_size| batch_size > 0)
}

fn vocab_sequential_trie_work_max_for_threads(threads: usize) -> usize {
    if threads >= VOCAB_SEQUENTIAL_TRIE_LARGE_POOL_THREADS {
        VOCAB_SEQUENTIAL_TRIE_WORK_MAX_LARGE_POOL
    } else {
        VOCAB_SEQUENTIAL_TRIE_WORK_MAX_DEFAULT
    }
}

fn vocab_sequential_trie_work_max() -> usize {
    if let Some(value) = std::env::var("GLRMASK_VOCAB_SEQUENTIAL_TRIE_WORK_MAX")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
    {
        return value;
    }
    vocab_sequential_trie_work_max_for_threads(rayon::current_num_threads())
}

fn first_transition_factor_enabled() -> bool {
    env_flag_override("GLRMASK_VOCAB_FIRST_TRANSITION_FACTOR").unwrap_or(true)
}

fn first_transition_factor_strict_reference_enabled() -> bool {
    env_flag_enabled("GLRMASK_VOCAB_FIRST_TRANSITION_FACTOR_STRICT_REFERENCE")
}

fn first_transition_literal_row_quotient_enabled() -> bool {
    !env_flag_enabled("GLRMASK_DISABLE_VOCAB_FIRST_TRANSITION_LITERAL_ROW_QUOTIENT")
}

fn first_transition_literal_row_quotient_max_work() -> usize {
    // The quotient scans every DFA state across every active suffix-byte class.
    // Keep that dense preprocessing bounded so it cannot dominate a factor pass
    // that already removed most token×state coordinates.
    std::env::var("GLRMASK_VOCAB_LITERAL_ROW_QUOTIENT_MAX_WORK")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(8_000_000)
}

fn literal_row_quotient_coarsening_rounds() -> usize {
    std::env::var("GLRMASK_VOCAB_LITERAL_ROW_COARSENING_ROUNDS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(1)
        .min(1)
}

fn first_transition_factor_min_bucket_tokens() -> usize {
    std::env::var("GLRMASK_VOCAB_FIRST_TRANSITION_FACTOR_MIN_BUCKET_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value >= 2)
        .unwrap_or(2)
}

fn first_transition_factor_max_work_ratio_override() -> Option<f64> {
    std::env::var("GLRMASK_VOCAB_FIRST_TRANSITION_FACTOR_MAX_WORK_RATIO")
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn first_transition_factor_moderate_extension_enabled(
    num_tokens: usize,
    num_states: usize,
    num_groups: usize,
) -> bool {
    (FIRST_TRANSITION_FACTOR_MODERATE_MIN_TOKENS
        ..=FIRST_TRANSITION_FACTOR_MODERATE_MAX_TOKENS)
        .contains(&num_tokens)
        && default_vocab_batch_size(num_states, num_groups, num_tokens) >= num_states
}

fn first_transition_factor_max_work_ratio(
    num_tokens: usize,
    num_states: usize,
    num_groups: usize,
) -> f64 {
    if let Some(value) = first_transition_factor_max_work_ratio_override() {
        return value;
    }

    // The first-transition prepartition has a second exact authority pass, so
    // its useful break-even point depends strongly on vocabulary size. Keep
    // the established conservative policy everywhere except moderate
    // vocabularies whose ordinary authority fits in one state batch. In that
    // shape the fixed authority cost is small enough that a 10% preliminary
    // state×token domain still gives a substantial reduction in trie work.
    // Multi-batch authorities already amortize/refine their trie work and do
    // not consistently repay the extra factor pass.
    if first_transition_factor_moderate_extension_enabled(num_tokens, num_states, num_groups) {
        FIRST_TRANSITION_FACTOR_MODERATE_MAX_WORK_RATIO
    } else {
        FIRST_TRANSITION_FACTOR_MAX_WORK_RATIO_DEFAULT
    }
}

fn first_transition_factor_parallel_buckets_override() -> Option<bool> {
    env_flag_override("GLRMASK_VOCAB_FIRST_TRANSITION_FACTOR_PARALLEL_BUCKETS")
}

fn first_transition_factor_token_chunk_size() -> usize {
    std::env::var("GLRMASK_VOCAB_FIRST_TRANSITION_FACTOR_TOKEN_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(1_024)
}

fn first_transition_factor_token_chunk_min_work() -> usize {
    std::env::var("GLRMASK_VOCAB_FIRST_TRANSITION_FACTOR_TOKEN_CHUNK_MIN_WORK")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(8_000_000)
}

fn first_transition_factor_final_single_batch_enabled() -> bool {
    env_flag_enabled("GLRMASK_VOCAB_FIRST_TRANSITION_FACTOR_FINAL_SINGLE_BATCH")
}

fn default_vocab_batch_size(
    num_states: usize,
    num_groups: usize,
    num_tokens: usize,
) -> usize {
    if num_states == 0 {
        return 0;
    }
    if num_groups == 0 {
        return num_states.min(VOCAB_DEFAULT_BATCH_MAX_STATES);
    }
    let bytes_per_state = num_groups.saturating_mul(std::mem::size_of::<u32>());
    let memory_bounded_states =
        (VOCAB_DEFAULT_BATCH_MATCH_POSITION_BYTES / bytes_per_state.max(1)).max(1);
    // On a large token × state product, refinement discovered in an early
    // state slice permanently removes singleton token classes from every later
    // slice.  Smaller slices therefore reduce total trie replay substantially.
    // For smaller vocabularies or state domains, the extra refinement rounds
    // are pure overhead, so retain the wider historical batch there.
    let refinement_bounded_states = if num_tokens >= 8_000 && num_states >= 2_000 {
        // Keep the dominant per-batch match-position slab within a typical
        // private cache. This was formerly a fixed 750-state tuning point;
        // deriving it from group width preserves the same locality benefit for
        // narrower and wider analysis DFAs.
        (VOCAB_LARGE_WORK_BATCH_MATCH_POSITION_BYTES / bytes_per_state.max(1)).max(1)
    } else {
        VOCAB_DEFAULT_BATCH_MAX_STATES
    };
    num_states
        .min(VOCAB_DEFAULT_BATCH_MAX_STATES)
        .min(refinement_bounded_states)
        .min(memory_bounded_states)
}

fn vocab_verify_token_pair_override() -> Option<(usize, usize)> {
    let value = std::env::var("GLRMASK_VOCAB_VERIFY_TOKEN_PAIR").ok()?;
    let mut parts = value.split(',').map(str::trim);
    let left = parts.next()?.parse::<usize>().ok()?;
    let right = parts.next()?.parse::<usize>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((left, right))
}

fn vocab_verify_token_pair_from_final_classes_enabled() -> bool {
    env_flag_enabled("GLRMASK_VOCAB_VERIFY_TOKEN_PAIR_FROM_FINAL_CLASSES")
}

fn vocab_state_group_size(num_states: usize, num_groups: usize) -> usize {
    if *VOCAB_UNGROUPED_BATCH || num_states <= 1 || num_groups == 0 {
        return num_states;
    }

    let bytes_per_state = num_groups.saturating_mul(std::mem::size_of::<u32>());
    let group_size = (VOCAB_MATCH_POSITIONS_GROUP_BYTES / bytes_per_state).max(1);
    group_size.min(num_states)
}

fn diversity_state_order_enabled() -> bool {
    !env_flag_enabled("GLRMASK_DISABLE_DIVERSITY_STATE_ORDER")
}

fn states_by_transition_diversity(dfa: &Dfa, states: &[usize]) -> Vec<usize> {
    let num_classes = dfa
        .byte_to_class
        .iter()
        .copied()
        .max()
        .map_or(0usize, |max_class| max_class as usize + 1);

    let mut ranked: Vec<(usize, usize)> = states
        .iter()
        .copied()
        .map(|state_id| {
            let mut targets = BTreeSet::new();
            for class in 0..num_classes {
                targets.insert(dfa.trans_by_class[class * dfa.num_states + state_id]);
            }
            (state_id, targets.len())
        })
        .collect();

    ranked.sort_unstable_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.into_iter().map(|(state_id, _)| state_id).collect()
}

/// Quotient source states by their exact observable value and literal
/// transition row over byte classes that can occur after the factored first
/// byte. This is exact for the finite-vocabulary suffix walk: equal rows enter
/// the same concrete target state for every observable next byte, so every
/// later suffix has identical behavior as well.
///
/// Dead-end states expose their current finalizers/completion but have no
/// future behavior, matching the vocabulary walk's early-death semantics.
fn exact_literal_state_row_quotient(
    dfa: &Dfa,
    active_byte_classes: &[bool],
    profiling: bool,
) -> Option<(Vec<u32>, Vec<usize>)> {
    let total_started_at = profiling.then(Instant::now);
    let num_states = dfa.num_states;
    if num_states == 0 || num_states > u32::MAX as usize {
        return None;
    }
    let num_classes = dfa
        .byte_to_class
        .iter()
        .copied()
        .max()
        .map_or(0usize, |max_class| max_class as usize + 1);
    if num_classes > 256 {
        return None;
    }
    if active_byte_classes.len() != num_classes {
        return None;
    }
    let active_class_count = active_byte_classes.iter().filter(|&&active| active).count();

    // Exact output labels: no probabilistic fingerprint is used here. The
    // future-group list determines the completion observation, while the
    // finalizer list is observable at the current consumed position.
    let labels_started_at = profiling.then(Instant::now);
    let mut label_ids = HashMap::<
        (bool, SmallVec<[usize; 4]>, SmallVec<[usize; 4]>),
        u32,
    >::with_capacity(num_states.min(16_384));
    let mut initial_classes = Vec::with_capacity(num_states);
    let mut next_label = 0u32;
    for state in 0..num_states {
        let key = (
            dfa.is_dead_end[state],
            dfa.finalizers[state].clone(),
            dfa.possible_future_groups[state].clone(),
        );
        let label = *label_ids.entry(key).or_insert_with(|| {
            let label = next_label;
            next_label += 1;
            label
        });
        initial_classes.push(label);
    }
    let labels_ms = labels_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let refine_started_at = profiling.then(Instant::now);
    let mut hashes = vec![HASH_SEED3; num_states];
    for state in 0..num_states {
        hashes[state] = hashes[state]
            .wrapping_mul(HASH_SEED1)
            .wrapping_add(initial_classes[state] as u64);
    }
    // Class-major transition storage lets us hash every state's literal
    // relevant-byte row with sequential reads and writes. Hash collisions are
    // resolved by the exact row comparison below.
    for (class, &active) in active_byte_classes.iter().enumerate() {
        if !active {
            continue;
        }
        let base = class * num_states;
        for state in 0..num_states {
            let target = if dfa.is_dead_end[state] {
                NONE
            } else {
                dfa.trans_by_class[base + state]
            };
            hashes[state] = hashes[state]
                .wrapping_mul(HASH_SEED1)
                .wrapping_add(target as u64);
        }
    }
    let mut order = (0..num_states).collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        initial_classes[left]
            .cmp(&initial_classes[right])
            .then_with(|| hashes[left].cmp(&hashes[right]))
            .then_with(|| left.cmp(&right))
    });
    let rows_equal = |left: usize, right: usize| {
        if initial_classes[left] != initial_classes[right] || hashes[left] != hashes[right] {
            return false;
        }
        for (class, &active) in active_byte_classes.iter().enumerate() {
            if !active {
                continue;
            }
            let base = class * num_states;
            let left_target = if dfa.is_dead_end[left] {
                NONE
            } else {
                dfa.trans_by_class[base + left]
            };
            let right_target = if dfa.is_dead_end[right] {
                NONE
            } else {
                dfa.trans_by_class[base + right]
            };
            if left_target != right_target {
                return false;
            }
        }
        true
    };
    let mut classes = vec![0u32; num_states];
    let mut class = 0u32;
    let mut previous = None::<usize>;
    for &state in &order {
        if previous.is_some_and(|left| !rows_equal(left, state)) {
            class += 1;
        }
        classes[state] = class;
        previous = Some(state);
    }
    let literal_class_count = class as usize + 1;

    // The literal-row classes above are already an exact behavioral
    // equivalence. Replacing each concrete target with its certified class and
    // regrouping by (observation, class-target row) can therefore only coarsen
    // that equivalence soundly. One round recovers useful recursively-equivalent
    // target rows cheaply; deeper rounds cost more than they save on the tail
    // workloads this path targets.
    let mut completed_coarsening_rounds = 0usize;
    for _ in 0..literal_row_quotient_coarsening_rounds() {
        hashes.fill(HASH_SEED3);
        for state in 0..num_states {
            hashes[state] = hashes[state]
                .wrapping_mul(HASH_SEED1)
                .wrapping_add(initial_classes[state] as u64);
        }
        for (byte_class, &active) in active_byte_classes.iter().enumerate() {
            if !active {
                continue;
            }
            let base = byte_class * num_states;
            for state in 0..num_states {
                let target_class = if dfa.is_dead_end[state] {
                    u32::MAX
                } else {
                    let target = dfa.trans_by_class[base + state];
                    if target == NONE {
                        u32::MAX
                    } else {
                        classes[target as usize]
                    }
                };
                hashes[state] = hashes[state]
                    .wrapping_mul(HASH_SEED1)
                    .wrapping_add(target_class as u64);
            }
        }
        order.sort_unstable_by(|&left, &right| {
            initial_classes[left]
                .cmp(&initial_classes[right])
                .then_with(|| hashes[left].cmp(&hashes[right]))
                .then_with(|| left.cmp(&right))
        });
        let quotient_rows_equal = |left: usize, right: usize| {
            if initial_classes[left] != initial_classes[right] || hashes[left] != hashes[right] {
                return false;
            }
            for (byte_class, &active) in active_byte_classes.iter().enumerate() {
                if !active {
                    continue;
                }
                let base = byte_class * num_states;
                let target_class = |state: usize| {
                    if dfa.is_dead_end[state] {
                        u32::MAX
                    } else {
                        let target = dfa.trans_by_class[base + state];
                        if target == NONE {
                            u32::MAX
                        } else {
                            classes[target as usize]
                        }
                    }
                };
                if target_class(left) != target_class(right) {
                    return false;
                }
            }
            true
        };
        let old_class_count = classes
            .iter()
            .copied()
            .max()
            .map_or(0usize, |id| id as usize + 1);
        let mut next_classes = vec![0u32; num_states];
        let mut next_class = 0u32;
        let mut previous = None::<usize>;
        for &state in &order {
            if previous.is_some_and(|left| !quotient_rows_equal(left, state)) {
                next_class += 1;
            }
            next_classes[state] = next_class;
            previous = Some(state);
        }
        let new_class_count = next_class as usize + 1;
        classes = next_classes;
        completed_coarsening_rounds += 1;
        if new_class_count == old_class_count {
            break;
        }
    }
    let refine_ms = refine_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let class_count = classes
        .iter()
        .copied()
        .max()
        .map_or(0usize, |class| class as usize + 1);
    let mut representatives = vec![usize::MAX; class_count];
    for (state, &class) in classes.iter().enumerate() {
        let representative = &mut representatives[class as usize];
        if *representative == usize::MAX {
            *representative = state;
        }
    }
    if profiling {
        eprintln!(
            "[glrmask/profile][vocab_literal_row_quotient_detail] states={} output_labels={} byte_classes={} active_byte_classes={} literal_classes={} classes={} coarsening_rounds={} labels_ms={:.3} refine_ms={:.3} total_ms={:.3}",
            num_states,
            next_label,
            num_classes,
            active_class_count,
            literal_class_count,
            class_count,
            completed_coarsening_rounds,
            labels_ms,
            refine_ms,
            total_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Some((classes, representatives))
}

#[inline]
fn hash_group_list(iter: impl ExactSizeIterator<Item = usize>) -> u64 {
    let mut h = new_hasher();
    h.write_u8(1);
    h.write_u64(iter.len() as u64);
    for v in iter {
        h.write_u64(v as u64);
    }
    h.finish()
}

#[inline]
fn hash_filtered_group_list(groups: &[usize], disallowed: &BitSet) -> u64 {
    let mut h = new_hasher();
    h.write_u8(1);
    let mut count = 0usize;
    for &gid in groups {
        if !disallowed.contains(gid) {
            count += 1;
        }
    }
    h.write_u64(count as u64);
    for &gid in groups {
        if !disallowed.contains(gid) {
            h.write_u64(gid as u64);
        }
    }
    h.finish()
}

fn build_dfa(
    tokenizer: &TokenizerView,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    byte_to_class_override: Option<&[u8; 256]>,
) -> Dfa {
    build_dfa_with_group_filter(tokenizer, disallowed_follows, byte_to_class_override, None, None)
}

/// Build the analysis DFA, optionally filtering to only active groups.
///
/// When `active_groups` is provided, only groups marked `true` are included
/// in finalizers and possible_future_groups. This is used for L2+-only
/// equivalence analysis where L1 terminal groups are excluded.
///
/// When `shared_cache` is provided, lazily initializes a SharedVocabDfaBase on
/// first call and reuses it for subsequent calls. This avoids redundant
/// computation of transition tables and self-loop bytes across partitions.
fn build_dfa_with_group_filter(
    tokenizer: &TokenizerView,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    byte_to_class_override: Option<&[u8; 256]>,
    active_groups: Option<&[bool]>,
    shared_cache: Option<&SharedVocabDfaCache>,
) -> Dfa {
    let dfa = tokenizer.dfa();
    assert!(dfa.states.len() <= u32::MAX as usize, "DFA too large");

    // Lazily initialize the shared base from this TokenizerView's DFA.
    let shared_base: Option<&SharedVocabDfaBase> = shared_cache.map(|cache| {
        cache.get_or_init(|| SharedVocabDfaBase::build_from_dfa(dfa))
    });

    // Compute the raw group axis before filtering. The active view removes
    // observations but does not change the terminal-ID coordinate used by
    // follow constraints or the cache key.
    let num_groups = dfa_num_groups(dfa);

    let mut finalizers = Vec::with_capacity(dfa.states.len());
    let mut is_dead_end = Vec::with_capacity(dfa.states.len());
    let mut possible_future_groups = Vec::with_capacity(dfa.states.len());
    let mut completion_hash = Vec::with_capacity(dfa.states.len());

    for (_state_idx, state) in dfa.states.iter().enumerate() {
        let filtered_finalizers: SmallVec<[usize; 4]> = if let Some(ag) = active_groups {
            state.finalizers.iter().copied().filter(|&gid| ag.get(gid).copied().unwrap_or(false)).collect()
        } else {
            state.finalizers.iter().copied().collect()
        };
        finalizers.push(filtered_finalizers);

        let future_groups: SmallVec<[usize; 4]> = if let Some(ag) = active_groups {
            state.possible_future_group_ids.iter().copied().filter(|&gid| ag.get(gid).copied().unwrap_or(false)).collect()
        } else {
            state.possible_future_group_ids.iter().copied().collect()
        };
        is_dead_end.push(future_groups.is_empty());
        completion_hash.push(hash_group_list(future_groups.iter().copied()));
        possible_future_groups.push(future_groups);
    }

    let num_dfa_states = dfa.states.len();

    // Use shared base if available and compatible, otherwise compute from scratch.
    let compatible_shared_base = shared_base.filter(|base| {
        base.is_compatible_with_dfa(dfa)
    });
    let (byte_to_class, trans_by_class, self_loop_bytes, none_completion_hash) =
        if let Some(base) = compatible_shared_base {
            let btc = byte_to_class_override.copied().unwrap_or(base.byte_to_class);
            (
                btc,
                base.transitions_by_class_arc(),
                Arc::clone(&base.self_loop_bytes),
                base.none_completion_hash,
            )
        } else {
            let btc = byte_to_class_override.copied().unwrap_or_else(|| compute_byte_classes(dfa));
            let num_classes = btc
                .iter()
                .copied()
                .max()
                .map_or(0usize, |max_class| max_class as usize + 1);
            let mut class_repr = vec![0u8; num_classes];
            let mut class_seen = vec![false; num_classes];
            for b in 0..=255u8 {
                let class = btc[b as usize] as usize;
                if !class_seen[class] {
                    class_seen[class] = true;
                    class_repr[class] = b;
                }
            }

            let mut tbc = vec![NONE; num_classes * num_dfa_states];
            for c in 0..num_classes {
                let repr = class_repr[c] as usize;
                let bbase = c * num_dfa_states;
                for s in 0..num_dfa_states {
                    tbc[bbase + s] = dfa.trans(s, repr);
                }
            }

            let mut slb = Vec::with_capacity(num_dfa_states);
            for s in 0..dfa.states.len() {
                let mut bits = U8Set::empty();
                for (byte_idx, &target) in dfa.transitions_for(s).iter().enumerate() {
                    if target == s as u32 {
                        bits.insert(byte_idx as u8);
                    }
                }
                slb.push(bits);
            }

            let nch = {
                let mut h = new_hasher();
                h.write_u8(0);
                h.finish()
            };

            (btc, Arc::from(tbc), Arc::from(slb), nch)
        };

    Dfa {
        start_state: dfa.start_state,
        num_states: num_dfa_states,
        byte_to_class,
        trans_by_class,
        finalizers,
        is_dead_end,
        num_groups,
        possible_future_groups,
        completion_hash,
        none_completion_hash,
        self_loop_bytes,
        disallowed_follows: normalize_disallowed_follows(num_groups, disallowed_follows),
    }
}

fn intersect_node_disallowed(
    scratch: &mut Scratch,
    pos: usize,
    incoming: &BitSet,
) {
    ensure_position_slot(&mut scratch.dag_disallowed, pos);
    if let Some(existing) = scratch.dag_disallowed[pos].as_mut() {
        intersect_disallowed(existing, incoming);
    } else {
        scratch.dag_disallowed[pos] = Some(incoming.clone());
    }
}

fn node_disallows_gid(scratch: &Scratch, pos: usize, gid: usize) -> bool {
    scratch
        .dag_disallowed
        .get(pos)
        .and_then(|bits| bits.as_ref())
        .map(|bits| bits.contains(gid))
        .unwrap_or(false)
}

#[inline]
fn ensure_position_slot<T>(slots: &mut Vec<Option<T>>, pos: usize) {
    if pos >= slots.len() {
        slots.resize_with(pos + 1, || None);
    }
}

impl Scratch {
    fn new(num_states: usize, num_groups: usize) -> Self {
        let dirty_words = (num_groups + 63) / 64;
        Scratch {
            current_states: vec![0; num_states],
            active_indices: Vec::new(),
            match_positions: vec![NONE; num_states * num_groups],
            dirty_state_flags: vec![0; num_states],
            dirty_state_bits: vec![0; num_states.div_ceil(64)],
            completion_weights: Vec::new(),
            dirty_group_masks: vec![0; num_states * dirty_words.max(1)],
            dirty_words,
            num_groups,
            trie_target_group_counts: Vec::new(),
            trie_target_group_bits: Vec::new(),
            trie_target_bits_enabled: *VOCAB_TRIE_TARGET_BITS_ENABLED,
            trie_target_position_counts: Vec::new(),
            targets: Vec::new(),
            target_gids: HashMap::new(),
            single_target_pos: usize::MAX,
            single_target_gids: SmallVec::new(),
            single_target_seen: vec![0; num_groups],
            single_target_seen_epoch: 1,
            dag_nodes: Vec::new(),
            dag_queue: Vec::new(),
            dag_disallowed: Vec::new(),
            dense_dag_disallowed: Vec::new(),
            dense_dag_disallowed_active: Vec::new(),
            single_target_disallowed: BitSet::new(num_groups),
            single_target_hash_pos: usize::MAX,
            single_target_hash: 0,
            suffix_match_positions: vec![NONE; num_groups],
            suffix_dirty_groups: SmallVec::new(),
        }
    }

    /// Reuse the allocation for a later state batch with the same vocabulary
    /// group axis. Token-signature cleanup leaves the active prefix clean.
    fn ensure_capacity(&mut self, num_states: usize, num_groups: usize) {
        let dirty_words = (num_groups + 63) / 64;
        if self.dirty_words != dirty_words || self.single_target_seen.len() != num_groups {
            *self = Self::new(num_states, num_groups);
            return;
        }
        if self.current_states.len() >= num_states {
            return;
        }
        self.current_states.resize(num_states, 0);
        self.match_positions.resize(num_states * num_groups, NONE);
        self.dirty_state_flags.resize(num_states, 0);
        self.dirty_state_bits.resize(num_states.div_ceil(64), 0);
        self.dirty_group_masks
            .resize(num_states * dirty_words.max(1), 0);
    }
}

#[inline(always)]
fn set_dirty_state_bit(bits: &mut [u64], state_idx: usize) {
    bits[state_idx / 64] |= 1u64 << (state_idx % 64);
}

#[inline(always)]
fn clear_dirty_state_bit(bits: &mut [u64], state_idx: usize) {
    bits[state_idx / 64] &= !(1u64 << (state_idx % 64));
}

fn ensure_completion_weights(scratch: &mut Scratch, num_states: usize) {
    if scratch.completion_weights.len() == num_states {
        return;
    }
    scratch.completion_weights.resize(num_states, 0);
    let mut weight = 1u64;
    for slot in scratch.completion_weights.iter_mut().rev() {
        *slot = weight;
        weight = weight.wrapping_mul(HASH_SEED1);
    }
}

fn reset_trie_target_aggregate(scratch: &mut Scratch, max_token_len: usize) {
    let positions = max_token_len + 1;
    let slots = positions.saturating_mul(scratch.num_groups);
    scratch.trie_target_group_counts.resize(slots, 0);
    scratch.trie_target_group_counts[..slots].fill(0);
    if scratch.trie_target_bits_enabled {
        let bit_slots = positions.saturating_mul(scratch.dirty_words);
        scratch.trie_target_group_bits.resize(bit_slots, 0);
        scratch.trie_target_group_bits[..bit_slots].fill(0);
    }
    scratch.trie_target_position_counts.resize(positions, 0);
    scratch.trie_target_position_counts[..positions].fill(0);
}

#[inline(always)]
fn update_trie_target_aggregate(
    scratch: &mut Scratch,
    gid: usize,
    previous_position: u32,
    next_position: u32,
) {
    if previous_position == next_position || gid >= scratch.num_groups {
        return;
    }

    let remove = |scratch: &mut Scratch, position: usize, gid: usize| {
        if position == 0 {
            return;
        }
        let index = position * scratch.num_groups + gid;
        debug_assert!(scratch.trie_target_group_counts[index] > 0);
        scratch.trie_target_group_counts[index] -= 1;
        if scratch.trie_target_group_counts[index] == 0 {
            debug_assert!(scratch.trie_target_position_counts[position] > 0);
            scratch.trie_target_position_counts[position] -= 1;
            if scratch.trie_target_bits_enabled {
                let bit_index = position * scratch.dirty_words + gid / 64;
                scratch.trie_target_group_bits[bit_index] &= !(1u64 << (gid % 64));
            }
        }
    };
    let add = |scratch: &mut Scratch, position: usize, gid: usize| {
        if position == 0 {
            return;
        }
        let index = position * scratch.num_groups + gid;
        if scratch.trie_target_group_counts[index] == 0 {
            scratch.trie_target_position_counts[position] += 1;
            if scratch.trie_target_bits_enabled {
                let bit_index = position * scratch.dirty_words + gid / 64;
                scratch.trie_target_group_bits[bit_index] |= 1u64 << (gid % 64);
            }
        }
        scratch.trie_target_group_counts[index] += 1;
    };

    if previous_position != NONE {
        remove(scratch, previous_position as usize, gid);
    }
    if next_position != NONE {
        add(scratch, next_position as usize, gid);
    }
}

/// Materialize the union of live `(position, group)` pairs maintained during a
/// trie walk. This is exactly what `collect_targets` obtains by scanning every
/// state-local dirty mask, but its work is proportional to live positions.
fn collect_trie_targets(scratch: &mut Scratch, token_len: usize, dense_suffix_dag: bool) {
    scratch.targets.clear();
    scratch.target_gids.clear();
    scratch.single_target_pos = usize::MAX;
    scratch.single_target_gids.clear();

    // Trie target positions are bounded by the current token length and are
    // already represented densely by the aggregate arrays. The dense suffix
    // DAG path consumes those arrays directly, so avoid materializing a
    // position -> gids HashMap for the overwhelmingly-common multi-target case.
    if dense_suffix_dag {
        for position in 1..=token_len {
            if scratch.trie_target_position_counts[position] != 0 {
                scratch.targets.push(position);
            }
        }
        if scratch.targets.len() == 1 {
            let position = scratch.targets[0];
            scratch.single_target_pos = position;
            if scratch.trie_target_bits_enabled {
                let bit_base = position * scratch.dirty_words;
                for word_index in 0..scratch.dirty_words {
                    let mut bits = scratch.trie_target_group_bits[bit_base + word_index];
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let gid = word_index * 64 + bit;
                        if gid < scratch.num_groups {
                            scratch.single_target_gids.push(gid);
                        }
                    }
                }
            } else {
                let base = position * scratch.num_groups;
                for gid in 0..scratch.num_groups {
                    if scratch.trie_target_group_counts[base + gid] != 0 {
                        scratch.single_target_gids.push(gid);
                    }
                }
            }
        }
        return;
    }

    let mut targets_with_gids: SmallVec<[(usize, SmallVec<[usize; 16]>); 4]> = SmallVec::new();
    for position in 1..=token_len {
        if scratch.trie_target_position_counts[position] == 0 {
            continue;
        }
        let mut gids = SmallVec::<[usize; 16]>::new();
        if scratch.trie_target_bits_enabled {
            let bit_base = position * scratch.dirty_words;
            for word_index in 0..scratch.dirty_words {
                let mut bits = scratch.trie_target_group_bits[bit_base + word_index];
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let gid = word_index * 64 + bit;
                    if gid < scratch.num_groups {
                        gids.push(gid);
                    }
                }
            }
        } else {
            let base = position * scratch.num_groups;
            for gid in 0..scratch.num_groups {
                if scratch.trie_target_group_counts[base + gid] != 0 {
                    gids.push(gid);
                }
            }
        }
        debug_assert!(!gids.is_empty());
        targets_with_gids.push((position, gids));
    }

    for (position, _) in &targets_with_gids {
        scratch.targets.push(*position);
    }
    if targets_with_gids.len() == 1 {
        let (position, gids) = targets_with_gids.pop().expect("single target exists");
        scratch.single_target_pos = position;
        scratch.single_target_gids = gids;
    } else {
        for (position, gids) in targets_with_gids {
            scratch.target_gids.insert(position, gids);
        }
    }
}

#[inline]
fn mark_dirty_group(scratch: &mut Scratch, state_idx: usize, gid: usize) {
    let word_idx = gid / 64;
    let bit = gid % 64;
    let flat_idx = state_idx * scratch.dirty_words + word_idx;
    scratch.dirty_state_flags[state_idx] = 1;
    scratch.dirty_group_masks[flat_idx] |= 1u64 << bit;
}

fn ensure_target_gids_map(
    target_gids: &mut HashMap<usize, SmallVec<[usize; 16]>>,
    single_target_pos: usize,
    single_target_gids: &[usize],
) {
    if target_gids.is_empty() && single_target_pos != usize::MAX {
        let mut gids = SmallVec::new();
        gids.extend(single_target_gids.iter().copied());
        target_gids.insert(single_target_pos, gids);
    }
}

fn advance_seen_epoch(seen: &mut [u32], epoch: &mut u32) {
    *epoch = epoch.wrapping_add(1);
    if *epoch == 0 {
        seen.fill(0);
        *epoch = 1;
    }
}

fn record_target_gid(
    targets: &mut Vec<usize>,
    target_gids: &mut HashMap<usize, SmallVec<[usize; 16]>>,
    single_target_pos: &mut usize,
    single_target_gids: &mut SmallVec<[usize; 16]>,
    single_target_seen: &mut [u32],
    single_target_seen_epoch: &mut u32,
    pos: usize,
    gid: usize,
) {
    if targets.is_empty() {
        targets.push(pos);
        *single_target_pos = pos;
        single_target_gids.clear();
        advance_seen_epoch(single_target_seen, single_target_seen_epoch);
        if gid < single_target_seen.len() {
            single_target_seen[gid] = *single_target_seen_epoch;
        }
        single_target_gids.push(gid);
        return;
    }

    if target_gids.is_empty() && targets.len() == 1 && *single_target_pos == pos {
        if gid < single_target_seen.len() {
            if single_target_seen[gid] != *single_target_seen_epoch {
                single_target_seen[gid] = *single_target_seen_epoch;
                single_target_gids.push(gid);
            }
        } else if !single_target_gids.contains(&gid) {
            single_target_gids.push(gid);
        }
        return;
    }

    ensure_target_gids_map(target_gids, *single_target_pos, single_target_gids.as_slice());

    let gids = target_gids.entry(pos).or_default();
    if gids.is_empty() {
        targets.push(pos);
    }
    if !gids.contains(&gid) {
        gids.push(gid);
    }
}

fn run_batch_inner(
    dfa: &Dfa,
    scratch: &mut Scratch,
    slice: &[u8],
    state_offset: usize,
    num_states: usize,
) {
    let num_groups = dfa.num_groups;
    let len = slice.len();
    let dirty_words = scratch.dirty_words;

    scratch.active_indices.clear();
    {
        let mask_start = state_offset * dirty_words;
        let mask_end = (state_offset + num_states) * dirty_words;
        for v in scratch.dirty_group_masks[mask_start..mask_end].iter_mut() {
            *v = 0;
        }
    }
    for flag in scratch.dirty_state_flags[state_offset..state_offset + num_states].iter_mut() {
        *flag = 0;
    }

    let has_bytes = !slice.is_empty();
    let first_byte = if has_bytes { slice[0] } else { 0 };

    // Initialize active states. Position-0 finalizer matches are NOT recorded
    // here because collect_targets filters them out (requires pv > 0) and
    // finish_token_signature skips them too. Omitting position-0 recording
    // avoids creating dirty-tracking entries for the ~97% of tokens that never
    // match at any position > 0, eliminating unnecessary bitmask
    // overhead in collect_targets and finish_token_signature.
    for i in state_offset..state_offset + num_states {
        let state = scratch.current_states[i];
        if dfa.is_dead_end[state] {
            scratch.current_states[i] = STATE_NONE;
            continue;
        }
        if has_bytes && dfa.transition(state, first_byte) == NONE {
            scratch.current_states[i] = STATE_NONE;
            continue;
        }
        scratch.active_indices.push(i);
    }

    // Walk each byte (hot path)
    if has_bytes && !scratch.active_indices.is_empty() {
        let mut active_len = scratch.active_indices.len();
        for (pos, &byte) in slice.iter().enumerate() {
            let position = (pos + 1) as u32;
            let mut next_len = 0usize;
            let class = dfa.byte_to_class[byte as usize] as usize;
            let class_base = class * dfa.num_states;
            for idx in 0..active_len {
                let i = scratch.active_indices[idx];
                let base = i * num_groups;
                let next_state = unsafe { *dfa.trans_by_class.get_unchecked(class_base + scratch.current_states[i]) };
                if next_state != NONE {
                    let ns = next_state as usize;
                    scratch.current_states[i] = ns;
                    for &gid in &dfa.finalizers[ns] {
                        if gid < num_groups {
                            let ix = base + gid;
                            if scratch.match_positions[ix] == NONE {
                                mark_dirty_group(scratch, i, gid);
                            }
                            scratch.match_positions[ix] = position;
                        }
                    }
                    if dfa.is_dead_end[ns] {
                        scratch.current_states[i] = STATE_NONE;
                    }
                } else {
                    scratch.current_states[i] = STATE_NONE;
                }
                if scratch.current_states[i] != STATE_NONE {
                    scratch.active_indices[next_len] = i;
                    next_len += 1;
                }
            }
            active_len = next_len;
            if active_len == 0 {
                break;
            }

            // Self-loop early exit: if all active states self-loop on every remaining byte,
            // greedy match positions advance to token_length and we can stop.
            if pos + 1 < len && active_len <= SELF_LOOP_ACTIVE_LEN_LIMIT {
                // Intersect self_loop_bytes for all active states
                let mut sl = U8Set::all();
                for idx in 0..active_len {
                    let i = scratch.active_indices[idx];
                    let s = scratch.current_states[i];
                    sl &= dfa.self_loop_bytes[s];
                }
                // Check if all remaining bytes are in the intersection
                let all_self_loop = slice[pos + 1..].iter().all(|&b| sl.contains(b));
                if all_self_loop {
                    let token_len = len as u32;
                    for idx in 0..active_len {
                        let i = scratch.active_indices[idx];
                        let base = i * num_groups;
                        let s = scratch.current_states[i];
                        for &gid in &dfa.finalizers[s] {
                            if gid < num_groups {
                                let ix = base + gid;
                                if scratch.match_positions[ix] == NONE {
                                    mark_dirty_group(scratch, i, gid);
                                }
                                scratch.match_positions[ix] = token_len;
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
}

fn collect_targets(
    scratch: &mut Scratch,
    num_groups: usize,
    len: usize,
    state_offset: usize,
    num_states: usize,
) {
    if num_groups == 0 {
        return;
    }

    let dirty_words = scratch.dirty_words;

    let Scratch {
        dirty_state_flags,
        dirty_group_masks,
        match_positions,
        targets,
        target_gids,
        single_target_pos,
        single_target_gids,
        single_target_seen,
        single_target_seen_epoch,
        ..
    } = scratch;

    for i in state_offset..state_offset + num_states {
        if dirty_state_flags[i] == 0 {
            continue;
        }
        let base = i * num_groups;
        let mask_base = i * dirty_words;
        for w in 0..dirty_words {
            let mut dirty_mask = dirty_group_masks[mask_base + w];
            while dirty_mask != 0 {
                let bit = dirty_mask.trailing_zeros() as usize;
                dirty_mask &= dirty_mask - 1;
                let gid = w * 64 + bit;
                if gid >= num_groups {
                    break;
                }
                let pv = match_positions[base + gid];
                if pv != NONE && pv > 0 && (pv as usize) <= len {
                    record_target_gid(
                        targets,
                        target_gids,
                        single_target_pos,
                        single_target_gids,
                        single_target_seen,
                        single_target_seen_epoch,
                        pv as usize,
                        gid,
                    );
                }
            }
        }
    }
}

/// Run DFA from all initial states on a token, recording end states and match positions.
/// Uses dirty bitmask tracking to avoid O(num_states * num_groups) memset.
/// INVARIANT: match_positions entries are NONE except for dirty entries from a previous
/// call that must have been cleaned up by the caller (token_signature does this).
fn run_batch(
    dfa: &Dfa,
    scratch: &mut Scratch,
    slice: &[u8],
    initial_states: &[usize],
    state_group_size: usize,
    _profile_signature_detail: bool,
) {
    let num_states = initial_states.len();
    let num_groups = dfa.num_groups;
    let len = slice.len();

    if num_states == 0 {
        scratch.targets.clear();
        return;
    }

    scratch.current_states[..num_states].clone_from_slice(initial_states);
    scratch.targets.clear();
    scratch.target_gids.clear();
    scratch.single_target_pos = usize::MAX;
    scratch.single_target_gids.clear();

    if state_group_size >= num_states {
        run_batch_inner(dfa, scratch, slice, 0, num_states);

        collect_targets(
            scratch,
            num_groups,
            len,
            0,
            num_states,
        );
    } else {
        for state_offset in (0..num_states).step_by(state_group_size) {
            let group_len = (state_offset + state_group_size).min(num_states) - state_offset;

            run_batch_inner(dfa, scratch, slice, state_offset, group_len);

            collect_targets(
                scratch,
                num_groups,
                len,
                state_offset,
                group_len,
            );
        }
    }
}

fn hash_suffixes(
    dfa: &Dfa,
    slice: &[u8],
    scratch: &mut Scratch,
    _profile_signature_detail: bool,
) -> usize {
    let len = slice.len();
    scratch.dag_nodes.clear();
    scratch.dag_queue.clear();
    scratch.dag_disallowed.clear();

    for (&pos, gids) in &scratch.target_gids {
        if let Some((&first_gid, rest)) = gids.split_first() {
            let mut combined = dfa.disallowed_for(first_gid).clone();
            for &gid in rest {
                intersect_disallowed(&mut combined, dfa.disallowed_for(gid));
            }
            ensure_position_slot(&mut scratch.dag_disallowed, pos);
            scratch.dag_disallowed[pos] = Some(combined);
        }
    }

    // BFS from target positions: run suffix DFA at each, discover new positions from edges
    for &pos in &scratch.targets {
        if pos < len {
            ensure_position_slot(&mut scratch.dag_nodes, pos);
            if scratch.dag_nodes[pos].is_some() {
                continue;
            }
            scratch.dag_queue.push(pos);
            scratch.dag_nodes[pos] = Some(DagNode {
                hash: 0,
                edges: EdgeList::new(),
                end_state: STATE_NONE,
            });
        }
    }

    let mut cursor = 0;
    while cursor < scratch.dag_queue.len() {
        let pos = scratch.dag_queue[cursor];
        cursor += 1;
        let (end_state, edges) = run_suffix(
            dfa,
            &slice[pos..],
            pos,
            &mut scratch.suffix_match_positions,
            &mut scratch.suffix_dirty_groups,
        );
        for &(_, target) in &edges {
            if target < len {
                ensure_position_slot(&mut scratch.dag_nodes, target);
                if scratch.dag_nodes[target].is_some() {
                    continue;
                }
                scratch.dag_queue.push(target);
                scratch.dag_nodes[target] = Some(DagNode {
                    hash: 0,
                    edges: EdgeList::new(),
                    end_state: STATE_NONE,
                });
            }
        }
        scratch.dag_nodes[pos] = Some(DagNode {
            hash: 0,
            edges,
            end_state: end_state.unwrap_or(STATE_NONE),
        });
    }

    scratch.dag_queue.sort_unstable();
    for idx in 0..scratch.dag_queue.len() {
        let pos = scratch.dag_queue[idx];
        let edges = scratch.dag_nodes[pos].as_ref().unwrap().edges.clone();

        for &(gid, target) in &edges {
            if node_disallows_gid(scratch, pos, gid) {
                continue;
            }
            if target < len {
                intersect_node_disallowed(scratch, target, dfa.disallowed_for(gid));
            }
        }
    }

    // Hash bottom-up: process deeper positions first. Reuse the ascending
    // topological order from propagation and walk it in reverse instead of
    // paying for a second sort.
    for idx in (0..scratch.dag_queue.len()).rev() {
        let pos = scratch.dag_queue[idx];
        let node = scratch.dag_nodes[pos].as_ref().unwrap();
        let edges = node.edges.clone();
        let end_state = node.end_state;
        let mut h = new_hasher();
        h.write_u64(dfa.completion_with_disallowed(
            end_state,
            scratch.dag_disallowed.get(pos).and_then(|bits| bits.as_ref()),
        ));

        for &(gid, target) in &edges {
            if node_disallows_gid(scratch, pos, gid) {
                continue;
            }
            h.write_u64(gid as u64);
            h.write_u64(
                scratch
                    .dag_nodes
                    .get(target)
                    .and_then(|node| node.as_ref())
                    .map_or(0, |node| node.hash),
            );
        }
        scratch.dag_nodes[pos].as_mut().unwrap().hash = h.finish();
    }

    scratch.dag_queue.len()
}

fn trie_target_disallowed_at(
    dfa: &Dfa,
    scratch: &Scratch,
    position: usize,
) -> Option<BitSet> {
    let mut combined = None::<BitSet>;
    if scratch.trie_target_bits_enabled {
        let bit_base = position * scratch.dirty_words;
        for word_index in 0..scratch.dirty_words {
            let mut bits = scratch.trie_target_group_bits[bit_base + word_index];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let gid = word_index * 64 + bit;
                if gid >= scratch.num_groups {
                    continue;
                }
                if let Some(current) = combined.as_mut() {
                    intersect_disallowed(current, dfa.disallowed_for(gid));
                } else {
                    combined = Some(dfa.disallowed_for(gid).clone());
                }
            }
        }
    } else {
        let base = position * scratch.num_groups;
        for gid in 0..scratch.num_groups {
            if scratch.trie_target_group_counts[base + gid] == 0 {
                continue;
            }
            if let Some(current) = combined.as_mut() {
                intersect_disallowed(current, dfa.disallowed_for(gid));
            } else {
                combined = Some(dfa.disallowed_for(gid).clone());
            }
        }
    }
    combined
}

fn fill_trie_target_disallowed_at(
    dfa: &Dfa,
    trie_target_bits_enabled: bool,
    dirty_words: usize,
    num_groups: usize,
    trie_target_group_bits: &[u64],
    trie_target_group_counts: &[u32],
    position: usize,
    out: &mut BitSet,
) -> bool {
    out.clear_all();
    let mut initialized = false;
    let mut observe_gid = |gid: usize| {
        if initialized {
            out.intersect_with(dfa.disallowed_for(gid));
        } else {
            out.union_with(dfa.disallowed_for(gid));
            initialized = true;
        }
    };

    if trie_target_bits_enabled {
        let bit_base = position * dirty_words;
        for word_index in 0..dirty_words {
            let mut bits = trie_target_group_bits[bit_base + word_index];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let gid = word_index * 64 + bit;
                if gid < num_groups {
                    observe_gid(gid);
                }
            }
        }
    } else {
        let base = position * num_groups;
        for gid in 0..num_groups {
            if trie_target_group_counts[base + gid] != 0 {
                observe_gid(gid);
            }
        }
    }
    initialized
}

fn ensure_dense_disallowed_slots(scratch: &mut Scratch, len: usize) {
    if scratch.dense_dag_disallowed.len() < len + 1 {
        scratch
            .dense_dag_disallowed
            .resize_with(len + 1, || BitSet::new(scratch.num_groups));
        scratch.dense_dag_disallowed_active.resize(len + 1, false);
    }
    scratch.dense_dag_disallowed_active[..len + 1].fill(false);
}

#[inline]
fn dense_disallowed_ref(scratch: &Scratch, pos: usize) -> Option<&BitSet> {
    scratch
        .dense_dag_disallowed_active
        .get(pos)
        .copied()
        .unwrap_or(false)
        .then(|| &scratch.dense_dag_disallowed[pos])
}

#[inline]
fn dense_node_disallows_gid(scratch: &Scratch, pos: usize, gid: usize) -> bool {
    dense_disallowed_ref(scratch, pos)
        .map(|bits| bits.contains(gid))
        .unwrap_or(false)
}

fn dense_intersect_node_disallowed(scratch: &mut Scratch, pos: usize, incoming: &BitSet) {
    if scratch.dense_dag_disallowed_active[pos] {
        scratch.dense_dag_disallowed[pos].intersect_with(incoming);
    } else {
        let slot = &mut scratch.dense_dag_disallowed[pos];
        slot.clear_all();
        slot.union_with(incoming);
        scratch.dense_dag_disallowed_active[pos] = true;
    }
}

/// Hash suffix observations using the token-position axis directly.
///
/// During a token suffix walk every emitted edge advances to a strictly later
/// byte position. Token positions are therefore already a topological order;
/// for the short finite-vocabulary tokens used by this path, a dense scan over
/// `1..token.len()` replaces the HashMap materialization, BFS queue, and sort in
/// the generic suffix DAG without changing the graph or its bottom-up hashes.
struct PrecomputedDenseSuffixRoot {
    pos: usize,
    end_state: usize,
    edges: EdgeList,
    disallowed: BitSet,
}

fn hash_trie_suffixes_dense(
    dfa: &Dfa,
    slice: &[u8],
    scratch: &mut Scratch,
    precomputed_root: Option<PrecomputedDenseSuffixRoot>,
) -> usize {
    hash_trie_suffixes_dense_impl(
        dfa,
        slice,
        scratch,
        precomputed_root,
        super::super::super::vocab_reuse_dense_disallowed_slots_enabled(),
    )
}

fn hash_trie_suffixes_dense_impl(
    dfa: &Dfa,
    slice: &[u8],
    scratch: &mut Scratch,
    precomputed_root: Option<PrecomputedDenseSuffixRoot>,
    reuse_disallowed_slots: bool,
) -> usize {
    let len = slice.len();
    scratch.dag_nodes.clear();
    scratch.dag_nodes.resize_with(len + 1, || None);
    if reuse_disallowed_slots {
        ensure_dense_disallowed_slots(scratch, len);
    } else {
        scratch.dag_disallowed.clear();
        scratch.dag_disallowed.resize_with(len + 1, || None);
    }

    let mut precomputed_root_pos = None;
    if let Some(root) = precomputed_root {
        if root.pos < len {
            precomputed_root_pos = Some(root.pos);
            if reuse_disallowed_slots {
                let slot = &mut scratch.dense_dag_disallowed[root.pos];
                slot.clear_all();
                slot.union_with(&root.disallowed);
                scratch.dense_dag_disallowed_active[root.pos] = true;
            } else {
                scratch.dag_disallowed[root.pos] = Some(root.disallowed);
            }
            scratch.dag_nodes[root.pos] = Some(DagNode {
                hash: 0,
                edges: root.edges,
                end_state: root.end_state,
            });
        }
    }

    for &pos in &scratch.targets {
        if pos >= len {
            continue;
        }
        if precomputed_root_pos == Some(pos) {
            continue;
        }
        if reuse_disallowed_slots {
            let initialized = fill_trie_target_disallowed_at(
                dfa,
                scratch.trie_target_bits_enabled,
                scratch.dirty_words,
                scratch.num_groups,
                &scratch.trie_target_group_bits,
                &scratch.trie_target_group_counts,
                pos,
                &mut scratch.dense_dag_disallowed[pos],
            );
            scratch.dense_dag_disallowed_active[pos] = initialized;
        } else {
            scratch.dag_disallowed[pos] = trie_target_disallowed_at(dfa, scratch, pos);
        }
        scratch.dag_nodes[pos] = Some(DagNode {
            hash: 0,
            edges: EdgeList::new(),
            end_state: STATE_NONE,
        });
    }

    let mut node_count = 0usize;
    for pos in 1..len {
        if scratch.dag_nodes[pos].is_none() {
            continue;
        }
        node_count += 1;
        let (end_state, edges) = if precomputed_root_pos == Some(pos) {
            let node = scratch.dag_nodes[pos]
                .as_ref()
                .expect("precomputed dense suffix root was initialized");
            (
                (node.end_state != STATE_NONE).then_some(node.end_state),
                node.edges.clone(),
            )
        } else {
            run_suffix(
                dfa,
                &slice[pos..],
                pos,
                &mut scratch.suffix_match_positions,
                &mut scratch.suffix_dirty_groups,
            )
        };
        for &(_, target) in &edges {
            if target < len && scratch.dag_nodes[target].is_none() {
                scratch.dag_nodes[target] = Some(DagNode {
                    hash: 0,
                    edges: EdgeList::new(),
                    end_state: STATE_NONE,
                });
            }
        }
        if precomputed_root_pos != Some(pos) {
            scratch.dag_nodes[pos] = Some(DagNode {
                hash: 0,
                edges,
                end_state: end_state.unwrap_or(STATE_NONE),
            });
        }
    }

    for pos in 1..len {
        let Some(node) = scratch.dag_nodes[pos].as_ref() else {
            continue;
        };
        let edges = node.edges.clone();
        for &(gid, target) in &edges {
            let disallows = if reuse_disallowed_slots {
                dense_node_disallows_gid(scratch, pos, gid)
            } else {
                node_disallows_gid(scratch, pos, gid)
            };
            if disallows {
                continue;
            }
            if target < len {
                if reuse_disallowed_slots {
                    dense_intersect_node_disallowed(scratch, target, dfa.disallowed_for(gid));
                } else {
                    intersect_node_disallowed(scratch, target, dfa.disallowed_for(gid));
                }
            }
        }
    }

    for pos in (1..len).rev() {
        let Some(node) = scratch.dag_nodes[pos].as_ref() else {
            continue;
        };
        let end_state = node.end_state;
        let edges = node.edges.clone();
        let mut h = new_hasher();
        h.write_u64(dfa.completion_with_disallowed(
            end_state,
            if reuse_disallowed_slots {
                dense_disallowed_ref(scratch, pos)
            } else {
                scratch.dag_disallowed[pos].as_ref()
            },
        ));
        for &(gid, target) in &edges {
            let disallows = if reuse_disallowed_slots {
                dense_node_disallows_gid(scratch, pos, gid)
            } else {
                node_disallows_gid(scratch, pos, gid)
            };
            if disallows {
                continue;
            }
            h.write_u64(gid as u64);
            h.write_u64(
                scratch
                    .dag_nodes
                    .get(target)
                    .and_then(|node| node.as_ref())
                    .map_or(0, |node| node.hash),
            );
        }
        scratch.dag_nodes[pos].as_mut().unwrap().hash = h.finish();
    }

    node_count
}

fn hash_single_target_suffix_dense(dfa: &Dfa, slice: &[u8], scratch: &mut Scratch) -> usize {
    hash_single_target_suffix_dense_impl(
        dfa,
        slice,
        scratch,
        super::super::super::vocab_reuse_single_target_disallowed_enabled(),
    )
}

fn hash_single_target_suffix_dense_impl(
    dfa: &Dfa,
    slice: &[u8],
    scratch: &mut Scratch,
    reuse_root_disallowed: bool,
) -> usize {
    let pos = scratch.single_target_pos;
    if pos == usize::MAX {
        return 0;
    }
    let len = slice.len();

    if pos >= len {
        scratch.single_target_hash_pos = pos;
        scratch.single_target_hash = 0;
        return 0;
    }

    let Some((&first_gid, rest)) = scratch.single_target_gids.split_first() else {
        return 0;
    };
    let root_disallowed_owned = if reuse_root_disallowed {
        scratch.single_target_disallowed.clear_all();
        scratch
            .single_target_disallowed
            .union_with(dfa.disallowed_for(first_gid));
        for &gid in rest {
            scratch
                .single_target_disallowed
                .intersect_with(dfa.disallowed_for(gid));
        }
        None
    } else {
        let mut root_disallowed = dfa.disallowed_for(first_gid).clone();
        for &gid in rest {
            intersect_disallowed(&mut root_disallowed, dfa.disallowed_for(gid));
        }
        Some(root_disallowed)
    };

    let (end_state, edges) = run_suffix(
        dfa,
        &slice[pos..],
        pos,
        &mut scratch.suffix_match_positions,
        &mut scratch.suffix_dirty_groups,
    );
    if edges.iter().any(|&(_, target)| target < len) {
        let root_disallowed = root_disallowed_owned
            .unwrap_or_else(|| scratch.single_target_disallowed.clone());
        return hash_trie_suffixes_dense(
            dfa,
            slice,
            scratch,
            Some(PrecomputedDenseSuffixRoot {
                pos,
                end_state: end_state.unwrap_or(STATE_NONE),
                edges,
                disallowed: root_disallowed,
            }),
        );
    }

    let root_disallowed = root_disallowed_owned
        .as_ref()
        .unwrap_or(&scratch.single_target_disallowed);

    let mut h = new_hasher();
    h.write_u64(dfa.completion_with_disallowed(
        end_state.unwrap_or(STATE_NONE),
        Some(&root_disallowed),
    ));
    for &(gid, _target) in &edges {
        if root_disallowed.contains(gid) {
            continue;
        }
        h.write_u64(gid as u64);
        h.write_u64(0);
    }

    scratch.single_target_hash_pos = pos;
    scratch.single_target_hash = h.finish();
    1
}

/// Run DFA on a suffix from start_state, returning (end_state, edges to match positions).
fn run_suffix(
    dfa: &Dfa,
    slice: &[u8],
    base_pos: usize,
    match_positions: &mut [u32],
    dirty_groups: &mut SmallVec<[usize; 16]>,
) -> (Option<usize>, EdgeList) {
    let num_groups = dfa.num_groups;
    dirty_groups.clear();
    let mut current = dfa.start_state;
    let mut done = dfa.is_dead_end[current];

    for &gid in &dfa.finalizers[current] {
        if gid < num_groups && match_positions[gid] == NONE {
            dirty_groups.push(gid);
            match_positions[gid] = 0;
        }
    }

    for (idx, &byte) in slice.iter().enumerate() {
        if done {
            break;
        }
        let ns = dfa.transition(current, byte);
        if ns == NONE {
            done = true;
            break;
        }
        current = ns as usize;
        let position = (idx + 1) as u32;
        for &gid in &dfa.finalizers[current] {
            if gid < num_groups {
                if match_positions[gid] == NONE {
                    dirty_groups.push(gid);
                }
                match_positions[gid] = position;
            }
        }
        if dfa.is_dead_end[current] {
            done = true;
        }
    }

    let end_state = if done { None } else { Some(current) };
    let edges: EdgeList = dirty_groups
        .iter()
        .filter_map(|&gid| {
            let pv = match_positions[gid];
            (pv != NONE && pv != 0).then(|| (gid, base_pos + pv as usize))
        })
        .collect();
    for &gid in dirty_groups.iter() {
        match_positions[gid] = NONE;
    }
    (end_state, edges)
}

fn try_hash_single_target_suffix(
    dfa: &Dfa,
    slice: &[u8],
    scratch: &mut Scratch,
) -> Option<usize> {
    let pos = scratch.single_target_pos;
    if pos == usize::MAX {
        return None;
    }
    let len = slice.len();

    if pos >= len {
        scratch.single_target_hash_pos = pos;
        scratch.single_target_hash = 0;
        return Some(0);
    }

    let (&first_gid, rest) = scratch.single_target_gids.split_first()?;
    let mut root_disallowed = dfa.disallowed_for(first_gid).clone();
    for &gid in rest {
        intersect_disallowed(&mut root_disallowed, dfa.disallowed_for(gid));
    }

    let (end_state, edges) = run_suffix(
        dfa,
        &slice[pos..],
        pos,
        &mut scratch.suffix_match_positions,
        &mut scratch.suffix_dirty_groups,
    );
    if edges.iter().any(|&(_, target)| target < len) {
        return None;
    }

    let mut h = new_hasher();
    h.write_u64(dfa.completion_with_disallowed(
        end_state.unwrap_or(STATE_NONE),
        Some(&root_disallowed),
    ));
    for &(gid, _target) in &edges {
        if root_disallowed.contains(gid) {
            continue;
        }
        h.write_u64(gid as u64);
        h.write_u64(0);
    }

    scratch.single_target_hash_pos = pos;
    scratch.single_target_hash = h.finish();
    Some(1)
}

/// Compute a token's full signature over a batch of initial states.
/// Also cleans up match_positions for dirty groups (maintaining the NONE invariant).
fn finish_token_signature(
    dfa: &Dfa,
    chunk_states: &[usize],
    scratch: &mut Scratch,
) -> u64 {
    let num_groups = dfa.num_groups;
    let dirty_words = scratch.dirty_words;
    let dag = &scratch.dag_nodes;
    let single_target_hash_pos = scratch.single_target_hash_pos;
    let single_target_hash = scratch.single_target_hash;
    let mut sig: u64 = HASH_SEED3;
    for i in 0..chunk_states.len() {
        let completion = dfa.completion(scratch.current_states[i]);
        let base = i * num_groups;
        let mask_base = i * dirty_words;

        let state_sig = if scratch.dirty_state_flags[i] != 0 {
            let mut h = new_hasher();
            h.write_u64(completion);
            for w in 0..dirty_words {
                let mut dirty_mask = scratch.dirty_group_masks[mask_base + w];
                while dirty_mask != 0 {
                    let bit = dirty_mask.trailing_zeros() as usize;
                    dirty_mask &= dirty_mask - 1;
                    let gid = w * 64 + bit;
                    if gid >= num_groups {
                        break;
                    }
                    let pv = scratch.match_positions[base + gid];
                    if pv != NONE && pv > 0 {
                        h.write_u64(gid as u64);
                        let target = pv as usize;
                        let target_hash = if single_target_hash_pos == target {
                            single_target_hash
                        } else {
                            dag.get(target)
                                .and_then(|node| node.as_ref())
                                .map_or(0, |node| node.hash)
                        };
                        h.write_u64(target_hash);
                    }
                    scratch.match_positions[base + gid] = NONE;
                }
                scratch.dirty_group_masks[mask_base + w] = 0;
            }
            scratch.dirty_state_flags[i] = 0;
            h.finish()
        } else {
            completion
        };

        sig = sig.wrapping_mul(HASH_SEED1).wrapping_add(state_sig);
    }
    sig
}

fn fill_state_observation_words_and_cleanup(
    dfa: &Dfa,
    batch_len: usize,
    scratch: &mut Scratch,
    out: &mut [u64],
) {
    let num_groups = dfa.num_groups;
    let dirty_words = scratch.dirty_words;
    let dag = &scratch.dag_nodes;
    let single_target_hash_pos = scratch.single_target_hash_pos;
    let single_target_hash = scratch.single_target_hash;

    for i in 0..batch_len {
        let completion = dfa.completion(scratch.current_states[i]);
        let base = i * num_groups;
        let mask_base = i * dirty_words;

        let state_sig = if scratch.dirty_state_flags[i] != 0 {
            let mut h = new_hasher();
            h.write_u64(completion);
            for w in 0..dirty_words {
                let mut dirty_mask = scratch.dirty_group_masks[mask_base + w];
                while dirty_mask != 0 {
                    let bit = dirty_mask.trailing_zeros() as usize;
                    dirty_mask &= dirty_mask - 1;
                    let gid = w * 64 + bit;
                    if gid >= num_groups {
                        break;
                    }
                    let pv = scratch.match_positions[base + gid];
                    if pv != NONE && pv > 0 {
                        h.write_u64(gid as u64);
                        let target = pv as usize;
                        let target_hash = if single_target_hash_pos == target {
                            single_target_hash
                        } else {
                            dag.get(target)
                                .and_then(|node| node.as_ref())
                                .map_or(0, |node| node.hash)
                        };
                        h.write_u64(target_hash);
                    }
                    scratch.match_positions[base + gid] = NONE;
                }
                scratch.dirty_group_masks[mask_base + w] = 0;
            }
            scratch.dirty_state_flags[i] = 0;
            h.finish()
        } else {
            completion
        };

        out[i] = state_sig;
    }
}

fn compute_token_state_observation_words(
    dfa: &Dfa,
    token: &[u8],
    batch: &[usize],
    state_group_size: usize,
    scratch: &mut Scratch,
    out: &mut [u64],
) {
    scratch.single_target_hash_pos = usize::MAX;
    scratch.single_target_hash = 0;
    run_batch(dfa, scratch, token, batch, state_group_size, false);

    let target_count = scratch.targets.len();
    if target_count == 1 {
        if try_hash_single_target_suffix(dfa, token, scratch).is_none() {
            ensure_target_gids_map(
                &mut scratch.target_gids,
                scratch.single_target_pos,
                scratch.single_target_gids.as_slice(),
            );
            hash_suffixes(dfa, token, scratch, false);
        }
    } else if target_count > 0 {
        hash_suffixes(dfa, token, scratch, false);
    }

    fill_state_observation_words_and_cleanup(dfa, batch.len(), scratch, out);
}

fn first_distinguishing_state_for_token_pair_with_count<S: AsRef<[u8]>>(
    dfa: &Dfa,
    strings: &[S],
    left_token_idx: usize,
    right_token_idx: usize,
    ordered_states: &[usize],
    ordered_original_states: &[usize],
    batch_size: usize,
) -> (Option<usize>, usize) {
    if left_token_idx >= strings.len() || right_token_idx >= strings.len() {
        return (None, 0);
    }

    let left_token = strings[left_token_idx].as_ref();
    let right_token = strings[right_token_idx].as_ref();
    let mut states_checked = 0usize;
    let mut left_words = vec![0u64; batch_size.max(1)];
    let mut right_words = vec![0u64; batch_size.max(1)];

    for batch_start in (0..ordered_states.len()).step_by(batch_size.max(1)) {
        let batch_end = (batch_start + batch_size.max(1)).min(ordered_states.len());
        let batch = &ordered_states[batch_start..batch_end];
        let batch_len = batch.len();
        let state_group_size = vocab_state_group_size(batch_len, dfa.num_groups);
        let mut left_scratch = Scratch::new(batch_len, dfa.num_groups);
        let mut right_scratch = Scratch::new(batch_len, dfa.num_groups);

        compute_token_state_observation_words(
            dfa,
            left_token,
            batch,
            state_group_size,
            &mut left_scratch,
            &mut left_words[..batch_len],
        );
        compute_token_state_observation_words(
            dfa,
            right_token,
            batch,
            state_group_size,
            &mut right_scratch,
            &mut right_words[..batch_len],
        );

        for i in 0..batch_len {
            states_checked += 1;
            if left_words[i] != right_words[i] {
                return (Some(ordered_original_states[batch_start + i]), states_checked);
            }
        }
    }

    (None, states_checked)
}

fn first_distinguishing_state_for_token_pair<S: AsRef<[u8]>>(
    dfa: &Dfa,
    strings: &[S],
    left_token_idx: usize,
    right_token_idx: usize,
    ordered_states: &[usize],
    ordered_original_states: &[usize],
    batch_size: usize,
) -> Option<usize> {
    first_distinguishing_state_for_token_pair_with_count(
        dfa,
        strings,
        left_token_idx,
        right_token_idx,
        ordered_states,
        ordered_original_states,
        batch_size,
    )
    .0
}

fn log_vocab_pair_verification<S: AsRef<[u8]>>(
    dfa: &Dfa,
    strings: &[S],
    left_token_idx: usize,
    right_token_idx: usize,
    ordered_states: &[usize],
    ordered_original_states: &[usize],
    batch_size: usize,
) {
    if left_token_idx >= strings.len() || right_token_idx >= strings.len() {
        eprintln!(
            "[glrmask/profile][vocab_pair_verify] left={} right={} result=invalid states={} total_ms=0.000",
            left_token_idx,
            right_token_idx,
            ordered_states.len(),
        );
        return;
    }

    let started_at = std::time::Instant::now();
    let (witness_state, states_checked) = first_distinguishing_state_for_token_pair_with_count(
        dfa,
        strings,
        left_token_idx,
        right_token_idx,
        ordered_states,
        ordered_original_states,
        batch_size,
    );

    if let Some(witness_state) = witness_state {
        eprintln!(
            "[glrmask/profile][vocab_pair_verify] left={} right={} result=different witness_state={} states_checked={} total_ms={:.3}",
            left_token_idx,
            right_token_idx,
            witness_state,
            states_checked,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    } else {
        eprintln!(
            "[glrmask/profile][vocab_pair_verify] left={} right={} result=equivalent states={} total_ms={:.3}",
            left_token_idx,
            right_token_idx,
            ordered_states.len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
}

fn run_vocab_row_cert_diag<S: AsRef<[u8]> + Sync>(
    dfa: &Dfa,
    strings: &[S],
    ordered_states: &[usize],
    batch_size: usize,
    state_group_size_for_batch: impl Fn(usize) -> usize,
    groups: &[Vec<usize>],
) {
    let started_at = std::time::Instant::now();
    let rep_token_indices: Vec<usize> = groups.iter().map(|group| group[0]).collect();
    let mut row_keys = vec![Vec::<u64>::with_capacity(rep_token_indices.len()); ordered_states.len()];

    for batch_start in (0..ordered_states.len()).step_by(batch_size.max(1)) {
        let batch_end = (batch_start + batch_size.max(1)).min(ordered_states.len());
        let batch = &ordered_states[batch_start..batch_end];
        let batch_len = batch.len();
        let state_group_size = state_group_size_for_batch(batch_len);
        let mut scratch = Scratch::new(batch_len, dfa.num_groups);
        let mut state_words = vec![0u64; batch_len];

        for &token_idx in &rep_token_indices {
            let token = strings[token_idx].as_ref();
            compute_token_state_observation_words(
                dfa,
                token,
                batch,
                state_group_size,
                &mut scratch,
                &mut state_words,
            );
            for i in 0..batch_len {
                row_keys[batch_start + i].push(state_words[i]);
            }
        }
    }

    let mut row_class_counts: HashMap<Vec<u64>, usize> =
        HashMap::with_capacity(row_keys.len() / 2);
    for row in row_keys {
        *row_class_counts.entry(row).or_insert(0) += 1;
    }

    let row_classes = row_class_counts.len();
    let largest_block = row_class_counts.values().copied().max().unwrap_or(0);
    let singleton_rows = row_class_counts
        .values()
        .copied()
        .filter(|&count| count == 1)
        .count();
    let reduction_pct = if ordered_states.is_empty() {
        0.0
    } else {
        100.0 * (1.0 - row_classes as f64 / ordered_states.len() as f64)
    };

    eprintln!(
        "[glrmask/profile][vocab_row_cert_diag] states={} rep_tokens={} row_classes={} reduction_pct={:.2} largest_block={} singleton_rows={} total_ms={:.3}",
        ordered_states.len(),
        rep_token_indices.len(),
        row_classes,
        reduction_pct,
        largest_block,
        singleton_rows,
        started_at.elapsed().as_secs_f64() * 1000.0,
    );
}

fn token_signature(
    dfa: &Dfa,
    token: &[u8],
    chunk_states: &[usize],
    state_group_size: usize,
    scratch: &mut Scratch,
    _profile_signature_detail: bool,
) -> u64 {
    scratch.single_target_hash_pos = usize::MAX;
    scratch.single_target_hash = 0;
    run_batch(
        dfa,
        scratch,
        token,
        chunk_states,
        state_group_size,
        false,
    );
    let target_count = scratch.targets.len();
    if target_count == 1 {
        if let Some(dag_nodes) = try_hash_single_target_suffix(dfa, token, scratch) {
            let _ = dag_nodes;
        } else {
            ensure_target_gids_map(
                &mut scratch.target_gids,
                scratch.single_target_pos,
                scratch.single_target_gids.as_slice(),
            );
            hash_suffixes(dfa, token, scratch, false);
        }
    } else if target_count > 0 {
        hash_suffixes(dfa, token, scratch, false);
    }

    finish_token_signature(dfa, chunk_states, scratch)
}

// ----- DFS Trie Walk for Prefix Sharing -----

const TRIE_CHUNK_SIZE: usize = 128;
const TRIE_WALK_MIN_TOKENS: usize = 256;

static TRIE_WALK_DISABLED: Lazy<bool> =
    Lazy::new(|| env_flag_enabled("GLRMASK_DISABLE_TRIE_WALK"));

struct DepthChangeLog {
    /// Batch indices whose DFA paths remain live after this depth.
    active_indices: Vec<usize>,
    /// (state_idx, old_state_value)
    state_changes: Vec<(usize, usize)>,
    /// (match_positions_flat_idx, old_value)
    match_changes: Vec<(usize, u32)>,
    /// (state_idx, old_dirty_mask)
    dirty_changes: Vec<(usize, u64)>,
    /// (state_idx, old_dirty_state_flag)
    dirty_state_flag_changes: Vec<(usize, u8)>,
}

impl DepthChangeLog {
    fn new() -> Self {
        Self {
            active_indices: Vec::new(),
            state_changes: Vec::new(),
            match_changes: Vec::new(),
            dirty_changes: Vec::new(),
            dirty_state_flag_changes: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.active_indices.clear();
        self.state_changes.clear();
        self.match_changes.clear();
        self.dirty_changes.clear();
        self.dirty_state_flag_changes.clear();
    }
}

struct TrieWalkState {
    root_active_indices: Vec<usize>,
    depth_logs: Vec<DepthChangeLog>,
}

impl TrieWalkState {
    fn new() -> Self {
        Self {
            root_active_indices: Vec::new(),
            depth_logs: Vec::new(),
        }
    }

    fn ensure_depth(&mut self, depth: usize) {
        while self.depth_logs.len() <= depth {
            self.depth_logs.push(DepthChangeLog::new());
        }
    }
}

struct ScratchWorker {
    scratch: Scratch,
    trie_state: TrieWalkState,
}

#[derive(Default)]
struct ScratchPool {
    available: Mutex<Vec<ScratchWorker>>,
    allocations: AtomicUsize,
    reuses: AtomicUsize,
}

impl ScratchPool {
    fn checkout(&self, num_states: usize, num_groups: usize) -> ScratchLease<'_> {
        let worker = self.available.lock().unwrap().pop();
        let mut worker = match worker {
            Some(worker) => {
                self.reuses.fetch_add(1, Ordering::Relaxed);
                worker
            }
            None => {
                self.allocations.fetch_add(1, Ordering::Relaxed);
                ScratchWorker {
                    scratch: Scratch::new(num_states, num_groups),
                    trie_state: TrieWalkState::new(),
                }
            }
        };
        worker.scratch.ensure_capacity(num_states, num_groups);
        ScratchLease {
            pool: self,
            worker: Some(worker),
        }
    }

    fn stats(&self) -> (usize, usize) {
        (
            self.allocations.load(Ordering::Relaxed),
            self.reuses.load(Ordering::Relaxed),
        )
    }
}

struct ScratchLease<'a> {
    pool: &'a ScratchPool,
    worker: Option<ScratchWorker>,
}

impl ScratchLease<'_> {
    fn worker_mut(&mut self) -> &mut ScratchWorker {
        self.worker.as_mut().expect("scratch lease must hold its worker")
    }
}

impl Drop for ScratchLease<'_> {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            self.pool.available.lock().unwrap().push(worker);
        }
    }
}

#[derive(Clone, Copy, Default)]
struct TrieWalkChunkStats {
    dfs_step_ms: f64,
    collect_targets_ms: f64,
    single_target_suffix_ms: f64,
    multi_target_suffix_ms: f64,
    finish_signature_ms: f64,
    dfs_steps: usize,
    dfs_steps_without_new_dirty: usize,
    dfs_states_visited: usize,
    dfs_dead_transitions: usize,
    dfs_dead_without_new_dirty: usize,
    dfs_new_dirty_groups: usize,
    dfs_new_dirty_states: usize,
    dfs_noop_self_loops: usize,
    clean_tokens: usize,
    dirty_tokens: usize,
    single_target_tokens: usize,
    multi_target_tokens: usize,
    total_targets: usize,
}

impl TrieWalkChunkStats {
    fn add_assign(&mut self, other: Self) {
        self.dfs_step_ms += other.dfs_step_ms;
        self.collect_targets_ms += other.collect_targets_ms;
        self.single_target_suffix_ms += other.single_target_suffix_ms;
        self.multi_target_suffix_ms += other.multi_target_suffix_ms;
        self.finish_signature_ms += other.finish_signature_ms;
        self.dfs_steps += other.dfs_steps;
        self.dfs_steps_without_new_dirty += other.dfs_steps_without_new_dirty;
        self.dfs_states_visited += other.dfs_states_visited;
        self.dfs_dead_transitions += other.dfs_dead_transitions;
        self.dfs_dead_without_new_dirty += other.dfs_dead_without_new_dirty;
        self.dfs_new_dirty_groups += other.dfs_new_dirty_groups;
        self.dfs_new_dirty_states += other.dfs_new_dirty_states;
        self.dfs_noop_self_loops += other.dfs_noop_self_loops;
        self.clean_tokens += other.clean_tokens;
        self.dirty_tokens += other.dirty_tokens;
        self.single_target_tokens += other.single_target_tokens;
        self.multi_target_tokens += other.multi_target_tokens;
        self.total_targets += other.total_targets;
    }
}

#[derive(Clone, Copy, Default)]
struct DfsStepStats {
    states_visited: usize,
    dead_transitions: usize,
    dead_without_new_dirty: usize,
    new_dirty_groups: usize,
    new_dirty_states: usize,
    noop_self_loops: usize,
}

/// Walk one byte forward at the given depth, recording all state changes
/// to the change log for later backtracking.
fn dfs_step(
    dfa: &Dfa,
    scratch: &mut Scratch,
    trie: &mut TrieWalkState,
    byte: u8,
    depth: usize,
    position_offset: usize,
    batch_len: usize,
) {
    trie.ensure_depth(depth);
    let TrieWalkState {
        root_active_indices,
        depth_logs,
    } = trie;
    let (source_indices, log) = if depth == 0 {
        (root_active_indices.as_slice(), &mut depth_logs[0])
    } else {
        let (previous, current) = depth_logs.split_at_mut(depth);
        (
            previous[depth - 1].active_indices.as_slice(),
            &mut current[0],
        )
    };
    log.clear();

    let num_groups = dfa.num_groups;
    let dirty_words = scratch.dirty_words;
    let position = (position_offset + depth + 1) as u32;
    let class = dfa.byte_to_class[byte as usize] as usize;
    let class_base = class * dfa.num_states;

    debug_assert!(source_indices.iter().all(|&i| i < batch_len));
    for &i in source_indices {
        let old_state = scratch.current_states[i];
        debug_assert_ne!(old_state, STATE_NONE);
        if dfa.finalizers[old_state].is_empty()
            && dfa.self_loop_bytes[old_state].contains(byte)
        {
            log.active_indices.push(i);
            continue;
        }

        let next_state_raw =
            unsafe { *dfa.trans_by_class.get_unchecked(class_base + old_state) };
        if next_state_raw == NONE {
            log.state_changes.push((i, old_state));
            scratch.current_states[i] = STATE_NONE;
            continue;
        }

        let ns = next_state_raw as usize;
        log.state_changes.push((i, old_state));
        scratch.current_states[i] = ns;

        let base = i * num_groups;
        for &gid in &dfa.finalizers[ns] {
            if gid < num_groups {
                let ix = base + gid;
                let old_mp = scratch.match_positions[ix];
                if old_mp == NONE {
                    let word_idx = gid / 64;
                    let bit = gid % 64;
                    let flat_idx = i * dirty_words + word_idx;
                    log.dirty_changes
                        .push((flat_idx, scratch.dirty_group_masks[flat_idx]));
                    if scratch.dirty_state_flags[i] == 0 {
                        log.dirty_state_flag_changes
                            .push((i, scratch.dirty_state_flags[i]));
                        scratch.dirty_state_flags[i] = 1;
                        set_dirty_state_bit(&mut scratch.dirty_state_bits, i);
                    }
                    scratch.dirty_group_masks[flat_idx] |= 1u64 << bit;
                }
                log.match_changes.push((ix, old_mp));
                update_trie_target_aggregate(scratch, gid, old_mp, position);
                scratch.match_positions[ix] = position;
            }
        }

        if dfa.is_dead_end[ns] {
            scratch.current_states[i] = STATE_NONE;
        } else {
            log.active_indices.push(i);
        }
    }
}

fn dfs_step_profiled(
    dfa: &Dfa,
    scratch: &mut Scratch,
    trie: &mut TrieWalkState,
    byte: u8,
    depth: usize,
    position_offset: usize,
    batch_len: usize,
) -> DfsStepStats {
    trie.ensure_depth(depth);
    let TrieWalkState {
        root_active_indices,
        depth_logs,
    } = trie;
    let (source_indices, log) = if depth == 0 {
        (root_active_indices.as_slice(), &mut depth_logs[0])
    } else {
        let (previous, current) = depth_logs.split_at_mut(depth);
        (
            previous[depth - 1].active_indices.as_slice(),
            &mut current[0],
        )
    };
    log.clear();

    let mut step_stats = DfsStepStats::default();

    let num_groups = dfa.num_groups;
    let dirty_words = scratch.dirty_words;
    let position = (position_offset + depth + 1) as u32;
    let class = dfa.byte_to_class[byte as usize] as usize;
    let class_base = class * dfa.num_states;

    debug_assert!(source_indices.iter().all(|&i| i < batch_len));
    for &i in source_indices {
        let old_state = scratch.current_states[i];
        debug_assert_ne!(old_state, STATE_NONE);
        step_stats.states_visited += 1;
        if dfa.finalizers[old_state].is_empty()
            && dfa.self_loop_bytes[old_state].contains(byte)
        {
            log.active_indices.push(i);
            step_stats.noop_self_loops += 1;
            continue;
        }

        let mut state_new_dirty_groups = 0usize;

        let next_state_raw =
            unsafe { *dfa.trans_by_class.get_unchecked(class_base + old_state) };
        if next_state_raw == NONE {
            log.state_changes.push((i, old_state));
            scratch.current_states[i] = STATE_NONE;
            step_stats.dead_transitions += 1;
            step_stats.dead_without_new_dirty += 1;
            continue;
        }

        let ns = next_state_raw as usize;
        log.state_changes.push((i, old_state));
        scratch.current_states[i] = ns;

        let base = i * num_groups;
        for &gid in &dfa.finalizers[ns] {
            if gid < num_groups {
                let ix = base + gid;
                let old_mp = scratch.match_positions[ix];
                if old_mp == NONE {
                    let word_idx = gid / 64;
                    let bit = gid % 64;
                    let flat_idx = i * dirty_words + word_idx;
                    log.dirty_changes
                        .push((flat_idx, scratch.dirty_group_masks[flat_idx]));
                    if scratch.dirty_state_flags[i] == 0 {
                        log.dirty_state_flag_changes
                            .push((i, scratch.dirty_state_flags[i]));
                        scratch.dirty_state_flags[i] = 1;
                        set_dirty_state_bit(&mut scratch.dirty_state_bits, i);
                        step_stats.new_dirty_states += 1;
                    }
                    scratch.dirty_group_masks[flat_idx] |= 1u64 << bit;
                    step_stats.new_dirty_groups += 1;
                    state_new_dirty_groups += 1;
                }
                log.match_changes.push((ix, old_mp));
                update_trie_target_aggregate(scratch, gid, old_mp, position);
                scratch.match_positions[ix] = position;
            }
        }

        if dfa.is_dead_end[ns] {
            scratch.current_states[i] = STATE_NONE;
            step_stats.dead_transitions += 1;
            if state_new_dirty_groups == 0 {
                step_stats.dead_without_new_dirty += 1;
            }
        } else {
            log.active_indices.push(i);
        }
    }

    step_stats
}

/// Undo all changes recorded at a single depth level.
fn dfs_undo_depth(scratch: &mut Scratch, log: &DepthChangeLog) {
    for &(ix, old_mp) in log.match_changes.iter().rev() {
        let current_mp = scratch.match_positions[ix];
        let gid = ix % scratch.num_groups;
        update_trie_target_aggregate(scratch, gid, current_mp, old_mp);
        scratch.match_positions[ix] = old_mp;
    }
    for &(flat_idx, old_dirty) in log.dirty_changes.iter().rev() {
        scratch.dirty_group_masks[flat_idx] = old_dirty;
    }
    for &(state_idx, old_flag) in log.dirty_state_flag_changes.iter().rev() {
        scratch.dirty_state_flags[state_idx] = old_flag;
        if old_flag == 0 {
            clear_dirty_state_bit(&mut scratch.dirty_state_bits, state_idx);
        } else {
            set_dirty_state_bit(&mut scratch.dirty_state_bits, state_idx);
        }
    }
    for &(i, old_state) in log.state_changes.iter().rev() {
        scratch.current_states[i] = old_state;
    }
}

/// Backtrack from current_depth to target_depth by undoing changes.
fn dfs_backtrack(
    scratch: &mut Scratch,
    trie: &TrieWalkState,
    current_depth: usize,
    target_depth: usize,
) {
    for depth in (target_depth..current_depth).rev() {
        dfs_undo_depth(scratch, &trie.depth_logs[depth]);
    }
}

/// Fast token signature when no dirty flags are set (no targets).
/// Uses 4-way loop unrolling to break the serial multiply-add dependency chain
/// in the original `finish_token_signature_no_cleanup`, reducing latency from
/// ~4 cycles/element to ~1 cycle/element on pipelined CPUs.
#[inline(never)]
fn finish_token_signature_clean(
    dfa: &Dfa,
    num_initial_states: usize,
    scratch: &Scratch,
) -> u64 {
    let c = HASH_SEED1;
    let c2 = c.wrapping_mul(c);
    let c3 = c2.wrapping_mul(c);
    let c4 = c3.wrapping_mul(c);
    let mut sig: u64 = HASH_SEED3;
    let n4 = (num_initial_states / 4) * 4;
    for i in (0..n4).step_by(4) {
        let x0 = dfa.completion(scratch.current_states[i]);
        let x1 = dfa.completion(scratch.current_states[i + 1]);
        let x2 = dfa.completion(scratch.current_states[i + 2]);
        let x3 = dfa.completion(scratch.current_states[i + 3]);
        let term = x0
            .wrapping_mul(c3)
            .wrapping_add(x1.wrapping_mul(c2))
            .wrapping_add(x2.wrapping_mul(c))
            .wrapping_add(x3);
        sig = sig.wrapping_mul(c4).wrapping_add(term);
    }
    for i in n4..num_initial_states {
        sig = sig
            .wrapping_mul(c)
            .wrapping_add(dfa.completion(scratch.current_states[i]));
    }
    sig
}

fn all_none_completion_signature(dfa: &Dfa, num_initial_states: usize) -> u64 {
    let mut signature = HASH_SEED3;
    for _ in 0..num_initial_states {
        signature = signature
            .wrapping_mul(HASH_SEED1)
            .wrapping_add(dfa.none_completion_hash);
    }
    signature
}

/// Compute the clean completion fold from the states that remain live at the
/// current trie node. Every omitted state is exactly `STATE_NONE`, so starting
/// from the all-dead fold and applying polynomial-position corrections is
/// algebraically identical to scanning the full batch.
fn finish_token_signature_sparse_live_clean(
    dfa: &Dfa,
    live_indices: &[usize],
    scratch: &Scratch,
    all_none_signature: u64,
) -> u64 {
    let mut signature = all_none_signature;
    for &index in live_indices {
        let state = scratch.current_states[index];
        debug_assert_ne!(state, STATE_NONE);
        let correction = dfa.completion_hash[state].wrapping_sub(dfa.none_completion_hash);
        signature = signature.wrapping_add(
            correction.wrapping_mul(scratch.completion_weights[index]),
        );
    }
    signature
}

/// Compute token signature without modifying scratch state (no cleanup).
/// Uses multi-word bitmask dirty tracking for any number of groups.
fn finish_token_signature_no_cleanup(
    dfa: &Dfa,
    num_initial_states: usize,
    scratch: &Scratch,
) -> u64 {
    let num_groups = dfa.num_groups;
    let dirty_words = scratch.dirty_words;
    let dag = &scratch.dag_nodes;
    let single_target_hash_pos = scratch.single_target_hash_pos;
    let single_target_hash = scratch.single_target_hash;
    let mut sig: u64 = HASH_SEED3;
    for i in 0..num_initial_states {
        let completion = dfa.completion(scratch.current_states[i]);
        let base = i * num_groups;
        let mask_base = i * dirty_words;

        let state_sig = if scratch.dirty_state_flags[i] != 0 {
            let mut h = new_hasher();
            h.write_u64(completion);
            for w in 0..dirty_words {
                let mut dm = scratch.dirty_group_masks[mask_base + w];
                while dm != 0 {
                    let bit = dm.trailing_zeros() as usize;
                    dm &= dm - 1;
                    let gid = w * 64 + bit;
                    if gid >= num_groups {
                        break;
                    }
                    let pv = scratch.match_positions[base + gid];
                    if pv != NONE && pv > 0 {
                        h.write_u64(gid as u64);
                        let target = pv as usize;
                        let target_hash = if single_target_hash_pos == target {
                            single_target_hash
                        } else {
                            dag.get(target)
                                .and_then(|node| node.as_ref())
                                .map_or(0, |node| node.hash)
                        };
                        h.write_u64(target_hash);
                    }
                }
            }
            h.finish()
        } else {
            completion
        };
        sig = sig.wrapping_mul(HASH_SEED1).wrapping_add(state_sig);
    }
    sig
}

/// Compute the clean completion fold with the 4-way fast path, then correct only
/// state positions whose token path recorded terminal edges. This is algebraically
/// identical to the full per-state fold because each correction is multiplied by
/// the original state's polynomial position weight.
fn finish_token_signature_sparse_dirty(
    dfa: &Dfa,
    num_initial_states: usize,
    live_indices: &[usize],
    scratch: &Scratch,
    all_none_signature: u64,
) -> u64 {
    let num_groups = dfa.num_groups;
    let dirty_words = scratch.dirty_words;
    let dag = &scratch.dag_nodes;
    let single_target_hash_pos = scratch.single_target_hash_pos;
    let single_target_hash = scratch.single_target_hash;
    // The clean component of a dirty token is the same polynomial fold as for
    // a clean token.  Every state omitted from `live_indices` is exactly
    // STATE_NONE, so start from the all-dead fold and correct only live states.
    // The dirty-state loop below then replaces each clean completion with the
    // full terminal-edge signature at that same polynomial position.
    let mut sig = if live_indices.len().saturating_mul(2) <= num_initial_states {
        finish_token_signature_sparse_live_clean(
            dfa,
            live_indices,
            scratch,
            all_none_signature,
        )
    } else {
        finish_token_signature_clean(dfa, num_initial_states, scratch)
    };
    let state_words = num_initial_states.div_ceil(64);
    for (word_idx, &dirty_word) in scratch.dirty_state_bits[..state_words].iter().enumerate() {
        let mut bits = dirty_word;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let i = word_idx * 64 + bit;
            if i >= num_initial_states {
                break;
            }

            let completion = dfa.completion(scratch.current_states[i]);
            let base = i * num_groups;
            let mask_base = i * dirty_words;
            let mut h = new_hasher();
            h.write_u64(completion);
            for w in 0..dirty_words {
                let mut dm = scratch.dirty_group_masks[mask_base + w];
                while dm != 0 {
                    let group_bit = dm.trailing_zeros() as usize;
                    dm &= dm - 1;
                    let gid = w * 64 + group_bit;
                    if gid >= num_groups {
                        break;
                    }
                    let pv = scratch.match_positions[base + gid];
                    if pv != NONE && pv > 0 {
                        h.write_u64(gid as u64);
                        let target = pv as usize;
                        let target_hash = if single_target_hash_pos == target {
                            single_target_hash
                        } else {
                            dag.get(target)
                                .and_then(|node| node.as_ref())
                                .map_or(0, |node| node.hash)
                        };
                        h.write_u64(target_hash);
                    }
                }
            }
            let state_sig = h.finish();
            let correction = state_sig.wrapping_sub(completion);
            sig = sig.wrapping_add(correction.wrapping_mul(scratch.completion_weights[i]));
        }
    }

    sig
}

fn token_indices_in_lexical_order<S: AsRef<[u8]>>(strings: &[S]) -> Vec<usize> {
    if strings.windows(2).all(|pair| pair[0].as_ref() <= pair[1].as_ref()) {
        return (0..strings.len()).collect();
    }
    let mut order = (0..strings.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        strings[left]
            .as_ref()
            .cmp(strings[right].as_ref())
            .then_with(|| left.cmp(&right))
    });
    order
}

fn active_indices_in_lexical_order(
    lexical_order: &[usize],
    active_tokens: &[bool],
) -> Vec<usize> {
    lexical_order
        .iter()
        .copied()
        .filter(|&token_idx| active_tokens[token_idx])
        .collect()
}

/// Process a sorted chunk of tokens using DFS trie walk with prefix sharing.
/// Tokens must be sorted by byte content. Returns (token_idx, signature) pairs.
fn trie_walk_chunk_signatures<S: AsRef<[u8]> + Sync>(
    dfa: &Dfa,
    strings: &[S],
    chunk: &[usize],
    batch: &[usize],
    state_group_size: usize,
    scratch: &mut Scratch,
    trie: &mut TrieWalkState,
    profile: bool,
) -> (Vec<(usize, u64)>, TrieWalkChunkStats) {
    trie_walk_chunk_signatures_from_prefix(
        dfa,
        strings,
        chunk,
        batch,
        state_group_size,
        0,
        scratch,
        trie,
        profile,
    )
}

/// Process tokens after a common already-consumed prefix.
///
/// `batch` is the DFA outcome coordinate after `position_offset` bytes. A
/// `STATE_NONE` entry denotes an already-dead path. For the first-transition
/// quotient `position_offset == 1`; finalizers of each live successor are
/// seeded at match position 1 before the suffix trie walk begins. Tokens remain
/// the original full byte strings so suffix-restart hashing observes the exact
/// original positions.
fn trie_walk_chunk_signatures_from_prefix<S: AsRef<[u8]> + Sync>(
    dfa: &Dfa,
    strings: &[S],
    chunk: &[usize],
    batch: &[usize],
    _state_group_size: usize,
    position_offset: usize,
    scratch: &mut Scratch,
    trie: &mut TrieWalkState,
    profile: bool,
) -> (Vec<(usize, u64)>, TrieWalkChunkStats) {
    let batch_len = batch.len();
    let num_groups = dfa.num_groups;
    let mut results = Vec::with_capacity(chunk.len());
    let elapsed_ms = |started_at: Option<Instant>| {
        started_at.map_or(0.0, |instant| instant.elapsed().as_secs_f64() * 1000.0)
    };
    let mut stats = TrieWalkChunkStats::default();

    scratch.current_states[..batch_len].clone_from_slice(batch);
    trie.root_active_indices.clear();
    {
        let dirty_words = scratch.dirty_words;
        let mask_end = batch_len * dirty_words;
        scratch.dirty_group_masks[..mask_end].fill(0);
    }
    scratch.dirty_state_flags[..batch_len].fill(0);
    let dirty_state_words = batch_len.div_ceil(64);
    scratch.dirty_state_bits[..dirty_state_words].fill(0);
    ensure_completion_weights(scratch, batch_len);
    let all_none_signature = all_none_completion_signature(dfa, batch_len);

    let max_token_len = chunk
        .iter()
        .map(|&token_idx| strings[token_idx].as_ref().len())
        .max()
        .unwrap_or(0);
    debug_assert!(chunk
        .iter()
        .all(|&token_idx| strings[token_idx].as_ref().len() >= position_offset));
    reset_trie_target_aggregate(scratch, max_token_len);

    let mut dirty_count = 0usize;
    if position_offset == 0 {
        for i in 0..batch_len {
            let state = scratch.current_states[i];
            debug_assert_ne!(state, STATE_NONE);
            if dfa.is_dead_end[state] {
                scratch.current_states[i] = STATE_NONE;
            } else {
                trie.root_active_indices.push(i);
            }
        }
    } else {
        let seeded_position = position_offset as u32;
        for i in 0..batch_len {
            let state = scratch.current_states[i];
            if state == STATE_NONE {
                continue;
            }
            debug_assert!(state < dfa.num_states);
            let base = i * num_groups;
            let mut state_became_dirty = false;
            for &gid in &dfa.finalizers[state] {
                if gid >= num_groups {
                    continue;
                }
                let ix = base + gid;
                debug_assert_eq!(scratch.match_positions[ix], NONE);
                mark_dirty_group(scratch, i, gid);
                update_trie_target_aggregate(scratch, gid, NONE, seeded_position);
                scratch.match_positions[ix] = seeded_position;
                state_became_dirty = true;
            }
            if state_became_dirty {
                set_dirty_state_bit(&mut scratch.dirty_state_bits, i);
                dirty_count += 1;
            }
            if dfa.is_dead_end[state] {
                scratch.current_states[i] = STATE_NONE;
            } else {
                trie.root_active_indices.push(i);
            }
        }
    }

    let initial_dirty_count = dirty_count;
    let mut current_depth = 0usize;
    let mut prev_suffix: &[u8] = &[];

    for &token_idx in chunk {
        let token = strings[token_idx].as_ref();
        let suffix = &token[position_offset..];
        let token_len = token.len();
        let suffix_len = suffix.len();
        let dfs_started_at = profile.then(Instant::now);

        let lcp = prev_suffix
            .iter()
            .zip(suffix.iter())
            .take_while(|(left, right)| left == right)
            .count();

        if current_depth > lcp {
            for depth in (lcp..current_depth).rev() {
                dirty_count -= trie.depth_logs[depth].dirty_state_flag_changes.len();
            }
            dfs_backtrack(scratch, trie, current_depth, lcp);
        }

        for depth in lcp..suffix_len {
            if profile {
                let step_stats = dfs_step_profiled(
                    dfa,
                    scratch,
                    trie,
                    suffix[depth],
                    depth,
                    position_offset,
                    batch_len,
                );
                dirty_count += trie.depth_logs[depth].dirty_state_flag_changes.len();
                stats.dfs_steps += 1;
                if step_stats.new_dirty_groups == 0 {
                    stats.dfs_steps_without_new_dirty += 1;
                }
                stats.dfs_states_visited += step_stats.states_visited;
                stats.dfs_dead_transitions += step_stats.dead_transitions;
                stats.dfs_dead_without_new_dirty += step_stats.dead_without_new_dirty;
                stats.dfs_new_dirty_groups += step_stats.new_dirty_groups;
                stats.dfs_new_dirty_states += step_stats.new_dirty_states;
                stats.dfs_noop_self_loops += step_stats.noop_self_loops;
            } else {
                dfs_step(
                    dfa,
                    scratch,
                    trie,
                    suffix[depth],
                    depth,
                    position_offset,
                    batch_len,
                );
                dirty_count += trie.depth_logs[depth].dirty_state_flag_changes.len();
            }
        }
        stats.dfs_step_ms += elapsed_ms(dfs_started_at);
        current_depth = suffix_len;
        let live_indices = if suffix_len == 0 {
            trie.root_active_indices.as_slice()
        } else {
            trie.depth_logs[suffix_len - 1].active_indices.as_slice()
        };

        let sig = if dirty_count == 0 {
            if profile {
                stats.clean_tokens += 1;
            }
            let finish_started_at = profile.then(Instant::now);
            let signature = finish_token_signature_sparse_live_clean(
                dfa,
                live_indices,
                scratch,
                all_none_signature,
            );
            stats.finish_signature_ms += elapsed_ms(finish_started_at);
            signature
        } else {
            if profile {
                stats.dirty_tokens += 1;
            }
            scratch.single_target_hash_pos = usize::MAX;
            scratch.single_target_hash = 0;

            let collect_targets_started_at = profile.then(Instant::now);
            let dense_suffix_dag = *VOCAB_DENSE_TRIE_SUFFIX_DAG_ENABLED;
            collect_trie_targets(scratch, token_len, dense_suffix_dag);
            stats.collect_targets_ms += elapsed_ms(collect_targets_started_at);

            let target_count = scratch.targets.len();
            if profile {
                stats.total_targets += target_count;
            }
            if target_count == 1 {
                if profile {
                    stats.single_target_tokens += 1;
                }
                let single_target_started_at = profile.then(Instant::now);
                if dense_suffix_dag {
                    hash_single_target_suffix_dense(dfa, token, scratch);
                } else if try_hash_single_target_suffix(dfa, token, scratch).is_none() {
                        ensure_target_gids_map(
                            &mut scratch.target_gids,
                            scratch.single_target_pos,
                            scratch.single_target_gids.as_slice(),
                        );
                        hash_suffixes(dfa, token, scratch, false);
                }
                stats.single_target_suffix_ms += elapsed_ms(single_target_started_at);
            } else if target_count > 0 {
                if profile {
                    stats.multi_target_tokens += 1;
                }
                let multi_target_started_at = profile.then(Instant::now);
                if dense_suffix_dag {
                    hash_trie_suffixes_dense(dfa, token, scratch, None);
                } else {
                    hash_suffixes(dfa, token, scratch, false);
                }
                stats.multi_target_suffix_ms += elapsed_ms(multi_target_started_at);
            }

            let finish_started_at = profile.then(Instant::now);
            let signature = if *VOCAB_SPARSE_DIRTY_FINISH_DISABLED {
                finish_token_signature_no_cleanup(dfa, batch_len, scratch)
            } else {
                finish_token_signature_sparse_dirty(
                    dfa,
                    batch_len,
                    live_indices,
                    scratch,
                    all_none_signature,
                )
            };
            stats.finish_signature_ms += elapsed_ms(finish_started_at);
            signature
        };

        results.push((token_idx, sig));
        prev_suffix = suffix;
    }

    if current_depth > 0 {
        for depth in (0..current_depth).rev() {
            dirty_count -= trie.depth_logs[depth].dirty_state_flag_changes.len();
        }
        dfs_backtrack(scratch, trie, current_depth, 0);
    }
    debug_assert_eq!(dirty_count, initial_dirty_count);

    // Prefix observations are the root baseline rather than trie-log changes,
    // so remove them explicitly before returning the scratch worker to the pool.
    if position_offset != 0 {
        let seeded_position = position_offset as u32;
        for (i, &state) in batch.iter().enumerate() {
            if state == STATE_NONE {
                continue;
            }
            let base = i * num_groups;
            for &gid in &dfa.finalizers[state] {
                if gid >= num_groups {
                    continue;
                }
                let ix = base + gid;
                debug_assert_eq!(scratch.match_positions[ix], seeded_position);
                update_trie_target_aggregate(
                    scratch,
                    gid,
                    scratch.match_positions[ix],
                    NONE,
                );
                scratch.match_positions[ix] = NONE;
            }
        }
        scratch.dirty_state_flags[..batch_len].fill(0);
        scratch.dirty_state_bits[..dirty_state_words].fill(0);
        let mask_end = batch_len * scratch.dirty_words;
        scratch.dirty_group_masks[..mask_end].fill(0);
    }

    (results, stats)
}

#[derive(Clone, Copy, Default)]
struct FirstTransitionFactorStats {
    semantic_buckets: usize,
    factored_buckets: usize,
    min_bucket_tokens: usize,
    source_state_buckets_before: usize,
    source_state_buckets_after: usize,
    full_state_token_pairs: usize,
    preliminary_state_token_pairs: usize,
    preliminary_classes: usize,
    parallel_buckets: bool,
    setup_ms: f64,
    preliminary_signature_ms: f64,
    preliminary_grouping_ms: f64,
    trie_walk: TrieWalkChunkStats,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FirstTransitionFactorMode {
    Disabled,
    Environment,
    Force,
}

struct FirstTransitionFactorPlan {
    preliminary_classes: Vec<Vec<usize>>,
    representative_tokens: Vec<usize>,
    preliminary_class_for_representative: Vec<usize>,
    stats: FirstTransitionFactorStats,
}

struct FirstTransitionBucket {
    token_indices: Vec<usize>,
    initial_outcomes: Vec<usize>,
}

struct FirstTransitionBucketResult {
    classes: Vec<Vec<usize>>,
    signature_ms: f64,
    grouping_ms: f64,
    trie_walk: TrieWalkChunkStats,
}

/// Build an exact token prepartition using deterministic first transitions.
///
/// The source coordinate is allowed to vary by semantic leading-byte class,
/// but remains static for the complete traversal of that class. This is
/// deliberately different from the rejected dynamic selective-frontier paths:
/// no per-node membership structure is maintained or restored. Each class is
/// classified once over one representative source per distinct first
/// successor, using the already-built analysis DFA, one shared lexical order,
/// and a shared scratch pool. A later ordinary all-source pass over one token
/// representative per preliminary class is still the exact authority.
fn try_first_transition_factor_plan<S: AsRef<[u8]> + Sync>(
    dfa: &Dfa,
    strings: &[S],
    initial_states: &[usize],
    lexical_order: &[usize],
    scratch_pool: &Arc<ScratchPool>,
    profiling: bool,
    enforce_work_ratio: bool,
) -> Option<FirstTransitionFactorPlan> {
    if *TRIE_WALK_DISABLED
        || strings.len() < 2
        || initial_states.len() < 2
        || strings.iter().any(|token| token.as_ref().is_empty())
    {
        return None;
    }
    let setup_started_at = Instant::now();

    let num_classes = dfa
        .byte_to_class
        .iter()
        .copied()
        .max()
        .map_or(0usize, |max_class| max_class as usize + 1);
    let mut tokens_by_class = (0..num_classes)
        .map(|_| Vec::<usize>::new())
        .collect::<Vec<_>>();
    for &token_idx in lexical_order {
        let first_byte = strings[token_idx].as_ref()[0];
        let class = dfa.byte_to_class[first_byte as usize] as usize;
        tokens_by_class[class].push(token_idx);
    }

    let min_bucket_tokens = first_transition_factor_min_bucket_tokens();
    let mut buckets = Vec::<FirstTransitionBucket>::new();
    let mut singleton_tokens = Vec::<usize>::new();
    let mut semantic_buckets = 0usize;
    let mut source_state_buckets_before = 0usize;
    let mut source_state_buckets_after = 0usize;
    let mut preliminary_state_token_pairs = 0usize;
    let mut preliminary_domain_by_token = vec![usize::MAX; strings.len()];
    let mut seen_outcomes = vec![0u32; dfa.num_states + 1];
    let mut seen_outcome_epoch = 0u32;

    for token_indices in tokens_by_class.into_iter().filter(|bucket| !bucket.is_empty()) {
        semantic_buckets += 1;
        for &token_idx in &token_indices {
            preliminary_domain_by_token[token_idx] = semantic_buckets;
        }
        if token_indices.len() < min_bucket_tokens {
            singleton_tokens.extend(token_indices);
            continue;
        }

        let first_byte = strings[token_indices[0]].as_ref()[0];
        let mut initial_outcomes = Vec::with_capacity(initial_states.len());
        advance_seen_epoch(&mut seen_outcomes, &mut seen_outcome_epoch);
        for &source in initial_states {
            // The ordinary engine suppresses a source marked dead before it
            // consumes any token byte. A pre-dead source and a live source with
            // no first transition therefore share STATE_NONE. A transition to
            // a real dead-end state remains distinct because its first-byte
            // finalizers are observable before the path dies.
            let effective_outcome = if dfa.is_dead_end[source] {
                STATE_NONE
            } else {
                let target = dfa.transition(source, first_byte);
                if target == NONE {
                    STATE_NONE
                } else {
                    target as usize
                }
            };
            let seen_index = if effective_outcome == STATE_NONE {
                dfa.num_states
            } else {
                effective_outcome
            };
            if seen_outcomes[seen_index] != seen_outcome_epoch {
                seen_outcomes[seen_index] = seen_outcome_epoch;
                initial_outcomes.push(effective_outcome);
            }
        }
        source_state_buckets_before = source_state_buckets_before
            .saturating_add(initial_states.len());
        source_state_buckets_after = source_state_buckets_after
            .saturating_add(initial_outcomes.len());
        preliminary_state_token_pairs = preliminary_state_token_pairs.saturating_add(
            token_indices.len().saturating_mul(initial_outcomes.len()),
        );
        buckets.push(FirstTransitionBucket {
            token_indices,
            initial_outcomes,
        });
    }

    if buckets.is_empty() {
        return None;
    }

    if buckets.len() > 1 {
        let mut merged_buckets = Vec::<FirstTransitionBucket>::with_capacity(buckets.len());
        let mut shape_to_bucket = HashMap::<Vec<usize>, usize>::with_capacity(buckets.len());
        for bucket in buckets {
            if let Some(&merged_idx) = shape_to_bucket.get(&bucket.initial_outcomes) {
                merged_buckets[merged_idx]
                    .token_indices
                    .extend(bucket.token_indices);
            } else {
                let merged_idx = merged_buckets.len();
                shape_to_bucket.insert(bucket.initial_outcomes.clone(), merged_idx);
                merged_buckets.push(bucket);
            }
        }
        buckets = merged_buckets;
    }

    let full_state_token_pairs = strings.len().saturating_mul(initial_states.len());
    let preliminary_ratio = preliminary_state_token_pairs as f64
        / full_state_token_pairs.max(1) as f64;
    let max_work_ratio = first_transition_factor_max_work_ratio(
        strings.len(),
        initial_states.len(),
        dfa.num_groups,
    );
    if enforce_work_ratio && preliminary_ratio > max_work_ratio {
        return None;
    }
    let uses_moderate_ratio_extension = enforce_work_ratio
        && first_transition_factor_max_work_ratio_override().is_none()
        && first_transition_factor_moderate_extension_enabled(
            strings.len(),
            initial_states.len(),
            dfa.num_groups,
        )
        && preliminary_ratio > FIRST_TRANSITION_FACTOR_MAX_WORK_RATIO_DEFAULT;

    // The admission decision above deliberately uses the historical raw state
    // counts. The exact literal-row quotient only makes an already accepted
    // large factor cheaper; it can never cause a new factor plan to be
    // selected. Restrict it to the large state×vocabulary shape where the
    // first-transition pass is a measured tail hotspot and the O(states ×
    // byte-classes) quotient cost is well amortized.
    if first_transition_literal_row_quotient_enabled()
        && strings.len() >= 10_000
        && initial_states.len() >= 10_000
        && dfa.num_states >= 20_000
    {
        let quotient_started_at = Instant::now();
        let quotient_num_classes = dfa
            .byte_to_class
            .iter()
            .copied()
            .max()
            .map_or(0usize, |max_class| max_class as usize + 1);
        let mut active_suffix_classes = vec![false; quotient_num_classes];
        for string in strings {
            for &byte in string.as_ref().iter().skip(1) {
                active_suffix_classes[dfa.byte_to_class[byte as usize] as usize] = true;
            }
        }
        let active_suffix_class_count = active_suffix_classes.iter().filter(|&&active| active).count();
        let quotient_row_work = dfa.num_states.saturating_mul(active_suffix_class_count);
        let quotient_max_work = first_transition_literal_row_quotient_max_work();
        let quotient = if quotient_row_work <= quotient_max_work {
            exact_literal_state_row_quotient(dfa, &active_suffix_classes, profiling)
        } else {
            None
        };
        if let Some((behavior_classes, representatives)) = quotient {
            let bucket_states_before = buckets
                .iter()
                .map(|bucket| bucket.initial_outcomes.len())
                .sum::<usize>();
            for bucket in &mut buckets {
                let mut seen_classes = vec![false; representatives.len()];
                let mut seen_none = false;
                let mut quotient_outcomes = Vec::with_capacity(bucket.initial_outcomes.len());
                for &state in &bucket.initial_outcomes {
                    if state == STATE_NONE {
                        if !seen_none {
                            seen_none = true;
                            quotient_outcomes.push(STATE_NONE);
                        }
                        continue;
                    }
                    let class = behavior_classes[state] as usize;
                    if !seen_classes[class] {
                        seen_classes[class] = true;
                        quotient_outcomes.push(representatives[class]);
                    }
                }
                bucket.initial_outcomes = quotient_outcomes;
            }

            // The quotient may make two previously distinct first-byte buckets
            // share the exact same source coordinate. Reuse one traversal while
            // retaining each token's semantic leading-byte domain in the later
            // grouping key.
            if buckets.len() > 1 {
                let mut merged_buckets =
                    Vec::<FirstTransitionBucket>::with_capacity(buckets.len());
                let mut shape_to_bucket =
                    HashMap::<Vec<usize>, usize>::with_capacity(buckets.len());
                for bucket in buckets {
                    if let Some(&merged_idx) = shape_to_bucket.get(&bucket.initial_outcomes) {
                        merged_buckets[merged_idx]
                            .token_indices
                            .extend(bucket.token_indices);
                    } else {
                        let merged_idx = merged_buckets.len();
                        shape_to_bucket.insert(bucket.initial_outcomes.clone(), merged_idx);
                        merged_buckets.push(bucket);
                    }
                }
                buckets = merged_buckets;
            }
            if profiling {
                let bucket_states_after = buckets
                    .iter()
                    .map(|bucket| bucket.initial_outcomes.len())
                    .sum::<usize>();
                eprintln!(
                    "[glrmask/profile][vocab_first_transition_literal_row_quotient] dfa_states={} row_classes={} bucket_states_before={} bucket_states_after={} buckets={} reduction_pct={:.2} total_ms={:.3}",
                    dfa.num_states,
                    representatives.len(),
                    bucket_states_before,
                    bucket_states_after,
                    buckets.len(),
                    if bucket_states_before == 0 {
                        0.0
                    } else {
                        100.0
                            * (1.0
                                - bucket_states_after as f64 / bucket_states_before as f64)
                    },
                    quotient_started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
        } else if profiling && quotient_row_work > quotient_max_work {
            eprintln!(
                "[glrmask/profile][vocab_first_transition_literal_row_quotient_skip] dfa_states={} active_byte_classes={} row_work={} max_work={}",
                dfa.num_states,
                active_suffix_class_count,
                quotient_row_work,
                quotient_max_work,
            );
        }
    }

    // Suffix sorting is required only for an accepted plan. Deferring it until
    // after the structural work-ratio test keeps default-off/rejected attempts
    // from sorting every large vocabulary a second time.
    for bucket in &mut buckets {
        bucket.token_indices.sort_unstable_by(|&left, &right| {
            strings[left].as_ref()[1..]
                .cmp(&strings[right].as_ref()[1..])
                .then_with(|| left.cmp(&right))
        });
    }
    let setup_ms = setup_started_at.elapsed().as_secs_f64() * 1000.0;

    let process_token_chunk = |bucket: &FirstTransitionBucket, token_indices: &[usize]| {
        let signature_started_at = Instant::now();
        let mut lease = scratch_pool.checkout(bucket.initial_outcomes.len(), dfa.num_groups);
        let worker = lease.worker_mut();
        let state_group_size = vocab_state_group_size(bucket.initial_outcomes.len(), dfa.num_groups);
        let (active_sigs, trie_walk) = trie_walk_chunk_signatures_from_prefix(
            dfa,
            strings,
            token_indices,
            &bucket.initial_outcomes,
            state_group_size,
            1,
            &mut worker.scratch,
            &mut worker.trie_state,
            profiling,
        );
        (
            active_sigs,
            trie_walk,
            signature_started_at.elapsed().as_secs_f64() * 1000.0,
        )
    };

    let group_active_sigs = |active_sigs: Vec<(usize, u64)>| {
        let grouping_started_at = Instant::now();
        // Equal outcome vectors may share one suffix trie walk, but retain the
        // original semantic leading-byte domain in the preliminary partition.
        // This captures traversal reuse without making the authority pass
        // resolve a deliberately coarser cross-domain class.
        let mut by_signature =
            HashMap::<(usize, u64), Vec<usize>>::with_capacity(active_sigs.len());
        for (token_idx, signature) in active_sigs {
            by_signature
                .entry((preliminary_domain_by_token[token_idx], signature))
                .or_default()
                .push(token_idx);
        }
        let mut classes = by_signature.into_values().collect::<Vec<_>>();
        for class in &mut classes {
            class.sort_unstable();
        }
        classes.sort_unstable_by_key(|class| class[0]);
        (
            classes,
            grouping_started_at.elapsed().as_secs_f64() * 1000.0,
        )
    };

    let process_bucket = |bucket: &FirstTransitionBucket| {
        let (active_sigs, trie_walk, signature_ms) =
            process_token_chunk(bucket, &bucket.token_indices);
        let (classes, grouping_ms) = group_active_sigs(active_sigs);
        FirstTransitionBucketResult {
            classes,
            signature_ms,
            grouping_ms,
            trie_walk,
        }
    };

    // The factor pass normally runs inside the partition-level Rayon DAG.
    // Preserve the existing scheduler for factors admitted by the established
    // 5% work-ratio gate. The moderate-vocabulary extension deliberately
    // admits more preliminary work; keeping those buckets sequential avoids
    // consuming the same Rayon workers that are concurrently building sibling
    // vocabulary partitions. Keep the explicit override for diagnostics.
    let parallel_buckets = buckets.len() > 1
        && first_transition_factor_parallel_buckets_override().unwrap_or_else(|| {
            !uses_moderate_ratio_extension
                && first_transition_factor_parallel_buckets_default(
                    rayon::current_num_threads(),
                    initial_states.len(),
                )
        });

    // A first-byte bucket can still dominate the factor critical path after the
    // source-coordinate reduction. Split only very large bucket work into a flat
    // Rayon task queue. This preserves the existing bucket partition and final
    // signature grouping while avoiding nested parallelism and retaining suffix
    // trie sharing within each token chunk. A zero chunk size disables the split
    // for controlled A/B diagnostics.
    let token_chunk_size = first_transition_factor_token_chunk_size();
    let token_chunk_min_work = first_transition_factor_token_chunk_min_work();
    let split_active = parallel_buckets
        && token_chunk_size > 0
        && buckets.iter().any(|bucket| {
            bucket.token_indices.len() > token_chunk_size
                && bucket
                    .token_indices
                    .len()
                    .saturating_mul(bucket.initial_outcomes.len())
                    >= token_chunk_min_work
        });

    let bucket_results = if split_active {
        let mut tasks = Vec::<(usize, usize, usize)>::new();
        let mut split_bucket_count = 0usize;
        for (bucket_idx, bucket) in buckets.iter().enumerate() {
            let len = bucket.token_indices.len();
            let bucket_work = len.saturating_mul(bucket.initial_outcomes.len());
            if len > token_chunk_size && bucket_work >= token_chunk_min_work {
                split_bucket_count += 1;
                for start in (0..len).step_by(token_chunk_size) {
                    tasks.push((bucket_idx, start, (start + token_chunk_size).min(len)));
                }
            } else {
                tasks.push((bucket_idx, 0, len));
            }
        }
        if profiling || env_flag_enabled("GLRMASK_PROFILE_VOCAB_FACTOR_TOKEN_SPLIT") {
            eprintln!(
                "[glrmask/profile][vocab_first_transition_factor_token_split] buckets={} split_buckets={} tasks={} chunk_size={} min_work={}",
                buckets.len(),
                split_bucket_count,
                tasks.len(),
                token_chunk_size,
                token_chunk_min_work,
            );
        }

        let task_results = tasks
            .par_iter()
            .map(|&(bucket_idx, start, end)| {
                let bucket = &buckets[bucket_idx];
                let (active_sigs, trie_walk, signature_ms) =
                    process_token_chunk(bucket, &bucket.token_indices[start..end]);
                (bucket_idx, active_sigs, trie_walk, signature_ms)
            })
            .collect::<Vec<_>>();
        let mut bucket_sigs = (0..buckets.len())
            .map(|_| Vec::<(usize, u64)>::new())
            .collect::<Vec<_>>();
        let mut bucket_walk = vec![TrieWalkChunkStats::default(); buckets.len()];
        let mut bucket_signature_ms = vec![0.0f64; buckets.len()];
        for (bucket_idx, task_sigs, task_walk, task_signature_ms) in task_results {
            bucket_sigs[bucket_idx].extend(task_sigs);
            bucket_walk[bucket_idx].add_assign(task_walk);
            bucket_signature_ms[bucket_idx] += task_signature_ms;
        }
        bucket_sigs
            .into_iter()
            .enumerate()
            .map(|(bucket_idx, active_sigs)| {
                let (classes, grouping_ms) = group_active_sigs(active_sigs);
                FirstTransitionBucketResult {
                    classes,
                    signature_ms: bucket_signature_ms[bucket_idx],
                    grouping_ms,
                    trie_walk: bucket_walk[bucket_idx],
                }
            })
            .collect::<Vec<_>>()
    } else if parallel_buckets {
        buckets.par_iter().map(process_bucket).collect::<Vec<_>>()
    } else {
        buckets.iter().map(process_bucket).collect::<Vec<_>>()
    };
    let mut stats = FirstTransitionFactorStats {
        semantic_buckets,
        factored_buckets: buckets.len(),
        min_bucket_tokens,
        source_state_buckets_before,
        source_state_buckets_after,
        full_state_token_pairs,
        preliminary_state_token_pairs,
        parallel_buckets,
        setup_ms,
        ..FirstTransitionFactorStats::default()
    };
    let mut preliminary_classes = singleton_tokens
        .into_iter()
        .map(|token_idx| vec![token_idx])
        .collect::<Vec<_>>();
    for result in bucket_results {
        stats.preliminary_signature_ms += result.signature_ms;
        stats.preliminary_grouping_ms += result.grouping_ms;
        stats.trie_walk.add_assign(result.trie_walk);
        preliminary_classes.extend(result.classes);
    }
    preliminary_classes.sort_unstable_by_key(|class| class[0]);
    stats.preliminary_classes = preliminary_classes.len();

    if preliminary_classes.len() >= strings.len() {
        return None;
    }

    let mut preliminary_class_for_representative = vec![usize::MAX; strings.len()];
    let representative_tokens = preliminary_classes
        .iter()
        .enumerate()
        .map(|(class_idx, class)| {
            let representative = class[0];
            preliminary_class_for_representative[representative] = class_idx;
            representative
        })
        .collect::<Vec<_>>();

    Some(FirstTransitionFactorPlan {
        preliminary_classes,
        representative_tokens,
        preliminary_class_for_representative,
        stats,
    })
}

/// Vocab equivalence with optional group filtering.
///
/// Minimum reduction ratio (compact/original) to trigger compaction.
const COMPACT_DFA_MAX_RATIO: f64 = 0.85;
/// Minimum number of initial states to consider compaction worthwhile.
const COMPACT_DFA_MIN_STATES: usize = 500;
/// Minimum work estimate (states × tokens) to justify the compaction overhead.
const COMPACT_DFA_MIN_WORK: usize = 10_000_000;
const INPUT_VIEW_COMPACT_MIN_STATES: usize = 512;
const INPUT_VIEW_COMPACT_MAX_RATIO: f64 = 0.85;
const INPUT_VIEW_COMPACT_MAX_TOKENS: usize = 16;
const SINGLETON_PROBE_MAX_TOKENS: usize = 64;
const SINGLETON_PROBE_STATES: usize = 16;
const PRE_DFA_SINGLETON_PROBE_MIN_INITIAL_STATES: usize = 64;
const PRE_DFA_SINGLETON_PROBE_MIN_WORK: usize = 8_192;
const FIRST_TRANSITION_FACTOR_MAX_WORK_RATIO_DEFAULT: f64 = 0.05;
const FIRST_TRANSITION_FACTOR_MODERATE_MIN_TOKENS: usize = 1_024;
const FIRST_TRANSITION_FACTOR_MODERATE_MAX_TOKENS: usize = 8_192;
const FIRST_TRANSITION_FACTOR_MODERATE_MAX_WORK_RATIO: f64 = 0.10;
const FIRST_TRANSITION_FACTOR_HIGH_THREAD_SEQUENTIAL_MIN_STATES: usize = 10_000;

fn first_transition_factor_parallel_buckets_default(
    num_threads: usize,
    num_initial_states: usize,
) -> bool {
    num_threads > 1
        && (num_threads <= 8
            || num_initial_states < FIRST_TRANSITION_FACTOR_HIGH_THREAD_SEQUENTIAL_MIN_STATES)
}

/// Restrict a tokenizer view to exactly the states reachable from the state
/// representatives and lexer start along bytes that appear in `strings`.
///
/// This happens before building the analysis DFA, rather than after it.  The
/// finite-vocabulary equivalence pass cannot observe any omitted transition:
/// every token walk starts from one of these roots and consumes only these
/// bytes.  Keeping the start root also preserves the suffix observations used
/// by the trellis hash.
fn compact_tokenizer_view_for_tokens<S: AsRef<[u8]>>(
    tokenizer: &TokenizerView,
    initial_states: &[usize],
    strings: &[S],
) -> Option<(TokenizerView, Vec<usize>, Vec<usize>)> {
    let source = tokenizer.dfa();
    if source.states.len() < INPUT_VIEW_COMPACT_MIN_STATES || strings.is_empty() {
        return None;
    }

    let mut relevant_bytes = [false; 256];
    for string in strings {
        for &byte in string.as_ref() {
            relevant_bytes[byte as usize] = true;
        }
    }
    if !relevant_bytes.iter().any(|&used| used) {
        return None;
    }

    let mut reachable = vec![false; source.states.len()];
    let mut queue = VecDeque::new();
    let visit = |state: usize, reachable: &mut [bool], queue: &mut VecDeque<usize>| {
        if state < reachable.len() && !reachable[state] {
            reachable[state] = true;
            queue.push_back(state);
        }
    };
    visit(source.start_state, &mut reachable, &mut queue);
    for &state in initial_states {
        visit(state, &mut reachable, &mut queue);
    }
    while let Some(state) = queue.pop_front() {
        for (byte, &used) in relevant_bytes.iter().enumerate() {
            if !used {
                continue;
            }
            let target = source.trans(state, byte);
            if target != NONE {
                visit(target as usize, &mut reachable, &mut queue);
            }
        }
    }

    let reachable_count = reachable.iter().filter(|&&state| state).count();
    if reachable_count as f64 / source.states.len() as f64 > INPUT_VIEW_COMPACT_MAX_RATIO {
        return None;
    }

    let mut original_to_compact = vec![NONE; source.states.len()];
    let mut compact_to_original = Vec::with_capacity(reachable_count);
    for (original, &is_reachable) in reachable.iter().enumerate() {
        if is_reachable {
            original_to_compact[original] = compact_to_original.len() as u32;
            compact_to_original.push(original);
        }
    }

    let mut transitions = vec![NONE; reachable_count * 256];
    let mut states = Vec::with_capacity(reachable_count);
    for (compact, &original) in compact_to_original.iter().enumerate() {
        states.push(source.states[original].clone());
        let compact_base = compact * 256;
        for (byte, &used) in relevant_bytes.iter().enumerate() {
            if !used {
                continue;
            }
            let target = source.trans(original, byte);
            if target != NONE {
                let mapped = original_to_compact[target as usize];
                debug_assert_ne!(mapped, NONE, "reachable token edge omitted from compact view");
                transitions[compact_base + byte] = mapped;
            }
        }
    }

    let compact_initial = initial_states
        .iter()
        .map(|&state| original_to_compact[state] as usize)
        .collect();
    let start_state = original_to_compact[source.start_state] as usize;
    Some((
        TokenizerView {
            flat_dfa: FlatDfa {
                states,
                start_state,
                transitions: Arc::from(transitions),
            },
        },
        compact_initial,
        compact_to_original,
    ))
}

fn raw_states_by_transition_diversity<S: AsRef<[u8]>>(
    tokenizer: &TokenizerView,
    initial_states: &[usize],
    strings: &[S],
) -> Vec<usize> {
    let mut relevant_bytes = U8Set::empty();
    for string in strings {
        for &byte in string.as_ref() {
            relevant_bytes.insert(byte);
        }
    }
    let source = tokenizer.dfa();
    let mut ranked = initial_states
        .iter()
        .copied()
        .map(|state| {
            let mut targets = SmallVec::<[u32; 64]>::new();
            for byte in relevant_bytes.iter() {
                let target = source.trans(state, byte as usize);
                if !targets.contains(&target) {
                    targets.push(target);
                }
            }
            (state, targets.len())
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.into_iter().map(|(state, _)| state).collect()
}

/// Build exactly the finite scanner subgraph observed by `token_signature` for
/// `initial_states` and `strings`.
///
/// Whole-token execution starts at each witness state. Match continuations are
/// summarized by `run_suffix`, which restarts from the tokenizer start at a
/// suffix position of the same token. No other byte sequence is observed by
/// the signature algorithm, so arbitrary closure under the union of token
/// bytes would add irrelevant cross-product states.
fn finite_token_probe_view<S: AsRef<[u8]>>(
    tokenizer: &TokenizerView,
    initial_states: &[usize],
    strings: &[S],
) -> (TokenizerView, Vec<usize>) {
    let source = tokenizer.dfa();
    let mut reached = vec![false; source.states.len()];
    let mut compact_to_original = Vec::new();
    let visit = |state: usize, reached: &mut [bool], compact_to_original: &mut Vec<usize>| {
        if !reached[state] {
            reached[state] = true;
            compact_to_original.push(state);
        }
    };

    visit(source.start_state, &mut reached, &mut compact_to_original);
    for &initial in initial_states {
        visit(initial, &mut reached, &mut compact_to_original);
        for string in strings {
            let mut state = initial;
            for &byte in string.as_ref() {
                let target = source.trans(state, byte as usize);
                if target == NONE {
                    break;
                }
                state = target as usize;
                visit(state, &mut reached, &mut compact_to_original);
            }
        }
    }
    for string in strings {
        let bytes = string.as_ref();
        for suffix_start in 0..bytes.len() {
            let mut state = source.start_state;
            for &byte in &bytes[suffix_start..] {
                let target = source.trans(state, byte as usize);
                if target == NONE {
                    break;
                }
                state = target as usize;
                visit(state, &mut reached, &mut compact_to_original);
            }
        }
    }

    compact_to_original.sort_unstable();
    let mut original_to_compact = vec![NONE; source.states.len()];
    for (compact, &original) in compact_to_original.iter().enumerate() {
        original_to_compact[original] = compact as u32;
    }

    let mut queried_edges = vec![U8Set::empty(); source.states.len()];
    for &initial in initial_states {
        for string in strings {
            let mut state = initial;
            for &byte in string.as_ref() {
                queried_edges[state].insert(byte);
                let target = source.trans(state, byte as usize);
                if target == NONE {
                    break;
                }
                state = target as usize;
            }
        }
    }
    for string in strings {
        let bytes = string.as_ref();
        for suffix_start in 0..bytes.len() {
            let mut state = source.start_state;
            for &byte in &bytes[suffix_start..] {
                queried_edges[state].insert(byte);
                let target = source.trans(state, byte as usize);
                if target == NONE {
                    break;
                }
                state = target as usize;
            }
        }
    }

    let mut transitions = vec![NONE; compact_to_original.len() * 256];
    let states = compact_to_original
        .iter()
        .enumerate()
        .map(|(compact, &original)| {
            let base = compact * 256;
            for byte in queried_edges[original].iter() {
                let target = source.trans(original, byte as usize);
                if target != NONE {
                    let mapped = original_to_compact[target as usize];
                    debug_assert_ne!(mapped, NONE, "queried token trajectory target omitted");
                    transitions[base + byte as usize] = mapped;
                }
            }
            source.states[original].clone()
        })
        .collect();
    let compact_initial = initial_states
        .iter()
        .map(|&state| original_to_compact[state] as usize)
        .collect();
    let start_state = original_to_compact[source.start_state] as usize;

    (
        TokenizerView {
            flat_dfa: FlatDfa {
                states,
                start_state,
                transitions: Arc::from(transitions),
            },
        },
        compact_initial,
    )
}

fn token_signatures_for_states<S: AsRef<[u8]>>(
    dfa: &Dfa,
    strings: &[S],
    states: &[usize],
) -> Vec<u64> {
    let state_group_size = vocab_state_group_size(states.len(), dfa.num_groups);
    let mut scratch = Scratch::new(states.len(), dfa.num_groups);
    strings
        .iter()
        .map(|token| {
            token_signature(
                dfa,
                token.as_ref(),
                states,
                state_group_size,
                &mut scratch,
                false,
            )
        })
        .collect()
}

fn token_signatures_are_pairwise_distinct<S: AsRef<[u8]>>(
    dfa: &Dfa,
    strings: &[S],
    states: &[usize],
) -> bool {
    let mut seen = HashMap::<u64, usize>::with_capacity(strings.len());
    token_signatures_for_states(dfa, strings, states)
        .into_iter()
        .enumerate()
        .all(|(token_idx, signature)| seen.insert(signature, token_idx).is_none())
}

/// Try to prove identity before constructing the full analysis DFA. A subset of
/// initial states is a sound distinguishing witness: adding states can only
/// refine token classes, never merge two signatures that already differ.
fn try_pre_dfa_singleton_identity_probe<S: AsRef<[u8]>>(
    tokenizer: &TokenizerView,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    active_groups: Option<&[bool]>,
    profiling: bool,
) -> Option<(VocabEquivalenceResult, f64)> {
    if strings.is_empty()
        || strings.len() > SINGLETON_PROBE_MAX_TOKENS
        || initial_states.len() < PRE_DFA_SINGLETON_PROBE_MIN_INITIAL_STATES
        || initial_states.len().saturating_mul(strings.len())
            < PRE_DFA_SINGLETON_PROBE_MIN_WORK
        || tokenizer.dfa().states.len() < INPUT_VIEW_COMPACT_MIN_STATES
    {
        return None;
    }

    let started_at = Instant::now();
    let state_order_started_at = profiling.then(Instant::now);
    let mut probe_states = raw_states_by_transition_diversity(tokenizer, initial_states, strings);
    probe_states.truncate(SINGLETON_PROBE_STATES);
    let state_order_ms = state_order_started_at
        .map_or(0.0, |started_at| started_at.elapsed().as_secs_f64() * 1000.0);

    let view_started_at = profiling.then(Instant::now);
    let (probe_view, probe_initial) = finite_token_probe_view(tokenizer, &probe_states, strings);
    let view_ms = view_started_at
        .map_or(0.0, |started_at| started_at.elapsed().as_secs_f64() * 1000.0);

    let build_started_at = Instant::now();
    let probe_dfa = build_dfa_with_group_filter(
        &probe_view,
        disallowed_follows,
        None,
        active_groups,
        None,
    );
    let build_dfa_ms = build_started_at.elapsed().as_secs_f64() * 1000.0;

    let signature_started_at = profiling.then(Instant::now);
    let distinct =
        token_signatures_are_pairwise_distinct(&probe_dfa, strings, &probe_initial);
    let signature_ms = signature_started_at
        .map_or(0.0, |started_at| started_at.elapsed().as_secs_f64() * 1000.0);

    if profiling {
        eprintln!(
            "[glrmask/profile][vocab_identity_probe] strings={} probe_states={} finite_states={} distinct={} state_order_ms={:.3} view_ms={:.3} build_dfa_ms={:.3} signature_ms={:.3} total_ms={:.3}",
            strings.len(),
            probe_states.len(),
            probe_view.dfa().states.len(),
            distinct,
            state_order_ms,
            view_ms,
            build_dfa_ms,
            signature_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    if !distinct {
        return None;
    }

    Some(((0..strings.len()).map(|token| vec![token]).collect(), build_dfa_ms))
}

/// Build a compact DFA containing only states reachable from `initial_states`
/// via byte classes actually used by the partition's tokens.  Returns the
/// compact DFA and the remapped initial state indices, or None if compaction
/// is not beneficial.
fn compact_dfa_for_tokens<S: AsRef<[u8]>>(
    dfa: &Dfa,
    initial_states: &[usize],
    strings: &[S],
) -> Option<(Dfa, Vec<usize>, Vec<usize>)> {
    if initial_states.len() < COMPACT_DFA_MIN_STATES
        || strings.is_empty()
        || initial_states.len() * strings.len() < COMPACT_DFA_MIN_WORK
    {
        return None;
    }

    // Relevant byte classes from the partition's tokens.
    let mut byte_used = [false; 256];
    for s in strings {
        for &b in s.as_ref() {
            byte_used[b as usize] = true;
        }
    }
    let num_classes = dfa.byte_to_class.iter().copied().max().map_or(0usize, |m| m as usize + 1);
    let mut class_used = vec![false; num_classes];
    for b in 0..=255u8 {
        if byte_used[b as usize] {
            class_used[dfa.byte_to_class[b as usize] as usize] = true;
        }
    }
    let relevant_classes: Vec<usize> = class_used
        .iter()
        .enumerate()
        .filter_map(|(i, &u)| if u { Some(i) } else { None })
        .collect();

    // BFS from initial_states following only relevant-class transitions.
    let mut visited = vec![false; dfa.num_states];
    let mut queue = std::collections::VecDeque::new();
    for &s in initial_states {
        if s < dfa.num_states && !visited[s] {
            visited[s] = true;
            queue.push_back(s);
        }
    }
    while let Some(s) = queue.pop_front() {
        for &c in &relevant_classes {
            let t = dfa.trans_by_class[c * dfa.num_states + s];
            if t != NONE {
                let t = t as usize;
                if !visited[t] {
                    visited[t] = true;
                    queue.push_back(t);
                }
            }
        }
    }

    let restricted_reachable = visited.iter().filter(|&&v| v).count();
    let ratio = restricted_reachable as f64 / dfa.num_states as f64;
    if ratio > COMPACT_DFA_MAX_RATIO {
        return None;
    }

    // Build state remapping.
    let mut original_to_compact = vec![u32::MAX; dfa.num_states];
    let mut compact_to_original = Vec::with_capacity(restricted_reachable);
    for (i, &v) in visited.iter().enumerate() {
        if v {
            original_to_compact[i] = compact_to_original.len() as u32;
            compact_to_original.push(i);
        }
    }
    let compact_num = compact_to_original.len();

    // Build compact transition table (all classes, compact states).
    let mut compact_trans = vec![NONE; num_classes * compact_num];
    for c in 0..num_classes {
        let src_base = c * dfa.num_states;
        let dst_base = c * compact_num;
        for cs in 0..compact_num {
            let orig = compact_to_original[cs];
            let t = dfa.trans_by_class[src_base + orig];
            if t != NONE {
                let mapped = original_to_compact[t as usize];
                // Only keep transition if target is in the compact set.
                if mapped != u32::MAX {
                    compact_trans[dst_base + cs] = mapped;
                }
            }
        }
    }

    // Compact per-state arrays.
    let compact_finalizers: Vec<SmallVec<[usize; 4]>> =
        compact_to_original.iter().map(|&s| dfa.finalizers[s].clone()).collect();
    let compact_is_dead_end: Vec<bool> =
        compact_to_original.iter().map(|&s| dfa.is_dead_end[s]).collect();
    let compact_completion_hash: Vec<u64> =
        compact_to_original.iter().map(|&s| dfa.completion_hash[s]).collect();
    let compact_self_loop: Vec<U8Set> =
        compact_to_original.iter().map(|&s| dfa.self_loop_bytes[s]).collect();
    let compact_pfg: Vec<SmallVec<[usize; 4]>> =
        compact_to_original.iter().map(|&s| dfa.possible_future_groups[s].clone()).collect();

    // Remap initial states.
    let compact_initial: Vec<usize> = initial_states
        .iter()
        .map(|&s| original_to_compact[s] as usize)
        .collect();

    let compact_dfa = Dfa {
        start_state: if dfa.start_state < original_to_compact.len() && original_to_compact[dfa.start_state] != u32::MAX {
            original_to_compact[dfa.start_state] as usize
        } else {
            0
        },
        num_states: compact_num,
        byte_to_class: dfa.byte_to_class,
        trans_by_class: Arc::from(compact_trans),
        finalizers: compact_finalizers,
        is_dead_end: compact_is_dead_end,
        num_groups: dfa.num_groups,
        possible_future_groups: compact_pfg,
        completion_hash: compact_completion_hash,
        none_completion_hash: dfa.none_completion_hash,
        self_loop_bytes: Arc::from(compact_self_loop),
        disallowed_follows: dfa.disallowed_follows.clone(),
    };

    Some((compact_dfa, compact_initial, compact_to_original))
}

/// When `active_groups` is provided, the DFA only tracks groups marked `true`.
/// L1 terminal groups can be excluded this way for a L2+-only analysis.
pub fn find_vocab_equivalence_classes_with_group_filter_profiled<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    byte_to_class: Option<&[u8; 256]>,
    active_groups: Option<&[bool]>,
    shared_cache: Option<&SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&SharedVocabAnalysisDfaCache>,
) -> (VocabEquivalenceResult, f64) {
    find_vocab_equivalence_classes_with_group_filter_profiled_impl(
        tokenizer,
        strings,
        initial_states,
        disallowed_follows,
        byte_to_class,
        active_groups,
        shared_cache,
        shared_analysis_dfa_cache,
        FirstTransitionFactorMode::Environment,
        None,
        false,
    )
}

/// Return only the exact-safe first-transition token prepartition.
///
/// The exact authority pass may merge these preliminary classes but does not
/// split them, so this is a conservative refinement of the final vocabulary
/// quotient. If factorization cannot prove a merge, return singletons.
pub fn find_vocab_first_transition_preclasses_with_group_filter_profiled<
    S: AsRef<[u8]> + Sync,
>(
    tokenizer: &TokenizerView,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    byte_to_class: Option<&[u8; 256]>,
    active_groups: Option<&[bool]>,
    shared_cache: Option<&SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&SharedVocabAnalysisDfaCache>,
) -> (VocabEquivalenceResult, f64) {
    find_vocab_equivalence_classes_with_group_filter_profiled_impl(
        tokenizer,
        strings,
        initial_states,
        disallowed_follows,
        byte_to_class,
        active_groups,
        shared_cache,
        shared_analysis_dfa_cache,
        FirstTransitionFactorMode::Force,
        None,
        true,
    )
}

/// Run the exact first-transition token prepartition, let the caller refine the
/// source-state coordinate using those token representatives, then resume the
/// ordinary exact token authority pass on one source position per returned
/// class. Returned positions index `initial_states`; the engine remaps them into
/// any internal compact-DFA coordinate without exposing that implementation
/// detail to the caller.
pub fn find_vocab_equivalence_classes_with_factor_state_refinement_profiled<
    S: AsRef<[u8]> + Sync,
    F: FnMut(&[usize]) -> Vec<usize>,
>(
    tokenizer: &TokenizerView,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    byte_to_class: Option<&[u8; 256]>,
    active_groups: Option<&[bool]>,
    shared_cache: Option<&SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&SharedVocabAnalysisDfaCache>,
    mut refine_states: F,
) -> (VocabEquivalenceResult, f64) {
    find_vocab_equivalence_classes_with_group_filter_profiled_impl(
        tokenizer,
        strings,
        initial_states,
        disallowed_follows,
        byte_to_class,
        active_groups,
        shared_cache,
        shared_analysis_dfa_cache,
        FirstTransitionFactorMode::Environment,
        Some(&mut refine_states),
        false,
    )
}

fn find_vocab_equivalence_classes_with_group_filter_profiled_impl<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    byte_to_class: Option<&[u8; 256]>,
    active_groups: Option<&[bool]>,
    shared_cache: Option<&SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&SharedVocabAnalysisDfaCache>,
    first_transition_factor_mode: FirstTransitionFactorMode,
    mut factor_state_refiner: Option<&mut dyn FnMut(&[usize]) -> Vec<usize>>,
    return_factor_prepartition: bool,
) -> (VocabEquivalenceResult, f64) {
    let input_state_count = tokenizer.dfa().states.len();
    if let Some((position, &state)) = initial_states
        .iter()
        .enumerate()
        .find(|(_, state)| **state >= input_state_count)
    {
        crate::error::fail_internal_invariant(format!(
            "vocabulary-equivalence input state is outside the analysis-view domain: \
             position={position} state={state} state_count={input_state_count} \
             sentinel={} token_count={}",
            state == u32::MAX as usize,
            strings.len(),
        ));
    }

    let profiling = compile_profile_enabled();
    let elapsed_ms = |started_at: Option<Instant>| {
        started_at.map_or(0.0, |instant| instant.elapsed().as_secs_f64() * 1000.0)
    };

    let total_started_at = profiling.then(Instant::now);
    if let Some(identity) = try_pre_dfa_singleton_identity_probe(
        tokenizer,
        strings,
        initial_states,
        disallowed_follows,
        active_groups,
        profiling,
    ) {
        return identity;
    }
    let build_dfa_started_at = Instant::now();
    let input_compacted = if shared_analysis_dfa_cache.is_none()
        && shared_cache.is_none()
        && active_groups.is_none()
        && strings.len() <= INPUT_VIEW_COMPACT_MAX_TOKENS
    {
        compact_tokenizer_view_for_tokens(tokenizer, initial_states, strings)
    } else {
        None
    };
    let (tokenizer_for_dfa, initial_states_for_dfa, input_compact_to_original): (
        &TokenizerView,
        &[usize],
        Option<&Vec<usize>>,
    ) = if let Some((ref compact_view, ref compact_initial, ref compact_to_original)) = input_compacted {
        (compact_view, compact_initial, Some(compact_to_original))
    } else {
        (tokenizer, initial_states, None)
    };
    let dfa = if let Some(cache) = shared_analysis_dfa_cache {
        let key = AnalysisDfaCacheKey::for_analysis_view(
            tokenizer_for_dfa,
            active_groups,
            disallowed_follows,
        );
        cache.get_or_init(key, || {
            build_dfa_with_group_filter(
                tokenizer_for_dfa,
                disallowed_follows,
                byte_to_class,
                active_groups,
                shared_cache,
            )
        })
    } else {
        Arc::new(build_dfa_with_group_filter(
            tokenizer_for_dfa,
            disallowed_follows,
            if input_compact_to_original.is_some() {
                None
            } else {
                byte_to_class
            },
            active_groups,
            shared_cache,
        ))
    };
    let build_dfa_ms = build_dfa_started_at.elapsed().as_secs_f64() * 1000.0;

    // Compact DFA: restrict to states reachable from initial_states via the
    // partition's token bytes.  This can dramatically shrink the transition
    // table (e.g. 77260 → 36598 for p0 of o62058) improving cache locality.
    let compact_dfa_started_at = profiling.then(Instant::now);
    let compacted = compact_dfa_for_tokens(dfa.as_ref(), initial_states_for_dfa, strings);
    let compact_dfa_ms = elapsed_ms(compact_dfa_started_at);
    let (dfa_ref, initial_states_ref, compact_to_original): (&Dfa, &[usize], Option<&Vec<usize>>) = if let Some((ref cdfa, ref cstates, ref compact_to_original)) = compacted {
        (cdfa, cstates, Some(compact_to_original))
    } else {
        (&dfa, initial_states_for_dfa, None)
    };
    if let Some((position, &state)) = initial_states_ref
        .iter()
        .enumerate()
        .find(|(_, state)| **state >= dfa_ref.num_states)
    {
        crate::error::fail_internal_invariant(format!(
            "vocabulary-equivalence state remapping produced an invalid compact-DFA coordinate: \
             position={position} state={state} compact_state_count={} source_state_count={} \
             compacted={} sentinel={} token_count={}",
            dfa_ref.num_states,
            dfa.num_states,
            compact_to_original.is_some(),
            state == u32::MAX as usize,
            strings.len(),
        ));
    }
    let compacted_states = compact_to_original.map_or(dfa.num_states, |states| states.len());
    let num_tokens = strings.len();
    let num_initial_states = initial_states_ref.len();

    if num_initial_states == 0 || num_tokens == 0 {
        return (BTreeSet::from_iter(vec![(0..num_tokens).collect()]), build_dfa_ms);
    }

    if initial_states_ref
        .iter()
        .all(|&state| dfa_ref.is_dead_end[state])
    {
        if profiling {
            eprintln!(
                "[glrmask/profile][vocab_equiv] strings={} initial_states={} batches=0 used_trie_walk=false active_final={} original_states={} effective_states={} compacted={} build_dfa_ms={:.3} compact_dfa_ms={:.3} state_order_ms=0.000 sort_tokens_ms=0.000 signature_ms=0.000 refinement_ms=0.000 final_groups_ms=0.000 dfs_step_ms=0.000 collect_targets_ms=0.000 single_target_suffix_ms=0.000 multi_target_suffix_ms=0.000 finish_signature_ms=0.000 dfs_steps=0 dfs_steps_without_new_dirty=0 dfs_states_visited=0 dfs_dead_transitions=0 dfs_dead_without_new_dirty=0 dfs_new_dirty_groups=0 dfs_new_dirty_states=0 clean_tokens={} dirty_tokens=0 single_target_tokens=0 multi_target_tokens=0 total_targets=0 total_ms={:.3}",
                num_tokens,
                num_initial_states,
                num_tokens,
                dfa.num_states,
                compacted_states,
                compact_to_original.is_some(),
                build_dfa_ms,
                compact_dfa_ms,
                num_tokens,
                elapsed_ms(total_started_at),
            );
        }
        return (BTreeSet::from_iter(vec![(0..num_tokens).collect()]), build_dfa_ms);
    }

    let lexical_order_started_at = profiling.then(Instant::now);
    let factor_enabled = match first_transition_factor_mode {
        FirstTransitionFactorMode::Disabled => false,
        FirstTransitionFactorMode::Environment => first_transition_factor_enabled(),
        FirstTransitionFactorMode::Force => true,
    };
    let lexical_order_needed = (num_tokens >= TRIE_WALK_MIN_TOKENS && !*TRIE_WALK_DISABLED)
        || factor_enabled;
    let lexical_order = lexical_order_needed.then(|| token_indices_in_lexical_order(strings));
    let mut sort_tokens_ms = elapsed_ms(lexical_order_started_at);
    let scratch_pool = Arc::new(ScratchPool::default());

    let preliminary_factor_started_at = profiling.then(Instant::now);
    let factor_plan = factor_enabled.then(|| {
        try_first_transition_factor_plan(
            dfa_ref,
            strings,
            initial_states_ref,
            lexical_order
                .as_deref()
                .expect("first-transition factorization requires lexical token order"),
            &scratch_pool,
            profiling,
            first_transition_factor_mode == FirstTransitionFactorMode::Environment,
        )
    });
    let factor_plan = factor_plan.flatten();
    let preliminary_factor_ms = elapsed_ms(preliminary_factor_started_at);
    if profiling {
        if factor_enabled {
            eprintln!(
                "[glrmask/profile][vocab_first_transition_factor_attempt] strings={} initial_states={} selected={} total_ms={:.3}",
                num_tokens,
                num_initial_states,
                factor_plan.is_some(),
                preliminary_factor_ms,
            );
        }
        if let Some(plan) = factor_plan.as_ref() {
            let stats = plan.stats;
            let reduction_pct = 100.0
                * (1.0
                    - stats.preliminary_state_token_pairs as f64
                        / stats.full_state_token_pairs.max(1) as f64);
            eprintln!(
                "[glrmask/profile][vocab_first_transition_factor] strings={} initial_states={} semantic_buckets={} factored_buckets={} parallel_buckets={} min_bucket_tokens={} source_state_buckets_before={} source_state_buckets_after={} full_state_token_pairs={} preliminary_state_token_pairs={} work_reduction_pct={:.2} preliminary_classes={} representative_tokens={} setup_ms={:.3} signature_cpu_ms={:.3} grouping_cpu_ms={:.3} wall_ms={:.3}",
                num_tokens,
                num_initial_states,
                stats.semantic_buckets,
                stats.factored_buckets,
                stats.parallel_buckets,
                stats.min_bucket_tokens,
                stats.source_state_buckets_before,
                stats.source_state_buckets_after,
                stats.full_state_token_pairs,
                stats.preliminary_state_token_pairs,
                reduction_pct,
                stats.preliminary_classes,
                plan.representative_tokens.len(),
                stats.setup_ms,
                stats.preliminary_signature_ms,
                stats.preliminary_grouping_ms,
                preliminary_factor_ms,
            );
        }
    }

    if return_factor_prepartition {
        let classes = factor_plan.map_or_else(
            || (0..num_tokens).map(|token| vec![token]).collect::<Vec<_>>(),
            |plan| plan.preliminary_classes,
        );
        return (BTreeSet::from_iter(classes), build_dfa_ms);
    }

    let factor_authority_states = factor_plan.as_ref().and_then(|plan| {
        factor_state_refiner.as_mut().map(|refine_states| {
            let positions = refine_states(&plan.representative_tokens);
            assert!(!positions.is_empty(), "factor state refinement returned no authority states");
            let mut seen = vec![false; initial_states_ref.len()];
            let mut states = Vec::with_capacity(positions.len());
            for position in positions {
                assert!(
                    position < initial_states_ref.len(),
                    "factor state authority position {position} exceeds input state count {}",
                    initial_states_ref.len(),
                );
                if !seen[position] {
                    seen[position] = true;
                    states.push(initial_states_ref[position]);
                }
            }
            states
        })
    });
    let authority_states = factor_authority_states
        .as_deref()
        .unwrap_or(initial_states_ref);
    if profiling && factor_plan.is_some() && factor_state_refiner.is_some() {
        eprintln!(
            "[glrmask/profile][vocab_factor_state_refinement] selected={} input_states={} authority_states={} preliminary_tokens={}",
            factor_authority_states.is_some(),
            num_initial_states,
            authority_states.len(),
            factor_plan.as_ref().map_or(num_tokens, |plan| plan.representative_tokens.len()),
        );
    }

    let analysis_token_count = factor_plan
        .as_ref()
        .map_or(num_tokens, |plan| plan.representative_tokens.len());
    let num_authority_states = authority_states.len();
    let state_order_started_at = profiling.then(Instant::now);
    let ordered_states = if diversity_state_order_enabled() {
        states_by_transition_diversity(dfa_ref, authority_states)
    } else {
        authority_states.to_vec()
    };
    let state_order_ms = elapsed_ms(state_order_started_at);
    let ordered_original_states = if let Some(compact_to_original) = compact_to_original {
        ordered_states
            .iter()
            .map(|&state| {
                let input_state = compact_to_original[state];
                input_compact_to_original
                    .map(|source| source[input_state])
                    .unwrap_or(input_state)
            })
            .collect::<Vec<_>>()
    } else if let Some(input_compact_to_original) = input_compact_to_original {
        ordered_states
            .iter()
            .map(|&state| input_compact_to_original[state])
            .collect::<Vec<_>>()
    } else {
        ordered_states.clone()
    };

    let num_groups = dfa_ref.num_groups;
    // Sparse live-state signatures make larger batches profitable by reducing
    // repeated trie traversal. Bound the default by the dominant
    // `match_positions` allocation so unusually wide terminal-group axes do
    // not inherit an unbounded memory increase. Keep the env override for A/B.
    let default_batch_size =
        default_vocab_batch_size(num_authority_states, num_groups, analysis_token_count);
    let factor_final_single_batch = factor_plan.is_some()
        && first_transition_factor_final_single_batch_enabled()
        && analysis_token_count.saturating_mul(num_authority_states)
            <= vocab_sequential_trie_work_max();
    let batch_size = vocab_batch_size_override().unwrap_or_else(|| {
        if factor_final_single_batch {
            num_authority_states
        } else {
            default_batch_size
        }
    });
    let mut active_indices: Vec<usize> = factor_plan.as_ref().map_or_else(
        || (0..num_tokens).collect(),
        |plan| plan.representative_tokens.clone(),
    );
    let mut active_tokens = vec![false; num_tokens];
    for &token_idx in &active_indices {
        active_tokens[token_idx] = true;
    }
    let mut partition = vec![0usize; num_tokens];
    let mut next_class_id = 1usize;
    let mut signature_ms = preliminary_factor_ms;
    let mut refinement_ms = 0.0;
    let mut batches = 0usize;
    let mut sequential_trie_batches = 0usize;
    let mut used_trie_walk = factor_plan
        .as_ref()
        .is_some_and(|plan| plan.stats.trie_walk.dfs_steps != 0);
    let mut trie_walk_stats = factor_plan
        .as_ref()
        .map_or_else(TrieWalkChunkStats::default, |plan| plan.stats.trie_walk);

    // Exact state-axis decomposition of the vocabulary observation matrix.
    //
    // The ordinary loop below refines token classes by one state batch at a
    // time. State batches do not depend on one another, so on the many-core
    // path we evaluate every batch concurrently and preserve their original
    // order. `append_state_signature_block` then recombines the slab hashes
    // algebraically into the exact same polynomial signature as one monolithic
    // state batch.
    // Each worker deliberately performs one sequential full-trie walk: the
    // outer state slabs are the parallelism, avoiding nested Rayon work on the
    // token axis.
    let parallel_state_batch_size = vocab_parallel_state_batch_size(
        num_authority_states,
        num_tokens,
    )
    .filter(|&size| {
        factor_plan.is_none()
            && num_tokens >= TRIE_WALK_MIN_TOKENS
            && !*TRIE_WALK_DISABLED
            && rayon::current_num_threads() > 1
            && num_authority_states > size
    });
    if let Some(parallel_batch_size) = parallel_state_batch_size {
        let use_parallel_singleton_probe = analysis_token_count <= SINGLETON_PROBE_MAX_TOKENS
            && num_authority_states > SINGLETON_PROBE_STATES
            && parallel_batch_size > SINGLETON_PROBE_STATES;
        let mut state_ranges = Vec::<std::ops::Range<usize>>::new();
        let mut start = 0usize;
        let mut batch_index = 0usize;
        while start < num_authority_states {
            let size = if batch_index == 0 && use_parallel_singleton_probe {
                SINGLETON_PROBE_STATES
            } else {
                parallel_batch_size
            };
            let end = (start + size).min(num_authority_states);
            state_ranges.push(start..end);
            start = end;
            batch_index += 1;
        }

        let sorted_indices = lexical_order
            .as_deref()
            .expect("parallel state batches require lexical token order");

        let signature_started_at = Instant::now();
        let batch_results = state_ranges
            .par_iter()
            .map(|range| {
                let batch = &ordered_states[range.clone()];
                let state_group_size = vocab_state_group_size(batch.len(), num_groups);
                let mut scratch = Scratch::new(batch.len(), num_groups);
                let mut trie_state = TrieWalkState::new();
                trie_walk_chunk_signatures(
                    dfa_ref,
                    strings,
                    &sorted_indices,
                    batch,
                    state_group_size,
                    &mut scratch,
                    &mut trie_state,
                    profiling,
                )
            })
            .collect::<Vec<_>>();
        let parallel_signature_wall_ms = signature_started_at.elapsed().as_secs_f64() * 1000.0;
        signature_ms += parallel_signature_wall_ms;
        batches += batch_results.len();
        sequential_trie_batches += batch_results.len();
        used_trie_walk = true;

        let refinement_started_at = Instant::now();
        let mut signatures_by_token = vec![HASH_SEED3; num_tokens];
        for (batch_index, (batch_signatures, batch_stats)) in
            batch_results.into_iter().enumerate()
        {
            if profiling {
                trie_walk_stats.add_assign(batch_stats);
            }
            debug_assert_eq!(batch_signatures.len(), num_tokens);
            let block_len = state_ranges[batch_index].len();
            let block_factor = state_signature_block_factor(block_len);
            for (token_idx, signature) in batch_signatures {
                signatures_by_token[token_idx] = append_state_signature_block(
                    signatures_by_token[token_idx],
                    signature,
                    block_factor,
                );
            }
        }

        let mut class_by_signature = HashMap::<u64, usize>::with_capacity(num_tokens);
        next_class_id = 0;
        for (token_idx, signature) in signatures_by_token.into_iter().enumerate() {
            let class_id = match class_by_signature.entry(signature) {
                hashbrown::hash_map::Entry::Occupied(entry) => *entry.get(),
                hashbrown::hash_map::Entry::Vacant(entry) => {
                    let class_id = next_class_id;
                    next_class_id += 1;
                    *entry.insert(class_id)
                }
            };
            partition[token_idx] = class_id;
        }
        active_indices.clear();
        active_tokens.fill(false);
        let parallel_refinement_ms = refinement_started_at.elapsed().as_secs_f64() * 1000.0;
        refinement_ms += parallel_refinement_ms;
        if profiling {
            eprintln!(
                "[glrmask/profile][vocab_parallel_state_batches] states={} tokens={} batch_size={} batches={} singleton_probe={} signature_wall_ms={:.3} refinement_ms={:.3} classes={}",
                num_authority_states,
                num_tokens,
                parallel_batch_size,
                batches,
                use_parallel_singleton_probe,
                parallel_signature_wall_ms,
                parallel_refinement_ms,
                next_class_id,
            );
        }
    }

    // A single Rayon worker previously rebuilt the full scratch arena for every
    // state batch. The arena is deliberately reset by the signature routines,
    // so one owner can safely reuse it across all batches without altering the
    // parallel path or the refinement semantics.
    let single_threaded = rayon::current_num_threads() == 1;
    let mut single_thread_scratch = single_threaded.then(|| Scratch::new(batch_size, num_groups));
    let mut single_thread_trie = single_threaded.then(TrieWalkState::new);

    // Small vocabularies often become provably singleton after observing only a
    // handful of high-diversity states. Start with a tiny witness batch, then
    // fall back to the normal large batches for any classes that remain
    // unresolved. A distinction found in the probe is permanent under further
    // refinement, so reaching `active_indices.is_empty()` is an exact identity
    // certificate, not a heuristic early exit.
    let use_singleton_probe = analysis_token_count <= SINGLETON_PROBE_MAX_TOKENS
        && num_authority_states > SINGLETON_PROBE_STATES
        && batch_size > SINGLETON_PROBE_STATES;
    let mut batch_start = 0usize;
    let mut batch_index = 0usize;
    while batch_start < num_authority_states {
        if active_indices.is_empty() {
            break;
        }
        batches += 1;

        let current_batch_size = if batch_index == 0 && use_singleton_probe {
            SINGLETON_PROBE_STATES
        } else {
            batch_size
        };
        let batch_end = (batch_start + current_batch_size).min(num_authority_states);
        let batch = &ordered_states[batch_start..batch_end];
        let state_group_size = vocab_state_group_size(batch.len(), num_groups);
        let use_trie_walk = active_indices.len() >= TRIE_WALK_MIN_TOKENS
            && !*TRIE_WALK_DISABLED;
        let use_sequential_trie = use_trie_walk
            && num_authority_states <= batch_size
            && vocab_sequential_trie_work_max() > 0
            && active_indices
                .len()
                .saturating_mul(batch.len())
                <= vocab_sequential_trie_work_max();
        sequential_trie_batches += usize::from(use_sequential_trie);
        used_trie_walk |= use_trie_walk;
        let active_sigs: Vec<(usize, u64)> = if use_trie_walk {
            let sort_started_at = profiling.then(Instant::now);
            let sorted_indices = active_indices_in_lexical_order(
                lexical_order
                    .as_deref()
                    .expect("trie walk must precompute lexical token order"),
                &active_tokens,
            );
            sort_tokens_ms += elapsed_ms(sort_started_at);
            debug_assert_eq!(sorted_indices.len(), active_indices.len());
            let signature_started_at = profiling.then(Instant::now);
            let mut flat_results = Vec::with_capacity(sorted_indices.len());
            if let (Some(scratch), Some(trie_state)) = (
                single_thread_scratch.as_mut(),
                single_thread_trie.as_mut(),
            ) {
                let (chunk_result, chunk_stats) = trie_walk_chunk_signatures(
                    dfa_ref,
                    strings,
                    &sorted_indices,
                    batch,
                    state_group_size,
                    scratch,
                    trie_state,
                    profiling,
                );
                flat_results.extend(chunk_result);
                if profiling {
                    trie_walk_stats.add_assign(chunk_stats);
                }
            } else if use_sequential_trie {
                let mut lease = scratch_pool.checkout(batch.len(), num_groups);
                let worker = lease.worker_mut();
                let (chunk_result, chunk_stats) = trie_walk_chunk_signatures(
                    dfa_ref,
                    strings,
                    &sorted_indices,
                    batch,
                    state_group_size,
                    &mut worker.scratch,
                    &mut worker.trie_state,
                    profiling,
                );
                flat_results.extend(chunk_result);
                if profiling {
                    trie_walk_stats.add_assign(chunk_stats);
                }
            } else {
                let scratch_pool = Arc::clone(&scratch_pool);
                let chunk_results: Vec<(Vec<(usize, u64)>, TrieWalkChunkStats)> = sorted_indices
                    .par_chunks(TRIE_CHUNK_SIZE)
                    .map_init(
                        || scratch_pool.checkout(batch.len(), num_groups),
                        |lease, chunk| {
                            let worker = lease.worker_mut();
                            trie_walk_chunk_signatures(
                                dfa_ref,
                                strings,
                                chunk,
                                batch,
                                state_group_size,
                                &mut worker.scratch,
                                &mut worker.trie_state,
                                profiling,
                            )
                        },
                    )
                    .collect();
                for (chunk_result, chunk_stats) in chunk_results {
                    flat_results.extend(chunk_result);
                    if profiling {
                        trie_walk_stats.add_assign(chunk_stats);
                    }
                }
            }
            signature_ms += elapsed_ms(signature_started_at);
            flat_results
        } else {
            let signature_started_at = profiling.then(Instant::now);
            // Dispatching one Rayon job per token is substantially more
            // expensive than the exact scan itself when both axes are tiny
            // (for example 58 tokens x 4 quotient states).  Select serial
            // execution from the actual state-token work, independent of
            // partition or grammar shape.  Larger matrices retain the
            // parallel path.
            const SEQUENTIAL_DIRECT_SIGNATURE_MAX_PAIRS: usize = 256;
            let use_sequential_direct = active_indices
                .len()
                .saturating_mul(batch.len())
                <= SEQUENTIAL_DIRECT_SIGNATURE_MAX_PAIRS;
            let result = if let Some(scratch) = single_thread_scratch.as_mut() {
                active_indices
                    .iter()
                    .map(|&token_idx| {
                        let token = strings[token_idx].as_ref();
                        (
                            token_idx,
                            token_signature(
                                dfa_ref,
                                token,
                                batch,
                                state_group_size,
                                scratch,
                                false,
                            ),
                        )
                    })
                    .collect()
            } else if use_sequential_direct {
                let mut scratch = Scratch::new(batch.len(), num_groups);
                active_indices
                    .iter()
                    .map(|&token_idx| {
                        let token = strings[token_idx].as_ref();
                        (
                            token_idx,
                            token_signature(
                                dfa_ref,
                                token,
                                batch,
                                state_group_size,
                                &mut scratch,
                                false,
                            ),
                        )
                    })
                    .collect()
            } else {
                let scratch_pool = Arc::clone(&scratch_pool);
                active_indices
                    .par_iter()
                    .map_init(
                        || scratch_pool.checkout(batch.len(), num_groups),
                        |lease, &token_idx| {
                            let token = strings[token_idx].as_ref();
                            let worker = lease.worker_mut();
                            (
                                token_idx,
                                token_signature(
                                    dfa_ref,
                                    token,
                                    batch,
                                    state_group_size,
                                    &mut worker.scratch,
                                    false,
                                ),
                            )
                        },
                    )
                    .collect()
            };
            signature_ms += elapsed_ms(signature_started_at);
            result
        };

        let refinement_started_at = profiling.then(Instant::now);
        let mut refinement: HashMap<(usize, u64), Vec<usize>> =
            HashMap::with_capacity(active_sigs.len() / 2);
        for (ti, sig) in active_sigs {
            refinement
                .entry((partition[ti], sig))
                .or_default()
                .push(ti);
        }

        let mut new_active = Vec::with_capacity(active_indices.len());
        let mut seen_classes = vec![false; next_class_id.max(1)];
        for ((old_class, _), tokens) in refinement {
            let class_id = if !seen_classes[old_class] {
                seen_classes[old_class] = true;
                old_class
            } else {
                let id = next_class_id;
                next_class_id += 1;
                id
            };
            for &ti in &tokens {
                partition[ti] = class_id;
            }
            if tokens.len() > 1 {
                new_active.extend(tokens);
            }
        }
        active_tokens.fill(false);
        for &token_idx in &new_active {
            active_tokens[token_idx] = true;
        }
        active_indices = new_active;
        refinement_ms += elapsed_ms(refinement_started_at);
        batch_start = batch_end;
        batch_index += 1;
    }

    let final_groups_started_at = profiling.then(Instant::now);
    let analyzed_tokens = factor_plan.as_ref().map_or_else(
        || (0..num_tokens).collect::<Vec<_>>(),
        |plan| plan.representative_tokens.clone(),
    );
    let mut representative_groups = vec![Vec::new(); next_class_id.max(1)];
    for token_idx in analyzed_tokens {
        representative_groups[partition[token_idx]].push(token_idx);
    }
    let representative_groups = representative_groups
        .into_iter()
        .filter(|group| !group.is_empty())
        .collect::<Vec<_>>();
    let groups = if let Some(plan) = factor_plan.as_ref() {
        representative_groups
            .into_iter()
            .map(|representatives| {
                let total_len = representatives
                    .iter()
                    .map(|&representative| {
                        let class_idx =
                            plan.preliminary_class_for_representative[representative];
                        debug_assert_ne!(class_idx, usize::MAX);
                        plan.preliminary_classes[class_idx].len()
                    })
                    .sum();
                let mut expanded = Vec::with_capacity(total_len);
                for representative in representatives {
                    let class_idx = plan.preliminary_class_for_representative[representative];
                    expanded.extend_from_slice(&plan.preliminary_classes[class_idx]);
                }
                expanded.sort_unstable();
                expanded
            })
            .collect::<Vec<_>>()
    } else {
        representative_groups
    };
    let final_groups_ms = elapsed_ms(final_groups_started_at);

    if let Some((left_token_idx, right_token_idx)) = vocab_verify_token_pair_override() {
        log_vocab_pair_verification(
            dfa_ref,
            strings,
            left_token_idx,
            right_token_idx,
            &ordered_states,
            &ordered_original_states,
            batch_size,
        );
    }

    if vocab_verify_token_pair_from_final_classes_enabled() {
        if let Some(group) = groups.iter().find(|group| group.len() >= 2) {
            log_vocab_pair_verification(
                dfa_ref,
                strings,
                group[0],
                group[1],
                &ordered_states,
                &ordered_original_states,
                batch_size,
            );
        }
        if groups.len() >= 2 {
            log_vocab_pair_verification(
                dfa_ref,
                strings,
                groups[0][0],
                groups[1][0],
                &ordered_states,
                &ordered_original_states,
                batch_size,
            );
        }
    }

    if *VOCAB_ROW_CERT_DIAG {
        run_vocab_row_cert_diag(
            dfa_ref,
            strings,
            &ordered_states,
            batch_size,
            |batch_len| vocab_state_group_size(batch_len, num_groups),
            &groups,
        );
    }

    let (scratch_pool_allocations, scratch_pool_reuses) = scratch_pool.stats();

    if profiling {
        eprintln!(
            "[glrmask/profile][vocab_equiv] strings={} initial_states={} batch_size={} factor_final_single_batch={} batches={} sequential_trie_batches={} used_trie_walk={} active_final={} original_states={} effective_states={} compacted={} build_dfa_ms={:.3} compact_dfa_ms={:.3} state_order_ms={:.3} sort_tokens_ms={:.3} signature_ms={:.3} refinement_ms={:.3} final_groups_ms={:.3} dfs_step_ms={:.3} collect_targets_ms={:.3} single_target_suffix_ms={:.3} multi_target_suffix_ms={:.3} finish_signature_ms={:.3} dfs_steps={} dfs_steps_without_new_dirty={} dfs_states_visited={} dfs_dead_transitions={} dfs_dead_without_new_dirty={} dfs_new_dirty_groups={} dfs_new_dirty_states={} dfs_noop_self_loops={} clean_tokens={} dirty_tokens={} single_target_tokens={} multi_target_tokens={} total_targets={} scratch_pool_allocations={} scratch_pool_reuses={} total_ms={:.3}",
            num_tokens,
            num_initial_states,
            batch_size,
            factor_final_single_batch,
            batches,
            sequential_trie_batches,
            used_trie_walk,
            active_indices.len(),
            dfa.num_states,
            dfa_ref.num_states,
            compact_to_original.is_some(),
            build_dfa_ms,
            compact_dfa_ms,
            state_order_ms,
            sort_tokens_ms,
            signature_ms,
            refinement_ms,
            final_groups_ms,
            trie_walk_stats.dfs_step_ms,
            trie_walk_stats.collect_targets_ms,
            trie_walk_stats.single_target_suffix_ms,
            trie_walk_stats.multi_target_suffix_ms,
            trie_walk_stats.finish_signature_ms,
            trie_walk_stats.dfs_steps,
            trie_walk_stats.dfs_steps_without_new_dirty,
            trie_walk_stats.dfs_states_visited,
            trie_walk_stats.dfs_dead_transitions,
            trie_walk_stats.dfs_dead_without_new_dirty,
            trie_walk_stats.dfs_new_dirty_groups,
            trie_walk_stats.dfs_new_dirty_states,
            trie_walk_stats.dfs_noop_self_loops,
            trie_walk_stats.clean_tokens,
            trie_walk_stats.dirty_tokens,
            trie_walk_stats.single_target_tokens,
            trie_walk_stats.multi_target_tokens,
            trie_walk_stats.total_targets,
            scratch_pool_allocations,
            scratch_pool_reuses,
            elapsed_ms(total_started_at),
        );
    }

    let result = groups.into_iter().collect::<VocabEquivalenceResult>();
    if first_transition_factor_mode != FirstTransitionFactorMode::Disabled
        && factor_plan.is_some()
        && first_transition_factor_strict_reference_enabled()
    {
        let strict_started_at = Instant::now();
        let (reference, _) = find_vocab_equivalence_classes_with_group_filter_profiled_impl(
            tokenizer,
            strings,
            initial_states,
            disallowed_follows,
            byte_to_class,
            active_groups,
            shared_cache,
            shared_analysis_dfa_cache,
            FirstTransitionFactorMode::Disabled,
            None,
            false,
        );
        assert_eq!(
            result, reference,
            "first-transition factored vocabulary partition differs from ordinary exact analysis",
        );
        if profiling {
            eprintln!(
                "[glrmask/profile][vocab_first_transition_factor_strict_reference] exact_classes={} reference_classes={} differs=false compare_ms={:.3}",
                result.len(),
                reference.len(),
                strict_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    (result, build_dfa_ms)
}

/// Result-only compatibility entry point. Callers that need precise phase
/// accounting should use the profiled sibling above.
pub fn find_vocab_equivalence_classes_with_group_filter<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    byte_to_class: Option<&[u8; 256]>,
    active_groups: Option<&[bool]>,
    shared_cache: Option<&SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&SharedVocabAnalysisDfaCache>,
) -> VocabEquivalenceResult {
    find_vocab_equivalence_classes_with_group_filter_profiled(
        tokenizer,
        strings,
        initial_states,
        disallowed_follows,
        byte_to_class,
        active_groups,
        shared_cache,
        shared_analysis_dfa_cache,
    )
    .0
}

#[cfg(test)]
mod shared_base_tests ;

#[cfg(test)]
mod scratch_pool_tests {
    use super::*;

    #[test]
    fn scratch_keeps_same_group_buffers_for_smaller_batches() {
        let mut scratch = Scratch::new(8, 5);
        let positions_ptr = scratch.match_positions.as_ptr();
        scratch.ensure_capacity(3, 5);
        assert_eq!(scratch.match_positions.as_ptr(), positions_ptr);
        assert!(scratch.current_states.len() >= 8);

        scratch.ensure_capacity(12, 5);
        assert!(scratch.current_states.len() >= 12);
        assert!(scratch.match_positions.len() >= 12 * 5);

        scratch.ensure_capacity(4, 6);
        assert_eq!(scratch.current_states.len(), 4);
        assert_eq!(scratch.match_positions.len(), 4 * 6);
        assert!(scratch.match_positions.iter().all(|&value| value == NONE));
    }

    #[test]
    fn scratch_pool_returns_leased_workers() {
        let pool = ScratchPool::default();
        {
            let _lease = pool.checkout(4, 3);
        }
        {
            let _lease = pool.checkout(2, 3);
        }
        assert_eq!(pool.stats(), (1, 1));
    }
}
