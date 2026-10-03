//! Byte-oriented lexer DFA used by the tokenizer and lexer compiler.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::ops::Index;
use std::sync::OnceLock;

use rustc_hash::{FxHashSet, FxHasher};
use rayon::prelude::*;
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smallvec::SmallVec;

use crate::ds::char_transitions::CharTransitions;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;

pub(super) type GroupId = u32;
pub(super) const DEAD: u32 = u32::MAX;

/// Exact singleton epsilon closures with a compact identity representation for
/// the overwhelmingly common case where only a dispatch state has a
/// non-singleton closure.  This avoids one heap allocation per raw tokenizer
/// state while preserving the slice-based caller API.
#[derive(Debug)]
pub enum SingletonEpsilonClosures {
    Dense(Box<[Box<[u32]>]>),
    IdentityWithOverrides {
        identities: Box<[u32]>,
        overrides: Box<[Option<Box<[u32]>>]>,
    },
}

impl SingletonEpsilonClosures {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Dense(rows) => rows.len(),
            Self::IdentityWithOverrides { identities, .. } => identities.len(),
        }
    }

    #[inline]
    pub fn get(&self, state: usize) -> Option<&[u32]> {
        match self {
            Self::Dense(rows) => rows.get(state).map(Box::as_ref),
            Self::IdentityWithOverrides {
                identities,
                overrides,
            } => {
                if state >= identities.len() {
                    None
                } else if let Some(closure) = overrides[state].as_deref() {
                    Some(closure)
                } else {
                    Some(&identities[state..state + 1])
                }
            }
        }
    }

    #[inline]
    pub fn iter(&self) -> SingletonEpsilonClosuresIter<'_> {
        SingletonEpsilonClosuresIter {
            closures: self,
            next: 0,
        }
    }

    /// Reuse an owned parent closure table while appending rebased child
    /// tables. This is exact for disjoint tokenizer composition: parent state
    /// IDs are unchanged, child state IDs receive fixed offsets, and only the
    /// retained parent reset state's closure gains the child start closures.
    pub fn append_rebased_children(
        self,
        parent_start: u32,
        children: &[(std::sync::Arc<SingletonEpsilonClosures>, u32, u32)],
    ) -> Option<Self> {
        let Self::IdentityWithOverrides {
            identities,
            overrides,
        } = self
        else {
            return None;
        };
        let mut identities = identities.into_vec();
        let mut overrides = overrides.into_vec();
        let parent_start_index = parent_start as usize;
        if parent_start_index >= identities.len() {
            return None;
        }
        let mut merged_start = overrides[parent_start_index]
            .as_deref()
            .unwrap_or(&identities[parent_start_index..parent_start_index + 1])
            .to_vec();

        for (child, state_offset, child_start) in children {
            if *state_offset as usize != identities.len() {
                return None;
            }
            let child_len = child.len();
            identities.extend(*state_offset..state_offset.saturating_add(child_len as u32));
            for local_state in 0..child_len {
                let closure = child.get(local_state)?;
                if closure.len() == 1 && closure[0] == local_state as u32 {
                    overrides.push(None);
                } else {
                    overrides.push(Some(
                        closure
                            .iter()
                            .map(|state| state_offset.saturating_add(*state))
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                    ));
                }
            }
            let child_start_closure = child.get(*child_start as usize)?;
            merged_start.extend(
                child_start_closure
                    .iter()
                    .map(|state| state_offset.saturating_add(*state)),
            );
        }
        merged_start.sort_unstable();
        merged_start.dedup();
        overrides[parent_start_index] = Some(merged_start.into_boxed_slice());
        Some(Self::IdentityWithOverrides {
            identities: identities.into_boxed_slice(),
            overrides: overrides.into_boxed_slice(),
        })
    }
}

impl Index<usize> for SingletonEpsilonClosures {
    type Output = [u32];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
            .expect("singleton epsilon closure state out of range")
    }
}

pub struct SingletonEpsilonClosuresIter<'a> {
    closures: &'a SingletonEpsilonClosures,
    next: usize,
}

impl<'a> Iterator for SingletonEpsilonClosuresIter<'a> {
    type Item = &'a [u32];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let state = self.next;
        let closure = self.closures.get(state)?;
        self.next += 1;
        Some(closure)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.closures.len().saturating_sub(self.next);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for SingletonEpsilonClosuresIter<'_> {}

fn resized_bitset(bits: &BitSet, num_groups: usize) -> BitSet {
    let mut resized = BitSet::new(num_groups);
    for bit in bits.iter() {
        resized.set(bit);
    }
    resized
}

fn project_bitset(bits: &BitSet, num_groups: usize) -> BitSet {
    let mut projected = BitSet::new(num_groups);
    for group_id in bits.iter().filter(|group_id| *group_id < num_groups) {
        projected.set(group_id);
    }
    projected
}

fn excluded_group_indices(
    finalizers: &BitSet,
    excludes: &BTreeMap<GroupId, BTreeSet<GroupId>>,
) -> Vec<usize> {
    let mut to_clear = Vec::new();
    for (&group_id, blocked_by) in excludes {
        let group_index = group_id as usize;
        if !finalizers.contains(group_index) {
            continue;
        }
        if blocked_by
            .iter()
            .any(|blocked_by_id| finalizers.contains(*blocked_by_id as usize))
        {
            to_clear.push(group_index);
        }
    }
    to_clear
}

fn intersection_missing_group_indices(
    finalizers: &BitSet,
    intersections: &BTreeMap<GroupId, BTreeSet<GroupId>>,
) -> Vec<usize> {
    let mut to_clear = Vec::new();
    for (&group_id, required) in intersections {
        let group_index = group_id as usize;
        if !finalizers.contains(group_index) {
            continue;
        }
        if required
            .iter()
            .any(|required_id| !finalizers.contains(*required_id as usize))
        {
            to_clear.push(group_index);
        }
    }
    to_clear
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub(super) struct DFAState {
    pub(super) transitions: CharTransitions<u32>,
    pub(super) finalizers: BitSet,
    pub(super) possible_future_group_ids: BitSet,
    /// Epsilon transitions are the only source of lexer nondeterminism. Byte
    /// transitions remain deterministic within an individual physical state.
    pub(super) epsilon_transitions: Vec<u32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct DfaDerivedStats {
    byte_transition_count: usize,
    epsilon_transition_count: usize,
    has_self_loops: bool,
}

#[derive(Clone, Default)]
pub struct DFA {
    states: Vec<DFAState>,
    group_id_to_u8set: Vec<U8Set>,
    derived_stats: OnceLock<DfaDerivedStats>,
    min_match_byte_len_cache: OnceLock<Option<usize>>,
    structural_hash_cache: OnceLock<u64>,
}

impl PartialEq for DFA {
    fn eq(&self, other: &Self) -> bool {
        self.states == other.states && self.group_id_to_u8set == other.group_id_to_u8set
    }
}

impl Eq for DFA {}

impl Hash for DFA {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.structural_hash().hash(state);
    }
}

/// The historical persisted representation of a lexer state. Keep this exact
/// field order: existing bincode artifacts encode nested structs without field
/// boundaries, so adding even a trailing field would consume data belonging to
/// the next state or the enclosing DFA.
#[derive(Serialize, Deserialize)]
struct DfaStateWire {
    transitions: CharTransitions<u32>,
    finalizers: BitSet,
    possible_future_group_ids: BitSet,
}

/// The historical persisted representation of a lexer DFA. Epsilon metadata is
/// encoded as validated synthetic trailing states, leaving the outer wire shape
/// and every real state byte-for-byte compatible with pre-epsilon artifacts.
#[derive(Serialize, Deserialize)]
struct DfaWire {
    states: Vec<DfaStateWire>,
    group_id_to_u8set: Vec<U8Set>,
}

#[derive(Serialize)]
struct DfaStateWireRef<'a> {
    transitions: &'a CharTransitions<u32>,
    finalizers: &'a BitSet,
    possible_future_group_ids: &'a BitSet,
}

struct DfaStatesWireRef<'a> {
    dfa: &'a DFA,
    metadata_state_count: usize,
}

impl Serialize for DfaStatesWireRef<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let trailer_count = usize::from(self.metadata_state_count != 0);
        let total_states = self
            .dfa
            .states
            .len()
            .checked_add(self.metadata_state_count)
            .and_then(|count| count.checked_add(trailer_count))
            .expect("serialized DFA state count should fit in usize");
        let mut sequence = serializer.serialize_seq(Some(total_states))?;

        for state in &self.dfa.states {
            sequence.serialize_element(&DfaStateWireRef {
                transitions: &state.transitions,
                finalizers: &state.finalizers,
                possible_future_group_ids: &state.possible_future_group_ids,
            })?;
        }

        for (source, state) in self.dfa.states.iter().enumerate() {
            for targets in state
                .epsilon_transitions
                .chunks(EPSILON_TARGETS_PER_WIRE_STATE)
            {
                if targets.is_empty() {
                    continue;
                }
                let mut entries = Vec::with_capacity(targets.len() + 3);
                entries.push((0, EPSILON_WIRE_MARKER));
                entries.push((1, EPSILON_WIRE_EDGE));
                entries.push((2, source as u32));
                entries.extend(
                    targets
                        .iter()
                        .enumerate()
                        .map(|(index, &target)| ((index + 3) as u8, target)),
                );
                sequence.serialize_element(&epsilon_wire_state(entries))?;
            }
        }

        if self.metadata_state_count != 0 {
            sequence.serialize_element(&epsilon_wire_state(vec![
                (0, EPSILON_WIRE_MARKER),
                (1, EPSILON_WIRE_TRAILER),
                (2, self.dfa.states.len() as u32),
                (3, self.metadata_state_count as u32),
            ]))?;
        }

        sequence.end()
    }
}

#[derive(Serialize)]
struct DfaWireRef<'a> {
    states: DfaStatesWireRef<'a>,
    group_id_to_u8set: &'a [U8Set],
}

const EPSILON_WIRE_MARKER: u32 = u32::MAX;
const EPSILON_WIRE_EDGE: u32 = 0x4550_5345; // "EPSE"
const EPSILON_WIRE_TRAILER: u32 = 0x4550_5354; // "EPST"
const EPSILON_TARGETS_PER_WIRE_STATE: usize = 253;

impl From<&DFAState> for DfaStateWire {
    fn from(state: &DFAState) -> Self {
        Self {
            transitions: state.transitions.clone(),
            finalizers: state.finalizers.clone(),
            possible_future_group_ids: state.possible_future_group_ids.clone(),
        }
    }
}

impl From<DfaStateWire> for DFAState {
    fn from(state: DfaStateWire) -> Self {
        Self {
            transitions: state.transitions,
            finalizers: state.finalizers,
            possible_future_group_ids: state.possible_future_group_ids,
            epsilon_transitions: Vec::new(),
        }
    }
}

fn epsilon_wire_state(entries: Vec<(u8, u32)>) -> DfaStateWire {
    DfaStateWire {
        transitions: CharTransitions::from_sorted_entries(entries),
        finalizers: BitSet::new(0),
        possible_future_group_ids: BitSet::new(0),
    }
}

fn epsilon_wire_trailer(state: &DfaStateWire) -> Result<Option<(usize, usize)>, &'static str> {
    let marker = state.transitions.get(0).copied() == Some(EPSILON_WIRE_MARKER);
    let trailer = state.transitions.get(1).copied() == Some(EPSILON_WIRE_TRAILER);
    if !marker || !trailer {
        return Ok(None);
    }
    if !state.finalizers.is_empty()
        || !state.possible_future_group_ids.is_empty()
        || state.transitions.len() != 4
    {
        return Err("malformed lexer epsilon metadata trailer");
    }
    let real_states = state
        .transitions
        .get(2)
        .copied()
        .ok_or("lexer epsilon metadata trailer omitted real-state count")?
        as usize;
    let metadata_states = state
        .transitions
        .get(3)
        .copied()
        .ok_or("lexer epsilon metadata trailer omitted metadata-state count")?
        as usize;
    Ok(Some((real_states, metadata_states)))
}

fn decode_epsilon_wire_states(
    states: &mut Vec<DfaStateWire>,
) -> Result<Option<Vec<Vec<u32>>>, &'static str> {
    let Some(last) = states.last() else {
        return Ok(None);
    };
    let Some((real_state_count, metadata_state_count)) = epsilon_wire_trailer(last)? else {
        return Ok(None);
    };
    if metadata_state_count == 0
        || real_state_count
            .checked_add(metadata_state_count)
            .and_then(|count| count.checked_add(1))
            != Some(states.len())
    {
        return Err("lexer epsilon metadata counts do not match serialized state count");
    }

    let mut epsilon_transitions = vec![Vec::new(); real_state_count];
    let mut seen_targets = vec![None::<FxHashSet<u32>>; real_state_count];
    for state in &states[real_state_count..real_state_count + metadata_state_count] {
        if !state.finalizers.is_empty()
            || !state.possible_future_group_ids.is_empty()
            || state.transitions.len() < 4
            || state.transitions.get(0).copied() != Some(EPSILON_WIRE_MARKER)
            || state.transitions.get(1).copied() != Some(EPSILON_WIRE_EDGE)
        {
            return Err("malformed lexer epsilon metadata state");
        }
        let source = state
            .transitions
            .get(2)
            .copied()
            .ok_or("lexer epsilon metadata omitted source state")?
            as usize;
        if source >= real_state_count {
            return Err("lexer epsilon metadata source is out of range");
        }
        let mut targets_in_state = 0usize;
        for (index, (byte, &target)) in state.transitions.iter().skip(3).enumerate() {
            if byte as usize != index + 3 {
                return Err("lexer epsilon metadata target slots are not contiguous");
            }
            if target as usize >= real_state_count {
                return Err("lexer epsilon metadata target is out of range");
            }
            if !seen_targets[source]
                .get_or_insert_with(FxHashSet::default)
                .insert(target)
            {
                return Err("lexer epsilon metadata contains a duplicate edge");
            }
            epsilon_transitions[source].push(target);
            targets_in_state += 1;
        }
        if targets_in_state == 0 {
            return Err("lexer epsilon metadata state contains no targets");
        }
    }

    states.truncate(real_state_count);
    Ok(Some(epsilon_transitions))
}

impl Serialize for DFA {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error;

        if self
            .states
            .iter()
            .flat_map(|state| state.transitions.values())
            .any(|&target| target == EPSILON_WIRE_MARKER)
        {
            return Err(S::Error::custom(
                "lexer DFA contains the reserved dead transition target",
            ));
        }

        let metadata_state_count = self
            .states
            .iter()
            .map(|state| {
                state
                    .epsilon_transitions
                    .len()
                    .div_ceil(EPSILON_TARGETS_PER_WIRE_STATE)
            })
            .sum();

        DfaWireRef {
            states: DfaStatesWireRef {
                dfa: self,
                metadata_state_count,
            },
            group_id_to_u8set: &self.group_id_to_u8set,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for DFA {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut wire = DfaWire::deserialize(deserializer)?;
        let epsilon_transitions = decode_epsilon_wire_states(&mut wire.states)
            .map_err(serde::de::Error::custom)?;
        let mut states = wire
            .states
            .into_iter()
            .map(DFAState::from)
            .collect::<Vec<_>>();
        if let Some(epsilon_transitions) = epsilon_transitions {
            for (state, epsilon_transitions) in
                states.iter_mut().zip(epsilon_transitions.into_iter())
            {
                state.epsilon_transitions = epsilon_transitions;
            }
        }
        Ok(Self {
            states,
            group_id_to_u8set: wire.group_id_to_u8set,
            derived_stats: OnceLock::new(),
            min_match_byte_len_cache: OnceLock::new(),
            structural_hash_cache: OnceLock::new(),
        })
    }
}

impl std::fmt::Debug for DFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DFA { .. }")
    }
}

impl DFA {
    pub(super) fn new(num_states: usize) -> Self {
        Self {
            states: vec![DFAState::default(); num_states],
            group_id_to_u8set: Vec::new(),
            derived_stats: OnceLock::new(),
            min_match_byte_len_cache: OnceLock::new(),
            structural_hash_cache: OnceLock::new(),
        }
    }

    /// Build the metadata-only DFA representation used by current static
    /// artifacts in one pass. Byte transitions live in the packed runtime
    /// sidecar, so constructing default states and then resizing/installing
    /// their terminal metadata is unnecessary load-time traffic.
    pub(super) fn new_from_sparse_metadata(
        group_id_to_u8set: Vec<U8Set>,
        epsilon_offsets: &[u32],
        epsilon_targets: &[u32],
        finalizer_offsets: &[u32],
        finalizers: &[u32],
        future_offsets: &[u32],
        futures: &[u32],
    ) -> Self {
        let state_count = epsilon_offsets.len().saturating_sub(1);
        debug_assert_eq!(finalizer_offsets.len(), state_count + 1);
        debug_assert_eq!(future_offsets.len(), state_count + 1);
        let group_count = group_id_to_u8set.len();
        let build_state = |state: usize| {
            let a = finalizer_offsets[state] as usize;
            let b = finalizer_offsets[state + 1] as usize;
            let finalizer_bits = BitSet::from_sparse_u32(group_count, &finalizers[a..b]);

            let a = future_offsets[state] as usize;
            let b = future_offsets[state + 1] as usize;
            let future_bits = BitSet::from_sparse_u32(group_count, &futures[a..b]);

            let a = epsilon_offsets[state] as usize;
            let b = epsilon_offsets[state + 1] as usize;
            DFAState {
                transitions: CharTransitions::default(),
                finalizers: finalizer_bits,
                possible_future_group_ids: future_bits,
                epsilon_transitions: epsilon_targets[a..b].to_vec(),
            }
        };
        const PARALLEL_SPARSE_METADATA_THRESHOLD: usize = 16_384;
        let states = if state_count >= PARALLEL_SPARSE_METADATA_THRESHOLD
            && rayon::current_num_threads() > 1
        {
            (0..state_count).into_par_iter().map(build_state).collect()
        } else {
            (0..state_count).map(build_state).collect()
        };
        Self {
            states,
            group_id_to_u8set,
            derived_stats: OnceLock::new(),
            min_match_byte_len_cache: OnceLock::new(),
            structural_hash_cache: OnceLock::new(),
        }
    }

    /// Construct a one-group DFA whose per-state acceptance and strict-future
    /// metadata are already known.
    ///
    /// Large compressed intersection products otherwise create default states,
    /// resize both bitsets for every state, then traverse the whole state array
    /// twice more to install acceptance and future flags. Building the final
    /// representation in one pass avoids that repeated memory traffic while
    /// preserving the ordinary DFA representation and wire format.
    pub(super) fn new_with_single_group_metadata(
        accepting: &[bool],
        possible_future: &[bool],
        group_u8set: U8Set,
    ) -> Self {
        assert_eq!(accepting.len(), possible_future.len());
        let build_state = |accepting: bool, possible_future: bool| {
            let mut finalizers = BitSet::new(1);
            if accepting {
                finalizers.set(0);
            }
            let mut possible_future_group_ids = BitSet::new(1);
            if possible_future {
                possible_future_group_ids.set(0);
            }
            DFAState {
                transitions: CharTransitions::default(),
                finalizers,
                possible_future_group_ids,
                epsilon_transitions: Vec::new(),
            }
        };
        const PARALLEL_METADATA_THRESHOLD: usize = 16_384;
        let states = if accepting.len() >= PARALLEL_METADATA_THRESHOLD {
            accepting
                .par_iter()
                .zip(possible_future.par_iter())
                .map(|(&accepting, &possible_future)| build_state(accepting, possible_future))
                .collect()
        } else {
            accepting
                .iter()
                .zip(possible_future)
                .map(|(&accepting, &possible_future)| build_state(accepting, possible_future))
                .collect()
        };
        Self {
            states,
            group_id_to_u8set: vec![group_u8set],
            derived_stats: OnceLock::new(),
            min_match_byte_len_cache: OnceLock::new(),
            structural_hash_cache: OnceLock::new(),
        }
    }

    /// Restore the compact one-group residual-oracle representation in one
    /// pass. This is deliberately narrower than the general persisted DFA
    /// format: bounded-code liveness DFAs are deterministic, epsilon-free and
    /// use only the Boolean observations "accepting" and "has future".
    pub(super) fn new_from_compact_single_group(
        state_count: usize,
        transition_offsets: &[u32],
        transition_bytes: &[u8],
        transition_targets: &[u16],
        accepting_words: &[u64],
        future_words: &[u64],
        group_u8set: U8Set,
    ) -> Option<Self> {
        if state_count == 0
            || state_count > u16::MAX as usize
            || transition_offsets.len() != state_count + 1
            || transition_offsets.first().copied() != Some(0)
            || transition_offsets.last().copied()? as usize != transition_bytes.len()
            || transition_bytes.len() != transition_targets.len()
            || accepting_words.len() != state_count.div_ceil(64)
            || future_words.len() != state_count.div_ceil(64)
        {
            return None;
        }
        let mut states = Vec::with_capacity(state_count);
        for state in 0..state_count {
            let start = transition_offsets[state] as usize;
            let end = transition_offsets[state + 1] as usize;
            if start > end || end > transition_bytes.len() {
                return None;
            }
            let mut entries = Vec::with_capacity(end - start);
            let mut previous = None;
            for index in start..end {
                let byte = transition_bytes[index];
                if previous.is_some_and(|previous| previous >= byte) {
                    return None;
                }
                previous = Some(byte);
                let target = transition_targets[index] as u32;
                if target as usize >= state_count {
                    return None;
                }
                entries.push((byte, target));
            }
            let mut finalizers = BitSet::new(1);
            if accepting_words[state / 64] & (1u64 << (state % 64)) != 0 {
                finalizers.set(0);
            }
            let mut possible_future_group_ids = BitSet::new(1);
            if future_words[state / 64] & (1u64 << (state % 64)) != 0 {
                possible_future_group_ids.set(0);
            }
            states.push(DFAState {
                transitions: CharTransitions::from_sorted_entries(entries),
                finalizers,
                possible_future_group_ids,
                epsilon_transitions: Vec::new(),
            });
        }
        Some(Self {
            states,
            group_id_to_u8set: vec![group_u8set],
            derived_stats: OnceLock::new(),
            min_match_byte_len_cache: OnceLock::new(),
            structural_hash_cache: OnceLock::new(),
        })
    }

    #[inline]
    fn invalidate_min_match_byte_len(&mut self) {
        let _ = self.min_match_byte_len_cache.take();
    }

    #[inline]
    fn invalidate_structural_hash(&mut self) {
        let _ = self.structural_hash_cache.take();
    }

    #[inline]
    fn invalidate_structural_caches(&mut self) {
        let _ = self.derived_stats.take();
        self.invalidate_min_match_byte_len();
        self.invalidate_structural_hash();
    }

    #[inline]
    fn structural_hash(&self) -> u64 {
        *self.structural_hash_cache.get_or_init(|| {
            let mut hasher = FxHasher::default();
            self.states.hash(&mut hasher);
            self.group_id_to_u8set.hash(&mut hasher);
            hasher.finish()
        })
    }

    fn compute_derived_stats(&self) -> DfaDerivedStats {
        let mut stats = DfaDerivedStats::default();
        for (state_id, state) in self.states.iter().enumerate() {
            stats.byte_transition_count += state.transitions.len();
            stats.epsilon_transition_count += state.epsilon_transitions.len();
            stats.has_self_loops |= state
                .transitions
                .iter()
                .any(|(_, &target)| target as usize == state_id);
        }
        stats
    }

    #[inline]
    fn derived_stats(&self) -> DfaDerivedStats {
        *self
            .derived_stats
            .get_or_init(|| self.compute_derived_stats())
    }

    pub fn transition_count(&self) -> usize {
        self.derived_stats().byte_transition_count
    }

    pub fn epsilon_transition_count(&self) -> usize {
        self.derived_stats().epsilon_transition_count
    }

    pub(super) fn set_derived_stats(
        &mut self,
        byte_transition_count: usize,
        epsilon_transition_count: usize,
        has_self_loops: bool,
    ) {
        let _ = self.derived_stats.take();
        self.derived_stats
            .set(DfaDerivedStats {
                byte_transition_count,
                epsilon_transition_count,
                has_self_loops,
            })
            .expect("DFA derived stats were unexpectedly initialized");
    }

    pub fn has_self_loops(&self) -> bool {
        self.derived_stats().has_self_loops
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    pub(super) fn add_state(&mut self) -> u32 {
        self.invalidate_structural_caches();
        let id = self.states.len() as u32;
        let groups = self.group_id_to_u8set.len();
        self.states.push(DFAState {
            transitions: CharTransitions::default(),
            finalizers: BitSet::new(groups),
            possible_future_group_ids: BitSet::new(groups),
            epsilon_transitions: Vec::new(),
        });
        id
    }

    pub(super) fn reserve_additional_states(&mut self, additional: usize) {
        self.states.reserve(additional);
    }

    /// Extend the global group metadata without resizing existing state bitsets.
    /// Existing states keep identity group IDs in their current shorter bitsets;
    /// out-of-range membership queries are false. This is exact for immutable
    /// disjoint composition and avoids rewriting every state in a large parent.
    pub(super) fn ensure_group_mapping_capacity(&mut self, num_groups: usize) {
        if self.group_id_to_u8set.len() < num_groups {
            self.invalidate_structural_hash();
            self.group_id_to_u8set.resize(num_groups, U8Set::empty());
        }
    }

    pub(super) fn ensure_group_capacity(&mut self, num_groups: usize) {
        // The mapping may already have been extended by
        // `ensure_group_mapping_capacity` while state metadata stayed narrow.
        // Resizing those bitsets changes structural equality even when the
        // mapping itself does not grow here.
        self.invalidate_structural_hash();
        if self.group_id_to_u8set.len() < num_groups {
            self.group_id_to_u8set.resize(num_groups, U8Set::empty());
        }
        for state in &mut self.states {
            Self::resize_state_group_bits(state, num_groups);
        }
    }

    pub(super) fn add_transition(&mut self, from: u32, byte: u8, to: u32) {
        self.invalidate_structural_caches();
        if let Some(state) = self.states.get_mut(from as usize) {
            state.transitions.insert(byte, to);
        }
    }

    pub(super) fn add_epsilon_transition(&mut self, from: u32, to: u32) {
        self.invalidate_structural_caches();
        if let Some(state) = self.states.get_mut(from as usize) {
            if !state.epsilon_transitions.contains(&to) {
                state.epsilon_transitions.push(to);
            }
        }
    }

    /// Move an independently compiled component into this DFA, rebasing its
    /// state targets and remapping local group IDs to the supplied global IDs.
    /// Transition buffers are retained rather than copied into newly allocated
    /// rows, which matters for very large exact runtime lexer components.
    pub(super) fn append_rebased_component(
        &mut self,
        mut component: DFA,
        global_group_ids: &[usize],
    ) -> u32 {
        self.invalidate_structural_caches();
        assert_eq!(
            component.group_id_to_u8set.len(),
            global_group_ids.len(),
            "one global group ID is required per component group",
        );
        let offset = u32::try_from(self.states.len()).expect("lexer DFA state ID overflow");

        for (local_group, &global_group) in global_group_ids.iter().enumerate() {
            assert!(
                global_group < self.group_id_to_u8set.len(),
                "global lexer group ID is out of range",
            );
            self.group_id_to_u8set[global_group] = component.group_id_to_u8set[local_group];
        }

        let total_groups = self.group_id_to_u8set.len();
        let identity_groups = component.group_id_to_u8set.len() == total_groups
            && global_group_ids
                .iter()
                .copied()
                .eq(0..total_groups);
        let rebase_state = |state: &mut DFAState| {
            for (_, target) in state.transitions.iter_mut() {
                *target = target
                    .checked_add(offset)
                    .expect("lexer DFA transition target overflow");
            }
            for target in &mut state.epsilon_transitions {
                *target = target
                    .checked_add(offset)
                    .expect("lexer DFA epsilon target overflow");
            }

            if !identity_groups {
                let mut finalizers = BitSet::new(total_groups);
                for local_group in state.finalizers.iter() {
                    finalizers.set(global_group_ids[local_group]);
                }
                let mut futures = BitSet::new(total_groups);
                for local_group in state.possible_future_group_ids.iter() {
                    futures.set(global_group_ids[local_group]);
                }
                state.finalizers = finalizers;
                state.possible_future_group_ids = futures;
            }
        };
        if component.states.len() >= 4096
            && rayon::current_num_threads() > 1
            && std::env::var_os("GLRMASK_SERIAL_APPEND_REBASE").is_none()
        {
            component.states.par_iter_mut().for_each(rebase_state);
        } else {
            component.states.iter_mut().for_each(rebase_state);
        }

        self.states.append(&mut component.states);
        offset
    }

    /// Append an independently compiled single-group component while mapping
    /// its one lexical language to several globally distinct terminal groups.
    /// This is exact language sharing: byte transitions remain one DFA, while
    /// accepting/future metadata carries every terminal whose residual language
    /// is identical.
    pub(super) fn append_rebased_shared_single_group_component(
        &mut self,
        mut component: DFA,
        global_group_ids: &[usize],
    ) -> u32 {
        self.invalidate_structural_caches();
        assert_eq!(component.group_id_to_u8set.len(), 1);
        assert!(!global_group_ids.is_empty());
        for &global_group in global_group_ids {
            assert!(global_group < self.group_id_to_u8set.len());
            self.group_id_to_u8set[global_group] = component.group_id_to_u8set[0];
        }
        let offset = u32::try_from(self.states.len()).expect("lexer DFA state ID overflow");
        let total_groups = self.group_id_to_u8set.len();
        let rebase_state = |state: &mut DFAState| {
            for (_, target) in state.transitions.iter_mut() {
                *target = target
                    .checked_add(offset)
                    .expect("lexer DFA transition target overflow");
            }
            for target in &mut state.epsilon_transitions {
                *target = target
                    .checked_add(offset)
                    .expect("lexer DFA epsilon target overflow");
            }
            let local_final = state.finalizers.contains(0);
            let local_future = state.possible_future_group_ids.contains(0);
            let mut finalizers = BitSet::new(total_groups);
            let mut futures = BitSet::new(total_groups);
            if local_final {
                for &global_group in global_group_ids {
                    finalizers.set(global_group);
                }
            }
            if local_future {
                for &global_group in global_group_ids {
                    futures.set(global_group);
                }
            }
            state.finalizers = finalizers;
            state.possible_future_group_ids = futures;
        };
        if component.states.len() >= 4096
            && rayon::current_num_threads() > 1
            && std::env::var_os("GLRMASK_SERIAL_APPEND_REBASE").is_none()
        {
            component.states.par_iter_mut().for_each(rebase_state);
        } else {
            component.states.iter_mut().for_each(rebase_state);
        }
        self.states.append(&mut component.states);
        offset
    }

    /// Clone a borrowed independently compiled component directly into this
    /// DFA while rebasing it. This avoids the owned helper's clone-then-rewrite
    /// double pass, which is material for composition of large cached lexers.
    pub(super) fn append_rebased_component_ref(
        &mut self,
        component: &DFA,
        global_group_ids: &[usize],
    ) -> u32 {
        self.invalidate_structural_caches();
        assert_eq!(
            component.group_id_to_u8set.len(),
            global_group_ids.len(),
            "one global group ID is required per component group",
        );
        let offset = u32::try_from(self.states.len()).expect("lexer DFA state ID overflow");

        for (local_group, &global_group) in global_group_ids.iter().enumerate() {
            assert!(
                global_group < self.group_id_to_u8set.len(),
                "global lexer group ID is out of range",
            );
            self.group_id_to_u8set[global_group] = component.group_id_to_u8set[local_group];
        }

        let total_groups = self.group_id_to_u8set.len();
        let identity_groups = component.group_id_to_u8set.len() == total_groups
            && global_group_ids
                .iter()
                .copied()
                .eq(0..total_groups);
        let clone_rebased = |source: &DFAState| {
            let mut state = source.clone();
            for (_, target) in state.transitions.iter_mut() {
                *target = target
                    .checked_add(offset)
                    .expect("lexer DFA transition target overflow");
            }
            for target in &mut state.epsilon_transitions {
                *target = target
                    .checked_add(offset)
                    .expect("lexer DFA epsilon target overflow");
            }
            if !identity_groups {
                let mut finalizers = BitSet::new(total_groups);
                for local_group in source.finalizers.iter() {
                    finalizers.set(global_group_ids[local_group]);
                }
                let mut futures = BitSet::new(total_groups);
                for local_group in source.possible_future_group_ids.iter() {
                    futures.set(global_group_ids[local_group]);
                }
                state.finalizers = finalizers;
                state.possible_future_group_ids = futures;
            }
            state
        };
        if component.states.len() < 4096 {
            // The sequential path does not need an intermediate state vector:
            // the iterator has an exact size, so `Vec::extend` reserves once
            // and writes the cloned/rebased states directly into the parent.
            self.states.reserve(component.states.len());
            self.states.extend(component.states.iter().map(clone_rebased));
            return offset;
        }
        let appended = if component.states.len() >= 4096
            && rayon::current_num_threads() > 1
            && std::env::var_os("GLRMASK_SERIAL_APPEND_REBASE").is_none()
        {
            component.states.par_iter().map(clone_rebased).collect::<Vec<_>>()
        } else {
            component.states.iter().map(clone_rebased).collect::<Vec<_>>()
        };
        self.states.extend(appended);
        offset
    }

    /// Append several independent single-group component templates in one
    /// deterministic batch.  Each logical component keeps its own terminal
    /// identity/state range; only the clone+rebase work is parallelized across
    /// components.  This is useful when many residual terminals share a finite
    /// DFA template but cannot share logical lexer states because their labels
    /// must remain correlated with parser branches.
    pub(super) fn append_rebased_single_group_component_refs_batch(
        &mut self,
        components: &[(&DFA, usize)],
    ) -> Option<Vec<u32>> {
        if components.is_empty() {
            return Some(Vec::new());
        }
        self.invalidate_structural_caches();
        let total_groups = self.group_id_to_u8set.len();
        let mut next_offset = u32::try_from(self.states.len()).ok()?;
        let mut offsets = Vec::with_capacity(components.len());
        for &(component, global_group) in components {
            if component.group_id_to_u8set.len() != 1 || global_group >= total_groups {
                return None;
            }
            offsets.push(next_offset);
            next_offset = next_offset.checked_add(u32::try_from(component.states.len()).ok()?)?;
            self.group_id_to_u8set[global_group] = component.group_id_to_u8set[0];
        }

        let appended = components
            .par_iter()
            .zip(offsets.par_iter().copied())
            .map(|(&(component, global_group), offset)| {
                component
                    .states
                    .iter()
                    .map(|source| {
                        let mut state = source.clone();
                        for (_, target) in state.transitions.iter_mut() {
                            *target = target
                                .checked_add(offset)
                                .expect("lexer DFA transition target overflow");
                        }
                        for target in &mut state.epsilon_transitions {
                            *target = target
                                .checked_add(offset)
                                .expect("lexer DFA epsilon target overflow");
                        }
                        let mut finalizers = BitSet::new(total_groups);
                        if source.finalizers.contains(0) {
                            finalizers.set(global_group);
                        }
                        let mut futures = BitSet::new(total_groups);
                        if source.possible_future_group_ids.contains(0) {
                            futures.set(global_group);
                        }
                        state.finalizers = finalizers;
                        state.possible_future_group_ids = futures;
                        state
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for mut states in appended {
            self.states.append(&mut states);
        }
        Some(offsets)
    }

    pub fn has_epsilon_transitions(&self) -> bool {
        self.epsilon_transition_count() != 0
    }

    /// Minimum number of consumed bytes needed to reach any accepting state.
    /// Epsilon transitions have zero cost and byte transitions have unit cost.
    pub fn min_match_byte_len(&self) -> Option<usize> {
        *self
            .min_match_byte_len_cache
            .get_or_init(|| self.compute_min_match_byte_len())
    }

    fn compute_min_match_byte_len(&self) -> Option<usize> {
        let mut distance = vec![usize::MAX; self.states.len()];
        let mut queue = std::collections::VecDeque::new();
        if self.states.is_empty() {
            return None;
        }
        distance[0] = 0;
        queue.push_front(0u32);

        while let Some(state_id) = queue.pop_front() {
            let state_index = state_id as usize;
            let current_distance = distance[state_index];
            let state = &self.states[state_index];
            if !state.finalizers.is_empty() {
                return Some(current_distance);
            }

            for &target in &state.epsilon_transitions {
                let target_index = target as usize;
                if current_distance < distance[target_index] {
                    distance[target_index] = current_distance;
                    queue.push_front(target);
                }
            }
            let next_distance = current_distance.saturating_add(1);
            for (_, &target) in state.transitions.iter() {
                let target_index = target as usize;
                if next_distance < distance[target_index] {
                    distance[target_index] = next_distance;
                    queue.push_back(target);
                }
            }
        }
        None
    }

    pub(super) fn epsilon_closure(&self, roots: &[u32]) -> SmallVec<[u32; 1]> {
        if roots.iter().all(|&root| {
            self.states
                .get(root as usize)
                .is_some_and(|state| state.epsilon_transitions.is_empty())
        }) {
            let mut closure = SmallVec::<[u32; 1]>::from_slice(roots);
            closure.sort_unstable();
            closure.dedup();
            return closure;
        }
        let mut closure = SmallVec::<[u32; 1]>::new();
        // Runtime tokenizers produced by exact bounded synthesis can have
        // hundreds of thousands of raw states while an epsilon closure usually
        // contains only a handful. A dense `seen` vector makes every such scan
        // O(total states) in allocation and zeroing. Track only states actually
        // reached so closure cost is proportional to the live epsilon graph.
        let mut seen = FxHashSet::<u32>::default();
        seen.reserve(roots.len().max(4));
        let mut stack = Vec::with_capacity(roots.len());
        for &root in roots {
            if (root as usize) < self.states.len() && seen.insert(root) {
                stack.push(root);
            }
        }
        while let Some(state) = stack.pop() {
            closure.push(state);
            for &target in &self.states[state as usize].epsilon_transitions {
                if seen.insert(target) {
                    stack.push(target);
                }
            }
        }
        closure.sort_unstable();
        closure
    }

    /// Compute every singleton epsilon closure in one pass.
    ///
    /// Lexer partition unions and bounded adaptive frontiers produce an
    /// acyclic epsilon graph.  For that common case, reverse-topological
    /// dynamic programming reuses each target closure instead of allocating a
    /// fresh `seen` vector and walking the same tails once per source state.
    /// Persisted/debug automata may contain epsilon cycles; retain the exact
    /// scalar closure as a defensive fallback when Kahn's order detects one.
    pub(super) fn all_singleton_epsilon_closures(&self) -> SingletonEpsilonClosures {
        let state_count = self.states.len();

        // Disjoint tokenizer composition creates one epsilon dispatcher whose
        // targets have no further epsilon edges.  In that common million-state
        // shape, every non-dispatch closure is the singleton `[state]`.  The
        // generic topological builder below allocates those rows serially; do
        // the exact independent construction in parallel instead.
        let mut epsilon_source = None;
        let mut shallow_dispatch = true;
        for (state, row) in self.states.iter().enumerate() {
            if row.epsilon_transitions.is_empty() {
                continue;
            }
            if epsilon_source.replace(state).is_some() {
                shallow_dispatch = false;
                break;
            }
            if row.epsilon_transitions.iter().any(|&target| {
                self.states
                    .get(target as usize)
                    .is_none_or(|target| !target.epsilon_transitions.is_empty())
            }) {
                shallow_dispatch = false;
                break;
            }
        }
        if shallow_dispatch
            && let Some(dispatch) = epsilon_source
        {
            let mut dispatch_closure = Vec::with_capacity(
                self.states[dispatch].epsilon_transitions.len() + 1,
            );
            dispatch_closure.push(dispatch as u32);
            dispatch_closure.extend(
                self.states[dispatch]
                    .epsilon_transitions
                    .iter()
                    .copied(),
            );
            dispatch_closure.sort_unstable();
            dispatch_closure.dedup();
            let identities = (0..state_count as u32)
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let mut overrides = (0..state_count)
                .map(|_| None)
                .collect::<Vec<Option<Box<[u32]>>>>();
            overrides[dispatch] = Some(dispatch_closure.into_boxed_slice());
            return SingletonEpsilonClosures::IdentityWithOverrides {
                identities,
                overrides: overrides.into_boxed_slice(),
            };
        }

        let mut indegree = vec![0usize; state_count];
        for state in &self.states {
            for &target in &state.epsilon_transitions {
                indegree[target as usize] += 1;
            }
        }
        let mut queue = std::collections::VecDeque::<u32>::new();
        for (state, &degree) in indegree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(state as u32);
            }
        }
        let mut topo = Vec::with_capacity(state_count);
        while let Some(state) = queue.pop_front() {
            topo.push(state);
            for &target in &self.states[state as usize].epsilon_transitions {
                let degree = &mut indegree[target as usize];
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(target);
                }
            }
        }
        if topo.len() != state_count {
            return SingletonEpsilonClosures::Dense(
                (0..state_count)
                    .map(|state| {
                        self.epsilon_closure(&[state as u32])
                            .into_vec()
                            .into_boxed_slice()
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );
        }

        let identities = (0..state_count as u32)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut overrides = (0..state_count)
            .map(|_| None)
            .collect::<Vec<Option<Box<[u32]>>>>();
        let mut marks = vec![0u32; state_count];
        let mut generation = 0u32;
        for &state in topo.iter().rev() {
            let epsilon_targets = &self.states[state as usize].epsilon_transitions;
            if epsilon_targets.is_empty() {
                continue;
            }
            generation = generation.wrapping_add(1);
            if generation == 0 {
                marks.fill(0);
                generation = 1;
            }
            let mut closure = Vec::<u32>::new();
            marks[state as usize] = generation;
            closure.push(state);
            for &target in epsilon_targets {
                let target_closure = overrides[target as usize]
                    .as_deref()
                    .unwrap_or(&identities[target as usize..target as usize + 1]);
                for &reachable in target_closure {
                    let slot = &mut marks[reachable as usize];
                    if *slot != generation {
                        *slot = generation;
                        closure.push(reachable);
                    }
                }
            }
            closure.sort_unstable();
            overrides[state as usize] = Some(closure.into_boxed_slice());
        }
        SingletonEpsilonClosures::IdentityWithOverrides {
            identities,
            overrides: overrides.into_boxed_slice(),
        }
    }

    pub(super) fn step_all(&self, states: &[u32], byte: u8) -> SmallVec<[u32; 1]> {
        if states.len() == 1 {
            let state = states[0];
            if self.states[state as usize].epsilon_transitions.is_empty() {
                let Some(target) = self.step(state, byte) else {
                    return SmallVec::new();
                };
                if self.states[target as usize].epsilon_transitions.is_empty() {
                    return SmallVec::from_buf([target]);
                }
            }
        }
        let closure = self.epsilon_closure(states);
        let mut targets = SmallVec::<[u32; 1]>::new();
        for state in closure {
            if let Some(target) = self.step(state, byte) {
                targets.push(target);
            }
        }
        if targets.is_empty() {
            return targets;
        }
        targets.sort_unstable();
        targets.dedup();
        self.epsilon_closure(&targets)
    }

    pub(super) fn set_transitions_from_sorted_entries(
        &mut self,
        state: u32,
        entries: Vec<(u8, u32)>,
    ) {
        self.invalidate_structural_caches();
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.transitions = CharTransitions::from_sorted_entries(entries);
        }
    }

    pub(super) fn clear_finalizers_for_state(&mut self, state: u32) -> BitSet {
        self.invalidate_min_match_byte_len();
        self.invalidate_structural_hash();
        let num_groups = self.group_id_to_u8set.len();
        if let Some(entry) = self.state_mut(state) {
            std::mem::replace(&mut entry.finalizers, BitSet::new(num_groups))
        } else {
            BitSet::empty(0)
        }
    }

    pub(super) fn overwrite_state_metadata(
        &mut self,
        state: u32,
        finalizers: BitSet,
        possible_future_group_ids: BitSet,
    ) {
        self.invalidate_min_match_byte_len();
        self.invalidate_structural_hash();
        if let Some(entry) = self.state_mut(state) {
            entry.finalizers = finalizers;
            entry.possible_future_group_ids = possible_future_group_ids;
        }
    }

    pub(super) fn canonicalize_group_aliases(&mut self, canonical: usize, aliases: &[usize]) {
        if aliases.is_empty() {
            return;
        }
        let alias_set = aliases.iter().copied().collect::<std::collections::BTreeSet<_>>();
        for state in &mut self.states {
            let finalizes_alias = state.finalizers.iter().any(|group| alias_set.contains(&group));
            let futures_alias = state
                .possible_future_group_ids
                .iter()
                .any(|group| alias_set.contains(&group));
            for &alias in aliases {
                state.finalizers.clear(alias);
                state.possible_future_group_ids.clear(alias);
            }
            if finalizes_alias {
                state.finalizers.set(canonical);
            }
            if futures_alias {
                state.possible_future_group_ids.set(canonical);
            }
        }
        let mut bytes = self.group_id_to_u8set(canonical as u32).clone();
        for &alias in aliases {
            bytes |= *self.group_id_to_u8set(alias as u32);
        }
        self.set_group_u8set(canonical as u32, bytes);
    }

    pub(super) fn set_group_u8set(&mut self, group_id: GroupId, set: U8Set) {
        self.invalidate_structural_hash();
        if let Some(entry) = self.group_id_to_u8set.get_mut(group_id as usize) {
            *entry = set;
        }
    }

    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.states
            .get(state as usize)
            .and_then(|state| state.transitions.get(byte).copied())
    }

    pub fn transitions(&self, state: u32) -> impl Iterator<Item = (u8, u32)> + '_ {
        self.states[state as usize]
            .transitions
            .iter()
            .map(|(byte, &target)| (byte, target))
    }

    pub(super) fn get_u8set(&self, state: u32) -> U8Set {
        let mut out = U8Set::empty();
        if let Some(state) = self.states.get(state as usize) {
            for (byte, _) in state.transitions.iter() {
                out.insert(byte);
            }
        }
        out
    }

    pub(super) fn get_transition(&self, state: u32, byte: u8) -> u32 {
        self.step(state, byte).unwrap_or(DEAD)
    }

    pub(super) fn group_id_to_u8set(&self, group_id: GroupId) -> &U8Set {
        &self.group_id_to_u8set[group_id as usize]
    }

    pub fn finalizers(&self, state: u32) -> &BitSet {
        &self.states[state as usize].finalizers
    }

    pub fn possible_future_group_ids(&self, state: u32) -> &BitSet {
        &self.states[state as usize].possible_future_group_ids
    }

    pub(super) fn states(&self) -> &[DFAState] {
        &self.states
    }

    pub(super) fn states_mut(&mut self) -> &mut Vec<DFAState> {
        self.invalidate_structural_caches();
        &mut self.states
    }

    pub(super) fn num_groups(&self) -> usize {
        self.group_id_to_u8set.len()
    }

    pub(super) fn set_possible_future_group_ids(&mut self, state: u32, ids: BitSet) {
        self.invalidate_structural_hash();
        if let Some(entry) = self.state_mut(state) {
            entry.possible_future_group_ids = ids;
        }
    }
    /// Mask all states' possible_future_group_ids with the given bitset.
    pub(super) fn mask_possible_futures(&mut self, mask: &BitSet) {
        self.invalidate_structural_hash();
        for state in &mut self.states {
            state.possible_future_group_ids.intersect_with(mask);
        }
    }
    /// Create a clone of an existing state (transitions, finalizers,
    /// possible_future_group_ids) and return the new state's id.
    pub(super) fn clone_state(&mut self, source: u32) -> u32 {
        self.invalidate_structural_caches();
        let cloned = self.states[source as usize].clone();
        let id = self.states.len() as u32;
        self.states.push(cloned);
        id
    }

    /// Rewrite every transition that targets `old_target` so it targets
    /// `new_target` instead.
    /// Redirect every incoming edge to `old_target`, returning whether any
    /// edge changed. The caller may use this to speculatively clone a state
    /// and discard the clone when no incoming edge exists.
    pub(super) fn redirect_transitions(&mut self, old_target: u32, new_target: u32) -> bool {
        self.invalidate_structural_caches();
        let mut changed = false;
        for state in &mut self.states {
            for (_, target) in state.transitions.iter_mut() {
                if *target == old_target {
                    *target = new_target;
                    changed = true;
                }
            }
            for target in &mut state.epsilon_transitions {
                if *target == old_target {
                    *target = new_target;
                    changed = true;
                }
            }
        }
        changed
    }

    /// Remove the final state when it is the expected freshly-created ID.
    pub(super) fn discard_last_state(&mut self, expected: u32) {
        self.invalidate_structural_caches();
        debug_assert_eq!(self.states.len(), expected as usize + 1);
        self.states.pop();
    }

    pub(super) fn apply_group_exclusions(
        &mut self,
        excludes: &BTreeMap<GroupId, BTreeSet<GroupId>>,
    ) -> bool {
        let mut changed = false;
        for state in &mut self.states {
            if state.finalizers.count_ones() < 2 {
                continue;
            }

            for group_index in excluded_group_indices(&state.finalizers, excludes) {
                if state.finalizers.contains(group_index) {
                    state.finalizers.clear(group_index);
                    changed = true;
                }
            }
        }
        if changed {
            self.invalidate_min_match_byte_len();
            self.invalidate_structural_hash();
        }
        changed
    }

    pub(super) fn apply_group_intersections(
        &mut self,
        intersections: &BTreeMap<GroupId, BTreeSet<GroupId>>,
    ) -> bool {
        let mut changed = false;
        for state in &mut self.states {
            for group_index in intersection_missing_group_indices(&state.finalizers, intersections) {
                if state.finalizers.contains(group_index) {
                    state.finalizers.clear(group_index);
                    changed = true;
                }
            }
        }
        if changed {
            self.invalidate_min_match_byte_len();
            self.invalidate_structural_hash();
        }
        changed
    }

    pub(super) fn project_groups(&self, num_groups: usize) -> DFA {
        let mut projected = DFA::new(self.num_states());
        projected.ensure_group_capacity(num_groups);

        for (state_index, state) in self.states.iter().enumerate() {
            let transitions = state
                .transitions
                .iter()
                .map(|(byte, &target)| (byte, target))
                .collect();
            projected.set_transitions_from_sorted_entries(state_index as u32, transitions);
            projected.states_mut()[state_index].epsilon_transitions =
                state.epsilon_transitions.clone();

            let finalizers = project_bitset(&state.finalizers, num_groups);
            let future = project_bitset(&state.possible_future_group_ids, num_groups);

            projected.overwrite_state_metadata(state_index as u32, finalizers, future);
        }

        for group_id in 0..num_groups {
            projected.set_group_u8set(group_id as u32, self.group_id_to_u8set[group_id]);
        }

        projected
    }

    /// Project to the leading `num_groups` in place. Transitions and state IDs
    /// are unchanged; only compiler metadata for hidden groups is discarded.
    pub(super) fn project_groups_in_place(&mut self, num_groups: usize) {
        assert!(num_groups <= self.num_groups());
        self.invalidate_structural_caches();
        for state in &mut self.states {
            state.finalizers.truncate(num_groups);
            state.possible_future_group_ids.truncate(num_groups);
        }
        self.group_id_to_u8set.truncate(num_groups);
    }

    fn state_mut(&mut self, state: u32) -> Option<&mut DFAState> {
        self.states.get_mut(state as usize)
    }

    fn resize_state_group_bits(state: &mut DFAState, num_groups: usize) {
        if state.finalizers.len() < num_groups {
            state.finalizers = resized_bitset(&state.finalizers, num_groups);
            state.possible_future_group_ids =
                resized_bitset(&state.possible_future_group_ids, num_groups);
        }
    }
}

#[cfg(test)]
mod tests {
    use serde::Serialize;

    use super::*;

    #[derive(Serialize)]
    struct LegacyDfaState {
        transitions: CharTransitions<u32>,
        finalizers: BitSet,
        possible_future_group_ids: BitSet,
    }

    #[derive(Serialize)]
    struct LegacyDfa {
        states: Vec<LegacyDfaState>,
        group_id_to_u8set: Vec<U8Set>,
    }

    fn owned_wire_bytes(dfa: &DFA) -> Vec<u8> {
        let mut states = dfa.states.iter().map(DfaStateWire::from).collect::<Vec<_>>();
        let real_state_count = states.len();
        let mut metadata_state_count = 0usize;
        for (source, state) in dfa.states.iter().enumerate() {
            for targets in state
                .epsilon_transitions
                .chunks(EPSILON_TARGETS_PER_WIRE_STATE)
            {
                if targets.is_empty() {
                    continue;
                }
                let mut entries = Vec::with_capacity(targets.len() + 3);
                entries.push((0, EPSILON_WIRE_MARKER));
                entries.push((1, EPSILON_WIRE_EDGE));
                entries.push((2, source as u32));
                entries.extend(
                    targets
                        .iter()
                        .enumerate()
                        .map(|(index, &target)| ((index + 3) as u8, target)),
                );
                states.push(epsilon_wire_state(entries));
                metadata_state_count += 1;
            }
        }
        if metadata_state_count != 0 {
            states.push(epsilon_wire_state(vec![
                (0, EPSILON_WIRE_MARKER),
                (1, EPSILON_WIRE_TRAILER),
                (2, real_state_count as u32),
                (3, metadata_state_count as u32),
            ]));
        }
        bincode::serialize(&DfaWire {
            states,
            group_id_to_u8set: dfa.group_id_to_u8set.clone(),
        })
        .unwrap()
    }

    #[test]
    fn structural_property_caches_invalidate_on_mutation() {
        let mut dfa = DFA::new(2);
        dfa.ensure_group_capacity(1);
        dfa.add_transition(0, b'a', 1);
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        dfa.overwrite_state_metadata(1, finalizers, BitSet::new(1));

        assert_eq!(dfa.transition_count(), 1);
        assert!(!dfa.has_epsilon_transitions());
        assert!(!dfa.has_self_loops());
        assert_eq!(dfa.min_match_byte_len(), Some(1));
        assert!(dfa.derived_stats.get().is_some());
        assert!(dfa.min_match_byte_len_cache.get().is_some());

        let old_finalizers = dfa.clear_finalizers_for_state(1);
        assert!(dfa.min_match_byte_len_cache.get().is_none());
        assert_eq!(dfa.min_match_byte_len(), None);
        dfa.overwrite_state_metadata(1, old_finalizers, BitSet::new(1));
        assert_eq!(dfa.min_match_byte_len(), Some(1));

        dfa.add_transition(1, b'b', 1);
        assert!(dfa.derived_stats.get().is_none());
        assert!(dfa.min_match_byte_len_cache.get().is_none());
        assert_eq!(dfa.transition_count(), 2);
        assert!(dfa.has_self_loops());
        assert_eq!(dfa.min_match_byte_len(), Some(1));

        dfa.add_epsilon_transition(0, 1);
        assert!(dfa.has_epsilon_transitions());
        assert_eq!(dfa.epsilon_transition_count(), 1);
    }

    #[test]
    fn in_place_group_projection_matches_owned_projection() {
        let mut dfa = DFA::new(3);
        dfa.ensure_group_capacity(3);
        dfa.add_transition(0, b'a', 1);
        dfa.add_transition(1, b'b', 2);

        let mut state1_finalizers = BitSet::new(3);
        state1_finalizers.set(2);
        let mut state1_futures = BitSet::new(3);
        state1_futures.set(0);
        state1_futures.set(2);
        dfa.overwrite_state_metadata(1, state1_finalizers, state1_futures);

        let mut state2_finalizers = BitSet::new(3);
        state2_finalizers.set(0);
        state2_finalizers.set(1);
        let mut state2_futures = BitSet::new(3);
        state2_futures.set(1);
        dfa.overwrite_state_metadata(2, state2_finalizers, state2_futures);

        let expected = dfa.project_groups(2);
        let mut actual = dfa;
        actual.project_groups_in_place(2);
        assert_eq!(actual, expected);
    }

    #[test]
    fn projection_before_future_recompute_matches_projection_after_recompute() {
        let mut dfa = DFA::new(4);
        dfa.ensure_group_capacity(3);
        dfa.add_transition(0, b'a', 1);
        dfa.add_transition(1, b'b', 2);
        dfa.add_transition(1, b'c', 3);

        let mut state2_finalizers = BitSet::new(3);
        state2_finalizers.set(0);
        state2_finalizers.set(2);
        dfa.overwrite_state_metadata(2, state2_finalizers, BitSet::new(3));

        let mut state3_finalizers = BitSet::new(3);
        state3_finalizers.set(1);
        dfa.overwrite_state_metadata(3, state3_finalizers, BitSet::new(3));

        let mut excludes = BTreeMap::<GroupId, BTreeSet<GroupId>>::new();
        excludes.entry(0).or_default().insert(2);

        let mut historical = dfa.clone();
        assert!(historical.apply_group_exclusions(&excludes));
        historical.recompute_possible_futures();
        let historical = historical.project_groups(2);

        let mut projected_first = dfa;
        assert!(projected_first.apply_group_exclusions(&excludes));
        projected_first.project_groups_in_place(2);
        projected_first.recompute_possible_futures();

        assert_eq!(projected_first, historical);
    }

    #[test]
    fn derived_caches_do_not_affect_equality_or_hashing() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut left = DFA::new(2);
        left.add_transition(0, b'a', 1);
        let right = left.clone();
        let _ = left.transition_count();
        let _ = left.min_match_byte_len();
        assert_eq!(left, right);

        let hash = |dfa: &DFA| {
            let mut hasher = DefaultHasher::new();
            dfa.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(hash(&left), hash(&right));
    }

    #[test]
    fn structural_hash_cache_invalidates_on_metadata_mutation() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash = |dfa: &DFA| {
            let mut hasher = DefaultHasher::new();
            dfa.hash(&mut hasher);
            hasher.finish()
        };
        let cache_hash = |dfa: &DFA| {
            let value = hash(dfa);
            assert!(dfa.structural_hash_cache.get().is_some());
            value
        };

        let mut dfa = DFA::new(2);
        dfa.ensure_group_capacity(1);
        dfa.add_transition(0, b'a', 1);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(1, accepting, BitSet::new(1));
        let original_hash = cache_hash(&dfa);

        dfa.ensure_group_mapping_capacity(2);
        assert!(dfa.structural_hash_cache.get().is_none());
        let mapping_hash = cache_hash(&dfa);
        assert_ne!(mapping_hash, original_hash);

        // The mapping is already wide here; this call changes only the state
        // metadata bitset widths and must still invalidate the hash.
        dfa.ensure_group_capacity(2);
        assert!(dfa.structural_hash_cache.get().is_none());
        let wide_hash = cache_hash(&dfa);
        assert_ne!(wide_hash, mapping_hash);

        let mut finalizers = BitSet::new(2);
        finalizers.set(0);
        finalizers.set(1);
        let mut futures = BitSet::new(2);
        futures.set(1);
        dfa.overwrite_state_metadata(1, finalizers, futures);
        assert!(dfa.structural_hash_cache.get().is_none());
        let metadata_hash = cache_hash(&dfa);
        assert_ne!(metadata_hash, wide_hash);

        dfa.set_group_u8set(1, U8Set::single(b'z'));
        assert!(dfa.structural_hash_cache.get().is_none());
        let group_hash = cache_hash(&dfa);
        assert_ne!(group_hash, metadata_hash);

        let mut new_futures = BitSet::new(2);
        new_futures.set(0);
        dfa.set_possible_future_group_ids(0, new_futures);
        assert!(dfa.structural_hash_cache.get().is_none());
        let futures_hash = cache_hash(&dfa);
        assert_ne!(futures_hash, group_hash);

        let mut mask = BitSet::new(2);
        mask.set(0);
        dfa.mask_possible_futures(&mask);
        assert!(dfa.structural_hash_cache.get().is_none());
        let _ = cache_hash(&dfa);

        let exclusions = BTreeMap::from([(0u32, BTreeSet::from([1u32]))]);
        assert!(dfa.apply_group_exclusions(&exclusions));
        assert!(dfa.structural_hash_cache.get().is_none());
        let _ = cache_hash(&dfa);

        let mut only_zero = BitSet::new(2);
        only_zero.set(0);
        dfa.overwrite_state_metadata(1, only_zero, BitSet::new(2));
        let _ = cache_hash(&dfa);
        let intersections = BTreeMap::from([(0u32, BTreeSet::from([1u32]))]);
        assert!(dfa.apply_group_intersections(&intersections));
        assert!(dfa.structural_hash_cache.get().is_none());
    }

    #[test]
    fn legacy_dfa_layout_decodes_with_empty_epsilon_edges() {
        let mut transitions = CharTransitions::default();
        transitions.insert(b'a', 1);
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        let mut future = BitSet::new(1);
        future.set(0);
        let legacy = LegacyDfa {
            states: vec![LegacyDfaState {
                transitions,
                finalizers,
                possible_future_group_ids: future,
            }],
            group_id_to_u8set: vec![U8Set::single(b'a')],
        };

        let bytes = bincode::serialize(&legacy).unwrap();
        let decoded: DFA = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.states.len(), 1);
        assert_eq!(decoded.states[0].transitions.get(b'a'), Some(&1));
        assert!(decoded.states[0].finalizers.contains(0));
        assert!(decoded.states[0].possible_future_group_ids.contains(0));
        assert!(decoded.states[0].epsilon_transitions.is_empty());
        assert_eq!(bincode::serialize(&decoded).unwrap(), bytes);

        // Epsilon-bearing DFAs use the compatible synthetic-state extension and
        // reconstruct the exact in-memory automaton.
        let mut current = decoded.clone();
        current.add_epsilon_transition(0, 0);
        let current_bytes = bincode::serialize(&current).unwrap();
        let current_decoded: DFA = bincode::deserialize(&current_bytes).unwrap();
        assert_eq!(current_decoded, current);
    }

    #[test]
    fn epsilon_closure_handles_fanout_chains_and_cycles() {
        let mut automaton = DFA::new(5);
        automaton.ensure_group_capacity(1);
        automaton.add_epsilon_transition(0, 1);
        automaton.add_epsilon_transition(0, 2);
        automaton.add_epsilon_transition(1, 3);
        automaton.add_epsilon_transition(2, 3);
        automaton.add_epsilon_transition(3, 1);
        automaton.add_transition(3, b'x', 4);

        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        automaton.overwrite_state_metadata(4, finalizers, BitSet::new(1));
        automaton.recompute_possible_futures();

        assert_eq!(automaton.epsilon_closure(&[0]).as_slice(), &[0, 1, 2, 3]);
        assert_eq!(automaton.step_all(&[0], b'x').as_slice(), &[4]);
        assert!(automaton.possible_future_group_ids(0).contains(0));
    }

    #[test]
    fn epsilon_wire_roundtrip_handles_more_than_one_metadata_chunk() {
        let mut automaton = DFA::new(302);
        for target in 1..=300 {
            automaton.add_epsilon_transition(0, target);
        }
        automaton.add_epsilon_transition(301, 0);

        let bytes = bincode::serialize(&automaton).unwrap();
        assert_eq!(bytes, owned_wire_bytes(&automaton));
        let decoded: DFA = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded, automaton);
        assert_eq!(decoded.states[0].epsilon_transitions.len(), 300);
    }

    #[test]
    fn epsilon_possible_futures_match_bruteforce_reachability() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *seed
        }

        let mut seed = 0x8f31_6a2d_901c_447bu64;
        for case in 0..96 {
            let states = 1 + (next(&mut seed) as usize % 8);
            let groups = 3usize;
            let mut dfa = DFA::new(states);
            dfa.ensure_group_capacity(groups);

            for state in 0..states {
                let mut finalizers = BitSet::new(groups);
                for group in 0..groups {
                    if next(&mut seed).is_multiple_of(4) {
                        finalizers.set(group);
                    }
                }
                dfa.overwrite_state_metadata(state as u32, finalizers, BitSet::new(groups));

                for byte in 0..3u8 {
                    if next(&mut seed).is_multiple_of(3) {
                        let target = (next(&mut seed) as usize % states) as u32;
                        dfa.add_transition(state as u32, byte, target);
                    }
                }
                for target in 0..states {
                    if next(&mut seed).is_multiple_of(9) {
                        dfa.add_epsilon_transition(state as u32, target as u32);
                    }
                }
            }

            dfa.recompute_possible_futures();

            for source in 0..states as u32 {
                let mut after_byte = Vec::new();
                for closure_state in dfa.epsilon_closure(&[source]) {
                    for (_, &target) in dfa.states[closure_state as usize].transitions.iter() {
                        after_byte.extend(dfa.epsilon_closure(&[target]));
                    }
                }
                after_byte.sort_unstable();
                after_byte.dedup();

                let mut seen = vec![false; states];
                let mut stack = after_byte;
                let mut expected = BitSet::new(groups);
                while let Some(state) = stack.pop() {
                    let state_index = state as usize;
                    if seen[state_index] {
                        continue;
                    }
                    seen[state_index] = true;
                    expected.union_with(&dfa.states[state_index].finalizers);
                    stack.extend(dfa.states[state_index].epsilon_transitions.iter().copied());
                    stack.extend(
                        dfa.states[state_index]
                            .transitions
                            .values()
                            .copied(),
                    );
                }

                assert_eq!(
                    dfa.possible_future_group_ids(source),
                    &expected,
                    "future mismatch in random case {case}, source {source}",
                );
            }
        }
    }
}
