//! Runtime-facing tokenizer API built on top of the lexer DFA.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, OnceLock};

use rustc_hash::{FxHashMap, FxHashSet};
use rayon::prelude::*;
use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};
use smallvec::SmallVec;

use super::dfa::DFA;
use super::runtime_unit_repeat::{
    VirtualZeroMinUnitRepeatMaskProjection, VirtualZeroMinUnitRepeatRuntime,
};
use super::runtime_repeat_product::{
    VirtualBinaryRepeatIntersectionDescriptor, VirtualBinaryRepeatIntersectionMaskProjection,
    VirtualBinaryRepeatIntersectionRuntime, VirtualRuntimeStateOwners, VirtualStateAllocator,
};
pub use super::runtime_residual::{
    VirtualResidualDirectCoordinate, VirtualResidualMaskProjection,
    VirtualResidualMaskProjectionArtifact, VirtualResidualMasterSliceArtifact,
};
#[doc(hidden)]
pub type VirtualResidualMaskProjectionArtifactRef<'a> =
    super::runtime_residual::VirtualResidualMaskProjectionArtifactRef<'a>;
use super::runtime_residual::{
    BoundedCodeIntersectionOracle, VirtualResidualRuntime, build_bounded_code_liveness_oracle,
};
pub use super::dfa::SingletonEpsilonClosures;
use crate::automata::regex::Expr;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;

thread_local! {
    /// Static constraint artifacts before v14 use the historical dense DFA
    /// serializer. v14+ may opt into the exact sparse/CSR tokenizer wire form
    /// while still reconstructing the same runtime Tokenizer type.
    static COMPACT_ARTIFACT_SERDE: Cell<bool> = const { Cell::new(false) };
    /// Current sectioned Constraint artifacts can carry the tokenizer in its
    /// own independently decodable section. In that mode the Constraint core
    /// contains only a one-byte placeholder.
    static EXTERNAL_ARTIFACT_SERDE: Cell<bool> = const { Cell::new(false) };
}

pub fn set_compact_artifact_serde(enabled: bool) -> bool {
    COMPACT_ARTIFACT_SERDE.with(|mode| mode.replace(enabled))
}

pub fn set_external_artifact_serde(enabled: bool) -> bool {
    EXTERNAL_ARTIFACT_SERDE.with(|mode| mode.replace(enabled))
}

fn external_artifact_serde_enabled() -> bool {
    EXTERNAL_ARTIFACT_SERDE.with(Cell::get)
}

fn compact_artifact_serde_enabled() -> bool {
    COMPACT_ARTIFACT_SERDE.with(Cell::get)
}

#[derive(Debug, Clone)]
pub(super) struct MatchedTerminalLists {
    offsets: Arc<[usize]>,
    entries: Arc<[TerminalID]>,
}

impl MatchedTerminalLists {
    #[inline]
    fn for_state(&self, state: u32) -> &[TerminalID] {
        let index = state as usize;
        let Some(&start) = self.offsets.get(index) else {
            return &[];
        };
        let Some(&end) = self.offsets.get(index + 1) else {
            return &[];
        };
        &self.entries[start..end]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[doc(hidden)]
pub enum VirtualTokenizerRuntimeKind {
    UnitRepeat,
    RepeatProduct,
    ResidualExpr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[doc(hidden)]
pub struct VirtualTokenizerRuntimeMetadata {
    pub kind: VirtualTokenizerRuntimeKind,
    pub terminal: TerminalID,
    pub root_state: u32,
}

/// Exact single-terminal residual-language quotient for one deterministic
/// tokenizer component.
///
/// The full tokenizer coordinate is stored sparsely rather than as one dense
/// row per tokenizer state. Keeping only states that actually belong to the
/// terminal's deterministic component avoids multiplying runtime memory by the
/// total tokenizer-state count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalProjectedQuotient {
    dfa: DFA,
    full_states: Box<[u32]>,
    projected_states: Box<[u32]>,
    /// Exact global byte-equivalence alphabet for `dfa`. Two bytes share a
    /// class iff their transition target is identical from every projected
    /// state (including both transitions being dead).
    byte_to_class: Box<[u8]>,
    class_representatives: Box<[u8]>,
    /// Dense projected-state × byte-class targets. `u32::MAX` is dead.
    class_targets: Box<[u32]>,
}

impl TerminalProjectedQuotient {
    #[inline]
    fn projected_state_live(dfa: &DFA, state: u32) -> bool {
        dfa.finalizers(state).contains(0) || dfa.possible_future_group_ids(state).contains(0)
    }

    #[inline]
    fn projected_step(dfa: &DFA, state: u32, byte: u8) -> Option<u32> {
        dfa.step(state, byte)
            .filter(|&target| Self::projected_state_live(dfa, target))
    }

    fn canonicalize_byte_class_map(classes: &[u8; 256]) -> ([u8; 256], Vec<u8>) {
        let mut remap = [u16::MAX; 256];
        let mut canonical = [0u8; 256];
        let mut representatives = Vec::<u8>::new();
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let old = classes[byte as usize] as usize;
            let new = if remap[old] == u16::MAX {
                let new = representatives.len() as u16;
                remap[old] = new;
                representatives.push(byte);
                new
            } else {
                remap[old]
            };
            canonical[byte as usize] = new as u8;
        }
        (canonical, representatives)
    }

    fn class_targets_for(
        dfa: &DFA,
        representatives: &[u8],
    ) -> Box<[u32]> {
        let mut targets = Vec::with_capacity(dfa.num_states().saturating_mul(representatives.len()));
        for state in 0..dfa.num_states() as u32 {
            for &representative in representatives {
                targets.push(Self::projected_step(dfa, state, representative).unwrap_or(u32::MAX));
            }
        }
        targets.into_boxed_slice()
    }

    fn exact_byte_classes_from_base(
        dfa: &DFA,
        base_byte_to_class: &[u8; 256],
    ) -> (Box<[u8]>, Box<[u8]>, Box<[u32]>) {
        let base_count = base_byte_to_class
            .iter()
            .copied()
            .max()
            .map_or(0usize, |class| class as usize + 1);
        if base_count == 0 {
            return Self::exact_byte_classes(dfa);
        }
        let mut base_representatives = vec![u8::MAX; base_count];
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let base = base_byte_to_class[byte as usize] as usize;
            if base_representatives[base] == u8::MAX {
                base_representatives[base] = byte;
            }
        }

        let mut symbol_classes = vec![0u8; base_count];
        let mut keyed = Vec::<(u64, usize)>::with_capacity(base_count);
        let mut refined = vec![0u8; base_count];
        for state in 0..dfa.num_states() as u32 {
            keyed.clear();
            for (symbol, &representative) in base_representatives.iter().enumerate() {
                let target_code =
                    Self::projected_step(dfa, state, representative).unwrap_or(u32::MAX) as u64
                        + 1;
                let key = ((symbol_classes[symbol] as u64) << 33) | target_code;
                keyed.push((key, symbol));
            }
            keyed.sort_unstable_by_key(|&(key, _)| key);
            let mut previous = None::<u64>;
            let mut next_class = 0usize;
            for &(key, symbol) in &keyed {
                if previous != Some(key) {
                    previous = Some(key);
                    next_class += 1;
                }
                refined[symbol] = (next_class - 1) as u8;
            }
            std::mem::swap(&mut symbol_classes, &mut refined);
        }

        let mut byte_classes = [0u8; 256];
        for byte in 0u16..=255 {
            let byte = byte as u8;
            byte_classes[byte as usize] =
                symbol_classes[base_byte_to_class[byte as usize] as usize];
        }
        let (byte_classes, representatives) = Self::canonicalize_byte_class_map(&byte_classes);
        let class_targets = Self::class_targets_for(dfa, &representatives);
        (
            byte_classes.to_vec().into_boxed_slice(),
            representatives.into_boxed_slice(),
            class_targets,
        )
    }

    fn exact_byte_classes(dfa: &DFA) -> (Box<[u8]>, Box<[u8]>, Box<[u32]>) {
        // Refine the byte partition incrementally by DFA row instead of
        // constructing one long state-target signature Vec for every byte.
        // `classes[a] == classes[b]` after row N iff bytes a and b have had the
        // same target from every state through N, so the final partition is the
        // exact global byte-equivalence relation.
        let mut classes = [0u8; 256];
        let mut class_count = 1usize;
        let mut targets = [u32::MAX; 256];
        let state_count = dfa.num_states();
        let target_stride = state_count + 1;
        let mut marks = vec![0u32; 256 * target_stride];
        let mut assigned = vec![0u8; 256 * target_stride];
        let mut generation = 0u32;
        let mut refined = [0u8; 256];

        for state in 0..state_count as u32 {
            if class_count == 256 {
                break;
            }
            generation = generation.wrapping_add(1);
            targets.fill(u32::MAX);
            for (byte, target) in dfa.transitions(state) {
                if Self::projected_state_live(dfa, target) {
                    targets[byte as usize] = target;
                }
            }
            let mut next_class = 0usize;
            for byte in 0u16..=255 {
                let byte = byte as u8;
                let old_class = classes[byte as usize] as usize;
                let target = targets[byte as usize];
                let target_index = if target == u32::MAX {
                    state_count
                } else {
                    target as usize
                };
                let key = old_class * target_stride + target_index;
                if marks[key] != generation {
                    marks[key] = generation;
                    assigned[key] = next_class as u8;
                    next_class += 1;
                }
                refined[byte as usize] = assigned[key];
            }
            classes = refined;
            class_count = next_class;
        }

        let (classes, class_representatives) = Self::canonicalize_byte_class_map(&classes);
        let class_targets = Self::class_targets_for(dfa, &class_representatives);
        (
            classes.to_vec().into_boxed_slice(),
            class_representatives.into_boxed_slice(),
            class_targets,
        )
    }

    fn from_source_terminal_subautomaton(
        tokenizer: &Tokenizer,
        terminal: TerminalID,
        component_states: &[u32],
    ) -> Option<Self> {
        let full_states = component_states
            .iter()
            .copied()
            .filter(|&state| tokenizer.state_live_for_terminal(state, terminal))
            .collect::<Vec<_>>();
        if full_states.is_empty() {
            return None;
        }
        let mut full_to_projected = vec![u32::MAX; tokenizer.num_states() as usize];
        for (projected, &full) in full_states.iter().enumerate() {
            full_to_projected[full as usize] = projected as u32;
        }

        // Exact byte equivalence over the terminal-live subgraph, refined row
        // by row without materializing a second transition-bearing DFA.
        let mut classes = [0u8; 256];
        let mut class_count = 1usize;
        let mut targets = [u32::MAX; 256];
        let state_count_for_classes = full_states.len();
        let target_stride = state_count_for_classes + 1;
        let mut marks = vec![0u32; 256 * target_stride];
        let mut assigned = vec![0u8; 256 * target_stride];
        let mut generation = 0u32;
        let mut refined = [0u8; 256];
        for &full_state in &full_states {
            if class_count == 256 {
                break;
            }
            generation = generation.wrapping_add(1);
            targets.fill(u32::MAX);
            for (byte, full_target) in tokenizer.transitions_from(full_state) {
                let projected = full_to_projected[full_target as usize];
                if projected != u32::MAX {
                    targets[byte as usize] = projected;
                }
            }
            let mut next_class = 0usize;
            for byte in 0u16..=255 {
                let byte = byte as u8;
                let old_class = classes[byte as usize] as usize;
                let target = targets[byte as usize];
                let target_index = if target == u32::MAX {
                    state_count_for_classes
                } else {
                    target as usize
                };
                let key = old_class * target_stride + target_index;
                if marks[key] != generation {
                    marks[key] = generation;
                    assigned[key] = next_class as u8;
                    next_class += 1;
                }
                refined[byte as usize] = assigned[key];
            }
            classes = refined;
            class_count = next_class;
        }
        let (classes, representatives) = Self::canonicalize_byte_class_map(&classes);

        let state_count = full_states.len();
        let mut dfa = DFA::new(state_count);
        dfa.ensure_group_capacity(1);
        if let Some(support) = tokenizer.terminal_byte_support(terminal) {
            dfa.set_group_u8set(0, support);
        }
        for (projected, &full_state) in full_states.iter().enumerate() {
            let mut finalizers = BitSet::new(1);
            if tokenizer
                .matched_terminal_bitset(full_state)
                .contains(terminal as usize)
            {
                finalizers.set(0);
            }
            let mut future = BitSet::new(1);
            if tokenizer
                .possible_future_terminals(full_state)
                .contains(terminal as usize)
            {
                future.set(0);
            }
            dfa.overwrite_state_metadata(projected as u32, finalizers, future);
        }

        let mut class_targets = Vec::with_capacity(state_count * representatives.len());
        for &full_state in &full_states {
            for &byte in &representatives {
                let projected = tokenizer
                    .step(full_state, byte)
                    .and_then(|full_target| {
                        let projected = full_to_projected[full_target as usize];
                        (projected != u32::MAX).then_some(projected)
                    })
                    .unwrap_or(u32::MAX);
                class_targets.push(projected);
            }
        }
        let projected_states = (0..state_count as u32).collect::<Vec<_>>();
        Some(Self {
            dfa,
            full_states: full_states.into_boxed_slice(),
            projected_states: projected_states.into_boxed_slice(),
            byte_to_class: classes.to_vec().into_boxed_slice(),
            class_representatives: representatives.into_boxed_slice(),
            class_targets: class_targets.into_boxed_slice(),
        })
    }

    fn from_dense_mapping(
        dfa: DFA,
        full_to_projected: Vec<u32>,
    ) -> Self {
        let mapped = full_to_projected
            .iter()
            .filter(|&&projected_state| projected_state != u32::MAX)
            .count();
        let mut full_states = Vec::with_capacity(mapped);
        let mut projected_states = Vec::with_capacity(mapped);
        for (full_state, projected_state) in full_to_projected.into_iter().enumerate() {
            if projected_state == u32::MAX {
                continue;
            }
            full_states.push(full_state as u32);
            projected_states.push(projected_state);
        }
        debug_assert_eq!(full_states.len(), projected_states.len());
        let (byte_to_class, class_representatives, class_targets) = Self::exact_byte_classes(&dfa);
        Self {
            dfa,
            full_states: full_states.into_boxed_slice(),
            projected_states: projected_states.into_boxed_slice(),
            byte_to_class,
            class_representatives,
            class_targets,
        }
    }

    fn from_dense_mapping_with_base_classes(
        dfa: DFA,
        full_to_projected: Vec<u32>,
        base_byte_to_class: &[u8; 256],
    ) -> Self {
        let mapped = full_to_projected
            .iter()
            .filter(|&&projected_state| projected_state != u32::MAX)
            .count();
        let mut full_states = Vec::with_capacity(mapped);
        let mut projected_states = Vec::with_capacity(mapped);
        for (full_state, projected_state) in full_to_projected.into_iter().enumerate() {
            if projected_state == u32::MAX {
                continue;
            }
            full_states.push(full_state as u32);
            projected_states.push(projected_state);
        }
        debug_assert_eq!(full_states.len(), projected_states.len());
        let (byte_to_class, class_representatives, class_targets) =
            Self::exact_byte_classes_from_base(&dfa, base_byte_to_class);
        Self {
            dfa,
            full_states: full_states.into_boxed_slice(),
            projected_states: projected_states.into_boxed_slice(),
            byte_to_class,
            class_representatives,
            class_targets,
        }
    }

    #[inline]
    fn class_target(&self, state: u32, class: u8) -> Option<u32> {
        let index = (state as usize)
            .checked_mul(self.class_representatives.len())?
            .checked_add(class as usize)?;
        let target = *self.class_targets.get(index)?;
        (target != u32::MAX).then_some(target)
    }

    fn classes_for_bytes(&self, bytes: impl Iterator<Item = u8>) -> Vec<u8> {
        let mut present = [false; 256];
        for byte in bytes {
            present[self.byte_to_class[byte as usize] as usize] = true;
        }
        present
            .iter()
            .enumerate()
            .filter_map(|(class, &is_present)| is_present.then_some(class as u8))
            .collect()
    }

    fn classes_for_ranges(&self, ranges: &[(u8, u8)]) -> Vec<u8> {
        self.classes_for_bytes(
            ranges
                .iter()
                .flat_map(|&(first, last)| first..=last),
        )
    }

    #[inline]
    fn projected_state(&self, full_source: u32) -> Option<u32> {
        let index = self.full_states.binary_search(&full_source).ok()?;
        self.projected_states.get(index).copied()
    }

    #[inline]
    pub fn contains_source(&self, source: u32) -> bool {
        self.projected_state(source).is_some()
    }

    #[doc(hidden)]
    #[inline]
    pub fn projected_state_for_source(&self, source: u32) -> Option<u32> {
        self.projected_state(source)
    }

    #[doc(hidden)]
    #[inline]
    pub fn projected_byte_class(&self, byte: u8) -> u8 {
        self.byte_to_class[byte as usize]
    }

    #[doc(hidden)]
    #[inline]
    pub fn projected_step_class(&self, state: u32, class: u8) -> Option<u32> {
        self.class_target(state, class)
    }

    #[doc(hidden)]
    #[inline]
    pub fn projected_state_is_accepting(&self, state: u32) -> bool {
        self.dfa.finalizers(state).contains(0)
    }

    #[doc(hidden)]
    #[inline]
    pub fn projected_state_has_future(&self, state: u32) -> bool {
        self.dfa.possible_future_group_ids(state).contains(0)
    }

    #[doc(hidden)]
    #[inline]
    pub fn projected_state_count(&self) -> usize {
        self.dfa.num_states()
    }

    #[doc(hidden)]
    #[inline]
    pub fn projected_source_states(&self) -> &[u32] {
        &self.full_states
    }

    #[doc(hidden)]
    #[inline]
    pub fn projected_states_for_sources(&self) -> &[u32] {
        &self.projected_states
    }

    fn state_counts(&self) -> (usize, usize) {
        (self.full_states.len(), self.dfa.num_states())
    }

    fn byte_class_count(&self) -> usize {
        self.class_representatives.len()
    }

    /// Validate that a deserialized quotient is still an exact homomorphic
    /// projection of `terminal` in `tokenizer`.
    ///
    /// Runtime subtree acceptance relies on this relationship, so wire data is
    /// not trusted merely because its internal arrays are in bounds.  The
    /// check also reconstructs the DFA's exact byte partition instead of
    /// trusting serialized class metadata.
    pub fn validate_exact_projection(
        &self,
        tokenizer: &Tokenizer,
        terminal: TerminalID,
    ) -> Result<(), String> {
        if terminal >= tokenizer.num_terminals() {
            return Err(format!(
                "projected-terminal quotient references terminal {terminal}, but tokenizer has {} terminals",
                tokenizer.num_terminals(),
            ));
        }
        if self.dfa.has_epsilon_transitions() {
            return Err("projected-terminal quotient DFA has epsilon transitions".to_owned());
        }
        if self.full_states.is_empty() || self.full_states.len() != self.projected_states.len() {
            return Err("projected-terminal quotient has an invalid sparse state map".to_owned());
        }
        if self
            .full_states
            .windows(2)
            .any(|states| states[0] >= states[1])
        {
            return Err(
                "projected-terminal quotient full-state map is not strictly sorted".to_owned(),
            );
        }
        if self.byte_to_class.len() != 256 {
            return Err(format!(
                "projected-terminal quotient has {} byte-class entries, expected 256",
                self.byte_to_class.len(),
            ));
        }
        let class_count = self.class_representatives.len();
        if class_count == 0 || class_count > 256 {
            return Err(format!(
                "projected-terminal quotient has invalid byte-class count {class_count}"
            ));
        }
        if self
            .byte_to_class
            .iter()
            .any(|&class| class as usize >= class_count)
        {
            return Err("projected-terminal quotient byte-class map is out of bounds".to_owned());
        }
        for (class, &representative) in self.class_representatives.iter().enumerate() {
            if self.byte_to_class[representative as usize] as usize != class {
                return Err(
                    "projected-terminal quotient class representative does not map to its class"
                        .to_owned(),
                );
            }
        }
        let expected_target_len = self
            .dfa
            .num_states()
            .checked_mul(class_count)
            .ok_or_else(|| "projected-terminal quotient class-target size overflow".to_owned())?;
        if self.class_targets.len() != expected_target_len {
            return Err(format!(
                "projected-terminal quotient has {} class targets, expected {expected_target_len}",
                self.class_targets.len(),
            ));
        }
        if self
            .class_targets
            .iter()
            .any(|&target| target != u32::MAX && target >= self.dfa.num_states() as u32)
        {
            return Err("projected-terminal quotient class target is out of bounds".to_owned());
        }

        for state in 0..self.dfa.num_states() as u32 {
            if self.dfa.finalizers(state).iter().any(|group| group != 0)
                || self
                    .dfa
                    .possible_future_group_ids(state)
                    .iter()
                    .any(|group| group != 0)
            {
                return Err(
                    "projected-terminal quotient DFA contains a nonzero terminal group".to_owned(),
                );
            }
        }

        for (&full_state, &projected_state) in
            self.full_states.iter().zip(self.projected_states.iter())
        {
            if full_state >= tokenizer.num_states() {
                return Err(format!(
                    "projected-terminal quotient references tokenizer state {full_state}, but tokenizer has {} states",
                    tokenizer.num_states(),
                ));
            }
            if projected_state >= self.dfa.num_states() as u32 {
                return Err(format!(
                    "projected-terminal quotient references projected state {projected_state}, but DFA has {} states",
                    self.dfa.num_states(),
                ));
            }

            let full_accepting = tokenizer
                .matched_terminal_bitset(full_state)
                .contains(terminal as usize);
            let projected_accepting = self.dfa.finalizers(projected_state).contains(0);
            let full_future = tokenizer
                .possible_future_terminals(full_state)
                .contains(terminal as usize);
            let projected_future = self
                .dfa
                .possible_future_group_ids(projected_state)
                .contains(0);
            if full_accepting != projected_accepting || full_future != projected_future {
                return Err(format!(
                    "projected-terminal quotient observation mismatch at tokenizer state {full_state}"
                ));
            }
            if !full_accepting && !full_future {
                return Err(format!(
                    "projected-terminal quotient maps dead tokenizer state {full_state}"
                ));
            }

            for byte in 0u16..=255 {
                let byte = byte as u8;
                let full_target =
                    tokenizer.terminal_projected_scalar_step(full_state, terminal, byte);
                let class = self.byte_to_class[byte as usize];
                let projected_target = self.class_target(projected_state, class);
                match (full_target, projected_target) {
                    (None, None) => {}
                    (Some(full_target), Some(projected_target))
                        if self.projected_state(full_target) == Some(projected_target) => {}
                    _ => {
                        return Err(format!(
                            "projected-terminal quotient transition mismatch at tokenizer state {full_state} on byte {byte}"
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Prove that every requested text continuation remains live in this exact
    /// single-terminal residual coordinate. The quotient has one accepting
    /// group, so residual distinctions belonging only to unrelated terminals
    /// are absent from the proof state.
    pub fn text_liveness_closed_bounded(
        &self,
        full_source: u32,
        ascii_bytes: U8Set,
        state_limit: usize,
        transition_work_limit: usize,
        include_utf8_scalars: bool,
    ) -> Option<bool> {
        let source = self.projected_state(full_source)?;
        if state_limit == 0 || transition_work_limit == 0 {
            return Some(false);
        }
        let safe_state = |state: u32| self.dfa.possible_future_group_ids(state).contains(0);
        if !safe_state(source) {
            return Some(false);
        }

        let utf8_branches: &[(&[(u8, u8)], &[(u8, u8)], usize)] = &[
            (&[(0xC2, 0xDF)], &[(0x80, 0xBF)], 0),
            (&[(0xE0, 0xE0)], &[(0xA0, 0xBF)], 1),
            (&[(0xE1, 0xEC), (0xEE, 0xEF)], &[(0x80, 0xBF)], 1),
            (&[(0xED, 0xED)], &[(0x80, 0x9F)], 1),
            (&[(0xF0, 0xF0)], &[(0x90, 0xBF)], 2),
            (&[(0xF1, 0xF3)], &[(0x80, 0xBF)], 2),
            (&[(0xF4, 0xF4)], &[(0x80, 0x8F)], 2),
        ];

        let ascii_classes =
            self.classes_for_bytes(ascii_bytes.iter().filter(|&byte| byte < 0x80));
        let continuation_classes = self.classes_for_ranges(&[(0x80, 0xBF)]);
        let utf8_class_branches = utf8_branches
            .iter()
            .map(|&(lead, second, extra_continuations)| {
                (
                    self.classes_for_ranges(lead),
                    self.classes_for_ranges(second),
                    extra_continuations,
                )
            })
            .collect::<Vec<_>>();

        let mut seen = FxHashSet::<u32>::default();
        let mut queue = VecDeque::from([source]);
        let mut work = 0usize;
        while let Some(state) = queue.pop_front() {
            if !seen.insert(state) {
                continue;
            }
            if seen.len() > state_limit {
                return None;
            }
            if !safe_state(state) {
                return Some(false);
            }

            for &class in &ascii_classes {
                work = work.saturating_add(1);
                if work > transition_work_limit {
                    return None;
                }
                let Some(target) = self.class_target(state, class) else {
                    return Some(false);
                };
                if !safe_state(target) {
                    return Some(false);
                }
                if !seen.contains(&target) {
                    queue.push_back(target);
                }
            }

            let advance_classes = |frontier: &[u32],
                                   classes: &[u8],
                                   work: &mut usize|
             -> Option<Option<Vec<u32>>> {
                let mut next = Vec::<u32>::new();
                for &frontier_state in frontier {
                    for &class in classes {
                        *work = work.saturating_add(1);
                        if *work > transition_work_limit {
                            return None;
                        }
                        let Some(target) = self.class_target(frontier_state, class) else {
                            return Some(None);
                        };
                        if !safe_state(target) {
                            return Some(None);
                        }
                        if !next.contains(&target) {
                            next.push(target);
                        }
                    }
                }
                next.sort_unstable();
                Some(Some(next))
            };

            if include_utf8_scalars {
                for (lead_classes, second_classes, extra_continuations) in &utf8_class_branches {
                    let Some(mut frontier) =
                        advance_classes(&[state], lead_classes, &mut work)?
                    else {
                        return Some(false);
                    };
                    let Some(after_second) =
                        advance_classes(&frontier, second_classes, &mut work)?
                    else {
                        return Some(false);
                    };
                    frontier = after_second;
                    for _ in 0..*extra_continuations {
                        let Some(after_continuation) = advance_classes(
                            &frontier,
                            &continuation_classes,
                            &mut work,
                        )?
                        else {
                            return Some(false);
                        };
                        frontier = after_continuation;
                    }
                    for target in frontier {
                        if !seen.contains(&target) {
                            queue.push_back(target);
                        }
                    }
                }
            }
        }
        Some(true)
    }

}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TerminalExclusionResidualState {
    pub(crate) left_state: u32,
    /// `u32::MAX` means the finite exclusion component is already dead.
    pub(crate) right_state: u32,
    /// Exact maximum bytes remaining to an exclusion match while `right_state`
    /// is live. `u32::MAX` is used only when the exclusion is already dead.
    pub(crate) right_max_remaining: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct TerminalExclusionCertificate {
    left_dfa: Arc<DFA>,
    states: Arc<[TerminalExclusionResidualState]>,
}

impl TerminalExclusionCertificate {
    pub(crate) fn new(
        left_dfa: Arc<DFA>,
        states: Vec<TerminalExclusionResidualState>,
    ) -> Option<Self> {
        (!states.is_empty()).then(|| Self {
            left_dfa,
            states: Arc::from(states.into_boxed_slice()),
        })
    }
}

/// Exact compile-time certificate for one raw tokenizer state inside a
/// top-level `Exclude(left, right)` terminal. This is intentionally small and
/// runtime-facing: all expensive expression/product analysis is completed
/// while the tokenizer is built.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalExclusionContinuation {
    pub terminal_residual_state: u32,
    pub left_state: u32,
    pub right_state: Option<u32>,
    pub right_max_remaining: Option<u32>,
}

/// Exact standalone-terminal residual coordinate paired with the raw combined
/// tokenizer state that produced it. Runtime direct walkers use this only when
/// the compiler-retained product-trace sidecar proves the mapping.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TerminalResidualDirectCoordinate {
    raw_state: u32,
    terminal: TerminalID,
    residual_state: u32,
}

#[derive(Debug, Clone)]
pub struct TerminalResidualCoordinates {
    offsets: Arc<[u32]>,
    entries: Arc<[(u32, u32)]>,
    terminal_dfas: Arc<[Arc<DFA>]>,
    terminal_groups: Arc<[u32]>,
    exclusion_certificates: Arc<[Option<Arc<TerminalExclusionCertificate>>]>,
}

impl TerminalResidualCoordinates {
    pub fn from_rows(rows: Vec<Vec<(u32, u32)>>) -> Self {
        Self::from_rows_and_dfas(rows, Vec::new())
    }

    pub fn from_rows_and_dfas(rows: Vec<Vec<(u32, u32)>>, terminal_dfas: Vec<Arc<DFA>>) -> Self {
        let terminal_groups = vec![0u32; terminal_dfas.len()];
        Self::from_rows_and_dfa_groups(rows, terminal_dfas, terminal_groups)
    }

    pub fn from_rows_and_dfa_groups(
        rows: Vec<Vec<(u32, u32)>>,
        terminal_dfas: Vec<Arc<DFA>>,
        terminal_groups: Vec<u32>,
    ) -> Self {
        debug_assert_eq!(terminal_dfas.len(), terminal_groups.len());
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        let mut entries = Vec::with_capacity(rows.iter().map(Vec::len).sum());
        offsets.push(0);
        for row in rows {
            entries.extend(row);
            offsets.push(entries.len() as u32);
        }
        let exclusion_certificates = vec![None; terminal_dfas.len()];
        Self {
            offsets: Arc::from(offsets.into_boxed_slice()),
            entries: Arc::from(entries.into_boxed_slice()),
            terminal_dfas: Arc::from(terminal_dfas.into_boxed_slice()),
            terminal_groups: Arc::from(terminal_groups.into_boxed_slice()),
            exclusion_certificates: Arc::from(exclusion_certificates.into_boxed_slice()),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    pub fn row(&self, state: u32) -> Option<&[(u32, u32)]> {
        let state = state as usize;
        let start = *self.offsets.get(state)? as usize;
        let end = *self.offsets.get(state + 1)? as usize;
        Some(&self.entries[start..end])
    }

    #[inline]
    pub fn terminal_dfa(&self, terminal: u32) -> Option<&DFA> {
        self.terminal_dfas.get(terminal as usize).map(Arc::as_ref)
    }

    #[inline]
    pub fn terminal_dfa_and_group(&self, terminal: u32) -> Option<(&DFA, u32)> {
        Some((
            self.terminal_dfas.get(terminal as usize)?.as_ref(),
            *self.terminal_groups.get(terminal as usize)?,
        ))
    }

    #[inline]
    pub fn terminal_dfa_count(&self) -> usize {
        self.terminal_dfas.len()
    }

    pub(crate) fn with_exclusion_certificates(
        mut self,
        certificates: Vec<Option<Arc<TerminalExclusionCertificate>>>,
    ) -> Option<Self> {
        if certificates.len() != self.terminal_dfas.len() {
            return None;
        }
        self.exclusion_certificates = Arc::from(certificates.into_boxed_slice());
        Some(self)
    }

    #[inline]
    pub(crate) fn terminal_exclusion_certificate(
        &self,
        terminal: TerminalID,
    ) -> Option<&TerminalExclusionCertificate> {
        self.exclusion_certificates
            .get(terminal as usize)?
            .as_deref()
    }

    fn replace_terminals_with_appended_dfas(
        &self,
        replacements: &[(TerminalID, Arc<DFA>)],
    ) -> Option<Self> {
        if replacements.is_empty() {
            return Some(self.clone());
        }

        // `install_direct_mask_components` appends every component first and
        // then updates this sidecar. Build the final coordinate table once,
        // rather than rebuilding all existing rows after every component.
        // Keep the exact sequential semantics even if a future caller supplies
        // the same terminal more than once: rows belonging to an earlier
        // replacement remain present but have that terminal coordinate removed,
        // while the final replacement owns the terminal's standalone identity
        // rows and DFA metadata.
        let terminal_count = self.terminal_dfas.len();
        let mut last_replacement = vec![usize::MAX; terminal_count];
        let mut appended_states = 0usize;
        for (index, (terminal, dfa)) in replacements.iter().enumerate() {
            let terminal_index = *terminal as usize;
            if terminal_index >= terminal_count {
                return None;
            }
            last_replacement[terminal_index] = index;
            appended_states = appended_states.checked_add(dfa.num_states())?;
        }

        let mut offsets = Vec::with_capacity(
            self.len()
                .checked_add(appended_states)?
                .checked_add(1)?,
        );
        let mut entries = Vec::with_capacity(self.entries.len().saturating_add(appended_states));
        offsets.push(0u32);
        for state in 0..self.len() {
            entries.extend(
                self.row(state as u32)?
                    .iter()
                    .copied()
                    .filter(|&(terminal, _)| {
                        last_replacement
                            .get(terminal as usize)
                            .map_or(true, |&replacement| replacement == usize::MAX)
                    }),
            );
            offsets.push(u32::try_from(entries.len()).ok()?);
        }

        for (index, (terminal, dfa)) in replacements.iter().enumerate() {
            let is_final = last_replacement[*terminal as usize] == index;
            for state in 0..dfa.num_states() {
                if is_final {
                    entries.push((*terminal, u32::try_from(state).ok()?));
                }
                offsets.push(u32::try_from(entries.len()).ok()?);
            }
        }
        let mut terminal_dfas = self.terminal_dfas.iter().cloned().collect::<Vec<_>>();
        let mut terminal_groups = self.terminal_groups.iter().copied().collect::<Vec<_>>();
        let mut exclusion_certificates = self.exclusion_certificates.iter().cloned().collect::<Vec<_>>();
        for (terminal, dfa) in replacements {
            let terminal_index = *terminal as usize;
            terminal_dfas[terminal_index] = Arc::clone(dfa);
            terminal_groups[terminal_index] = 0;
            // The appended standalone DFA has different state numbering from
            // the compile-time exclusion projection. Fail closed rather than
            // retaining stale proof metadata.
            exclusion_certificates[terminal_index] = None;
        }
        Some(Self {
            offsets: Arc::from(offsets.into_boxed_slice()),
            entries: Arc::from(entries.into_boxed_slice()),
            terminal_dfas: Arc::from(terminal_dfas.into_boxed_slice()),
            terminal_groups: Arc::from(terminal_groups.into_boxed_slice()),
            exclusion_certificates: Arc::from(exclusion_certificates.into_boxed_slice()),
        })
    }

    fn replace_terminal_alias_groups_with_appended_dfas(
        &self,
        replacements: &[(Vec<TerminalID>, Arc<DFA>)],
    ) -> Option<Self> {
        if replacements.is_empty() {
            return Some(self.clone());
        }

        let terminal_count = self.terminal_dfas.len();
        let mut replaced = vec![false; terminal_count];
        let mut appended_states = 0usize;
        let mut appended_entries = 0usize;
        for (terminals, dfa) in replacements {
            if terminals.is_empty() {
                return None;
            }
            appended_states = appended_states.checked_add(dfa.num_states())?;
            appended_entries = appended_entries.checked_add(
                dfa.num_states().checked_mul(terminals.len())?,
            )?;
            for &terminal in terminals {
                let terminal_index = terminal as usize;
                if terminal_index >= terminal_count || replaced[terminal_index] {
                    return None;
                }
                replaced[terminal_index] = true;
            }
        }

        let mut offsets = Vec::with_capacity(
            self.len()
                .checked_add(appended_states)?
                .checked_add(1)?,
        );
        let mut entries = Vec::with_capacity(self.entries.len().saturating_add(appended_entries));
        offsets.push(0u32);
        for state in 0..self.len() {
            entries.extend(
                self.row(state as u32)?
                    .iter()
                    .copied()
                    .filter(|&(terminal, _)| {
                        replaced
                            .get(terminal as usize)
                            .is_none_or(|&is_replaced| !is_replaced)
                    }),
            );
            offsets.push(u32::try_from(entries.len()).ok()?);
        }

        for (terminals, dfa) in replacements {
            for state in 0..dfa.num_states() {
                let local_state = u32::try_from(state).ok()?;
                entries.extend(terminals.iter().map(|&terminal| (terminal, local_state)));
                offsets.push(u32::try_from(entries.len()).ok()?);
            }
        }

        let mut terminal_dfas = self.terminal_dfas.iter().cloned().collect::<Vec<_>>();
        let mut terminal_groups = self.terminal_groups.iter().copied().collect::<Vec<_>>();
        let mut exclusion_certificates = self.exclusion_certificates.iter().cloned().collect::<Vec<_>>();
        for (terminals, dfa) in replacements {
            for &terminal in terminals {
                let terminal_index = terminal as usize;
                terminal_dfas[terminal_index] = Arc::clone(dfa);
                terminal_groups[terminal_index] = 0;
                exclusion_certificates[terminal_index] = None;
            }
        }

        Some(Self {
            offsets: Arc::from(offsets.into_boxed_slice()),
            entries: Arc::from(entries.into_boxed_slice()),
            terminal_dfas: Arc::from(terminal_dfas.into_boxed_slice()),
            terminal_groups: Arc::from(terminal_groups.into_boxed_slice()),
            exclusion_certificates: Arc::from(exclusion_certificates.into_boxed_slice()),
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Tokenizer {
    pub(super) dfa: DFA,
    pub(super) num_terminals: u32,
    /// Current static artifacts may retain their canonical packed transition
    /// topology directly. The DFA then carries only state metadata/epsilon
    /// structure; scalar byte lookup and transition iteration use this sidecar.
    #[serde(default, skip)]
    pub(super) packed_runtime_transitions: Option<Arc<PackedRuntimeTransitions>>,
    /// Rebased borrowed packed transition blocks retained by structural
    /// tokenizer composition. Loaded tokenizers keep ordinary sparse byte
    /// transitions in `packed_runtime_transitions` while their DFA rows are
    /// metadata-only stubs; dropping that sidecar during composition silently
    /// deletes lexer edges. Segments preserve those blocks without expanding
    /// them into dense DFA rows.
    #[serde(default, skip)]
    pub(super) packed_runtime_transition_segments: Arc<[PackedRuntimeTransitionSegment]>,
    /// Runtime-only exact byte-class transition segments. The historical
    /// serialized tokenizer shape contains only `dfa` and `num_terminals`; the
    /// custom serializer expands these segments into that same DFA wire form.
    #[serde(default, skip)]
    pub(super) compressed_transition_segments: Arc<[CompressedTransitionSegment]>,
    /// Current giant-tokenizer artifacts keep per-state labels/epsilon rows in
    /// a compact dictionary-backed sidecar instead of allocating one DFAState
    /// for every serialized state. Freshly compiled tokenizers leave this
    /// empty and continue to use the ordinary DFA representation.
    #[serde(default, skip)]
    pub(super) packed_runtime_metadata: Option<Arc<PackedTokenizerMetadata>>,
    /// Rebased packed metadata blocks retained by structural composition of
    /// loaded tokenizers. Like packed transition segments, these keep TKS3
    /// finalizer/future/epsilon metadata zero-copy instead of expanding one DFA
    /// state record per serialized state.
    #[serde(default, skip)]
    pub(super) packed_runtime_metadata_segments: Arc<[PackedTokenizerMetadataSegment]>,
    /// Dictionary-backed byte-class transition segments used by the compact
    /// giant-tokenizer wire. These are semantically identical to
    /// `compressed_transition_segments`, but intern repeated rows and encode
    /// targets as source-relative deltas.
    #[serde(default, skip)]
    pub(super) packed_compressed_transition_segments: Arc<[PackedCompressedTransitionSegment]>,
    /// Exact arithmetic residuals for the supported standalone bounded-repeat
    /// runtime lane. Shared across clones; it has no per-residual allocation.
    #[serde(default, skip)]
    pub(super) virtual_unit_repeat: Option<Arc<VirtualZeroMinUnitRepeatRuntime>>,
    /// Exact lazily interned product residuals for pathological bounded-repeat
    /// terminals. Bounds do not size these stores; only states actually
    /// reached by runtime input are interned, with globally unique handles
    /// allocated across all components.
    #[serde(default, skip)]
    pub(super) virtual_repeat_intersections: Vec<Arc<VirtualBinaryRepeatIntersectionRuntime>>,
    /// General exact lazy regex residuals. Specialized repeat runtimes are
    /// selected first; this is the compositional fallback for expressions that
    /// should stay symbolic instead of being eagerly determinized.
    #[serde(default, skip)]
    pub(super) virtual_residuals: Vec<Arc<VirtualResidualRuntime>>,
    /// Per-terminal regex expressions used to (re)build this tokenizer.
    /// Skipped during (de)serialization because they are only needed during
    /// compile-time simplification for active-terminal rebuilds.
    #[serde(default, skip)]
    pub(super) exprs: Option<Arc<[Expr]>>,
    /// Compile-time-only exact mapping from tokenizer states to the residual
    /// states of independently compiled terminal DFAs. This is populated only
    /// by tokenizer builders that preserve product coordinates and is never
    /// serialized into runtime artifacts.
    #[serde(default, skip)]
    pub(super) terminal_residual_coordinates: Option<Arc<TerminalResidualCoordinates>>,
    /// Derived epsilon closures are shared by compile-time analyses.  A
    /// partitioned lexer is queried by many concurrent compiler lanes; without
    /// this cache each lane independently walks the same epsilon DAG for every
    /// raw state.
    #[serde(default, skip)]
    pub(super) singleton_epsilon_closures: OnceLock<Arc<SingletonEpsilonClosures>>,
    /// Sparse accepting-terminal labels per raw state. Runtime commit scans
    /// iterate these lists instead of rescanning a terminal-domain-sized bitset.
    #[serde(default, skip)]
    pub(super) matched_terminals_cache: OnceLock<Arc<MatchedTerminalLists>>,
    /// Exact epsilon-closed frontier after consuming one byte from tokenizer reset.
    #[serde(default, skip)]
    pub(super) initial_byte_frontiers: OnceLock<Arc<[TokenizerStateSet]>>,
    /// Per-state byte sets whose transitions loop to the same raw tokenizer
    /// state. Compiler partitions reuse this table instead of rescanning every
    /// transition row independently.
    #[serde(default, skip)]
    pub(super) all_self_loop_bytes_cache: OnceLock<Arc<[U8Set]>>,
    /// Exact expanded byte-transition count, including compressed segments.
    /// This is immutable after construction except in the two structural
    /// tokenizer transforms, which invalidate all derived caches together.
    #[serde(default, skip)]
    pub(super) transition_count_cache: OnceLock<usize>,
    /// State count after a forced full DFA minimization. Used only by explicit
    /// diagnostics/regression tests; the expensive computation is cached.
    #[serde(default, skip)]
    pub(super) forced_minimized_state_count_cache: OnceLock<usize>,
    /// Whether reset dispatch is followed only by scalar byte transitions.
    /// Several compiler paths query this structural invariant repeatedly; cache
    /// the full reachable-graph proof rather than walking the tokenizer per
    /// vocabulary node or build stage.
    #[serde(default, skip)]
    pub(super) scalar_deterministic_dispatch_cache: OnceLock<bool>,
    /// Sorted/deduplicated reset-dispatch roots. The dynamic scalar-dispatch
    /// mask path needs this order on every mask; cache the pure function of
    /// the immutable reset structure instead of reallocating/sorting per mask.
    #[serde(default, skip)]
    pub(super) sorted_dispatch_roots_cache: OnceLock<Arc<Vec<u32>>>,
    /// Per-state first-byte sets (union of `transitions_from` bytes). The
    /// dynamic pre-collapse gate unions these over root physical states on
    /// every scalar mask; per-state lazy cells avoid both per-mask transition
    /// rescans and a first-mask full-table build spike.
    #[serde(default, skip)]
    pub(super) state_first_bytes_cache: OnceLock<Arc<[OnceLock<U8Set>]>>,
}

/// Exact dynamic residual component whose bounded-code liveness proof has
/// already been constructed. The oracle stays opaque outside the lexer crate;
/// callers can prepare these independently of a physical tokenizer and later
/// install them without rebuilding the proof.
#[doc(hidden)]
pub struct PreparedVirtualResidualComponent {
    expression: Expr,
    terminal: TerminalID,
    liveness_oracle: BoundedCodeIntersectionOracle,
}

#[derive(Debug, Clone)]
pub struct PackedRuntimeTransitions {
    byte_offsets: Arc<[u32]>,
    bytes: PackedRuntimeBytes,
    targets: PackedRuntimeTargets,
}

#[derive(Debug, Clone)]
pub(super) struct PackedRuntimeTransitionSegment {
    state_offset: u32,
    transitions: Arc<PackedRuntimeTransitions>,
}

impl PackedRuntimeTransitionSegment {
    #[inline]
    fn contains_state(&self, state: u32) -> bool {
        state >= self.state_offset
            && (state - self.state_offset) < self.transitions.state_count() as u32
    }

    #[inline]
    fn transition(&self, state: u32, byte: u8) -> Option<u32> {
        self.transitions
            .transition(state - self.state_offset, byte)
            .and_then(|target| self.state_offset.checked_add(target))
    }

    #[inline]
    fn row(&self, state: u32) -> Option<(&[u8], PackedRuntimeTargetSlice<'_>)> {
        self.transitions.row(state - self.state_offset)
    }
}

#[derive(Debug, Clone)]
enum PackedRowIds {
    U8(Arc<[u8]>),
    U16(Arc<[u16]>),
    U32(Arc<[u32]>),
    BackedU8 {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
    BackedU16 {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
    BackedU32 {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

impl PackedRowIds {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::U8(ids) => ids.len(),
            Self::U16(ids) => ids.len(),
            Self::U32(ids) => ids.len(),
            Self::BackedU8 { len, .. }
            | Self::BackedU16 { len, .. }
            | Self::BackedU32 { len, .. } => *len,
        }
    }

    #[inline]
    fn get(&self, index: usize) -> Option<usize> {
        match self {
            Self::U8(ids) => ids.get(index).copied().map(usize::from),
            Self::U16(ids) => ids.get(index).copied().map(usize::from),
            Self::U32(ids) => ids.get(index).copied().map(|value| value as usize),
            Self::BackedU8 {
                backing,
                start,
                len,
            } => (index < *len)
                .then(|| backing[*start + index] as usize),
            Self::BackedU16 {
                backing,
                start,
                len,
            } => {
                if index >= *len {
                    return None;
                }
                let offset = *start + index * 2;
                let bytes = backing.get(offset..offset + 2)?;
                Some(u16::from_le_bytes([bytes[0], bytes[1]]) as usize)
            }
            Self::BackedU32 {
                backing,
                start,
                len,
            } => {
                if index >= *len {
                    return None;
                }
                let offset = *start + index * 4;
                let bytes = backing.get(offset..offset + 4)?;
                Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize)
            }
        }
    }

    fn all_lt(&self, limit: usize) -> bool {
        match self {
            Self::U8(ids) => ids.iter().all(|&id| (id as usize) < limit),
            Self::U16(ids) => ids.iter().all(|&id| (id as usize) < limit),
            Self::U32(ids) => ids.iter().all(|&id| (id as usize) < limit),
            Self::BackedU8 {
                backing,
                start,
                len,
            } => backing
                .get(*start..*start + *len)
                .is_some_and(|ids| ids.iter().all(|&id| (id as usize) < limit)),
            Self::BackedU16 {
                backing,
                start,
                len,
            } => backing
                .get(*start..*start + *len * 2)
                .is_some_and(|bytes| {
                    bytes
                        .chunks_exact(2)
                        .all(|b| (u16::from_le_bytes([b[0], b[1]]) as usize) < limit)
                }),
            Self::BackedU32 {
                backing,
                start,
                len,
            } => backing
                .get(*start..*start + *len * 4)
                .is_some_and(|bytes| {
                    bytes.chunks_exact(4).all(|b| {
                        (u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize) < limit
                    })
                }),
        }
    }
}

#[derive(Debug, Clone)]
enum PackedI16Values {
    Owned(Arc<[i16]>),
    Backed {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

impl PackedI16Values {
    #[inline]
    fn get(&self, index: usize) -> Option<i16> {
        match self {
            Self::Owned(values) => values.get(index).copied(),
            Self::Backed {
                backing,
                start,
                len,
            } => {
                if index >= *len {
                    return None;
                }
                let offset = *start + index * 2;
                let bytes = backing.get(offset..offset + 2)?;
                Some(i16::from_le_bytes([bytes[0], bytes[1]]))
            }
        }
    }

    #[inline]
    fn backed_le_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Owned(_) => None,
            Self::Backed {
                backing,
                start,
                len,
            } => backing.get(*start..start.checked_add(len.checked_mul(2)?)?),
        }
    }

    #[inline]
    fn owned_values(&self) -> Option<&[i16]> {
        match self {
            Self::Owned(values) => Some(values),
            Self::Backed { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct PackedTokenizerMetadata {
    state_count: u32,
    finalizer_row_ids: PackedRowIds,
    finalizer_rows: Arc<[BitSet]>,
    finalizer_lists: Arc<[Box<[TerminalID]>]>,
    future_row_ids: PackedRowIds,
    future_rows: Arc<[BitSet]>,
    epsilon_states: Arc<[u32]>,
    epsilon_offsets: Arc<[u32]>,
    epsilon_targets: Arc<[u32]>,
}

#[derive(Debug, Clone)]
pub(super) struct PackedTokenizerMetadataSegment {
    state_offset: u32,
    metadata: Arc<PackedTokenizerMetadata>,
}

impl PackedTokenizerMetadataSegment {
    #[inline]
    fn contains_state(&self, state: u32) -> bool {
        state >= self.state_offset && state - self.state_offset < self.metadata.state_count
    }

    #[inline]
    fn local_state(&self, state: u32) -> u32 {
        state - self.state_offset
    }
}

impl PackedTokenizerMetadata {
    #[inline]
    fn finalizers(&self, state: u32) -> Option<&BitSet> {
        let row = self.finalizer_row_ids.get(state as usize)?;
        self.finalizer_rows.get(row)
    }

    #[inline]
    fn finalizer_list(&self, state: u32) -> Option<&[TerminalID]> {
        let row = self.finalizer_row_ids.get(state as usize)?;
        self.finalizer_lists.get(row).map(Box::as_ref)
    }

    #[inline]
    fn futures(&self, state: u32) -> Option<&BitSet> {
        let row = self.future_row_ids.get(state as usize)?;
        self.future_rows.get(row)
    }

    #[inline]
    fn epsilon_targets(&self, state: u32) -> &[u32] {
        let Ok(index) = self.epsilon_states.binary_search(&state) else {
            return &[];
        };
        let Some(&start) = self.epsilon_offsets.get(index) else {
            return &[];
        };
        let Some(&end) = self.epsilon_offsets.get(index + 1) else {
            return &[];
        };
        self.epsilon_targets
            .get(start as usize..end as usize)
            .unwrap_or(&[])
    }

    #[inline]
    fn has_epsilon_transitions(&self) -> bool {
        !self.epsilon_states.is_empty()
    }

    fn rebased_terminals(
        &self,
        terminal_offset: TerminalID,
        total_terminals: u32,
    ) -> Arc<Self> {
        let offset = terminal_offset as usize;
        let total = total_terminals as usize;
        let rebase_bits = |bits: &BitSet| {
            let mut rebased = BitSet::new(total);
            for terminal in bits.iter() {
                let target = offset
                    .checked_add(terminal)
                    .expect("packed tokenizer terminal offset overflow");
                assert!(target < total, "rebased packed tokenizer terminal exceeds merged domain");
                rebased.set(target);
            }
            rebased
        };
        let finalizer_rows = self
            .finalizer_rows
            .iter()
            .map(rebase_bits)
            .collect::<Vec<_>>();
        let finalizer_lists = self
            .finalizer_lists
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&terminal| {
                        terminal_offset
                            .checked_add(terminal)
                            .expect("packed tokenizer terminal offset overflow")
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>();
        let future_rows = self.future_rows.iter().map(rebase_bits).collect::<Vec<_>>();
        Arc::new(Self {
            state_count: self.state_count,
            finalizer_row_ids: self.finalizer_row_ids.clone(),
            finalizer_rows: Arc::from(finalizer_rows.into_boxed_slice()),
            finalizer_lists: Arc::from(finalizer_lists.into_boxed_slice()),
            future_row_ids: self.future_row_ids.clone(),
            future_rows: Arc::from(future_rows.into_boxed_slice()),
            epsilon_states: Arc::clone(&self.epsilon_states),
            epsilon_offsets: Arc::clone(&self.epsilon_offsets),
            epsilon_targets: Arc::clone(&self.epsilon_targets),
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct PackedCompressedTransitionSegment {
    state_offset: u32,
    state_count: u32,
    byte_to_class: PackedRuntimeBytes,
    class_members: Arc<[Box<[u8]>]>,
    row_ids: PackedRowIds,
    row_offsets: Arc<[u32]>,
    classes: PackedRuntimeBytes,
    deltas: PackedI16Values,
    overflow_indices: Arc<[u32]>,
    overflow_deltas: Arc<[i32]>,
    expanded_transition_count: usize,
}

impl PackedCompressedTransitionSegment {
    /// Re-encode each class row, not its expanded byte row. TKS2 can represent
    /// separated compressed regions that the contiguous-suffix TKS3 wire
    /// deliberately declines. Packed deltas are relative to the current row;
    /// the older segment wire stores targets relative to the component base.
    fn to_compressed_segment(&self) -> CompressedTransitionSegment {
        let mut offsets = Vec::with_capacity(self.state_count as usize + 1);
        let mut classes = Vec::new();
        let mut targets = Vec::new();
        offsets.push(0);
        for local_state in 0..self.state_count {
            let state = self.state_offset + local_state;
            let (begin, end) = self.row_range(state)
                .expect("validated packed segment covers every state");
            for index in begin..end {
                let target = i64::from(local_state) + i64::from(
                    self.delta(index).expect("validated packed segment has every delta"),
                );
                assert!(target >= 0 && target < i64::from(self.state_count),
                    "packed segment target must remain in its component");
                classes.push(self.classes.as_slice()[index]);
                targets.push(target as u32);
            }
            offsets.push(u32::try_from(classes.len()).expect("compressed segment entry count exceeds u32"));
        }
        CompressedTransitionSegment {
            state_offset: self.state_offset,
            state_count: self.state_count,
            byte_to_class: Arc::from(self.byte_to_class.as_slice()),
            class_members: Arc::clone(&self.class_members),
            row_offsets: Arc::from(offsets.into_boxed_slice()),
            entries: CompressedTransitionEntries::from_parts(classes, targets),
            expanded_transition_count: self.expanded_transition_count,
        }
    }

    #[inline]
    fn contains_state(&self, state: u32) -> bool {
        state >= self.state_offset && state - self.state_offset < self.state_count
    }

    #[inline]
    fn row_range(&self, state: u32) -> Option<(usize, usize)> {
        let local = (state - self.state_offset) as usize;
        let row = self.row_ids.get(local)?;
        let start = *self.row_offsets.get(row)? as usize;
        let end = *self.row_offsets.get(row + 1)? as usize;
        Some((start, end))
    }

    #[inline]
    fn delta(&self, index: usize) -> Option<i32> {
        let delta = self.deltas.get(index)?;
        if delta != i16::MIN {
            return Some(delta as i32);
        }
        let overflow = self.overflow_indices.binary_search(&(index as u32)).ok()?;
        self.overflow_deltas.get(overflow).copied()
    }

    #[inline]
    fn transition(&self, state: u32, byte: u8) -> Option<u32> {
        let class = *self.byte_to_class.as_slice().get(byte as usize)?;
        let (start, end) = self.row_range(state)?;
        let local = self.classes.as_slice().get(start..end)?;
        let in_row = local.binary_search(&class).ok()?;
        let delta = self.delta(start + in_row)? as i64;
        let target = state as i64 + delta;
        (target >= self.state_offset as i64
            && target < (self.state_offset + self.state_count) as i64)
            .then_some(target as u32)
    }

    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) {
        row.fill(u32::MAX);
        let Some((start, end)) = self.row_range(state) else {
            return;
        };
        for index in start..end {
            let Some(&class) = self.classes.as_slice().get(index) else {
                continue;
            };
            let Some(delta) = self.delta(index) else {
                continue;
            };
            let target = (state as i64 + delta as i64) as u32;
            for &byte in self.class_members[class as usize].iter() {
                row[byte as usize] = target;
            }
        }
    }

    fn self_loop_bytes(&self, state: u32) -> U8Set {
        let mut bytes = U8Set::empty();
        let Some((start, end)) = self.row_range(state) else {
            return bytes;
        };
        for index in start..end {
            if self.delta(index) != Some(0) {
                continue;
            }
            let class = self.classes.as_slice()[index] as usize;
            for &byte in self.class_members[class].iter() {
                bytes.insert(byte);
            }
        }
        bytes
    }

    fn transition_count(&self, state: u32) -> usize {
        let Some((start, end)) = self.row_range(state) else {
            return 0;
        };
        self.classes.as_slice()[start..end]
            .iter()
            .map(|&class| self.class_members[class as usize].len())
            .sum()
    }
}

#[derive(Debug, Clone)]
enum PackedRuntimeBytes {
    Owned(Arc<[u8]>),
    Backed {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

impl PackedRuntimeBytes {
    #[inline(always)]
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(values) => values,
            Self::Backed {
                backing,
                start,
                len,
            } => &backing[*start..*start + *len],
        }
    }
}

#[derive(Debug, Clone)]
enum PackedRuntimeTargets {
    U16(Arc<[u16]>),
    U32(Arc<[u32]>),
    BackedU16 {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
    BackedU32 {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

#[derive(Clone, Copy)]
enum PackedRuntimeTargetSlice<'a> {
    U16(&'a [u16]),
    U32(&'a [u32]),
    BackedU16(&'a [u8]),
    BackedU32(&'a [u8]),
}

impl PackedRuntimeTargetSlice<'_> {
    #[inline]
    fn len(self) -> usize {
        match self {
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::BackedU16(bytes) => bytes.len() / 2,
            Self::BackedU32(bytes) => bytes.len() / 4,
        }
    }

    #[inline(always)]
    fn get(self, index: usize) -> Option<u32> {
        match self {
            Self::U16(values) => values.get(index).map(|&value| value as u32),
            Self::U32(values) => values.get(index).copied(),
            Self::BackedU16(bytes) => {
                let start = index.checked_mul(2)?;
                let pair = bytes.get(start..start + 2)?;
                Some(u16::from_le_bytes([pair[0], pair[1]]) as u32)
            }
            Self::BackedU32(bytes) => {
                let start = index.checked_mul(4)?;
                let word = bytes.get(start..start + 4)?;
                Some(u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            }
        }
    }
}

impl PackedRuntimeTransitions {
    #[inline(always)]
    fn row(&self, state: u32) -> Option<(&[u8], PackedRuntimeTargetSlice<'_>)> {
        let state = state as usize;
        let byte_start = *self.byte_offsets.get(state)? as usize;
        let byte_end = *self.byte_offsets.get(state + 1)? as usize;
        let bytes = self.bytes.as_slice().get(byte_start..byte_end)?;
        let targets = match &self.targets {
            PackedRuntimeTargets::U16(values) => {
                PackedRuntimeTargetSlice::U16(values.get(byte_start..byte_end)?)
            }
            PackedRuntimeTargets::U32(values) => {
                PackedRuntimeTargetSlice::U32(values.get(byte_start..byte_end)?)
            }
            PackedRuntimeTargets::BackedU16 {
                backing,
                start,
                len,
            } => {
                let all = backing.get(*start..*start + *len * 2)?;
                PackedRuntimeTargetSlice::BackedU16(
                    all.get(byte_start * 2..byte_end * 2)?,
                )
            }
            PackedRuntimeTargets::BackedU32 {
                backing,
                start,
                len,
            } => {
                let all = backing.get(*start..*start + *len * 4)?;
                PackedRuntimeTargetSlice::BackedU32(
                    all.get(byte_start * 4..byte_end * 4)?,
                )
            }
        };
        (bytes.len() == targets.len()).then_some((bytes, targets))
    }

    #[inline(always)]
    fn transition(&self, state: u32, byte: u8) -> Option<u32> {
        let (bytes, targets) = self.row(state)?;
        // Dense regex rows are frequently one contiguous byte interval. Avoid
        // a ~7-step binary search on every member of a lazy union when the
        // sorted row itself proves that direct indexing is exact.
        if bytes.len() >= 32 {
            let first = *bytes.first()?;
            let last = *bytes.last()?;
            if usize::from(last.wrapping_sub(first)) + 1 == bytes.len() {
                if byte < first || byte > last {
                    return None;
                }
                return targets.get(usize::from(byte - first));
            }
        }
        let index = bytes.binary_search(&byte).ok()?;
        targets.get(index)
    }

    #[inline]
    fn state_count(&self) -> usize {
        self.byte_offsets.len().saturating_sub(1)
    }
}

pub struct FullTokenizerDeterminization {
    pub tokenizer: Tokenizer,
    /// Exact epsilon-closed source-state subset represented by each new state.
    pub source_subsets: Vec<Box<[u32]>>,
    /// First state of an appended exact copy of the source tokenizer.  The
    /// copy is a correctness fallback for parser histories that cease to be
    /// uniform across one determinized subset.
    pub source_state_offset: u32,
    /// A source state whose singleton epsilon closure is exactly the product
    /// subset, or `u32::MAX` when no such scalar representative exists.
    pub exact_source_states: Vec<u32>,
}

/// Exact deterministic transition rows over a byte-equivalence-class alphabet.
/// Targets and row coordinates are local to one DFA component; `state_offset`
/// rebases both into the final partitioned runtime tokenizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedTransitionSegment {
    pub state_offset: u32,
    pub state_count: u32,
    pub byte_to_class: Arc<[u8]>,
    pub class_members: Arc<[Box<[u8]>]>,
    pub row_offsets: Arc<[u32]>,
    pub entries: CompressedTransitionEntries,
    pub expanded_transition_count: usize,
}

/// Structure-of-arrays storage for compressed transition entries. Rust pads a
/// `(u8, u32)` tuple to eight bytes; keeping the fields separately uses five
/// bytes per entry while retaining the historical serialized sequence of
/// `(class, target)` pairs.
#[derive(Debug, Clone, Default)]
pub struct CompressedTransitionEntries {
    classes: Arc<[u8]>,
    targets: Arc<[u32]>,
}

impl CompressedTransitionEntries {
    pub fn from_parts(classes: Vec<u8>, targets: Vec<u32>) -> Self {
        assert_eq!(classes.len(), targets.len());
        Self {
            classes: Arc::from(classes.into_boxed_slice()),
            targets: Arc::from(targets.into_boxed_slice()),
        }
    }

    #[inline]
    fn class_slice(&self, start: usize, end: usize) -> &[u8] {
        &self.classes[start..end]
    }

    #[inline]
    fn target(&self, index: usize) -> u32 {
        self.targets[index]
    }

    #[inline]
    pub(super) fn iter_range(
        &self,
        start: usize,
        end: usize,
    ) -> impl Iterator<Item = (u8, u32)> + '_ {
        self.classes[start..end]
            .iter()
            .copied()
            .zip(self.targets[start..end].iter().copied())
    }

    #[inline]
    fn len(&self) -> usize {
        self.classes.len()
    }
}

impl Serialize for CompressedTransitionEntries {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.len()))?;
        for entry in self
            .classes
            .iter()
            .copied()
            .zip(self.targets.iter().copied())
        {
            sequence.serialize_element(&entry)?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for CompressedTransitionEntries {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let entries = Vec::<(u8, u32)>::deserialize(deserializer)?;
        let mut classes = Vec::with_capacity(entries.len());
        let mut targets = Vec::with_capacity(entries.len());
        for (class, target) in entries {
            classes.push(class);
            targets.push(target);
        }
        Ok(Self::from_parts(classes, targets))
    }
}

pub mod artifact_serde {
    use super::*;

    const FAST_WIRE_FLAG_SCALAR_DETERMINISTIC_DISPATCH: u8 = 1 << 0;
    const FAST_WIRE_KNOWN_FLAGS: u8 = FAST_WIRE_FLAG_SCALAR_DETERMINISTIC_DISPATCH;
    pub(super) const FAST_WIRE_FLAGS_OFFSET: usize = 30;

    #[inline]
    fn u16_values_all_below(bytes: &[u8], limit: usize) -> bool {
        debug_assert_eq!(bytes.len() % 2, 0);
        debug_assert!(limit <= u16::MAX as usize + 1);
        #[cfg(target_arch = "x86_64")]
        {
            // x86_64 guarantees SSE2. TKF2 target slabs are not necessarily
            // aligned (the preceding byte-transition slab has arbitrary
            // length), so use unaligned vector loads. Convert the unsigned
            // comparison to a signed one by flipping the sign bit, and test
            // `value > limit - 1` eight targets at a time.
            if limit != 0 && limit <= u16::MAX as usize {
                unsafe {
                    use std::arch::x86_64::{
                        __m128i, _mm_cmpgt_epi16, _mm_loadu_si128, _mm_movemask_epi8,
                        _mm_set1_epi16, _mm_xor_si128,
                    };
                    let sign = _mm_set1_epi16(i16::MIN);
                    let threshold = _mm_xor_si128(
                        _mm_set1_epi16((limit as u16 - 1) as i16),
                        sign,
                    );
                    let mut pos = 0usize;
                    while pos + 16 <= bytes.len() {
                        let values = _mm_loadu_si128(bytes.as_ptr().add(pos).cast::<__m128i>());
                        let unsigned_ordered = _mm_xor_si128(values, sign);
                        let invalid = _mm_cmpgt_epi16(unsigned_ordered, threshold);
                        if _mm_movemask_epi8(invalid) != 0 {
                            return false;
                        }
                        pos += 16;
                    }
                    return bytes[pos..].chunks_exact(2).all(|word| {
                        (u16::from_le_bytes([word[0], word[1]]) as usize) < limit
                    });
                }
            }
        }
        bytes
            .chunks_exact(2)
            .all(|word| (u16::from_le_bytes([word[0], word[1]]) as usize) < limit)
    }
    use serde::{Deserializer, Serializer};

    #[derive(Serialize)]
    struct TokenizerArtifactRef<'a> {
        dfa: &'a DFA,
        num_terminals: u32,
        compressed_transition_segments: &'a [CompressedTransitionSegment],
    }

    #[derive(Deserialize)]
    struct TokenizerArtifact {
        dfa: DFA,
        num_terminals: u32,
        compressed_transition_segments: Vec<CompressedTransitionSegment>,
    }

    pub fn serialize<S>(tokenizer: &Tokenizer, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if external_artifact_serde_enabled() {
            return 0u8.serialize(serializer);
        }
        if compact_artifact_serde_enabled() {
            return packed_artifact_serde::serialize(tokenizer, serializer);
        }
        TokenizerArtifactRef {
            dfa: &tokenizer.dfa,
            num_terminals: tokenizer.num_terminals,
            compressed_transition_segments: &tokenizer.compressed_transition_segments,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Tokenizer, D::Error>
    where
        D: Deserializer<'de>,
    {
        if external_artifact_serde_enabled() {
            let marker = u8::deserialize(deserializer)?;
            if marker != 0 {
                return Err(serde::de::Error::custom(
                    "invalid external tokenizer placeholder",
                ));
            }
            return Ok(Tokenizer {
                dfa: DFA::new(1),
                num_terminals: 0,
                packed_runtime_transitions: None,
                packed_runtime_transition_segments: Arc::from([]),
                compressed_transition_segments: Arc::from([]),
                packed_runtime_metadata: None,
                packed_runtime_metadata_segments: Arc::from([]),
                packed_compressed_transition_segments: Arc::from([]),
                virtual_unit_repeat: None,
                virtual_repeat_intersections: Vec::new(),
                virtual_residuals: Vec::new(),
                exprs: None,
                terminal_residual_coordinates: None,
                singleton_epsilon_closures: OnceLock::new(),
                matched_terminals_cache: OnceLock::new(),
                initial_byte_frontiers: OnceLock::new(),
                all_self_loop_bytes_cache: OnceLock::new(),
                transition_count_cache: OnceLock::new(),
                forced_minimized_state_count_cache: OnceLock::new(),
                scalar_deterministic_dispatch_cache: OnceLock::new(),
                sorted_dispatch_roots_cache: OnceLock::new(),
                state_first_bytes_cache: OnceLock::new(),
            });
        }
        if compact_artifact_serde_enabled() {
            return packed_artifact_serde::deserialize(deserializer);
        }
        let artifact = TokenizerArtifact::deserialize(deserializer)?;
        Ok(Tokenizer {
            dfa: artifact.dfa,
            num_terminals: artifact.num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from(
                artifact.compressed_transition_segments.into_boxed_slice(),
            ),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: None,
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
            sorted_dispatch_roots_cache: OnceLock::new(),
            state_first_bytes_cache: OnceLock::new(),
        })
    }

    #[derive(Clone, Copy)]
    #[doc(hidden)]
    pub struct FastLayout {
        state_count: usize,
        transition_count: usize,
        epsilon_count: usize,
        finalizer_count: usize,
        future_count: usize,
        terminal_count: usize,
        state_id_width: usize,
        terminal_id_width: usize,
        len: usize,
    }

    impl FastLayout {
        #[inline]
        pub fn len(self) -> usize {
            self.len
        }
    }

    fn fast_layout(tokenizer: &Tokenizer, dfa: &DFA) -> FastLayout {
        const HEADER_LEN: usize = 32;
        let states = dfa.states();
        let state_count = states.len();
        let transition_count = states.iter().map(|state| state.transitions.len()).sum::<usize>();
        let epsilon_count = states.iter().map(|state| state.epsilon_transitions.len()).sum::<usize>();
        let finalizer_count = states
            .iter()
            .map(|state| state.finalizers.iter().count())
            .sum::<usize>();
        let future_count = states
            .iter()
            .map(|state| state.possible_future_group_ids.iter().count())
            .sum::<usize>();
        let terminal_count = tokenizer.num_terminals as usize;
        let state_id_width = if state_count <= u16::MAX as usize + 1 {
            2usize
        } else {
            4usize
        };
        let terminal_id_width = if terminal_count <= u16::MAX as usize + 1 {
            2usize
        } else {
            4usize
        };
        let offsets_bytes = (state_count + 1) * 4;
        let len = HEADER_LEN
            + terminal_count * 32
            + offsets_bytes + transition_count
            + transition_count * state_id_width
            + offsets_bytes + epsilon_count * state_id_width
            + offsets_bytes + finalizer_count * terminal_id_width
            + offsets_bytes + future_count * terminal_id_width;
        FastLayout {
            state_count,
            transition_count,
            epsilon_count,
            finalizer_count,
            future_count,
            terminal_count,
            state_id_width,
            terminal_id_width,
            len,
        }
    }

    /// Estimated eventual TKF2 wire length without materializing compressed
    /// transition segments. The estimate is exact because compressed segments
    /// retain their expanded transition counts while all state metadata stays
    /// resident in the DFA.
    pub fn estimated_fast_len(tokenizer: &Tokenizer) -> usize {
        const HEADER_LEN: usize = 32;
        let states = tokenizer.dfa.states();
        let state_count = states.len();
        let transition_count = tokenizer.transition_count();
        let epsilon_count = states
            .iter()
            .map(|state| state.epsilon_transitions.len())
            .sum::<usize>();
        let finalizer_count = states
            .iter()
            .map(|state| state.finalizers.iter().count())
            .sum::<usize>();
        let future_count = states
            .iter()
            .map(|state| state.possible_future_group_ids.iter().count())
            .sum::<usize>();
        let terminal_count = tokenizer.num_terminals as usize;
        let state_id_width = if state_count <= u16::MAX as usize + 1 { 2 } else { 4 };
        let terminal_id_width = if terminal_count <= u16::MAX as usize + 1 { 2 } else { 4 };
        let offsets_bytes = (state_count + 1) * 4;
        HEADER_LEN
            + terminal_count * 32
            + offsets_bytes
            + transition_count
            + transition_count * state_id_width
            + offsets_bytes
            + epsilon_count * state_id_width
            + offsets_bytes
            + finalizer_count * terminal_id_width
            + offsets_bytes
            + future_count * terminal_id_width
    }

    /// Exact TKF2 wire length without constructing the wire buffer. Returns
    /// `None` only for the uncommon compressed-segment representation, whose
    /// direct writer requires materialized transition rows.
    pub fn fast_layout_for_write(tokenizer: &Tokenizer) -> Option<FastLayout> {
        if !tokenizer.compressed_transition_segments.is_empty()
            || !tokenizer.packed_compressed_transition_segments.is_empty()
        {
            return None;
        }
        if tokenizer.packed_runtime_transitions.is_some()
            || !tokenizer.packed_runtime_transition_segments.is_empty()
        {
            let materialized = tokenizer.materialized_dfa();
            return Some(fast_layout(tokenizer, &materialized));
        }
        Some(fast_layout(tokenizer, &tokenizer.dfa))
    }

    pub fn fast_len(tokenizer: &Tokenizer) -> Option<usize> {
        fast_layout_for_write(tokenizer).map(FastLayout::len)
    }

    #[inline]
    fn fast_transition_prefix_len(layout: FastLayout) -> usize {
        32usize
            + layout.terminal_count * 32
            + (layout.state_count + 1) * 4
            + layout.transition_count
            + layout.transition_count * layout.state_id_width
    }

    fn write_fast_bytes_for_dfa(
        tokenizer: &Tokenizer,
        dfa: &DFA,
        layout: FastLayout,
        out: &mut [u8],
        transition_prefix_only: bool,
    ) -> Result<(), String> {
        let expected_len = if transition_prefix_only {
            fast_transition_prefix_len(layout)
        } else {
            layout.len
        };
        if out.len() != expected_len {
            return Err(format!(
                "fast tokenizer output has length {}, expected {}",
                out.len(), expected_len
            ));
        }
        let states = dfa.states();
        let mut pos = 0usize;
        let put = |out: &mut [u8], pos: &mut usize, bytes: &[u8]| {
            let end = *pos + bytes.len();
            out[*pos..end].copy_from_slice(bytes);
            *pos = end;
        };
        let put_u32 = |out: &mut [u8], pos: &mut usize, value: u32| {
            put(out, pos, &value.to_le_bytes());
        };
        let put_id = |out: &mut [u8], pos: &mut usize, value: u32, width: usize| match width {
            2 => put(out, pos, &(value as u16).to_le_bytes()),
            4 => put(out, pos, &value.to_le_bytes()),
            _ => unreachable!(),
        };
        let put_offsets =
            |out: &mut [u8], pos: &mut usize, lengths: &mut dyn Iterator<Item = usize>| {
                let mut end = 0u32;
                put_u32(out, pos, end);
                for len in lengths {
                    end = end.saturating_add(len as u32);
                    put_u32(out, pos, end);
                }
            };

        put(out, &mut pos, b"TKF2");
        for value in [
            tokenizer.num_terminals,
            layout.state_count as u32,
            layout.transition_count as u32,
            layout.epsilon_count as u32,
            layout.finalizer_count as u32,
            layout.future_count as u32,
        ] {
            put_u32(out, &mut pos, value);
        }
        put(
            out,
            &mut pos,
            &[layout.state_id_width as u8, layout.terminal_id_width as u8, 0, 0],
        );
        for terminal in 0..layout.terminal_count {
            for word in dfa.group_id_to_u8set(terminal as u32).to_words() {
                put(out, &mut pos, &word.to_le_bytes());
            }
        }
        let packed_transitions = tokenizer
            .packed_runtime_transitions
            .as_deref()
            .filter(|packed| {
                packed.state_count() == states.len()
                    && packed.bytes.as_slice().len() == layout.transition_count
            });
        if let Some(packed) = packed_transitions {
            for &offset in packed.byte_offsets.iter() {
                put_u32(out, &mut pos, offset);
            }
            put(out, &mut pos, packed.bytes.as_slice());
            match (&packed.targets, layout.state_id_width) {
                (PackedRuntimeTargets::U16(values), 2) if cfg!(target_endian = "little") => {
                    // SAFETY: u16 has no padding, the source allocation remains
                    // alive for the copy, and TKF2 is explicitly little-endian.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            values.as_ptr().cast::<u8>(),
                            values.len() * std::mem::size_of::<u16>(),
                        )
                    };
                    put(out, &mut pos, bytes);
                }
                (PackedRuntimeTargets::U32(values), 4) if cfg!(target_endian = "little") => {
                    // SAFETY: u32 has no padding; see the U16 branch above.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            values.as_ptr().cast::<u8>(),
                            values.len() * std::mem::size_of::<u32>(),
                        )
                    };
                    put(out, &mut pos, bytes);
                }
                (
                    PackedRuntimeTargets::BackedU16 {
                        backing,
                        start,
                        len,
                    },
                    2,
                ) => put(out, &mut pos, &backing[*start..*start + *len * 2]),
                (
                    PackedRuntimeTargets::BackedU32 {
                        backing,
                        start,
                        len,
                    },
                    4,
                ) => put(out, &mut pos, &backing[*start..*start + *len * 4]),
                _ => {
                    for state in 0..states.len() as u32 {
                        let (_, targets) = packed
                            .row(state)
                            .expect("packed runtime transition rows must cover every state");
                        for index in 0..targets.len() {
                            put_id(
                                out,
                                &mut pos,
                                targets
                                    .get(index)
                                    .expect("packed runtime target row length was validated"),
                                layout.state_id_width,
                            );
                        }
                    }
                }
            }
        } else {
            put_offsets(
                out,
                &mut pos,
                &mut states.iter().map(|state| state.transitions.len()),
            );
            for state in states {
                for (byte, _) in state.transitions.iter() {
                    out[pos] = byte;
                    pos += 1;
                }
            }
            for state in states {
                for &target in state.transitions.values() {
                    debug_assert!(layout.state_id_width == 4 || target <= u16::MAX as u32);
                    put_id(out, &mut pos, target, layout.state_id_width);
                }
            }
        }
        if transition_prefix_only {
            debug_assert_eq!(pos, out.len());
            return Ok(());
        }
        put_offsets(
            out,
            &mut pos,
            &mut states.iter().map(|state| state.epsilon_transitions.len()),
        );
        for state in states {
            for &target in &state.epsilon_transitions {
                debug_assert!(layout.state_id_width == 4 || target <= u16::MAX as u32);
                put_id(out, &mut pos, target, layout.state_id_width);
            }
        }
        put_offsets(
            out,
            &mut pos,
            &mut states.iter().map(|state| state.finalizers.iter().count()),
        );
        for state in states {
            for terminal in state.finalizers.iter() {
                let terminal = terminal as u32;
                debug_assert!(layout.terminal_id_width == 4 || terminal <= u16::MAX as u32);
                put_id(out, &mut pos, terminal, layout.terminal_id_width);
            }
        }
        put_offsets(
            out,
            &mut pos,
            &mut states
                .iter()
                .map(|state| state.possible_future_group_ids.iter().count()),
        );
        for state in states {
            for terminal in state.possible_future_group_ids.iter() {
                let terminal = terminal as u32;
                debug_assert!(layout.terminal_id_width == 4 || terminal <= u16::MAX as u32);
                put_id(out, &mut pos, terminal, layout.terminal_id_width);
            }
        }
        debug_assert_eq!(pos, out.len());
        Ok(())
    }

    /// Write TKF2 directly into an exactly-sized destination. Used by the
    /// constraint serializer to overlap tokenizer encoding with independent
    /// final-section copies.
    pub fn write_fast_bytes(tokenizer: &Tokenizer, out: &mut [u8]) -> Result<(), String> {
        if !tokenizer.compressed_transition_segments.is_empty() {
            return Err("direct fast tokenizer write requires materialized transitions".to_owned());
        }
        let layout = fast_layout(tokenizer, &tokenizer.dfa);
        write_fast_bytes_for_dfa(tokenizer, &tokenizer.dfa, layout, out, false)
    }

    /// Write TKF2 using an exact layout already computed by the caller. This
    /// avoids rescanning every DFA state after `fast_layout_for_write()` was
    /// used to size the final artifact section.
    pub fn write_fast_bytes_with_layout(
        tokenizer: &Tokenizer,
        layout: FastLayout,
        out: &mut [u8],
    ) -> Result<(), String> {
        if !tokenizer.compressed_transition_segments.is_empty()
            || !tokenizer.packed_compressed_transition_segments.is_empty()
        {
            return Err("direct fast tokenizer write requires materialized transitions".to_owned());
        }
        if tokenizer.packed_runtime_transitions.is_some()
            || !tokenizer.packed_runtime_transition_segments.is_empty()
        {
            let materialized = tokenizer.materialized_dfa();
            return write_fast_bytes_for_dfa(tokenizer, &materialized, layout, out, false);
        }
        write_fast_bytes_for_dfa(tokenizer, &tokenizer.dfa, layout, out, false)
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    struct FastPackedMetadataArtifact {
        finalizer_rows: Vec<Box<[u32]>>,
        finalizer_row_ids: Vec<u32>,
        future_rows: Vec<Box<[u32]>>,
        future_row_ids: Vec<u32>,
        epsilon_states: Vec<u32>,
        epsilon_offsets: Vec<u32>,
        epsilon_targets: Vec<u32>,
    }

    fn fast_packed_metadata_artifact(dfa: &DFA) -> FastPackedMetadataArtifact {
        let states = dfa.states();
        let pack_rows = |futures: bool| {
            let mut row_map = FxHashMap::<Box<[u32]>, u32>::default();
            let mut rows = Vec::<Box<[u32]>>::new();
            let mut row_ids = Vec::<u32>::with_capacity(states.len());
            for state in states {
                let sparse = if futures {
                    state
                        .possible_future_group_ids
                        .iter()
                        .map(|terminal| terminal as u32)
                        .collect::<Vec<_>>()
                } else {
                    state
                        .finalizers
                        .iter()
                        .map(|terminal| terminal as u32)
                        .collect::<Vec<_>>()
                };
                let row = if let Some(&row) = row_map.get(sparse.as_slice()) {
                    row
                } else {
                    let row = rows.len() as u32;
                    let boxed = sparse.into_boxed_slice();
                    row_map.insert(boxed.clone(), row);
                    rows.push(boxed);
                    row
                };
                row_ids.push(row);
            }
            (rows, row_ids)
        };
        let (finalizer_rows, finalizer_row_ids) = pack_rows(false);
        let (future_rows, future_row_ids) = pack_rows(true);
        let mut epsilon_states = Vec::new();
        let mut epsilon_offsets = vec![0u32];
        let mut epsilon_targets = Vec::new();
        for (state_index, state) in states.iter().enumerate() {
            if state.epsilon_transitions.is_empty() {
                continue;
            }
            epsilon_states.push(state_index as u32);
            epsilon_targets.extend_from_slice(&state.epsilon_transitions);
            epsilon_offsets.push(epsilon_targets.len() as u32);
        }
        FastPackedMetadataArtifact {
            finalizer_rows,
            finalizer_row_ids,
            future_rows,
            future_row_ids,
            epsilon_states,
            epsilon_offsets,
            epsilon_targets,
        }
    }

    /// TKF3 keeps TKF2's directly-backed transition arrays, but persists the
    /// already-interned state metadata rows used by the loaded runtime. This is
    /// used for giant finite residual mask tokenizers where rebuilding those
    /// rows on every load is measurable.
    pub fn to_fast_bytes_with_packed_metadata(tokenizer: &Tokenizer) -> Vec<u8> {
        let materialized;
        let dfa = if tokenizer.compressed_transition_segments.is_empty()
            && tokenizer.packed_compressed_transition_segments.is_empty()
            && tokenizer.packed_runtime_transitions.is_none()
            && tokenizer.packed_runtime_transition_segments.is_empty()
        {
            &tokenizer.dfa
        } else {
            materialized = tokenizer.materialized_dfa();
            &materialized
        };
        let layout = fast_layout(tokenizer, dfa);
        let transition_end = fast_transition_prefix_len(layout);
        let mut out = vec![0u8; transition_end];
        write_fast_bytes_for_dfa(tokenizer, dfa, layout, &mut out, true)
            .expect("exact TKF3 transition-prefix layout should always write successfully");
        out[..4].copy_from_slice(b"TKF3");
        if tokenizer.scalar_deterministic_dispatch_cache.get() == Some(&true) {
            out[FAST_WIRE_FLAGS_OFFSET] |= FAST_WIRE_FLAG_SCALAR_DETERMINISTIC_DISPATCH;
        }
        let metadata = fast_packed_metadata_artifact(dfa);
        bincode::serialize_into(&mut out, &metadata)
            .expect("TKF3 packed metadata serialization should succeed");
        out
    }

    /// Add the already-proven scalar-dispatch certificate to an ordinary TKF3
    /// wire after its expensive payload has been serialized. This lets callers
    /// overlap the proof with serialization without making the serializer wait
    /// for it.
    #[doc(hidden)]
    pub fn mark_fast_wire_scalar_deterministic_dispatch(out: &mut [u8]) {
        assert!(
            out.len() > FAST_WIRE_FLAGS_OFFSET && out.starts_with(b"TKF3"),
            "scalar-dispatch fast-wire flag requires a TKF3 payload",
        );
        out[FAST_WIRE_FLAGS_OFFSET] |= FAST_WIRE_FLAG_SCALAR_DETERMINISTIC_DISPATCH;
    }

    /// Runtime-native current-format tokenizer wire. Unlike the older packed
    /// serializer this does no row hashing and no per-row varint target
    /// encoding: the compiler already owns the exact DFA, so save is a linear
    /// fixed-width copy and load reconstructs the existing packed runtime
    /// transition sidecar directly.
    pub fn to_fast_bytes(tokenizer: &Tokenizer) -> Vec<u8> {
        let materialized;
        let dfa = if tokenizer.compressed_transition_segments.is_empty()
            && tokenizer.packed_compressed_transition_segments.is_empty()
            && tokenizer.packed_runtime_transitions.is_none()
            && tokenizer.packed_runtime_transition_segments.is_empty()
        {
            &tokenizer.dfa
        } else {
            materialized = tokenizer.materialized_dfa();
            &materialized
        };
        let layout = fast_layout(tokenizer, dfa);
        let len = layout.len;
        let mut out = Vec::<u8>::with_capacity(len);
        unsafe {
            out.set_len(len);
        }
        write_fast_bytes_for_dfa(tokenizer, dfa, layout, &mut out, false)
            .expect("exact fast tokenizer layout should always write successfully");
        out
    }

    const PACKED_WIRE_MAGIC: &[u8; 4] = b"TKP1";
    const SEGMENT_WIRE_MAGIC: &[u8; 4] = b"TKS2";

    struct PackedWireRef<'a>(&'a Tokenizer);

    impl Serialize for PackedWireRef<'_> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            packed_artifact_serde::serialize(self.0, serializer)
        }
    }

    struct PackedWireOwned(Tokenizer);

    impl<'de> Deserialize<'de> for PackedWireOwned {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            packed_artifact_serde::deserialize(deserializer).map(Self)
        }
    }

    /// Compact current tokenizer wire for exceptionally large DFAs. The
    /// historical packed codec interns repeated byte rows and varint-encodes
    /// targets, avoiding TKF2's fixed-width expansion when state ids require
    /// four bytes.
    pub fn to_packed_bytes(tokenizer: &Tokenizer) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(PACKED_WIRE_MAGIC);
        bincode::serialize_into(&mut out, &PackedWireRef(tokenizer))
            .expect("packed tokenizer serialization should succeed");
        out
    }

    fn from_packed_bytes(input: &[u8]) -> Result<Tokenizer, String> {
        let body = input
            .strip_prefix(PACKED_WIRE_MAGIC)
            .ok_or_else(|| "invalid packed tokenizer header".to_owned())?;
        bincode::deserialize::<PackedWireOwned>(body)
            .map(|wire| wire.0)
            .map_err(|err| err.to_string())
    }

    /// Persist an already-compressed tokenizer without expanding its byte-class
    /// transition segments. State metadata uses the same flat CSR shape as the
    /// fast tokenizer wire, while only transition rows outside compressed
    /// segments are stored explicitly.
    pub fn to_segment_bytes(tokenizer: &Tokenizer) -> Vec<u8> {
        // Runtime compaction removes owned compressed rows. If those packed
        // regions are not one contiguous suffix, TKS3 declines and this is the
        // authoritative fallback. Reading only `dfa`/owned segments here would
        // silently save empty byte rows while retaining live finalizer/future
        // metadata. Materialize only metadata and noncompressed rows, retaining
        // compressed regions as class rows rather than expanding their bytes.
        if !tokenizer.packed_compressed_transition_segments.is_empty()
            || tokenizer.packed_runtime_transitions.is_some()
            || !tokenizer.packed_runtime_transition_segments.is_empty()
            || tokenizer.packed_runtime_metadata.is_some()
            || !tokenizer.packed_runtime_metadata_segments.is_empty()
        {
            let mut segments = tokenizer.compressed_transition_segments.to_vec();
            segments.extend(tokenizer.packed_compressed_transition_segments.iter()
                .map(PackedCompressedTransitionSegment::to_compressed_segment));
            segments.sort_unstable_by_key(|segment| segment.state_offset);
            let physical = Tokenizer::from_parts_with_compressed_transitions(
                tokenizer.materialized_noncompressed_dfa(),
                tokenizer.num_terminals,
                None,
                segments,
            );
            return to_segment_bytes(&physical);
        }
        const HEADER_LEN: usize = 36;
        let states = tokenizer.dfa.states();
        let state_count = states.len();
        let state_id_width = if state_count <= u16::MAX as usize + 1 { 2 } else { 4 };
        let terminal_count = tokenizer.num_terminals as usize;
        let terminal_id_width = if terminal_count <= u16::MAX as usize + 1 { 2 } else { 4 };

        let mut compressed = vec![false; state_count];
        for segment in tokenizer.compressed_transition_segments.iter() {
            let start = segment.state_offset as usize;
            let end = start.saturating_add(segment.state_count as usize).min(state_count);
            compressed[start..end].fill(true);
        }
        let residual_transition_count = states
            .iter()
            .enumerate()
            .filter(|(state, _)| !compressed[*state])
            .map(|(_, state)| state.transitions.len())
            .sum::<usize>();
        let epsilon_count = states.iter().map(|state| state.epsilon_transitions.len()).sum::<usize>();
        let finalizer_count = states
            .iter()
            .map(|state| state.finalizers.iter().count())
            .sum::<usize>();
        let future_count = states
            .iter()
            .map(|state| state.possible_future_group_ids.iter().count())
            .sum::<usize>();
        let segment_blob = bincode::serialize(tokenizer.compressed_transition_segments.as_ref())
            .expect("compressed tokenizer segments should serialize");
        let offsets_bytes = (state_count + 1) * 4;
        let len = HEADER_LEN
            + terminal_count * 32
            + offsets_bytes
            + residual_transition_count
            + residual_transition_count * state_id_width
            + offsets_bytes
            + epsilon_count * state_id_width
            + offsets_bytes
            + finalizer_count * terminal_id_width
            + offsets_bytes
            + future_count * terminal_id_width
            + segment_blob.len();
        let mut out = Vec::with_capacity(len);
        out.extend_from_slice(SEGMENT_WIRE_MAGIC);
        for value in [
            tokenizer.num_terminals,
            state_count as u32,
            residual_transition_count as u32,
            epsilon_count as u32,
            finalizer_count as u32,
            future_count as u32,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&(segment_blob.len() as u64).to_le_bytes());
        debug_assert_eq!(out.len(), HEADER_LEN);
        for terminal in 0..terminal_count {
            for word in tokenizer.dfa.group_id_to_u8set(terminal as u32).to_words() {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        let put_id = |out: &mut Vec<u8>, value: u32, width: usize| match width {
            2 => out.extend_from_slice(&(value as u16).to_le_bytes()),
            4 => out.extend_from_slice(&value.to_le_bytes()),
            _ => unreachable!(),
        };
        let mut end = 0u32;
        out.extend_from_slice(&end.to_le_bytes());
        for (index, state) in states.iter().enumerate() {
            if !compressed[index] {
                end += state.transitions.len() as u32;
            }
            out.extend_from_slice(&end.to_le_bytes());
        }
        for (index, state) in states.iter().enumerate() {
            if !compressed[index] {
                out.extend(state.transitions.iter().map(|(byte, _)| byte));
            }
        }
        for (index, state) in states.iter().enumerate() {
            if !compressed[index] {
                for &target in state.transitions.values() {
                    put_id(&mut out, target, state_id_width);
                }
            }
        }
        let mut put_sparse = |counts: &mut dyn Iterator<Item = usize>, ids: &mut dyn Iterator<Item = u32>, width: usize| {
            let mut end = 0u32;
            out.extend_from_slice(&end.to_le_bytes());
            for count in counts {
                end += count as u32;
                out.extend_from_slice(&end.to_le_bytes());
            }
            for id in ids {
                put_id(&mut out, id, width);
            }
        };
        put_sparse(
            &mut states.iter().map(|state| state.epsilon_transitions.len()),
            &mut states.iter().flat_map(|state| state.epsilon_transitions.iter().copied()),
            state_id_width,
        );
        put_sparse(
            &mut states.iter().map(|state| state.finalizers.iter().count()),
            &mut states.iter().flat_map(|state| state.finalizers.iter().map(|id| id as u32)),
            terminal_id_width,
        );
        put_sparse(
            &mut states.iter().map(|state| state.possible_future_group_ids.iter().count()),
            &mut states
                .iter()
                .flat_map(|state| state.possible_future_group_ids.iter().map(|id| id as u32)),
            terminal_id_width,
        );
        out.extend_from_slice(&segment_blob);
        debug_assert_eq!(out.len(), len);
        out
    }

    fn from_segment_bytes(input: &[u8]) -> Result<Tokenizer, String> {
        const HEADER_LEN: usize = 36;
        if input.len() < HEADER_LEN || !input.starts_with(SEGMENT_WIRE_MAGIC) {
            return Err("invalid compressed tokenizer header".to_owned());
        }
        let mut pos = 4usize;
        let take_u32 = |input: &[u8], pos: &mut usize| -> Result<u32, String> {
            let end = pos.checked_add(4).ok_or_else(|| "compressed tokenizer offset overflow".to_owned())?;
            let bytes = input.get(*pos..end).ok_or_else(|| "truncated compressed tokenizer".to_owned())?;
            *pos = end;
            Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
        };
        let num_terminals = take_u32(input, &mut pos)?;
        let state_count = take_u32(input, &mut pos)? as usize;
        let residual_transition_count = take_u32(input, &mut pos)? as usize;
        let epsilon_count = take_u32(input, &mut pos)? as usize;
        let finalizer_count = take_u32(input, &mut pos)? as usize;
        let future_count = take_u32(input, &mut pos)? as usize;
        let segment_len_end = pos + 8;
        let segment_blob_len = u64::from_le_bytes(
            input.get(pos..segment_len_end)
                .ok_or_else(|| "truncated compressed tokenizer segment length".to_owned())?
                .try_into().unwrap(),
        ) as usize;
        pos = segment_len_end;
        if state_count == 0 {
            return Err("compressed tokenizer has no states".to_owned());
        }
        let state_id_width = if state_count <= u16::MAX as usize + 1 { 2 } else { 4 };
        let terminal_count = num_terminals as usize;
        let terminal_id_width = if terminal_count <= u16::MAX as usize + 1 { 2 } else { 4 };
        let mut group_id_to_u8set = Vec::with_capacity(terminal_count);
        for _ in 0..terminal_count {
            let mut words = [0u64; 4];
            for word in &mut words {
                let end = pos + 8;
                *word = u64::from_le_bytes(
                    input.get(pos..end)
                        .ok_or_else(|| "truncated compressed tokenizer groups".to_owned())?
                        .try_into().unwrap(),
                );
                pos = end;
            }
            group_id_to_u8set.push(U8Set::from_words(words));
        }
        let read_u32_vec = |input: &[u8], pos: &mut usize, count: usize| -> Result<Vec<u32>, String> {
            let bytes_len = count.checked_mul(4).ok_or_else(|| "compressed tokenizer vector overflow".to_owned())?;
            let end = pos.checked_add(bytes_len).ok_or_else(|| "compressed tokenizer offset overflow".to_owned())?;
            let bytes = input.get(*pos..end).ok_or_else(|| "truncated compressed tokenizer vector".to_owned())?;
            let mut out = Vec::with_capacity(count);
            out.extend(bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())));
            *pos = end;
            Ok(out)
        };
        let read_ids = |input: &[u8], pos: &mut usize, count: usize, width: usize| -> Result<Vec<u32>, String> {
            let bytes_len = count.checked_mul(width).ok_or_else(|| "compressed tokenizer id vector overflow".to_owned())?;
            let end = pos.checked_add(bytes_len).ok_or_else(|| "compressed tokenizer offset overflow".to_owned())?;
            let bytes = input.get(*pos..end).ok_or_else(|| "truncated compressed tokenizer ids".to_owned())?;
            let out = match width {
                2 => bytes.chunks_exact(2).map(|b| u16::from_le_bytes(b.try_into().unwrap()) as u32).collect(),
                4 => bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect(),
                _ => unreachable!(),
            };
            *pos = end;
            Ok(out)
        };
        let validate_offsets = |offsets: &[u32], count: usize, label: &str| -> Result<(), String> {
            if offsets.len() != state_count + 1
                || offsets.first().copied() != Some(0)
                || offsets.last().copied() != Some(count as u32)
                || offsets.windows(2).any(|pair| pair[0] > pair[1])
            {
                return Err(format!("invalid compressed tokenizer {label} offsets"));
            }
            Ok(())
        };
        let transition_offsets = read_u32_vec(input, &mut pos, state_count + 1)?;
        validate_offsets(&transition_offsets, residual_transition_count, "transition")?;
        let transition_bytes_end = pos + residual_transition_count;
        let transition_bytes = input.get(pos..transition_bytes_end)
            .ok_or_else(|| "truncated compressed tokenizer transition bytes".to_owned())?;
        pos = transition_bytes_end;
        let transition_targets = read_ids(input, &mut pos, residual_transition_count, state_id_width)?;
        if transition_targets.iter().any(|&target| target as usize >= state_count) {
            return Err("compressed tokenizer transition target is out of range".to_owned());
        }
        let epsilon_offsets = read_u32_vec(input, &mut pos, state_count + 1)?;
        validate_offsets(&epsilon_offsets, epsilon_count, "epsilon")?;
        let epsilon_targets = read_ids(input, &mut pos, epsilon_count, state_id_width)?;
        if epsilon_targets.iter().any(|&target| target as usize >= state_count) {
            return Err("compressed tokenizer epsilon target is out of range".to_owned());
        }
        let finalizer_offsets = read_u32_vec(input, &mut pos, state_count + 1)?;
        validate_offsets(&finalizer_offsets, finalizer_count, "finalizer")?;
        let finalizers = read_ids(input, &mut pos, finalizer_count, terminal_id_width)?;
        let future_offsets = read_u32_vec(input, &mut pos, state_count + 1)?;
        validate_offsets(&future_offsets, future_count, "future")?;
        let futures = read_ids(input, &mut pos, future_count, terminal_id_width)?;
        if finalizers.iter().chain(&futures).any(|&id| id as usize >= terminal_count) {
            return Err("compressed tokenizer terminal id is out of range".to_owned());
        }
        let segment_end = pos.checked_add(segment_blob_len)
            .ok_or_else(|| "compressed tokenizer segment length overflow".to_owned())?;
        let segment_blob = input.get(pos..segment_end)
            .ok_or_else(|| "truncated compressed tokenizer segments".to_owned())?;
        if segment_end != input.len() {
            return Err("trailing bytes in compressed tokenizer".to_owned());
        }
        let compressed_transition_segments =
            bincode::deserialize::<Vec<CompressedTransitionSegment>>(segment_blob)
                .map_err(|err| err.to_string())?;
        let mut previous_end = 0usize;
        for segment in &compressed_transition_segments {
            let start = segment.state_offset as usize;
            let end = start.checked_add(segment.state_count as usize)
                .ok_or_else(|| "compressed tokenizer segment state range overflow".to_owned())?;
            if start < previous_end || end > state_count || segment.row_offsets.len() != segment.state_count as usize + 1 {
                return Err("invalid compressed tokenizer segment state range".to_owned());
            }
            previous_end = end;
        }
        let mut dfa = DFA::new_from_sparse_metadata(
            group_id_to_u8set,
            &epsilon_offsets,
            &epsilon_targets,
            &finalizer_offsets,
            &finalizers,
            &future_offsets,
            &futures,
        );
        for state in 0..state_count {
            let a = transition_offsets[state] as usize;
            let b = transition_offsets[state + 1] as usize;
            if a != b {
                dfa.set_transitions_from_sorted_entries(
                    state as u32,
                    transition_bytes[a..b]
                        .iter()
                        .copied()
                        .zip(transition_targets[a..b].iter().copied())
                        .collect(),
                );
            }
        }
        Ok(Tokenizer {
            dfa,
            num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from(compressed_transition_segments.into_boxed_slice()),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: None,
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
            sorted_dispatch_roots_cache: OnceLock::new(),
            state_first_bytes_cache: OnceLock::new(),
        })
    }


    const HUGE_WIRE_MAGIC: &[u8; 4] = b"TKS3";
    const HUGE_WIRE_FLAG_SCALAR_DETERMINISTIC_DISPATCH: u8 = 1 << 0;
    const HUGE_WIRE_KNOWN_FLAGS: u8 = HUGE_WIRE_FLAG_SCALAR_DETERMINISTIC_DISPATCH;
    const HUGE_WIRE_HEADER_LEN: usize = 52;

    struct PackedSegmentBuild {
        row_ids: Vec<u32>,
        row_offsets: Vec<u32>,
        classes: Vec<u8>,
        deltas: Vec<i16>,
        overflow_indices: Vec<u32>,
        overflow_deltas: Vec<i32>,
    }

    #[inline]
    fn packed_row_id_width(row_count: usize) -> usize {
        if row_count <= u8::MAX as usize + 1 {
            1
        } else if row_count <= u16::MAX as usize + 1 {
            2
        } else {
            4
        }
    }

    fn write_packed_row_ids(out: &mut Vec<u8>, ids: &[u32], width: usize) {
        match width {
            1 => out.extend(ids.iter().map(|&id| id as u8)),
            2 => {
                for &id in ids {
                    out.extend_from_slice(&(id as u16).to_le_bytes());
                }
            }
            4 => {
                for &id in ids {
                    out.extend_from_slice(&id.to_le_bytes());
                }
            }
            _ => unreachable!(),
        }
    }

    fn read_packed_row_ids(
        input: &[u8],
        pos: &mut usize,
        count: usize,
        width: usize,
        backing: Option<(&Arc<Vec<u8>>, usize)>,
    ) -> Result<PackedRowIds, String> {
        let local_start = *pos;
        let bytes_len = count
            .checked_mul(width)
            .ok_or_else(|| "packed tokenizer row-id length overflow".to_owned())?;
        let end = pos
            .checked_add(bytes_len)
            .ok_or_else(|| "packed tokenizer row-id offset overflow".to_owned())?;
        let bytes = input
            .get(*pos..end)
            .ok_or_else(|| "truncated packed tokenizer row ids".to_owned())?;
        *pos = end;
        if let Some((artifact, section_start)) = backing {
            let start = section_start
                .checked_add(local_start)
                .ok_or_else(|| "packed tokenizer backing offset overflow".to_owned())?;
            return match width {
                1 => Ok(PackedRowIds::BackedU8 {
                    backing: Arc::clone(artifact),
                    start,
                    len: count,
                }),
                2 => Ok(PackedRowIds::BackedU16 {
                    backing: Arc::clone(artifact),
                    start,
                    len: count,
                }),
                4 => Ok(PackedRowIds::BackedU32 {
                    backing: Arc::clone(artifact),
                    start,
                    len: count,
                }),
                _ => Err("invalid packed tokenizer row-id width".to_owned()),
            };
        }
        match width {
            1 => Ok(PackedRowIds::U8(Arc::from(bytes))),
            2 => Ok(PackedRowIds::U16(Arc::from(
                bytes
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ))),
            4 => Ok(PackedRowIds::U32(Arc::from(
                bytes
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ))),
            _ => Err("invalid packed tokenizer row-id width".to_owned()),
        }
    }

    fn packed_segment_build(segment: &CompressedTransitionSegment) -> Option<PackedSegmentBuild> {
        use std::hash::{Hash, Hasher};
        if segment.state_count > i32::MAX as u32 || segment.class_members.len() > u8::MAX as usize {
            return None;
        }
        let mut by_hash = FxHashMap::<u64, Vec<(u32, u32)>>::default();
        let mut row_ids = Vec::with_capacity(segment.state_count as usize);
        let mut row_offsets = vec![0u32];
        let mut classes = Vec::<u8>::new();
        let mut deltas = Vec::<i16>::new();
        let mut overflow_indices = Vec::<u32>::new();
        let mut overflow_deltas = Vec::<i32>::new();

        let rows_equal = |left: u32, right: u32| {
            let la = segment.row_offsets[left as usize] as usize;
            let lb = segment.row_offsets[left as usize + 1] as usize;
            let ra = segment.row_offsets[right as usize] as usize;
            let rb = segment.row_offsets[right as usize + 1] as usize;
            if lb - la != rb - ra || segment.entries.class_slice(la, lb) != segment.entries.class_slice(ra, rb) {
                return false;
            }
            (0..lb - la).all(|offset| {
                let lhs = segment.entries.target(la + offset) as i64 - left as i64;
                let rhs = segment.entries.target(ra + offset) as i64 - right as i64;
                lhs == rhs
            })
        };

        for local_state in 0..segment.state_count {
            let start = segment.row_offsets[local_state as usize] as usize;
            let end = segment.row_offsets[local_state as usize + 1] as usize;
            let mut hasher = rustc_hash::FxHasher::default();
            (end - start).hash(&mut hasher);
            for index in start..end {
                segment.entries.class_slice(index, index + 1)[0].hash(&mut hasher);
                let delta = segment.entries.target(index) as i64 - local_state as i64;
                delta.hash(&mut hasher);
            }
            let hash = hasher.finish();
            let existing = by_hash.get(&hash).and_then(|candidates| {
                candidates
                    .iter()
                    .find(|(_, representative)| rows_equal(local_state, *representative))
                    .map(|(row, _)| *row)
            });
            let row = if let Some(row) = existing {
                row
            } else {
                let row = u32::try_from(row_offsets.len() - 1).ok()?;
                for index in start..end {
                    let class = segment.entries.class_slice(index, index + 1)[0];
                    let delta64 = segment.entries.target(index) as i64 - local_state as i64;
                    let delta = i32::try_from(delta64).ok()?;
                    let entry_index = u32::try_from(classes.len()).ok()?;
                    classes.push(class);
                    if let Ok(short) = i16::try_from(delta) {
                        if short != i16::MIN {
                            deltas.push(short);
                            continue;
                        }
                    }
                    deltas.push(i16::MIN);
                    overflow_indices.push(entry_index);
                    overflow_deltas.push(delta);
                }
                row_offsets.push(u32::try_from(classes.len()).ok()?);
                by_hash.entry(hash).or_default().push((row, local_state));
                row
            };
            row_ids.push(row);
        }
        Some(PackedSegmentBuild {
            row_ids,
            row_offsets,
            classes,
            deltas,
            overflow_indices,
            overflow_deltas,
        })
    }

    fn metadata_rows(tokenizer: &Tokenizer) -> Option<(Vec<[u64; 2]>, Vec<u32>, Vec<[u64; 2]>, Vec<u32>)> {
        if tokenizer.num_terminals > 128 {
            return None;
        }
        fn words2(bits: &BitSet) -> [u64; 2] {
            let words = bits.words();
            [words.first().copied().unwrap_or(0), words.get(1).copied().unwrap_or(0)]
        }
        let mut final_map = FxHashMap::<[u64; 2], u32>::default();
        let mut final_rows = Vec::<[u64; 2]>::new();
        let mut final_ids = Vec::<u32>::with_capacity(tokenizer.dfa.num_states());
        let mut future_map = FxHashMap::<[u64; 2], u32>::default();
        let mut future_rows = Vec::<[u64; 2]>::new();
        let mut future_ids = Vec::<u32>::with_capacity(tokenizer.dfa.num_states());
        for state in tokenizer.dfa.states() {
            let final_key = words2(&state.finalizers);
            let final_id = if let Some(&id) = final_map.get(&final_key) {
                id
            } else {
                let id = u32::try_from(final_rows.len()).ok()?;
                final_rows.push(final_key);
                final_map.insert(final_key, id);
                id
            };
            final_ids.push(final_id);
            let future_key = words2(&state.possible_future_group_ids);
            let future_id = if let Some(&id) = future_map.get(&future_key) {
                id
            } else {
                let id = u32::try_from(future_rows.len()).ok()?;
                future_rows.push(future_key);
                future_map.insert(future_key, id);
                id
            };
            future_ids.push(future_id);
        }
        Some((final_rows, final_ids, future_rows, future_ids))
    }

    /// General metadata-row interning for the compact TKS3 wire. The older
    /// in-memory compaction path intentionally stays on its <=128-terminal
    /// two-word specialization, but a freshly built compressed tokenizer may
    /// have an arbitrary terminal domain. TKS3 itself already stores/read the
    /// exact dynamic word count, so only the builder-side row key needed to be
    /// widened.
    fn metadata_rows_wide(
        tokenizer: &Tokenizer,
    ) -> Option<(Vec<Box<[u64]>>, Vec<u32>, Vec<Box<[u64]>>, Vec<u32>)> {
        let mut final_map = FxHashMap::<Box<[u64]>, u32>::default();
        let mut final_rows = Vec::<Box<[u64]>>::new();
        let mut final_ids = Vec::<u32>::with_capacity(tokenizer.dfa.num_states());
        let mut future_map = FxHashMap::<Box<[u64]>, u32>::default();
        let mut future_rows = Vec::<Box<[u64]>>::new();
        let mut future_ids = Vec::<u32>::with_capacity(tokenizer.dfa.num_states());
        for state in tokenizer.dfa.states() {
            let final_words = state.finalizers.words();
            let final_id = if let Some(&id) = final_map.get(final_words) {
                id
            } else {
                let final_key = final_words.to_vec().into_boxed_slice();
                let id = u32::try_from(final_rows.len()).ok()?;
                final_map.insert(final_key.clone(), id);
                final_rows.push(final_key);
                id
            };
            final_ids.push(final_id);

            let future_words = state.possible_future_group_ids.words();
            let future_id = if let Some(&id) = future_map.get(future_words) {
                id
            } else {
                let future_key = future_words.to_vec().into_boxed_slice();
                let id = u32::try_from(future_rows.len()).ok()?;
                future_map.insert(future_key.clone(), id);
                future_rows.push(future_key);
                id
            };
            future_ids.push(future_id);
        }
        Some((final_rows, final_ids, future_rows, future_ids))
    }

    fn packed_row_ids_from_u32(ids: Vec<u32>, row_count: usize) -> PackedRowIds {
        match packed_row_id_width(row_count) {
            1 => PackedRowIds::U8(Arc::from(
                ids.into_iter()
                    .map(|id| id as u8)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )),
            2 => PackedRowIds::U16(Arc::from(
                ids.into_iter()
                    .map(|id| id as u16)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )),
            4 => PackedRowIds::U32(Arc::from(ids.into_boxed_slice())),
            _ => unreachable!(),
        }
    }

    fn bitset_from_words2(words: [u64; 2], terminal_count: usize) -> BitSet {
        let mut bits = BitSet::new(terminal_count);
        for (slot, value) in bits.words_mut().iter_mut().zip(words) {
            *slot = value;
        }
        bits
    }

    /// Convert a large compressed-suffix tokenizer into the same packed form
    /// used by freshly loaded TKS3 artifacts. This is a runtime storage change,
    /// not a serialization cache: subsequent tokenization reads packed metadata
    /// and packed compressed transition rows directly.
    pub fn compact_large_runtime(tokenizer: &mut Tokenizer) -> bool {
        const MIN_COMPRESSED_STATES: usize = 100_000;
        if tokenizer.compressed_transition_segments.is_empty()
            || tokenizer.packed_runtime_metadata.is_some()
            || !tokenizer.packed_compressed_transition_segments.is_empty()
            || tokenizer.num_terminals > 128
        {
            return false;
        }
        let compressed_states = tokenizer
            .compressed_transition_segments
            .iter()
            .map(|segment| segment.state_count as usize)
            .sum::<usize>();
        if compressed_states < MIN_COMPRESSED_STATES {
            return false;
        }

        let packed_builds = match tokenizer
            .compressed_transition_segments
            .iter()
            .map(packed_segment_build)
            .collect::<Option<Vec<_>>>()
        {
            Some(value) => value,
            None => return false,
        };
        let (final_rows_raw, final_ids, future_rows_raw, future_ids) =
            match metadata_rows(tokenizer) {
                Some(value) => value,
                None => return false,
            };

        let terminal_count = tokenizer.num_terminals as usize;
        let finalizer_rows = final_rows_raw
            .into_iter()
            .map(|words| bitset_from_words2(words, terminal_count))
            .collect::<Vec<_>>();
        let finalizer_lists = finalizer_rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|terminal| terminal as TerminalID)
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>();
        let future_rows = future_rows_raw
            .into_iter()
            .map(|words| bitset_from_words2(words, terminal_count))
            .collect::<Vec<_>>();

        let mut epsilon_states = Vec::<u32>::new();
        let mut epsilon_offsets = vec![0u32];
        let mut epsilon_targets = Vec::<u32>::new();
        for (state, row) in tokenizer.dfa.states().iter().enumerate() {
            if row.epsilon_transitions.is_empty() {
                continue;
            }
            epsilon_states.push(state as u32);
            epsilon_targets.extend_from_slice(&row.epsilon_transitions);
            let Some(end) = u32::try_from(epsilon_targets.len()).ok() else {
                return false;
            };
            epsilon_offsets.push(end);
        }

        let metadata = Arc::new(PackedTokenizerMetadata {
            state_count: tokenizer.dfa.num_states() as u32,
            finalizer_row_ids: packed_row_ids_from_u32(final_ids, finalizer_rows.len()),
            finalizer_rows: Arc::from(finalizer_rows.into_boxed_slice()),
            finalizer_lists: Arc::from(finalizer_lists.into_boxed_slice()),
            future_row_ids: packed_row_ids_from_u32(future_ids, future_rows.len()),
            future_rows: Arc::from(future_rows.into_boxed_slice()),
            epsilon_states: Arc::from(epsilon_states.into_boxed_slice()),
            epsilon_offsets: Arc::from(epsilon_offsets.into_boxed_slice()),
            epsilon_targets: Arc::from(epsilon_targets.into_boxed_slice()),
        });

        let packed_segments = tokenizer
            .compressed_transition_segments
            .iter()
            .zip(packed_builds)
            .map(|(segment, packed)| PackedCompressedTransitionSegment {
                state_offset: segment.state_offset,
                state_count: segment.state_count,
                byte_to_class: PackedRuntimeBytes::Owned(Arc::clone(&segment.byte_to_class)),
                class_members: Arc::clone(&segment.class_members),
                row_ids: packed_row_ids_from_u32(
                    packed.row_ids,
                    packed.row_offsets.len().saturating_sub(1),
                ),
                row_offsets: Arc::from(packed.row_offsets.into_boxed_slice()),
                classes: PackedRuntimeBytes::Owned(Arc::from(packed.classes.into_boxed_slice())),
                deltas: PackedI16Values::Owned(Arc::from(packed.deltas.into_boxed_slice())),
                overflow_indices: Arc::from(packed.overflow_indices.into_boxed_slice()),
                overflow_deltas: Arc::from(packed.overflow_deltas.into_boxed_slice()),
                expanded_transition_count: segment.expanded_transition_count,
            })
            .collect::<Vec<_>>();

        tokenizer.packed_runtime_metadata = Some(metadata);
        tokenizer.packed_compressed_transition_segments =
            Arc::from(packed_segments.into_boxed_slice());
        tokenizer.compressed_transition_segments = Arc::from([]);
        tokenizer.invalidate_derived_caches();
        true
    }

    /// Add contiguous runtime transition storage for large ordinary DFAs.  The
    /// full DFA is retained for compiler/analysis consumers, but runtime byte
    /// stepping immediately prefers this sidecar.  This is therefore a real
    /// runtime representation, not serialized-wire precomputation.
    pub fn compact_large_fast_runtime(tokenizer: &mut Tokenizer) -> bool {
        const MIN_TRANSITIONS: usize = 1_000_000;
        if tokenizer.packed_runtime_transitions.is_some()
            || !tokenizer.packed_runtime_transition_segments.is_empty()
            || !tokenizer.compressed_transition_segments.is_empty()
            || !tokenizer.packed_compressed_transition_segments.is_empty()
        {
            return false;
        }
        // A byte DFA has at most 256 labeled byte transitions per state.
        // Avoid an O(states + transitions) count for the common small-tokenizer
        // case when the threshold is impossible to reach.
        const MAX_BYTE_TRANSITIONS_PER_STATE: usize = 256;
        if tokenizer.dfa.num_states()
            < MIN_TRANSITIONS.div_ceil(MAX_BYTE_TRANSITIONS_PER_STATE)
        {
            return false;
        }
        let transition_count = tokenizer.dfa.transition_count();
        if transition_count < MIN_TRANSITIONS {
            return false;
        }
        let states = tokenizer.dfa.states();
        let state_count = states.len();
        let mut offsets = Vec::<u32>::with_capacity(state_count + 1);
        let mut bytes = Vec::<u8>::with_capacity(transition_count);
        offsets.push(0);
        if state_count <= u16::MAX as usize + 1 {
            let mut targets = Vec::<u16>::with_capacity(transition_count);
            for state in states {
                bytes.extend(state.transitions.iter().map(|(byte, _)| byte));
                targets.extend(state.transitions.values().map(|&target| {
                    u16::try_from(target).expect("u16-sized DFA must have u16 transition targets")
                }));
                offsets.push(
                    u32::try_from(bytes.len())
                        .expect("runtime tokenizer transition count must fit u32"),
                );
            }
            tokenizer.packed_runtime_transitions = Some(Arc::new(PackedRuntimeTransitions {
                byte_offsets: Arc::from(offsets.into_boxed_slice()),
                bytes: PackedRuntimeBytes::Owned(Arc::from(bytes.into_boxed_slice())),
                targets: PackedRuntimeTargets::U16(Arc::from(targets.into_boxed_slice())),
            }));
        } else {
            let mut targets = Vec::<u32>::with_capacity(transition_count);
            for state in states {
                bytes.extend(state.transitions.iter().map(|(byte, _)| byte));
                targets.extend(state.transitions.values().copied());
                offsets.push(
                    u32::try_from(bytes.len())
                        .expect("runtime tokenizer transition count must fit u32"),
                );
            }
            tokenizer.packed_runtime_transitions = Some(Arc::new(PackedRuntimeTransitions {
                byte_offsets: Arc::from(offsets.into_boxed_slice()),
                bytes: PackedRuntimeBytes::Owned(Arc::from(bytes.into_boxed_slice())),
                targets: PackedRuntimeTargets::U32(Arc::from(targets.into_boxed_slice())),
            }));
        }
        true
    }

    fn write_packed_row_ids_any(out: &mut Vec<u8>, ids: &PackedRowIds, width: usize) {
        match (ids, width) {
            (PackedRowIds::U8(values), 1) => out.extend_from_slice(values),
            (PackedRowIds::U16(values), 2) if cfg!(target_endian = "little") => {
                // SAFETY: u16 has no padding and TKS3 is little-endian.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        values.as_ptr().cast::<u8>(),
                        values.len() * std::mem::size_of::<u16>(),
                    )
                };
                out.extend_from_slice(bytes);
            }
            (PackedRowIds::U32(values), 4) if cfg!(target_endian = "little") => {
                // SAFETY: u32 has no padding and TKS3 is little-endian.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        values.as_ptr().cast::<u8>(),
                        values.len() * std::mem::size_of::<u32>(),
                    )
                };
                out.extend_from_slice(bytes);
            }
            _ => {
                for index in 0..ids.len() {
                    let id = ids
                        .get(index)
                        .expect("packed row id was validated to cover every state")
                        as u32;
                    match width {
                        1 => out.push(id as u8),
                        2 => out.extend_from_slice(&(id as u16).to_le_bytes()),
                        4 => out.extend_from_slice(&id.to_le_bytes()),
                        _ => unreachable!(),
                    }
                }
            }
        }
    }

    #[inline]
    fn extend_u32_le(out: &mut Vec<u8>, values: &[u32]) {
        if cfg!(target_endian = "little") {
            // SAFETY: u32 has no padding and the source slice outlives the copy.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            };
            out.extend_from_slice(bytes);
        } else {
            for &value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }

    #[inline]
    fn extend_i16_le(out: &mut Vec<u8>, values: &[i16]) {
        if cfg!(target_endian = "little") {
            // SAFETY: i16 has no padding and the source slice outlives the copy.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            };
            out.extend_from_slice(bytes);
        } else {
            for &value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }

    #[inline]
    fn extend_i32_le(out: &mut Vec<u8>, values: &[i32]) {
        if cfg!(target_endian = "little") {
            // SAFETY: i32 has no padding and the source slice outlives the copy.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            };
            out.extend_from_slice(bytes);
        } else {
            for &value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }

    fn build_huge_bytes_from_packed_runtime(tokenizer: &Tokenizer) -> Option<Vec<u8>> {
        let metadata = tokenizer.packed_runtime_metadata.as_deref()?;
        let segments = tokenizer.packed_compressed_transition_segments.as_ref();
        if segments.is_empty()
            || !tokenizer.packed_runtime_transition_segments.is_empty()
            || !tokenizer.packed_runtime_metadata_segments.is_empty()
        {
            return None;
        }
        // A decoded TKS3 tokenizer owns only its prefix DFA states; the
        // compressed suffix lives in backed packed segments. Use the runtime
        // state count here rather than the owned-DFA count so re-encoding a
        // loaded artifact validates against the full metadata coordinate.
        let state_count = tokenizer.num_states() as usize;
        if metadata.state_count as usize != state_count
            || metadata.finalizer_row_ids.len() != state_count
            || metadata.future_row_ids.len() != state_count
        {
            return None;
        }
        let prefix_state_count = segments.first()?.state_offset as usize;
        let mut expected = prefix_state_count;
        for segment in segments {
            if segment.state_offset as usize != expected {
                return None;
            }
            expected = expected.checked_add(segment.state_count as usize)?;
        }
        if expected != state_count {
            return None;
        }

        let prefix_states = &tokenizer.dfa.states()[..prefix_state_count];
        let packed_prefix = tokenizer.packed_runtime_transitions.as_deref();
        if packed_prefix.is_some_and(|prefix| prefix.state_count() != prefix_state_count) {
            return None;
        }
        let residual_transition_count = if let Some(prefix) = packed_prefix {
            (0..prefix_state_count)
                .map(|state| prefix.row(state as u32).map(|row| row.0.len()))
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .sum::<usize>()
        } else {
            prefix_states
                .iter()
                .map(|state| state.transitions.len())
                .sum::<usize>()
        };
        let expanded_transition_count = residual_transition_count
            + segments
                .iter()
                .map(|segment| segment.expanded_transition_count)
                .sum::<usize>();
        let final_width = packed_row_id_width(metadata.finalizer_rows.len());
        let future_width = packed_row_id_width(metadata.future_rows.len());

        let metadata_word_count = (tokenizer.num_terminals as usize).div_ceil(64);
        let mut exact_len = HUGE_WIRE_HEADER_LEN
            + tokenizer.num_terminals as usize * 32
            + (prefix_state_count + 1) * 4
            + residual_transition_count
            + residual_transition_count * 4
            + metadata.finalizer_rows.len() * metadata_word_count * 8
            + metadata.future_rows.len() * metadata_word_count * 8
            + state_count * final_width
            + state_count * future_width
            + metadata.epsilon_states.len() * 4
            + metadata.epsilon_offsets.len() * 4
            + metadata.epsilon_targets.len() * 4;
        for segment in segments {
            let row_count = segment.row_offsets.len().checked_sub(1)?;
            let entry_count = segment.classes.as_slice().len();
            let row_width = packed_row_id_width(row_count);
            exact_len = exact_len
                .checked_add(28)?
                .checked_add(segment.byte_to_class.as_slice().len())?
                .checked_add(segment.row_ids.len().checked_mul(row_width)?)?
                .checked_add(segment.row_offsets.len().checked_mul(4)?)?
                .checked_add(entry_count)?
                .checked_add(entry_count.checked_mul(2)?)?
                .checked_add(segment.overflow_indices.len().checked_mul(4)?)?
                .checked_add(segment.overflow_deltas.len().checked_mul(4)?)?;
        }
        let mut out = Vec::<u8>::with_capacity(exact_len);
        out.extend_from_slice(HUGE_WIRE_MAGIC);
        for value in [
            tokenizer.num_terminals,
            state_count as u32,
            prefix_state_count as u32,
            residual_transition_count as u32,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&(expanded_transition_count as u64).to_le_bytes());
        for value in [
            segments.len() as u32,
            metadata.finalizer_rows.len() as u32,
            metadata.future_rows.len() as u32,
            metadata.epsilon_states.len() as u32,
            metadata.epsilon_targets.len() as u32,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let flags = if tokenizer.scalar_deterministic_dispatch_cache.get() == Some(&true) {
            HUGE_WIRE_FLAG_SCALAR_DETERMINISTIC_DISPATCH
        } else {
            0
        };
        out.extend_from_slice(&[final_width as u8, future_width as u8, flags, 0]);
        debug_assert_eq!(out.len(), HUGE_WIRE_HEADER_LEN);
        for terminal in 0..tokenizer.num_terminals {
            for word in tokenizer.dfa.group_id_to_u8set(terminal).to_words() {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        let mut end = 0u32;
        out.extend_from_slice(&end.to_le_bytes());
        if let Some(prefix) = packed_prefix {
            for state in 0..prefix_state_count as u32 {
                let (bytes, _) = prefix.row(state)?;
                end = end.checked_add(u32::try_from(bytes.len()).ok()?)?;
                out.extend_from_slice(&end.to_le_bytes());
            }
            for state in 0..prefix_state_count as u32 {
                let (bytes, _) = prefix.row(state)?;
                out.extend_from_slice(bytes);
            }
            for state in 0..prefix_state_count as u32 {
                let (_, targets) = prefix.row(state)?;
                for index in 0..targets.len() {
                    out.extend_from_slice(&targets.get(index)?.to_le_bytes());
                }
            }
        } else {
            for state in prefix_states {
                end = end.checked_add(u32::try_from(state.transitions.len()).ok()?)?;
                out.extend_from_slice(&end.to_le_bytes());
            }
            for state in prefix_states {
                out.extend(state.transitions.iter().map(|(byte, _)| byte));
            }
            for state in prefix_states {
                for &target in state.transitions.values() {
                    out.extend_from_slice(&target.to_le_bytes());
                }
            }
        }
        for row in metadata.finalizer_rows.iter() {
            for word in row.words().iter().take(metadata_word_count) {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        for row in metadata.future_rows.iter() {
            for word in row.words().iter().take(metadata_word_count) {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        write_packed_row_ids_any(&mut out, &metadata.finalizer_row_ids, final_width);
        write_packed_row_ids_any(&mut out, &metadata.future_row_ids, future_width);
        extend_u32_le(&mut out, &metadata.epsilon_states);
        extend_u32_le(&mut out, &metadata.epsilon_offsets);
        extend_u32_le(&mut out, &metadata.epsilon_targets);
        for segment in segments {
            let class_count = u16::try_from(segment.class_members.len()).ok()?;
            let row_count = segment.row_offsets.len().checked_sub(1)?;
            let entry_count = segment.classes.as_slice().len();
            let row_width = packed_row_id_width(row_count);
            for value in [
                segment.state_offset,
                segment.state_count,
                u32::try_from(segment.expanded_transition_count).ok()?,
                u32::try_from(row_count).ok()?,
                u32::try_from(entry_count).ok()?,
                u32::try_from(segment.overflow_indices.len()).ok()?,
            ] {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out.extend_from_slice(&class_count.to_le_bytes());
            out.extend_from_slice(&[row_width as u8, 0]);
            out.extend_from_slice(segment.byte_to_class.as_slice());
            write_packed_row_ids_any(&mut out, &segment.row_ids, row_width);
            extend_u32_le(&mut out, &segment.row_offsets);
            out.extend_from_slice(segment.classes.as_slice());
            if let Some(bytes) = segment.deltas.backed_le_bytes() {
                out.extend_from_slice(bytes);
            } else if let Some(values) = segment.deltas.owned_values() {
                extend_i16_le(&mut out, values);
            } else {
                for index in 0..entry_count {
                    out.extend_from_slice(
                        &segment
                            .deltas
                            .get(index)
                            .expect("packed delta row must cover every entry")
                            .to_le_bytes(),
                    );
                }
            }
            extend_u32_le(&mut out, &segment.overflow_indices);
            extend_i32_le(&mut out, &segment.overflow_deltas);
        }
        debug_assert_eq!(out.len(), exact_len);
        Some(out)
    }

    /// Build the compact giant-tokenizer artifact. This is intentionally gated
    /// to the already-compressed suffix representation; normal tokenizers stay
    /// on TKF2 byte-for-byte.
    pub fn build_huge_bytes(tokenizer: &Tokenizer) -> Option<Vec<u8>> {
        if tokenizer.compressed_transition_segments.is_empty() {
            return build_huge_bytes_from_packed_runtime(tokenizer);
        }
        let segments = tokenizer.compressed_transition_segments.as_ref();
        if segments.is_empty() {
            return None;
        }
        let state_count = tokenizer.dfa.num_states();
        let prefix_state_count = segments.first()?.state_offset as usize;
        let mut expected = prefix_state_count;
        for segment in segments {
            if segment.state_offset as usize != expected {
                return None;
            }
            expected = expected.checked_add(segment.state_count as usize)?;
        }
        if expected != state_count {
            return None;
        }
        let packed_segments = segments
            .iter()
            .map(packed_segment_build)
            .collect::<Option<Vec<_>>>()?;
        let (final_rows, final_ids, future_rows, future_ids) = metadata_rows_wide(tokenizer)?;
        let final_width = packed_row_id_width(final_rows.len());
        let future_width = packed_row_id_width(future_rows.len());

        let prefix_states = &tokenizer.dfa.states()[..prefix_state_count];
        let residual_transition_count = prefix_states
            .iter()
            .map(|state| state.transitions.len())
            .sum::<usize>();
        let expanded_transition_count = residual_transition_count
            + segments
                .iter()
                .map(|segment| segment.expanded_transition_count)
                .sum::<usize>();
        let mut epsilon_states = Vec::<u32>::new();
        let mut epsilon_offsets = vec![0u32];
        let mut epsilon_targets = Vec::<u32>::new();
        for (state, row) in tokenizer.dfa.states().iter().enumerate() {
            if row.epsilon_transitions.is_empty() {
                continue;
            }
            epsilon_states.push(state as u32);
            epsilon_targets.extend_from_slice(&row.epsilon_transitions);
            epsilon_offsets.push(u32::try_from(epsilon_targets.len()).ok()?);
        }

        let mut out = Vec::<u8>::new();
        out.extend_from_slice(HUGE_WIRE_MAGIC);
        for value in [
            tokenizer.num_terminals,
            state_count as u32,
            prefix_state_count as u32,
            residual_transition_count as u32,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&(expanded_transition_count as u64).to_le_bytes());
        for value in [
            segments.len() as u32,
            final_rows.len() as u32,
            future_rows.len() as u32,
            epsilon_states.len() as u32,
            epsilon_targets.len() as u32,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let flags = if tokenizer.scalar_deterministic_dispatch_cache.get() == Some(&true) {
            HUGE_WIRE_FLAG_SCALAR_DETERMINISTIC_DISPATCH
        } else {
            0
        };
        out.extend_from_slice(&[final_width as u8, future_width as u8, flags, 0]);
        debug_assert_eq!(out.len(), HUGE_WIRE_HEADER_LEN);
        for terminal in 0..tokenizer.num_terminals {
            for word in tokenizer.dfa.group_id_to_u8set(terminal).to_words() {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        let mut end = 0u32;
        out.extend_from_slice(&end.to_le_bytes());
        for state in prefix_states {
            end += state.transitions.len() as u32;
            out.extend_from_slice(&end.to_le_bytes());
        }
        for state in prefix_states {
            out.extend(state.transitions.iter().map(|(byte, _)| byte));
        }
        for state in prefix_states {
            for &target in state.transitions.values() {
                out.extend_from_slice(&target.to_le_bytes());
            }
        }
        let metadata_word_count = (tokenizer.num_terminals as usize).div_ceil(64);
        for row in &final_rows {
            for word in row.iter().take(metadata_word_count) {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        for row in &future_rows {
            for word in row.iter().take(metadata_word_count) {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        write_packed_row_ids(&mut out, &final_ids, final_width);
        write_packed_row_ids(&mut out, &future_ids, future_width);
        for &state in &epsilon_states {
            out.extend_from_slice(&state.to_le_bytes());
        }
        for &offset in &epsilon_offsets {
            out.extend_from_slice(&offset.to_le_bytes());
        }
        for &target in &epsilon_targets {
            out.extend_from_slice(&target.to_le_bytes());
        }
        for (segment, packed) in segments.iter().zip(&packed_segments) {
            let class_count = u16::try_from(segment.class_members.len()).ok()?;
            let row_count = packed.row_offsets.len().checked_sub(1)?;
            let row_width = packed_row_id_width(row_count);
            for value in [
                segment.state_offset,
                segment.state_count,
                u32::try_from(segment.expanded_transition_count).ok()?,
                row_count as u32,
                packed.classes.len() as u32,
                packed.overflow_indices.len() as u32,
            ] {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out.extend_from_slice(&class_count.to_le_bytes());
            out.push(row_width as u8);
            out.push(0);
            out.extend_from_slice(&segment.byte_to_class);
            write_packed_row_ids(&mut out, &packed.row_ids, row_width);
            for &offset in &packed.row_offsets {
                out.extend_from_slice(&offset.to_le_bytes());
            }
            out.extend_from_slice(&packed.classes);
            for &delta in &packed.deltas {
                out.extend_from_slice(&delta.to_le_bytes());
            }
            for &index in &packed.overflow_indices {
                out.extend_from_slice(&index.to_le_bytes());
            }
            for &delta in &packed.overflow_deltas {
                out.extend_from_slice(&delta.to_le_bytes());
            }
        }
        Some(out)
    }

    fn from_huge_bytes(
        input: &[u8],
        backing: Option<(Arc<Vec<u8>>, usize)>,
    ) -> Result<Tokenizer, String> {
        use rayon::prelude::*;
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let huge_started = profile.then(std::time::Instant::now);
        if input.len() < HUGE_WIRE_HEADER_LEN || !input.starts_with(HUGE_WIRE_MAGIC) {
            return Err("invalid giant tokenizer header".to_owned());
        }
        let mut pos = 4usize;
        let take_u32 = |input: &[u8], pos: &mut usize| -> Result<u32, String> {
            let end = pos.checked_add(4).ok_or_else(|| "giant tokenizer offset overflow".to_owned())?;
            let bytes = input.get(*pos..end).ok_or_else(|| "truncated giant tokenizer".to_owned())?;
            *pos = end;
            Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
        };
        let take_u64 = |input: &[u8], pos: &mut usize| -> Result<u64, String> {
            let end = pos.checked_add(8).ok_or_else(|| "giant tokenizer offset overflow".to_owned())?;
            let bytes = input.get(*pos..end).ok_or_else(|| "truncated giant tokenizer".to_owned())?;
            *pos = end;
            Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
        };
        let num_terminals = take_u32(input, &mut pos)?;
        let state_count = take_u32(input, &mut pos)? as usize;
        let prefix_state_count = take_u32(input, &mut pos)? as usize;
        let residual_transition_count = take_u32(input, &mut pos)? as usize;
        let expanded_transition_count = take_u64(input, &mut pos)? as usize;
        let segment_count = take_u32(input, &mut pos)? as usize;
        let final_row_count = take_u32(input, &mut pos)? as usize;
        let future_row_count = take_u32(input, &mut pos)? as usize;
        let epsilon_state_count = take_u32(input, &mut pos)? as usize;
        let epsilon_target_count = take_u32(input, &mut pos)? as usize;
        let final_width = *input.get(pos).ok_or_else(|| "truncated giant tokenizer widths".to_owned())? as usize;
        let future_width = *input.get(pos + 1).ok_or_else(|| "truncated giant tokenizer widths".to_owned())? as usize;
        let flags = *input.get(pos + 2).ok_or_else(|| "truncated giant tokenizer flags".to_owned())?;
        let reserved = *input.get(pos + 3).ok_or_else(|| "truncated giant tokenizer flags".to_owned())?;
        if flags & !HUGE_WIRE_KNOWN_FLAGS != 0
            || reserved != 0
            || !matches!(final_width, 1 | 2 | 4)
            || !matches!(future_width, 1 | 2 | 4)
        {
            return Err("invalid giant tokenizer row-id widths".to_owned());
        }
        pos += 4;
        if pos != HUGE_WIRE_HEADER_LEN || state_count == 0 || prefix_state_count > state_count {
            return Err("invalid giant tokenizer dimensions".to_owned());
        }
        let terminal_count = num_terminals as usize;
        let mut groups = Vec::with_capacity(terminal_count);
        for _ in 0..terminal_count {
            let mut words = [0u64; 4];
            for word in &mut words {
                let end = pos + 8;
                let bytes = input.get(pos..end).ok_or_else(|| "truncated giant tokenizer groups".to_owned())?;
                *word = u64::from_le_bytes(bytes.try_into().unwrap());
                pos = end;
            }
            groups.push(U8Set::from_words(words));
        }
        fn read_u32_vec_at(
            input: &[u8],
            pos: &mut usize,
            count: usize,
        ) -> Result<Vec<u32>, String> {
            let end = pos
                .checked_add(
                    count
                        .checked_mul(4)
                        .ok_or_else(|| "giant vector overflow".to_owned())?,
                )
                .ok_or_else(|| "giant tokenizer offset overflow".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated giant tokenizer vector".to_owned())?;
            *pos = end;
            Ok(bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect())
        }
        let prefix_offsets = read_u32_vec_at(input, &mut pos, prefix_state_count + 1)?;
        if prefix_offsets.first().copied() != Some(0)
            || prefix_offsets.last().copied() != Some(residual_transition_count as u32)
            || prefix_offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err("invalid giant tokenizer residual offsets".to_owned());
        }
        let bytes_end = pos.checked_add(residual_transition_count).ok_or_else(|| "giant tokenizer offset overflow".to_owned())?;
        let prefix_bytes = Arc::<[u8]>::from(input.get(pos..bytes_end).ok_or_else(|| "truncated giant tokenizer residual bytes".to_owned())?);
        pos = bytes_end;
        let prefix_targets = read_u32_vec_at(input, &mut pos, residual_transition_count)?;
        if prefix_targets.iter().any(|&target| target as usize >= state_count) {
            return Err("giant tokenizer residual target out of range".to_owned());
        }
        let mut read_metadata_rows = |count: usize| -> Result<Vec<BitSet>, String> {
            let mut rows = Vec::with_capacity(count);
            for _ in 0..count {
                let mut bits = BitSet::new(terminal_count);
                for word in bits.words_mut() {
                    let end = pos + 8;
                    let bytes = input.get(pos..end).ok_or_else(|| "truncated giant tokenizer metadata".to_owned())?;
                    *word = u64::from_le_bytes(bytes.try_into().unwrap());
                    pos = end;
                }
                rows.push(bits);
            }
            Ok(rows)
        };
        let final_rows = read_metadata_rows(final_row_count)?;
        let future_rows = read_metadata_rows(future_row_count)?;
        let backed = backing.as_ref().map(|(artifact, start)| (artifact, *start));
        let final_row_ids =
            read_packed_row_ids(input, &mut pos, state_count, final_width, backed)?;
        let future_row_ids =
            read_packed_row_ids(input, &mut pos, state_count, future_width, backed)?;
        let meta_validate_started = profile.then(std::time::Instant::now);
        let (final_ids_valid, future_ids_valid) = if state_count >= 100_000
            && rayon::current_num_threads() > 1
        {
            rayon::join(
                || final_row_ids.all_lt(final_rows.len()),
                || future_row_ids.all_lt(future_rows.len()),
            )
        } else {
            (
                final_row_ids.all_lt(final_rows.len()),
                future_row_ids.all_lt(future_rows.len()),
            )
        };
        if !final_ids_valid || !future_ids_valid {
            return Err("giant tokenizer metadata row id out of range".to_owned());
        }
        if let Some(started) = meta_validate_started {
            eprintln!("[glrmask/profile][tks3] meta_ids_validate_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
        }
        let epsilon_states = read_u32_vec_at(input, &mut pos, epsilon_state_count)?;
        let epsilon_offsets = read_u32_vec_at(input, &mut pos, epsilon_state_count + 1)?;
        let epsilon_targets = read_u32_vec_at(input, &mut pos, epsilon_target_count)?;
        if epsilon_offsets.first().copied() != Some(0)
            || epsilon_offsets.last().copied() != Some(epsilon_target_count as u32)
            || epsilon_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || epsilon_states.windows(2).any(|pair| pair[0] >= pair[1])
            || epsilon_states.iter().any(|&state| state as usize >= state_count)
            || epsilon_targets.iter().any(|&state| state as usize >= state_count)
        {
            return Err("invalid giant tokenizer epsilon metadata".to_owned());
        }
        let finalizer_lists = final_rows
            .iter()
            .map(|row| row.iter().map(|id| id as TerminalID).collect::<Vec<_>>().into_boxed_slice())
            .collect::<Vec<_>>();
        let metadata = Arc::new(PackedTokenizerMetadata {
            state_count: state_count as u32,
            finalizer_row_ids: final_row_ids,
            finalizer_rows: Arc::from(final_rows.into_boxed_slice()),
            finalizer_lists: Arc::from(finalizer_lists.into_boxed_slice()),
            future_row_ids,
            future_rows: Arc::from(future_rows.into_boxed_slice()),
            epsilon_states: Arc::from(epsilon_states.into_boxed_slice()),
            epsilon_offsets: Arc::from(epsilon_offsets.into_boxed_slice()),
            epsilon_targets: Arc::from(epsilon_targets.into_boxed_slice()),
        });
        let mut packed_segments = Vec::with_capacity(segment_count);
        let mut expected_state = prefix_state_count as u32;
        let mut segment_expanded = 0usize;
        for _ in 0..segment_count {
            let segment_started = profile.then(std::time::Instant::now);
            let state_offset = take_u32(input, &mut pos)?;
            let segment_state_count = take_u32(input, &mut pos)?;
            let segment_expanded_count = take_u32(input, &mut pos)? as usize;
            let row_count = take_u32(input, &mut pos)? as usize;
            let entry_count = take_u32(input, &mut pos)? as usize;
            let overflow_count = take_u32(input, &mut pos)? as usize;
            let class_end = pos + 2;
            let class_count = u16::from_le_bytes(input.get(pos..class_end).ok_or_else(|| "truncated giant segment class count".to_owned())?.try_into().unwrap()) as usize;
            pos = class_end;
            let row_width = *input.get(pos).ok_or_else(|| "truncated giant segment row width".to_owned())? as usize;
            let reserved = *input.get(pos + 1).ok_or_else(|| "truncated giant segment row width".to_owned())?;
            pos += 2;
            if state_offset != expected_state || segment_state_count == 0 || class_count == 0 || class_count > 255 || reserved != 0 || !matches!(row_width, 1 | 2 | 4) {
                return Err("invalid giant tokenizer segment header".to_owned());
            }
            expected_state = expected_state.checked_add(segment_state_count).ok_or_else(|| "giant tokenizer state range overflow".to_owned())?;
            let map_end = pos + 256;
            let byte_to_class_slice = input
                .get(pos..map_end)
                .ok_or_else(|| "truncated giant tokenizer byte classes".to_owned())?;
            let byte_to_class = if let Some((artifact, section_start)) = backed {
                PackedRuntimeBytes::Backed {
                    backing: Arc::clone(artifact),
                    start: section_start + pos,
                    len: 256,
                }
            } else {
                PackedRuntimeBytes::Owned(Arc::from(byte_to_class_slice))
            };
            pos = map_end;
            if byte_to_class
                .as_slice()
                .iter()
                .any(|&class| class != u8::MAX && class as usize >= class_count)
            {
                return Err("giant tokenizer byte class out of range".to_owned());
            }
            let row_ids = read_packed_row_ids(
                input,
                &mut pos,
                segment_state_count as usize,
                row_width,
                backed,
            )?;
            let row_offsets = read_u32_vec_at(input, &mut pos, row_count + 1)?;
            if row_offsets.first().copied() != Some(0)
                || row_offsets.last().copied() != Some(entry_count as u32)
                || row_offsets.windows(2).any(|pair| pair[0] > pair[1])
            {
                return Err("invalid giant tokenizer transition dictionary offsets".to_owned());
            }
            let classes_end = pos + entry_count;
            let classes_slice = input
                .get(pos..classes_end)
                .ok_or_else(|| "truncated giant tokenizer transition classes".to_owned())?;
            let classes = if let Some((artifact, section_start)) = backed {
                PackedRuntimeBytes::Backed {
                    backing: Arc::clone(artifact),
                    start: section_start + pos,
                    len: entry_count,
                }
            } else {
                PackedRuntimeBytes::Owned(Arc::from(classes_slice))
            };
            pos = classes_end;
            if classes
                .as_slice()
                .iter()
                .any(|&class| class as usize >= class_count)
            {
                return Err("giant tokenizer transition class out of range".to_owned());
            }
            let delta_end = pos.checked_add(entry_count * 2).ok_or_else(|| "giant tokenizer delta overflow".to_owned())?;
            let delta_bytes = input.get(pos..delta_end).ok_or_else(|| "truncated giant tokenizer deltas".to_owned())?;
            let deltas = if let Some((artifact, section_start)) = backed {
                PackedI16Values::Backed {
                    backing: Arc::clone(artifact),
                    start: section_start + pos,
                    len: entry_count,
                }
            } else {
                PackedI16Values::Owned(Arc::from(
                    delta_bytes
                        .chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]]))
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                ))
            };
            pos = delta_end;
            let overflow_indices = read_u32_vec_at(input, &mut pos, overflow_count)?;
            let overflow_raw = read_u32_vec_at(input, &mut pos, overflow_count)?;
            let overflow_deltas = Arc::<[i32]>::from(overflow_raw.into_iter().map(|v| v as i32).collect::<Vec<_>>().into_boxed_slice());
            let mut class_members = vec![Vec::<u8>::new(); class_count];
            for (byte, &class) in byte_to_class.as_slice().iter().enumerate() {
                if class != u8::MAX {
                    class_members[class as usize].push(byte as u8);
                }
            }
            let class_members = Arc::from(
                class_members
                    .into_iter()
                    .map(Vec::into_boxed_slice)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );
            let segment = PackedCompressedTransitionSegment {
                state_offset,
                state_count: segment_state_count,
                byte_to_class,
                class_members,
                row_ids,
                row_offsets: Arc::from(row_offsets.into_boxed_slice()),
                classes,
                deltas,
                overflow_indices: Arc::from(overflow_indices.into_boxed_slice()),
                overflow_deltas,
                expanded_transition_count: segment_expanded_count,
            };
            // Validate row ids and target bounds together. The historical
            // decoder first scanned every physical state only to validate the
            // row id, then scanned all of them again for target bounds.
            let mut min_delta = vec![0i32; row_count];
            let mut max_delta = vec![0i32; row_count];
            let delta_validate_started = profile.then(std::time::Instant::now);
            if let Some(delta_bytes) = segment.deltas.backed_le_bytes() {
                // The section length and row-offset coverage have already been
                // validated. Walk the fixed-width slab once, resolving the
                // sparse overflow table monotonically instead of calling
                // `segment.delta()` for every entry (which performs bounds
                // checks and binary-searches overflow sentinels).
                let scan_rows = |row_base: usize,
                                 min_chunk: &mut [i32],
                                 max_chunk: &mut [i32]|
                 -> Result<(), String> {
                    let first_entry = segment.row_offsets[row_base] as usize;
                    let end_entry = segment.row_offsets[row_base + min_chunk.len()] as usize;
                    let mut overflow = segment
                        .overflow_indices
                        .partition_point(|&index| (index as usize) < first_entry);
                    let expected_overflow_end = segment
                        .overflow_indices
                        .partition_point(|&index| (index as usize) < end_entry);
                    for offset in 0..min_chunk.len() {
                        let row = row_base + offset;
                        let start = segment.row_offsets[row] as usize;
                        let end = segment.row_offsets[row + 1] as usize;
                        if start == end {
                            continue;
                        }
                        let mut min = i32::MAX;
                        let mut max = i32::MIN;
                        for index in start..end {
                            let byte = index * 2;
                            let raw = i16::from_le_bytes([delta_bytes[byte], delta_bytes[byte + 1]]);
                            let delta = if raw != i16::MIN {
                                if segment
                                    .overflow_indices
                                    .get(overflow)
                                    .is_some_and(|&overflow_index| overflow_index as usize == index)
                                {
                                    return Err("invalid giant tokenizer overflow table".to_owned());
                                }
                                raw as i32
                            } else {
                                let Some(&overflow_index) = segment.overflow_indices.get(overflow) else {
                                    return Err("missing giant tokenizer overflow record".to_owned());
                                };
                                if overflow_index as usize != index {
                                    return Err("invalid giant tokenizer overflow table".to_owned());
                                }
                                let Some(&delta) = segment.overflow_deltas.get(overflow) else {
                                    return Err("missing giant tokenizer overflow delta".to_owned());
                                };
                                overflow += 1;
                                delta
                            };
                            min = min.min(delta);
                            max = max.max(delta);
                        }
                        min_chunk[offset] = min;
                        max_chunk[offset] = max;
                    }
                    if overflow != expected_overflow_end {
                        return Err("unused giant tokenizer overflow record".to_owned());
                    }
                    Ok(())
                };

                if entry_count >= 100_000 && rayon::current_num_threads() > 1 {
                    let workers = rayon::current_num_threads().min(8).max(1);
                    let chunk_rows = row_count.div_ceil(workers);
                    min_delta
                        .par_chunks_mut(chunk_rows)
                        .zip(max_delta.par_chunks_mut(chunk_rows))
                        .enumerate()
                        .try_for_each(|(chunk, (min_chunk, max_chunk))| {
                            scan_rows(chunk * chunk_rows, min_chunk, max_chunk)
                        })?;
                } else {
                    scan_rows(0, &mut min_delta, &mut max_delta)?;
                }

                if segment.overflow_indices.len() != segment.overflow_deltas.len() {
                    return Err("unused giant tokenizer overflow record".to_owned());
                }
            } else {
                for row in 0..row_count {
                    let start = segment.row_offsets[row] as usize;
                    let end = segment.row_offsets[row + 1] as usize;
                    if start == end {
                        continue;
                    }
                    let first = segment
                        .delta(start)
                        .ok_or_else(|| "invalid giant tokenizer row delta".to_owned())?;
                    let mut min = first;
                    let mut max = first;
                    for index in start + 1..end {
                        let delta = segment
                            .delta(index)
                            .ok_or_else(|| "invalid giant tokenizer row delta".to_owned())?;
                        min = min.min(delta);
                        max = max.max(delta);
                    }
                    min_delta[row] = min;
                    max_delta[row] = max;
                }
            }
            if let Some(started) = delta_validate_started {
                eprintln!("[glrmask/profile][tks3] segment={} delta_extrema_ms={:.3}", state_offset, started.elapsed().as_secs_f64() * 1000.0);
            }
            let row_validate_started = profile.then(std::time::Instant::now);
            let validate_local = |local: usize| -> bool {
                let Some(row) = segment.row_ids.get(local) else {
                    return false;
                };
                if row >= row_count {
                    return false;
                }
                let lo = local as i64 + min_delta[row] as i64;
                let hi = local as i64 + max_delta[row] as i64;
                lo >= 0 && hi < segment_state_count as i64
            };
            let rows_valid = if segment_state_count as usize >= 100_000
                && rayon::current_num_threads() > 1
            {
                (0..segment_state_count as usize)
                    .into_par_iter()
                    .all(validate_local)
            } else {
                (0..segment_state_count as usize).all(validate_local)
            };
            if !rows_valid {
                return Err("giant tokenizer transition row or target out of range".to_owned());
            }
            if let Some(started) = row_validate_started {
                eprintln!("[glrmask/profile][tks3] segment={} row_ids_validate_ms={:.3}", state_offset, started.elapsed().as_secs_f64() * 1000.0);
            }
            segment_expanded += segment_expanded_count;
            packed_segments.push(segment);
            if let Some(started) = segment_started {
                eprintln!("[glrmask/profile][tks3] segment={} total_segment_ms={:.3}", state_offset, started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        if expected_state as usize != state_count
            || residual_transition_count + segment_expanded != expanded_transition_count
            || pos != input.len()
        {
            return Err("invalid giant tokenizer coverage or trailing bytes".to_owned());
        }
        let mut dfa = DFA::new(prefix_state_count);
        dfa.ensure_group_capacity(terminal_count);
        for (terminal, group) in groups.into_iter().enumerate() {
            dfa.set_group_u8set(terminal as u32, group);
        }
        for state in 0..prefix_state_count as u32 {
            dfa.overwrite_state_metadata(
                state,
                metadata.finalizers(state).unwrap().clone(),
                metadata.futures(state).unwrap().clone(),
            );
            for &target in metadata.epsilon_targets(state) {
                dfa.add_epsilon_transition(state, target);
            }
        }
        let prefix_transitions = PackedRuntimeTransitions {
            byte_offsets: Arc::from(prefix_offsets.into_boxed_slice()),
            bytes: PackedRuntimeBytes::Owned(prefix_bytes),
            targets: PackedRuntimeTargets::U32(Arc::from(prefix_targets.into_boxed_slice())),
        };
        let transition_count_cache = OnceLock::new();
        let _ = transition_count_cache.set(expanded_transition_count);
        let scalar_deterministic_dispatch_cache = OnceLock::new();
        if flags & HUGE_WIRE_FLAG_SCALAR_DETERMINISTIC_DISPATCH != 0 {
            let _ = scalar_deterministic_dispatch_cache.set(true);
        }
        let tokenizer = Tokenizer {
            dfa,
            num_terminals,
            packed_runtime_transitions: Some(Arc::new(prefix_transitions)),
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: Some(metadata),
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from(packed_segments.into_boxed_slice()),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: None,
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache,
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache,
            sorted_dispatch_roots_cache: OnceLock::new(),
            state_first_bytes_cache: OnceLock::new(),
        };
        if let Some(started) = huge_started {
            eprintln!("[glrmask/profile][tks3] total_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(tokenizer)
    }

    pub fn from_fast_bytes(input: &[u8]) -> Result<Tokenizer, String> {
        from_fast_bytes_impl(input, None)
    }

    /// Decode a current tokenizer section while retaining transition payloads
    /// directly in the owned constraint artifact. The caller must supply the
    /// exact byte offset of `input` inside `backing`; this is validated before
    /// any backed view is installed.
    pub fn from_fast_bytes_backed(
        input: &[u8],
        backing: Arc<Vec<u8>>,
        section_start: usize,
    ) -> Result<Tokenizer, String> {
        let section_end = section_start
            .checked_add(input.len())
            .ok_or_else(|| "fast tokenizer backing range overflow".to_owned())?;
        let backed = backing
            .get(section_start..section_end)
            .ok_or_else(|| "fast tokenizer section is outside artifact backing".to_owned())?;
        if backed.as_ptr() != input.as_ptr() || backed.len() != input.len() {
            return Err("fast tokenizer section does not match artifact backing".to_owned());
        }
        from_fast_bytes_impl(input, Some((backing, section_start)))
    }

    fn from_fast_bytes_impl(
        input: &[u8],
        backing: Option<(Arc<Vec<u8>>, usize)>,
    ) -> Result<Tokenizer, String> {
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        if input.starts_with(HUGE_WIRE_MAGIC) {
            return from_huge_bytes(input, backing);
        }
        if input.starts_with(PACKED_WIRE_MAGIC) {
            return from_packed_bytes(input);
        }
        if input.starts_with(SEGMENT_WIRE_MAGIC) {
            return from_segment_bytes(input);
        }
        let tkf3 = input.starts_with(b"TKF3");
        let tkf2 = input.starts_with(b"TKF2");
        if input.len() < 28 || (!tkf3 && !tkf2 && !input.starts_with(b"TKF1")) {
            return Err("invalid fast tokenizer header".to_owned());
        }
        let mut pos = 4usize;
        let take_u32 = |input: &[u8], pos: &mut usize| -> Result<u32, String> {
            let end = pos.checked_add(4).ok_or_else(|| "fast tokenizer offset overflow".to_owned())?;
            let bytes = input.get(*pos..end).ok_or_else(|| "truncated fast tokenizer".to_owned())?;
            *pos = end;
            Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
        };
        let num_terminals = take_u32(input, &mut pos)?;
        let state_count = take_u32(input, &mut pos)? as usize;
        let transition_count = take_u32(input, &mut pos)? as usize;
        let epsilon_count = take_u32(input, &mut pos)? as usize;
        let finalizer_count = take_u32(input, &mut pos)? as usize;
        let future_count = take_u32(input, &mut pos)? as usize;
        if state_count == 0 {
            return Err("fast tokenizer has no states".to_owned());
        }
        let mut fast_wire_flags = 0u8;
        let (state_id_width, terminal_id_width) = if tkf3 || tkf2 {
            let state_width = *input
                .get(pos)
                .ok_or_else(|| "truncated fast tokenizer state-id width".to_owned())?
                as usize;
            let terminal_width = *input
                .get(pos + 1)
                .ok_or_else(|| "truncated fast tokenizer terminal-id width".to_owned())?
                as usize;
            let flags = *input
                .get(pos + 2)
                .ok_or_else(|| "truncated fast tokenizer flags".to_owned())?;
            let reserved = *input
                .get(pos + 3)
                .ok_or_else(|| "truncated fast tokenizer reserved byte".to_owned())?;
            pos += 4;
            let invalid_flags = if tkf3 {
                flags & !FAST_WIRE_KNOWN_FLAGS != 0
            } else {
                flags != 0
            };
            if !matches!(state_width, 2 | 4)
                || !matches!(terminal_width, 2 | 4)
                || invalid_flags
                || reserved != 0
            {
                return Err("invalid fast tokenizer id widths or flags".to_owned());
            }
            fast_wire_flags = flags;
            if state_width == 2 && state_count > u16::MAX as usize + 1 {
                return Err("fast tokenizer u16 state ids cannot address all states".to_owned());
            }
            if terminal_width == 2 && num_terminals as usize > u16::MAX as usize + 1 {
                return Err("fast tokenizer u16 terminal ids cannot address all terminals".to_owned());
            }
            (state_width, terminal_width)
        } else {
            (4usize, 4usize)
        };
        let dfa_alloc_started = profile.then(std::time::Instant::now);
        let mut group_id_to_u8set = Vec::with_capacity(num_terminals as usize);
        for _ in 0..num_terminals as usize {
            let mut words = [0u64; 4];
            for word in &mut words {
                let end = pos.checked_add(8).ok_or_else(|| "fast tokenizer offset overflow".to_owned())?;
                let bytes = input.get(pos..end).ok_or_else(|| "truncated fast tokenizer groups".to_owned())?;
                *word = u64::from_le_bytes(bytes.try_into().unwrap());
                pos = end;
            }
            group_id_to_u8set.push(U8Set::from_words(words));
        }
        let dfa_alloc_groups_ms = dfa_alloc_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let read_u32_vec = |input: &[u8], pos: &mut usize, count: usize| -> Result<Vec<u32>, String> {
            let bytes_len = count.checked_mul(4).ok_or_else(|| "fast tokenizer vector overflow".to_owned())?;
            let end = pos.checked_add(bytes_len).ok_or_else(|| "fast tokenizer offset overflow".to_owned())?;
            let bytes = input.get(*pos..end).ok_or_else(|| "truncated fast tokenizer vector".to_owned())?;
            let mut out = Vec::<u32>::with_capacity(count);
            if cfg!(target_endian = "little") {
                unsafe {
                    out.set_len(count);
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().cast::<u8>(), bytes_len);
                }
            } else {
                out.extend(bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())));
            }
            *pos = end;
            Ok(out)
        };
        let read_u16_as_u32_vec =
            |input: &[u8], pos: &mut usize, count: usize| -> Result<Vec<u32>, String> {
            let bytes_len = count
                .checked_mul(2)
                .ok_or_else(|| "fast tokenizer u16 vector overflow".to_owned())?;
            let end = pos
                .checked_add(bytes_len)
                .ok_or_else(|| "fast tokenizer offset overflow".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast tokenizer u16 vector".to_owned())?;
            let mut out = Vec::<u32>::with_capacity(count);
            out.extend(
                bytes
                    .chunks_exact(2)
                    .map(|b| u32::from(u16::from_le_bytes([b[0], b[1]]))),
            );
            *pos = end;
            Ok(out)
        };
        let read_ids_as_u32 =
            |input: &[u8], pos: &mut usize, count: usize, width: usize| -> Result<Vec<u32>, String> {
                match width {
                    2 => read_u16_as_u32_vec(input, pos, count),
                    4 => read_u32_vec(input, pos, count),
                    _ => Err("invalid fast tokenizer id width".to_owned()),
                }
            };
        let transitions_started = profile.then(std::time::Instant::now);
        let transition_offsets = read_u32_vec(input, &mut pos, state_count + 1)?;
        if transition_offsets.first().copied() != Some(0)
            || transition_offsets.last().copied() != Some(transition_count as u32)
            || transition_offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err("invalid fast tokenizer transition offsets".to_owned());
        }
        let transition_bytes_start = pos;
        let transition_bytes_end = pos
            .checked_add(transition_count)
            .ok_or_else(|| "fast tokenizer offset overflow".to_owned())?;
        let transition_bytes_slice = input
            .get(pos..transition_bytes_end)
            .ok_or_else(|| "truncated fast tokenizer transition bytes".to_owned())?;
        let transition_bytes = if let Some((artifact, section_start)) = &backing {
            PackedRuntimeBytes::Backed {
                backing: Arc::clone(artifact),
                start: section_start + transition_bytes_start,
                len: transition_count,
            }
        } else {
            PackedRuntimeBytes::Owned(Arc::from(transition_bytes_slice))
        };
        pos = transition_bytes_end;
        let transition_targets = match state_id_width {
            2 => {
                let bytes_len = transition_count
                    .checked_mul(2)
                    .ok_or_else(|| "fast tokenizer target vector overflow".to_owned())?;
                let start = pos;
                let end = pos
                    .checked_add(bytes_len)
                    .ok_or_else(|| "fast tokenizer target offset overflow".to_owned())?;
                let bytes = input
                    .get(start..end)
                    .ok_or_else(|| "truncated fast tokenizer targets".to_owned())?;
                if !u16_values_all_below(bytes, state_count) {
                    return Err("fast tokenizer transition target out of range".to_owned());
                }
                pos = end;
                if let Some((artifact, section_start)) = &backing {
                    PackedRuntimeTargets::BackedU16 {
                        backing: Arc::clone(artifact),
                        start: section_start + start,
                        len: transition_count,
                    }
                } else {
                    let mut targets = Vec::with_capacity(transition_count);
                    targets.extend(
                        bytes
                            .chunks_exact(2)
                            .map(|word| u16::from_le_bytes([word[0], word[1]])),
                    );
                    PackedRuntimeTargets::U16(Arc::from(targets.into_boxed_slice()))
                }
            }
            4 => {
                let bytes_len = transition_count
                    .checked_mul(4)
                    .ok_or_else(|| "fast tokenizer target vector overflow".to_owned())?;
                let start = pos;
                let end = pos
                    .checked_add(bytes_len)
                    .ok_or_else(|| "fast tokenizer target offset overflow".to_owned())?;
                let bytes = input
                    .get(start..end)
                    .ok_or_else(|| "truncated fast tokenizer targets".to_owned())?;
                let validate_chunk = |chunk: &[u8]| {
                    debug_assert_eq!(chunk.len() % 4, 0);
                    chunk.chunks_exact(4).all(|word| {
                        (u32::from_le_bytes([word[0], word[1], word[2], word[3]]) as usize)
                            < state_count
                    })
                };
                let targets_valid = if transition_count >= 100_000
                    && rayon::current_num_threads() > 1
                {
                    const WORDS_PER_CHUNK: usize = 262_144;
                    bytes
                        .par_chunks(WORDS_PER_CHUNK * 4)
                        .all(validate_chunk)
                } else {
                    validate_chunk(bytes)
                };
                if !targets_valid {
                    return Err("fast tokenizer transition target out of range".to_owned());
                }
                pos = end;
                if let Some((artifact, section_start)) = &backing {
                    PackedRuntimeTargets::BackedU32 {
                        backing: Arc::clone(artifact),
                        start: section_start + start,
                        len: transition_count,
                    }
                } else {
                    let mut targets = Vec::with_capacity(transition_count);
                    targets.extend(bytes.chunks_exact(4).map(|word| {
                        u32::from_le_bytes([word[0], word[1], word[2], word[3]])
                    }));
                    PackedRuntimeTargets::U32(Arc::from(targets.into_boxed_slice()))
                }
            }
            _ => return Err("invalid fast tokenizer state-id width".to_owned()),
        };
        let transitions_ms = transitions_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        if tkf3 {
            let metadata_started = profile.then(std::time::Instant::now);
            let metadata: FastPackedMetadataArtifact = bincode::deserialize(&input[pos..])
                .map_err(|err| format!("invalid TKF3 metadata: {err}"))?;
            if metadata.finalizer_row_ids.len() != state_count
                || metadata.future_row_ids.len() != state_count
                || metadata.epsilon_offsets.len() != metadata.epsilon_states.len() + 1
                || metadata.epsilon_offsets.first().copied() != Some(0)
                || metadata.epsilon_offsets.last().copied().map(|value| value as usize)
                    != Some(metadata.epsilon_targets.len())
                || metadata.epsilon_offsets.windows(2).any(|pair| pair[0] > pair[1])
                || metadata.epsilon_states.windows(2).any(|pair| pair[0] >= pair[1])
                || metadata.epsilon_states.iter().any(|&state| state as usize >= state_count)
                || metadata.epsilon_targets.iter().any(|&state| state as usize >= state_count)
                || metadata.finalizer_row_ids.iter().any(|&row| row as usize >= metadata.finalizer_rows.len())
                || metadata.future_row_ids.iter().any(|&row| row as usize >= metadata.future_rows.len())
                || metadata
                    .finalizer_rows
                    .iter()
                    .chain(&metadata.future_rows)
                    .any(|row| {
                        row.windows(2).any(|pair| pair[0] >= pair[1])
                            || row.iter().any(|&terminal| terminal >= num_terminals)
                    })
            {
                return Err("invalid TKF3 packed metadata".to_owned());
            }
            let build_rows = |rows: Vec<Box<[u32]>>| {
                rows.into_iter()
                    .map(|row| {
                        let mut bits = BitSet::new(num_terminals as usize);
                        for terminal in row.iter().copied() {
                            bits.set(terminal as usize);
                        }
                        bits
                    })
                    .collect::<Vec<_>>()
            };
            let finalizer_rows = build_rows(metadata.finalizer_rows);
            let finalizer_lists = finalizer_rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|terminal| terminal as TerminalID)
                        .collect::<Vec<_>>()
                        .into_boxed_slice()
                })
                .collect::<Vec<_>>();
            let future_rows = build_rows(metadata.future_rows);
            let packed_metadata = Arc::new(PackedTokenizerMetadata {
                state_count: state_count as u32,
                finalizer_row_ids: packed_row_ids_from_u32(
                    metadata.finalizer_row_ids,
                    finalizer_rows.len(),
                ),
                finalizer_rows: Arc::from(finalizer_rows.into_boxed_slice()),
                finalizer_lists: Arc::from(finalizer_lists.into_boxed_slice()),
                future_row_ids: packed_row_ids_from_u32(
                    metadata.future_row_ids,
                    future_rows.len(),
                ),
                future_rows: Arc::from(future_rows.into_boxed_slice()),
                epsilon_states: Arc::from(metadata.epsilon_states.into_boxed_slice()),
                epsilon_offsets: Arc::from(metadata.epsilon_offsets.into_boxed_slice()),
                epsilon_targets: Arc::from(metadata.epsilon_targets.into_boxed_slice()),
            });
            let mut dfa = DFA::new(1);
            dfa.ensure_group_capacity(num_terminals as usize);
            for (terminal, group) in group_id_to_u8set.into_iter().enumerate() {
                dfa.set_group_u8set(terminal as u32, group);
            }
            if let Some(started) = metadata_started {
                eprintln!(
                    "[glrmask/profile][tokenizer_tkf3_decode] states={} transitions={} final_rows={} future_rows={} metadata_ms={:.3} total_ms={:.3}",
                    state_count,
                    transition_count,
                    packed_metadata.finalizer_rows.len(),
                    packed_metadata.future_rows.len(),
                    started.elapsed().as_secs_f64() * 1000.0,
                    total_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                );
            }
            let scalar_deterministic_dispatch_cache = OnceLock::new();
            if fast_wire_flags & FAST_WIRE_FLAG_SCALAR_DETERMINISTIC_DISPATCH != 0 {
                let _ = scalar_deterministic_dispatch_cache.set(true);
            }
            return Ok(Tokenizer {
                dfa,
                num_terminals,
                packed_runtime_transitions: Some(Arc::new(PackedRuntimeTransitions {
                    byte_offsets: Arc::from(transition_offsets.into_boxed_slice()),
                    bytes: transition_bytes,
                    targets: transition_targets,
                })),
                packed_runtime_transition_segments: Arc::from([]),
                compressed_transition_segments: Arc::from([]),
                packed_runtime_metadata: Some(packed_metadata),
                packed_runtime_metadata_segments: Arc::from([]),
                packed_compressed_transition_segments: Arc::from([]),
                virtual_unit_repeat: None,
                virtual_repeat_intersections: Vec::new(),
                virtual_residuals: Vec::new(),
                exprs: None,
                terminal_residual_coordinates: None,
                singleton_epsilon_closures: OnceLock::new(),
                matched_terminals_cache: OnceLock::new(),
                initial_byte_frontiers: OnceLock::new(),
                all_self_loop_bytes_cache: OnceLock::new(),
                transition_count_cache: OnceLock::new(),
                forced_minimized_state_count_cache: OnceLock::new(),
                scalar_deterministic_dispatch_cache,
                sorted_dispatch_roots_cache: OnceLock::new(),
                state_first_bytes_cache: OnceLock::new(),
            });
        }

        let metadata_wire_started = profile.then(std::time::Instant::now);
        let mut read_state_values =
            |count: usize, width: usize| -> Result<(Vec<u32>, Vec<u32>), String> {
            let offsets = read_u32_vec(input, &mut pos, state_count + 1)?;
            if offsets.first().copied() != Some(0)
                || offsets.last().copied() != Some(count as u32)
                || offsets.windows(2).any(|w| w[0] > w[1])
            {
                return Err("invalid fast tokenizer metadata offsets".to_owned());
            }
            let values = read_ids_as_u32(input, &mut pos, count, width)?;
            Ok((offsets, values))
        };
        let (epsilon_offsets, epsilon_targets) = read_state_values(epsilon_count, state_id_width)?;
        let (finalizer_offsets, finalizers) =
            read_state_values(finalizer_count, terminal_id_width)?;
        let (future_offsets, futures) = read_state_values(future_count, terminal_id_width)?;
        let metadata_wire_ms = metadata_wire_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if pos != input.len() {
            return Err("trailing bytes in fast tokenizer".to_owned());
        }
        if epsilon_targets.iter().any(|&target| target as usize >= state_count) {
            return Err("fast tokenizer epsilon target out of range".to_owned());
        }
        if finalizers
            .iter()
            .chain(&futures)
            .any(|&terminal| terminal >= num_terminals)
        {
            return Err("fast tokenizer terminal id out of range".to_owned());
        }
        let metadata_build_started = profile.then(std::time::Instant::now);
        // Current fast artifacts already carry sparse per-state observation
        // metadata. Keep that information in the runtime's compact metadata
        // sidecar instead of eagerly allocating one full DFAState (including
        // two terminal bitsets) for every state. Byte transitions already live
        // in PackedRuntimeTransitions, so the structural DFA only needs the
        // reset-state stub and terminal byte-group mapping until a later
        // composition/structural mutation explicitly asks to materialize it.
        let pack_rows = |offsets: &[u32], values: &[u32]| {
            let mut row_map = FxHashMap::<Box<[u32]>, u32>::default();
            let mut rows = Vec::<BitSet>::new();
            let mut row_ids = Vec::<u32>::with_capacity(state_count);
            for state in 0..state_count {
                let start = offsets[state] as usize;
                let end = offsets[state + 1] as usize;
                let sparse = &values[start..end];
                let row = if let Some(&row) = row_map.get(sparse) {
                    row
                } else {
                    let row = rows.len() as u32;
                    let mut bits = BitSet::new(num_terminals as usize);
                    for &terminal in sparse {
                        bits.set(terminal as usize);
                    }
                    rows.push(bits);
                    row_map.insert(sparse.into(), row);
                    row
                };
                row_ids.push(row);
            }
            (rows, row_ids)
        };
        let (finalizer_rows, finalizer_row_ids) = pack_rows(&finalizer_offsets, &finalizers);
        let finalizer_lists = finalizer_rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|terminal| terminal as TerminalID)
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>();
        let (future_rows, future_row_ids) = pack_rows(&future_offsets, &futures);
        let mut epsilon_states = Vec::<u32>::new();
        let mut packed_epsilon_offsets = vec![0u32];
        for state in 0..state_count {
            if epsilon_offsets[state] == epsilon_offsets[state + 1] {
                continue;
            }
            epsilon_states.push(state as u32);
            packed_epsilon_offsets.push(epsilon_offsets[state + 1]);
        }
        let packed_metadata = Arc::new(PackedTokenizerMetadata {
            state_count: state_count as u32,
            finalizer_row_ids: packed_row_ids_from_u32(finalizer_row_ids, finalizer_rows.len()),
            finalizer_rows: Arc::from(finalizer_rows.into_boxed_slice()),
            finalizer_lists: Arc::from(finalizer_lists.into_boxed_slice()),
            future_row_ids: packed_row_ids_from_u32(future_row_ids, future_rows.len()),
            future_rows: Arc::from(future_rows.into_boxed_slice()),
            epsilon_states: Arc::from(epsilon_states.into_boxed_slice()),
            epsilon_offsets: Arc::from(packed_epsilon_offsets.into_boxed_slice()),
            epsilon_targets: Arc::from(epsilon_targets.into_boxed_slice()),
        });
        let mut dfa = DFA::new(1);
        dfa.ensure_group_capacity(num_terminals as usize);
        for (terminal, group) in group_id_to_u8set.into_iter().enumerate() {
            dfa.set_group_u8set(terminal as u32, group);
        }
        let metadata_build_ms = metadata_build_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let singleton_epsilon_closures = OnceLock::new();
        if let Some(total_started) = total_started {
            let epsilon_rows = epsilon_offsets
                .windows(2)
                .filter(|row| row[0] != row[1])
                .count();
            let finalizer_rows = finalizer_offsets
                .windows(2)
                .filter(|row| row[0] != row[1])
                .count();
            let future_rows = future_offsets
                .windows(2)
                .filter(|row| row[0] != row[1])
                .count();
            eprintln!(
                "[glrmask/profile][tokenizer_fast_decode] states={} transitions={} epsilon={} epsilon_rows={} finalizers={} finalizer_rows={} futures={} future_rows={} dfa_groups_ms={:.3} transitions_ms={:.3} metadata_wire_ms={:.3} metadata_build_ms={:.3} total_ms={:.3}",
                state_count,
                transition_count,
                epsilon_count,
                epsilon_rows,
                finalizer_count,
                finalizer_rows,
                future_count,
                future_rows,
                dfa_alloc_groups_ms,
                transitions_ms,
                metadata_wire_ms,
                metadata_build_ms,
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(Tokenizer {
            dfa,
            num_terminals,
            packed_runtime_transitions: Some(Arc::new(PackedRuntimeTransitions {
                byte_offsets: Arc::from(transition_offsets.into_boxed_slice()),
                bytes: transition_bytes,
                targets: transition_targets,
            })),
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: Some(packed_metadata),
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: None,
            terminal_residual_coordinates: None,
            singleton_epsilon_closures,
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
            sorted_dispatch_roots_cache: OnceLock::new(),
            state_first_bytes_cache: OnceLock::new(),
        })
    }
}


/// Compact tokenizer wire form for dynamic artifacts.
///
/// The ordinary historical `Tokenizer` serializer stores one dense terminal
/// bitset for every lexer state.  That is appropriate for backwards-compatible
/// static artifacts, but source-specialized grammars can have hundreds of
/// thousands of states and thousands of terminals while setting only a handful
/// of metadata bits per state.  This wire form stores exactly the set bits and
/// the actual graph edges.
pub mod compact_artifact_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    #[derive(Serialize, Deserialize)]
    struct CompactTokenizerArtifact {
        num_terminals: u32,
        state_count: u32,
        group_id_to_u8set: Vec<U8Set>,
        transition_offsets: Vec<u32>,
        transition_bytes: Vec<u8>,
        transition_targets: Vec<u32>,
        epsilon_offsets: Vec<u32>,
        epsilon_targets: Vec<u32>,
        finalizer_offsets: Vec<u32>,
        finalizers: Vec<u32>,
        future_offsets: Vec<u32>,
        futures: Vec<u32>,
        compressed_transition_segments: Vec<CompressedTransitionSegment>,
    }

    fn offset(value: usize, label: &str) -> Result<u32, String> {
        u32::try_from(value).map_err(|_| format!("{label} count exceeds u32"))
    }

    fn validate_offsets(
        offsets: &[u32],
        state_count: usize,
        entry_count: usize,
        label: &str,
    ) -> Result<(), String> {
        if offsets.len() != state_count.saturating_add(1) {
            return Err(format!(
                "{label} offsets have length {}, expected {}",
                offsets.len(),
                state_count.saturating_add(1),
            ));
        }
        if offsets.first().copied() != Some(0) {
            return Err(format!("{label} offsets do not start at zero"));
        }
        if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(format!("{label} offsets are not monotonic"));
        }
        if offsets.last().copied().map(|value| value as usize) != Some(entry_count) {
            return Err(format!(
                "{label} offsets end at {:?}, expected {entry_count}",
                offsets.last(),
            ));
        }
        Ok(())
    }

    pub fn serialize<S>(tokenizer: &Tokenizer, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error as _;

        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);

        let state_count = tokenizer.dfa.num_states();
        let mut transition_offsets = Vec::with_capacity(state_count + 1);
        let mut transition_bytes = Vec::new();
        let mut transition_targets = Vec::new();
        let mut epsilon_offsets = Vec::with_capacity(state_count + 1);
        let mut epsilon_targets = Vec::new();
        let mut finalizer_offsets = Vec::with_capacity(state_count + 1);
        let mut finalizers = Vec::new();
        let mut future_offsets = Vec::with_capacity(state_count + 1);
        let mut futures = Vec::new();
        transition_offsets.push(0);
        epsilon_offsets.push(0);
        finalizer_offsets.push(0);
        future_offsets.push(0);

        for state in tokenizer.dfa.states() {
            for (byte, &target) in state.transitions.iter() {
                transition_bytes.push(byte);
                transition_targets.push(target);
            }
            transition_offsets.push(
                offset(transition_bytes.len(), "tokenizer transition")
                    .map_err(S::Error::custom)?,
            );

            epsilon_targets.extend_from_slice(&state.epsilon_transitions);
            epsilon_offsets.push(
                offset(epsilon_targets.len(), "tokenizer epsilon transition")
                    .map_err(S::Error::custom)?,
            );

            finalizers.extend(state.finalizers.iter().map(|terminal| terminal as u32));
            finalizer_offsets.push(
                offset(finalizers.len(), "tokenizer finalizer")
                    .map_err(S::Error::custom)?,
            );

            futures.extend(
                state
                    .possible_future_group_ids
                    .iter()
                    .map(|terminal| terminal as u32),
            );
            future_offsets.push(
                offset(futures.len(), "tokenizer future")
                    .map_err(S::Error::custom)?,
            );
        }

        let group_id_to_u8set = (0..tokenizer.dfa.num_groups())
            .map(|group| *tokenizer.dfa.group_id_to_u8set(group as u32))
            .collect();
        CompactTokenizerArtifact {
            num_terminals: tokenizer.num_terminals,
            state_count: u32::try_from(state_count).map_err(S::Error::custom)?,
            group_id_to_u8set,
            transition_offsets,
            transition_bytes,
            transition_targets,
            epsilon_offsets,
            epsilon_targets,
            finalizer_offsets,
            finalizers,
            future_offsets,
            futures,
            compressed_transition_segments: tokenizer.compressed_transition_segments.to_vec(),
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Tokenizer, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error as _;

        let artifact = CompactTokenizerArtifact::deserialize(deserializer)?;
        let state_count = artifact.state_count as usize;
        let terminal_count = artifact.num_terminals as usize;
        if artifact.group_id_to_u8set.len() != terminal_count {
            return Err(D::Error::custom(format!(
                "compact tokenizer has {} group byte sets for {terminal_count} terminals",
                artifact.group_id_to_u8set.len(),
            )));
        }
        if artifact.transition_bytes.len() != artifact.transition_targets.len() {
            return Err(D::Error::custom(
                "compact tokenizer transition byte/target lengths differ",
            ));
        }
        validate_offsets(
            &artifact.transition_offsets,
            state_count,
            artifact.transition_targets.len(),
            "transition",
        )
        .map_err(D::Error::custom)?;
        validate_offsets(
            &artifact.epsilon_offsets,
            state_count,
            artifact.epsilon_targets.len(),
            "epsilon",
        )
        .map_err(D::Error::custom)?;
        validate_offsets(
            &artifact.finalizer_offsets,
            state_count,
            artifact.finalizers.len(),
            "finalizer",
        )
        .map_err(D::Error::custom)?;
        validate_offsets(
            &artifact.future_offsets,
            state_count,
            artifact.futures.len(),
            "future",
        )
        .map_err(D::Error::custom)?;

        for &target in artifact
            .transition_targets
            .iter()
            .chain(&artifact.epsilon_targets)
        {
            if target as usize >= state_count {
                return Err(D::Error::custom(format!(
                    "compact tokenizer transition target {target} is out of range for {state_count} states",
                )));
            }
        }
        for &terminal in artifact.finalizers.iter().chain(&artifact.futures) {
            if terminal as usize >= terminal_count {
                return Err(D::Error::custom(format!(
                    "compact tokenizer terminal {terminal} is out of range for {terminal_count} terminals",
                )));
            }
        }
        for segment in &artifact.compressed_transition_segments {
            let end = segment
                .state_offset
                .checked_add(segment.state_count)
                .ok_or_else(|| D::Error::custom("compressed tokenizer segment state range overflow"))?;
            if end as usize > state_count {
                return Err(D::Error::custom(format!(
                    "compressed tokenizer segment ends at {end}, beyond {state_count} states",
                )));
            }
        }

        let mut dfa = DFA::new(state_count);
        dfa.ensure_group_capacity(terminal_count);
        for (group, set) in artifact.group_id_to_u8set.into_iter().enumerate() {
            dfa.set_group_u8set(group as u32, set);
        }
        for state in 0..state_count {
            let start = artifact.transition_offsets[state] as usize;
            let end = artifact.transition_offsets[state + 1] as usize;
            let entries = artifact.transition_bytes[start..end]
                .iter()
                .copied()
                .zip(artifact.transition_targets[start..end].iter().copied())
                .collect();
            dfa.set_transitions_from_sorted_entries(state as u32, entries);
        }
        {
            let states = dfa.states_mut();
            for (state_index, state) in states.iter_mut().enumerate() {
                let start = artifact.epsilon_offsets[state_index] as usize;
                let end = artifact.epsilon_offsets[state_index + 1] as usize;
                state.epsilon_transitions = artifact.epsilon_targets[start..end].to_vec();

                let start = artifact.finalizer_offsets[state_index] as usize;
                let end = artifact.finalizer_offsets[state_index + 1] as usize;
                for &terminal in &artifact.finalizers[start..end] {
                    state.finalizers.set(terminal as usize);
                }

                let start = artifact.future_offsets[state_index] as usize;
                let end = artifact.future_offsets[state_index + 1] as usize;
                for &terminal in &artifact.futures[start..end] {
                    state.possible_future_group_ids.set(terminal as usize);
                }
            }
        }

        Ok(Tokenizer {
            dfa,
            num_terminals: artifact.num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from(
                artifact.compressed_transition_segments.into_boxed_slice(),
            ),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: None,
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
            sorted_dispatch_roots_cache: OnceLock::new(),
            state_first_bytes_cache: OnceLock::new(),
        })
    }
}

/// Packed tokenizer wire form used only by current static-constraint
/// artifacts. Runtime and compiler representations remain unchanged.
///
/// The byte labels of lexer transition rows are highly repetitive (large
/// schema tokenizers commonly have only ~100 distinct byte patterns across
/// thousands of states), while target ids are small. Store each byte pattern
/// once and encode the matching target sequence as varints. The old
/// `compact_artifact_serde` remains untouched because dynamic-constraint
/// artifacts use it as a persisted format.
mod packed_artifact_serde {
    use super::*;
    use rustc_hash::FxHashMap;
    use serde::{Deserializer, Serializer};

    #[derive(Serialize, Deserialize)]
    struct PackedTokenizerArtifact {
        num_terminals: u32,
        state_count: u32,
        group_id_to_u8set: Vec<U8Set>,
        transition_byte_rows: Vec<Vec<u8>>,
        transition_byte_row_ids: Vec<u32>,
        transition_target_offsets: Vec<u32>,
        transition_targets: Vec<u8>,
        epsilon_offsets: Vec<u32>,
        epsilon_targets: Vec<u32>,
        finalizer_offsets: Vec<u32>,
        finalizers: Vec<u32>,
        future_offsets: Vec<u32>,
        futures: Vec<u32>,
        compressed_transition_segments: Vec<CompressedTransitionSegment>,
    }

    #[inline]
    fn put_var_u32(out: &mut Vec<u8>, mut value: u32) {
        while value >= 0x80 {
            out.push((value as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }

    #[inline]
    fn put_var_i64(out: &mut Vec<u8>, value: i64) {
        let zigzag = ((value << 1) ^ (value >> 63)) as u64;
        let mut value = zigzag;
        while value >= 0x80 {
            out.push((value as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }

    #[inline]
    fn var_u32_len(mut value: u32) -> usize {
        let mut len = 1usize;
        while value >= 0x80 {
            value >>= 7;
            len += 1;
        }
        len
    }

    #[inline]
    fn var_i64_len(value: i64) -> usize {
        let mut value = ((value << 1) ^ (value >> 63)) as u64;
        let mut len = 1usize;
        while value >= 0x80 {
            value >>= 7;
            len += 1;
        }
        len
    }

    #[inline]
    fn take_var_u32(input: &[u8], pos: &mut usize) -> Result<u32, String> {
        let mut value = 0u32;
        let mut shift = 0u32;
        for _ in 0..5 {
            let byte = *input
                .get(*pos)
                .ok_or_else(|| "truncated packed tokenizer target".to_owned())?;
            *pos += 1;
            if shift == 28 && byte > 0x0f {
                return Err("overflowing packed tokenizer target".to_owned());
            }
            value |= ((byte & 0x7f) as u32) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
        }
        Err("overflowing packed tokenizer target".to_owned())
    }

    #[inline]
    fn take_var_i64(input: &[u8], pos: &mut usize) -> Result<i64, String> {
        let mut value = 0u64;
        let mut shift = 0u32;
        for index in 0..10 {
            let byte = *input
                .get(*pos)
                .ok_or_else(|| "truncated packed tokenizer target delta".to_owned())?;
            *pos += 1;
            if index == 9 && byte > 1 {
                return Err("overflowing packed tokenizer target delta".to_owned());
            }
            value |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(((value >> 1) as i64) ^ (-((value & 1) as i64)));
            }
            shift += 7;
        }
        Err("overflowing packed tokenizer target delta".to_owned())
    }

    fn encode_targets(targets: &[u32], out: &mut Vec<u8>) {
        let absolute_len = targets.iter().map(|&target| var_u32_len(target)).sum::<usize>();
        let mut previous = 0i64;
        let delta_len = targets
            .iter()
            .map(|&target| {
                let target = target as i64;
                let len = var_i64_len(target - previous);
                previous = target;
                len
            })
            .sum::<usize>();
        let use_delta = delta_len < absolute_len;
        out.push(u8::from(use_delta));
        if use_delta {
            let mut previous = 0i64;
            for &target in targets {
                let target = target as i64;
                put_var_i64(out, target - previous);
                previous = target;
            }
        } else {
            for &target in targets {
                put_var_u32(out, target);
            }
        }
    }

    fn decode_targets_into(
        body: &[u8],
        count: usize,
        state_count: usize,
        targets: &mut Vec<u32>,
    ) -> Result<(), String> {
        let (&mode, rest) = body
            .split_first()
            .ok_or_else(|| "missing packed tokenizer target mode".to_owned())?;
        let start_len = targets.len();
        let mut pos = 0usize;
        match mode {
            0 => {
                for _ in 0..count {
                    targets.push(take_var_u32(rest, &mut pos)?);
                }
            }
            1 => {
                let mut previous = 0i64;
                for _ in 0..count {
                    let delta = take_var_i64(rest, &mut pos)?;
                    let target = previous
                        .checked_add(delta)
                        .ok_or_else(|| "overflowing packed tokenizer target".to_owned())?;
                    let target = u32::try_from(target)
                        .map_err(|_| "invalid packed tokenizer target".to_owned())?;
                    targets.push(target);
                    previous = target as i64;
                }
            }
            _ => return Err("invalid packed tokenizer target mode".to_owned()),
        }
        if pos != rest.len() {
            return Err("trailing bytes in packed tokenizer target row".to_owned());
        }
        if targets[start_len..]
            .iter()
            .any(|&target| target as usize >= state_count)
        {
            return Err("packed tokenizer transition target is out of range".to_owned());
        }
        Ok(())
    }

    fn offset(value: usize, label: &str) -> Result<u32, String> {
        u32::try_from(value).map_err(|_| format!("{label} count exceeds u32"))
    }

    fn validate_offsets(
        offsets: &[u32],
        state_count: usize,
        entry_count: usize,
        label: &str,
    ) -> Result<(), String> {
        if offsets.len() != state_count.saturating_add(1) {
            return Err(format!(
                "{label} offsets have length {}, expected {}",
                offsets.len(),
                state_count.saturating_add(1),
            ));
        }
        if offsets.first().copied() != Some(0) {
            return Err(format!("{label} offsets do not start at zero"));
        }
        if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(format!("{label} offsets are not monotonic"));
        }
        if offsets.last().copied().map(|value| value as usize) != Some(entry_count) {
            return Err(format!(
                "{label} offsets end at {:?}, expected {entry_count}",
                offsets.last(),
            ));
        }
        Ok(())
    }

    pub fn serialize<S>(tokenizer: &Tokenizer, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error as _;

        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);

        // Current static artifacts should already have materialized rows, but
        // preserve exact behavior for the uncommon compressed-segment case.
        let materialized;
        let dfa = if tokenizer.compressed_transition_segments.is_empty() {
            &tokenizer.dfa
        } else {
            materialized = tokenizer.materialized_dfa();
            &materialized
        };
        let state_count = dfa.num_states();
        if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_RAW").is_some() {
            let started = std::time::Instant::now();
            let transition_count = dfa
                .states()
                .iter()
                .map(|state| state.transitions.len())
                .sum::<usize>();
            let mut raw = Vec::with_capacity(
                12usize
                    .saturating_add((state_count + 1).saturating_mul(4))
                    .saturating_add(transition_count.saturating_mul(5)),
            );
            raw.extend_from_slice(b"TRW1");
            raw.extend_from_slice(&(state_count as u32).to_le_bytes());
            raw.extend_from_slice(&(transition_count as u32).to_le_bytes());
            let mut end = 0u32;
            raw.extend_from_slice(&end.to_le_bytes());
            for state in dfa.states() {
                end = end.saturating_add(state.transitions.len() as u32);
                raw.extend_from_slice(&end.to_le_bytes());
            }
            for state in dfa.states() {
                for (byte, &target) in state.transitions.iter() {
                    raw.push(byte);
                    raw.extend_from_slice(&target.to_le_bytes());
                }
            }
            eprintln!(
                "[glrmask/profile][tokenizer_raw_candidate] states={} transitions={} bytes={} ms={:.3}",
                state_count,
                transition_count,
                raw.len(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        let mut byte_rows = Vec::<Vec<u8>>::new();
        let mut byte_row_ids = Vec::with_capacity(state_count);
        let mut byte_row_by_value = FxHashMap::<Vec<u8>, u32>::default();
        let mut target_offsets = Vec::with_capacity(state_count + 1);
        let mut target_bytes = Vec::<u8>::new();
        let mut epsilon_offsets = Vec::with_capacity(state_count + 1);
        let mut epsilon_targets = Vec::new();
        let mut finalizer_offsets = Vec::with_capacity(state_count + 1);
        let mut finalizers = Vec::new();
        let mut future_offsets = Vec::with_capacity(state_count + 1);
        let mut futures = Vec::new();
        target_offsets.push(0);
        epsilon_offsets.push(0);
        finalizer_offsets.push(0);
        future_offsets.push(0);

        for state in dfa.states() {
            let bytes = state.transitions.iter().map(|(byte, _)| byte).collect::<Vec<_>>();
            let byte_row = if let Some(&row) = byte_row_by_value.get(&bytes) {
                row
            } else {
                let row = u32::try_from(byte_rows.len()).map_err(S::Error::custom)?;
                byte_row_by_value.insert(bytes.clone(), row);
                byte_rows.push(bytes);
                row
            };
            byte_row_ids.push(byte_row);
            let targets = state.transitions.values().copied().collect::<Vec<_>>();
            encode_targets(&targets, &mut target_bytes);
            target_offsets.push(
                offset(target_bytes.len(), "tokenizer target byte").map_err(S::Error::custom)?,
            );

            epsilon_targets.extend_from_slice(&state.epsilon_transitions);
            epsilon_offsets.push(
                offset(epsilon_targets.len(), "tokenizer epsilon transition")
                    .map_err(S::Error::custom)?,
            );
            finalizers.extend(state.finalizers.iter().map(|terminal| terminal as u32));
            finalizer_offsets.push(
                offset(finalizers.len(), "tokenizer finalizer").map_err(S::Error::custom)?,
            );
            futures.extend(
                state
                    .possible_future_group_ids
                    .iter()
                    .map(|terminal| terminal as u32),
            );
            future_offsets.push(
                offset(futures.len(), "tokenizer future").map_err(S::Error::custom)?,
            );
        }

        let artifact = PackedTokenizerArtifact {
            num_terminals: tokenizer.num_terminals,
            state_count: u32::try_from(state_count).map_err(S::Error::custom)?,
            group_id_to_u8set: (0..dfa.num_groups())
                .map(|group| *dfa.group_id_to_u8set(group as u32))
                .collect(),
            transition_byte_rows: byte_rows,
            transition_byte_row_ids: byte_row_ids,
            transition_target_offsets: target_offsets,
            transition_targets: target_bytes,
            epsilon_offsets,
            epsilon_targets,
            finalizer_offsets,
            finalizers,
            future_offsets,
            futures,
            compressed_transition_segments: Vec::new(),
        };
        let result = artifact.serialize(serializer);
        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][tokenizer_encode] states={} transition_rows={} target_bytes={} ms={:.3}",
                state_count,
                artifact.transition_byte_rows.len(),
                artifact.transition_targets.len(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Tokenizer, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error as _;

        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let artifact_started = profile.then(std::time::Instant::now);
        let artifact = PackedTokenizerArtifact::deserialize(deserializer)?;
        let artifact_ms = artifact_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);
        let state_count = artifact.state_count as usize;
        let terminal_count = artifact.num_terminals as usize;
        if artifact.group_id_to_u8set.len() != terminal_count {
            return Err(D::Error::custom(format!(
                "packed tokenizer has {} group byte sets for {terminal_count} terminals",
                artifact.group_id_to_u8set.len(),
            )));
        }
        if artifact.transition_byte_row_ids.len() != state_count {
            return Err(D::Error::custom("packed tokenizer byte-row id count differs from state count"));
        }
        validate_offsets(
            &artifact.transition_target_offsets,
            state_count,
            artifact.transition_targets.len(),
            "transition target",
        )
        .map_err(D::Error::custom)?;
        validate_offsets(
            &artifact.epsilon_offsets,
            state_count,
            artifact.epsilon_targets.len(),
            "epsilon",
        )
        .map_err(D::Error::custom)?;
        validate_offsets(
            &artifact.finalizer_offsets,
            state_count,
            artifact.finalizers.len(),
            "finalizer",
        )
        .map_err(D::Error::custom)?;
        validate_offsets(
            &artifact.future_offsets,
            state_count,
            artifact.futures.len(),
            "future",
        )
        .map_err(D::Error::custom)?;
        if artifact
            .epsilon_targets
            .iter()
            .any(|&target| target as usize >= state_count)
        {
            return Err(D::Error::custom("packed tokenizer epsilon target is out of range"));
        }
        if artifact
            .finalizers
            .iter()
            .chain(&artifact.futures)
            .any(|&terminal| terminal as usize >= terminal_count)
        {
            return Err(D::Error::custom("packed tokenizer terminal is out of range"));
        }
        if artifact
            .transition_byte_rows
            .iter()
            .any(|bytes| bytes.windows(2).any(|pair| pair[0] >= pair[1]))
        {
            return Err(D::Error::custom(
                "packed tokenizer byte row is not sorted unique",
            ));
        }

        let transitions_started = profile.then(std::time::Instant::now);
        let mut target_count = 0usize;
        for &byte_row in &artifact.transition_byte_row_ids {
            target_count = target_count
                .checked_add(
                    artifact
                        .transition_byte_rows
                        .get(byte_row as usize)
                        .ok_or_else(|| D::Error::custom("packed tokenizer byte-row id is out of range"))?
                        .len(),
                )
                .ok_or_else(|| D::Error::custom("packed tokenizer target count overflow"))?;
        }
        let mut runtime_byte_offsets = Vec::with_capacity(state_count + 1);
        let mut runtime_bytes = Vec::with_capacity(target_count);
        let mut runtime_target_offsets = Vec::with_capacity(state_count + 1);
        let mut runtime_targets = Vec::with_capacity(target_count);
        runtime_byte_offsets.push(0u32);
        runtime_target_offsets.push(0u32);
        for state in 0..state_count {
            let byte_row = artifact.transition_byte_row_ids[state] as usize;
            let bytes = &artifact.transition_byte_rows[byte_row];
            let count = bytes.len();
            runtime_bytes.extend_from_slice(bytes);
            runtime_byte_offsets.push(
                u32::try_from(runtime_bytes.len())
                    .map_err(|_| D::Error::custom("packed tokenizer runtime byte count exceeds u32"))?,
            );
            let start = artifact.transition_target_offsets[state] as usize;
            let end = artifact.transition_target_offsets[state + 1] as usize;
            decode_targets_into(
                &artifact.transition_targets[start..end],
                count,
                state_count,
                &mut runtime_targets,
            )
            .map_err(D::Error::custom)?;
            runtime_target_offsets.push(
                u32::try_from(runtime_targets.len())
                    .map_err(|_| D::Error::custom("packed tokenizer runtime target count exceeds u32"))?,
            );
        }
        let transitions_ms = transitions_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);

        let dfa_started = profile.then(std::time::Instant::now);
        let mut dfa = DFA::new(state_count);
        dfa.ensure_group_capacity(terminal_count);
        for (group, set) in artifact.group_id_to_u8set.into_iter().enumerate() {
            dfa.set_group_u8set(group as u32, set);
        }
        {
            let states = dfa.states_mut();
            for (state_index, state) in states.iter_mut().enumerate() {
                let start = artifact.epsilon_offsets[state_index] as usize;
                let end = artifact.epsilon_offsets[state_index + 1] as usize;
                state.epsilon_transitions = artifact.epsilon_targets[start..end].to_vec();
                let start = artifact.finalizer_offsets[state_index] as usize;
                let end = artifact.finalizer_offsets[state_index + 1] as usize;
                for &terminal in &artifact.finalizers[start..end] {
                    state.finalizers.set(terminal as usize);
                }
                let start = artifact.future_offsets[state_index] as usize;
                let end = artifact.future_offsets[state_index + 1] as usize;
                for &terminal in &artifact.futures[start..end] {
                    state.possible_future_group_ids.set(terminal as usize);
                }
            }
        }
        let packed_runtime_transitions = Arc::new(PackedRuntimeTransitions {
            byte_offsets: Arc::from(runtime_byte_offsets.into_boxed_slice()),
            bytes: PackedRuntimeBytes::Owned(Arc::from(runtime_bytes.into_boxed_slice())),
            targets: PackedRuntimeTargets::U32(Arc::from(runtime_targets.into_boxed_slice())),
        });
        let dfa_ms = dfa_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);

        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][tokenizer_decode] states={} artifact_ms={artifact_ms:.3} transitions_ms={transitions_ms:.3} dfa_ms={dfa_ms:.3} total_ms={:.3}",
                state_count,
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }

        Ok(Tokenizer {
            dfa,
            num_terminals: artifact.num_terminals,
            packed_runtime_transitions: Some(packed_runtime_transitions),
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from(
                artifact.compressed_transition_segments.into_boxed_slice(),
            ),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: None,
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
            sorted_dispatch_roots_cache: OnceLock::new(),
            state_first_bytes_cache: OnceLock::new(),
        })
    }
}

impl CompressedTransitionSegment {
    #[inline]
    pub(super) fn contains_state(&self, state: u32) -> bool {
        state >= self.state_offset && state - self.state_offset < self.state_count
    }

    #[inline]
    fn local_transition(&self, local_state: u32, byte: u8) -> Option<u32> {
        let class = self.byte_to_class[byte as usize];
        let start = self.row_offsets[local_state as usize] as usize;
        let end = self.row_offsets[local_state as usize + 1] as usize;
        self.entries
            .class_slice(start, end)
            .binary_search(&class)
            .ok()
            .map(|index| self.entries.target(start + index))
    }

    #[inline]
    pub(super) fn transition(&self, state: u32, byte: u8) -> Option<u32> {
        self.local_transition(state - self.state_offset, byte)
            .map(|target| self.state_offset + target)
    }

    fn expanded_entries(&self, state: u32) -> Vec<(u8, u32)> {
        let local_state = state - self.state_offset;
        let start = self.row_offsets[local_state as usize] as usize;
        let end = self.row_offsets[local_state as usize + 1] as usize;
        let mut target_by_class = vec![u32::MAX; self.class_members.len()];
        let mut capacity = 0usize;
        for (class, target) in self.entries.iter_range(start, end) {
            target_by_class[class as usize] = target;
            capacity += self.class_members[class as usize].len();
        }
        let mut entries = Vec::with_capacity(capacity);
        for byte in 0u16..=255 {
            let class = self.byte_to_class[byte as usize] as usize;
            let target = target_by_class[class];
            if target != u32::MAX {
                entries.push((byte as u8, self.state_offset + target));
            }
        }
        entries
    }

    pub fn materialize_into_dfa(&self, dfa: &mut DFA) {
        for local_state in 0..self.state_count {
            let state = self.state_offset + local_state;
            dfa.set_transitions_from_sorted_entries(state, self.expanded_entries(state));
        }
    }

    pub(super) fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) {
        row.fill(u32::MAX);
        let local_state = state - self.state_offset;
        let start = self.row_offsets[local_state as usize] as usize;
        let end = self.row_offsets[local_state as usize + 1] as usize;
        for (class, target) in self.entries.iter_range(start, end) {
            let target = self.state_offset + target;
            for &byte in self.class_members[class as usize].iter() {
                row[byte as usize] = target;
            }
        }
    }

    pub(super) fn transition_count(&self, state: u32) -> usize {
        let local_state = state - self.state_offset;
        let start = self.row_offsets[local_state as usize] as usize;
        let end = self.row_offsets[local_state as usize + 1] as usize;
        self.entries
            .class_slice(start, end)
            .iter()
            .map(|class| self.class_members[*class as usize].len())
            .sum()
    }

    pub(super) fn transitions_satisfy(
        &self,
        state: u32,
        mut predicate: impl FnMut(u8, u32) -> bool,
    ) -> bool {
        let local_state = state - self.state_offset;
        let start = self.row_offsets[local_state as usize] as usize;
        let end = self.row_offsets[local_state as usize + 1] as usize;
        for (class, target) in self.entries.iter_range(start, end) {
            let target = self.state_offset + target;
            for &byte in self.class_members[class as usize].iter() {
                if !predicate(byte, target) {
                    return false;
                }
            }
        }
        true
    }
}

enum TokenizerTransitionsIterInner<'a> {
    Dense(crate::ds::char_transitions::CharTransitionsIter<'a, u32>),
    Packed {
        bytes: &'a [u8],
        targets: PackedRuntimeTargetSlice<'a>,
        next: usize,
    },
    PackedSegment {
        bytes: &'a [u8],
        targets: PackedRuntimeTargetSlice<'a>,
        target_offset: u32,
        next: usize,
    },
    Compressed {
        segment: &'a CompressedTransitionSegment,
        state: u32,
        next_byte: u16,
    },
    PackedCompressed {
        segment: &'a PackedCompressedTransitionSegment,
        state: u32,
        next_byte: u16,
    },
    Virtual {
        bytes: crate::ds::u8set::U8SetIter,
        target: u32,
    },
    VirtualProduct(std::vec::IntoIter<(u8, u32)>),
    Empty,
}

pub struct TokenizerTransitionsIter<'a> {
    inner: TokenizerTransitionsIterInner<'a>,
}

impl Iterator for TokenizerTransitionsIter<'_> {
    type Item = (u8, u32);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            TokenizerTransitionsIterInner::Dense(iter) => {
                iter.next().map(|(byte, target)| (byte, *target))
            }
            TokenizerTransitionsIterInner::Packed {
                bytes,
                targets,
                next,
            } => {
                let index = *next;
                let byte = *bytes.get(index)?;
                let target = targets.get(index)?;
                *next += 1;
                Some((byte, target))
            }
            TokenizerTransitionsIterInner::PackedSegment {
                bytes,
                targets,
                target_offset,
                next,
            } => {
                let index = *next;
                let byte = *bytes.get(index)?;
                let target = targets.get(index)?.checked_add(*target_offset)?;
                *next += 1;
                Some((byte, target))
            }
            TokenizerTransitionsIterInner::Compressed {
                segment,
                state,
                next_byte,
            } => {
                while *next_byte <= 255 {
                    let byte = *next_byte as u8;
                    *next_byte += 1;
                    if let Some(target) = segment.transition(*state, byte) {
                        return Some((byte, target));
                    }
                }
                None
            }
            TokenizerTransitionsIterInner::PackedCompressed {
                segment,
                state,
                next_byte,
            } => {
                while *next_byte <= 255 {
                    let byte = *next_byte as u8;
                    *next_byte += 1;
                    if let Some(target) = segment.transition(*state, byte) {
                        return Some((byte, target));
                    }
                }
                None
            }
            TokenizerTransitionsIterInner::Virtual { bytes, target } => {
                bytes.next().map(|byte| (byte, *target))
            }
            TokenizerTransitionsIterInner::VirtualProduct(iter) => iter.next(),
            TokenizerTransitionsIterInner::Empty => None,
        }
    }


    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            TokenizerTransitionsIterInner::Dense(iter) => iter.size_hint(),
            TokenizerTransitionsIterInner::Packed { bytes, next, .. }
            | TokenizerTransitionsIterInner::PackedSegment { bytes, next, .. } => {
                let count = bytes.len().saturating_sub(*next);
                (count, Some(count))
            }
            TokenizerTransitionsIterInner::Compressed { segment, state, .. } => {
                let count = segment.transition_count(*state);
                (count, Some(count))
            }
            TokenizerTransitionsIterInner::PackedCompressed { segment, state, .. } => {
                let count = segment.transition_count(*state);
                (count, Some(count))
            }
            TokenizerTransitionsIterInner::Virtual { bytes, .. } => bytes.size_hint(),
            TokenizerTransitionsIterInner::VirtualProduct(iter) => iter.size_hint(),
            TokenizerTransitionsIterInner::Empty => (0, Some(0)),
        }
    }

    fn count(self) -> usize {
        match self.inner {
            TokenizerTransitionsIterInner::Dense(iter) => iter.count(),
            TokenizerTransitionsIterInner::Packed { bytes, next, .. }
            | TokenizerTransitionsIterInner::PackedSegment { bytes, next, .. } => {
                bytes.len().saturating_sub(next)
            }
            TokenizerTransitionsIterInner::Compressed { segment, state, .. } => {
                segment.transition_count(state)
            }
            TokenizerTransitionsIterInner::PackedCompressed { segment, state, .. } => {
                segment.transition_count(state)
            }
            TokenizerTransitionsIterInner::Virtual { bytes, .. } => bytes.count(),
            TokenizerTransitionsIterInner::VirtualProduct(iter) => iter.count(),
            TokenizerTransitionsIterInner::Empty => 0,
        }
    }
}

impl Serialize for Tokenizer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let materialized;
        let dfa = if self.compressed_transition_segments.is_empty() {
            &self.dfa
        } else {
            materialized = self.materialized_dfa();
            &materialized
        };
        // Match the historical derived-serialization field order exactly.
        let mut state = serializer.serialize_struct("Tokenizer", 2)?;
        state.serialize_field("dfa", dfa)?;
        state.serialize_field("num_terminals", &self.num_terminals)?;
        state.end()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerMatch {
    pub id: TerminalID,
    pub width: usize,
    pub end_state: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerExecResult {
    pub end_state: TokenizerStateSet,
    pub matches: Vec<TokenizerMatch>,
}

pub type TokenizerStateSet = SmallVec<[u32; 1]>;

/// Exact disjoint union used only by cross-tokenizer compile-time analyses.
/// Source state `s` is represented by `left_offset + s` or
/// `right_offset + s`; state zero is a fresh epsilon dispatcher.
pub struct TokenizerAnalysisUnion {
    pub tokenizer: Tokenizer,
    pub left_offset: u32,
    pub right_offset: u32,
}

/// A compile-time induced graph with original lexical observations.
/// This is not a replacement runtime lexer: only queries whose complete
/// trajectories were proved to lie in the retained set may use it.
#[doc(hidden)]
pub struct TokenizerObservationView {
    pub tokenizer: Tokenizer,
    pub original_to_view: Vec<u32>,
    pub view_to_original: Vec<u32>,
}

pub trait Lexer {
    fn start_state(&self) -> u32;
    fn num_terminals(&self) -> u32;
    fn has_epsilon_transitions(&self) -> bool;
    fn state_has_epsilon_transitions(&self, state: u32) -> bool;
    fn transitions_from(&self, state: u32) -> impl Iterator<Item = (u8, u32)> + '_;

    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) {
        row.fill(u32::MAX);
        for (byte, target) in self.transitions_from(state) {
            row[byte as usize] = target;
        }
    }

    fn transition_row(&self, state: u32) -> Box<[u32; 256]> {
        let mut row = Box::new([u32::MAX; 256]);
        self.fill_transition_row(state, &mut row);
        row
    }

    fn self_loop_bytes(&self, state: u32) -> U8Set {
        let mut bytes = U8Set::empty();
        for (byte, target) in self.transitions_from(state) {
            if target == state {
                bytes.insert(byte);
            }
        }
        bytes
    }

    /// Largest H<=`max_horizon` such that every byte string over `bytes` of
    /// length at most H preserves the lexer observation of `source`.
    ///
    /// This is an exact local proof for scalar deterministic states. It follows
    /// only states actually reachable from `source` under the requested byte
    /// alphabet, deduplicating the frontier at every depth. Bounded-repeat
    /// chains therefore cost O(H) states rather than an O(all-tokenizer-states)
    /// prepass. If the reachable frontier becomes too wide, return the already
    /// proved shorter horizon and let the ordinary exact runtime walk handle
    /// the subtree.
    fn bounded_observation_safe_horizon_from_state(
        &self,
        source: u32,
        bytes: U8Set,
        active_terminals: &BitSet,
        max_horizon: u8,
    ) -> u8 {
        const MAX_FRONTIER_STATES: usize = 4_096;

        #[inline]
        fn equal_under_mask(left: &BitSet, right: &BitSet, mask: &BitSet) -> bool {
            debug_assert_eq!(left.len(), right.len());
            debug_assert_eq!(left.len(), mask.len());
            left.words()
                .iter()
                .zip(right.words())
                .zip(mask.words())
                .all(|((&left, &right), &mask)| ((left ^ right) & mask) == 0)
        }

        if max_horizon == 0
            || bytes.is_empty()
            || source >= self.num_states()
            || self.state_has_epsilon_transitions(source)
        {
            return 0;
        }

        let mut frontier = vec![source];
        for depth in 1..=max_horizon {
            let mut next = Vec::<u32>::new();
            for &state in &frontier {
                if self.state_has_epsilon_transitions(state) {
                    return depth - 1;
                }
                for byte in bytes.iter() {
                    let target = self.get_transition(state, byte);
                    if target == u32::MAX || self.state_has_epsilon_transitions(target) {
                        return depth - 1;
                    }
                    if !equal_under_mask(
                        self.matched_terminal_bitset(target),
                        self.matched_terminal_bitset(source),
                        active_terminals,
                    ) || !equal_under_mask(
                        self.possible_future_terminals(target),
                        self.possible_future_terminals(source),
                        active_terminals,
                    )
                    {
                        return depth - 1;
                    }
                    next.push(target);
                }
            }
            next.sort_unstable();
            next.dedup();
            if next.len() > MAX_FRONTIER_STATES {
                return depth - 1;
            }
            // The same closed deterministic frontier has reappeared. Every
            // requested byte remains inside this already-validated set, so the
            // certificate is valid for every longer horizon too.
            if next == frontier {
                return max_horizon;
            }
            frontier = next;
        }
        max_horizon
    }

    fn transition_count(&self) -> usize {
        (0..self.num_states())
            .map(|state| self.transitions_from(state).count())
            .sum()
    }

    fn step(&self, state: u32, byte: u8) -> Option<u32>;
    fn step_all(&self, states: &[u32], byte: u8) -> TokenizerStateSet;
    fn get_transition(&self, state: u32, byte: u8) -> u32;
    fn matched_terminal_bitset(&self, state: u32) -> &BitSet;
    fn matched_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_;
    fn possible_future_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_;
    fn possible_future_terminals(&self, state: u32) -> &BitSet;

    fn is_end(&self, state: u32) -> bool {
        self.possible_future_terminals(state).is_empty()
    }

    fn num_states(&self) -> u32;
    fn compute_forced_minimized_state_count(&self) -> usize;
    fn execute_from_state_all_widths(
        &self,
        input: &[u8],
        start: u32,
    ) -> TokenizerExecResult;
    fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult;
    fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> TokenizerStateSet;
    fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult;

    fn initial_state(&self) -> u32 {
        self.start_state()
    }

    fn initial_state_id(&self) -> u32 {
        self.initial_state()
    }

    fn tokens_accessible_from_state(&self, state: u32) -> &BitSet {
        self.possible_future_terminals(state)
    }

    fn scan_terminal_matches_from_state(
        &self,
        input: &[u8],
        start: u32,
        terminals_of_interest: &BitSet,
    ) -> (BitSet, TokenizerStateSet);
}

fn into_longest_matches(
    matches: FxHashMap<TerminalID, (usize, TokenizerStateSet)>,
) -> Vec<TokenizerMatch> {
    matches
        .into_iter()
        .flat_map(|(id, (width, end_states))| {
            end_states.into_iter().map(move |end_state| TokenizerMatch {
                id,
                width,
                end_state,
            })
        })
        .collect()
}

fn group_matches_by_width(matches: Vec<TokenizerMatch>) -> Vec<(usize, BTreeSet<TerminalID>)> {
    let mut grouped = std::collections::BTreeMap::<usize, BTreeSet<TerminalID>>::new();
    for matched in matches {
        grouped.entry(matched.width).or_default().insert(matched.id);
    }
    grouped.into_iter().collect()
}

impl Tokenizer {
    #[inline]
    fn virtual_repeat_runtime_for_state(
        &self,
        state: u32,
    ) -> Option<&VirtualBinaryRepeatIntersectionRuntime> {
        let owner = self.virtual_repeat_intersections.first()?.owner_index(state)?;
        self.virtual_repeat_intersections.get(owner).map(Arc::as_ref)
    }

    #[inline]
    fn virtual_residual_runtime_for_state(&self, state: u32) -> Option<&VirtualResidualRuntime> {
        let owner = self.virtual_residuals.first()?.owner_index(state)?;
        self.virtual_residuals.get(owner).map(Arc::as_ref)
    }

    /// Exact parser-relative observation equivalence of two lexer residual
    /// configurations.  Unlike the bounded counterpart, `Some(true)` is only
    /// returned after the complete finite graph of reachable epsilon-closed
    /// configuration pairs has been exhausted.  It therefore proves equality
    /// of `(matched, possible future)` for every continuation byte string, for
    /// exactly the terminal labels selected by `terminals`.
    ///
    /// `None` is a conservative resource-limit decline.  It is not evidence
    /// of inequivalence.
    pub fn state_sets_observation_equivalent_exact(
        &self,
        left: &[u32],
        right: &[u32],
        terminals: &BitSet,
        pair_limit: usize,
    ) -> Option<bool> {
        if left.is_empty()
            || right.is_empty()
            || pair_limit == 0
            || terminals.len() != self.num_terminals as usize
            || left.iter().any(|&state| state >= self.num_states())
            || right.iter().any(|&state| state >= self.num_states())
        {
            return None;
        }

        let observation_equal = |left: &[u32], right: &[u32]| {
            terminals.iter().all(|terminal| {
                let left_matched = left
                    .iter()
                    .any(|&state| self.matched_terminal_bitset(state).contains(terminal));
                let right_matched = right
                    .iter()
                    .any(|&state| self.matched_terminal_bitset(state).contains(terminal));
                if left_matched != right_matched {
                    return false;
                }
                let left_future = left
                    .iter()
                    .any(|&state| self.possible_future_terminals(state).contains(terminal));
                let right_future = right
                    .iter()
                    .any(|&state| self.possible_future_terminals(state).contains(terminal));
                left_future == right_future
            })
        };
        let observation_live = |states: &[u32]| {
            terminals.iter().any(|terminal| {
                states.iter().any(|&state| {
                    self.matched_terminal_bitset(state).contains(terminal)
                        || self.possible_future_terminals(state).contains(terminal)
                })
            })
        };

        // The ordinary dynamic JSON tokenizer is deterministic and epsilon
        // free. Keep that commit-time cache-preservation proof allocation-light:
        // the exact product is just pairs of raw states.
        if !self.has_any_virtual_runtime()
            && !self.has_epsilon_transitions()
            && let ([left], [right]) = (left, right)
        {
            if !observation_equal(&[*left], &[*right]) {
                return Some(false);
            }
            let mut seen = FxHashSet::<(u32, u32)>::default();
            seen.insert((*left, *right));
            let mut frontier = vec![(*left, *right)];
            while let Some((left, right)) = frontier.pop() {
                for byte in 0u16..=255 {
                    let left_target = self.step(left, byte as u8);
                    let right_target = self.step(right, byte as u8);
                    let left_live = left_target.is_some_and(|state| observation_live(&[state]));
                    let right_live = right_target.is_some_and(|state| observation_live(&[state]));
                    if left_live != right_live {
                        return Some(false);
                    }
                    if !left_live {
                        continue;
                    }
                    let (Some(left_target), Some(right_target)) = (left_target, right_target) else {
                        return Some(false);
                    };
                    if !observation_equal(&[left_target], &[right_target]) {
                        return Some(false);
                    }
                    if left_target == right_target {
                        continue;
                    }
                    if seen.insert((left_target, right_target)) {
                        if seen.len() > pair_limit {
                            return None;
                        }
                        frontier.push((left_target, right_target));
                    }
                }
            }
            return Some(true);
        }

        let normalize = |states: &[u32]| -> TokenizerStateSet {
            let mut closure = TokenizerStateSet::new();
            for &state in states {
                closure.extend(self.epsilon_closure_states(&[state]));
            }
            closure.sort_unstable();
            closure.dedup();
            closure
        };
        let left = normalize(left);
        let right = normalize(right);
        if !observation_equal(&left, &right) {
            return Some(false);
        }

        type ConfigPair = (Box<[u32]>, Box<[u32]>);
        let to_pair = |left: &TokenizerStateSet, right: &TokenizerStateSet| -> ConfigPair {
            (
                left.iter().copied().collect::<Vec<_>>().into_boxed_slice(),
                right.iter().copied().collect::<Vec<_>>().into_boxed_slice(),
            )
        };
        let mut seen = FxHashSet::<ConfigPair>::default();
        seen.insert(to_pair(&left, &right));
        let mut frontier = vec![(left, right)];
        while let Some((left, right)) = frontier.pop() {
            for byte in 0u16..=255 {
                let left_target = self.step_all(left.as_slice(), byte as u8);
                let right_target = self.step_all(right.as_slice(), byte as u8);
                let left_live = !left_target.is_empty() && observation_live(&left_target);
                let right_live = !right_target.is_empty() && observation_live(&right_target);
                if left_live != right_live {
                    return Some(false);
                }
                if !left_live {
                    continue;
                }
                if !observation_equal(&left_target, &right_target) {
                    return Some(false);
                }
                if left_target == right_target {
                    continue;
                }
                let pair = to_pair(&left_target, &right_target);
                if seen.insert(pair) {
                    if seen.len() > pair_limit {
                        return None;
                    }
                    frontier.push((left_target, right_target));
                }
            }
        }
        Some(true)
    }

    /// Diagnostic form of [`Self::state_sets_observation_equivalent_exact`].
    /// `Err(bytes)` is a concrete distinguishing continuation prefix;
    /// `Ok(())` proves equivalence; `None` is a resource-limit decline.
    #[cfg(test)]
    pub fn state_sets_observation_equivalence_exact_witness(
        &self,
        left: &[u32],
        right: &[u32],
        terminals: &BitSet,
        pair_limit: usize,
    ) -> Option<Result<(), Box<[u8]>>> {
        if left.is_empty() || right.is_empty() || pair_limit == 0 {
            return None;
        }
        if terminals.len() != self.num_terminals as usize {
            return None;
        }

        let normalize = |states: &[u32]| -> Option<TokenizerStateSet> {
            if states.iter().any(|&state| state >= self.num_states()) {
                return None;
            }
            let mut closure = TokenizerStateSet::new();
            for &state in states {
                closure.extend(self.dfa.epsilon_closure(&[state]));
            }
            closure.sort_unstable();
            closure.dedup();
            Some(closure)
        };
        let observation_equal = |left: &[u32], right: &[u32]| {
            terminals.iter().all(|terminal| {
                let left_matched = left
                    .iter()
                    .any(|&state| self.matched_terminal_bitset(state).contains(terminal));
                let right_matched = right
                    .iter()
                    .any(|&state| self.matched_terminal_bitset(state).contains(terminal));
                if left_matched != right_matched {
                    return false;
                }
                let left_future = left
                    .iter()
                    .any(|&state| self.possible_future_terminals(state).contains(terminal));
                let right_future = right
                    .iter()
                    .any(|&state| self.possible_future_terminals(state).contains(terminal));
                left_future == right_future
            })
        };
        let observation_live = |states: &[u32]| {
            terminals.iter().any(|terminal| {
                states.iter().any(|&state| {
                    self.matched_terminal_bitset(state).contains(terminal)
                        || self.possible_future_terminals(state).contains(terminal)
                })
            })
        };

        let left = normalize(left)?;
        let right = normalize(right)?;
        if !observation_equal(&left, &right) {
            return Some(Err(Box::new([])));
        }

        type ConfigPair = (Box<[u32]>, Box<[u32]>);
        let to_pair = |left: &TokenizerStateSet, right: &TokenizerStateSet| -> ConfigPair {
            (
                left.iter().copied().collect::<Vec<_>>().into_boxed_slice(),
                right.iter().copied().collect::<Vec<_>>().into_boxed_slice(),
            )
        };
        let mut seen = FxHashSet::<ConfigPair>::default();
        seen.insert(to_pair(&left, &right));
        let mut frontier = vec![(left, right, Vec::<u8>::new())];
        while let Some((left, right, prefix)) = frontier.pop() {
            for byte in 0u16..=255 {
                let left_target = self.step_all(left.as_slice(), byte as u8);
                let right_target = self.step_all(right.as_slice(), byte as u8);
                let left_live = !left_target.is_empty() && observation_live(&left_target);
                let right_live = !right_target.is_empty() && observation_live(&right_target);
                if left_live != right_live {
                    let mut witness = prefix.clone();
                    witness.push(byte as u8);
                    return Some(Err(witness.into_boxed_slice()));
                }
                if !left_live {
                    continue;
                }
                if !observation_equal(&left_target, &right_target) {
                    let mut witness = prefix.clone();
                    witness.push(byte as u8);
                    return Some(Err(witness.into_boxed_slice()));
                }
                if left_target == right_target {
                    continue;
                }
                let pair = to_pair(&left_target, &right_target);
                if seen.insert(pair) {
                    if seen.len() > pair_limit {
                        return None;
                    }
                    let mut next_prefix = prefix.clone();
                    next_prefix.push(byte as u8);
                    frontier.push((left_target, right_target, next_prefix));
                }
            }
        }
        Some(Ok(()))
    }

    /// Exact quotient of raw lexer states for the complete observation of one
    /// terminal: `(matched now, possible in the future)`.
    ///
    /// Each raw state is first projected to the selected terminal and closed
    /// under epsilon transitions.  Those finite residual configurations form a
    /// deterministic Moore machine under byte input.  We discover that machine
    /// from every raw singleton residual, then refine by output + byte-successor
    /// class until a true fixed point.  Equal returned nonzero class IDs are
    /// therefore an exact arbitrary-continuation equivalence proof.  Class zero
    /// is the terminal-dead residual.
    ///
    /// `None` is a conservative resource-limit decline.
    pub fn exact_terminal_observation_partition(
        &self,
        terminal: TerminalID,
        config_limit: usize,
        transition_work_limit: usize,
    ) -> Option<(Box<[u32]>, usize, usize)> {
        if self.has_any_virtual_runtime()
            || terminal >= self.num_terminals
            || config_limit == 0
            || transition_work_limit == 0
        {
            return None;
        }

        // Fast exact path for the common partitioned lexer shape.  Once the
        // reset dispatcher selects this terminal's unique component,
        // `has_scalar_deterministic_dispatch()` proves that every byte-reachable
        // state is scalar and has no epsilon fan-out.  We can therefore refine
        // the terminal Moore machine directly on raw states, with no subset
        // construction or configuration hash-consing.  Raw states outside the
        // certified component deliberately retain class zero and can never be
        // used as aliases by callers.
        if let Some(root) = self.terminal_scalar_dispatch_root(terminal) {
            let state_count = self.num_states() as usize;
            let mut local_of_raw = vec![u32::MAX; state_count];
            let mut raws = Vec::<u32>::new();
            let mut transitions = Vec::<Box<[(u8, u32)]>>::new();
            let mut observations = Vec::<u8>::new();
            let mut pending = VecDeque::from([root]);
            local_of_raw[root as usize] = 0;
            raws.push(root);
            let mut work = 0usize;

            let mut index = 0usize;
            while let Some(state) = pending.pop_front() {
                debug_assert_eq!(raws[index], state);
                if raws.len() > config_limit {
                    return None;
                }
                let matched = self.matched_terminal_bitset(state).contains(terminal as usize);
                let future = self
                    .possible_future_terminals(state)
                    .contains(terminal as usize);
                observations.push((u8::from(matched) << 1) | u8::from(future));

                let mut row = Vec::<(u8, u32)>::new();
                for (byte, target) in self.transitions_from(state) {
                    work = work.saturating_add(1);
                    if work > transition_work_limit {
                        return None;
                    }
                    if !self.state_live_for_terminal(target, terminal) {
                        continue;
                    }
                    let target_index = target as usize;
                    let local = if local_of_raw[target_index] == u32::MAX {
                        let next = u32::try_from(raws.len()).ok()?;
                        local_of_raw[target_index] = next;
                        raws.push(target);
                        pending.push_back(target);
                        next
                    } else {
                        local_of_raw[target_index]
                    };
                    row.push((byte, local));
                }
                transitions.push(row.into_boxed_slice());
                index += 1;
            }
            let classes = super::minimize::hopcroft_projected_observation_partition(
                &transitions,
                &observations,
            );
            let mut mapped = vec![0u32; state_count];
            for (local, &raw) in raws.iter().enumerate() {
                mapped[raw as usize] = classes[local];
            }
            // Preserve the public class-zero contract: zero means this
            // terminal is dead.  A scalar dispatch certificate only covers
            // the reset-reachable component, while structural synthesis may
            // append externally-entered live residuals elsewhere in the raw
            // DFA.  Give each such residual a fresh singleton class.  That is
            // intentionally incomplete (two equivalent external residuals may
            // not be merged), but remains an exact equality certificate and
            // prevents an uncovered live state from being confused with dead.
            let mut next_uncovered_class = classes
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
                .saturating_add(1);
            for raw in 0..self.num_states() {
                if mapped[raw as usize] == 0 && self.state_live_for_terminal(raw, terminal) {
                    mapped[raw as usize] = next_uncovered_class;
                    next_uncovered_class = next_uncovered_class.saturating_add(1);
                }
            }
            return Some((mapped.into_boxed_slice(), raws.len(), 0));
        }

        type Config = Box<[u32]>;
        let mut config_ids = FxHashMap::<Config, u32>::default();
        let mut configs = Vec::<Config>::new();
        let mut raw_config = vec![0u32; self.num_states() as usize];

        let intern = |config: Config,
                      config_ids: &mut FxHashMap<Config, u32>,
                      configs: &mut Vec<Config>|
         -> Option<u32> {
            if config.is_empty() {
                return Some(0);
            }
            if let Some(&id) = config_ids.get(config.as_ref()) {
                return Some(id);
            }
            if configs.len() >= config_limit {
                return None;
            }
            let id = u32::try_from(configs.len() + 1).ok()?;
            config_ids.insert(config.clone(), id);
            configs.push(config);
            Some(id)
        };

        for raw in 0..self.num_states() {
            let config = self.terminal_projected_epsilon_closure(&[raw], terminal);
            raw_config[raw as usize] = intern(config, &mut config_ids, &mut configs)?;
        }
        let mut transitions = Vec::<Box<[(u8, u32)]>>::new();
        let mut observations = Vec::<u8>::new();
        let mut work = 0usize;
        let mut index = 0usize;
        while index < configs.len() {
            let config = configs[index].clone();
            let matched = config
                .iter()
                .any(|&state| self.matched_terminal_bitset(state).contains(terminal as usize));
            let future = config.iter().any(|&state| {
                self.possible_future_terminals(state)
                    .contains(terminal as usize)
            });
            observations.push((u8::from(matched) << 1) | u8::from(future));

            let mut bytes = U8Set::empty();
            for &state in config.iter() {
                for (byte, _) in self.transitions_from(state) {
                    bytes.insert(byte);
                }
            }
            let mut row = Vec::<(u8, u32)>::with_capacity(bytes.len());
            for byte in bytes.iter() {
                let target = self.terminal_projected_step(
                    config.as_ref(),
                    terminal,
                    byte,
                    &mut work,
                    transition_work_limit,
                )?;
                let target = intern(target, &mut config_ids, &mut configs)?;
                if target != 0 {
                    row.push((byte, target));
                }
            }
            transitions.push(row.into_boxed_slice());
            index += 1;
        }
        let mut classes = observations.iter().map(|&obs| u32::from(obs)).collect::<Vec<_>>();
        let mut rounds = 0usize;
        loop {
            rounds += 1;
            let mut ids = FxHashMap::<(u8, Vec<(u8, u32)>), u32>::default();
            let mut next_id = 1u32;
            let mut next = vec![0u32; configs.len()];
            for state in 0..configs.len() {
                let mut signature = Vec::<(u8, u32)>::new();
                for &(byte, target) in transitions[state].iter() {
                    let target_class = classes[(target - 1) as usize];
                    signature.push((byte, target_class));
                }
                let key = (observations[state], signature);
                next[state] = if let Some(&id) = ids.get(&key) {
                    id
                } else {
                    let id = next_id;
                    next_id = next_id.saturating_add(1);
                    ids.insert(key, id);
                    id
                };
            }

            let mut old_to_new = FxHashMap::<u32, u32>::default();
            let mut new_to_old = FxHashMap::<u32, u32>::default();
            let stable = classes.iter().zip(next.iter()).all(|(&old, &new)| {
                let forward = old_to_new.entry(old).or_insert(new);
                if *forward != new {
                    return false;
                }
                let backward = new_to_old.entry(new).or_insert(old);
                *backward == old
            });
            classes = next;
            if stable {
                break;
            }
        }
        let mapped = raw_config
            .into_iter()
            .map(|config| {
                if config == 0 {
                    0
                } else {
                    classes[(config - 1) as usize]
                }
            })
            .collect::<Vec<_>>();
        Some((mapped.into_boxed_slice(), configs.len(), rounds))
    }

    /// Exact finite-horizon quotient for the Boolean observation
    /// `terminal is still a possible future terminal`.
    ///
    /// Class zero is the dead class: states from which `terminal` is already
    /// impossible, epsilon-bearing states, and missing byte transitions all
    /// behave identically for the no-finalization continuation proof used by
    /// dynamic trie projections.  For each subsequent round we partition live
    /// states by the byte -> previous-class map, omitting edges into class zero.
    /// After H rounds, equal class ids therefore imply equal terminal-liveness
    /// for every byte string of length at most H.
    pub fn bounded_terminal_future_partition(
        &self,
        terminal: TerminalID,
        horizon: u8,
    ) -> Box<[u32]> {
        use rustc_hash::FxHashMap;

        let state_count = self.num_states() as usize;
        let mut classes = vec![0u32; state_count];
        for state in 0..state_count {
            let state_u32 = state as u32;
            if !self.state_has_epsilon_transitions(state_u32)
                && self
                    .possible_future_terminals(state_u32)
                    .contains(terminal as usize)
            {
                classes[state] = 1;
            }
        }
        if horizon == 0 {
            return classes.into_boxed_slice();
        }

        // Most synthesized bounded-repeat rows have roughly 90 live bytes.
        // Keep those signatures inline; only unusually broad rows allocate.
        type Signature = SmallVec<[(u8, u32); 128]>;
        let mut next = vec![0u32; state_count];
        for _ in 0..horizon {
            let mut ids = FxHashMap::<Signature, u32>::default();
            let mut next_id = 1u32;
            for state in 0..state_count {
                if classes[state] == 0 {
                    continue;
                }
                let mut signature = Signature::new();
                for (byte, target) in self.transitions_from(state as u32) {
                    let target_class = classes.get(target as usize).copied().unwrap_or(0);
                    if target_class != 0 {
                        signature.push((byte, target_class));
                    }
                }
                let id = if let Some(&id) = ids.get(&signature) {
                    id
                } else {
                    let id = next_id;
                    next_id = next_id.saturating_add(1);
                    ids.insert(signature, id);
                    id
                };
                next[state] = id;
            }
            std::mem::swap(&mut classes, &mut next);
            next.fill(0);
        }
        classes.into_boxed_slice()
    }

    fn merged_terminal_exprs(
        tokenizers: &[(&Tokenizer, TerminalID)],
        total_terminals: TerminalID,
    ) -> Option<Arc<[Expr]>> {
        let mut merged = vec![None::<Expr>; total_terminals as usize];
        for &(tokenizer, terminal_offset) in tokenizers {
            let exprs = tokenizer.exprs.as_deref()?;
            if exprs.len() != tokenizer.num_terminals as usize {
                return None;
            }
            for (terminal, expr) in exprs.iter().enumerate() {
                let slot = merged.get_mut(terminal_offset as usize + terminal)?;
                if slot.is_some() {
                    return None;
                }
                *slot = Some(expr.clone());
            }
        }
        merged
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .map(Arc::from)
    }

    pub fn canonicalize_terminal_aliases(
        &mut self,
        canonical: TerminalID,
        aliases: &[TerminalID],
    ) {
        let aliases = aliases
            .iter()
            .copied()
            .filter(|&alias| alias != canonical)
            .map(|alias| alias as usize)
            .collect::<Vec<_>>();
        self.dfa
            .canonicalize_group_aliases(canonical as usize, &aliases);
        self.invalidate_derived_caches();
    }

    /// Form an exact disjoint union of independently compiled tokenizers while
    /// keeping every terminal ID distinct.
    ///
    /// `terminal_offsets[i]` is added to every terminal/group ID in
    /// `tokenizers[i]`. A fresh epsilon root dispatches to each source start
    /// state. No DFA states or terminals are identified across inputs.
    ///
    /// The returned state offsets map each source raw tokenizer state into the
    /// merged raw-state domain.
    /// Form one ordinary flattened tokenizer by consuming the parent as the
    /// destination and appending borrowed child components. Parent state IDs and
    /// transition buffers remain unchanged; only child states are cloned and
    /// rebased. Terminal IDs in the parent remain identity-mapped from zero.
    pub fn disjoint_union_with_owned_parent(
        mut parent: Tokenizer,
        parent_terminal_offset: TerminalID,
        children: &[(&Tokenizer, TerminalID)],
    ) -> (Tokenizer, Vec<u32>) {
        assert_eq!(
            parent_terminal_offset, 0,
            "owned-parent tokenizer composition requires the parent terminal domain to start at zero",
        );
        // Fast-loaded tokenizers may keep finalizer/future/epsilon metadata in
        // packed runtime storage while retaining only the structural DFA
        // skeleton.  This function mutates the parent's epsilon graph in
        // place, so leaving packed metadata authoritative would hide every new
        // child edge (and can make the start state look like a pure dispatcher
        // even when its byte transitions still live in packed storage).
        // Materialize metadata only; byte-transition rows remain packed.
        parent.materialize_runtime_metadata_for_structural_mutation();
        let total_terminals = std::iter::once((&parent, parent_terminal_offset))
            .chain(children.iter().copied())
            .map(|(tokenizer, terminal_offset)| {
                terminal_offset
                    .checked_add(tokenizer.num_terminals)
                    .expect("merged tokenizer terminal ID overflow")
            })
            .max()
            .unwrap_or(0);
        let expr_components = std::iter::once((&parent, parent_terminal_offset))
            .chain(children.iter().copied())
            .collect::<Vec<_>>();
        let merged_exprs = Self::merged_terminal_exprs(&expr_components, total_terminals);

        let mut merged_byte_transition_count = parent.transition_count();
        let mut merged_epsilon_transition_count = parent.dfa.epsilon_transition_count();
        let mut merged_has_self_loops = parent.dfa.has_self_loops();
        for &(tokenizer, _) in children {
            merged_byte_transition_count = merged_byte_transition_count
                .checked_add(tokenizer.transition_count())
                .expect("merged tokenizer byte-transition count overflow");
            merged_epsilon_transition_count = merged_epsilon_transition_count
                .checked_add(tokenizer.dfa.epsilon_transition_count())
                .and_then(|count| count.checked_add(1))
                .expect("merged tokenizer epsilon-transition count overflow");
            merged_has_self_loops |= tokenizer.dfa.has_self_loops();
        }

        parent
            .dfa
            .ensure_group_mapping_capacity(total_terminals as usize);
        let parent_closures = std::mem::take(&mut parent.singleton_epsilon_closures)
            .into_inner()
            .and_then(|closures| Arc::try_unwrap(closures).ok());
        let mut child_closures = Vec::with_capacity(children.len());
        let mut state_offsets = Vec::with_capacity(children.len() + 1);
        state_offsets.push(0);
        let mut compressed_segments = parent
            .compressed_transition_segments
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mut packed_runtime_segments = parent
            .packed_runtime_transition_segments
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mut packed_metadata_segments = parent
            .packed_runtime_metadata_segments
            .iter()
            .map(|segment| PackedTokenizerMetadataSegment {
                state_offset: segment.state_offset,
                metadata: segment
                    .metadata
                    .rebased_terminals(parent_terminal_offset, total_terminals),
            })
            .collect::<Vec<_>>();
        let mut packed_compressed_segments = parent
            .packed_compressed_transition_segments
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mut root_finalizers = BitSet::new(total_terminals as usize);
        for terminal in parent.matched_terminals_iter(parent.start_state()) {
            root_finalizers.set(terminal as usize);
        }
        let mut root_futures = BitSet::new(total_terminals as usize);
        for terminal in parent.possible_future_terminals_iter(parent.start_state()) {
            root_futures.set(terminal as usize);
        }

        for &(tokenizer, terminal_offset) in children {
            let global_groups = (0..tokenizer.num_terminals)
                .map(|terminal| (terminal_offset + terminal) as usize)
                .collect::<Vec<_>>();
            let mut component = tokenizer.dfa.clone();
            while component.num_states() < tokenizer.num_states() as usize {
                component.add_state();
            }
            let state_offset = parent
                .dfa
                .append_rebased_component(component, &global_groups);
            state_offsets.push(state_offset);
            if let Some(closures) = tokenizer.cached_singleton_epsilon_closures() {
                child_closures.push((Arc::clone(closures), state_offset, tokenizer.start_state()));
            }
            parent
                .dfa
                .add_epsilon_transition(parent.start_state(), state_offset + tokenizer.start_state());
            if let Some(transitions) = tokenizer.packed_runtime_transitions.as_ref() {
                packed_runtime_segments.push(PackedRuntimeTransitionSegment {
                    state_offset,
                    transitions: Arc::clone(transitions),
                });
            }
            packed_runtime_segments.extend(
                tokenizer
                    .packed_runtime_transition_segments
                    .iter()
                    .cloned()
                    .map(|mut segment| {
                        segment.state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged packed tokenizer state offset overflow");
                        segment
                    }),
            );
            if let Some(metadata) = tokenizer.packed_runtime_metadata.as_ref() {
                packed_metadata_segments.push(PackedTokenizerMetadataSegment {
                    state_offset,
                    metadata: metadata.rebased_terminals(terminal_offset, total_terminals),
                });
            }
            packed_metadata_segments.extend(
                tokenizer
                    .packed_runtime_metadata_segments
                    .iter()
                    .map(|segment| {
                        let rebased_state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged packed tokenizer metadata state offset overflow");
                        PackedTokenizerMetadataSegment {
                            state_offset: rebased_state_offset,
                            metadata: segment
                                .metadata
                                .rebased_terminals(terminal_offset, total_terminals),
                        }
                    }),
            );
            packed_compressed_segments.extend(
                tokenizer
                    .packed_compressed_transition_segments
                    .iter()
                    .cloned()
                    .map(|mut segment| {
                        segment.state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged packed-compressed tokenizer state offset overflow");
                        segment
                    }),
            );
            for terminal in tokenizer.matched_terminals_iter(tokenizer.start_state()) {
                root_finalizers.set(terminal_offset as usize + terminal as usize);
            }
            for terminal in tokenizer.possible_future_terminals_iter(tokenizer.start_state()) {
                root_futures.set(terminal_offset as usize + terminal as usize);
            }
            compressed_segments.extend(
                tokenizer
                    .compressed_transition_segments
                    .iter()
                    .cloned()
                    .map(|mut segment| {
                        segment.state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged compressed tokenizer state offset overflow");
                        segment
                    }),
            );
        }
        parent
            .dfa
            .overwrite_state_metadata(parent.start_state(), root_finalizers, root_futures);
        parent.dfa.set_derived_stats(
            merged_byte_transition_count,
            merged_epsilon_transition_count,
            merged_has_self_loops,
        );
        parent.num_terminals = total_terminals;
        parent.compressed_transition_segments =
            Arc::from(compressed_segments.into_boxed_slice());
        packed_runtime_segments.sort_unstable_by_key(|segment| segment.state_offset);
        parent.packed_runtime_transition_segments =
            Arc::from(packed_runtime_segments.into_boxed_slice());
        packed_metadata_segments.sort_unstable_by_key(|segment| segment.state_offset);
        parent.packed_runtime_metadata_segments =
            Arc::from(packed_metadata_segments.into_boxed_slice());
        packed_compressed_segments.sort_unstable_by_key(|segment| segment.state_offset);
        parent.packed_compressed_transition_segments =
            Arc::from(packed_compressed_segments.into_boxed_slice());
        parent.exprs = merged_exprs;
        parent.invalidate_derived_caches();
        if child_closures.len() == children.len()
            && let Some(closures) = parent_closures.and_then(|closures| {
            closures.append_rebased_children(
                parent.start_state(),
                &child_closures,
            )
        }) {
            let _ = parent
                .singleton_epsilon_closures
                .set(Arc::new(closures));
        }
        (parent, state_offsets)
    }

    pub fn disjoint_union_with_terminal_offsets(
        tokenizers: &[(&Tokenizer, TerminalID)],
    ) -> (Tokenizer, Vec<u32>) {
        let total_terminals = tokenizers
            .iter()
            .map(|(tokenizer, terminal_offset)| {
                terminal_offset
                    .checked_add(tokenizer.num_terminals)
                    .expect("merged tokenizer terminal ID overflow")
            })
            .max()
            .unwrap_or(0);
        let merged_exprs = Self::merged_terminal_exprs(tokenizers, total_terminals);

        let total_states = 1usize.saturating_add(
            tokenizers
                .iter()
                .map(|(tokenizer, _)| tokenizer.num_states() as usize)
                .sum::<usize>(),
        );
        let mut merged = DFA::new(total_states.min(1));
        merged.ensure_group_capacity(total_terminals as usize);
        let mut state_offsets = Vec::with_capacity(tokenizers.len());
        let mut compressed_segments = Vec::<CompressedTransitionSegment>::new();
        let mut packed_runtime_segments = Vec::<PackedRuntimeTransitionSegment>::new();
        let mut packed_metadata_segments = Vec::<PackedTokenizerMetadataSegment>::new();
        let mut packed_compressed_segments = Vec::<PackedCompressedTransitionSegment>::new();
        let mut root_finalizers = BitSet::new(total_terminals as usize);
        let mut root_futures = BitSet::new(total_terminals as usize);

        for &(tokenizer, terminal_offset) in tokenizers {
            // Preserve compact runtime transition segments. Materializing them
            // here expands every byte-class row before immediately rebuilding
            // equivalent runtime caches, which is catastrophic for million-state
            // composed tokenizers. Segment targets are local to the segment, so
            // rebasing only its state offset is exact.
            let mut component = tokenizer.dfa.clone();
            while component.num_states() < tokenizer.num_states() as usize {
                component.add_state();
            }
            let global_groups = (0..tokenizer.num_terminals)
                .map(|terminal| (terminal_offset + terminal) as usize)
                .collect::<Vec<_>>();
            let state_offset = merged.append_rebased_component(component, &global_groups);
            if let Some(transitions) = tokenizer.packed_runtime_transitions.as_ref() {
                packed_runtime_segments.push(PackedRuntimeTransitionSegment {
                    state_offset,
                    transitions: Arc::clone(transitions),
                });
            }
            packed_runtime_segments.extend(
                tokenizer
                    .packed_runtime_transition_segments
                    .iter()
                    .cloned()
                    .map(|mut segment| {
                        segment.state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged packed tokenizer state offset overflow");
                        segment
                    }),
            );
            if let Some(metadata) = tokenizer.packed_runtime_metadata.as_ref() {
                packed_metadata_segments.push(PackedTokenizerMetadataSegment {
                    state_offset,
                    metadata: metadata.rebased_terminals(terminal_offset, total_terminals),
                });
            }
            packed_metadata_segments.extend(
                tokenizer
                    .packed_runtime_metadata_segments
                    .iter()
                    .map(|segment| {
                        let rebased_state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged packed tokenizer metadata state offset overflow");
                        PackedTokenizerMetadataSegment {
                            state_offset: rebased_state_offset,
                            metadata: segment
                                .metadata
                                .rebased_terminals(terminal_offset, total_terminals),
                        }
                    }),
            );
            compressed_segments.extend(
                tokenizer
                    .compressed_transition_segments
                    .iter()
                    .cloned()
                    .map(|mut segment| {
                        segment.state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged compressed tokenizer state offset overflow");
                        segment
                    }),
            );
            packed_compressed_segments.extend(
                tokenizer
                    .packed_compressed_transition_segments
                    .iter()
                    .cloned()
                    .map(|mut segment| {
                        segment.state_offset = segment
                            .state_offset
                            .checked_add(state_offset)
                            .expect("merged packed-compressed tokenizer state offset overflow");
                        segment
                    }),
            );
            state_offsets.push(state_offset);
            merged.add_epsilon_transition(0, state_offset + tokenizer.start_state());

            for terminal in tokenizer.matched_terminals_iter(tokenizer.start_state()) {
                root_finalizers.set(terminal_offset as usize + terminal as usize);
            }
            for terminal in tokenizer.possible_future_terminals_iter(tokenizer.start_state()) {
                root_futures.set(terminal_offset as usize + terminal as usize);
            }
        }
        merged.overwrite_state_metadata(0, root_finalizers, root_futures);
        packed_runtime_segments.sort_unstable_by_key(|segment| segment.state_offset);
        packed_metadata_segments.sort_unstable_by_key(|segment| segment.state_offset);
        packed_compressed_segments.sort_unstable_by_key(|segment| segment.state_offset);

        let mut tokenizer = Tokenizer::from_parts_with_compressed_transitions(
            merged,
            total_terminals,
            merged_exprs,
            compressed_segments,
        );
        tokenizer.packed_runtime_transition_segments =
            Arc::from(packed_runtime_segments.into_boxed_slice());
        tokenizer.packed_runtime_metadata_segments =
            Arc::from(packed_metadata_segments.into_boxed_slice());
        tokenizer.packed_compressed_transition_segments =
            Arc::from(packed_compressed_segments.into_boxed_slice());
        (tokenizer, state_offsets)
    }

    #[inline]
    fn invalidate_derived_caches(&mut self) {
        let _ = self.singleton_epsilon_closures.take();
        let _ = self.matched_terminals_cache.take();
        let _ = self.initial_byte_frontiers.take();
        let _ = self.all_self_loop_bytes_cache.take();
        let _ = self.transition_count_cache.take();
        let _ = self.forced_minimized_state_count_cache.take();
        let _ = self.scalar_deterministic_dispatch_cache.take();
        let _ = self.sorted_dispatch_roots_cache.take();
        let _ = self.state_first_bytes_cache.take();
    }

    /// Exact byte successors of one epsilon-closed subset, grouped sparsely
    /// from the source tokenizer's actual outgoing transitions.
    ///
    /// Subset construction used to probe all 256 bytes and every source state
    /// for every product state. Residual and ordinary lexer states are often
    /// sparse, so that turns a modest exact determinization into millions of
    /// dead `step()` calls. Reuse one 256-bucket scratch table instead: visit
    /// only real source transitions, union their precomputed singleton epsilon
    /// closures, then emit the non-empty byte rows in byte order.
    fn for_each_determinization_subset_successor(
        &self,
        subset: &[u32],
        closures: &SingletonEpsilonClosures,
        byte_targets: &mut [SmallVec<[u32; 8]>; 256],
        mut visit: impl FnMut(u8, &[u32]) -> bool,
    ) -> bool {
        for &source_state in subset {
            for (byte, target) in self.transitions_from(source_state) {
                byte_targets[byte as usize]
                    .extend_from_slice(&closures[target as usize]);
            }
        }

        // Keep normalized successor sets in the reusable scratch buckets and
        // expose them by slice. Most derivatives already exist in the subset
        // interner, so allocating an owned box before the lookup is pure waste.
        // Callers clone only the genuinely new subsets they need to retain.
        for (byte, targets) in byte_targets.iter_mut().enumerate() {
            if targets.is_empty() {
                continue;
            }
            if !targets.as_slice().is_sorted() {
                targets.sort_unstable();
            }
            targets.dedup();
            let keep_going = visit(byte as u8, targets.as_slice());
            targets.clear();
            if !keep_going {
                return false;
            }
        }
        true
    }

    /// The same exact successor construction over a fully materialized source
    /// DFA. Finite-horizon mask determinization already pays to materialize the
    /// tokenizer before expansion; using those rows avoids repeatedly decoding
    /// packed/compressed transition segments for every subset member.
    fn for_each_materialized_determinization_subset_successor(
        source_dfa: &DFA,
        subset: &[u32],
        closures: &SingletonEpsilonClosures,
        byte_targets: &mut [SmallVec<[u32; 8]>; 256],
        mut visit: impl FnMut(u8, &[u32]) -> bool,
    ) -> bool {
        for &source_state in subset {
            for (byte, target) in source_dfa.transitions(source_state) {
                byte_targets[byte as usize]
                    .extend_from_slice(&closures[target as usize]);
            }
        }

        for (byte, targets) in byte_targets.iter_mut().enumerate() {
            if targets.is_empty() {
                continue;
            }
            if !targets.as_slice().is_sorted() {
                targets.sort_unstable();
            }
            targets.dedup();
            let keep_going = visit(byte as u8, targets.as_slice());
            targets.clear();
            if !keep_going {
                return false;
            }
        }
        true
    }

    /// Pre-group one materialized DFA row by its exact singleton epsilon-
    /// closed target. When every byte edge in the source lands in a singleton
    /// closure, subset determinization can refine the fixed byte alphabet with
    /// `U8Set` intersections instead of replaying every literal byte edge for
    /// every product state.
    fn singleton_closed_transition_groups(
        source_dfa: &DFA,
        closures: &SingletonEpsilonClosures,
    ) -> Option<Vec<(SmallVec<[(u32, U8Set); 8]>, U8Set)>> {
        let mut rows = Vec::with_capacity(source_dfa.num_states());
        for state in 0..source_dfa.num_states() as u32 {
            let mut groups = SmallVec::<[(u32, U8Set); 8]>::new();
            let mut support = U8Set::empty();
            for (byte, target) in source_dfa.transitions(state) {
                let closure = closures.get(target as usize)?;
                if closure.len() != 1 {
                    return None;
                }
                let closed_target = closure[0];
                support.insert(byte);
                if let Some((_, bytes)) = groups
                    .iter_mut()
                    .find(|(seen_target, _)| *seen_target == closed_target)
                {
                    bytes.insert(byte);
                } else {
                    groups.push((closed_target, U8Set::from_byte(byte)));
                }
            }
            rows.push((groups, support));
        }
        Some(rows)
    }

    /// Exact subset successors for the singleton-closure source above.
    /// `classes` remains a disjoint partition of the 256-byte alphabet, so the
    /// result is exactly the same per-byte successor relation as the generic
    /// closure-union path.
    fn for_each_singleton_closed_subset_successor(
        subset: &[u32],
        grouped_rows: &[(SmallVec<[(u32, U8Set); 8]>, U8Set)],
        mut visit: impl FnMut(U8Set, &[u32]) -> bool,
    ) -> bool {
        let mut classes = Vec::<(U8Set, SmallVec<[u32; 16]>)>::with_capacity(16);
        classes.push((U8Set::all(), SmallVec::new()));

        for &source_state in subset {
            let Some((groups, support)) = grouped_rows.get(source_state as usize) else {
                return false;
            };
            let mut refined = Vec::<(U8Set, SmallVec<[u32; 16]>)>::with_capacity(
                classes.len().saturating_mul(groups.len().saturating_add(1)).min(256),
            );
            for (bytes, targets) in classes.drain(..) {
                let dead = bytes.difference(support);
                if !dead.is_empty() {
                    refined.push((dead, targets.clone()));
                }
                for &(target, target_bytes) in groups {
                    let overlap = bytes.intersection(&target_bytes);
                    if overlap.is_empty() {
                        continue;
                    }
                    let mut next_targets = targets.clone();
                    next_targets.push(target);
                    refined.push((overlap, next_targets));
                }
            }
            classes = refined;
        }

        let mut merged = FxHashMap::<SmallVec<[u32; 16]>, U8Set>::default();
        for (bytes, mut targets) in classes {
            if targets.is_empty() {
                continue;
            }
            targets.sort_unstable();
            targets.dedup();
            *merged.entry(targets).or_insert_with(U8Set::empty) |= bytes;
        }
        let mut merged = merged.into_iter().collect::<Vec<_>>();
        merged.sort_unstable_by_key(|(_, bytes)| bytes.iter().next().unwrap_or(u8::MAX));
        for (targets, bytes) in merged {
            if !visit(bytes, targets.as_slice()) {
                return false;
            }
        }
        true
    }

    /// Fully determinize the current runtime tokenizer by exact subset
    /// construction.  Each returned DFA state carries the epsilon-closed set of
    /// source states it represents so callers can transport the already-final
    /// tokenizer-state ID map without rebuilding compiler analyses.
    pub fn try_full_determinization(
        &self,
        state_limit: usize,
        transition_limit: usize,
    ) -> Option<FullTokenizerDeterminization> {
        // Virtual states live outside the materialized DFA state array. A
        // physical subset construction cannot represent or epsilon-close
        // those coordinates, so it must never be used as an implicit
        // materialization fallback for any virtual runtime family.
        if self.virtual_unit_repeat.is_some()
            || !self.virtual_repeat_intersections.is_empty()
            || !self.virtual_residuals.is_empty()
            || state_limit == 0
            || transition_limit == 0
            || !self.has_epsilon_transitions()
        {
            return None;
        }

        let closures = self.all_singleton_epsilon_closures();
        let start = closures[self.initial_state_id() as usize].to_vec();
        if start.is_empty() {
            return None;
        }

        let mut dfa = DFA::new(1);
        dfa.ensure_group_capacity(self.num_terminals as usize);
        for terminal in 0..self.num_terminals {
            dfa.set_group_u8set(
                terminal,
                *self.dfa.group_id_to_u8set(terminal),
            );
        }

        let metadata = |subset: &[u32]| {
            let mut finalizers = BitSet::new(self.num_terminals as usize);
            let mut futures = BitSet::new(self.num_terminals as usize);
            for &state in subset {
                // Structural tokenizer composition may append DFA components
                // whose per-state metadata bitsets were created in a smaller
                // local terminal domain. Their set bits are already rebased to
                // global group IDs, but the backing BitSet width need not have
                // been eagerly widened on every historical state. Product
                // construction needs set union, not identical storage widths;
                // materialize that union into the known global domain.
                for terminal in self.state_finalizers(state).iter() {
                    finalizers.set(terminal);
                }
                for terminal in self.state_futures(state).iter() {
                    futures.set(terminal);
                }
            }
            (finalizers, futures)
        };
        let (start_finalizers, start_futures) = metadata(&start);
        dfa.overwrite_state_metadata(0, start_finalizers, start_futures);

        let start: Box<[u32]> = start.into_boxed_slice();
        let mut source_subsets = vec![start.clone()];
        let mut state_by_subset = FxHashMap::<Box<[u32]>, u32>::default();
        state_by_subset.insert(start, 0);
        let mut worklist = VecDeque::from([0u32]);
        let mut transitions_built = 0usize;
        let mut byte_targets: [SmallVec<[u32; 8]>; 256] =
            std::array::from_fn(|_| SmallVec::new());
        while let Some(determinized_state) = worklist.pop_front() {
            let subset = source_subsets[determinized_state as usize].clone();
            let mut transitions = Vec::<(u8, u32)>::new();
            let completed = self.for_each_determinization_subset_successor(
                &subset,
                closures.as_ref(),
                &mut byte_targets,
                |byte, closed| {
                    transitions_built = transitions_built.saturating_add(1);
                    if transitions_built > transition_limit {
                        return false;
                    }
                    let target = if let Some(&existing) = state_by_subset.get(closed) {
                        existing
                    } else {
                        if source_subsets.len() >= state_limit {
                            return false;
                        }
                        let new_state = dfa.add_state();
                        debug_assert_eq!(new_state as usize, source_subsets.len());
                        let (finalizers, futures) = metadata(closed);
                        dfa.overwrite_state_metadata(new_state, finalizers, futures);
                        let closed = closed.to_vec().into_boxed_slice();
                        state_by_subset.insert(closed.clone(), new_state);
                        source_subsets.push(closed);
                        worklist.push_back(new_state);
                        new_state
                    };
                    transitions.push((byte, target));
                    true
                },
            );
            if !completed {
                return None;
            }
            dfa.set_transitions_from_sorted_entries(determinized_state, transitions);
        }

        let mut source_by_closure = FxHashMap::<Box<[u32]>, u32>::default();
        // Prefer the true initial state when another raw state happens to have
        // the same closure: accumulator state keys are observable at commit.
        let initial = self.initial_state_id();
        source_by_closure.insert(
            closures[initial as usize].to_vec().into_boxed_slice(),
            initial,
        );
        for (state, closure) in closures.iter().enumerate() {
            source_by_closure
                .entry(closure.to_vec().into_boxed_slice())
                .or_insert(state as u32);
        }
        let exact_source_states = source_subsets
            .iter()
            .map(|subset| source_by_closure.get(subset).copied().unwrap_or(u32::MAX))
            .collect();

        Some(FullTokenizerDeterminization {
            tokenizer: Tokenizer {
                dfa,
                num_terminals: self.num_terminals,
                packed_runtime_transitions: None,
                packed_runtime_transition_segments: Arc::from([]),
                compressed_transition_segments: Arc::from([]),
                packed_runtime_metadata: None,
                packed_runtime_metadata_segments: Arc::from([]),
                packed_compressed_transition_segments: Arc::from([]),
                virtual_unit_repeat: None,
                virtual_repeat_intersections: Vec::new(),
                virtual_residuals: Vec::new(),
                exprs: self.exprs.clone(),
                terminal_residual_coordinates: None,
                singleton_epsilon_closures: OnceLock::new(),
                matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
                all_self_loop_bytes_cache: OnceLock::new(),
                transition_count_cache: OnceLock::new(),
                forced_minimized_state_count_cache: OnceLock::new(),
                scalar_deterministic_dispatch_cache: OnceLock::new(),
                sorted_dispatch_roots_cache: OnceLock::new(),
                state_first_bytes_cache: OnceLock::new(),
            },
            source_subsets,
            source_state_offset: u32::MAX,
            exact_source_states,
        })
    }

    /// Fully determinize the physical tokenizer from every raw runtime state.
    ///
    /// The ordinary full determinization only contains subsets reachable from
    /// the global reset state. Dynamic masking can begin at any exact raw lexer
    /// state retained by the committed parser frontier, so a mask-only
    /// deterministic coordinate also needs every singleton epsilon closure as
    /// a legal entry state. This extends the ordinary exact subset DFA with
    /// those additional roots and anything reachable from them.
    pub fn try_full_determinization_all_starts(
        &self,
        state_limit: usize,
        transition_limit: usize,
    ) -> Option<(FullTokenizerDeterminization, Vec<u32>)> {
        // Every singleton epsilon closure is a required legal entry state in
        // the all-starts coordinate.  When the epsilon graph is acyclic those
        // closures are necessarily distinct: equal closures for two distinct
        // raw states would make each state epsilon-reachable from the other,
        // hence form a cycle.  Therefore an acyclic source with more raw
        // states than the output state budget provably cannot fit.  Reject it
        // before paying for subset construction; cyclic graphs deliberately
        // retain the general path because an SCC can collapse several raw
        // states to one singleton closure.
        if self.num_states() as usize > state_limit && self.epsilon_graph_is_acyclic() {
            return None;
        }
        let mut built = self.try_full_determinization(state_limit, transition_limit)?;
        let closures = self.all_singleton_epsilon_closures();
        let mut state_by_subset = FxHashMap::<Box<[u32]>, u32>::default();
        state_by_subset.reserve(state_limit.min(closures.len().saturating_mul(2)));
        for (state, subset) in built.source_subsets.iter().enumerate() {
            state_by_subset.insert(subset.clone(), state as u32);
        }

        let metadata = |subset: &[u32]| {
            let mut finalizers = BitSet::new(self.num_terminals as usize);
            let mut futures = BitSet::new(self.num_terminals as usize);
            for &state in subset {
                for terminal in self.state_finalizers(state).iter() {
                    finalizers.set(terminal);
                }
                for terminal in self.state_futures(state).iter() {
                    futures.set(terminal);
                }
            }
            (finalizers, futures)
        };

        let mut full_to_determinized = vec![u32::MAX; closures.len()];
        let mut worklist = VecDeque::<u32>::new();
        for (raw_state, closure) in closures.iter().enumerate() {
            if let Some(&state) = state_by_subset.get(closure.as_ref()) {
                full_to_determinized[raw_state] = state;
                continue;
            }
            if built.source_subsets.len() >= state_limit {
                return None;
            }
            let state = built.tokenizer.dfa.add_state();
            debug_assert_eq!(state as usize, built.source_subsets.len());
            let key = closure.to_vec().into_boxed_slice();
            let (finalizers, futures) = metadata(&key);
            built
                .tokenizer
                .dfa
                .overwrite_state_metadata(state, finalizers, futures);
            state_by_subset.insert(key.clone(), state);
            built.source_subsets.push(key);
            built.exact_source_states.push(raw_state as u32);
            full_to_determinized[raw_state] = state;
            worklist.push_back(state);
        }

        let mut transitions_built = (0..built.tokenizer.num_states())
            .map(|state| built.tokenizer.transitions_from(state).count())
            .sum::<usize>();
        let mut byte_targets: [SmallVec<[u32; 8]>; 256] =
            std::array::from_fn(|_| SmallVec::new());
        while let Some(determinized_state) = worklist.pop_front() {
            let subset = built.source_subsets[determinized_state as usize].clone();
            let mut transitions = Vec::<(u8, u32)>::new();
            let completed = self.for_each_determinization_subset_successor(
                &subset,
                closures.as_ref(),
                &mut byte_targets,
                |byte, closed| {
                    transitions_built = transitions_built.saturating_add(1);
                    if transitions_built > transition_limit {
                        return false;
                    }
                    let target = if let Some(&existing) = state_by_subset.get(closed) {
                        existing
                    } else {
                        if built.source_subsets.len() >= state_limit {
                            return false;
                        }
                        let new_state = built.tokenizer.dfa.add_state();
                        debug_assert_eq!(new_state as usize, built.source_subsets.len());
                        let (finalizers, futures) = metadata(closed);
                        built
                            .tokenizer
                            .dfa
                            .overwrite_state_metadata(new_state, finalizers, futures);
                        let closed = closed.to_vec().into_boxed_slice();
                        state_by_subset.insert(closed.clone(), new_state);
                        built.source_subsets.push(closed);
                        built.exact_source_states.push(u32::MAX);
                        worklist.push_back(new_state);
                        new_state
                    };
                    transitions.push((byte, target));
                    true
                },
            );
            if !completed {
                return None;
            }
            built
                .tokenizer
                .dfa
                .set_transitions_from_sorted_entries(determinized_state, transitions);
        }

        // Some raw states share an epsilon closure. Preserve one exact scalar
        // representative per deterministic subset for existing source-fallback
        // users, while the returned dense map retains every raw entry state.
        let initial = self.initial_state_id();
        let mut source_by_closure = FxHashMap::<Box<[u32]>, u32>::default();
        source_by_closure.insert(
            closures[initial as usize].to_vec().into_boxed_slice(),
            initial,
        );
        for (state, closure) in closures.iter().enumerate() {
            source_by_closure
                .entry(closure.to_vec().into_boxed_slice())
                .or_insert(state as u32);
        }
        built.exact_source_states = built
            .source_subsets
            .iter()
            .map(|subset| source_by_closure.get(subset).copied().unwrap_or(u32::MAX))
            .collect();
        built.tokenizer.invalidate_derived_caches();
        Some((built, full_to_determinized))
    }

    /// Exact finite-horizon determinization which preserves existing raw state
    /// IDs whenever possible. Each raw state is reinterpreted as its singleton
    /// epsilon closure at the same ID; states without epsilon fan-out therefore
    /// keep their existing deterministic rows unchanged. Only genuinely new
    /// multi-state subsets are appended. This is substantially cheaper for the
    /// common partitioned-lexer shape where a small epsilon dispatcher fans out
    /// into large independently deterministic components.
    pub fn try_reusing_horizon_determinization_all_starts(
        &self,
        horizon: usize,
        state_limit: usize,
        transition_limit: usize,
    ) -> Option<(FullTokenizerDeterminization, Vec<u32>)> {
        if self.has_any_virtual_runtime()
            || horizon == 0
            || state_limit == 0
            || transition_limit == 0
            || !self.has_epsilon_transitions()
        {
            return None;
        }

        let closures = self.all_singleton_epsilon_closures();
        let raw_state_count = self.num_states() as usize;
        if raw_state_count == 0
            || raw_state_count > state_limit
            || closures.len() != raw_state_count
        {
            return None;
        }

        // Materialize compressed component rows once, then retain every raw ID.
        // Epsilon edges are removed only after their closure has been captured.
        let mut dfa = self.materialized_dfa();
        if dfa.num_states() != raw_state_count {
            return None;
        }
        for state in dfa.states_mut() {
            state.epsilon_transitions.clear();
        }

        let metadata = |subset: &[u32]| {
            let mut finalizers = BitSet::new(self.num_terminals as usize);
            let mut futures = BitSet::new(self.num_terminals as usize);
            for &state in subset {
                for terminal in self.state_finalizers(state).iter() {
                    finalizers.set(terminal);
                }
                for terminal in self.state_futures(state).iter() {
                    futures.set(terminal);
                }
            }
            (finalizers, futures)
        };

        let mut source_subsets = Vec::<Box<[u32]>>::with_capacity(raw_state_count);
        let mut exact_source_states = Vec::<u32>::with_capacity(raw_state_count);
        let mut state_by_subset = FxHashMap::<Box<[u32]>, u32>::default();
        state_by_subset.reserve(raw_state_count.saturating_mul(2).min(state_limit));
        let mut worklist = VecDeque::<(u32, usize)>::new();

        for (raw_state, closure) in closures.iter().enumerate() {
            let raw_state = raw_state as u32;
            let key = closure.to_vec().into_boxed_slice();
            source_subsets.push(key.clone());
            exact_source_states.push(raw_state);
            // Any representative of an identical closure is exact. Keep the
            // first so successor lookup remains stable while every raw entry
            // itself still maps to its preserved state ID below.
            state_by_subset.entry(key.clone()).or_insert(raw_state);
            if key.len() != 1 || key[0] != raw_state {
                let (finalizers, futures) = metadata(&key);
                dfa.overwrite_state_metadata(raw_state, finalizers, futures);
                worklist.push_back((raw_state, 0));
            }
        }

        let mut transitions_built = 0usize;
        let mut byte_targets: [SmallVec<[u32; 8]>; 256] =
            std::array::from_fn(|_| SmallVec::new());
        let singleton_closed_rows =
            Self::singleton_closed_transition_groups(&dfa, closures.as_ref());
        // Raw states with non-singleton epsilon closures must receive rewritten
        // deterministic rows, but those same raw rows remain the immutable
        // source semantics for every later subset expansion. Delay only those
        // row writes until construction is complete; appended deterministic
        // states can be populated immediately.
        let mut raw_transition_overrides = Vec::<(u32, Vec<(u8, u32)>)>::new();
        while let Some((determinized_state, depth)) = worklist.pop_front() {
            if depth >= horizon {
                continue;
            }
            let subset = source_subsets[determinized_state as usize].clone();

            let mut transitions = Vec::<(u8, u32)>::new();
            let mut pending_states = Vec::new();
            let mut intern_closed = |closed: &[u32]| -> Option<u32> {
                if let Some(&existing) = state_by_subset.get(closed) {
                    return Some(existing);
                }
                if source_subsets.len() >= state_limit {
                    return None;
                }
                let new_state = source_subsets.len() as u32;
                let (finalizers, futures) = metadata(closed);
                let closed = closed.to_vec().into_boxed_slice();
                state_by_subset.insert(closed.clone(), new_state);
                source_subsets.push(closed);
                exact_source_states.push(u32::MAX);
                worklist.push_back((new_state, depth + 1));
                pending_states.push((new_state, finalizers, futures));
                Some(new_state)
            };
            let completed = if let Some(grouped_rows) = singleton_closed_rows.as_deref() {
                Self::for_each_singleton_closed_subset_successor(
                    &subset,
                    grouped_rows,
                    |bytes, closed| {
                        transitions_built = transitions_built.saturating_add(bytes.len());
                        if transitions_built > transition_limit {
                            return false;
                        }
                        let Some(target) = intern_closed(closed) else {
                            return false;
                        };
                        transitions.extend(bytes.iter().map(|byte| (byte, target)));
                        true
                    },
                )
            } else {
                Self::for_each_materialized_determinization_subset_successor(
                    &dfa,
                    &subset,
                    closures.as_ref(),
                    &mut byte_targets,
                    |byte, closed| {
                        transitions_built = transitions_built.saturating_add(1);
                        if transitions_built > transition_limit {
                            return false;
                        }
                        let Some(target) = intern_closed(closed) else {
                            return false;
                        };
                        transitions.push((byte, target));
                        true
                    },
                )
            };
            if !completed {
                return None;
            }
            for (new_state, finalizers, futures) in pending_states {
                let actual = dfa.add_state();
                debug_assert_eq!(actual, new_state);
                dfa.overwrite_state_metadata(new_state, finalizers, futures);
            }
            if determinized_state < raw_state_count as u32 {
                raw_transition_overrides.push((determinized_state, transitions));
            } else {
                dfa.set_transitions_from_sorted_entries(determinized_state, transitions);
            }
        }
        for (state, transitions) in raw_transition_overrides {
            dfa.set_transitions_from_sorted_entries(state, transitions);
        }


        // Existing scalar rows remain valid because a byte target raw ID now
        // denotes that target's exact epsilon closure at the same ID. Newly
        // appended rows were constructed directly in that same coordinate.
        let mut tokenizer = Tokenizer::from_parts(
            dfa,
            self.num_terminals,
            self.exprs.clone(),
        );
        tokenizer.invalidate_derived_caches();
        let full_to_determinized = (0..raw_state_count as u32).collect::<Vec<_>>();
        Some((
            FullTokenizerDeterminization {
                tokenizer,
                source_subsets,
                source_state_offset: u32::MAX,
                exact_source_states,
            },
            full_to_determinized,
        ))
    }

    /// Determinize every raw runtime entry state only to the finite byte
    /// horizon observable by one mask walk. Unlike full all-start
    /// determinization, this deliberately does not construct transitions out
    /// of subsets first reached at `horizon`: no vocabulary token can observe
    /// them. The returned states and metadata are otherwise exact subset-DFA
    /// coordinates, so callers may use them exactly like a full
    /// determinization during a bounded vocabulary walk.
    pub fn try_horizon_determinization_all_starts(
        &self,
        horizon: usize,
        state_limit: usize,
        transition_limit: usize,
    ) -> Option<(FullTokenizerDeterminization, Vec<u32>)> {
        if self.virtual_unit_repeat.is_some()
            || !self.virtual_repeat_intersections.is_empty()
            || !self.virtual_residuals.is_empty()
            || horizon == 0
            || state_limit == 0
            || transition_limit == 0
            || !self.has_epsilon_transitions()
        {
            return None;
        }

        let closures = self.all_singleton_epsilon_closures();
        let initial = self.initial_state_id();
        let initial_closure = closures.get(initial as usize)?.to_vec().into_boxed_slice();
        if initial_closure.is_empty() {
            return None;
        }

        let mut dfa = DFA::new(1);
        dfa.ensure_group_capacity(self.num_terminals as usize);
        for terminal in 0..self.num_terminals {
            dfa.set_group_u8set(terminal, *self.dfa.group_id_to_u8set(terminal));
        }

        let metadata = |subset: &[u32]| {
            let mut finalizers = BitSet::new(self.num_terminals as usize);
            let mut futures = BitSet::new(self.num_terminals as usize);
            for &state in subset {
                for terminal in self.state_finalizers(state).iter() {
                    finalizers.set(terminal);
                }
                for terminal in self.state_futures(state).iter() {
                    futures.set(terminal);
                }
            }
            (finalizers, futures)
        };

        let (initial_finalizers, initial_futures) = metadata(&initial_closure);
        dfa.overwrite_state_metadata(0, initial_finalizers, initial_futures);
        let mut source_subsets = vec![initial_closure.clone()];
        let mut exact_source_states = vec![initial];
        let mut state_by_subset = FxHashMap::<Box<[u32]>, u32>::default();
        state_by_subset.insert(initial_closure, 0);
        let mut full_to_determinized = vec![u32::MAX; closures.len()];
        full_to_determinized[initial as usize] = 0;

        // Seed every raw runtime state before expanding anything. This makes
        // each singleton epsilon closure a depth-zero root, exactly matching
        // the fact that a committed dynamic state may begin its next mask walk
        // at any raw lexer coordinate.
        let mut worklist = VecDeque::<(u32, usize)>::new();
        worklist.push_back((0, 0));
        for (raw_state, closure) in closures.iter().enumerate() {
            if raw_state == initial as usize {
                continue;
            }
            if let Some(&state) = state_by_subset.get(closure.as_ref()) {
                full_to_determinized[raw_state] = state;
                continue;
            }
            if source_subsets.len() >= state_limit {
                return None;
            }
            let key = closure.to_vec().into_boxed_slice();
            let state = dfa.add_state();
            debug_assert_eq!(state as usize, source_subsets.len());
            let (finalizers, futures) = metadata(&key);
            dfa.overwrite_state_metadata(state, finalizers, futures);
            state_by_subset.insert(key.clone(), state);
            source_subsets.push(key);
            exact_source_states.push(raw_state as u32);
            full_to_determinized[raw_state] = state;
            worklist.push_back((state, 0));
        }

        let mut transitions_built = 0usize;
        let mut byte_targets: [SmallVec<[u32; 8]>; 256] =
            std::array::from_fn(|_| SmallVec::new());
        while let Some((determinized_state, depth)) = worklist.pop_front() {
            if depth >= horizon {
                continue;
            }
            let subset = source_subsets[determinized_state as usize].clone();
            let mut transitions = Vec::<(u8, u32)>::new();
            let completed = self.for_each_determinization_subset_successor(
                &subset,
                closures.as_ref(),
                &mut byte_targets,
                |byte, closed| {
                    transitions_built = transitions_built.saturating_add(1);
                    if transitions_built > transition_limit {
                        return false;
                    }
                    let target = if let Some(&existing) = state_by_subset.get(closed) {
                        existing
                    } else {
                        if source_subsets.len() >= state_limit {
                            return false;
                        }
                        let new_state = dfa.add_state();
                        debug_assert_eq!(new_state as usize, source_subsets.len());
                        let (finalizers, futures) = metadata(closed);
                        dfa.overwrite_state_metadata(new_state, finalizers, futures);
                        let closed = closed.to_vec().into_boxed_slice();
                        state_by_subset.insert(closed.clone(), new_state);
                        source_subsets.push(closed);
                        exact_source_states.push(u32::MAX);
                        worklist.push_back((new_state, depth + 1));
                        new_state
                    };
                    transitions.push((byte, target));
                    true
                },
            );
            if !completed {
                return None;
            }
            dfa.set_transitions_from_sorted_entries(determinized_state, transitions);
        }

        // Prefer a real raw-state representative for subsets which correspond
        // exactly to one singleton epsilon closure. This is diagnostic/fallback
        // metadata only; `full_to_determinized` above retains every raw entry.
        let mut source_by_closure = FxHashMap::<Box<[u32]>, u32>::default();
        source_by_closure.insert(
            closures[initial as usize].to_vec().into_boxed_slice(),
            initial,
        );
        for (state, closure) in closures.iter().enumerate() {
            source_by_closure
                .entry(closure.to_vec().into_boxed_slice())
                .or_insert(state as u32);
        }
        exact_source_states = source_subsets
            .iter()
            .map(|subset| source_by_closure.get(subset).copied().unwrap_or(u32::MAX))
            .collect();

        let mut tokenizer = Tokenizer {
            dfa,
            num_terminals: self.num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: self.exprs.clone(),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
            sorted_dispatch_roots_cache: OnceLock::new(),
            state_first_bytes_cache: OnceLock::new(),
        };
        tokenizer.invalidate_derived_caches();
        Some((
            FullTokenizerDeterminization {
                tokenizer,
                source_subsets,
                source_state_offset: u32::MAX,
                exact_source_states,
            },
            full_to_determinized,
        ))
    }

    /// Move the exact source tokenizer behind a completed subset tokenizer.
    ///
    /// Product states are safe only while one parser language is uniformly
    /// associated with every source state in their subset. Runtime commit can
    /// expand such a state into this appended source coordinate, execute the
    /// historical NFA semantics unchanged, and re-coalesce only exact uniform
    /// subsets afterward. The product start state remains state zero.
    pub fn finish_full_determinization_with_source_fallback(
        &mut self,
        mut built: FullTokenizerDeterminization,
    ) -> FullTokenizerDeterminization {
        debug_assert_eq!(built.source_subsets.len(), built.tokenizer.num_states() as usize);
        debug_assert_eq!(built.exact_source_states.len(), built.source_subsets.len());

        let mut source_dfa = std::mem::replace(&mut self.dfa, DFA::new(0));
        // Immutable disjoint tokenizer composition deliberately permits old
        // component states to retain shorter finalizer/future bitsets: group
        // membership beyond that shorter local domain is simply false. Once
        // those states are appended behind a deterministic runtime product,
        // however, product and fallback states coexist in one live tokenizer
        // frontier and runtime admission unions their metadata directly. Make
        // the source fallback use the product tokenizer's one canonical
        // terminal domain before publishing it.
        source_dfa.ensure_group_capacity(self.num_terminals as usize);
        let global_groups = (0..self.num_terminals as usize).collect::<Vec<_>>();
        built.source_state_offset = built
            .tokenizer
            .dfa
            .append_rebased_component(source_dfa, &global_groups);
        let mut compressed_segments = built
            .tokenizer
            .compressed_transition_segments
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        compressed_segments.extend(self.compressed_transition_segments.iter().cloned().map(
            |mut segment| {
                segment.state_offset = segment
                    .state_offset
                    .checked_add(built.source_state_offset)
                    .expect("runtime tokenizer compressed state offset overflow");
                segment
            },
        ));
        built.tokenizer.compressed_transition_segments = Arc::from(compressed_segments);
        self.compressed_transition_segments = Arc::from([]);
        self.invalidate_derived_caches();
        built.tokenizer.invalidate_derived_caches();
        built
    }

    /// Materialize a deterministic compile-time analysis view as a tokenizer.
    /// The view may be a powerset of this tokenizers epsilon-NFA. State zero is
    /// reserved for the supplied start state, and the returned old-to-new map
    /// lets callers lift raw-start mappings into the materialized coordinate.
    pub fn materialize_deterministic_view(
        &self,
        start_state: usize,
        finalizers: &[Vec<usize>],
        futures: &[Vec<usize>],
        edge_offsets: &[u32],
        edges: &[(u8, u32)],
        active_terminals: &[bool],
    ) -> Option<(Tokenizer, Vec<u32>)> {
        let state_count = finalizers.len();
        if state_count == 0
            || futures.len() != state_count
            || edge_offsets.len() != state_count + 1
            || start_state >= state_count
            || active_terminals.len() != self.num_terminals as usize
        {
            return None;
        }
        let mut new_to_old = Vec::with_capacity(state_count);
        new_to_old.push(start_state);
        new_to_old.extend((0..state_count).filter(|&state| state != start_state));
        let mut old_to_new = vec![u32::MAX; state_count];
        for (new, &old) in new_to_old.iter().enumerate() {
            old_to_new[old] = new as u32;
        }

        let mut dfa = DFA::new(state_count);
        dfa.ensure_group_capacity(self.num_terminals as usize);
        for terminal in 0..self.num_terminals as usize {
            if active_terminals[terminal] {
                dfa.set_group_u8set(
                    terminal as u32,
                    *self.dfa.group_id_to_u8set(terminal as u32),
                );
            }
        }
        for (new_state, &old_state) in new_to_old.iter().enumerate() {
            let start = *edge_offsets.get(old_state)? as usize;
            let end = *edge_offsets.get(old_state + 1)? as usize;
            let transitions = edges
                .get(start..end)?
                .iter()
                .map(|&(byte, target)| {
                    old_to_new
                        .get(target as usize)
                        .copied()
                        .filter(|&target| target != u32::MAX)
                        .map(|target| (byte, target))
                })
                .collect::<Option<Vec<_>>>()?;
            dfa.set_transitions_from_sorted_entries(new_state as u32, transitions);
            let to_bits = |groups: &[usize]| {
                let mut bits = BitSet::new(self.num_terminals as usize);
                for &group in groups {
                    if group >= active_terminals.len() || !active_terminals[group] {
                        return None;
                    }
                    bits.set(group);
                }
                Some(bits)
            };
            dfa.overwrite_state_metadata(
                new_state as u32,
                to_bits(&finalizers[old_state])?,
                to_bits(&futures[old_state])?,
            );
        }
        Some((
            Tokenizer {
                dfa,
                num_terminals: self.num_terminals,
                packed_runtime_transitions: None,
                packed_runtime_transition_segments: Arc::from([]),
                compressed_transition_segments: Arc::from([]),
                packed_runtime_metadata: None,
                packed_runtime_metadata_segments: Arc::from([]),
                packed_compressed_transition_segments: Arc::from([]),
                virtual_unit_repeat: None,
                virtual_repeat_intersections: Vec::new(),
                virtual_residuals: Vec::new(),
                exprs: None,
                terminal_residual_coordinates: None,
                singleton_epsilon_closures: OnceLock::new(),
                matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
                all_self_loop_bytes_cache: OnceLock::new(),
                transition_count_cache: OnceLock::new(),
                forced_minimized_state_count_cache: OnceLock::new(),
                scalar_deterministic_dispatch_cache: OnceLock::new(),
                sorted_dispatch_roots_cache: OnceLock::new(),
                state_first_bytes_cache: OnceLock::new(),
            },
            old_to_new,
        ))
    }

    /// Materialize an exact deterministic quotient for one compile-time branch.
    ///
    /// `original_to_quotient` must be a congruence for every vocabulary-relevant
    /// byte after filtering labels to `active_terminals`. The method verifies
    /// that property over every class member before constructing the smaller
    /// tokenizer, so callers can fail closed to the original tokenizer.
    pub fn materialize_active_quotient(
        &self,
        original_to_quotient: &[u32],
        representatives: &[u32],
        active_terminals: &[bool],
        relevant_bytes: &[bool; 256],
    ) -> Option<Tokenizer> {
        if original_to_quotient.len() != self.num_states() as usize
            || active_terminals.len() != self.num_terminals as usize
            || representatives.is_empty()
            || original_to_quotient.get(self.start_state() as usize).copied() != Some(0)
        {
            return None;
        }
        let quotient_states = representatives.len();
        if original_to_quotient
            .iter()
            .any(|&state| state == u32::MAX || state as usize >= quotient_states)
        {
            return None;
        }

        let filtered = |bits: &BitSet| {
            let mut result = BitSet::new(self.num_terminals as usize);
            for terminal in bits.iter() {
                if active_terminals.get(terminal).copied().unwrap_or(false) {
                    result.set(terminal);
                }
            }
            result
        };

        // Verify output labels and every relevant transition for all members,
        // rather than trusting the refinement implementation as an implicit
        // construction contract.
        let mut class_members = vec![Vec::<u32>::new(); quotient_states];
        for (original, &quotient) in original_to_quotient.iter().enumerate() {
            class_members[quotient as usize].push(original as u32);
        }
        for (class, members) in class_members.iter().enumerate() {
            let representative = *representatives.get(class)?;
            if !members.contains(&representative) {
                return None;
            }
            let representative_finalizers = filtered(self.dfa.finalizers(representative));
            let representative_futures =
                filtered(self.dfa.possible_future_group_ids(representative));
            for &member in members {
                if filtered(self.dfa.finalizers(member)) != representative_finalizers
                    || filtered(self.dfa.possible_future_group_ids(member))
                        != representative_futures
                {
                    return None;
                }
                for byte in 0u16..=255 {
                    if !relevant_bytes[byte as usize] {
                        continue;
                    }
                    let mapped = self
                        .step(member, byte as u8)
                        .map(|target| original_to_quotient[target as usize]);
                    let representative_mapped = self
                        .step(representative, byte as u8)
                        .map(|target| original_to_quotient[target as usize]);
                    if mapped != representative_mapped {
                        return None;
                    }
                }
                let mapped_epsilon = |state: u32| {
                    let mut targets = self.dfa.states()[state as usize]
                        .epsilon_transitions
                        .iter()
                        .map(|&target| original_to_quotient[target as usize])
                        .collect::<Vec<_>>();
                    targets.sort_unstable();
                    targets.dedup();
                    targets
                };
                if mapped_epsilon(member) != mapped_epsilon(representative) {
                    return None;
                }
            }
        }

        let mut dfa = DFA::new(quotient_states);
        dfa.ensure_group_capacity(self.num_terminals as usize);
        for terminal in 0..self.num_terminals as usize {
            if active_terminals[terminal] {
                dfa.set_group_u8set(
                    terminal as u32,
                    *self.dfa.group_id_to_u8set(terminal as u32),
                );
            }
        }
        for (class, &representative) in representatives.iter().enumerate() {
            let transitions = (0u16..=255)
                .filter(|&byte| relevant_bytes[byte as usize])
                .filter_map(|byte| {
                    self.step(representative, byte as u8).map(|target| {
                        (byte as u8, original_to_quotient[target as usize])
                    })
                })
                .collect::<Vec<_>>();
            dfa.set_transitions_from_sorted_entries(class as u32, transitions);
            let mut epsilon_targets = self.dfa.states()[representative as usize]
                .epsilon_transitions
                .iter()
                .map(|&target| original_to_quotient[target as usize])
                .collect::<Vec<_>>();
            epsilon_targets.sort_unstable();
            epsilon_targets.dedup();
            for target in epsilon_targets {
                dfa.add_epsilon_transition(class as u32, target);
            }
            dfa.overwrite_state_metadata(
                class as u32,
                filtered(self.dfa.finalizers(representative)),
                filtered(self.dfa.possible_future_group_ids(representative)),
            );
        }
        Some(Tokenizer {
            dfa,
            num_terminals: self.num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: None,
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
            sorted_dispatch_roots_cache: OnceLock::new(),
            state_first_bytes_cache: OnceLock::new(),
        })
    }

    /// Verify that `source_to_self` is a raw tokenizer homomorphism.
    ///
    /// Whole-vocabulary token equivalence is not sufficient here: callers use
    /// the target tokenizer as the byte-transition coordinate while building a
    /// partition-local terminal DWA. Every mapped source state must therefore
    /// preserve labels, epsilon successors, and every byte successor exactly.
    fn certifies_mapped_prefix_homomorphism_from(
        &self,
        source: &Tokenizer,
        source_to_self: &[u32],
        source_to_rebuilt: &[u32],
    ) -> bool {
        if source.num_terminals != self.num_terminals
            || source_to_self.len() != source.num_states() as usize
            || source_to_rebuilt.len() != source.num_states() as usize
            || source_to_self
                .iter()
                .any(|&state| state as usize >= self.num_states() as usize)
            || source_to_self.get(source.start_state() as usize).copied()
                != Some(self.start_state())
        {
            return false;
        }

        let mut mapped_epsilon = Vec::<u32>::new();
        let mut target_epsilon = Vec::<u32>::new();
        for source_state in 0..source.num_states() {
            // States absent from the rebuilt prefix were appended below by
            // copying their exact metadata and transition rows through the
            // completed source-to-self map. Only states represented by the
            // synthesized prefix can violate the raw homomorphism.
            if source_to_rebuilt[source_state as usize] == u32::MAX {
                continue;
            }
            let target_state = source_to_self[source_state as usize];
            if source.dfa.finalizers(source_state) != self.dfa.finalizers(target_state)
                || source.dfa.possible_future_group_ids(source_state)
                    != self.dfa.possible_future_group_ids(target_state)
            {
                return false;
            }

            mapped_epsilon.clear();
            mapped_epsilon.extend(
                source.dfa.states()[source_state as usize]
                    .epsilon_transitions
                    .iter()
                    .map(|&next| source_to_self[next as usize]),
            );
            mapped_epsilon.sort_unstable();
            mapped_epsilon.dedup();
            target_epsilon.clear();
            target_epsilon.extend_from_slice(
                &self.dfa.states()[target_state as usize].epsilon_transitions,
            );
            target_epsilon.sort_unstable();
            target_epsilon.dedup();
            if mapped_epsilon != target_epsilon {
                return false;
            }

            let mut source_transitions = source.transitions_from(source_state);
            let mut target_transitions = self.transitions_from(target_state);
            loop {
                match (source_transitions.next(), target_transitions.next()) {
                    (None, None) => break,
                    (
                        Some((source_byte, source_next)),
                        Some((target_byte, target_next)),
                    ) if source_byte == target_byte
                        && source_to_self[source_next as usize] == target_next => {}
                    _ => return false,
                }
            }
        }
        true
    }

    /// Extend `self` with the source-only residual states that were appended to
    /// `source` after `rebuilt` was constructed.  `rebuilt_to_self` must be a
    /// structural state map from the rebuilt expression DFA into `self`.
    ///
    /// Protected residual synthesis appends externally-entered product states
    /// to otherwise identical deterministic dispatch components.  The original
    /// component states remain an exact prefix.  Verify that prefix relation
    /// state-for-state, then clone only the appended states while redirecting
    /// every edge through the completed source-to-self map.  The result is a
    /// transition homomorphism over the actual source tokenizer, not a bounded
    /// semantic approximation.
    pub fn augment_from_verified_component_prefixes(
        &mut self,
        source: &Tokenizer,
        rebuilt: &Tokenizer,
        rebuilt_to_self: &[u32],
    ) -> Option<Vec<u32>> {
        if source.num_terminals != rebuilt.num_terminals
            || source.num_terminals != self.num_terminals
            || rebuilt_to_self.len() != rebuilt.num_states() as usize
        {
            return None;
        }

        let source_components = source.disjoint_dispatch_components()?;
        let rebuilt_components = rebuilt.disjoint_dispatch_components()?;
        if source_components.len() != rebuilt_components.len() {
            return None;
        }

        let mut source_to_rebuilt = vec![u32::MAX; source.num_states() as usize];
        source_to_rebuilt[source.start_state() as usize] = rebuilt.start_state();
        for (source_states, rebuilt_states) in
            source_components.iter().zip(&rebuilt_components)
        {
            if rebuilt_states.len() > source_states.len() {
                return None;
            }
            for (&source_state, &rebuilt_state) in source_states.iter().zip(rebuilt_states) {
                source_to_rebuilt[source_state as usize] = rebuilt_state;
            }
        }

        // Verify that the mapped prefix is exactly the rebuilt DFA after state
        // renumbering.  This guards the append-only invariant rather than
        // relying on component construction order as an undocumented fact.
        for (source_state, &rebuilt_state) in source_to_rebuilt.iter().enumerate() {
            if rebuilt_state == u32::MAX {
                continue;
            }
            let source_state = source_state as u32;
            if source.dfa.finalizers(source_state) != rebuilt.dfa.finalizers(rebuilt_state)
                || source.dfa.possible_future_group_ids(source_state)
                    != rebuilt.dfa.possible_future_group_ids(rebuilt_state)
                || source.state_has_epsilon_transitions(source_state)
                    != rebuilt.state_has_epsilon_transitions(rebuilt_state)
            {
                return None;
            }
            let source_epsilon = &source.dfa.states()[source_state as usize].epsilon_transitions;
            let rebuilt_epsilon = &rebuilt.dfa.states()[rebuilt_state as usize].epsilon_transitions;
            let mapped_epsilon = source_epsilon
                .iter()
                .map(|&target| *source_to_rebuilt.get(target as usize).unwrap_or(&u32::MAX))
                .collect::<Vec<_>>();
            if mapped_epsilon != *rebuilt_epsilon {
                return None;
            }
            let source_transitions = source
                .transitions_from(source_state)
                .map(|(byte, target)| {
                    Some((byte, *source_to_rebuilt.get(target as usize)?))
                })
                .collect::<Option<Vec<_>>>()?;
            if source_transitions.iter().any(|&(_, target)| target == u32::MAX)
                || source_transitions
                    != rebuilt.transitions_from(rebuilt_state).collect::<Vec<_>>()
            {
                return None;
            }
        }

        for (source_state, &rebuilt_state) in source_to_rebuilt.iter().enumerate() {
            if rebuilt_state == u32::MAX
                && source.state_has_epsilon_transitions(source_state as u32)
            {
                return None;
            }
        }

        self.invalidate_derived_caches();
        let original_self_states = self.num_states() as usize;
        let mut source_to_self = vec![u32::MAX; source.num_states() as usize];
        for (source_state, &rebuilt_state) in source_to_rebuilt.iter().enumerate() {
            if rebuilt_state != u32::MAX {
                source_to_self[source_state] = *rebuilt_to_self.get(rebuilt_state as usize)?;
            }
        }
        for source_state in 0..source.num_states() as usize {
            if source_to_self[source_state] == u32::MAX {
                source_to_self[source_state] = self.dfa.add_state();
            }
        }

        for source_state in 0..source.num_states() as usize {
            if source_to_rebuilt[source_state] != u32::MAX {
                continue;
            }
            let target_state = source_to_self[source_state];
            let source_state_u32 = source_state as u32;
            let transitions = source
                .transitions_from(source_state_u32)
                .map(|(byte, target)| (byte, source_to_self[target as usize]))
                .collect::<Vec<_>>();
            self.dfa
                .set_transitions_from_sorted_entries(target_state, transitions);
            self.dfa.overwrite_state_metadata(
                target_state,
                source.dfa.finalizers(source_state_u32).clone(),
                source
                    .dfa
                    .possible_future_group_ids(source_state_u32)
                    .clone(),
            );
        }
        debug_assert_eq!(
            self.num_states() as usize - original_self_states,
            source_to_rebuilt
                .iter()
                .filter(|&&state| state == u32::MAX)
                .count(),
        );
        if !self.certifies_mapped_prefix_homomorphism_from(
            source,
            &source_to_self,
            &source_to_rebuilt,
        ) {
            return None;
        }
        Some(source_to_self)
    }

    pub(super) fn from_parts(
        dfa: DFA,
        num_terminals: u32,
        exprs: Option<Arc<[Expr]>>,
    ) -> Self {
        Self {
            dfa,
            num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs,
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
            sorted_dispatch_roots_cache: OnceLock::new(),
            state_first_bytes_cache: OnceLock::new(),
        }
    }

    pub fn from_parts_with_compressed_transitions(
        dfa: DFA,
        num_terminals: u32,
        exprs: Option<Arc<[Expr]>>,
        compressed_transition_segments: Vec<CompressedTransitionSegment>,
    ) -> Self {
        debug_assert!(compressed_transition_segments
            .windows(2)
            .all(|pair| pair[0].state_offset + pair[0].state_count <= pair[1].state_offset));
        Self {
            dfa,
            num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from(compressed_transition_segments),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs,
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: OnceLock::new(),
            matched_terminals_cache: OnceLock::new(),
            initial_byte_frontiers: OnceLock::new(),
            all_self_loop_bytes_cache: OnceLock::new(),
            transition_count_cache: OnceLock::new(),
            forced_minimized_state_count_cache: OnceLock::new(),
            scalar_deterministic_dispatch_cache: OnceLock::new(),
            sorted_dispatch_roots_cache: OnceLock::new(),
            state_first_bytes_cache: OnceLock::new(),
        }
    }

    fn compressed_segment_for_state(
        &self,
        state: u32,
    ) -> Option<&CompressedTransitionSegment> {
        let index = self
            .compressed_transition_segments
            .partition_point(|segment| segment.state_offset <= state);
        index.checked_sub(1).and_then(|index| {
            let segment = &self.compressed_transition_segments[index];
            segment.contains_state(state).then_some(segment)
        })
    }

    #[inline]
    fn packed_compressed_segment_for_state(
        &self,
        state: u32,
    ) -> Option<&PackedCompressedTransitionSegment> {
        let index = self
            .packed_compressed_transition_segments
            .partition_point(|segment| segment.state_offset <= state);
        index.checked_sub(1).and_then(|index| {
            let segment = &self.packed_compressed_transition_segments[index];
            segment.contains_state(state).then_some(segment)
        })
    }

    #[inline]
    fn packed_runtime_transition_segment_for_state(
        &self,
        state: u32,
    ) -> Option<&PackedRuntimeTransitionSegment> {
        let index = self
            .packed_runtime_transition_segments
            .partition_point(|segment| segment.state_offset <= state);
        index.checked_sub(1).and_then(|index| {
            let segment = &self.packed_runtime_transition_segments[index];
            segment.contains_state(state).then_some(segment)
        })
    }

    #[inline]
    fn packed_runtime_metadata_segment_for_state(
        &self,
        state: u32,
    ) -> Option<&PackedTokenizerMetadataSegment> {
        let index = self
            .packed_runtime_metadata_segments
            .partition_point(|segment| segment.state_offset <= state);
        index.checked_sub(1).and_then(|index| {
            let segment = &self.packed_runtime_metadata_segments[index];
            segment.contains_state(state).then_some(segment)
        })
    }

    #[inline]
    pub fn has_packed_runtime_metadata(&self) -> bool {
        self.packed_runtime_metadata.is_some() || !self.packed_runtime_metadata_segments.is_empty()
    }

    pub fn has_compressed_transition_state(&self, state: u32) -> bool {
        self.compressed_segment_for_state(state).is_some()
            || self.packed_compressed_segment_for_state(state).is_some()
    }

    #[inline]
    pub fn has_packed_runtime_transitions(&self) -> bool {
        self.packed_runtime_transitions.is_some()
            || !self.packed_runtime_transition_segments.is_empty()
    }

    /// Return whether runtime transitions borrow their byte rows directly from
    /// an artifact backing. Fresh compiler tokenizers can also own packed
    /// transition sidecars, so this is intentionally narrower than
    /// `has_packed_runtime_transitions()`.
    pub fn has_backed_runtime_transitions(&self) -> bool {
        self.packed_runtime_transitions.as_deref().is_some_and(|packed| {
            matches!(packed.bytes, PackedRuntimeBytes::Backed { .. })
        }) || self.packed_runtime_transition_segments.iter().any(|segment| {
            matches!(segment.transitions.bytes, PackedRuntimeBytes::Backed { .. })
        })
    }

    /// Move packed observation/epsilon metadata back into the structural DFA
    /// without expanding packed byte-transition rows. This is needed before a
    /// tokenizer is structurally mutated: packed metadata is otherwise the
    /// authoritative source for epsilon closure and would shadow newly added
    /// DFA epsilon edges.
    fn materialize_runtime_metadata_for_structural_mutation(&mut self) {
        if let Some(metadata) = self.packed_runtime_metadata.take() {
            while self.dfa.num_states() < metadata.state_count as usize {
                self.dfa.add_state();
            }
            for state in 0..metadata.state_count {
                let finalizers = metadata
                    .finalizers(state)
                    .expect("packed tokenizer metadata covers every state")
                    .clone();
                let futures = metadata
                    .futures(state)
                    .expect("packed tokenizer metadata covers every state")
                    .clone();
                self.dfa.overwrite_state_metadata(state, finalizers, futures);
                for &target in metadata.epsilon_targets(state) {
                    let already_present = self
                        .dfa
                        .states()
                        .get(state as usize)
                        .is_some_and(|row| row.epsilon_transitions.contains(&target));
                    if !already_present {
                        self.dfa.add_epsilon_transition(state, target);
                    }
                }
            }
        }

        let segments = std::mem::take(&mut self.packed_runtime_metadata_segments);
        for segment in segments.iter() {
            for local_state in 0..segment.metadata.state_count {
                let state = segment.state_offset + local_state;
                let finalizers = segment
                    .metadata
                    .finalizers(local_state)
                    .expect("packed tokenizer metadata segment covers every state")
                    .clone();
                let futures = segment
                    .metadata
                    .futures(local_state)
                    .expect("packed tokenizer metadata segment covers every state")
                    .clone();
                self.dfa.overwrite_state_metadata(state, finalizers, futures);
                for &local_target in segment.metadata.epsilon_targets(local_state) {
                    let target = segment
                        .state_offset
                        .checked_add(local_target)
                        .expect("packed tokenizer epsilon target overflow");
                    let already_present = self
                        .dfa
                        .states()
                        .get(state as usize)
                        .is_some_and(|row| row.epsilon_transitions.contains(&target));
                    if !already_present {
                        self.dfa.add_epsilon_transition(state, target);
                    }
                }
            }
        }
    }

    /// Restore physical state metadata and ordinary rows only. Compressed
    /// regions are left in their sidecars so serializers need not expand a
    /// small class alphabet into hundreds of byte transitions per state.
    fn materialized_noncompressed_dfa(&self) -> DFA {
        let mut dfa = self.dfa.clone();
        while dfa.num_states() < self.num_states() as usize {
            dfa.add_state();
        }
        if let Some(metadata) = self.packed_runtime_metadata.as_deref() {
            for state in 0..metadata.state_count {
                dfa.overwrite_state_metadata(
                    state,
                    metadata.finalizers(state).expect("packed metadata covers every state").clone(),
                    metadata.futures(state).expect("packed metadata covers every state").clone(),
                );
                for &target in metadata.epsilon_targets(state) {
                    dfa.add_epsilon_transition(state, target);
                }
            }
        }
        for segment in self.packed_runtime_metadata_segments.iter() {
            for local_state in 0..segment.metadata.state_count {
                let state = segment.state_offset + local_state;
                dfa.overwrite_state_metadata(
                    state,
                    segment.metadata.finalizers(local_state).expect("packed metadata segment covers every state").clone(),
                    segment.metadata.futures(local_state).expect("packed metadata segment covers every state").clone(),
                );
                for &target in segment.metadata.epsilon_targets(local_state) {
                    dfa.add_epsilon_transition(state, segment.state_offset + target);
                }
            }
        }
        if let Some(packed) = &self.packed_runtime_transitions {
            for state in 0..packed.state_count() as u32 {
                if let Some((bytes, targets)) = packed.row(state) {
                    dfa.set_transitions_from_sorted_entries(
                        state,
                        bytes
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(index, byte)| {
                                (
                                    byte,
                                    targets
                                        .get(index)
                                        .expect("packed tokenizer row lengths were validated"),
                                )
                            })
                            .collect(),
                    );
                }
            }
        }
        for segment in self.packed_runtime_transition_segments.iter() {
            for local_state in 0..segment.transitions.state_count() as u32 {
                let state = segment.state_offset + local_state;
                if let Some((bytes, targets)) = segment.transitions.row(local_state) {
                    dfa.set_transitions_from_sorted_entries(
                        state,
                        bytes
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(index, byte)| {
                                (
                                    byte,
                                    segment.state_offset
                                        + targets
                                            .get(index)
                                            .expect("packed tokenizer row lengths were validated"),
                                )
                            })
                            .collect(),
                    );
                }
            }
        }
        dfa
    }

    fn materialized_dfa(&self) -> DFA {
        let mut dfa = self.materialized_noncompressed_dfa();
        for segment in self.compressed_transition_segments.iter() {
            for local_state in 0..segment.state_count {
                let state = segment.state_offset + local_state;
                dfa.set_transitions_from_sorted_entries(state, segment.expanded_entries(state));
            }
        }
        for segment in self.packed_compressed_transition_segments.iter() {
            for local_state in 0..segment.state_count {
                let state = segment.state_offset + local_state;
                let mut row = [u32::MAX; 256];
                segment.fill_transition_row(state, &mut row);
                dfa.set_transitions_from_sorted_entries(
                    state,
                    row.iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(byte, target)| {
                            (target != u32::MAX).then_some((byte as u8, target))
                        })
                        .collect(),
                );
            }
        }
        dfa
    }

    /// Put two tokenizers with the same terminal-id domain under one fresh
    /// epsilon root without identifying any source states. This lets the exact
    /// state-equivalence machinery compare residual states across independently
    /// built full and synthesized lexers.
    pub fn disjoint_union_for_analysis(
        left: &Tokenizer,
        right: &Tokenizer,
    ) -> TokenizerAnalysisUnion {
        assert_eq!(
            left.num_terminals, right.num_terminals,
            "cross-tokenizer analysis requires one shared terminal-id domain",
        );

        let left_offset = 1u32;
        let right_offset = left_offset + left.dfa.num_states() as u32;
        let mut dfa = DFA::new(
            1usize
                .saturating_add(left.dfa.num_states())
                .saturating_add(right.dfa.num_states()),
        );
        let num_groups = left.num_terminals as usize;
        dfa.ensure_group_capacity(num_groups);

        for group in 0..num_groups {
            let left_set = *left.dfa.group_id_to_u8set(group as u32);
            let right_set = *right.dfa.group_id_to_u8set(group as u32);
            dfa.set_group_u8set(group as u32, left_set.union(&right_set));
        }

        let copy_source = |target: &mut DFA, source: &DFA, offset: u32| {
            for (state_index, state) in source.states().iter().enumerate() {
                let target_state = offset + state_index as u32;
                target.set_transitions_from_sorted_entries(
                    target_state,
                    state
                        .transitions
                        .iter()
                        .map(|(byte, &destination)| (byte, offset + destination))
                        .collect(),
                );
                for &destination in &state.epsilon_transitions {
                    target.add_epsilon_transition(target_state, offset + destination);
                }
                target.overwrite_state_metadata(
                    target_state,
                    state.finalizers.clone(),
                    source
                        .possible_future_group_ids(state_index as u32)
                        .clone(),
                );
            }
        };
        copy_source(&mut dfa, &left.dfa, left_offset);
        copy_source(&mut dfa, &right.dfa, right_offset);
        dfa.add_epsilon_transition(0, left_offset + left.start_state());
        dfa.add_epsilon_transition(0, right_offset + right.start_state());

        let mut root_futures = BitSet::new(num_groups);
        for terminal in left
            .possible_future_terminals_iter(left.start_state())
            .chain(right.possible_future_terminals_iter(right.start_state()))
        {
            root_futures.set(terminal as usize);
        }
        dfa.overwrite_state_metadata(0, BitSet::new(num_groups), root_futures);

        TokenizerAnalysisUnion {
            tokenizer: Tokenizer::from_parts(dfa, left.num_terminals, None),
            left_offset,
            right_offset,
        }
    }

    fn start_state(&self) -> u32 {
        0
    }

    fn num_terminals(&self) -> u32 {
        self.num_terminals
    }

    pub fn has_epsilon_transitions(&self) -> bool {
        self.packed_runtime_metadata
            .as_deref()
            .is_some_and(PackedTokenizerMetadata::has_epsilon_transitions)
            || self
                .packed_runtime_metadata_segments
                .iter()
                .any(|segment| segment.metadata.has_epsilon_transitions())
            || self.dfa.has_epsilon_transitions()
    }

    #[inline]
    #[doc(hidden)]
    pub fn contains_runtime_state(&self, state: u32) -> bool {
        state < self.num_states()
            || self.virtual_residual_runtime_for_state(state).is_some()
            || self.virtual_repeat_runtime_for_state(state).is_some()
            || self
                .virtual_unit_repeat
                .as_deref()
                .is_some_and(|runtime| runtime.handles_state(state))
    }

    #[inline]
    #[doc(hidden)]
    pub fn state_is_virtual_runtime(&self, state: u32) -> bool {
        self.virtual_residual_runtime_for_state(state).is_some()
            || self.virtual_repeat_runtime_for_state(state).is_some()
            || self
                .virtual_unit_repeat
                .as_deref()
                .is_some_and(|runtime| runtime.is_virtual_state(state))
    }

    #[inline]
    pub fn state_has_epsilon_transitions(&self, state: u32) -> bool {
        if self.virtual_residual_runtime_for_state(state).is_some() {
            return false;
        }
        if self.virtual_repeat_runtime_for_state(state).is_some() {
            return false;
        }
        if self
            .virtual_unit_repeat
            .as_deref()
            .is_some_and(|runtime| runtime.is_virtual_state(state))
        {
            return false;
        }
        if let Some(metadata) = self
            .packed_runtime_metadata
            .as_deref()
            .filter(|metadata| state < metadata.state_count)
        {
            return !metadata.epsilon_targets(state).is_empty();
        }
        if let Some(segment) = self.packed_runtime_metadata_segment_for_state(state) {
            return !segment
                .metadata
                .epsilon_targets(segment.local_state(state))
                .is_empty();
        }
        self.dfa
            .states()
            .get(state as usize)
            .is_some_and(|state| !state.epsilon_transitions.is_empty())
    }

    /// Preserve an epsilon-closed induced subset with exact matched/future
    /// observations. The caller must prove that its queried byte paths stay
    /// inside this subset. Full original futures are intentional: a token may
    /// end at a live residual whose later bytes lie outside the current query.
    /// Every direct epsilon edge is copied once; a transitive closure must not
    /// be substituted as direct topology. Packed and ordinary inputs share the
    /// same authoritative accessors. No wire-layout construction is involved.
    pub fn induced_observation_view(&self, retained: &[bool]) -> Option<TokenizerObservationView> {
        let n=self.num_states() as usize;
        if n==0 || n>200_000 || retained.len()!=n || self.has_virtual_residual_runtime(){return None;}
        let closures=self.all_singleton_epsilon_closures();
        let mut keep=retained.to_vec();keep[self.initial_state_id() as usize]=true;
        let mut closure_work=0usize;
        for q in 0..n {if keep[q]{
            closure_work=closure_work.checked_add(closures[q].len())?;
            if closure_work>8_000_000{return None;}
            for &r in &closures[q]{if r as usize>=n{return None;}keep[r as usize]=true;}
        }}
        let start=self.initial_state_id();
        let mut inverse=vec![start];let mut mapping=vec![u32::MAX;n];mapping[start as usize]=0;
        for (q,&yes) in keep.iter().enumerate(){if yes && q!=start as usize{
            mapping[q]=inverse.len() as u32;inverse.push(q as u32);
        }}
        let mut dfa=DFA::new(inverse.len());dfa.ensure_group_capacity(self.num_terminals as usize);
        for terminal in 0..self.num_terminals{dfa.set_group_u8set(terminal,*self.dfa.group_id_to_u8set(terminal));}
        let mut edge_work=0usize;
        for (view,&raw) in inverse.iter().enumerate(){
            if self.state_is_virtual_runtime(raw){return None;}
            let mut edges=Vec::new();
            for (byte,target) in self.transitions_from(raw){
                if target as usize>=n{return None;}
                let target=mapping[target as usize];
                if target!=u32::MAX{edges.push((byte,target));}
            }
            edges.sort_unstable();
            edge_work=edge_work.checked_add(edges.len())?;
            if edge_work>8_000_000{return None;}
            dfa.set_transitions_from_sorted_entries(view as u32,edges);
            dfa.overwrite_state_metadata(view as u32,self.state_finalizers(raw).clone(),self.state_futures(raw).clone());
            let mut valid=true;
            self.for_each_epsilon_target(raw,|target|{
                let mapped=mapping.get(target as usize).copied().unwrap_or(u32::MAX);
                if mapped==u32::MAX{valid=false;}else{dfa.add_epsilon_transition(view as u32,mapped);edge_work+=1;}
            });
            if !valid || edge_work>8_000_000{return None;}
        }
        Some(TokenizerObservationView{tokenizer:Tokenizer::from_parts(dfa,self.num_terminals,self.exprs.clone()),
            original_to_view:mapping,view_to_original:inverse})
    }

    fn for_each_epsilon_target(&self, state: u32, mut visit: impl FnMut(u32)) {
        if let Some(metadata) = self
            .packed_runtime_metadata
            .as_deref()
            .filter(|metadata| state < metadata.state_count)
        {
            for &target in metadata.epsilon_targets(state) {
                visit(target);
            }
            return;
        }
        if let Some(segment) = self.packed_runtime_metadata_segment_for_state(state) {
            for &target in segment.metadata.epsilon_targets(segment.local_state(state)) {
                visit(segment.state_offset + target);
            }
            return;
        }
        if let Some(dfa_state) = self.dfa.states().get(state as usize) {
            for &target in &dfa_state.epsilon_transitions {
                visit(target);
            }
        }
    }

    fn epsilon_graph_is_acyclic(&self) -> bool {
        let state_count = self.num_states() as usize;
        let mut indegree = vec![0u32; state_count];
        for state in 0..self.num_states() {
            self.for_each_epsilon_target(state, |target| {
                indegree[target as usize] += 1;
            });
        }
        let mut queue = VecDeque::<u32>::new();
        for (state, &degree) in indegree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(state as u32);
            }
        }
        let mut visited = 0usize;
        while let Some(state) = queue.pop_front() {
            visited += 1;
            self.for_each_epsilon_target(state, |target| {
                let degree = &mut indegree[target as usize];
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(target);
                }
            });
        }
        visited == state_count
    }

    #[inline]
    fn state_finalizers(&self, state: u32) -> &BitSet {
        if let Some(finalizers) = self
            .virtual_residual_runtime_for_state(state)
            .and_then(|runtime| runtime.finalizers(state))
        {
            return finalizers;
        }
        if let Some(finalizers) = self
            .virtual_repeat_runtime_for_state(state)
            .and_then(|runtime| runtime.finalizers(state))
        {
            return finalizers;
        }
        if let Some(finalizers) = self
            .virtual_unit_repeat
            .as_deref()
            .and_then(|runtime| runtime.finalizers(state))
        {
            return finalizers;
        }
        if let Some(metadata) = self
            .packed_runtime_metadata
            .as_deref()
            .filter(|metadata| state < metadata.state_count)
        {
            return metadata
                .finalizers(state)
                .expect("packed tokenizer finalizer row must cover every state");
        }
        if let Some(segment) = self.packed_runtime_metadata_segment_for_state(state) {
            return segment
                .metadata
                .finalizers(segment.local_state(state))
                .expect("packed tokenizer metadata segment must cover every state");
        }
        self.dfa.finalizers(state)
    }

    #[inline]
    fn state_futures(&self, state: u32) -> &BitSet {
        if let Some(futures) = self
            .virtual_residual_runtime_for_state(state)
            .and_then(|runtime| runtime.futures(state))
        {
            return futures;
        }
        if let Some(futures) = self
            .virtual_repeat_runtime_for_state(state)
            .and_then(|runtime| runtime.futures(state))
        {
            return futures;
        }
        if let Some(futures) = self
            .virtual_unit_repeat
            .as_deref()
            .and_then(|runtime| runtime.futures(state))
        {
            return futures;
        }
        if let Some(metadata) = self
            .packed_runtime_metadata
            .as_deref()
            .filter(|metadata| state < metadata.state_count)
        {
            return metadata
                .futures(state)
                .expect("packed tokenizer future row must cover every state");
        }
        if let Some(segment) = self.packed_runtime_metadata_segment_for_state(state) {
            return segment
                .metadata
                .futures(segment.local_state(state))
                .expect("packed tokenizer metadata segment must cover every state");
        }
        self.dfa.possible_future_group_ids(state)
    }

    fn epsilon_closure_states(&self, roots: &[u32]) -> SmallVec<[u32; 1]> {
        if self.packed_runtime_metadata.is_none()
            && self.packed_runtime_metadata_segments.is_empty()
            && roots.iter().all(|&state| state < self.num_states())
        {
            return self.dfa.epsilon_closure(roots);
        }
        if roots.iter().all(|&state| !self.state_has_epsilon_transitions(state)) {
            let mut result = SmallVec::from_slice(roots);
            result.sort_unstable();
            result.dedup();
            return result;
        }
        let mut result = SmallVec::<[u32; 1]>::new();
        let mut stack = SmallVec::<[u32; 8]>::from_slice(roots);
        while let Some(state) = stack.pop() {
            if result.contains(&state) {
                continue;
            }
            result.push(state);
            if let Some(metadata) = self
                .packed_runtime_metadata
                .as_deref()
                .filter(|metadata| state < metadata.state_count)
            {
                stack.extend_from_slice(metadata.epsilon_targets(state));
            } else if let Some(segment) = self.packed_runtime_metadata_segment_for_state(state) {
                stack.extend(
                    segment
                        .metadata
                        .epsilon_targets(segment.local_state(state))
                        .iter()
                        .map(|target| segment.state_offset + *target),
                );
            } else if let Some(dfa_state) = self.dfa.states().get(state as usize) {
                stack.extend_from_slice(&dfa_state.epsilon_transitions);
            }
        }
        result.sort_unstable();
        result
    }

    pub fn terminal_expr(&self, terminal: TerminalID) -> Option<&Expr> {
        self.exprs.as_deref()?.get(terminal as usize)
    }

    /// Compile-time terminal expressions, when retained by the artifact.
    pub fn terminal_exprs(&self) -> Option<&[Expr]> {
        self.exprs.as_deref()
    }

    pub fn terminal_residual_coordinates(&self) -> Option<&TerminalResidualCoordinates> {
        self.terminal_residual_coordinates.as_deref()
    }

    /// Exact retained standalone-terminal residual for one physical combined
    /// tokenizer state. Returns `None` when the compiler did not retain an
    /// unambiguous coordinate or when the terminal is dead at this state.
    #[doc(hidden)]
    pub fn terminal_residual_direct_coordinate(
        &self,
        state: u32,
        terminal: TerminalID,
    ) -> Option<TerminalResidualDirectCoordinate> {
        if state >= self.num_states() || self.state_has_epsilon_transitions(state) {
            return None;
        }
        let coordinates = self.terminal_residual_coordinates.as_deref()?;
        let row = coordinates.row(state)?;
        let index = row
            .binary_search_by_key(&terminal, |&(candidate, _)| candidate)
            .ok()?;
        let residual_state = row[index].1;
        let (dfa, group) = coordinates.terminal_dfa_and_group(terminal)?;
        let live = dfa.finalizers(residual_state).contains(group as usize)
            || dfa
                .possible_future_group_ids(residual_state)
                .contains(group as usize);
        live.then_some(TerminalResidualDirectCoordinate {
            raw_state: state,
            terminal,
            residual_state,
        })
    }

    /// Advance a retained physical terminal-residual coordinate by one byte.
    /// The direct path is accepted only when the raw combined tokenizer target
    /// and the compile-time residual sidecar continue to agree exactly.
    #[doc(hidden)]
    pub fn terminal_residual_direct_coordinate_step(
        &self,
        coordinate: TerminalResidualDirectCoordinate,
        byte: u8,
    ) -> Option<TerminalResidualDirectCoordinate> {
        let coordinates = self.terminal_residual_coordinates.as_deref()?;
        let (dfa, group) = coordinates.terminal_dfa_and_group(coordinate.terminal)?;
        let residual_target = dfa.step(coordinate.residual_state, byte)?;
        let live = dfa.finalizers(residual_target).contains(group as usize)
            || dfa
                .possible_future_group_ids(residual_target)
                .contains(group as usize);
        if !live {
            return None;
        }
        let raw_target = self.step(coordinate.raw_state, byte)?;
        if self.state_has_epsilon_transitions(raw_target) {
            return None;
        }
        let row = coordinates.row(raw_target)?;
        let index = row
            .binary_search_by_key(&coordinate.terminal, |&(candidate, _)| candidate)
            .ok()?;
        if row[index].1 != residual_target {
            return None;
        }
        Some(TerminalResidualDirectCoordinate {
            raw_state: raw_target,
            terminal: coordinate.terminal,
            residual_state: residual_target,
        })
    }

    #[doc(hidden)]
    pub fn terminal_residual_direct_coordinate_accepting(
        &self,
        coordinate: TerminalResidualDirectCoordinate,
    ) -> Option<bool> {
        let coordinates = self.terminal_residual_coordinates.as_deref()?;
        let (dfa, group) = coordinates.terminal_dfa_and_group(coordinate.terminal)?;
        Some(dfa.finalizers(coordinate.residual_state).contains(group as usize))
    }

    #[doc(hidden)]
    pub fn terminal_residual_direct_coordinate_has_future(
        &self,
        coordinate: TerminalResidualDirectCoordinate,
    ) -> Option<bool> {
        let coordinates = self.terminal_residual_coordinates.as_deref()?;
        let (dfa, group) = coordinates.terminal_dfa_and_group(coordinate.terminal)?;
        Some(
            dfa.possible_future_group_ids(coordinate.residual_state)
                .contains(group as usize),
        )
    }

    #[doc(hidden)]
    #[inline]
    pub fn terminal_residual_direct_coordinate_raw_state(
        &self,
        coordinate: TerminalResidualDirectCoordinate,
    ) -> u32 {
        coordinate.raw_state
    }

    /// Exact compile-time exclusion certificate for a raw tokenizer state and
    /// terminal. `Some` is returned only when the partitioned compiler retained
    /// an unambiguous top-level `Exclude(left, right)` product coordinate and
    /// proved the right language finite.
    #[doc(hidden)]
    pub fn terminal_exclusion_continuation(
        &self,
        state: u32,
        terminal: TerminalID,
    ) -> Option<TerminalExclusionContinuation> {
        let coordinates = self.terminal_residual_coordinates.as_deref()?;
        let row = coordinates.row(state)?;
        let index = row
            .binary_search_by_key(&terminal, |&(candidate, _)| candidate)
            .ok()?;
        let terminal_residual_state = row[index].1;
        let certificate = coordinates.terminal_exclusion_certificate(terminal)?;
        let residual = *certificate.states.get(terminal_residual_state as usize)?;
        let right_live = residual.right_state != u32::MAX;
        Some(TerminalExclusionContinuation {
            terminal_residual_state,
            left_state: residual.left_state,
            right_state: right_live.then_some(residual.right_state),
            right_max_remaining: right_live.then_some(residual.right_max_remaining),
        })
    }

    /// Left operand DFA for a retained top-level exclusion certificate. State
    /// numbers match `TerminalExclusionContinuation::left_state`.
    #[doc(hidden)]
    pub fn terminal_exclusion_left_dfa(&self, terminal: TerminalID) -> Option<&DFA> {
        self.terminal_residual_coordinates
            .as_deref()?
            .terminal_exclusion_certificate(terminal)
            .map(|certificate| certificate.left_dfa.as_ref())
    }

    /// Sound, potentially non-maximal observation partition for one terminal
    /// derived directly from retained standalone-terminal residual coordinates.
    /// Equal nonzero IDs imply equal exact terminal residuals. Raw states that
    /// are terminal-live but lack a retained coordinate remain distinct rather
    /// than being confused with the class-zero dead residual.
    pub fn terminal_residual_coordinate_observation_partition(
        &self,
        terminal: TerminalID,
    ) -> Option<(Box<[u32]>, usize, bool)> {
        let coordinates = self.terminal_residual_coordinates.as_deref()?;
        let (terminal_dfa, group) = coordinates.terminal_dfa_and_group(terminal)?;
        let state_count = self.num_states() as usize;
        let mut classes = vec![0u32; state_count];
        let mut seen_residuals = vec![false; terminal_dfa.num_states()];
        let mut distinct = 0usize;
        let mut useful_alias = false;
        let physical_rows = coordinates.len().min(state_count);

        for raw in 0..physical_rows {
            let row = coordinates.row(raw as u32)?;
            let Ok(index) = row.binary_search_by_key(&terminal, |&(candidate, _)| candidate)
            else {
                continue;
            };
            let residual = row[index].1;
            if residual >= terminal_dfa.num_states() as u32 {
                return None;
            }
            let live = terminal_dfa.finalizers(residual).contains(group as usize)
                || terminal_dfa
                    .possible_future_group_ids(residual)
                    .contains(group as usize);
            if live {
                classes[raw] = residual.checked_add(1)?;
                let seen = seen_residuals.get_mut(residual as usize)?;
                if *seen {
                    useful_alias = true;
                } else {
                    *seen = true;
                    distinct += 1;
                }
            }
        }

        let mut next_uncovered_class = u32::try_from(terminal_dfa.num_states())
            .ok()?
            .checked_add(1)?;
        for raw in 0..self.num_states() {
            if classes[raw as usize] == 0 && self.state_live_for_terminal(raw, terminal) {
                classes[raw as usize] = next_uncovered_class;
                next_uncovered_class = next_uncovered_class.checked_add(1)?;
                distinct += 1;
            }
        }
        Some((classes.into_boxed_slice(), distinct, useful_alias))
    }

    pub fn set_terminal_residual_coordinates(
        &mut self,
        coordinates: TerminalResidualCoordinates,
    ) {
        debug_assert_eq!(coordinates.len(), self.num_states() as usize);
        self.terminal_residual_coordinates = Some(Arc::new(coordinates));
    }

    #[doc(hidden)]
    pub fn virtual_runtime_metadata(&self) -> Vec<VirtualTokenizerRuntimeMetadata> {
        let mut metadata = Vec::with_capacity(
            self.virtual_repeat_intersections.len()
                + self.virtual_residuals.len()
                + usize::from(self.virtual_unit_repeat.is_some()),
        );
        if let Some(runtime) = self.virtual_unit_repeat.as_deref() {
            metadata.push(VirtualTokenizerRuntimeMetadata {
                kind: VirtualTokenizerRuntimeKind::UnitRepeat,
                terminal: runtime.terminal(),
                root_state: runtime.root_state(),
            });
        }
        metadata.extend(self.virtual_repeat_intersections.iter().map(|runtime| {
            VirtualTokenizerRuntimeMetadata {
                kind: VirtualTokenizerRuntimeKind::RepeatProduct,
                terminal: runtime.terminal(),
                root_state: runtime.root_state(),
            }
        }));
        metadata.extend(self.virtual_residuals.iter().map(|runtime| {
            VirtualTokenizerRuntimeMetadata {
                kind: VirtualTokenizerRuntimeKind::ResidualExpr,
                terminal: runtime.terminal(),
                root_state: runtime.root_state(),
            }
        }));
        metadata.sort_unstable_by_key(|entry| (entry.terminal, entry.root_state));
        metadata
    }

    #[doc(hidden)]
    pub fn virtual_residual_runtime_oracles(&self) -> Vec<(TerminalID, Vec<u8>)> {
        self.virtual_residuals
            .iter()
            .filter(|runtime| runtime.has_bounded_code_liveness_oracle())
            .map(|runtime| (runtime.terminal(), runtime.serialized_bounded_code_oracle()))
            .collect()
    }

    #[doc(hidden)]
    pub fn virtual_residual_master_slice_artifacts(
        &self,
    ) -> Vec<VirtualResidualMasterSliceArtifact> {
        self.virtual_residuals
            .iter()
            .filter_map(|runtime| runtime.master_slice_artifact())
            .collect()
    }

    #[doc(hidden)]
    pub fn restore_virtual_residual_master_slice_artifacts(
        &self,
        artifacts: Vec<VirtualResidualMasterSliceArtifact>,
    ) -> Result<(), String> {
        for artifact in artifacts {
            let runtime = self
                .virtual_residuals
                .iter()
                .find(|runtime| {
                    runtime.terminal() == artifact.terminal
                        && runtime.root_state() == artifact.root_state
                })
                .ok_or_else(|| {
                    "virtual residual master-slice artifact references unknown runtime".to_owned()
                })?;
            runtime.restore_master_slice_artifact(artifact)?;
        }
        Ok(())
    }

    fn restore_terminal_exprs_arc_only(
        &mut self,
        exprs: Option<Arc<[Expr]>>,
    ) -> Result<(), String> {
        let Some(exprs) = exprs else {
            self.exprs = None;
            return Ok(());
        };
        if exprs.len() != self.num_terminals as usize {
            return Err(format!(
                "serialized tokenizer has {} terminal expressions for {} terminals",
                exprs.len(), self.num_terminals,
            ));
        }
        self.exprs = Some(exprs);
        Ok(())
    }

    fn restore_terminal_exprs_only(&mut self, exprs: Option<Vec<Expr>>) -> Result<(), String> {
        self.restore_terminal_exprs_arc_only(
            exprs.map(|exprs| Arc::from(exprs.into_boxed_slice())),
        )
    }

    /// Restore terminal expressions carried by a versioned outer artifact.
    /// Invalid-length metadata is rejected rather than silently associating
    /// expressions with the wrong terminal IDs.
    pub fn restore_terminal_exprs(&mut self, exprs: Option<Vec<Expr>>) -> Result<(), String> {
        self.restore_terminal_exprs_only(exprs)?;
        if self.exprs.is_none() {
            return Ok(());
        }
        self.restore_virtual_unit_repeat_runtime()?;
        self.restore_virtual_repeat_intersection_runtime()?;
        Ok(())
    }

    /// Attach compile-time terminal expressions without invoking legacy
    /// structural virtual-runtime reconstruction. Fresh dynamic compilation
    /// uses this before explicitly installing the selected virtual runtime
    /// family, so ownership is deterministic rather than inferred from proxy
    /// shape.
    #[doc(hidden)]
    pub fn restore_terminal_exprs_without_virtual_runtime(
        &mut self,
        exprs: Option<Vec<Expr>>,
    ) -> Result<(), String> {
        self.restore_terminal_exprs_only(exprs)
    }

    #[doc(hidden)]
    pub fn restore_terminal_exprs_arc_without_virtual_runtime(
        &mut self,
        exprs: Option<Arc<[Expr]>>,
    ) -> Result<(), String> {
        self.restore_terminal_exprs_arc_only(exprs)
    }

    /// Restore virtual sidecars from explicit outer-artifact metadata. Unlike
    /// the legacy structural heuristic, every virtualizable giant terminal is
    /// a mandatory owner. A below-threshold residual owner is additionally
    /// permitted only when the exact bounded-code liveness oracle certifies its
    /// terminal expression; such owners are representation choices, not
    /// mandatory semantics, so older materialized artifacts may omit them.
    #[doc(hidden)]
    pub fn restore_terminal_exprs_with_virtual_runtime_metadata(
        &mut self,
        exprs: Option<Vec<Expr>>,
        metadata: &[VirtualTokenizerRuntimeMetadata],
        allow_legacy_exact_dead_residual_roots: bool,
    ) -> Result<(), String> {
        self.restore_terminal_exprs_with_virtual_runtime_metadata_impl(
            exprs,
            metadata,
            allow_legacy_exact_dead_residual_roots,
            false,
            None,
            None,
        )
    }

    #[doc(hidden)]
    pub fn restore_terminal_exprs_with_virtual_runtime_metadata_and_oracles(
        &mut self,
        exprs: Option<Vec<Expr>>,
        metadata: &[VirtualTokenizerRuntimeMetadata],
        residual_oracles: &[(TerminalID, Vec<u8>)],
        allow_legacy_exact_dead_residual_roots: bool,
    ) -> Result<(), String> {
        self.restore_terminal_exprs_with_virtual_runtime_metadata_and_oracles_preserving_coordinates(
            exprs,
            metadata,
            residual_oracles,
            allow_legacy_exact_dead_residual_roots,
            false,
        )
    }

    #[doc(hidden)]
    pub fn restore_terminal_exprs_with_virtual_runtime_metadata_and_oracles_preserving_coordinates(
        &mut self,
        exprs: Option<Vec<Expr>>,
        metadata: &[VirtualTokenizerRuntimeMetadata],
        residual_oracles: &[(TerminalID, Vec<u8>)],
        allow_legacy_exact_dead_residual_roots: bool,
        preserve_oracle_coordinate: bool,
    ) -> Result<(), String> {
        self.restore_terminal_exprs_with_virtual_runtime_metadata_impl(
            exprs,
            metadata,
            allow_legacy_exact_dead_residual_roots,
            preserve_oracle_coordinate,
            None,
            Some(residual_oracles),
        )
    }

    /// Static compiled constraints use the same exact residual language as the
    /// dynamic runtime, but retain the bounded-code oracle coordinate in the
    /// lazy state identity so every runtime state has a deterministic finite
    /// Static TSID projection. This is a representation policy, not additional
    /// serialized grammar semantics, so it is selected by the Static loader.
    #[doc(hidden)]
    pub fn restore_terminal_exprs_with_virtual_runtime_metadata_preserving_residual_coordinates(
        &mut self,
        exprs: Option<Vec<Expr>>,
        metadata: &[VirtualTokenizerRuntimeMetadata],
        allow_legacy_exact_dead_residual_roots: bool,
    ) -> Result<(), String> {
        self.restore_terminal_exprs_with_virtual_runtime_metadata_impl(
            exprs,
            metadata,
            allow_legacy_exact_dead_residual_roots,
            true,
            None,
            None,
        )
    }

    /// Restore a current Static residual runtime directly from its compiled
    /// artifact program without materializing the constraint's complete list
    /// of terminal source expressions. The full expression blob remains
    /// deferred for later composition; only each residual owner's tiny source
    /// expression is decoded to rebuild the derivative arena.
    #[doc(hidden)]
    pub fn restore_compiled_static_residual_runtimes(
        &mut self,
        metadata: &[VirtualTokenizerRuntimeMetadata],
        projections: &[VirtualResidualMaskProjectionArtifact],
    ) -> Result<(), String> {
        if metadata.is_empty() || metadata.len() != projections.len() {
            return Err("compiled static residual runtime/projection count mismatch".to_owned());
        }
        if metadata.iter().any(|entry| entry.kind != VirtualTokenizerRuntimeKind::ResidualExpr) {
            return Err("compiled static residual fast path requires residual-only virtual runtimes".to_owned());
        }
        self.virtual_unit_repeat = None;
        self.virtual_repeat_intersections.clear();
        self.virtual_residuals.clear();
        self.exprs = None;

        let physical_state_count = self.num_states();
        let start_state = self.start_state();
        let reset_closure = self.epsilon_closure_states(&[start_state]);
        let mut seen_terminals = BTreeSet::new();
        let mut seen_roots = BTreeSet::new();
        for entry in metadata {
            if entry.terminal >= self.num_terminals
                || !seen_terminals.insert(entry.terminal)
                || !seen_roots.insert(entry.root_state)
                || entry.root_state == start_state
                || entry.root_state >= physical_state_count
                || !reset_closure.contains(&entry.root_state)
                || self.state_has_epsilon_transitions(entry.root_state)
                || self.transitions_from(entry.root_state).next().is_some()
                || !self.state_finalizers(entry.root_state).is_empty()
            {
                return Err("compiled static residual runtime has invalid terminal/root ownership".to_owned());
            }
        }
        let projection_terminals = projections
            .iter()
            .map(VirtualResidualMaskProjectionArtifact::terminal)
            .collect::<BTreeSet<_>>();
        if projection_terminals != seen_terminals || projection_terminals.len() != projections.len() {
            return Err("compiled static residual projection terminal ownership mismatch".to_owned());
        }

        let allocator = Arc::new(
            VirtualStateAllocator::new(physical_state_count)
                .ok_or_else(|| "compiled static residual runtime has no virtual state namespace".to_owned())?,
        );
        let roots = metadata.iter().map(|entry| entry.root_state).collect::<Vec<_>>();
        let owners = Arc::new(
            VirtualRuntimeStateOwners::new(physical_state_count, &roots)
                .ok_or_else(|| "compiled static residual runtime has invalid state ownership".to_owned())?,
        );
        let mut runtimes = Vec::with_capacity(metadata.len());
        for (runtime_index, entry) in metadata.iter().enumerate() {
            let projection = projections
                .iter()
                .find(|projection| projection.terminal() == entry.terminal)
                .ok_or_else(|| format!("missing compiled residual program for terminal {}", entry.terminal))?;
            if projection.runtime_expr_bytes().is_empty() {
                return Err(format!("compiled residual program for terminal {} has no expression", entry.terminal));
            }
            let expression: Expr = bincode::deserialize(projection.runtime_expr_bytes())
                .map_err(|err| format!("invalid compiled residual expression: {err}"))?;
            if self.terminal_byte_support(entry.terminal) != Some(super::compile::expr_u8set(&expression)) {
                return Err(format!("compiled residual terminal {} has inconsistent byte support", entry.terminal));
            }
            let runtime_index = u32::try_from(runtime_index)
                .map_err(|_| "compiled residual runtime count exceeds u32".to_owned())?;
            let runtime = Arc::new(
                VirtualResidualRuntime::new_preserving_oracle_coordinate_from_oracle_bytes(
                    &expression, projection.oracle_bytes(), runtime_index, entry.terminal,
                    self.num_terminals, physical_state_count, entry.root_state,
                    Arc::clone(&allocator), Arc::clone(&owners),
                )
                .ok_or_else(|| "compiled static residual runtime program is invalid".to_owned())?,
            );
            let mut expected_future = BitSet::new(self.num_terminals as usize);
            if runtime.root_has_future() { expected_future.set(entry.terminal as usize); }
            if self.state_futures(entry.root_state) != &expected_future {
                return Err(format!("compiled residual root {} has inconsistent future metadata", entry.root_state));
            }
            if runtime.root_has_future() && !self.state_futures(start_state).contains(entry.terminal as usize) {
                return Err(format!("compiled residual terminal {} is missing from reset-state futures", entry.terminal));
            }
            runtimes.push(runtime);
        }
        self.virtual_residuals = runtimes;
        self.invalidate_derived_caches();
        Ok(())
    }

    /// Restore a current Dynamic residual runtime from the already-certified
    /// runtime-owner list and serialized bounded-code oracles. Unlike the
    /// generic compatibility loader, this path must not rescan every terminal
    /// expression to rediscover which terminals require a virtual runtime;
    /// that classification is compile-time work and is already represented by
    /// `metadata` plus the one oracle carried for each residual owner.
    #[doc(hidden)]
    pub fn restore_compiled_dynamic_residual_runtimes(
        &mut self,
        expressions: &[Expr],
        metadata: &[VirtualTokenizerRuntimeMetadata],
        residual_oracles: &[(TerminalID, Vec<u8>)],
        master_slice_artifacts: &[VirtualResidualMasterSliceArtifact],
    ) -> Result<(), String> {
        let profile_load = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_LOAD").is_some();
        let total_started = profile_load.then(std::time::Instant::now);
        if metadata.is_empty() {
            return Err("compiled dynamic residual runtime list is empty".to_owned());
        }
        if metadata
            .iter()
            .any(|entry| entry.kind != VirtualTokenizerRuntimeKind::ResidualExpr)
        {
            return Err(
                "compiled dynamic residual fast path requires residual-only virtual runtimes"
                    .to_owned(),
            );
        }
        self.virtual_unit_repeat = None;
        self.virtual_repeat_intersections.clear();
        self.virtual_residuals.clear();
        self.exprs = None;

        let physical_state_count = self.num_states();
        let start_state = self.start_state();
        let validate_started = profile_load.then(std::time::Instant::now);
        let reset_closure = self.epsilon_closure_states(&[start_state]);
        let mut seen_terminals = BTreeSet::new();
        let mut seen_roots = BTreeSet::new();
        for entry in metadata {
            if entry.terminal >= self.num_terminals
                || entry.terminal as usize >= expressions.len()
                || !seen_terminals.insert(entry.terminal)
                || !seen_roots.insert(entry.root_state)
                || entry.root_state == start_state
                || entry.root_state >= physical_state_count
                || !reset_closure.contains(&entry.root_state)
                || self.state_has_epsilon_transitions(entry.root_state)
                || self.transitions_from(entry.root_state).next().is_some()
                || !self.state_finalizers(entry.root_state).is_empty()
            {
                return Err(
                    "compiled dynamic residual runtime has invalid terminal/root ownership"
                        .to_owned(),
                );
            }
        }
        if !residual_oracles.is_empty() {
            let oracle_terminals = residual_oracles
                .iter()
                .map(|(terminal, _)| *terminal)
                .collect::<BTreeSet<_>>();
            if oracle_terminals != seen_terminals
                || oracle_terminals.len() != residual_oracles.len()
            {
                return Err(
                    "compiled dynamic residual oracle terminal ownership mismatch".to_owned(),
                );
            }
        }
        let validate_ms = validate_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let owners_started = profile_load.then(std::time::Instant::now);
        let allocator = Arc::new(
            VirtualStateAllocator::new(physical_state_count).ok_or_else(|| {
                "compiled dynamic residual runtime has no virtual state namespace".to_owned()
            })?,
        );
        let roots = metadata
            .iter()
            .map(|entry| entry.root_state)
            .collect::<Vec<_>>();
        let owners = Arc::new(
            VirtualRuntimeStateOwners::new(physical_state_count, &roots).ok_or_else(|| {
                "compiled dynamic residual runtime has invalid state ownership".to_owned()
            })?,
        );
        let owners_ms = owners_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let mut runtimes = Vec::with_capacity(metadata.len());
        for (runtime_index, entry) in metadata.iter().enumerate() {
            let runtime_started = profile_load.then(std::time::Instant::now);
            let expression = &expressions[entry.terminal as usize];
            let support_started = profile_load.then(std::time::Instant::now);
            if self.terminal_byte_support(entry.terminal)
                != Some(super::compile::expr_u8set(expression))
            {
                return Err(format!(
                    "compiled dynamic residual terminal {} has inconsistent byte support",
                    entry.terminal,
                ));
            }
            let support_ms = support_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            let oracle_bytes = residual_oracles
                .iter()
                .find(|(terminal, _)| *terminal == entry.terminal)
                .map(|(_, bytes)| bytes.as_slice());
            let runtime_index = u32::try_from(runtime_index)
                .map_err(|_| "compiled dynamic residual runtime count exceeds u32".to_owned())?;
            let master_slice = master_slice_artifacts.iter().find(|artifact| {
                artifact.terminal == entry.terminal && artifact.root_state == entry.root_state
            });
            let oracle_started = profile_load.then(std::time::Instant::now);
            let runtime = if let Some(master_slice) = master_slice {
                let oracle = VirtualResidualRuntime::compact_liveness_oracle_from_master_slice_artifact(
                    master_slice,
                )
                .ok_or_else(|| {
                    format!(
                        "compiled dynamic residual master-slice program for terminal {} is invalid",
                        entry.terminal,
                    )
                })?;
                VirtualResidualRuntime::new_with_liveness_oracle(
                    expression,
                    oracle,
                    runtime_index,
                    entry.terminal,
                    self.num_terminals,
                    physical_state_count,
                    entry.root_state,
                    Arc::clone(&allocator),
                    Arc::clone(&owners),
                )
            } else if let Some(oracle_bytes) = oracle_bytes {
                VirtualResidualRuntime::new_preserving_oracle_coordinate_from_oracle_bytes(
                    expression,
                    oracle_bytes,
                    runtime_index,
                    entry.terminal,
                    self.num_terminals,
                    physical_state_count,
                    entry.root_state,
                    Arc::clone(&allocator),
                    Arc::clone(&owners),
                )
            } else {
                VirtualResidualRuntime::new_preserving_oracle_coordinate(
                    expression,
                    runtime_index,
                    entry.terminal,
                    self.num_terminals,
                    physical_state_count,
                    entry.root_state,
                    Arc::clone(&allocator),
                    Arc::clone(&owners),
                )
            };
            let oracle_and_runtime_ms = oracle_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            let runtime = Arc::new(
                runtime.ok_or_else(|| {
                    "compiled dynamic residual runtime program is invalid".to_owned()
                })?,
            );
            let restore_started = profile_load.then(std::time::Instant::now);
            if let Some(master_slice) = master_slice {
                runtime.restore_master_slice_artifact(master_slice.clone())?;
            }
            let restore_ms = restore_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            let mut expected_future = BitSet::new(self.num_terminals as usize);
            if runtime.root_has_future() {
                expected_future.set(entry.terminal as usize);
            }
            if self.state_futures(entry.root_state) != &expected_future {
                return Err(format!(
                    "compiled dynamic residual root {} has inconsistent future metadata",
                    entry.root_state,
                ));
            }
            if runtime.root_has_future()
                && !self.state_futures(start_state).contains(entry.terminal as usize)
            {
                return Err(format!(
                    "compiled dynamic residual terminal {} is missing from reset-state futures",
                    entry.terminal,
                ));
            }
            runtimes.push(runtime);
            if profile_load {
                eprintln!(
                    "[glrmask/profile][dynamic_residual_restore_runtime] terminal={} support_ms={:.3} oracle_runtime_ms={:.3} restore_rows_ms={:.3} total_ms={:.3}",
                    entry.terminal,
                    support_ms,
                    oracle_and_runtime_ms,
                    restore_ms,
                    runtime_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                );
            }
        }
        self.virtual_residuals = runtimes;
        let invalidate_started = profile_load.then(std::time::Instant::now);
        self.invalidate_derived_caches();
        let invalidate_ms = invalidate_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile_load {
            eprintln!(
                "[glrmask/profile][dynamic_residual_restore] runtimes={} validate_ms={:.3} owners_ms={:.3} invalidate_ms={:.3} total_ms={:.3}",
                metadata.len(),
                validate_ms,
                owners_ms,
                invalidate_ms,
                total_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        Ok(())
    }

    #[doc(hidden)]
    pub fn restore_terminal_exprs_with_precompiled_static_residual_oracles(
        &mut self,
        exprs: Option<Vec<Expr>>,
        metadata: &[VirtualTokenizerRuntimeMetadata],
        projections: &[VirtualResidualMaskProjectionArtifact],
        allow_legacy_exact_dead_residual_roots: bool,
    ) -> Result<(), String> {
        self.restore_terminal_exprs_with_virtual_runtime_metadata_impl(
            exprs,
            metadata,
            allow_legacy_exact_dead_residual_roots,
            true,
            Some(projections),
            None,
        )
    }

    fn restore_terminal_exprs_with_virtual_runtime_metadata_impl(
        &mut self,
        exprs: Option<Vec<Expr>>,
        metadata: &[VirtualTokenizerRuntimeMetadata],
        allow_legacy_exact_dead_residual_roots: bool,
        preserve_residual_oracle_coordinates: bool,
        precompiled_static_residual_projections: Option<&[VirtualResidualMaskProjectionArtifact]>,
        precompiled_dynamic_residual_oracles: Option<&[(TerminalID, Vec<u8>)]>,
    ) -> Result<(), String> {
        let load_profile = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_LOAD").is_some();
        let whole_started = load_profile.then(std::time::Instant::now);
        self.restore_terminal_exprs_only(exprs)?;
        self.virtual_unit_repeat = None;
        self.virtual_repeat_intersections.clear();
        self.virtual_residuals.clear();
        let Some(expressions) = self.exprs.as_deref() else {
            if metadata.is_empty() {
                return Ok(());
            }
            return Err("serialized virtual runtime metadata has no terminal expressions".to_owned());
        };

        let declared_terminals = metadata
            .iter()
            .map(|entry| entry.terminal)
            .collect::<BTreeSet<_>>();
        let physical_state_count = self.num_states();
        let mut required_virtual_terminals = BTreeSet::<TerminalID>::new();
        let mut certified_bounded_code_terminals = BTreeSet::<TerminalID>::new();
        // Dynamic transfer loading used to construct an exact bounded-code
        // oracle here solely to certify each below-threshold residual owner,
        // drop it, and then construct the identical oracle again while
        // attaching the runtime below. Retain the certification result and
        // move it into the runtime instead.
        let mut certified_dynamic_residual_oracles =
            BTreeMap::<TerminalID, super::runtime_residual::BoundedCodeIntersectionOracle>::new();
        let mut any_finalizer = BitSet::new(self.num_terminals as usize);
        let mut any_future = BitSet::new(self.num_terminals as usize);
        let state_scan_started = load_profile.then(std::time::Instant::now);
        if precompiled_static_residual_projections.is_some() {
            for state in 0..physical_state_count {
                any_finalizer.union_with(self.state_finalizers(state));
                any_future.union_with(self.state_futures(state));
            }
        }
        let state_scan_ms = state_scan_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let expr_scan_started = load_profile.then(std::time::Instant::now);
        for (terminal, expression) in expressions.iter().enumerate() {
            let terminal = terminal as TerminalID;
            if super::compile::expression_contains_large_bounded_repeat(expression) {
                required_virtual_terminals.insert(terminal);
                continue;
            }
            if let Some(projections) = precompiled_static_residual_projections {
                if projections.iter().any(|projection| projection.terminal() == terminal) {
                    certified_bounded_code_terminals.insert(terminal);
                    if !any_finalizer.contains(terminal as usize) && any_future.contains(terminal as usize) {
                        required_virtual_terminals.insert(terminal);
                    }
                }
                continue;
            }
            let Some(oracle) = build_bounded_code_liveness_oracle(expression) else {
                continue;
            };
            certified_bounded_code_terminals.insert(terminal);
            if declared_terminals.contains(&terminal) {
                certified_dynamic_residual_oracles.insert(terminal, oracle);
            }
            let mut has_physical_finalizer = false;
            let mut has_physical_future = false;
            for state in 0..physical_state_count {
                has_physical_finalizer |= self.state_finalizers(state).contains(terminal as usize);
                has_physical_future |= self.state_futures(state).contains(terminal as usize);
                if has_physical_finalizer && has_physical_future { break; }
            }
            if !has_physical_finalizer && has_physical_future {
                required_virtual_terminals.insert(terminal);
            }
        }
        let expr_scan_ms = expr_scan_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if metadata.len() != declared_terminals.len() {
            return Err("serialized virtual runtime metadata contains duplicate terminal owners".to_owned());
        }
        let missing_required = required_virtual_terminals
            .difference(&declared_terminals)
            .copied()
            .collect::<BTreeSet<_>>();
        if !missing_required.is_empty() {
            return Err(format!(
                "serialized virtual runtime terminal ownership mismatch: missing required {:?}, found {:?}",
                missing_required, declared_terminals,
            ));
        }
        for entry in metadata {
            if required_virtual_terminals.contains(&entry.terminal) {
                continue;
            }
            if entry.kind != VirtualTokenizerRuntimeKind::ResidualExpr
                || !certified_bounded_code_terminals.contains(&entry.terminal)
            {
                return Err(format!(
                    "serialized virtual runtime terminal ownership mismatch: terminal {} is not a required giant component or a certified bounded-code residual",
                    entry.terminal,
                ));
            }
        }
        if metadata.is_empty() {
            return Ok(());
        }

        let start_state = self.start_state();
        let root_validation_started = load_profile.then(std::time::Instant::now);
        let reset_closure = self.epsilon_closure_states(&[start_state]);
        let validate_root = |tokenizer: &Self,
                             entry: &VirtualTokenizerRuntimeMetadata|
         -> Result<(), String> {
            let standalone_unit_root = metadata.len() == 1
                && entry.kind == VirtualTokenizerRuntimeKind::UnitRepeat
                && physical_state_count == 1
                && self.num_terminals == 1
                && entry.terminal == 0
                && entry.root_state == start_state
                && start_state == 0;
            if (!standalone_unit_root && entry.root_state == start_state)
                || entry.root_state >= physical_state_count
                || !reset_closure.contains(&entry.root_state)
                || tokenizer.state_has_epsilon_transitions(entry.root_state)
                || tokenizer.transitions_from(entry.root_state).next().is_some()
                || !tokenizer.state_finalizers(entry.root_state).is_empty()
            {
                return Err(format!(
                    "serialized virtual runtime has invalid physical proxy root {}",
                    entry.root_state,
                ));
            }
            Ok(())
        };
        let mut seen_roots = BTreeSet::new();
        for entry in metadata {
            if entry.terminal >= self.num_terminals || !seen_roots.insert(entry.root_state) {
                return Err("serialized virtual runtime metadata has invalid terminal/root ownership".to_owned());
            }
            validate_root(self, entry)?;
        }
        let root_validation_ms = root_validation_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if load_profile {
            eprintln!(
                "[glrmask/profile][virtual_runtime_restore_preamble] state_scan_ms={:.3} expr_scan_ms={:.3} root_validation_ms={:.3} elapsed_ms={:.3}",
                state_scan_ms,
                expr_scan_ms,
                root_validation_ms,
                whole_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }

        if metadata.iter().any(|entry| entry.kind == VirtualTokenizerRuntimeKind::UnitRepeat) {
            let [entry] = metadata else {
                return Err("serialized arithmetic unit runtime cannot coexist with other virtual runtimes".to_owned());
            };
            if entry.kind != VirtualTokenizerRuntimeKind::UnitRepeat {
                unreachable!();
            }
            let expression = expressions
                .get(entry.terminal as usize)
                .ok_or_else(|| "serialized virtual unit terminal is out of range".to_owned())?;
            let (body, min, max) = super::compile::virtual_unit_repeat_descriptor(expression)
                .ok_or_else(|| "serialized virtual unit runtime descriptor no longer matches terminal expression".to_owned())?;
            if self.terminal_byte_support(entry.terminal) != Some(body) {
                return Err(
                    "serialized virtual unit runtime byte support does not match terminal metadata"
                        .to_owned(),
                );
            }
            if !super::compile::virtual_zero_min_unit_repeat_fits_state_ids(max, physical_state_count) {
                return Err("serialized virtual unit runtime exceeds tokenizer state-id space".to_owned());
            }
            let mut expected_future = BitSet::new(self.num_terminals as usize);
            expected_future.set(entry.terminal as usize);
            // Validate the serialized physical metadata before attaching the
            // reconstructed sidecar. Once installed, `state_futures()` for a
            // unit-runtime state intentionally dispatches through the sidecar,
            // which would make these checks self-validating instead of checking
            // the artifact that was actually loaded.
            let serialized_root_futures = self.state_futures(entry.root_state).clone();
            let serialized_start_futures = self.state_futures(start_state).clone();
            if serialized_root_futures != expected_future
                || !serialized_start_futures.contains(entry.terminal as usize)
            {
                return Err(
                    "serialized virtual unit runtime has inconsistent future metadata".to_owned(),
                );
            }
            let runtime = VirtualZeroMinUnitRepeatRuntime::new(
                body,
                min,
                max,
                entry.terminal,
                self.num_terminals,
                physical_state_count,
                entry.root_state,
            )
            .ok_or_else(|| "serialized virtual unit runtime metadata is invalid".to_owned())?;
            self.virtual_unit_repeat = Some(Arc::new(runtime));
            self.invalidate_derived_caches();
            return Ok(());
        }

        if metadata
            .iter()
            .any(|entry| entry.kind == VirtualTokenizerRuntimeKind::ResidualExpr)
        {
            if metadata
                .iter()
                .any(|entry| entry.kind != VirtualTokenizerRuntimeKind::ResidualExpr)
            {
                return Err(
                    "serialized residual runtime cannot coexist with another virtual runtime family"
                        .to_owned(),
                );
            }
            let allocator = Arc::new(
                VirtualStateAllocator::new(physical_state_count).ok_or_else(|| {
                    "serialized residual runtime has no virtual state-id namespace".to_owned()
                })?,
            );
            let roots = metadata.iter().map(|entry| entry.root_state).collect::<Vec<_>>();
            let owners = Arc::new(
                VirtualRuntimeStateOwners::new(physical_state_count, &roots).ok_or_else(|| {
                    "serialized residual runtime has invalid state ownership metadata".to_owned()
                })?,
            );
            let mut runtimes = Vec::with_capacity(metadata.len());
            let mut legacy_exact_dead_roots = Vec::<(u32, TerminalID, BitSet)>::new();
            for (runtime_index, entry) in metadata.into_iter().enumerate() {
                let runtime_profile = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_LOAD").is_some();
                let runtime_started = runtime_profile.then(std::time::Instant::now);
                let expression = expressions
                    .get(entry.terminal as usize)
                    .ok_or_else(|| "serialized residual terminal is out of range".to_owned())?;
                let support_started = runtime_profile.then(std::time::Instant::now);
                let expected_support = super::compile::expr_u8set(expression);
                let support_ms = support_started
                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                if self.terminal_byte_support(entry.terminal) != Some(expected_support) {
                    return Err(format!(
                        "serialized residual terminal {} has inconsistent byte support",
                        entry.terminal,
                    ));
                }
                let runtime_index = u32::try_from(runtime_index).map_err(|_| {
                    "serialized residual runtime count exceeds u32".to_owned()
                })?;
                let construct_started = runtime_profile.then(std::time::Instant::now);
                let runtime = if preserve_residual_oracle_coordinates {
                    if let Some(projections) = precompiled_static_residual_projections {
                        let projection = projections
                            .iter()
                            .find(|projection| projection.terminal() == entry.terminal)
                            .ok_or_else(|| format!("missing precompiled residual oracle for terminal {}", entry.terminal))?;
                        VirtualResidualRuntime::new_preserving_oracle_coordinate_from_oracle_bytes(
                            expression,
                            projection.oracle_bytes(),
                            runtime_index,
                            entry.terminal,
                            self.num_terminals,
                            physical_state_count,
                            entry.root_state,
                            Arc::clone(&allocator),
                            Arc::clone(&owners),
                        )
                    } else {
                        VirtualResidualRuntime::new_preserving_oracle_coordinate(
                            expression, runtime_index, entry.terminal, self.num_terminals,
                            physical_state_count, entry.root_state, Arc::clone(&allocator),
                            Arc::clone(&owners),
                        )
                    }
                } else if let Some(oracle_bytes) = precompiled_dynamic_residual_oracles
                    .and_then(|oracles| {
                        oracles
                            .iter()
                            .find(|(terminal, _)| *terminal == entry.terminal)
                            .map(|(_, bytes)| bytes.as_slice())
                    })
                {
                    VirtualResidualRuntime::new_from_oracle_bytes(
                        expression,
                        oracle_bytes,
                        runtime_index,
                        entry.terminal,
                        self.num_terminals,
                        physical_state_count,
                        entry.root_state,
                        Arc::clone(&allocator),
                        Arc::clone(&owners),
                    )
                } else if let Some(oracle) =
                    certified_dynamic_residual_oracles.remove(&entry.terminal)
                {
                    VirtualResidualRuntime::new_with_liveness_oracle(
                        expression,
                        oracle,
                        runtime_index,
                        entry.terminal,
                        self.num_terminals,
                        physical_state_count,
                        entry.root_state,
                        Arc::clone(&allocator),
                        Arc::clone(&owners),
                    )
                } else {
                    VirtualResidualRuntime::new_dynamic(
                        expression,
                        runtime_index,
                        entry.terminal,
                        self.num_terminals,
                        physical_state_count,
                        entry.root_state,
                        Arc::clone(&allocator),
                        Arc::clone(&owners),
                    )
                };
                let construct_ms = construct_started
                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                let runtime = Arc::new(
                    runtime.ok_or_else(|| "serialized residual runtime metadata is invalid".to_owned())?,
                );
                if let Some(started) = runtime_started {
                    eprintln!(
                        "[glrmask/profile][virtual_residual_restore] terminal={} support_ms={:.3} construct_ms={:.3} total_ms={:.3}",
                        entry.terminal,
                        support_ms,
                        construct_ms,
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                let mut expected_future = BitSet::new(self.num_terminals as usize);
                if runtime.root_has_future() {
                    expected_future.set(entry.terminal as usize);
                }
                if self.state_futures(entry.root_state) != &expected_future {
                    // Residual artifacts written before conservative root
                    // futures were introduced stored the exact root liveness
                    // bit. Accept only that one legacy mismatch: current
                    // metadata is conservatively live, the serialized root is
                    // dead, and an exact residual query proves the root really
                    // has no positive continuation. Any missing bit for a live
                    // root (or any other bit mismatch) remains corruption.
                    let serialized_root_future = self.state_futures(entry.root_state).clone();
                    let legacy_exact_dead = allow_legacy_exact_dead_residual_roots
                        && runtime.root_has_future()
                        && serialized_root_future.is_empty()
                        && !self.state_futures(start_state).contains(entry.terminal as usize)
                        && runtime.exact_has_future(entry.root_state)? == Some(false);
                    if !legacy_exact_dead {
                        return Err(format!(
                            "serialized residual root {} has inconsistent future metadata",
                            entry.root_state,
                        ));
                    }
                    legacy_exact_dead_roots.push((
                        entry.root_state,
                        entry.terminal,
                        expected_future.clone(),
                    ));
                }
                if runtime.root_has_future()
                    && !self.state_futures(start_state).contains(entry.terminal as usize)
                    && !legacy_exact_dead_roots
                        .iter()
                        .any(|&(root, terminal, _)| {
                            root == entry.root_state && terminal == entry.terminal
                        })
                {
                    return Err(format!(
                        "serialized residual terminal {} is missing from reset-state futures",
                        entry.terminal,
                    ));
                }
                runtimes.push(runtime);
            }
            if !legacy_exact_dead_roots.is_empty() {
                let start_finalizers = self.state_finalizers(start_state).clone();
                let mut start_futures = self.state_futures(start_state).clone();
                for (root_state, terminal, expected_future) in legacy_exact_dead_roots {
                    let root_finalizers = self.state_finalizers(root_state).clone();
                    self.dfa.overwrite_state_metadata(
                        root_state,
                        root_finalizers,
                        expected_future,
                    );
                    start_futures.set(terminal as usize);
                }
                self.dfa.overwrite_state_metadata(
                    start_state,
                    start_finalizers,
                    start_futures,
                );
            }
            self.virtual_residuals = runtimes;
            self.invalidate_derived_caches();
            return Ok(());
        }

        let allocator = Arc::new(
            VirtualStateAllocator::new(physical_state_count)
                .ok_or_else(|| "serialized virtual runtime has no state-id namespace".to_owned())?,
        );
        let roots = metadata.iter().map(|entry| entry.root_state).collect::<Vec<_>>();
        let owners = Arc::new(
            VirtualRuntimeStateOwners::new(physical_state_count, &roots).ok_or_else(|| {
                "serialized virtual product runtime has invalid state ownership metadata".to_owned()
            })?,
        );
        let mut runtimes = Vec::with_capacity(metadata.len());
        for (runtime_index, entry) in metadata.iter().enumerate() {
            if entry.kind != VirtualTokenizerRuntimeKind::RepeatProduct {
                return Err("serialized virtual runtime has unknown mixed runtime kind".to_owned());
            }
            let expression = expressions
                .get(entry.terminal as usize)
                .ok_or_else(|| "serialized virtual product terminal is out of range".to_owned())?;
            let descriptor = super::compile::virtual_binary_bounded_repeat_intersection_descriptor(expression)
                .or_else(|| super::compile::virtual_large_bounded_repeat_descriptor(expression))
                .ok_or_else(|| "serialized virtual product descriptor no longer matches terminal expression".to_owned())?;
            if self.terminal_byte_support(entry.terminal) != Some(descriptor.byte_support) {
                return Err(format!(
                    "serialized virtual product terminal {} has inconsistent byte support",
                    entry.terminal,
                ));
            }
            let runtime = Arc::new(
                VirtualBinaryRepeatIntersectionRuntime::new(
                    descriptor,
                    u32::try_from(runtime_index).map_err(|_| {
                        "serialized virtual product runtime count exceeds u32".to_owned()
                    })?,
                    entry.terminal,
                    self.num_terminals,
                    physical_state_count,
                    entry.root_state,
                    Arc::clone(&allocator),
                    Arc::clone(&owners),
                )
                .ok_or_else(|| "serialized virtual product runtime metadata is invalid".to_owned())?,
            );
            let expected_future = {
                let mut bits = BitSet::new(self.num_terminals as usize);
                if runtime.root_has_future() {
                    bits.set(entry.terminal as usize);
                }
                bits
            };
            if self.state_futures(entry.root_state) != &expected_future {
                return Err(format!(
                    "serialized virtual product root {} has inconsistent future metadata",
                    entry.root_state,
                ));
            }
            if runtime.root_has_future()
                && !self.state_futures(start_state).contains(entry.terminal as usize)
            {
                return Err(format!(
                    "serialized virtual product terminal {} is missing from reset-state futures",
                    entry.terminal,
                ));
            }
            runtimes.push(runtime);
        }
        self.virtual_repeat_intersections = runtimes;
        self.invalidate_derived_caches();
        Ok(())
    }

    pub(super) fn install_virtual_unit_repeat(
        &mut self,
        body: U8Set,
        min: usize,
        max: usize,
    ) -> Option<()> {
        if !self.virtual_repeat_intersections.is_empty()
            || !self.virtual_residuals.is_empty()
            || self.num_terminals != 1
            || self.num_states() != 1
            || self.dfa.has_epsilon_transitions()
            || self.dfa.states()[0].transitions.iter().next().is_some()
        {
            return None;
        }
        let runtime = VirtualZeroMinUnitRepeatRuntime::new(body, min, max, 0, 1, 1, 0)?;
        self.dfa.ensure_group_capacity(1);
        self.dfa.set_group_u8set(0, body);
        let mut futures = BitSet::new(1);
        futures.set(0);
        self.dfa
            .overwrite_state_metadata(0, BitSet::new(1), futures);
        self.virtual_unit_repeat = Some(Arc::new(runtime));
        self.invalidate_derived_caches();
        Some(())
    }

    pub(super) fn install_virtual_zero_min_unit_repeat(
        &mut self,
        body: U8Set,
        max: usize,
    ) -> Option<()> {
        self.install_virtual_unit_repeat(body, 0, max)
    }

    /// Add one exact arithmetic repeat as an NFA component beside the already
    /// materialized physical lexer components. The caller must have drained
    /// ordinary nullable terminals first: adding physical states afterward
    /// would collide with the arithmetic state interval which begins at the
    /// final physical state count recorded here.
    #[doc(hidden)]
    pub fn install_virtual_unit_repeat_component(
        &mut self,
        body: U8Set,
        min: usize,
        max: usize,
        terminal: TerminalID,
    ) -> Option<()> {
        if self.virtual_unit_repeat.is_some()
            || !self.virtual_repeat_intersections.is_empty()
            || !self.virtual_residuals.is_empty()
            || terminal >= self.num_terminals
            || body.is_empty()
            || max == 0
            || min > max
        {
            return None;
        }
        self.invalidate_derived_caches();
        self.dfa.ensure_group_capacity(self.num_terminals as usize);
        self.dfa.set_group_u8set(terminal, body);
        let root_state = self.dfa.add_state();
        let physical_state_count = self.dfa.num_states() as u32;

        let mut root_futures = BitSet::new(self.num_terminals as usize);
        root_futures.set(terminal as usize);
        self.dfa.overwrite_state_metadata(
            root_state,
            BitSet::new(self.num_terminals as usize),
            root_futures,
        );
        self.dfa.add_epsilon_transition(self.start_state(), root_state);

        // `recompute_possible_futures` cannot see the arithmetic byte edges.
        // Propagate the new component's future bit to the reset state by hand;
        // the epsilon closure exposes the proxy root everywhere else.
        let mut start_futures = self.dfa.possible_future_group_ids(self.start_state()).clone();
        start_futures.set(terminal as usize);
        let start_finalizers = self.dfa.finalizers(self.start_state()).clone();
        self.dfa.overwrite_state_metadata(
            self.start_state(),
            start_finalizers,
            start_futures,
        );

        self.virtual_unit_repeat = Some(Arc::new(VirtualZeroMinUnitRepeatRuntime::new(
            body,
            min,
            max,
            terminal,
            self.num_terminals,
            physical_state_count,
            root_state,
        )?));
        self.invalidate_derived_caches();
        Some(())
    }

    #[doc(hidden)]
    pub fn install_virtual_zero_min_unit_repeat_component(
        &mut self,
        body: U8Set,
        max: usize,
        terminal: TerminalID,
    ) -> Option<()> {
        self.install_virtual_unit_repeat_component(body, 0, max, terminal)
    }

    /// Add one exact lazy binary bounded-repeat intersection beside the
    /// already-materialized lexer components. The product-state IDs are stable
    /// runtime handles; the two repetition bounds never size an allocation.
    #[doc(hidden)]
    pub fn install_virtual_binary_repeat_intersection_component(
        &mut self,
        descriptor: VirtualBinaryRepeatIntersectionDescriptor,
        terminal: TerminalID,
    ) -> Option<()> {
        self.install_virtual_binary_repeat_intersection_components(vec![(descriptor, terminal)])
    }

    /// Add multiple independent exact lazy bounded-repeat components. All
    /// physical proxy roots are installed first; their lazily reached product
    /// residuals then draw globally unique handles from one shared allocator
    /// below the dynamic-NFA high-bit tag.
    #[doc(hidden)]
    pub fn install_virtual_binary_repeat_intersection_components(
        &mut self,
        components: Vec<(VirtualBinaryRepeatIntersectionDescriptor, TerminalID)>,
    ) -> Option<()> {
        if self.virtual_unit_repeat.is_some()
            || !self.virtual_repeat_intersections.is_empty()
            || !self.virtual_residuals.is_empty()
            || components.is_empty()
            || components.iter().any(|(descriptor, terminal)| {
                *terminal >= self.num_terminals || descriptor.byte_support.is_empty()
            })
        {
            return None;
        }
        let mut seen_terminals = BTreeSet::new();
        if components
            .iter()
            .any(|(_, terminal)| !seen_terminals.insert(*terminal))
        {
            return None;
        }
        // Prove every runtime can be constructed before mutating the physical
        // tokenizer. This keeps the virtual selection transaction-like: a
        // rejected descriptor cannot leave proxy roots behind before the
        // caller chooses another exact path.
        let existing_physical_state_count = u32::try_from(self.dfa.num_states()).ok()?;
        let physical_state_count = existing_physical_state_count
            .checked_add(u32::try_from(components.len()).ok()?)?;
        let allocator = Arc::new(VirtualStateAllocator::new(physical_state_count)?);
        let roots = (0..components.len())
            .map(|index| existing_physical_state_count.checked_add(u32::try_from(index).ok()?))
            .collect::<Option<Vec<_>>>()?;
        let owners = Arc::new(VirtualRuntimeStateOwners::new(physical_state_count, &roots)?);
        let mut pending = Vec::with_capacity(components.len());
        for (index, (descriptor, terminal)) in components.into_iter().enumerate() {
            let root_state = roots[index];
            let byte_support = descriptor.byte_support;
            let runtime = Arc::new(VirtualBinaryRepeatIntersectionRuntime::new(
                descriptor,
                u32::try_from(index).ok()?,
                terminal,
                self.num_terminals,
                physical_state_count,
                root_state,
                Arc::clone(&allocator),
                Arc::clone(&owners),
            )?);
            pending.push((terminal, root_state, byte_support, runtime));
        }

        self.invalidate_derived_caches();
        self.dfa.ensure_group_capacity(self.num_terminals as usize);
        for &(terminal, expected_root, byte_support, _) in &pending {
            self.dfa.set_group_u8set(terminal, byte_support);
            let root_state = self.dfa.add_state();
            debug_assert_eq!(root_state, expected_root);
            self.dfa.add_epsilon_transition(self.start_state(), root_state);
        }

        let mut runtimes = Vec::with_capacity(pending.len());
        let mut start_futures = self.dfa.possible_future_group_ids(self.start_state()).clone();
        for (terminal, root_state, _, runtime) in pending {
            let mut root_futures = BitSet::new(self.num_terminals as usize);
            if runtime.root_has_future() {
                root_futures.set(terminal as usize);
                start_futures.set(terminal as usize);
            }
            self.dfa.overwrite_state_metadata(
                root_state,
                BitSet::new(self.num_terminals as usize),
                root_futures,
            );
            runtimes.push(runtime);
        }
        let start_finalizers = self.dfa.finalizers(self.start_state()).clone();
        self.dfa.overwrite_state_metadata(
            self.start_state(),
            start_finalizers,
            start_futures,
        );
        self.virtual_repeat_intersections = runtimes;
        self.invalidate_derived_caches();
        Some(())
    }

    /// Add general exact symbolic regex components beside the materialized
    /// ordinary lexer. Every component shares one virtual-state allocator;
    /// bounds inside `Expr::Repeat` never determine an allocation size.
    #[doc(hidden)]
    pub fn install_virtual_residual_components(
        &mut self,
        components: Vec<(Expr, TerminalID)>,
    ) -> Option<()> {
        self.install_virtual_residual_components_impl(components, false, false)
    }

    /// Construct exact bounded-code liveness proofs before the physical
    /// tokenizer exists. Indexed parallel collection preserves component order
    /// so later installation assigns the same runtime/root IDs as the ordinary
    /// dynamic path.
    #[doc(hidden)]
    pub fn prepare_virtual_residual_components(
        components: Vec<(Expr, TerminalID)>,
    ) -> Option<Vec<PreparedVirtualResidualComponent>> {
        components
            .into_par_iter()
            .map(|(expression, terminal)| {
                let liveness_oracle = build_bounded_code_liveness_oracle(&expression)?;
                Some(PreparedVirtualResidualComponent {
                    expression,
                    terminal,
                    liveness_oracle,
                })
            })
            .collect()
    }

    /// Install residual components whose exact dynamic liveness proofs were
    /// prepared independently of this tokenizer.
    #[doc(hidden)]
    pub fn install_prepared_virtual_residual_components(
        &mut self,
        components: Vec<PreparedVirtualResidualComponent>,
    ) -> Option<()> {
        if self.virtual_unit_repeat.is_some()
            || !self.virtual_repeat_intersections.is_empty()
            || !self.virtual_residuals.is_empty()
            || components.is_empty()
            || components
                .iter()
                .any(|component| component.terminal >= self.num_terminals)
        {
            return None;
        }
        let mut seen_terminals = BTreeSet::new();
        if components
            .iter()
            .any(|component| !seen_terminals.insert(component.terminal))
        {
            return None;
        }

        let existing_physical_state_count = u32::try_from(self.dfa.num_states()).ok()?;
        let physical_state_count = existing_physical_state_count
            .checked_add(u32::try_from(components.len()).ok()?)?;
        let allocator = Arc::new(VirtualStateAllocator::new(physical_state_count)?);
        let roots = (0..components.len())
            .map(|index| existing_physical_state_count.checked_add(u32::try_from(index).ok()?))
            .collect::<Option<Vec<_>>>()?;
        let owners = Arc::new(VirtualRuntimeStateOwners::new(physical_state_count, &roots)?);

        let pending = components
            .into_iter()
            .enumerate()
            .map(|(index, component)| {
                let root_state = roots[index];
                let byte_support = super::compile::expr_u8set(&component.expression);
                let runtime_index = u32::try_from(index).ok()?;
                let runtime = VirtualResidualRuntime::new_with_liveness_oracle(
                    &component.expression,
                    component.liveness_oracle,
                    runtime_index,
                    component.terminal,
                    self.num_terminals,
                    physical_state_count,
                    root_state,
                    Arc::clone(&allocator),
                    Arc::clone(&owners),
                )?;
                Some((
                    component.terminal,
                    root_state,
                    byte_support,
                    Arc::new(runtime),
                ))
            })
            .collect::<Option<Vec<_>>>()?;

        self.invalidate_derived_caches();
        self.dfa.ensure_group_capacity(self.num_terminals as usize);
        for &(terminal, expected_root, byte_support, _) in &pending {
            self.dfa.set_group_u8set(terminal, byte_support);
            let root_state = self.dfa.add_state();
            debug_assert_eq!(root_state, expected_root);
            self.dfa.add_epsilon_transition(self.start_state(), root_state);
        }

        let mut runtimes = Vec::with_capacity(pending.len());
        let mut start_futures = self.dfa.possible_future_group_ids(self.start_state()).clone();
        for (terminal, root_state, _, runtime) in pending {
            let mut root_futures = BitSet::new(self.num_terminals as usize);
            if runtime.root_has_future() {
                root_futures.set(terminal as usize);
                start_futures.set(terminal as usize);
            }
            self.dfa.overwrite_state_metadata(
                root_state,
                BitSet::new(self.num_terminals as usize),
                root_futures,
            );
            runtimes.push(runtime);
        }
        let start_finalizers = self.dfa.finalizers(self.start_state()).clone();
        self.dfa.overwrite_state_metadata(
            self.start_state(),
            start_finalizers,
            start_futures,
        );
        self.virtual_residuals = runtimes;
        self.invalidate_derived_caches();
        Some(())
    }

    /// Install residual proxy/runtime metadata for a transfer-producing
    /// dynamic compile without eagerly constructing bounded-code liveness
    /// oracles that are not serialized in the compact artifact.
    #[doc(hidden)]
    pub fn install_virtual_residual_components_without_liveness_oracles(
        &mut self,
        components: Vec<(Expr, TerminalID)>,
    ) -> Option<()> {
        self.install_virtual_residual_components_impl(components, false, true)
    }

    #[doc(hidden)]
    pub fn install_virtual_residual_components_preserving_oracle_coordinates(
        &mut self,
        components: Vec<(Expr, TerminalID)>,
    ) -> Option<()> {
        self.install_virtual_residual_components_impl(components, true, false)
    }

    fn install_virtual_residual_components_impl(
        &mut self,
        components: Vec<(Expr, TerminalID)>,
        preserve_oracle_coordinate: bool,
        suppress_dynamic_liveness_oracle: bool,
    ) -> Option<()> {
        if self.virtual_unit_repeat.is_some()
            || !self.virtual_repeat_intersections.is_empty()
            || !self.virtual_residuals.is_empty()
            || components.is_empty()
            || components
                .iter()
                .any(|(_, terminal)| *terminal >= self.num_terminals)
        {
            return None;
        }
        let mut seen_terminals = BTreeSet::new();
        if components
            .iter()
            .any(|(_, terminal)| !seen_terminals.insert(*terminal))
        {
            return None;
        }

        let existing_physical_state_count = u32::try_from(self.dfa.num_states()).ok()?;
        let physical_state_count = existing_physical_state_count
            .checked_add(u32::try_from(components.len()).ok()?)?;
        let allocator = Arc::new(VirtualStateAllocator::new(physical_state_count)?);
        let roots = (0..components.len())
            .map(|index| existing_physical_state_count.checked_add(u32::try_from(index).ok()?))
            .collect::<Option<Vec<_>>>()?;
        let owners = Arc::new(VirtualRuntimeStateOwners::new(physical_state_count, &roots)?);
        // General residual components are independent until their proxy roots
        // are appended below. Building a bounded-code oracle can involve DFA
        // construction and relation doubling, so doing that serially makes a
        // multi-string schema pay the sum of all component setup costs even
        // though the resulting runtime ordering is fixed. Indexed parallel
        // collection preserves input order and therefore root/runtime IDs.
        let pending = components
            .into_par_iter()
            .enumerate()
            .map(|(index, (expression, terminal))| {
                let root_state = roots[index];
                let byte_support = super::compile::expr_u8set(&expression);
                let runtime_index = u32::try_from(index).ok()?;
                let runtime = if preserve_oracle_coordinate {
                    VirtualResidualRuntime::new_preserving_oracle_coordinate(
                        &expression,
                        runtime_index,
                        terminal,
                        self.num_terminals,
                        physical_state_count,
                        root_state,
                        Arc::clone(&allocator),
                        Arc::clone(&owners),
                    )?
                } else if suppress_dynamic_liveness_oracle {
                    VirtualResidualRuntime::new_without_liveness_oracle(
                        &expression,
                        runtime_index,
                        terminal,
                        self.num_terminals,
                        physical_state_count,
                        root_state,
                        Arc::clone(&allocator),
                        Arc::clone(&owners),
                    )?
                } else {
                    VirtualResidualRuntime::new_dynamic(
                        &expression,
                        runtime_index,
                        terminal,
                        self.num_terminals,
                        physical_state_count,
                        root_state,
                        Arc::clone(&allocator),
                        Arc::clone(&owners),
                    )?
                };
                Some((terminal, root_state, byte_support, Arc::new(runtime)))
            })
            .collect::<Option<Vec<_>>>()?;

        self.invalidate_derived_caches();
        self.dfa.ensure_group_capacity(self.num_terminals as usize);
        for &(terminal, expected_root, byte_support, _) in &pending {
            self.dfa.set_group_u8set(terminal, byte_support);
            let root_state = self.dfa.add_state();
            debug_assert_eq!(root_state, expected_root);
            self.dfa.add_epsilon_transition(self.start_state(), root_state);
        }

        let mut runtimes = Vec::with_capacity(pending.len());
        let mut start_futures = self.dfa.possible_future_group_ids(self.start_state()).clone();
        for (terminal, root_state, _, runtime) in pending {
            let mut root_futures = BitSet::new(self.num_terminals as usize);
            if runtime.root_has_future() {
                root_futures.set(terminal as usize);
                start_futures.set(terminal as usize);
            }
            self.dfa.overwrite_state_metadata(
                root_state,
                BitSet::new(self.num_terminals as usize),
                root_futures,
            );
            runtimes.push(runtime);
        }
        let start_finalizers = self.dfa.finalizers(self.start_state()).clone();
        self.dfa.overwrite_state_metadata(
            self.start_state(),
            start_finalizers,
            start_futures,
        );
        self.virtual_residuals = runtimes;
        self.invalidate_derived_caches();
        Some(())
    }

    #[doc(hidden)]
    pub fn has_any_virtual_runtime(&self) -> bool {
        self.virtual_unit_repeat.is_some()
            || !self.virtual_repeat_intersections.is_empty()
            || !self.virtual_residuals.is_empty()
    }

    #[doc(hidden)]
    pub fn has_compressed_transition_segments(&self) -> bool {
        !self.compressed_transition_segments.is_empty()
            || !self.packed_compressed_transition_segments.is_empty()
    }

    #[doc(hidden)]
    pub fn has_virtual_residual_runtime(&self) -> bool {
        !self.virtual_residuals.is_empty()
    }

    #[doc(hidden)]
    pub fn virtual_residual_interned_state_count(&self) -> usize {
        self.virtual_residuals
            .iter()
            .map(|runtime| runtime.interned_state_count())
            .sum()
    }

    #[doc(hidden)]
    pub fn virtual_residual_bounded_code_liveness_oracle_count(&self) -> usize {
        self.virtual_residuals
            .iter()
            .filter(|runtime| runtime.has_bounded_code_liveness_oracle())
            .count()
    }

    /// Cheap upper-coordinate work estimate for the finite one-token mask
    /// projection of every virtual residual component. `None` means at least
    /// one component cannot use that projection at this horizon.
    #[doc(hidden)]
    pub fn virtual_residual_mask_projection_dense_state_work(
        &self,
        horizon: usize,
    ) -> Option<usize> {
        if self.virtual_residuals.is_empty() {
            return None;
        }
        self.virtual_residuals.iter().try_fold(0usize, |total, runtime| {
            total.checked_add(runtime.finite_mask_projection_dense_state_count(horizon)?)
        })
    }

    /// Exact liveness at a dynamic token boundary. Ordinary, specialized, and
    /// certified bounded-code residual states already expose exact future
    /// metadata. Other general symbolic residuals deliberately expose a
    /// conservative infallible future bitset, so this boundary resolves them
    /// through the bounded exact derivative search. Resource exhaustion is an
    /// error rather than a false dead/live answer.
    #[doc(hidden)]
    pub fn exact_dynamic_state_has_future(&self, state: u32) -> Result<bool, String> {
        if let Some(runtime) = self.virtual_residual_runtime_for_state(state) {
            return runtime
                .exact_has_future(state)?
                .ok_or_else(|| "residual runtime owner index lost state ownership".to_owned());
        }
        Ok(!self.is_end(state))
    }

    /// Exact single-byte transition used by the dynamic masker's
    /// first-match raw-residual lane. This deliberately exposes only the
    /// ordinary tokenizer transition semantics; it does not consult any DWA
    /// or token-effect cache.
    #[doc(hidden)]
    #[inline]
    pub fn dynamic_direct_transition(&self, state: u32, byte: u8) -> u32 {
        self.get_transition(state, byte)
    }

    /// Cheap outgoing-byte count for ordinary physical runtime states.
    /// Returns `None` for virtual/compressed coordinates where answering this
    /// exactly would itself require expanding transition classes.
    #[doc(hidden)]
    #[inline]
    pub fn dynamic_direct_transition_count(&self, state: u32) -> Option<usize> {
        if self.virtual_residual_runtime_for_state(state).is_some()
            || self.virtual_repeat_runtime_for_state(state).is_some()
            || self
                .virtual_unit_repeat
                .as_deref()
                .is_some_and(|runtime| runtime.handles_state(state))
        {
            return None;
        }
        if let Some(packed) = &self.packed_runtime_transitions
            && let Some((bytes, _)) = packed.row(state)
        {
            return Some(bytes.len());
        }
        if let Some(segment) = self.packed_runtime_transition_segment_for_state(state)
            && let Some((bytes, _)) = segment.row(state)
        {
            return Some(bytes.len());
        }
        if self.compressed_segment_for_state(state).is_some()
            || self.packed_compressed_segment_for_state(state).is_some()
        {
            return None;
        }
        self.dfa
            .states()
            .get(state as usize)
            .map(|state| state.transitions.len())
    }

    /// Parser-transparent finite-horizon byte-family proof for an exact
    /// bounded-code virtual residual state. This deliberately avoids building
    /// the finite mask projection; `None` means this state cannot be certified
    /// cheaply and callers should use the ordinary exact walk.
    #[doc(hidden)]
    pub fn virtual_residual_parser_transparent_byte_family(
        &self,
        state: u32,
        bytes: U8Set,
        max_horizon: u32,
    ) -> Option<bool> {
        self.virtual_residual_runtime_for_state(state)?
            .parser_transparent_byte_family(state, bytes, max_horizon)
    }

    #[doc(hidden)]
    pub fn virtual_residual_parser_transparent_byte_dfa(
        &self,
        state: u32,
        slice_start: u32,
        slice_class_count: usize,
        slice_byte_to_class: &[u8; 256],
        slice_transitions: &[u32],
        slice_can_reach_accepting: &[bool],
        slice_language_finite: bool,
        work_limit: usize,
    ) -> Option<bool> {
        self.virtual_residual_runtime_for_state(state)?
            .parser_transparent_byte_dfa(
                state,
                slice_start,
                slice_class_count,
                slice_byte_to_class,
                slice_transitions,
                slice_can_reach_accepting,
                slice_language_finite,
                work_limit,
            )
    }

    /// Exact whole-atom length ceiling for one virtual residual, if its
    /// finite code envelope and closing suffix establish that certificate.
    #[doc(hidden)]
    pub fn virtual_residual_safe_atom_length_upper_bound(
        &self,
        state: u32,
        slice_start: u32,
        slice_class_count: usize,
        slice_byte_to_class: &[u8; 256],
        slice_transitions: &[u32],
        slice_accepting: &[bool],
        slice_can_reach_accepting: &[bool],
    ) -> Option<u32> {
        self.virtual_residual_runtime_for_state(state)?.safe_atom_length_upper_bound(
            state, slice_start, slice_class_count, slice_byte_to_class,
            slice_transitions, slice_accepting, slice_can_reach_accepting,
        )
    }

    #[doc(hidden)]
    pub fn virtual_residual_parser_transparent_byte_dfa_repeat_radius(
        &self,
        state: u32,
        slice_start: u32,
        slice_class_count: usize,
        slice_byte_to_class: &[u8; 256],
        slice_transitions: &[u32],
        slice_accepting: &[bool],
        slice_can_reach_accepting: &[bool],
        max_repetitions: u32,
        work_limit: usize,
    ) -> Option<u32> {
        self.virtual_residual_runtime_for_state(state)?
            .parser_transparent_byte_dfa_repeat_radius(
                state,
                slice_start,
                slice_class_count,
                slice_byte_to_class,
                slice_transitions,
                slice_accepting,
                slice_can_reach_accepting,
                max_repetitions,
                work_limit,
            )
    }

    #[doc(hidden)]
    pub fn prepare_virtual_residual_master_slice_artifacts(
        &self,
        slice_start: u32,
        slice_class_count: usize,
        slice_byte_to_class: &[u8; 256],
        slice_transitions: &[u32],
        slice_accepting: &[bool],
        slice_can_reach_accepting: &[bool],
        max_repetitions: u32,
    ) {
        for runtime in &self.virtual_residuals {
            runtime.prepare_master_slice_artifacts(
                slice_start,
                slice_class_count,
                slice_byte_to_class,
                slice_transitions,
                slice_accepting,
                slice_can_reach_accepting,
                max_repetitions,
            );
        }
    }

    /// Owning terminal for an exact virtual-residual state. This is runtime
    /// provenance only: callers still need to establish parser admission before
    /// using a residual-language certificate as a mask shortcut.
    #[doc(hidden)]
    pub fn virtual_residual_terminal_for_state(&self, state: u32) -> Option<TerminalID> {
        self.virtual_residual_runtime_for_state(state)
            .map(VirtualResidualRuntime::terminal)
    }

    #[doc(hidden)]
    pub fn virtual_residual_direct_coordinate(
        &self,
        state: u32,
    ) -> Option<VirtualResidualDirectCoordinate> {
        self.virtual_residual_runtime_for_state(state)?
            .direct_coordinate_for_state(state)
    }

    #[doc(hidden)]
    #[inline(always)]
    pub fn virtual_residual_direct_coordinate_step(
        &self,
        source: VirtualResidualDirectCoordinate,
        byte: u8,
    ) -> Option<VirtualResidualDirectCoordinate> {
        self.virtual_residuals
            .get(source.runtime_index as usize)?
            .direct_coordinate_step(source, byte)
    }

    #[doc(hidden)]
    #[inline(always)]
    pub fn virtual_residual_direct_coordinate_accepting(
        &self,
        source: VirtualResidualDirectCoordinate,
    ) -> Option<bool> {
        self.virtual_residuals
            .get(source.runtime_index as usize)?
            .direct_coordinate_accepting(source)
    }

    #[doc(hidden)]
    pub fn virtual_residual_direct_coordinate_has_future(
        &self,
        source: VirtualResidualDirectCoordinate,
    ) -> Option<bool> {
        self.virtual_residuals
            .get(source.runtime_index as usize)?
            .direct_coordinate_has_future(source)
    }

    #[doc(hidden)]
    pub fn virtual_residual_direct_coordinate_finite_mask_dense_key(
        &self,
        source: VirtualResidualDirectCoordinate,
        max_token_len: usize,
    ) -> Option<(u32, u32)> {
        self.virtual_residuals
            .get(source.runtime_index as usize)?
            .direct_coordinate_finite_mask_dense_key(source, max_token_len)
    }

    /// Result-cache-only exact finite-vocabulary quotient. Recognition,
    /// compiled projections and byte-transition caches keep the global stencil.
    #[doc(hidden)]
    pub fn virtual_residual_direct_coordinate_mask_result_dense_key(
        &self,
        source: VirtualResidualDirectCoordinate,
        max_token_len: usize,
    ) -> Option<(u32, u32)> {
        self.virtual_residuals
            .get(source.runtime_index as usize)?
            .direct_coordinate_mask_result_dense_key(source, max_token_len)
    }

    #[doc(hidden)]
    pub fn virtual_residual_direct_coordinate_parser_transparent_byte_dfa(
        &self,
        source: VirtualResidualDirectCoordinate,
        slice_start: u32,
        slice_class_count: usize,
        slice_byte_to_class: &[u8; 256],
        slice_transitions: &[u32],
        slice_can_reach_accepting: &[bool],
        slice_language_finite: bool,
        work_limit: usize,
    ) -> Option<bool> {
        self.virtual_residuals
            .get(source.runtime_index as usize)?
            .direct_coordinate_parser_transparent_byte_dfa(
                source,
                slice_start,
                slice_class_count,
                slice_byte_to_class,
                slice_transitions,
                slice_can_reach_accepting,
                slice_language_finite,
                work_limit,
            )
    }

    #[doc(hidden)]
    pub fn virtual_residual_state_for_direct_coordinate(
        &self,
        source: VirtualResidualDirectCoordinate,
    ) -> Option<u32> {
        self.virtual_residuals
            .get(source.runtime_index as usize)?
            .state_for_direct_coordinate(source)
    }

    #[doc(hidden)]
    pub fn has_virtual_binary_repeat_intersection(&self) -> bool {
        !self.virtual_repeat_intersections.is_empty()
    }

    #[doc(hidden)]
    pub fn virtual_binary_repeat_intersection_interned_state_count(&self) -> usize {
        self.virtual_repeat_intersections
            .iter()
            .map(|runtime| runtime.interned_state_count())
            .sum()
    }

    #[doc(hidden)]
    pub fn virtual_binary_repeat_intersection_mask_tokenizer(
        &self,
        horizon: usize,
    ) -> Option<(Tokenizer, VirtualBinaryRepeatIntersectionMaskProjection)> {
        let (tokenizer, mut projections) =
            self.virtual_binary_repeat_intersections_mask_tokenizer(horizon)?;
        (projections.len() == 1).then(|| (tokenizer, projections.remove(0)))
    }

    #[doc(hidden)]
    pub fn restore_compiled_virtual_residual_mask_projections(
        &self,
        mask_tokenizer: &Tokenizer,
        artifacts: Vec<VirtualResidualMaskProjectionArtifact>,
    ) -> Result<Vec<VirtualResidualMaskProjection>, String> {
        if artifacts.len() != self.virtual_residuals.len() {
            return Err(format!(
                "compiled virtual residual projection count mismatch: artifact={} runtime={}",
                artifacts.len(), self.virtual_residuals.len(),
            ));
        }
        if mask_tokenizer.num_terminals() != self.num_terminals || mask_tokenizer.has_any_virtual_runtime() {
            return Err("compiled virtual residual mask tokenizer is incompatible".to_owned());
        }
        let mask_states = mask_tokenizer.num_states();
        let offsets = artifacts.iter().map(VirtualResidualMaskProjectionArtifact::state_offset).collect::<Vec<_>>();
        if offsets.windows(2).any(|pair| pair[0] >= pair[1])
            || offsets.last().is_some_and(|&offset| offset >= mask_states)
        {
            return Err("compiled virtual residual projection offsets are invalid".to_owned());
        }
        let mut restored = Vec::with_capacity(artifacts.len());
        for (index, (runtime, artifact)) in self.virtual_residuals.iter().zip(artifacts).enumerate() {
            let start = offsets[index];
            let end = offsets.get(index + 1).copied().unwrap_or(mask_states);
            if end <= start {
                return Err("compiled virtual residual projection component is empty".to_owned());
            }
            restored.push(runtime.restore_compiled_finite_mask_projection(end - start, artifact)?);
        }
        Ok(restored)
    }

    #[doc(hidden)]
    pub fn restore_virtual_residual_mask_projections(
        &self,
        horizon: usize,
        mask_tokenizer: &Tokenizer,
        artifacts: Vec<VirtualResidualMaskProjectionArtifact>,
    ) -> Result<Vec<VirtualResidualMaskProjection>, String> {
        if artifacts.len() != self.virtual_residuals.len() {
            return Err(format!(
                "virtual residual projection count mismatch: artifact={} runtime={}",
                artifacts.len(),
                self.virtual_residuals.len(),
            ));
        }
        if mask_tokenizer.num_terminals() != self.num_terminals {
            return Err(format!(
                "virtual residual mask tokenizer terminal mismatch: mask={} source={}",
                mask_tokenizer.num_terminals(),
                self.num_terminals,
            ));
        }
        if mask_tokenizer.has_any_virtual_runtime() {
            return Err("serialized virtual residual mask tokenizer must be finite".to_owned());
        }
        let mask_states = mask_tokenizer.num_states();
        let offsets = artifacts
            .iter()
            .map(VirtualResidualMaskProjectionArtifact::state_offset)
            .collect::<Vec<_>>();
        if offsets.windows(2).any(|pair| pair[0] >= pair[1])
            || offsets.last().is_some_and(|&offset| offset >= mask_states)
        {
            return Err("virtual residual projection offsets are invalid".to_owned());
        }
        let mut restored = Vec::with_capacity(artifacts.len());
        for (index, (runtime, artifact)) in self
            .virtual_residuals
            .iter()
            .zip(artifacts.into_iter())
            .enumerate()
        {
            let start = offsets[index];
            let end = offsets.get(index + 1).copied().unwrap_or(mask_states);
            if end <= start {
                return Err("virtual residual projection component is empty".to_owned());
            }
            restored.push(runtime.restore_finite_mask_projection(
                horizon,
                end - start,
                artifact,
            )?);
        }
        Ok(restored)
    }

    pub fn virtual_residuals_mask_tokenizer(
        &self,
        horizon: usize,
    ) -> Option<(Tokenizer, Vec<VirtualResidualMaskProjection>)> {
        self.virtual_residuals_mask_tokenizer_with_vocab(horizon, None)
    }

    pub fn virtual_residuals_mask_tokenizer_with_vocab(
        &self,
        horizon: usize,
        vocab: Option<&crate::Vocab>,
    ) -> Option<(Tokenizer, Vec<VirtualResidualMaskProjection>)> {
        if self.virtual_residuals.is_empty() {
            return None;
        }
        if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_VIRTUAL_RESIDUAL_PROJECTION").is_some() {
            let estimates = self
                .virtual_residuals
                .iter()
                .map(|runtime| runtime.finite_mask_projection_dense_state_count(horizon))
                .collect::<Vec<_>>();
            eprintln!(
                "[glrmask/profile][virtual_residual_projection_estimate] source_states={} horizon={} residuals={} dense_state_estimates={:?}",
                self.num_states(),
                horizon,
                self.virtual_residuals.len(),
                estimates,
            );
        }
        let profile_projection = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_VIRTUAL_RESIDUAL_PROJECTION").is_some();
        let projection_total_started = std::time::Instant::now();
        let clone_started = std::time::Instant::now();
        let mut mask = self.clone();
        let clone_ms = clone_started.elapsed().as_secs_f64() * 1000.0;
        mask.virtual_unit_repeat = None;
        mask.virtual_repeat_intersections.clear();
        mask.virtual_residuals.clear();
        // The projection below replaces virtual proxy epsilon roots and appends
        // finite components. Artifact-loaded tokenizers may keep epsilon closure
        // metadata packed outside `dfa`; materialize that metadata before any
        // structural mutation or the raw DFA appears to have no proxy roots and
        // newly-added epsilon edges would be shadowed by the packed metadata.
        mask.materialize_runtime_metadata_for_structural_mutation();
        let start = mask.start_state();
        let repeat_horizons = super::compile::VocabularyRepeatHorizonCache::new();

        // Disconnect every symbolic residual proxy before appending the finite
        // replacements. This lets us certify the retained physical graph while
        // it is still small; appended finite projection components are ordinary
        // epsilon-free DFAs by construction. The combined certificate is the
        // same invariant checked by `has_scalar_deterministic_dispatch()`, but
        // avoids walking every appended state again on first mask use.
        {
            let root_eps = &mut mask.dfa.states_mut()[start as usize].epsilon_transitions;
            for runtime in &self.virtual_residuals {
                let proxy_root = runtime.root_state();
                let before = root_eps.len();
                root_eps.retain(|&state| state != proxy_root);
                if root_eps.len() == before {
                    return None;
                }
            }
        }
        let retained_roots = mask.dfa.states()[start as usize].epsilon_transitions.clone();
        let mut retained_scalar = mask.transitions_from(start).next().is_none();
        if retained_scalar {
            let mut seen = vec![false; mask.num_states() as usize];
            let mut pending = retained_roots;
            while let Some(state) = pending.pop() {
                let Some(slot) = seen.get_mut(state as usize) else {
                    retained_scalar = false;
                    break;
                };
                if *slot {
                    continue;
                }
                *slot = true;
                if mask.state_has_epsilon_transitions(state) {
                    retained_scalar = false;
                    break;
                }
                pending.extend(mask.transitions_from(state).map(|(_, target)| target));
            }
        }

        // Residual components are independent symbolic languages, but JSON
        // schemas commonly contain many distinct terminals with exactly the
        // same bounded-code residual language.  Terminal identity must remain
        // distinct at runtime, so we cannot collapse their logical lexer state
        // ranges; however the expensive finite DFA construction is identical.
        // Build one exact template per semantic language key, then clone/rebase
        // that template with terminal-local metadata for every runtime.
        let build_started = std::time::Instant::now();
        let mut grouped = Vec::<Vec<(usize, Arc<VirtualResidualRuntime>, Option<usize>)>>::new();
        let mut group_by_key = FxHashMap::default();
        for (index, runtime) in self.virtual_residuals.iter().enumerate() {
            let key = runtime.finite_mask_projection_language_key()?;
            let crossed_boundaries = vocab.and_then(|vocab| {
                runtime.vocabulary_repeat_boundary_horizon(vocab, &repeat_horizons)
            });
            let group = if let Some(&group) = group_by_key.get(&key) {
                group
            } else {
                let group = grouped.len();
                group_by_key.insert(key, group);
                grouped.push(Vec::new());
                group
            };
            grouped[group].push((index, Arc::clone(runtime), crossed_boundaries));
        }
        let built = grouped
            .par_iter()
            .map(|members| {
                let (_, runtime, crossed_boundaries) = members.first()?;
                debug_assert!(members
                    .iter()
                    .all(|(_, _, other)| other == crossed_boundaries));
                let (component, segment, local_root, projection) = if let Some(crossed_boundaries) = crossed_boundaries {
                    runtime.build_finite_mask_projection_for_crossed_boundaries(*crossed_boundaries, 0)?
                } else {
                    runtime.build_finite_mask_projection(horizon, 0)?
                };
                Some((component, segment, local_root, projection))
            })
            .collect::<Option<Vec<_>>>()?;
        let build_ms = build_started.elapsed().as_secs_f64() * 1000.0;
        if profile_projection {
            for (group, (component, _, local_root, _)) in built.iter().enumerate() {
                let terminals = grouped[group]
                    .iter()
                    .map(|(_, runtime, _)| runtime.terminal())
                    .collect::<Vec<_>>();
                eprintln!(
                    "[glrmask/profile][virtual_residual_projection_component] group={} terminals={:?} states={} local_root={}",
                    group,
                    terminals,
                    component.num_states(),
                    local_root,
                );
            }
        }

        let append_started = std::time::Instant::now();
        let mut projections = vec![None; self.virtual_residuals.len()];
        let mut appended_components_scalar = true;
        for (component, _, _, _) in &built {
            appended_components_scalar &= component
                .states()
                .iter()
                .all(|state| state.epsilon_transitions.is_empty());
        }
        let mut member_to_group = vec![0usize; self.virtual_residuals.len()];
        for (group, members) in grouped.iter().enumerate() {
            for (member_index, _, _) in members {
                member_to_group[*member_index] = group;
            }
        }
        let mut append_components = Vec::<(&DFA, usize)>::with_capacity(self.virtual_residuals.len());
        for (member_index, runtime) in self.virtual_residuals.iter().enumerate() {
            let group = member_to_group[member_index];
            append_components.push((&built[group].0, runtime.terminal() as usize));
        }
        let offsets = mask
            .dfa
            .append_rebased_single_group_component_refs_batch(&append_components)?;
        let mut compressed_segments = mask.compressed_transition_segments.to_vec();
        for ((member_index, runtime), offset) in
            self.virtual_residuals.iter().enumerate().zip(offsets)
        {
            let group = member_to_group[member_index];
            let mut segment = built[group].1.clone();
            segment.state_offset = offset;
            compressed_segments.push(segment);
            let local_root = built[group].2;
            mask.dfa.add_epsilon_transition(start, offset.checked_add(local_root)?);
            let mut projection = built[group]
                .3
                .clone_for_equivalent_runtime(Arc::clone(runtime))?;
            projection.set_state_offset(offset);
            projections[member_index] = Some(projection);
        }
        mask.compressed_transition_segments = Arc::from(compressed_segments.into_boxed_slice());
        let projections = projections.into_iter().collect::<Option<Vec<_>>>()?;
        let append_ms = append_started.elapsed().as_secs_f64() * 1000.0;
        // Do not globally recompute futures here. `mask` may retain packed
        // byte-transition rows for the ordinary physical states; the raw DFA
        // intentionally does not contain those rows, so a DFA-only fixpoint
        // would erase valid physical future-terminal metadata. The existing
        // physical metadata is already exact, each appended finite component
        // carries exact remapped metadata, and replacing a residual proxy root
        // with the finite root preserves the same terminal future at `start`.
        let futures_ms = 0.0;
        mask.invalidate_derived_caches();
        if retained_scalar
            && appended_components_scalar
            && mask.dfa.states()[start as usize].epsilon_transitions.len() >= 2
        {
            let _ = mask.scalar_deterministic_dispatch_cache.set(true);
        }
        if profile_projection {
            eprintln!(
                "[glrmask/profile][virtual_residual_projection_phases] clone_ms={:.3} build_ms={:.3} append_ms={:.3} futures_ms={:.3} total_ms={:.3} states={}",
                clone_ms,
                build_ms,
                append_ms,
                futures_ms,
                projection_total_started.elapsed().as_secs_f64() * 1000.0,
                mask.num_states(),
            );
        }
        Some((mask, projections))
    }

    /// Attach already-finite one-token observation components to an ordinary
    /// physical tokenizer. This is intentionally projection-free: the caller
    /// has no exact runtime coordinate to map from.
    #[doc(hidden)]
    pub fn install_direct_mask_components(
        &mut self,
        components: Vec<(DFA, u32, TerminalID)>,
    ) -> Option<()> {
        self.install_direct_mask_components_impl(components, false)
    }

    /// Shared-component form used when multiple terminals have the exact same
    /// finite mask language. Each terminal still receives its own rebased copy
    /// in the physical tokenizer, while the residual sidecar can retain one
    /// immutable component allocation for all aliases.
    #[doc(hidden)]
    pub fn install_direct_mask_components_shared(
        &mut self,
        components: Vec<(Arc<DFA>, u32, TerminalID)>,
    ) -> Option<()> {
        self.install_direct_mask_component_alias_groups_impl(
            components
                .into_iter()
                .map(|(component, root, terminal)| (component, root, vec![terminal]))
                .collect(),
            false,
        )
    }

    /// Attach one physical finite component for several terminals with an
    /// exactly identical language. The shared states finalize every alias and
    /// retain one residual coordinate per alias/local-state pair.
    #[doc(hidden)]
    pub fn install_direct_mask_component_alias_groups(
        &mut self,
        components: Vec<(Arc<DFA>, u32, Vec<TerminalID>)>,
    ) -> Option<()> {
        self.install_direct_mask_component_alias_groups_impl(components, false)
    }

    fn install_direct_mask_components_impl(
        &mut self,
        components: Vec<(DFA, u32, TerminalID)>,
        force_full_future_recompute: bool,
    ) -> Option<()> {
        self.install_direct_mask_component_alias_groups_impl(
            components
                .into_iter()
                .map(|(component, root, terminal)| {
                    (Arc::new(component), root, vec![terminal])
                })
                .collect(),
            force_full_future_recompute,
        )
    }

    fn install_direct_mask_component_alias_groups_impl(
        &mut self,
        components: Vec<(Arc<DFA>, u32, Vec<TerminalID>)>,
        force_full_future_recompute: bool,
    ) -> Option<()> {
        if components.is_empty() {
            return Some(());
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let physical_component_count = components.len();
        let component_count = components
            .iter()
            .map(|(_, _, terminals)| terminals.len())
            .sum::<usize>();
        let appended_state_count = components
            .iter()
            .map(|(component, _, _)| component.num_states())
            .sum::<usize>();
        self.dfa.reserve_additional_states(appended_state_count);
        let start = self.start_state();
        // The VocabPartition caller isolates the dispatch start before appending
        // components. If that start has no incoming byte or epsilon edge, no
        // pre-existing state can newly reach an appended component: only the
        // start state's strict future set changes. Verify that invariant rather
        // than assuming it so arbitrary/debug tokenizers retain the exact full
        // recomputation fallback.
        let incoming_scan_started = profile.then(std::time::Instant::now);
        let start_has_incoming = self.dfa.states().iter().any(|state| {
            state
                .transitions
                .iter()
                .any(|(_, &target)| target == start)
                || state.epsilon_transitions.contains(&start)
        });
        let incoming_scan_ms = incoming_scan_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let incremental_futures = !force_full_future_recompute && !start_has_incoming;
        let mut start_futures = incremental_futures
            .then(|| self.dfa.possible_future_group_ids(start).clone());
        let mut coordinate_replacements = self
            .terminal_residual_coordinates
            .as_ref()
            .map(|_| Vec::with_capacity(components.len()));
        let append_started = profile.then(std::time::Instant::now);
        for (component, local_root, terminals) in components {
            if terminals.is_empty()
                || terminals.iter().any(|&terminal| terminal >= self.num_terminals)
                || local_root as usize >= component.num_states()
            {
                return None;
            }
            let canonical_terminal = terminals[0];
            let root_has_future = component.possible_future_group_ids(local_root).contains(0);
            let offset = u32::try_from(self.dfa.num_states()).ok()?;
            let actual_offset = self
                .dfa
                .append_rebased_component_ref(component.as_ref(), &[canonical_terminal as usize]);
            if actual_offset != offset {
                return None;
            }
            if terminals.len() > 1 {
                let group_bytes = component.group_id_to_u8set(0).clone();
                for &terminal in &terminals[1..] {
                    self.dfa
                        .set_group_u8set(terminal, group_bytes.clone());
                }
                let end = offset as usize + component.num_states();
                for state in &mut self.dfa.states_mut()[offset as usize..end] {
                    let finalizes = state.finalizers.contains(canonical_terminal as usize);
                    let has_future = state
                        .possible_future_group_ids
                        .contains(canonical_terminal as usize);
                    for &terminal in &terminals[1..] {
                        if finalizes {
                            state.finalizers.set(terminal as usize);
                        }
                        if has_future {
                            state.possible_future_group_ids.set(terminal as usize);
                        }
                    }
                }
            }
            self.dfa
                .add_epsilon_transition(start, offset.checked_add(local_root)?);
            if root_has_future && let Some(start_futures) = start_futures.as_mut() {
                for &terminal in &terminals {
                    start_futures.set(terminal as usize);
                }
            }
            if let Some(replacements) = coordinate_replacements.as_mut() {
                replacements.push((terminals, Arc::clone(&component)));
            }
        }
        let append_ms = append_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let coordinates_started = profile.then(std::time::Instant::now);
        if let (Some(coordinates), Some(replacements)) = (
            self.terminal_residual_coordinates.as_ref().map(Arc::clone),
            coordinate_replacements.as_deref(),
        ) {
            self.terminal_residual_coordinates = Some(Arc::new(
                coordinates.replace_terminal_alias_groups_with_appended_dfas(replacements)?,
            ));
        }
        let coordinates_ms = coordinates_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let futures_started = profile.then(std::time::Instant::now);
        let future_mode = if let Some(start_futures) = start_futures {
            self.dfa.set_possible_future_group_ids(start, start_futures);
            "incremental"
        } else {
            self.dfa.recompute_possible_futures();
            "full"
        };
        let futures_ms = futures_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        self.invalidate_derived_caches();
        if profile {
            eprintln!(
                "[glrmask/profile][direct_mask_install] components={} physical_components={} appended_states={} future_mode={} start_has_incoming={} incoming_scan_ms={:.3} append_ms={:.3} coordinates_ms={:.3} futures_ms={:.3} total_ms={:.3}",
                component_count,
                physical_component_count,
                appended_state_count,
                future_mode,
                start_has_incoming,
                incoming_scan_ms,
                append_ms,
                coordinates_ms,
                futures_ms,
                total_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        Some(())
    }
    pub fn virtual_binary_repeat_intersections_mask_tokenizer(
        &self,
        horizon: usize,
    ) -> Option<(Tokenizer, Vec<VirtualBinaryRepeatIntersectionMaskProjection>)> {
        if self.virtual_repeat_intersections.is_empty() {
            return None;
        }
        let mut dfa = self.materialized_dfa();
        let mut projections = Vec::with_capacity(self.virtual_repeat_intersections.len());
        for runtime in &self.virtual_repeat_intersections {
            let (next_dfa, projection) =
                runtime.build_mask_projection(horizon, dfa, self.num_terminals)?;
            dfa = next_dfa;
            projections.push(projection);
        }
        Some((Tokenizer::from_parts(dfa, self.num_terminals, None), projections))
    }

    /// Build the ordinary finite DFA used only while generating one model-token
    /// mask for an arithmetic unit repeat. Commit retains this tokenizer's
    /// exact virtual counter; callers reproject that counter at the beginning
    /// of every mask operation.
    #[doc(hidden)]
    pub fn virtual_unit_repeat_mask_tokenizer(
        &self,
        horizon: usize,
    ) -> Option<(Tokenizer, VirtualZeroMinUnitRepeatMaskProjection)> {
        let runtime = self.virtual_unit_repeat.as_deref()?;
        let projection = VirtualZeroMinUnitRepeatMaskProjection::new(
            runtime.min(),
            runtime.max(),
            horizon,
            runtime.physical_state_count(),
        )?;
        let mut dfa = self.materialized_dfa();
        let physical_state_count = runtime.physical_state_count();
        debug_assert_eq!(dfa.num_states() as u32, physical_state_count);
        let positive_state_count = projection.mask_state_count() - physical_state_count;
        for offset in 0..positive_state_count {
            let state = dfa.add_state();
            debug_assert_eq!(state, physical_state_count + offset);
        }
        let first_positive = projection.first_positive_state()?;
        for byte in runtime.body().iter() {
            dfa.add_transition(runtime.root_state(), byte, first_positive);
        }
        for offset in 0..positive_state_count {
            let state = physical_state_count + offset;
            let exact_consumed = projection.full_consumed_for_exact_mask_state(state);
            let self_looping = projection.deep_lower_state() == Some(state)
                || projection.interior_state() == Some(state);
            if self_looping {
                for byte in runtime.body().iter() {
                    dfa.add_transition(state, byte, state);
                }
            } else if let Some(consumed) = exact_consumed
                && consumed < runtime.max() as u32
            {
                let next_full_state = physical_state_count.checked_add(consumed)?;
                let target = projection.project(next_full_state)?;
                for byte in runtime.body().iter() {
                    dfa.add_transition(state, byte, target);
                }
            }

            let mut finalizers = BitSet::new(self.num_terminals as usize);
            let accepting = projection.interior_state() == Some(state)
                || exact_consumed.is_some_and(|consumed| consumed >= runtime.min() as u32);
            if accepting {
                finalizers.set(runtime.terminal() as usize);
            }
            let mut futures = BitSet::new(self.num_terminals as usize);
            let live = self_looping
                || exact_consumed.is_some_and(|consumed| consumed < runtime.max() as u32);
            if live {
                futures.set(runtime.terminal() as usize);
            }
            dfa.overwrite_state_metadata(state, finalizers, futures);
        }
        Some((Tokenizer::from_parts(dfa, self.num_terminals, None), projection))
    }

    #[doc(hidden)]
    pub fn virtual_zero_min_unit_repeat_mask_tokenizer(
        &self,
        horizon: usize,
    ) -> Option<(Tokenizer, VirtualZeroMinUnitRepeatMaskProjection)> {
        let runtime = self.virtual_unit_repeat.as_deref()?;
        (runtime.min() == 0)
            .then(|| self.virtual_unit_repeat_mask_tokenizer(horizon))
            .flatten()
    }

    fn restore_virtual_unit_repeat_runtime(&mut self) -> Result<(), String> {
        if self.virtual_unit_repeat.is_some() {
            return Ok(());
        }
        let Some(expressions) = self.exprs.as_deref() else {
            return Ok(());
        };
        let descriptors = expressions
            .iter()
            .enumerate()
            .filter_map(|(terminal, expression)| {
                super::compile::virtual_unit_repeat_descriptor(expression)
                    .map(|(body, min, max)| (terminal as TerminalID, body, min, max))
            })
            .collect::<Vec<_>>();
        if descriptors.is_empty() {
            return Ok(());
        }
        // If more than one terminal is eligible for the general exact repeat
        // product lane, a multi-component artifact deliberately represents all
        // of them through the shared product-runtime allocator. Do not restore
        // one arithmetic unit sidecar first and thereby steal its proxy root.
        let product_descriptor_count = expressions
            .iter()
            .filter(|expression| {
                super::compile::virtual_binary_bounded_repeat_intersection_descriptor(expression)
                    .or_else(|| super::compile::virtual_large_bounded_repeat_descriptor(expression))
                    .is_some()
            })
            .count();
        if product_descriptor_count > 1 {
            return Ok(());
        }

        let physical_state_count = self.num_states();
        let mut restored = Vec::<(TerminalID, U8Set, usize, usize, u32)>::new();
        for (terminal, body, min, max) in descriptors {
            // The same one-byte repeat expression may deliberately have used
            // the repeat-product runtime when a hybrid proxy left too little
            // room below the reserved high state bit. Do not misidentify that
            // proxy as an arithmetic-unit runtime during reconstruction; the
            // repeat-product restoration pass below will recover it exactly.
            if !super::compile::virtual_zero_min_unit_repeat_fits_state_ids(
                max,
                physical_state_count,
            ) {
                continue;
            }
            if physical_state_count == 1 && self.num_terminals == 1 && terminal == 0 {
                restored.push((terminal, body, min, max, 0));
                continue;
            }
            let expected_future = {
                let mut bits = BitSet::new(self.num_terminals as usize);
                bits.set(terminal as usize);
                bits
            };
            let candidates = self
                .epsilon_closure_states(&[self.start_state()])
                .into_iter()
                .filter(|&state| state != self.start_state())
                .filter(|&state| !self.state_has_epsilon_transitions(state))
                .filter(|&state| self.transitions_from(state).next().is_none())
                .filter(|&state| self.state_finalizers(state).is_empty())
                .filter(|&state| self.state_futures(state) == &expected_future)
                .collect::<Vec<_>>();
            if let [root] = candidates.as_slice() {
                restored.push((terminal, body, min, max, *root));
            }
        }
        let ([] | [_, _, ..]) = restored.as_slice() else {
            let (terminal, body, min, max, root_state) = restored[0];
            self.virtual_unit_repeat = Some(Arc::new(
                VirtualZeroMinUnitRepeatRuntime::new(
                    body,
                    min,
                    max,
                    terminal,
                    self.num_terminals,
                    physical_state_count,
                    root_state,
                )
                .ok_or_else(|| {
                    "serialized virtual bounded-repeat tokenizer has an invalid physical proxy"
                        .to_owned()
                })?,
            ));
            self.invalidate_derived_caches();
            return Ok(());
        };
        if restored.len() > 1 {
            return Err(
                "serialized tokenizer contains multiple virtual bounded-repeat proxy roots"
                    .to_owned(),
            );
        }
        // No proxy root means these expressions were materialized normally;
        // there is no virtual runtime state to restore.
        Ok(())
    }

    fn restore_virtual_repeat_intersection_runtime(&mut self) -> Result<(), String> {
        if self.virtual_unit_repeat.is_some() || !self.virtual_repeat_intersections.is_empty() {
            return Ok(());
        }
        let Some(expressions) = self.exprs.as_deref() else {
            return Ok(());
        };
        let descriptors = expressions
            .iter()
            .enumerate()
            .filter_map(|(terminal, expression)| {
                super::compile::virtual_binary_bounded_repeat_intersection_descriptor(expression)
                    .or_else(|| super::compile::virtual_large_bounded_repeat_descriptor(expression))
                    .map(|descriptor| (terminal as TerminalID, descriptor))
            })
            .collect::<Vec<_>>();
        if descriptors.is_empty() {
            return Ok(());
        }

        let physical_state_count = self.num_states();
        let mut restored = Vec::<(
            TerminalID,
            VirtualBinaryRepeatIntersectionDescriptor,
            u32,
        )>::new();
        for (terminal, descriptor) in descriptors {
            let expected_future = {
                let mut bits = BitSet::new(self.num_terminals as usize);
                bits.set(terminal as usize);
                bits
            };
            let candidates = self
                .epsilon_closure_states(&[self.start_state()])
                .into_iter()
                .filter(|&state| state != self.start_state())
                .filter(|&state| !self.state_has_epsilon_transitions(state))
                .filter(|&state| self.transitions_from(state).next().is_none())
                .filter(|&state| self.state_finalizers(state).is_empty())
                .filter(|&state| self.state_futures(state) == &expected_future)
                .collect::<Vec<_>>();
            if let [root] = candidates.as_slice() {
                restored.push((terminal, descriptor, *root));
            }
        }
        if restored.is_empty() {
            return Ok(());
        }
        let allocator = Arc::new(VirtualStateAllocator::new(physical_state_count).ok_or_else(|| {
            "serialized virtual repeat-intersection tokenizer has no virtual state-id space"
                .to_owned()
        })?);
        let mut seen_roots = BTreeSet::new();
        for &(_, _, root_state) in &restored {
            if !seen_roots.insert(root_state) {
                return Err(
                    "serialized tokenizer maps multiple virtual repeat components to one proxy root"
                        .to_owned(),
                );
            }
        }
        let roots = restored.iter().map(|(_, _, root)| *root).collect::<Vec<_>>();
        let owners = Arc::new(
            VirtualRuntimeStateOwners::new(physical_state_count, &roots).ok_or_else(|| {
                "serialized virtual repeat-intersection tokenizer has invalid state ownership"
                    .to_owned()
            })?,
        );
        let mut runtimes = Vec::with_capacity(restored.len());
        for (runtime_index, (terminal, descriptor, root_state)) in restored.into_iter().enumerate() {
            runtimes.push(Arc::new(
                VirtualBinaryRepeatIntersectionRuntime::new(
                    descriptor,
                    u32::try_from(runtime_index).map_err(|_| {
                        "serialized virtual repeat-intersection runtime count exceeds u32"
                            .to_owned()
                    })?,
                    terminal,
                    self.num_terminals,
                    physical_state_count,
                    root_state,
                    Arc::clone(&allocator),
                    Arc::clone(&owners),
                )
                .ok_or_else(|| {
                    "serialized virtual repeat-intersection tokenizer has an invalid physical proxy"
                        .to_owned()
                })?,
            ));
        }
        self.virtual_repeat_intersections = runtimes;
        self.invalidate_derived_caches();
        Ok(())
    }

    /// Exact syntactic byte support retained for one terminal.
    ///
    /// This is a cheap necessary condition for terminal-language equality. It
    /// is not itself used as an equivalence proof.
    pub fn terminal_byte_support(&self, terminal: TerminalID) -> Option<U8Set> {
        (terminal < self.num_terminals)
            .then(|| *self.dfa.group_id_to_u8set(terminal))
    }

    #[inline]
    fn state_live_for_terminal(&self, state: u32, terminal: TerminalID) -> bool {
        self.state_finalizers(state).contains(terminal as usize)
            || self.state_futures(state).contains(terminal as usize)
    }

    fn terminal_projected_epsilon_closure(
        &self,
        states: &[u32],
        terminal: TerminalID,
    ) -> Box<[u32]> {
        let mut closure = self.epsilon_closure_states(states);
        closure.retain(|state| self.state_live_for_terminal(*state, terminal));
        closure.sort_unstable();
        closure.dedup();
        closure.into_vec().into_boxed_slice()
    }

    fn terminal_projected_subset_accepting(
        &self,
        states: &[u32],
        terminal: TerminalID,
    ) -> bool {
        states
            .iter()
            .any(|&state| self.state_finalizers(state).contains(terminal as usize))
    }

    /// Return the unique scalar deterministic reset branch containing this
    /// terminal, when the tokenizer's structural certificate proves such a
    /// branch exists. This avoids powerset construction for the common
    /// partitioned-lexer representation.
    fn terminal_dispatch_root_candidate(&self, terminal: TerminalID) -> Option<u32> {
        let start = self.start_state();
        if self.state_finalizers(start).contains(terminal as usize) {
            return None;
        }
        if !self.has_epsilon_transitions() {
            return Some(start);
        }
        // Use the runtime representation of the reset dispatcher. Fast
        // transfer loads may keep epsilon metadata in packed tables rather
        // than in `dfa.states()[start]`; quotient construction must see the
        // same roots as ordinary scanner execution.
        let mut live_roots = self
            .deterministic_dispatch_roots()?
            .iter()
            .copied()
            .filter(|&root| self.state_live_for_terminal(root, terminal));
        let root = live_roots.next()?;
        live_roots.next().is_none().then_some(root)
    }

    fn terminal_scalar_dispatch_root(&self, terminal: TerminalID) -> Option<u32> {
        let root = self.terminal_dispatch_root_candidate(terminal)?;
        self.scalar_physical_component_root(root).then_some(root)
    }

    fn terminal_scalar_dispatch_root_with_states(
        &self,
        terminal: TerminalID,
    ) -> Option<(u32, Vec<u32>)> {
        let root = self.terminal_dispatch_root_candidate(terminal)?;
        let states = self.scalar_physical_component_states(root)?;
        Some((root, states))
    }

    /// Certify scalar execution only for one selected reset component. This is
    /// deliberately local: unrelated virtual/epsilon components under the same
    /// global reset dispatcher must not disable exact single-terminal quotient
    /// construction for an ordinary deterministic component.
    fn scalar_physical_component_root(&self, root: u32) -> bool {
        static GLOBAL_PROOF: OnceLock<bool> = OnceLock::new();
        static ASSERT_PROOF: OnceLock<bool> = OnceLock::new();
        let result = if *GLOBAL_PROOF.get_or_init(|| {
            std::env::var_os("GLRMASK_DISABLE_GLOBAL_SCALAR_PROOF").is_none()
        }) {
            self.scalar_physical_component_root_impl::<true>(root)
        } else {
            self.scalar_physical_component_root_impl::<false>(root)
        };
        if *ASSERT_PROOF.get_or_init(|| {
            std::env::var_os("GLRMASK_ASSERT_GLOBAL_SCALAR_PROOF").is_some()
        }) {
            assert_eq!(result, self.scalar_physical_component_root_impl::<false>(root),
                "global scalar proof differs from exact component traversal at {root}");
        }
        result
    }

    fn scalar_physical_component_root_impl<const GLOBAL_PROOF: bool>(&self, root: u32) -> bool {
        // Valid tokenizer byte transitions stay in their physical DFA unless a
        // virtual runtime owns the destination. With neither virtual runtimes
        // nor epsilon edges anywhere (including packed segments), every root's
        // byte-reachable component is already certified scalar. No component
        // enumeration or allocation is needed solely to re-prove this boolean.
        if GLOBAL_PROOF && root < self.num_states()
            && !self.has_any_virtual_runtime() && !self.has_epsilon_transitions()
        {
            return true;
        }
        self.scalar_physical_component_states(root).is_some()
    }

    /// Discover a complete small physical component, or decline before large
    /// traversal/quotient scratch allocation. Each distinct state is queued at
    /// most once; byte edges are charged before visiting their targets.
    fn scalar_physical_component_states_bounded(
        &self,
        root: u32,
        state_limit: usize,
        transition_limit: usize,
    ) -> Option<Vec<u32>> {
        if state_limit == 0 { return None; }
        let mut seen = FxHashSet::<u32>::default();
        let mut pending = vec![root];
        seen.insert(root);
        let mut transition_work = 0usize;
        while let Some(state) = pending.pop() {
            if state >= self.num_states()
                || self.state_is_virtual_runtime(state)
                || self.state_has_epsilon_transitions(state)
            { return None; }
            for (_, target) in self.transitions_from(state) {
                if transition_work >= transition_limit { return None; }
                transition_work += 1;
                if !seen.contains(&target) {
                    if seen.len() >= state_limit { return None; }
                    seen.insert(target);
                    pending.push(target);
                }
            }
        }
        let mut states = seen.into_iter().collect::<Vec<_>>();
        states.sort_unstable();
        Some(states)
    }

    /// An optional acceleration only: None means use the normal exact walker.
    /// Importantly there is NO expression-compiler or retained-coordinate
    /// fallback here: those would defeat the cold-construction ceiling.
    /// Accepted components use the unchanged exact quotient constructor.
    #[doc(hidden)]
    pub fn build_terminal_projected_quotient_for_containment_bounded(
        &self,
        terminal: TerminalID,
        state_limit: usize,
        transition_limit: usize,
    ) -> Option<TerminalProjectedQuotient> {
        // The existing constructor creates a source-coordinate map. Bound
        // this independent allocation too, even for a tiny selected component.
        const MAX_SOURCE_MAP_STATES: u32 = 1 << 20;
        if terminal >= self.num_terminals
            || state_limit == 0
            || self.num_states() > MAX_SOURCE_MAP_STATES
        { return None; }
        let root = self.terminal_dispatch_root_candidate(terminal)?;
        let states = self.scalar_physical_component_states_bounded(
            root, state_limit, transition_limit,
        )?;
        self.terminal_live_subautomaton_quotient_from_component(terminal, &states)
    }

    fn scalar_physical_component_states(&self, root: u32) -> Option<Vec<u32>> {
        let mut seen = FxHashSet::<u32>::default();
        let mut pending = vec![root];
        while let Some(state) = pending.pop() {
            if !seen.insert(state) {
                continue;
            }
            if state >= self.num_states()
                || self.virtual_residual_runtime_for_state(state).is_some()
                || self.virtual_repeat_runtime_for_state(state).is_some()
                || self
                    .virtual_unit_repeat
                    .as_deref()
                    .is_some_and(|runtime| runtime.handles_state(state))
                || self.state_has_epsilon_transitions(state)
            {
                return None;
            }
            pending.extend(self.transitions_from(state).map(|(_, target)| target));
        }
        let mut states = seen.into_iter().collect::<Vec<_>>();
        states.sort_unstable();
        Some(states)
    }


    /// Exact source-state certificate for containment of `bytes+` in one
    /// scalar physical terminal residual. The result covers every state in the
    /// terminal's physical component; `true` means every nonempty byte string
    /// over `bytes` remains terminal-live from that source.
    #[doc(hidden)]
    pub fn terminal_byte_plus_containment_rows(
        &self,
        terminal: TerminalID,
        bytes: U8Set,
    ) -> Option<Vec<(u32, bool)>> {
        if bytes.is_empty() {
            return None;
        }
        let root = self.terminal_scalar_dispatch_root(terminal)?;
        let states = self.scalar_physical_component_states(root)?;
        let mut index = vec![u32::MAX; self.num_states() as usize];
        for (local, &state) in states.iter().enumerate() {
            index[state as usize] = local as u32;
        }

        let mut bad = vec![false; states.len()];
        let mut reverse = (0..states.len()).map(|_| Vec::<u32>::new()).collect::<Vec<_>>();
        for (local, &state) in states.iter().enumerate() {
            if !self.state_live_for_terminal(state, terminal) {
                bad[local] = true;
                continue;
            }
            let mut required_seen = 0usize;
            for (byte, target) in self.transitions_from(state) {
                if !bytes.contains(byte) {
                    continue;
                }
                required_seen += 1;
                if !self.state_live_for_terminal(target, terminal) {
                    bad[local] = true;
                    continue;
                }
                let target_local = *index.get(target as usize)?;
                if target_local == u32::MAX {
                    return None;
                }
                reverse[target_local as usize].push(local as u32);
            }
            if required_seen != bytes.len() {
                bad[local] = true;
            }
        }

        let mut queue = VecDeque::<u32>::new();
        for (local, &is_bad) in bad.iter().enumerate() {
            if is_bad {
                queue.push_back(local as u32);
            }
        }
        while let Some(target) = queue.pop_front() {
            for &predecessor in &reverse[target as usize] {
                let predecessor = predecessor as usize;
                if !bad[predecessor] {
                    bad[predecessor] = true;
                    queue.push_back(predecessor as u32);
                }
            }
        }

        Some(
            states
                .into_iter()
                .zip(bad.into_iter().map(|is_bad| !is_bad))
                .collect(),
        )
    }

    /// Necessary-condition test for regex-slice containment: does any exact
    /// physical source state of `terminal` keep the terminal live after every
    /// byte in `bytes`? If false, a slice accepting all of those one-byte
    /// strings cannot be contained at any source in this terminal component.
    #[doc(hidden)]
    pub fn terminal_has_source_covering_bytes(
        &self,
        terminal: TerminalID,
        bytes: U8Set,
    ) -> bool {
        if bytes.is_empty() {
            return true;
        }
        let Some(states) = self.scalar_physical_component_states_for_terminal(terminal) else {
            return false;
        };
        states.into_iter().any(|state| {
            self.state_live_for_terminal(state, terminal)
                && bytes.iter().all(|byte| {
                    self.step(state, byte)
                        .is_some_and(|target| self.state_live_for_terminal(target, terminal))
                })
        })
    }

    #[doc(hidden)]
    pub fn scalar_physical_component_states_for_terminal(
        &self,
        terminal: TerminalID,
    ) -> Option<Vec<u32>> {
        let root = self.terminal_scalar_dispatch_root(terminal)?;
        self.scalar_physical_component_states(root)
    }

    #[doc(hidden)]
    #[inline]
    pub fn terminal_state_is_live(&self, state: u32, terminal: TerminalID) -> bool {
        self.state_live_for_terminal(state, terminal)
    }

    /// Cheap exact necessary condition for language containment at one
    /// *physical* residual state. `Some(false)` proves that a candidate
    /// language containing every byte in `bytes` as a possible first byte
    /// cannot be contained in this terminal residual. Virtual/epsilon states
    /// return `None` so callers conservatively fall back to the full proof.
    #[doc(hidden)]
    pub fn physical_terminal_residual_covers_first_bytes(
        &self,
        state: u32,
        terminal: TerminalID,
        bytes: U8Set,
    ) -> Option<bool> {
        if state >= self.num_states()
            || terminal >= self.num_terminals
            || self.state_is_virtual_runtime(state)
            || self.state_has_epsilon_transitions(state)
            || !self.state_live_for_terminal(state, terminal)
        {
            return None;
        }
        let mut missing = bytes;
        for (byte, target) in self.transitions_from(state) {
            if missing.contains(byte) && self.state_live_for_terminal(target, terminal) {
                missing.remove(byte);
                if missing.is_empty() {
                    return Some(true);
                }
            }
        }
        Some(missing.is_empty())
    }

    #[inline]
    fn terminal_projected_scalar_step(
        &self,
        state: u32,
        terminal: TerminalID,
        byte: u8,
    ) -> Option<u32> {
        self.step(state, byte)
            .filter(|&target| self.state_live_for_terminal(target, terminal))
    }

    fn terminal_scalar_prefix_fingerprint_from_state(
        &self,
        state: u32,
        terminal: TerminalID,
        depth: u8,
        memo: &mut FxHashMap<(u32, u8), u64>,
    ) -> u64 {
        if let Some(&fingerprint) = memo.get(&(state, depth)) {
            return fingerprint;
        }
        // Fixed deterministic mixer. Hash collisions merely admit an extra
        // exact equivalence check; this fingerprint is never a proof.
        #[inline]
        fn mix(mut hash: u64, value: u64) -> u64 {
            hash ^= value.wrapping_add(0x9e37_79b9_7f4a_7c15);
            hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            hash ^ (hash >> 29)
        }

        let accepting = self.state_finalizers(state).contains(terminal as usize);
        let mut hash = if accepting {
            0x6a09_e667_f3bc_c909
        } else {
            0xbb67_ae85_84ca_a73b
        };
        if depth > 0 {
            let mut transitions = self
                .transitions_from(state)
                .filter_map(|(byte, target)| {
                    self.state_live_for_terminal(target, terminal)
                        .then_some((byte, target))
                })
                .collect::<Vec<_>>();
            transitions.sort_unstable_by_key(|&(byte, _)| byte);
            for (byte, target) in transitions {
                let child = self.terminal_scalar_prefix_fingerprint_from_state(
                    target,
                    terminal,
                    depth - 1,
                    memo,
                );
                hash = mix(hash, byte as u64 + 1);
                hash = mix(hash, child);
            }
            hash = mix(hash, 0x100 + depth as u64);
        }
        memo.insert((state, depth), hash);
        hash
    }

    /// Build the same exact single-terminal residual coordinate directly from
    /// the retained terminal expression, then prove a homomorphism from this
    /// tokenizer's terminal projection into that standalone DFA.
    ///
    /// This avoids minimizing the much larger combined tokenizer component.
    /// The mapping is accepted only when every reachable projected byte edge
    /// and accepting observation agrees exactly in the two automata.
    fn terminal_expr_projected_quotient(
        &self,
        source: u32,
        terminal: TerminalID,
    ) -> Option<TerminalProjectedQuotient> {
        self.terminal_expr_projected_quotient_with_byte_classes(source, terminal, None)
    }

    fn terminal_expr_projected_quotient_with_byte_classes(
        &self,
        source: u32,
        terminal: TerminalID,
        _full_byte_classes: Option<&[u8; 256]>,
    ) -> Option<TerminalProjectedQuotient> {
        if source >= self.num_states()
            || terminal >= self.num_terminals
            || self.state_has_epsilon_transitions(source)
        {
            return None;
        }
        let expr = self.terminal_expr(terminal)?.clone();
        let projected = expr.build().into_tokenizer(1, None);
        if projected.has_epsilon_transitions() || projected.has_any_virtual_runtime() {
            return None;
        }

        // Exact sparse homomorphism proof. The selected full component and the
        // standalone terminal DFA are deterministic and epsilon-free. Their
        // terminal languages agree at a state pair iff accepting/future
        // observations agree and the complete sets of *live* outgoing
        // byte-labelled edges agree. Comparing sparse rows therefore proves all
        // 256 byte observations at once without probing dead bytes individually.
        let full_root = self.terminal_scalar_dispatch_root(terminal)?;
        let projected_root = projected.initial_state();
        let mut full_to_projected = vec![u32::MAX; self.num_states() as usize];
        let mut queue = VecDeque::from([(full_root, projected_root)]);
        full_to_projected[full_root as usize] = projected_root;
        while let Some((full_state, projected_state)) = queue.pop_front() {
            let full_accepting = self
                .matched_terminal_bitset(full_state)
                .contains(terminal as usize);
            let projected_accepting = projected.matched_terminal_bitset(projected_state).contains(0);
            if full_accepting != projected_accepting {
                return None;
            }
            let full_future = self
                .possible_future_terminals(full_state)
                .contains(terminal as usize);
            let projected_future = projected.possible_future_terminals(projected_state).contains(0);
            if full_future != projected_future {
                return None;
            }

            let mut full_edges = self
                .transitions_from(full_state)
                .filter(|&(_, target)| self.state_live_for_terminal(target, terminal))
                .collect::<Vec<_>>();
            let mut projected_edges = projected
                .transitions_from(projected_state)
                .filter(|&(_, target)| projected.state_live_for_terminal(target, 0))
                .collect::<Vec<_>>();
            full_edges.sort_unstable_by_key(|&(byte, _)| byte);
            projected_edges.sort_unstable_by_key(|&(byte, _)| byte);
            if full_edges.len() != projected_edges.len() {
                return None;
            }
            for ((full_byte, full_target), (projected_byte, projected_target)) in
                full_edges.into_iter().zip(projected_edges)
            {
                if full_byte != projected_byte {
                    return None;
                }
                let slot = &mut full_to_projected[full_target as usize];
                if *slot == u32::MAX {
                    *slot = projected_target;
                    queue.push_back((full_target, projected_target));
                } else if *slot != projected_target {
                    return None;
                }
            }
        }

        if full_to_projected[source as usize] == u32::MAX {
            return None;
        }
        let quotient = TerminalProjectedQuotient::from_dense_mapping(
            projected.dfa,
            full_to_projected,
        );
        Some(quotient)
    }

    fn deterministic_component_byte_classes(&self, states: &[u32]) -> [u8; 256] {
        let mut classes = [0u8; 256];
        let mut class_count = 1usize;
        let mut targets = [u32::MAX; 256];
        let mut keyed = Vec::<(u64, u8)>::with_capacity(256);
        let mut refined = [0u8; 256];
        for &state in states {
            if class_count == 256 {
                break;
            }
            targets.fill(u32::MAX);
            for (byte, target) in self.transitions_from(state) {
                targets[byte as usize] = target;
            }
            keyed.clear();
            for byte in 0u16..=255 {
                let byte = byte as u8;
                let target_code = targets[byte as usize] as u64 + 1;
                let key = ((classes[byte as usize] as u64) << 33) | target_code;
                keyed.push((key, byte));
            }
            keyed.sort_unstable_by_key(|&(key, _)| key);
            let mut previous = None::<u64>;
            let mut next_class = 0usize;
            for &(key, byte) in &keyed {
                if previous != Some(key) {
                    previous = Some(key);
                    next_class += 1;
                }
                refined[byte as usize] = (next_class - 1) as u8;
            }
            classes = refined;
            class_count = next_class;
        }
        TerminalProjectedQuotient::canonicalize_byte_class_map(&classes).0
    }

    /// Extract the exact effective single-terminal residual automaton directly
    /// from one certified scalar physical tokenizer component. This preserves
    /// lexer priority/exclusion semantics by construction and avoids rebuilding
    /// the retained source expression plus a separate homomorphism proof.
    fn terminal_live_subautomaton_quotient_from_component(
        &self,
        terminal: TerminalID,
        component_states: &[u32],
    ) -> Option<TerminalProjectedQuotient> {
        let quotient = TerminalProjectedQuotient::from_source_terminal_subautomaton(
            self,
            terminal,
            component_states,
        )?;
        Some(quotient)
    }

    fn terminal_live_subautomaton_quotient(
        &self,
        terminal: TerminalID,
        _base_byte_classes: Option<&[u8; 256]>,
    ) -> Option<TerminalProjectedQuotient> {
        let (_root, component_states) =
            self.terminal_scalar_dispatch_root_with_states(terminal)?;
        self.terminal_live_subautomaton_quotient_from_component(terminal, &component_states)
    }

    /// Build the exact expression-projected quotient from this terminal's
    /// deterministic dispatch root.  Kept as a diagnostic/compile-time helper
    /// so callers do not need access to the raw dispatch-root coordinate.
    fn terminal_expr_projected_quotient_from_root(
        &self,
        terminal: TerminalID,
    ) -> Option<TerminalProjectedQuotient> {
        if let Some(quotient) = self.terminal_live_subautomaton_quotient(terminal, None) {
            return Some(quotient);
        }
        if let Some(quotient) = self.terminal_projected_quotient_from_retained_coordinates(terminal)
        {
            return Some(quotient);
        }
        let source = self.terminal_scalar_dispatch_root(terminal)?;
        self.terminal_expr_projected_quotient(source, terminal)
    }

    /// Reuse the exact product coordinates retained by the partitioned lexer
    /// builder instead of reconstructing and re-proving the same terminal
    /// homomorphism. Each row records `(terminal, standalone_terminal_state)`
    /// for one combined tokenizer state. States appended later for unrelated
    /// virtual runtimes simply remain unmapped and conservatively fall back.
    fn terminal_projected_quotient_from_retained_coordinates(
        &self,
        terminal: TerminalID,
    ) -> Option<TerminalProjectedQuotient> {
        let coordinates = self.terminal_residual_coordinates.as_deref()?;
        let terminal_dfa = coordinates.terminal_dfa(terminal)?.clone();
        if terminal_dfa.has_epsilon_transitions() {
            return None;
        }

        let mut full_to_projected = vec![u32::MAX; self.num_states() as usize];
        let mut mapped = 0usize;
        let physical_rows = coordinates.len().min(full_to_projected.len());
        for state in 0..physical_rows {
            let row = coordinates.row(state as u32)?;
            let Ok(index) = row.binary_search_by_key(&terminal, |&(candidate, _)| candidate)
            else {
                continue;
            };
            let projected = row[index].1;
            if projected >= terminal_dfa.num_states() as u32 {
                return None;
            }
            let live = terminal_dfa.finalizers(projected).contains(0)
                || terminal_dfa.possible_future_group_ids(projected).contains(0);
            if live {
                full_to_projected[state] = projected;
                mapped += 1;
            }
        }
        (mapped != 0).then(|| {
            TerminalProjectedQuotient::from_dense_mapping(terminal_dfa, full_to_projected)
        })
    }

    /// Build exact single-terminal residual coordinates only for terminals in
    /// sufficiently large shared deterministic dispatch components.
    ///
    /// A single-terminal component cannot contain residual distinctions caused
    /// by other terminals, so projecting it buys nothing. Every returned
    /// quotient is certified by the exact expression-projected homomorphism.
    pub fn build_shared_component_terminal_projected_quotients(
        &self,
        min_component_states: usize,
    ) -> Vec<(TerminalID, TerminalProjectedQuotient)> {
        if min_component_states == 0 {
            return Vec::new();
        }
        let Some(roots) = self.deterministic_dispatch_roots() else {
            return Vec::new();
        };
        let Some(components) = self.disjoint_dispatch_components() else {
            return Vec::new();
        };
        if roots.len() != components.len() {
            return Vec::new();
        }

        let mut candidates = Vec::<(TerminalID, Arc<[u8; 256]>)>::new();
        for (&root, states) in roots.iter().zip(&components) {
            if states.len() < min_component_states {
                continue;
            }
            let terminals = (0..self.num_terminals)
                .filter(|&terminal| self.terminal_scalar_dispatch_root(terminal) == Some(root))
                .collect::<Vec<_>>();
            if terminals.len() < 2 {
                continue;
            }
            let byte_classes = Arc::new(self.deterministic_component_byte_classes(states));
            candidates.extend(
                terminals
                    .into_iter()
                    .map(|terminal| (terminal, Arc::clone(&byte_classes))),
            );
        }
        candidates.sort_unstable_by_key(|(terminal, _)| *terminal);
        candidates.dedup_by_key(|(terminal, _)| *terminal);

        let profile =
            std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROJECTED_QUOTIENT_BUILD").is_some();
        let results = candidates
            .into_par_iter()
            .filter_map(|(terminal, byte_classes)| {
                let source = self.terminal_scalar_dispatch_root(terminal)?;
                let quotient = self.terminal_expr_projected_quotient_with_byte_classes(
                    source,
                    terminal,
                    Some(byte_classes.as_ref()),
                )?;
                let (mapped, projected) = quotient.state_counts();
                if profile {
                    eprintln!(
                        "[glrmask/profile][dynamic_projected_quotient_build] terminal={} mapped={} projected={} retained={}",
                        terminal,
                        mapped,
                        projected,
                        projected < mapped,
                    );
                }
                (projected < mapped).then_some((terminal, quotient))
            })
            .collect::<Vec<_>>();
        if profile {
            eprintln!(
                "[glrmask/profile][dynamic_projected_quotient_build_summary] retained={}",
                results.len(),
            );
        }
        results
    }

    /// Build every exact scalar single-terminal quotient needed by runtime
    /// regular-language containment. Unlike the older projection optimization,
    /// this deliberately retains quotients even when they do not reduce the
    /// underlying component state count: containment needs an exact residual
    /// coordinate, not compression.
    pub fn build_terminal_projected_quotients_for_containment(
        &self,
    ) -> Vec<(TerminalID, TerminalProjectedQuotient)> {
        let terminals = (0..self.num_terminals).collect::<Vec<_>>();
        self.build_terminal_projected_quotients_for_containment_candidates(&terminals)
    }

    /// Build exact single-terminal quotients only for caller-selected
    /// containment candidates. Candidate selection is allowed to be incomplete
    /// only in the conservative direction: omitted terminals simply lose the
    /// acceleration and fall back to the exact vocabulary walk.
    pub fn build_terminal_projected_quotients_for_containment_candidates(
        &self,
        terminals: &[TerminalID],
    ) -> Vec<(TerminalID, TerminalProjectedQuotient)> {
        self.build_containment_candidates_with_limits(terminals, None)
    }

    /// Prepare independent exact proofs in parallel, after a bounded shared
    /// component preflight. An omitted component has no certificate and must
    /// use the ordinary exact walk; a resource limit is never an acceptance.
    /// The bounded path deliberately declines unbounded expression fallback.
    pub fn build_terminal_projected_quotients_for_containment_candidates_bounded(
        &self,
        terminals: &[TerminalID],
        max_states: usize,
        max_edges: usize,
    ) -> Vec<(TerminalID, TerminalProjectedQuotient)> {
        self.build_containment_candidates_with_limits(terminals, Some((max_states, max_edges)))
    }

    fn build_containment_candidates_with_limits(
        &self,
        terminals: &[TerminalID],
        limits: Option<(usize, usize)>,
    ) -> Vec<(TerminalID, TerminalProjectedQuotient)> {
        let profile =
            std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROJECTED_QUOTIENT_BUILD").is_some();
        let mut candidates = terminals
            .iter()
            .copied()
            .filter(|&terminal| terminal < self.num_terminals)
            .collect::<Vec<_>>();
        candidates.sort_unstable();
        candidates.dedup();

        let mut by_root = BTreeMap::<u32, Vec<TerminalID>>::new();
        let mut fallback = Vec::<TerminalID>::new();
        for terminal in candidates {
            if let Some(root) = self.terminal_dispatch_root_candidate(terminal) {
                by_root.entry(root).or_default().push(terminal);
            } else {
                fallback.push(terminal);
            }
        }

        // Certify each shared physical component once. Unlike the rejected
        // shared-byte-alphabet experiment, this only reuses the state list; each
        // terminal still derives its own exact byte partition independently.
        let component_jobs = by_root
            .into_par_iter()
            .filter_map(|(root, terminals)| {
                let states = match limits {
                    Some((states, edges)) => self.scalar_physical_component_states_bounded(root, states, edges),
                    None => self.scalar_physical_component_states(root),
                }?;
                Some((terminals, Arc::new(states)))
            })
            .collect::<Vec<_>>();
        let mut jobs = Vec::<(TerminalID, Arc<Vec<u32>>)>::new();
        for (terminals, states) in component_jobs {
            jobs.extend(
                terminals
                    .into_iter()
                    .map(|terminal| (terminal, Arc::clone(&states))),
            );
        }

        let mut results = jobs
            .into_par_iter()
            .filter_map(|(terminal, states)| {
                let quotient = self
                    .terminal_live_subautomaton_quotient_from_component(terminal, states.as_ref())?;
                if profile {
                    let (mapped, projected) = quotient.state_counts();
                    eprintln!(
                        "[glrmask/profile][containment_quotient_build] terminal={} mapped={} projected={}",
                        terminal, mapped, projected,
                    );
                }
                Some((terminal, quotient))
            })
            .collect::<Vec<_>>();

        if limits.is_none() {
            results.extend(
                fallback
                    .into_par_iter()
                    .filter_map(|terminal| {
                        let quotient = self.terminal_expr_projected_quotient_from_root(terminal)?;
                        Some((terminal, quotient))
                    })
                    .collect::<Vec<_>>(),
            );
        }
        results.sort_unstable_by_key(|(terminal, _)| *terminal);
        if profile {
            eprintln!(
                "[glrmask/profile][containment_quotient_build_summary] retained={}",
                results.len(),
            );
        }
        results
    }

    /// A bounded language-observation fingerprint for candidate indexing.
    ///
    /// For scalar deterministic terminal branches this recursively records the
    /// accepting bit and every live byte derivative to `depth`. Equal terminal
    /// languages necessarily produce the same value. Hash collisions or a
    /// shallow horizon only create extra exact checks; equality of the value is
    /// never used as an equivalence certificate.
    pub fn terminal_scalar_prefix_fingerprint(
        &self,
        terminal: TerminalID,
        depth: u8,
    ) -> Option<u64> {
        let root = self.terminal_scalar_dispatch_root(terminal)?;
        let mut memo = FxHashMap::default();
        Some(self.terminal_scalar_prefix_fingerprint_from_state(
            root,
            terminal,
            depth,
            &mut memo,
        ))
    }

    /// Canonical rooted-graph certificate for one scalar terminal DFA.
    ///
    /// States are renumbered by deterministic BFS from the terminal's unique
    /// dispatch root, following live transitions in byte order. The returned
    /// sparse serialization records each state's accepting bit and every
    /// byte-labelled edge to the canonical target ID. Equality of two returned
    /// vectors is therefore an exact rooted labelled-DFA isomorphism proof and
    /// implies terminal-language equality. Non-isomorphic but language-
    /// equivalent DFAs may produce different certificates; this is an
    /// intentionally sufficient, not complete, proof.
    pub fn terminal_scalar_structural_certificate(
        &self,
        terminal: TerminalID,
        state_limit: usize,
        transition_limit: usize,
    ) -> Option<Vec<u64>> {
        if state_limit == 0 || transition_limit == 0 {
            return None;
        }
        let root = self.terminal_scalar_dispatch_root(terminal)?;
        let mut canonical = FxHashMap::<u32, u32>::default();
        canonical.insert(root, 0);
        let mut queue = VecDeque::from([root]);
        let mut encoded = Vec::<u64>::new();
        let mut transition_count = 0usize;

        while let Some(state) = queue.pop_front() {
            if canonical.len() > state_limit {
                return None;
            }
            let accepting = self.state_finalizers(state).contains(terminal as usize);
            let mut transitions = self
                .transitions_from(state)
                .filter_map(|(byte, target)| {
                    self.state_live_for_terminal(target, terminal)
                        .then_some((byte, target))
                })
                .collect::<Vec<_>>();
            transitions.sort_unstable_by_key(|&(byte, _)| byte);
            transition_count = transition_count.saturating_add(transitions.len());
            if transition_count > transition_limit {
                return None;
            }
            encoded.push(
                ((accepting as u64) << 63)
                    | u64::try_from(transitions.len()).ok()?.min((1u64 << 63) - 1),
            );
            for (byte, target) in transitions {
                let next_id = if let Some(&existing) = canonical.get(&target) {
                    existing
                } else {
                    let next = canonical.len() as u32;
                    canonical.insert(target, next);
                    queue.push_back(target);
                    next
                };
                encoded.push(((byte as u64) << 32) | next_id as u64);
            }
        }
        Some(encoded)
    }

    fn terminal_scalar_language_equivalent_bounded(
        &self,
        terminal: TerminalID,
        left_root: u32,
        other: &Tokenizer,
        other_terminal: TerminalID,
        right_root: u32,
        pair_limit: usize,
        transition_work_limit: usize,
    ) -> Option<bool> {
        let mut seen = rustc_hash::FxHashSet::<(Option<u32>, Option<u32>)>::default();
        let mut queue = VecDeque::from([(Some(left_root), Some(right_root))]);
        let mut work = 0usize;

        while let Some(pair @ (left, right)) = queue.pop_front() {
            if !seen.insert(pair) {
                continue;
            }
            if seen.len() > pair_limit {
                return None;
            }
            let left_accepting =
                left.is_some_and(|state| self.state_finalizers(state).contains(terminal as usize));
            let right_accepting = right.is_some_and(|state| {
                other.state_finalizers(state).contains(other_terminal as usize)
            });
            if left_accepting != right_accepting {
                return Some(false);
            }

            let mut bytes = U8Set::empty();
            if let Some(state) = left {
                for (byte, _) in self.transitions_from(state) {
                    bytes.insert(byte);
                }
            }
            if let Some(state) = right {
                for (byte, _) in other.transitions_from(state) {
                    bytes.insert(byte);
                }
            }
            for byte in bytes.iter() {
                work = work.saturating_add(1);
                if work > transition_work_limit {
                    return None;
                }
                let next = (
                    left.and_then(|state| {
                        self.terminal_projected_scalar_step(state, terminal, byte)
                    }),
                    right.and_then(|state| {
                        other.terminal_projected_scalar_step(state, other_terminal, byte)
                    }),
                );
                if next != (None, None) && !seen.contains(&next) {
                    queue.push_back(next);
                }
            }
        }
        Some(true)
    }

    fn terminal_projected_step(
        &self,
        states: &[u32],
        terminal: TerminalID,
        byte: u8,
        work: &mut usize,
        work_limit: usize,
    ) -> Option<Box<[u32]>> {
        let mut targets = SmallVec::<[u32; 8]>::new();
        for &state in states {
            *work = work.saturating_add(1);
            if *work > work_limit {
                return None;
            }
            if let Some(target) = self.step(state, byte) {
                targets.push(target);
            }
        }
        if targets.is_empty() {
            return Some(Box::new([]));
        }
        targets.sort_unstable();
        targets.dedup();
        Some(self.terminal_projected_epsilon_closure(&targets, terminal))
    }

    /// Prove whether two compiled terminals denote the same byte language.
    ///
    /// This works directly from the serialized tokenizer automata and does not
    /// require the compile-time `Expr` sidecar. The proof is exact: each
    /// epsilon-NFA is projected to the chosen terminal, then the symmetric
    /// difference of their on-the-fly subset constructions is searched by BFS.
    /// `Some(true)` means equivalence was proved; `Some(false)` means a concrete
    /// distinguishing byte word exists. `None` means the supplied resource
    /// bounds were exhausted, in which case callers must conservatively decline
    /// any optimization that requires equivalence.
    ///
    /// `possible_future_group_ids` is used only to remove states from which the
    /// selected terminal can neither finalize now nor in the future. By its
    /// tokenizer invariant such states cannot contribute to the selected
    /// terminal's language.
    pub fn terminal_language_equivalent_bounded(
        &self,
        terminal: TerminalID,
        other: &Tokenizer,
        other_terminal: TerminalID,
        pair_limit: usize,
        transition_work_limit: usize,
    ) -> Option<bool> {
        if terminal >= self.num_terminals
            || other_terminal >= other.num_terminals
            || pair_limit == 0
            || transition_work_limit == 0
        {
            return None;
        }

        // Equal languages necessarily consume the same set of byte values.
        // This metadata is a cheap rejection filter only; equality here is not
        // treated as a proof.
        if self.terminal_byte_support(terminal)? != other.terminal_byte_support(other_terminal)? {
            return Some(false);
        }

        if let (Some(left_root), Some(right_root)) = (
            self.terminal_scalar_dispatch_root(terminal),
            other.terminal_scalar_dispatch_root(other_terminal),
        ) {
            return self.terminal_scalar_language_equivalent_bounded(
                terminal,
                left_root,
                other,
                other_terminal,
                right_root,
                pair_limit,
                transition_work_limit,
            );
        }

        let left_start =
            self.terminal_projected_epsilon_closure(&[self.start_state()], terminal);
        let right_start = other
            .terminal_projected_epsilon_closure(&[other.start_state()], other_terminal);

        type SubsetPair = (Box<[u32]>, Box<[u32]>);
        let mut seen = rustc_hash::FxHashSet::<SubsetPair>::default();
        let mut queue = VecDeque::<SubsetPair>::from([(left_start, right_start)]);
        let mut work = 0usize;

        while let Some((left, right)) = queue.pop_front() {
            if !seen.insert((left.clone(), right.clone())) {
                continue;
            }
            if seen.len() > pair_limit {
                return None;
            }
            if self.terminal_projected_subset_accepting(&left, terminal)
                != other.terminal_projected_subset_accepting(&right, other_terminal)
            {
                return Some(false);
            }

            // Explore exactly the bytes that have an outgoing transition from
            // either current subset. This is equivalent to scanning all 256
            // bytes while avoiding dead/dead product edges.
            let mut bytes = U8Set::empty();
            for &state in left.iter() {
                for (byte, _) in self.transitions_from(state) {
                    bytes.insert(byte);
                }
            }
            for &state in right.iter() {
                for (byte, _) in other.transitions_from(state) {
                    bytes.insert(byte);
                }
            }

            for byte in bytes.iter() {
                let next_left = self.terminal_projected_step(
                    &left,
                    terminal,
                    byte,
                    &mut work,
                    transition_work_limit,
                )?;
                let next_right = other.terminal_projected_step(
                    &right,
                    other_terminal,
                    byte,
                    &mut work,
                    transition_work_limit,
                )?;
                if next_left.is_empty() && next_right.is_empty() {
                    continue;
                }
                if !seen.contains(&(next_left.clone(), next_right.clone())) {
                    queue.push_back((next_left, next_right));
                }
            }
        }
        Some(true)
    }

    pub fn initial_epsilon_branch_count(&self) -> usize {
        self.dfa
            .states()
            .get(self.start_state() as usize)
            .map_or(0, |state| state.epsilon_transitions.len())
    }

    /// Return the deterministic scanner roots behind the special epsilon
    /// dispatch state produced by `build_regex_partitioned`.
    ///
    /// This is deliberately narrower than "has epsilon transitions".  The
    /// compiler can retain its scalar-state fast paths when the only live
    /// nondeterminism is a zero-byte fan-out from the global reset state into
    /// independently deterministic components.  Nullable-start isolation may
    /// leave an unreachable cloned dispatch state elsewhere in the DFA, so the
    /// predicate is based on the live reset shape rather than a whole-DFA scan.
    pub fn deterministic_dispatch_roots(&self) -> Option<&[u32]> {
        let start_state = self.start_state();
        // Fast artifact decoding can move epsilon metadata out of the owned
        // DFA states into the packed runtime table. Structural runtime queries
        // must therefore consult the same representation used by scanning,
        // rather than assuming `dfa.states()[start].epsilon_transitions` is
        // populated after load.
        let roots = if let Some(metadata) = self
            .packed_runtime_metadata
            .as_deref()
            .filter(|metadata| start_state < metadata.state_count)
        {
            metadata.epsilon_targets(start_state)
        } else if let Some(segment) = self.packed_runtime_metadata_segment_for_state(start_state) {
            // The global reset belongs to the zero-offset segment when packed
            // metadata is segmented; local and global target IDs then coincide.
            if segment.state_offset != 0 {
                return None;
            }
            segment.metadata.epsilon_targets(segment.local_state(start_state))
        } else {
            &self.dfa.states().get(start_state as usize)?.epsilon_transitions
        };
        if roots.len() < 2 || self.transitions_from(start_state).next().is_some() {
            return None;
        }
        if roots
            .iter()
            .any(|&root| self.state_has_epsilon_transitions(root))
        {
            return None;
        }
        Some(roots)
    }

    #[inline]
    pub fn has_deterministic_dispatch(&self) -> bool {
        self.deterministic_dispatch_roots().is_some()
    }

    /// Sorted/deduplicated reset-dispatch roots, cached. Pure function of the
    /// immutable reset structure; the dynamic scalar-dispatch mask path needs
    /// this order on every mask.
    pub fn sorted_deterministic_dispatch_roots(&self) -> Option<Arc<Vec<u32>>> {
        let roots = self.deterministic_dispatch_roots()?;
        Some(Arc::clone(self.sorted_dispatch_roots_cache.get_or_init(|| {
            let mut sorted = roots.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            Arc::new(sorted)
        })))
    }

    /// First-byte set of one physical tokenizer state (union of
    /// `transitions_from` bytes), cached per state on first touch. Used by the
    /// dynamic pre-collapse gate instead of rescanning transitions per mask.
    /// States with epsilon transitions are included as ordinary rows; callers
    /// that must skip epsilon states keep their own filter.
    pub fn state_first_bytes(&self, state: u32) -> U8Set {
        let table = self.state_first_bytes_cache.get_or_init(|| {
            Arc::from(
                (0..self.num_states())
                    .map(|_| OnceLock::new())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        });
        let Some(cell) = table.get(state as usize) else {
            return U8Set::empty();
        };
        *cell.get_or_init(|| {
            let mut bytes = U8Set::empty();
            for (byte, _) in self.transitions_from(state) {
                bytes.insert(byte);
            }
            bytes
        })
    }

    /// Return whether selecting one reset-dispatch root leaves a genuinely
    /// scalar scanner for the rest of the byte stream.
    ///
    /// `deterministic_dispatch_roots()` only certifies the shape at reset: a
    /// zero-byte fan-out whose immediate roots have no epsilon edges.  A
    /// depth-limited adaptive product can have that same reset shape while
    /// introducing epsilon fan-out later, at its determinization frontier.
    /// Whole-token flat-transition walkers are sound only under the stronger
    /// condition checked here: no state byte-reachable from a dispatch root has
    /// an epsilon transition.
    pub fn has_scalar_deterministic_dispatch(&self) -> bool {
        *self.scalar_deterministic_dispatch_cache.get_or_init(|| {
            let Some(roots) = self.deterministic_dispatch_roots() else {
                return false;
            };
            let mut seen = vec![false; self.num_states() as usize];
            let mut pending = roots.to_vec();
            while let Some(state) = pending.pop() {
                let Some(slot) = seen.get_mut(state as usize) else {
                    return false;
                };
                if *slot {
                    continue;
                }
                *slot = true;
                if self.state_has_epsilon_transitions(state) {
                    return false;
                }
                pending.extend(self.transitions_from(state).map(|(_, target)| target));
            }
            true
        })
    }

    /// Return the closed, pairwise-disjoint state sets below the global
    /// epsilon dispatcher. Components may contain internal epsilon structure,
    /// but no byte or epsilon edge may cross between returned sets.
    pub fn disjoint_dispatch_components(&self) -> Option<Vec<Vec<u32>>> {
        let roots = self.deterministic_dispatch_roots()?;
        let mut owner = vec![usize::MAX; self.dfa.states().len()];
        owner[self.start_state() as usize] = roots.len();
        let mut components = Vec::with_capacity(roots.len());

        for (component_index, &root) in roots.iter().enumerate() {
            if owner.get(root as usize).copied().unwrap_or(roots.len()) != usize::MAX {
                return None;
            }
            let mut states = Vec::new();
            let mut stack = vec![root];
            while let Some(state) = stack.pop() {
                let slot = owner.get_mut(state as usize)?;
                if *slot == component_index {
                    continue;
                }
                if *slot != usize::MAX {
                    return None;
                }
                *slot = component_index;
                states.push(state);
                let dfa_state = self.dfa.states().get(state as usize)?;
                stack.extend(dfa_state.transitions.iter().map(|(_, &target)| target));
                stack.extend(dfa_state.epsilon_transitions.iter().copied());
            }
            if states.is_empty() {
                return None;
            }
            states.sort_unstable();
            components.push(states);
        }
        Some(components)
    }

    /// Scanner states to use after a terminal boundary.  A conventional DFA
    /// has one reset state.  A partitioned lexer has one deterministic reset
    /// state per component; keeping them separate avoids materializing their
    /// product while preserving cross-component terminal sequences inside one
    /// vocabulary token.
    pub fn deterministic_reset_states(&self) -> TokenizerStateSet {
        self.deterministic_dispatch_roots()
            .map(TokenizerStateSet::from_slice)
            .unwrap_or_else(|| TokenizerStateSet::from_buf([self.initial_state_id()]))
    }

    fn transitions_from(&self, state: u32) -> TokenizerTransitionsIter<'_> {
        if let Some(transitions) = self
            .virtual_residual_runtime_for_state(state)
            .and_then(|runtime| runtime.transitions(state))
        {
            return TokenizerTransitionsIter {
                inner: TokenizerTransitionsIterInner::VirtualProduct(transitions.into_iter()),
            };
        }
        if let Some(transitions) = self
            .virtual_repeat_runtime_for_state(state)
            .and_then(|runtime| runtime.transitions(state))
        {
            return TokenizerTransitionsIter {
                inner: TokenizerTransitionsIterInner::VirtualProduct(transitions.into_iter()),
            };
        }
        if let Some(runtime) = self.virtual_unit_repeat.as_deref()
            && let Some(target) = runtime.transition_target(state)
        {
            return TokenizerTransitionsIter {
                inner: TokenizerTransitionsIterInner::Virtual {
                    bytes: runtime.body().iter(),
                    target,
                },
            };
        }
        if let Some(packed) = &self.packed_runtime_transitions {
            if let Some((bytes, targets)) = packed.row(state) {
                return TokenizerTransitionsIter {
                    inner: TokenizerTransitionsIterInner::Packed {
                        bytes,
                        targets,
                        next: 0,
                    },
                };
            }
        }
        if let Some(segment) = self.packed_runtime_transition_segment_for_state(state)
            && let Some((bytes, targets)) = segment.row(state)
        {
            return TokenizerTransitionsIter {
                inner: TokenizerTransitionsIterInner::PackedSegment {
                    bytes,
                    targets,
                    target_offset: segment.state_offset,
                    next: 0,
                },
            };
        }
        if let Some(segment) = self.compressed_segment_for_state(state) {
            return TokenizerTransitionsIter {
                inner: TokenizerTransitionsIterInner::Compressed {
                    segment,
                    state,
                    next_byte: 0,
                },
            };
        }
        if let Some(segment) = self.packed_compressed_segment_for_state(state) {
            return TokenizerTransitionsIter {
                inner: TokenizerTransitionsIterInner::PackedCompressed {
                    segment,
                    state,
                    next_byte: 0,
                },
            };
        }
        TokenizerTransitionsIter {
            inner: self
                .dfa
                .states()
                .get(state as usize)
                .map_or(TokenizerTransitionsIterInner::Empty, |state| {
                    TokenizerTransitionsIterInner::Dense(state.transitions.iter())
                }),
        }
    }

    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) {
        if let Some(segment) = self.compressed_segment_for_state(state) {
            segment.fill_transition_row(state, row);
            return;
        }
        if let Some(segment) = self.packed_compressed_segment_for_state(state) {
            segment.fill_transition_row(state, row);
            return;
        }
        row.fill(u32::MAX);
        for (byte, target) in self.transitions_from(state) {
            row[byte as usize] = target;
        }
    }

    fn transition_row(&self, state: u32) -> Box<[u32; 256]> {
        let mut row = Box::new([u32::MAX; 256]);
        self.fill_transition_row(state, &mut row);
        row
    }

    fn self_loop_bytes(&self, state: u32) -> U8Set {
        if let Some(segment) = self.compressed_segment_for_state(state) {
            let mut bytes = U8Set::empty();
            let local_state = state - segment.state_offset;
            let start = segment.row_offsets[local_state as usize] as usize;
            let end = segment.row_offsets[local_state as usize + 1] as usize;
            for (class, target) in segment.entries.iter_range(start, end) {
                if target == local_state {
                    for &byte in segment.class_members[class as usize].iter() {
                        bytes.insert(byte);
                    }
                }
            }
            return bytes;
        }
        if let Some(segment) = self.packed_compressed_segment_for_state(state) {
            return segment.self_loop_bytes(state);
        }
        let mut bytes = U8Set::empty();
        for (byte, target) in self.transitions_from(state) {
            if target == state {
                bytes.insert(byte);
            }
        }
        bytes
    }

    /// Optimized exact finite-horizon observation certificate for one scalar
    /// tokenizer state. This shadows the generic [`Lexer`] default for the
    /// concrete runtime tokenizer so compressed transition segments can be
    /// consumed in their native byte-class form.
    pub fn bounded_observation_safe_horizon_from_state(
        &self,
        source: u32,
        bytes: U8Set,
        active_terminals: &BitSet,
        max_horizon: u8,
    ) -> u8 {
        self.bounded_observation_safe_horizon_with_witnesses(
            source,
            bytes,
            active_terminals,
            max_horizon,
        )
        .0
    }

    /// Precompute one canonical full-observation-stable byte alphabet per raw
    /// tokenizer state for the two runtime horizons used by dynamic masking.
    ///
    /// `safe_h[state]` is a conservative alphabet B such that every B-string
    /// of length at most H keeps both the complete finalizer set and complete
    /// possible-future-terminal set equal to their values at `state`.
    ///
    /// A single alphabet cannot represent every safe subset exactly (safe
    /// alphabets are not union-closed), so this computes one deterministic
    /// closed alphabet by refinement.  Round 1 retains all observation-
    /// preserving byte transitions.  Each later round intersects that set with
    /// the previous-round safe alphabet of every destination reachable by a
    /// currently retained byte.  The result is therefore sound for arbitrary
    /// mixed byte sequences drawn from the returned set.
    pub fn precompute_bounded_observation_safe_byte_sets(
        &self,
    ) -> (Box<[U8Set]>, Box<[U8Set]>) {
        const DEAD: u32 = u32::MAX;
        let state_count = self.num_states() as usize;
        if state_count == 0 {
            return (Box::new([]), Box::new([]));
        }

        // Literal self-loops are safe for every horizon and remain the fallback
        // when a finite advancing family shrinks away before H.
        let mut horizon16 = (0..state_count)
            .into_par_iter()
            .map(|state| self.self_loop_bytes(state as u32))
            .collect::<Vec<_>>();
        let mut horizon64 = horizon16.clone();

        // Pick one canonical one-byte continuation family at every raw state:
        // the largest set of bytes that all go to the same target while keeping
        // the complete lexer observation unchanged. The target relation is
        // therefore functional for every byte in the selected family.
        let mut selected_sets = vec![U8Set::empty(); state_count];
        let mut selected_targets = vec![DEAD; state_count];
        let mut compressed = vec![false; state_count];
        for segment in self.compressed_transition_segments.iter() {
            let start = segment.state_offset as usize;
            let end = start + segment.state_count as usize;
            compressed[start..end].fill(true);
        }

        // Ordinary states are relatively few. Group their explicit byte edges
        // by raw target directly.
        for state in 0..state_count {
            if compressed[state] || self.state_has_epsilon_transitions(state as u32) {
                continue;
            }
            let mut groups = SmallVec::<[(u32, U8Set); 8]>::new();
            for (byte, target) in self.transitions_from(state as u32) {
                if let Some((_, bytes)) = groups.iter_mut().find(|(seen, _)| *seen == target) {
                    bytes.insert(byte);
                } else {
                    let mut bytes = U8Set::empty();
                    bytes.insert(byte);
                    groups.push((target, bytes));
                }
            }
            let source_finalizers = self.matched_terminal_bitset(state as u32);
            let source_futures = self.possible_future_terminals(state as u32);
            let mut best = U8Set::empty();
            let mut best_target = DEAD;
            for (target, candidate) in groups {
                if self.state_has_epsilon_transitions(target)
                    || self.matched_terminal_bitset(target) != source_finalizers
                    || self.possible_future_terminals(target) != source_futures
                {
                    continue;
                }
                if candidate.len() > best.len()
                    || (candidate.len() == best.len() && target < best_target)
                {
                    best = candidate;
                    best_target = target;
                }
            }
            selected_sets[state] = best;
            selected_targets[state] = best_target;
        }

        // Large synthesized bounded-repeat products use compressed transition
        // segments. Collapse all tokenizer byte classes sharing a raw target
        // without expanding every byte transition for every million-state row.
        for segment in self.compressed_transition_segments.iter() {
            let segment_states = segment.state_count as usize;
            if segment_states == 0 {
                continue;
            }
            let class_sets = segment
                .class_members
                .iter()
                .map(|members| U8Set::from_bytes(members))
                .collect::<Vec<_>>();

            // Scratch indexed by local target avoids one hash table allocation
            // per row. `target_epoch` lazily clears `target_sets`.
            let mut target_sets = vec![U8Set::empty(); segment_states];
            let mut target_epoch = vec![0u32; segment_states];
            let mut epoch = 0u32;
            let mut touched = SmallVec::<[u32; 16]>::new();

            for local_state in 0..segment_states {
                let source_global = segment.state_offset + local_state as u32;
                if self.state_has_epsilon_transitions(source_global) {
                    continue;
                }
                epoch = epoch.wrapping_add(1);
                if epoch == 0 {
                    target_epoch.fill(0);
                    epoch = 1;
                }
                touched.clear();
                let row_start = segment.row_offsets[local_state] as usize;
                let row_end = segment.row_offsets[local_state + 1] as usize;
                for entry in row_start..row_end {
                    let class = segment.entries.classes[entry] as usize;
                    let target = segment.entries.target(entry) as usize;
                    if target_epoch[target] != epoch {
                        target_epoch[target] = epoch;
                        target_sets[target] = U8Set::empty();
                        touched.push(target as u32);
                    }
                    target_sets[target] |= class_sets[class];
                }

                let source_finalizers = self.matched_terminal_bitset(source_global);
                let source_futures = self.possible_future_terminals(source_global);
                let mut best = U8Set::empty();
                let mut best_target = DEAD;
                for &target_local in &touched {
                    let target_global = segment.state_offset + target_local;
                    if self.state_has_epsilon_transitions(target_global)
                        || self.matched_terminal_bitset(target_global) != source_finalizers
                        || self.possible_future_terminals(target_global) != source_futures
                    {
                        continue;
                    }
                    let candidate = target_sets[target_local as usize];
                    if candidate.len() > best.len()
                        || (candidate.len() == best.len() && target_global < best_target)
                    {
                        best = candidate;
                        best_target = target_global;
                    }
                }
                selected_sets[source_global as usize] = best;
                selected_targets[source_global as usize] = best_target;
            }
        }

        // For a chain S --B0--> T --B1--> U ..., any byte alphabet contained
        // in every Bi follows the same raw-state chain and preserves the same
        // full lexer observation at every prefix. Pointer doubling computes the
        // intersection along 2^k edges, so rounds 4 and 6 give exact conservative
        // 16- and 64-byte alphabets for this canonical chain.
        let mut path_sets = selected_sets;
        let mut jump = selected_targets;
        let mut doubled_sets = vec![U8Set::empty(); state_count];
        let mut doubled_jump = vec![DEAD; state_count];

        for round in 1..=6 {
            doubled_sets
                .par_iter_mut()
                .zip(doubled_jump.par_iter_mut())
                .enumerate()
                .for_each(|(state, (set_slot, jump_slot))| {
                    let middle = jump[state];
                    if middle == DEAD {
                        *set_slot = U8Set::empty();
                        *jump_slot = DEAD;
                        return;
                    }
                    let middle = middle as usize;
                    *set_slot = path_sets[state].intersection(&path_sets[middle]);
                    *jump_slot = jump[middle];
                });
            std::mem::swap(&mut path_sets, &mut doubled_sets);
            std::mem::swap(&mut jump, &mut doubled_jump);

            if round == 4 {
                horizon16
                    .par_iter_mut()
                    .zip(path_sets.par_iter())
                    .for_each(|(current, candidate)| {
                        if candidate.len() > current.len() {
                            *current = *candidate;
                        }
                    });
            } else if round == 6 {
                horizon64
                    .par_iter_mut()
                    .zip(path_sets.par_iter())
                    .for_each(|(current, candidate)| {
                        if candidate.len() > current.len() {
                            *current = *candidate;
                        }
                    });
            }
        }

        (horizon16.into_boxed_slice(), horizon64.into_boxed_slice())
    }

    /// The same exact finite-horizon proof as
    /// [`Self::bounded_observation_safe_horizon_from_state`], additionally
    /// returning states reached while the proof remained valid and the depth
    /// at which each was first observed.
    ///
    /// If the returned source horizon is `H`, a witness `(state, d)` with
    /// `d <= H` is itself proved safe for at least `H - d` bytes: every such
    /// continuation is also a continuation of the source of total length at
    /// most `H`, and all observations through depth `H` equal the source
    /// observation. Runtime masking uses these conservative lower bounds to
    /// amortize one bounded-repeat proof across following counter layers.
    pub fn bounded_observation_safe_horizon_with_witnesses(
        &self,
        source: u32,
        bytes: U8Set,
        active_terminals: &BitSet,
        max_horizon: u8,
    ) -> (u8, Vec<(u32, u8)>) {
        const MAX_FRONTIER_STATES: usize = 4_096;
        // Witnesses are an optional cache accelerator, never part of the
        // correctness proof. Avoid retaining pathological wide-frontier
        // traversals; the source result remains exact if this cap is exceeded.
        const MAX_WITNESSES: usize = 16_384;

        #[inline]
        fn equal_under_mask(left: &BitSet, right: &BitSet, mask: &BitSet) -> bool {
            debug_assert_eq!(left.len(), right.len());
            debug_assert_eq!(left.len(), mask.len());
            left.words()
                .iter()
                .zip(right.words())
                .zip(mask.words())
                .all(|((&left, &right), &mask)| ((left ^ right) & mask) == 0)
        }

        if max_horizon == 0
            || bytes.is_empty()
            || source >= self.num_states()
            || self.state_has_epsilon_transitions(source)
        {
            return (0, Vec::new());
        }

        let source_finalizers = self.matched_terminal_bitset(source);
        let source_futures = self.possible_future_terminals(source);

        // Bounded-repeat products are represented by one compressed segment
        // whose byte equivalence classes are stable across every state in the
        // counter chain. Compute which of those classes intersect B once. A
        // 53-byte ASCII alphabet commonly collapses to one class, replacing 53
        // transition lookups at every depth with one row lookup.
        let source_segment = self.compressed_segment_for_state(source);
        let mut requested_source_classes = SmallVec::<[u8; 16]>::new();
        if let Some(segment) = source_segment {
            for byte in bytes.iter() {
                let class = segment.byte_to_class[byte as usize];
                if !requested_source_classes.contains(&class) {
                    requested_source_classes.push(class);
                }
            }
            requested_source_classes.sort_unstable();
        }

        let mut frontier = vec![source];
        let mut next = Vec::<u32>::with_capacity(16);
        let mut witnesses = Vec::<(u32, u8)>::with_capacity(128);
        witnesses.push((source, 0));
        let mut retain_witnesses = true;
        for depth in 1..=max_horizon {
            next.clear();
            for &state in &frontier {
                if self.state_has_epsilon_transitions(state) {
                    return (depth - 1, witnesses);
                }

                let mut visit_target = |target: u32| -> bool {
                    if target == u32::MAX || self.state_has_epsilon_transitions(target) {
                        return false;
                    }
                    if !equal_under_mask(
                        self.matched_terminal_bitset(target),
                        source_finalizers,
                        active_terminals,
                    ) || !equal_under_mask(
                        self.possible_future_terminals(target),
                        source_futures,
                        active_terminals,
                    ) {
                        return false;
                    }
                    next.push(target);
                    true
                };

                if let Some(segment) = source_segment.filter(|segment| segment.contains_state(state)) {
                    let local_state = state - segment.state_offset;
                    let row_start = segment.row_offsets[local_state as usize] as usize;
                    let row_end = segment.row_offsets[local_state as usize + 1] as usize;
                    let row_classes = segment.entries.class_slice(row_start, row_end);
                    for &class in &requested_source_classes {
                        let Ok(index) = row_classes.binary_search(&class) else {
                            return (depth - 1, witnesses);
                        };
                        let local_target = segment.entries.target(row_start + index);
                        if !visit_target(segment.state_offset + local_target) {
                            return (depth - 1, witnesses);
                        }
                    }
                } else {
                    for byte in bytes.iter() {
                        if !visit_target(self.get_transition(state, byte)) {
                            return (depth - 1, witnesses);
                        }
                    }
                }
            }

            next.sort_unstable();
            next.dedup();
            if next.len() > MAX_FRONTIER_STATES {
                return (depth - 1, witnesses);
            }
            if retain_witnesses {
                if witnesses.len().saturating_add(next.len()) <= MAX_WITNESSES {
                    witnesses.extend(next.iter().copied().map(|state| (state, depth)));
                } else {
                    // Keep the already collected prefix. It remains a valid
                    // lower-bound witness set; simply stop adding more.
                    retain_witnesses = false;
                }
            }
            if next == frontier {
                return (max_horizon, witnesses);
            }
            std::mem::swap(&mut frontier, &mut next);
        }
        (max_horizon, witnesses)
    }

    fn transition_count(&self) -> usize {
        *self.transition_count_cache.get_or_init(|| {
            let packed_segments = self
                .packed_runtime_transition_segments
                .iter()
                .map(|segment| {
                    (0..segment.transitions.state_count() as u32)
                        .map(|state| {
                            segment
                                .transitions
                                .row(state)
                                .map_or(0, |(bytes, _)| bytes.len())
                        })
                        .sum::<usize>()
                })
                .sum::<usize>();
            let packed_compressed = self
                .packed_compressed_transition_segments
                .iter()
                .map(|segment| segment.expanded_transition_count)
                .sum::<usize>();
            if self.packed_runtime_metadata.is_some() {
                let residual = self
                    .packed_runtime_transitions
                    .as_ref()
                    .map_or_else(|| self.dfa.transition_count(), |transitions| {
                        (0..transitions.state_count() as u32)
                            .map(|state| transitions.row(state).map_or(0, |(bytes, _)| bytes.len()))
                            .sum()
                    });
                return residual + packed_compressed + packed_segments;
            }
            let compressed = self
                .compressed_transition_segments
                .iter()
                .map(|segment| segment.expanded_transition_count)
                .sum::<usize>();
            let stored_inside_compressed_segments = self
                .compressed_transition_segments
                .iter()
                .map(|segment| {
                    let start = segment.state_offset as usize;
                    let end = start + segment.state_count as usize;
                    self.dfa.states()[start..end]
                        .iter()
                        .map(|state| state.transitions.len())
                        .sum::<usize>()
                })
                .sum::<usize>();
            self.dfa
                .transition_count()
                .saturating_sub(stored_inside_compressed_segments)
                + compressed
                + packed_segments
                + packed_compressed
        })
    }

    /// Detect nullable terminals (those that match the empty string) by
    /// inspecting start-state finalizers, remove them from the DFA, and return
    /// the set.  After this call the tokenizer no longer reports those
    /// terminals as matched at state 0.
    pub fn isolate_start_state_and_drain_nullable_terminals(&mut self) -> BTreeSet<TerminalID> {
        let start = self.start_state();
        let initial_closure = self.dfa.epsilon_closure(&[start]);
        let mut nullable = BTreeSet::new();
        for &state in &initial_closure {
            nullable.extend(
                self.dfa
                    .finalizers(state)
                    .iter()
                    .map(|terminal| terminal as TerminalID),
            );
        }
        if nullable.is_empty() {
            return nullable;
        }
        self.invalidate_derived_caches();

        // The whole initial epsilon closure represents the zero-byte scanner
        // configuration. A component root can also be reached later after a
        // byte transition (for example, a nullable `a*` terminal looping to its
        // root). Clearing its finalizers in place would then remove legitimate
        // non-empty matches. Clone the closure as the post-consumption version,
        // redirect byte entries and external epsilon entries to those clones,
        // and drain finalizers only from the original zero-byte closure.
        let original_state_count = self.dfa.num_states();
        let mut post_byte_state = vec![u32::MAX; original_state_count];
        for &state in &initial_closure {
            let clone = self.dfa.clone_state(state);
            post_byte_state[state as usize] = clone;
        }

        let in_initial_closure = |state: u32| {
            (state as usize) < post_byte_state.len()
                && post_byte_state[state as usize] != u32::MAX
        };

        // Rewrite the cloned closure so all of its internal epsilon structure
        // remains in the post-byte coordinate.
        for &state in &initial_closure {
            let clone = post_byte_state[state as usize];
            let clone_state = &mut self.dfa.states_mut()[clone as usize];
            for (_, target) in clone_state.transitions.iter_mut() {
                if in_initial_closure(*target) {
                    *target = post_byte_state[*target as usize];
                }
            }
            for target in &mut clone_state.epsilon_transitions {
                if in_initial_closure(*target) {
                    *target = post_byte_state[*target as usize];
                }
            }
        }

        // A byte edge always enters the post-byte coordinate. An epsilon edge
        // from outside the initial closure can only be traversed after input has
        // already been consumed, so it does too. Epsilon edges within the
        // original closure remain untouched for the initial zero-byte closure.
        for source in 0..original_state_count {
            let source_in_initial_closure = in_initial_closure(source as u32);
            let state = &mut self.dfa.states_mut()[source];
            for (_, target) in state.transitions.iter_mut() {
                if in_initial_closure(*target) {
                    *target = post_byte_state[*target as usize];
                }
            }
            if !source_in_initial_closure {
                for target in &mut state.epsilon_transitions {
                    if in_initial_closure(*target) {
                        *target = post_byte_state[*target as usize];
                    }
                }
            }
        }

        for state in initial_closure {
            self.dfa.clear_finalizers_for_state(state);
        }
        // `possible_future_group_ids` is a strict (one-or-more-byte) property.
        // Draining the initial closure changes only zero-byte acceptance. Every
        // byte transition that formerly entered that closure now enters an
        // exact clone with the original finalizers and future metadata, and
        // epsilon entries from post-byte states are redirected likewise. Thus
        // every language reachable after consuming at least one byte is
        // unchanged, so the existing strict-future metadata remains exact.
        // Recomputing it here is especially expensive for a partitioned lexer:
        // the epsilon-aware generic fixpoint repeatedly closes every state of
        // the whole union even though this transformation preserves all strict
        // futures by construction.
        nullable
    }

    fn step(&self, state: u32, byte: u8) -> Option<u32> {
        if let Some(runtime) = self.virtual_residual_runtime_for_state(state) {
            return runtime.step(state, byte);
        }
        if let Some(runtime) = self.virtual_repeat_runtime_for_state(state) {
            return runtime.step(state, byte);
        }
        if let Some(runtime) = self.virtual_unit_repeat.as_deref()
            && runtime.handles_state(state)
        {
            return runtime.step(state, byte);
        }
        if let Some(packed) = &self.packed_runtime_transitions {
            if (state as usize) < packed.state_count() {
                return packed.transition(state, byte);
            }
        }
        if let Some(segment) = self.packed_runtime_transition_segment_for_state(state) {
            return segment.transition(state, byte);
        }
        if let Some(segment) = self.packed_compressed_segment_for_state(state) {
            return segment.transition(state, byte);
        }
        self.compressed_segment_for_state(state)
            .map_or_else(|| self.dfa.step(state, byte), |segment| segment.transition(state, byte))
    }

    fn step_all(&self, states: &[u32], byte: u8) -> TokenizerStateSet {
        let has_any_compressed = !self.compressed_transition_segments.is_empty()
            || !self.packed_compressed_transition_segments.is_empty()
            || !self.packed_runtime_transition_segments.is_empty();
        if self.virtual_unit_repeat.is_none()
            && self.virtual_repeat_intersections.is_empty()
            && self.virtual_residuals.is_empty()
            && !has_any_compressed
            && self.packed_runtime_metadata.is_none()
            && self.packed_runtime_metadata_segments.is_empty()
            && self.packed_runtime_transitions.is_none()
        {
            return self.dfa.step_all(states, byte);
        }
        if states.len() == 1 {
            let state = states[0];
            if !self.state_has_epsilon_transitions(state)
                && let Some(target) = self.step(state, byte)
                && !self.state_has_epsilon_transitions(target)
            {
                return TokenizerStateSet::from_buf([target]);
            }
        }
        let closure = self.epsilon_closure_states(states);
        let mut targets = TokenizerStateSet::new();
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
        self.epsilon_closure_states(&targets)
    }

    fn get_transition(&self, state: u32, byte: u8) -> u32 {
        self.step(state, byte).unwrap_or(u32::MAX)
    }

    pub fn run(&self, input: &[u8]) -> TokenizerStateSet {
        self.scan_input(input, self.start_state(), &mut (), |_, _, _, _| {})
    }

    #[doc(hidden)]
    pub fn artifact_metadata_stats(&self) -> (usize, usize, usize, usize) {
        let mut finalizer_bits = 0usize;
        let mut future_bits = 0usize;
        let mut max_finalizers = 0usize;
        let mut max_futures = 0usize;
        for state in 0..self.num_states() {
            let finalizers = self.state_finalizers(state).count_ones();
            let futures = self.state_futures(state).count_ones();
            finalizer_bits += finalizers;
            future_bits += futures;
            max_finalizers = max_finalizers.max(finalizers);
            max_futures = max_futures.max(futures);
        }
        (finalizer_bits, future_bits, max_finalizers, max_futures)
    }

    pub fn matched_terminals(&self, state: u32) -> BTreeSet<TerminalID> {
        self.epsilon_closure_states(&[state])
            .into_iter()
            .flat_map(|state| self.matched_terminals_iter(state))
            .collect()
    }

    #[inline]
    pub fn matched_terminals_slice(&self, state: u32) -> &[TerminalID] {
        if let Some(finalizers) = self
            .virtual_residual_runtime_for_state(state)
            .and_then(|runtime| runtime.finalizer_list(state))
        {
            return finalizers;
        }
        if let Some(finalizers) = self
            .virtual_repeat_runtime_for_state(state)
            .and_then(|runtime| runtime.finalizer_list(state))
        {
            return finalizers;
        }
        if let Some(finalizers) = self
            .virtual_unit_repeat
            .as_deref()
            .and_then(|runtime| runtime.finalizer_list(state))
        {
            return finalizers;
        }
        if let Some(metadata) = self
            .packed_runtime_metadata
            .as_deref()
            .filter(|metadata| state < metadata.state_count)
        {
            return metadata
                .finalizer_list(state)
                .expect("packed tokenizer finalizer row must cover every state");
        }
        if let Some(segment) = self.packed_runtime_metadata_segment_for_state(state) {
            return segment
                .metadata
                .finalizer_list(segment.local_state(state))
                .expect("packed tokenizer metadata segment must cover every state");
        }
        self.matched_terminals_cache
            .get_or_init(|| {
                let state_count = self.num_states() as usize;
                let mut offsets = Vec::with_capacity(state_count + 1);
                let mut entries = Vec::<TerminalID>::new();
                offsets.push(0);
                for raw_state in 0..self.num_states() {
                    entries.extend(
                        self.state_finalizers(raw_state)
                            .iter()
                            .map(|terminal| terminal as TerminalID),
                    );
                    offsets.push(entries.len());
                }
                Arc::new(MatchedTerminalLists {
                    offsets: offsets.into(),
                    entries: entries.into(),
                })
            })
            .for_state(state)
    }


    pub fn all_singleton_epsilon_closures(&self) -> Arc<SingletonEpsilonClosures> {
        Arc::clone(
            self.singleton_epsilon_closures
                .get_or_init(|| {
                    if self.packed_runtime_metadata.is_none()
                        && self.packed_runtime_metadata_segments.is_empty()
                    {
                        return Arc::new(self.dfa.all_singleton_epsilon_closures());
                    }
                    Arc::new(SingletonEpsilonClosures::Dense(
                        (0..self.num_states())
                            .map(|state| self.epsilon_closure_states(&[state]).into_boxed_slice())
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                    ))
                }),
        )

    }


    /// Exact epsilon-closed tokenizer frontiers after one byte from reset.
    /// Entry `b` is epsilon_closure(move(epsilon_closure(reset), b)).
    pub fn initial_byte_frontiers(&self) -> Arc<[TokenizerStateSet]> {
        Arc::clone(self.initial_byte_frontiers.get_or_init(|| {
            let closures = self.all_singleton_epsilon_closures();
            let reset = self.initial_state();
            let reset_closure = closures
                .get(reset as usize)
                .expect("tokenizer reset state must have an epsilon closure");
            let mut rows = Vec::with_capacity(256);
            for byte in 0u16..=255 {
                let byte = byte as u8;
                let mut targets = TokenizerStateSet::new();
                for &state in reset_closure {
                    let Some(target) = self.step(state, byte) else {
                        continue;
                    };
                    if let Some(closure) = closures.get(target as usize) {
                        targets.extend(closure.iter().copied());
                    } else {
                        targets.extend(self.epsilon_closure_states(&[target]));
                    }
                }
                targets.sort_unstable();
                targets.dedup();
                rows.push(targets);
            }
            Arc::from(rows.into_boxed_slice())
        }))
    }

    pub fn cached_singleton_epsilon_closures(
        &self,
    ) -> Option<&Arc<SingletonEpsilonClosures>> {
        self.singleton_epsilon_closures.get()
    }

    /// Return one exact self-loop byte set per raw tokenizer state.
    ///
    /// This is deliberately separate from `self_loop_bytes(state)`: callers
    /// needing one state keep the local O(out-degree) query, while whole-DFA
    /// compiler passes explicitly opt into one cached O(transitions) build.
    pub fn all_self_loop_bytes(&self) -> Arc<[U8Set]> {
        Arc::clone(self.all_self_loop_bytes_cache.get_or_init(|| {
            Arc::from(
                (0..self.num_states())
                    .map(|state| self.self_loop_bytes(state))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        }))
    }

    pub fn singleton_epsilon_closure(&self, state: u32) -> Box<[u32]> {
        self.epsilon_closure_states(&[state]).into_boxed_slice()
    }

    fn matched_terminals_iter(
        &self,
        state: u32,
    ) -> impl Iterator<Item = TerminalID> + '_ {
        self.state_finalizers(state)
            .iter()
            .map(|terminal| terminal as TerminalID)
    }

    fn matched_terminal_bitset(&self, state: u32) -> &BitSet {
        self.state_finalizers(state)
    }

    fn possible_future_terminals_iter(
        &self,
        state: u32,
    ) -> impl Iterator<Item = TerminalID> + '_ {
        self.state_futures(state)
            .iter()
            .map(|terminal| terminal as TerminalID)
    }

    fn possible_future_terminals(&self, state: u32) -> &BitSet {
        self.state_futures(state)
    }

    fn is_end(&self, state: u32) -> bool {
        self.possible_future_terminals(state).is_empty()
    }

    fn num_states(&self) -> u32 {
        let mut count = self.dfa.num_states() as u32;
        if let Some(metadata) = self.packed_runtime_metadata.as_deref() {
            count = count.max(metadata.state_count);
        }
        for segment in self.packed_runtime_metadata_segments.iter() {
            count = count.max(segment.state_offset.saturating_add(segment.metadata.state_count));
        }
        count
    }

    fn compute_forced_minimized_state_count(&self) -> usize {
        *self
            .forced_minimized_state_count_cache
            .get_or_init(|| self.dfa.minimize().num_states())
    }

    fn execute_from_state_all_widths(
        &self,
        input: &[u8],
        start: u32,
    ) -> TokenizerExecResult {
        let mut matches = Vec::new();
        let mut end_states = self.scan_input(input, start, &mut matches, |tokenizer, matches, state, width| {
            tokenizer.record_all_matches(matches, state, width);
        });
        end_states.retain(|state| !self.is_end(*state));

        TokenizerExecResult {
            end_state: end_states,
            matches,
        }
    }

    fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult {
        let mut matches = FxHashMap::<TerminalID, (usize, TokenizerStateSet)>::default();
        let end_states = self.scan_input(input, start, &mut matches, |tokenizer, matches, state, width| {
            tokenizer.record_longest_matches(matches, state, width);
        });

        TokenizerExecResult {
            end_state: end_states,
            matches: into_longest_matches(matches),
        }
    }

    /// Exact compact residual scan for compiler analyses that need only each
    /// terminal's longest width and the live states after the complete input.
    /// Unlike `execute_from_state`, this does not retain one matching end-state
    /// set per terminal only to discard it afterward.
    pub fn execute_summary_from_state(
        &self,
        input: &[u8],
        start: u32,
    ) -> (TokenizerStateSet, SmallVec<[(TerminalID, usize); 4]>) {
        let mut matches = SmallVec::<[(TerminalID, usize); 4]>::new();
        let end_states = self.scan_input(
            input,
            start,
            &mut matches,
            |tokenizer, matches, state, width| {
                for terminal in tokenizer.matched_terminals_iter(state) {
                    if let Some((_, longest)) = matches
                        .iter_mut()
                        .find(|(candidate, _)| *candidate == terminal)
                    {
                        *longest = (*longest).max(width);
                    } else {
                        matches.push((terminal, width));
                    }
                }
            },
        );
        matches.sort_unstable_by_key(|(terminal, _)| *terminal);
        (end_states, matches)
    }

    /// Execute the same exact compact residual scan for many starting states,
    /// merging starts as soon as their live lexer states and accumulated
    /// longest matches become identical.
    ///
    /// The returned tuple is `(end_states, longest_matches, starts)`. Expanding
    /// `starts` and comparing each entry with [`Self::execute_summary_from_state`]
    /// yields identical results. This is intended for compiler analyses where
    /// thousands of residual states are tested against the same byte string and
    /// rapidly converge after the first few bytes.
    pub fn execute_summary_groups_from_states(
        &self,
        input: &[u8],
        starts: &[u32],
    ) -> Vec<(
        TokenizerStateSet,
        SmallVec<[(TerminalID, usize); 4]>,
        Vec<u32>,
    )> {
        type ScanKey = (
            TokenizerStateSet,
            SmallVec<[(TerminalID, usize); 4]>,
        );

        let mut active = FxHashMap::<ScanKey, Vec<u32>>::default();
        // Compiler analyses often materialize the singleton-closure table
        // before scanning many residual starts. Reuse it when already present;
        // do not force construction here so callers that only need a handful of
        // scans retain the historical lazy behavior.
        let cached_singleton_closures = self
            .cached_singleton_epsilon_closures()
            .map(Arc::as_ref);
        for &start in starts {
            let states = cached_singleton_closures
                .and_then(|closures| closures.get(start as usize))
                .map(TokenizerStateSet::from_slice)
                .unwrap_or_else(|| self.epsilon_closure_states(&[start]));
            active
                .entry((states, SmallVec::new()))
                .or_default()
                .push(start);
        }
        let mut finished = FxHashMap::<ScanKey, Vec<u32>>::default();

        for (index, &byte) in input.iter().enumerate() {
            let width = index + 1;
            let mut next = FxHashMap::<ScanKey, Vec<u32>>::default();
            for ((states, mut matches), support) in active {
                let end_states = self.step_all(&states, byte);
                if end_states.is_empty() {
                    finished
                        .entry((end_states, matches))
                        .or_default()
                        .extend(support);
                    continue;
                }
                for &state in &end_states {
                    for terminal in self.matched_terminals_iter(state) {
                        if let Some((_, longest)) = matches
                            .iter_mut()
                            .find(|(candidate, _)| *candidate == terminal)
                        {
                            *longest = (*longest).max(width);
                        } else {
                            matches.push((terminal, width));
                        }
                    }
                }
                matches.sort_unstable_by_key(|(terminal, _)| *terminal);
                next.entry((end_states, matches))
                    .or_default()
                    .extend(support);
            }
            active = next;
            if active.is_empty() {
                break;
            }
        }
        for (key, support) in active {
            finished.entry(key).or_default().extend(support);
        }

        let mut groups = finished
            .into_iter()
            .map(|((states, matches), mut support)| {
                support.sort_unstable();
                support.dedup();
                (states, matches, support)
            })
            .collect::<Vec<_>>();
        groups.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });
        groups
    }

    fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> TokenizerStateSet {
        self.scan_input(input, start, &mut (), |_, _, _, _| {})
    }

    fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult {
        let exec = self.execute_from_state_all_widths(input, start);
        let end_states = if exec.end_state.is_empty() {
            SmallVec::from_buf([start])
        } else {
            exec.end_state
        };
        TokenizerResult {
            end_state: end_states,
            matches: group_matches_by_width(exec.matches),
        }
    }

    fn initial_state(&self) -> u32 {
        self.start_state()
    }

    fn initial_state_id(&self) -> u32 {
        self.initial_state()
    }

    fn tokens_accessible_from_state(&self, state: u32) -> &BitSet {
        self.possible_future_terminals(state)
    }

    /// Scan input bytes and report which terminals of interest matched/finalized.
    ///
    /// Returns a bitset of matched terminals and an optional end state.
    ///
    /// Algorithm:
    /// 1. `remaining = terminals_of_interest`.
    /// 2. `matched = empty`.
    /// 3. For each byte:
    ///    - Check if current state's possible futures overlap `remaining`.
    ///      If not, return `(matched, None)`.
    ///    - Consume byte -> next state.
    ///    - If no transition, return `(matched, None)`.
    ///    - Get finalizers at next state, intersect with `remaining`.
    ///    - Add intersection to `matched`, remove from `remaining`.
    /// 4. After all bytes, check futures at end state overlap `remaining`.
    ///    If not, return `(matched, None)`. Otherwise `(matched, Some(end_state))`.
    ///
    /// Important: initial-state finalizers are intentionally ignored.
    /// Only post-byte finalizers count.
    ///
    /// `terminals_of_interest` must have length equal to `self.num_terminals`.
    fn scan_terminal_matches_from_state(
        &self,
        input: &[u8],
        start: u32,
        terminals_of_interest: &BitSet,
    ) -> (BitSet, TokenizerStateSet) {
        debug_assert_eq!(terminals_of_interest.len(), self.num_terminals as usize);
        let mut remaining = terminals_of_interest.clone();
        let mut matched = BitSet::new(self.num_terminals as usize);
        let mut states = self.epsilon_closure_states(&[start]);

        for &byte in input {
            let any_future = states
                .iter()
                .any(|&state| !self.possible_future_terminals(state).is_disjoint(&remaining));
            if !any_future {
                return (matched, TokenizerStateSet::new());
            }

            states = self.step_all(&states, byte);
            if states.is_empty() {
                return (matched, states);
            }

            let mut finals = BitSet::new(self.num_terminals as usize);
            for &state in &states {
                finals.union_with(&self.state_finalizers(state).intersection(&remaining));
            }
            matched.union_with(&finals);
            remaining = remaining.difference(&finals);
        }

        states.retain(|state| !self.possible_future_terminals(*state).is_disjoint(&remaining));
        (matched, states)
    }

    fn record_all_matches(&self, matches: &mut Vec<TokenizerMatch>, state: u32, width: usize) {
        matches.extend(self.matched_terminals_iter(state).map(|id| TokenizerMatch {
            id,
            width,
            end_state: state,
        }));
    }

    fn record_longest_matches(
        &self,
        matches: &mut FxHashMap<TerminalID, (usize, TokenizerStateSet)>,
        state: u32,
        width: usize,
    ) {
        for terminal in self.matched_terminals_iter(state) {
            let entry = matches
                .entry(terminal)
                .or_insert_with(|| (width, TokenizerStateSet::new()));
            if width > entry.0 {
                entry.0 = width;
                entry.1.clear();
            }
            if width == entry.0 && !entry.1.contains(&state) {
                entry.1.push(state);
            }
        }
    }

    fn scan_input<R>(
        &self,
        input: &[u8],
        start: u32,
        mut matches: &mut R,
        mut record_matches: impl FnMut(&Self, &mut R, u32, usize),
    ) -> TokenizerStateSet {
        // The partitioned runtime tokenizer has a zero-byte dispatcher whose
        // outgoing roots are already deterministic and epsilon-free. Once at
        // least one byte will be consumed, the dispatcher itself cannot remain
        // live, so enter those roots directly instead of materializing its
        // epsilon closure. On very large synthesized tokenizers that closure's
        // generic dense `seen` scratch is otherwise proportional to every raw
        // tokenizer state even though only a handful of roots are live.
        let mut states = if !input.is_empty() && start == self.initial_state_id() {
            self.deterministic_dispatch_roots()
                .map(TokenizerStateSet::from_slice)
                .unwrap_or_else(|| self.epsilon_closure_states(&[start]))
        } else {
            self.epsilon_closure_states(&[start])
        };
        for (index, &byte) in input.iter().enumerate() {
            states = self.step_all(&states, byte);
            if states.is_empty() {
                return states;
            }
            for &state in &states {
                record_matches(self, &mut matches, state, index + 1);
            }
        }
        states
    }


}

impl Lexer for Tokenizer {
    fn start_state(&self) -> u32 { self.start_state() }
    fn num_terminals(&self) -> u32 { self.num_terminals() }
    fn has_epsilon_transitions(&self) -> bool { self.has_epsilon_transitions() }
    fn state_has_epsilon_transitions(&self, state: u32) -> bool { self.state_has_epsilon_transitions(state) }
    fn transitions_from(&self, state: u32) -> impl Iterator<Item = (u8, u32)> + '_ { self.transitions_from(state) }
    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) { self.fill_transition_row(state, row); }
    fn transition_row(&self, state: u32) -> Box<[u32; 256]> { self.transition_row(state) }
    fn self_loop_bytes(&self, state: u32) -> U8Set { self.self_loop_bytes(state) }
    fn transition_count(&self) -> usize { self.transition_count() }
    fn step(&self, state: u32, byte: u8) -> Option<u32> { self.step(state, byte) }
    fn step_all(&self, states: &[u32], byte: u8) -> TokenizerStateSet { self.step_all(states, byte) }
    fn get_transition(&self, state: u32, byte: u8) -> u32 { self.get_transition(state, byte) }
    fn matched_terminal_bitset(&self, state: u32) -> &BitSet { self.matched_terminal_bitset(state) }
    fn matched_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_ { self.matched_terminals_iter(state) }
    fn possible_future_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_ { self.possible_future_terminals_iter(state) }
    fn possible_future_terminals(&self, state: u32) -> &BitSet { self.possible_future_terminals(state) }
    fn is_end(&self, state: u32) -> bool { self.is_end(state) }
    fn num_states(&self) -> u32 { self.num_states() }
    fn compute_forced_minimized_state_count(&self) -> usize { self.compute_forced_minimized_state_count() }
    fn execute_from_state_all_widths(&self, input: &[u8], start: u32) -> TokenizerExecResult { self.execute_from_state_all_widths(input, start) }
    fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult { self.execute_from_state(input, start) }
    fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> TokenizerStateSet { self.execute_from_state_end_only(input, start) }
    fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult { self.execute_all_matches(input, start) }
    fn initial_state(&self) -> u32 { self.initial_state() }
    fn initial_state_id(&self) -> u32 { self.initial_state_id() }
    fn tokens_accessible_from_state(&self, state: u32) -> &BitSet { self.tokens_accessible_from_state(state) }
    fn scan_terminal_matches_from_state(&self, input: &[u8], start: u32, terminals_of_interest: &BitSet) -> (BitSet, TokenizerStateSet) {
        self.scan_terminal_matches_from_state(input, start, terminals_of_interest)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerResult {
    pub end_state: TokenizerStateSet,
    pub matches: Vec<(usize, BTreeSet<TerminalID>)>,
}

pub fn arbitrary_epsilon_l1_test_tokenizer() -> Tokenizer {
    let mut dfa = DFA::new(7);
    dfa.ensure_group_capacity(2);
    dfa.add_epsilon_transition(0, 1);
    dfa.add_epsilon_transition(1, 2);
    dfa.add_epsilon_transition(1, 4);
    dfa.add_transition(2, b'a', 3);
    dfa.add_transition(4, b'a', 5);
    dfa.add_transition(2, b'b', 6);

    let mut terminal_zero = BitSet::new(2);
    terminal_zero.set(0);
    dfa.overwrite_state_metadata(3, terminal_zero.clone(), BitSet::new(2));
    dfa.overwrite_state_metadata(6, terminal_zero, BitSet::new(2));
    let mut terminal_one = BitSet::new(2);
    terminal_one.set(1);
    dfa.overwrite_state_metadata(5, terminal_one, BitSet::new(2));
    dfa.recompute_possible_futures();

    let tokenizer = Tokenizer::from_parts(dfa, 2, None);
    assert!(tokenizer.has_epsilon_transitions());
    assert!(!tokenizer.has_deterministic_dispatch());
    tokenizer
}

#[doc(hidden)]
pub fn arbitrary_flat32_test_tokenizer() -> Tokenizer {
    const TARGET: u32 = 32_768;
    let mut dfa = DFA::new(TARGET as usize + 1);
    dfa.ensure_group_capacity(1);
    dfa.add_transition(0, b'a', TARGET);
    let mut finalizers = BitSet::new(1);
    finalizers.set(0);
    dfa.overwrite_state_metadata(TARGET, finalizers, BitSet::new(1));
    dfa.recompute_possible_futures();
    Tokenizer::from_parts(dfa, 1, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::{bytes, plus};
    use crate::automata::lexer::compile::build_regex;

    fn tokenizer_from_exprs(exprs: Vec<Expr>) -> Tokenizer {
        let num_terminals = exprs.len() as u32;
        build_regex(&exprs).into_tokenizer(
            num_terminals,
            Some(Arc::from(exprs.into_boxed_slice())),
        )
    }


    #[test]
    fn global_scalar_proof_matches_local_scan_on_generated_graphs() {
        // Includes loops, absent bytes, shared targets and reset-unreachable
        // states. Every root is compared, not just the tokenizer's start.
        for states in [1usize, 2, 7, 31] {
            for seed in 0..12u64 {
                let mut bits = seed + 1;
                let mut dfa = DFA::new(states);
                dfa.ensure_group_capacity(2);
                for source in 0..states as u32 {
                    for byte in [0u8, b'a', b'b', b'"', 128, 255] {
                        bits = bits.wrapping_mul(6364136223846793005).wrapping_add(1);
                        if bits & 3 != 0 {
                            dfa.add_transition(source, byte, ((bits >> 32) % states as u64) as u32);
                        }
                    }
                }
                let tokenizer = Tokenizer::from_parts(dfa, 2, None);
                assert!(!tokenizer.has_epsilon_transitions());
                assert!(!tokenizer.has_any_virtual_runtime());
                for root in 0..states as u32 {
                    assert!(tokenizer.scalar_physical_component_root_impl::<true>(root));
                    assert_eq!(tokenizer.scalar_physical_component_root_impl::<true>(root),
                        tokenizer.scalar_physical_component_root_impl::<false>(root));
                }
                for root in [states as u32, u32::MAX] {
                    assert!(!tokenizer.scalar_physical_component_root_impl::<true>(root));
                    assert!(!tokenizer.scalar_physical_component_root_impl::<false>(root));
                }
            }
        }
    }

    #[test]
    fn global_scalar_proof_preserves_mixed_component_refusals() {
        let mut dfa = DFA::new(5);
        dfa.ensure_group_capacity(1);
        dfa.add_transition(0, b'a', 1);
        dfa.add_transition(1, b'b', 2);
        dfa.add_epsilon_transition(2, 3);
        // Root 4 remains scalar even though unrelated roots are not.
        dfa.add_transition(4, b'z', 4);
        let tokenizer = Tokenizer::from_parts(dfa, 1, None);
        assert!(tokenizer.has_epsilon_transitions());
        for root in 0..5 {
            assert_eq!(tokenizer.scalar_physical_component_root_impl::<true>(root),
                tokenizer.scalar_physical_component_root_impl::<false>(root));
        }
        assert!(!tokenizer.scalar_physical_component_root_impl::<true>(0));
        assert!(tokenizer.scalar_physical_component_root_impl::<true>(4));

        let mut tokenizer = tokenizer_from_exprs(vec![
            Expr::U8Seq(b"b".to_vec()), Expr::U8Class(U8Set::empty()),
        ]);
        tokenizer.isolate_start_state_and_drain_nullable_terminals();
        tokenizer.install_virtual_zero_min_unit_repeat_component(
            U8Set::single(b'a'), 1_000_000_000, 1,
        ).unwrap();
        assert!(tokenizer.has_any_virtual_runtime());
        for root in 0..tokenizer.num_states() {
            assert_eq!(tokenizer.scalar_physical_component_root_impl::<true>(root),
                tokenizer.scalar_physical_component_root_impl::<false>(root));
        }
    }

    #[test]
    fn bounded_eager_candidates_preserve_exact_rows_and_single_worker_execution() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| {
            let t = Tokenizer::from_parts(one_byte_component(b'a'), 1, None);
            let original = t.build_terminal_projected_quotients_for_containment_candidates(&[0]);
            let bounded = t.build_terminal_projected_quotients_for_containment_candidates_bounded(&[0, 0, 7], 2, 1);
            assert_eq!(original.len(), 1);
            assert_eq!(bounded.len(), 1);
            assert_eq!(bincode::serialize(&original).unwrap(), bincode::serialize(&bounded).unwrap());
            assert!(t.build_terminal_projected_quotients_for_containment_candidates_bounded(&[0], 1, 1).is_empty());
            assert!(t.build_terminal_projected_quotients_for_containment_candidates_bounded(&[0], 2, 0).is_empty());
        });
    }

    #[test]
    fn coldcap_component_limits_are_exact_and_cycles_are_deduplicated() {
        let t = Tokenizer::from_parts(one_byte_component(b'a'), 1, None);
        let full = t.scalar_physical_component_states(0).unwrap();
        assert_eq!(full, vec![0, 1]);
        assert!(t.scalar_physical_component_states_bounded(0, 0, 10).is_none());
        assert!(t.scalar_physical_component_states_bounded(0, 1, 10).is_none());
        assert!(t.scalar_physical_component_states_bounded(0, 2, 0).is_none());
        assert_eq!(t.scalar_physical_component_states_bounded(0, 2, 1), Some(full));
        let looped = tokenizer_from_exprs(vec![plus(bytes(&[b'a', b'b', b'c']))]);
        let root = looped.start_state();
        let states = looped.scalar_physical_component_states(root).unwrap();
        let edges = states.iter().map(|&s| looped.transitions_from(s).count()).sum();
        assert_eq!(looped.scalar_physical_component_states_bounded(root, states.len(), edges), Some(states));
    }

    #[test]
    fn coldcap_accepted_quotient_is_identical_to_unbounded_constructor() {
        let t = Tokenizer::from_parts(one_byte_component(b'a'), 1, None);
        let old = t.build_terminal_projected_quotients_for_containment_candidates(&[0]);
        let new = t.build_terminal_projected_quotient_for_containment_bounded(0, 2, 1).unwrap();
        assert_eq!(old.len(), 1);
        assert_eq!(new.full_states, old[0].1.full_states);
        assert_eq!(new.projected_states, old[0].1.projected_states);
        assert_eq!(new.byte_to_class, old[0].1.byte_to_class);
        assert_eq!(new.class_representatives, old[0].1.class_representatives);
        assert_eq!(new.class_targets, old[0].1.class_targets);
        assert!(t.build_terminal_projected_quotient_for_containment_bounded(0, 1, 1).is_none());
        assert!(t.build_terminal_projected_quotient_for_containment_bounded(7, 2, 1).is_none());
    }

    #[test]
    fn coldcap_rejects_epsilon_component_without_partial_certificate() {
        let mut dfa = one_byte_component(b'a');
        dfa.add_epsilon_transition(0, 1);
        let t = Tokenizer::from_parts(dfa, 1, None);
        assert!(t.scalar_physical_component_states_bounded(0, 100, 1000).is_none());
    }

    fn one_byte_component(byte: u8) -> DFA {
        let mut dfa = DFA::new(2);
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, U8Set::single(byte));
        dfa.add_transition(0, byte, 1);
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        dfa.overwrite_state_metadata(1, finalizers, BitSet::new(1));
        dfa.recompute_possible_futures();
        dfa
    }

    #[test]
    fn direct_mask_component_isolated_start_future_update_matches_full_recompute() {
        let mut host = DFA::new(2);
        host.ensure_group_capacity(2);
        host.set_group_u8set(0, U8Set::single(b'a'));
        host.set_group_u8set(1, U8Set::single(b'b'));
        host.add_epsilon_transition(0, 1);
        host.add_transition(1, b'a', 1);
        let mut finalizers = BitSet::new(2);
        finalizers.set(0);
        host.overwrite_state_metadata(1, finalizers, BitSet::new(2));
        host.recompute_possible_futures();
        let mut tokenizer = Tokenizer::from_parts(host, 2, None);

        tokenizer
            .install_direct_mask_components(vec![(one_byte_component(b'b'), 0, 1)])
            .expect("isolated-start direct component must install");

        let mut reference = tokenizer.dfa.clone();
        reference.recompute_possible_futures();
        assert_eq!(tokenizer.dfa.num_states(), reference.num_states());
        for state in 0..tokenizer.dfa.num_states() as u32 {
            assert_eq!(
                tokenizer.dfa.possible_future_group_ids(state),
                reference.possible_future_group_ids(state),
                "future mismatch at state {state}",
            );
        }
    }

    #[test]
    fn direct_mask_component_nonisolated_start_falls_back_to_full_recompute() {
        let mut host = DFA::new(2);
        host.ensure_group_capacity(2);
        host.set_group_u8set(0, U8Set::single(b'a'));
        host.set_group_u8set(1, U8Set::single(b'b'));
        host.add_transition(1, b'a', 0);
        host.recompute_possible_futures();
        let mut tokenizer = Tokenizer::from_parts(host, 2, None);

        tokenizer
            .install_direct_mask_components(vec![(one_byte_component(b'b'), 0, 1)])
            .expect("nonisolated-start direct component must install via fallback");

        let mut reference = tokenizer.dfa.clone();
        reference.recompute_possible_futures();
        for state in 0..tokenizer.dfa.num_states() as u32 {
            assert_eq!(
                tokenizer.dfa.possible_future_group_ids(state),
                reference.possible_future_group_ids(state),
                "future mismatch at state {state}",
            );
        }
    }

    #[test]
    fn batched_terminal_residual_replacement_preserves_appended_state_order() {
        let base_dfas = (0..3)
            .map(|_| Arc::new(DFA::new(2)))
            .collect::<Vec<_>>();
        let coordinates = TerminalResidualCoordinates::from_rows_and_dfas(
            vec![
                vec![(0, 0), (1, 0), (2, 0)],
                vec![(0, 1), (1, 1)],
                vec![(2, 1)],
            ],
            base_dfas,
        );
        let replacement_one = Arc::new(DFA::new(3));
        let replacement_two = Arc::new(DFA::new(2));
        let replaced = coordinates
            .replace_terminals_with_appended_dfas(&[
                (1, Arc::clone(&replacement_one)),
                (2, Arc::clone(&replacement_two)),
            ])
            .expect("valid batched replacement");

        let expected = [
            vec![(0, 0)],
            vec![(0, 1)],
            vec![],
            vec![(1, 0)],
            vec![(1, 1)],
            vec![(1, 2)],
            vec![(2, 0)],
            vec![(2, 1)],
        ];
        assert_eq!(replaced.len(), expected.len());
        for (state, expected_row) in expected.iter().enumerate() {
            assert_eq!(replaced.row(state as u32), Some(expected_row.as_slice()));
        }
        assert_eq!(replaced.terminal_dfa(1).unwrap().num_states(), 3);
        assert_eq!(replaced.terminal_dfa(2).unwrap().num_states(), 2);
    }

    #[test]
    fn batched_terminal_residual_replacement_matches_sequential_duplicate_semantics() {
        let coordinates = TerminalResidualCoordinates::from_rows_and_dfas(
            vec![vec![(0, 0), (1, 7)]],
            vec![Arc::new(DFA::new(1)), Arc::new(DFA::new(8))],
        );
        let replaced = coordinates
            .replace_terminals_with_appended_dfas(&[
                (1, Arc::new(DFA::new(2))),
                (1, Arc::new(DFA::new(3))),
            ])
            .expect("duplicate replacements remain well defined");

        let expected = [
            vec![(0, 0)],
            vec![],
            vec![],
            vec![(1, 0)],
            vec![(1, 1)],
            vec![(1, 2)],
        ];
        assert_eq!(replaced.len(), expected.len());
        for (state, expected_row) in expected.iter().enumerate() {
            assert_eq!(replaced.row(state as u32), Some(expected_row.as_slice()));
        }
        assert_eq!(replaced.terminal_dfa(1).unwrap().num_states(), 3);
    }

    #[test]
    fn grouped_terminal_residual_replacement_shares_physical_rows() {
        let base = Arc::new(DFA::new(1));
        let coordinates = TerminalResidualCoordinates::from_rows_and_dfas(
            vec![vec![(0, 0), (1, 0), (2, 0)]],
            vec![Arc::clone(&base), Arc::clone(&base), base],
        );
        let shared = Arc::new(DFA::new(2));
        let replaced = coordinates
            .replace_terminal_alias_groups_with_appended_dfas(&[(
                vec![1, 2],
                Arc::clone(&shared),
            )])
            .expect("valid grouped alias replacement");

        assert_eq!(replaced.len(), 3);
        assert_eq!(replaced.row(0), Some(&[(0, 0)][..]));
        assert_eq!(replaced.row(1), Some(&[(1, 0), (2, 0)][..]));
        assert_eq!(replaced.row(2), Some(&[(1, 1), (2, 1)][..]));
        assert!(Arc::ptr_eq(&replaced.terminal_dfas[1], &shared));
        assert!(Arc::ptr_eq(&replaced.terminal_dfas[2], &shared));
    }

    fn direct_mask_future_test_tokenizer(start_has_incoming: bool) -> Tokenizer {
        let mut dfa = DFA::new(2);
        dfa.ensure_group_capacity(2);
        let mut a = U8Set::empty();
        a.insert(b'a');
        dfa.set_group_u8set(0, a);
        let mut x = U8Set::empty();
        x.insert(b'x');
        dfa.set_group_u8set(1, x);
        dfa.add_transition(0, b'a', 1);
        if start_has_incoming {
            dfa.add_transition(1, b'b', 0);
        }
        let mut finalizers = BitSet::new(2);
        finalizers.set(0);
        dfa.overwrite_state_metadata(1, finalizers, BitSet::new(2));
        dfa.recompute_possible_futures();
        Tokenizer::from_parts(dfa, 2, None)
    }

    fn direct_mask_future_test_component() -> (DFA, u32, TerminalID) {
        let mut dfa = DFA::new(2);
        dfa.ensure_group_capacity(1);
        let mut x = U8Set::empty();
        x.insert(b'x');
        dfa.set_group_u8set(0, x);
        dfa.add_transition(0, b'x', 1);
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        dfa.overwrite_state_metadata(1, finalizers, BitSet::new(1));
        dfa.recompute_possible_futures();
        (dfa, 0, 1)
    }

    fn assert_same_future_metadata(left: &Tokenizer, right: &Tokenizer) {
        assert_eq!(left.dfa.num_states(), right.dfa.num_states());
        for state in 0..left.dfa.num_states() as u32 {
            assert_eq!(left.dfa.finalizers(state), right.dfa.finalizers(state));
            assert_eq!(
                left.dfa.possible_future_group_ids(state),
                right.dfa.possible_future_group_ids(state),
                "future metadata differs at state {state}",
            );
        }
    }

    #[test]
    fn direct_mask_incremental_start_futures_match_full_recompute() {
        let base = direct_mask_future_test_tokenizer(false);
        let mut incremental = base.clone();
        let mut full = base;
        incremental
            .install_direct_mask_components_impl(vec![direct_mask_future_test_component()], false)
            .unwrap();
        full.install_direct_mask_components_impl(vec![direct_mask_future_test_component()], true)
            .unwrap();
        assert_same_future_metadata(&incremental, &full);
        assert!(incremental.dfa.possible_future_group_ids(0).contains(1));
    }

    #[test]
    fn direct_mask_incremental_futures_fall_back_when_start_has_incoming_edge() {
        let base = direct_mask_future_test_tokenizer(true);
        let mut guarded = base.clone();
        let mut full = base;
        guarded
            .install_direct_mask_components_impl(vec![direct_mask_future_test_component()], false)
            .unwrap();
        full.install_direct_mask_components_impl(vec![direct_mask_future_test_component()], true)
            .unwrap();
        assert_same_future_metadata(&guarded, &full);
        assert!(
            guarded.dfa.possible_future_group_ids(1).contains(1),
            "predecessor of start must see the newly installed terminal future",
        );
    }

    #[test]
    fn direct_mask_exact_aliases_share_physical_component_without_changing_terminal_outcomes() {
        let mut base_dfa = DFA::new(1);
        base_dfa.ensure_group_capacity(3);
        let mut base = Tokenizer::from_parts(base_dfa, 3, None);
        let dummy = Arc::new(DFA::new(1));
        base.terminal_residual_coordinates = Some(Arc::new(
            TerminalResidualCoordinates::from_rows_and_dfas(
                vec![vec![(0, 0), (1, 0), (2, 0)]],
                vec![Arc::clone(&dummy), Arc::clone(&dummy), dummy],
            ),
        ));

        let component = Arc::new(direct_mask_future_test_component().0);
        let mut separate = base.clone();
        separate
            .install_direct_mask_components_shared(vec![
                (Arc::clone(&component), 0, 1),
                (Arc::clone(&component), 0, 2),
            ])
            .unwrap();
        let mut grouped = base;
        grouped
            .install_direct_mask_component_alias_groups(vec![(
                Arc::clone(&component),
                0,
                vec![1, 2],
            )])
            .unwrap();

        assert_eq!(
            separate.dfa.num_states(),
            grouped.dfa.num_states() + component.num_states(),
        );
        let separate_closure = separate.dfa.epsilon_closure(&[0]);
        let grouped_closure = grouped.dfa.epsilon_closure(&[0]);
        let separate_after_x = separate.dfa.step_all(&separate_closure, b'x');
        let grouped_after_x = grouped.dfa.step_all(&grouped_closure, b'x');
        let mut separate_finalizers = BitSet::new(3);
        for state in separate_after_x {
            separate_finalizers.union_with(separate.dfa.finalizers(state));
        }
        let mut grouped_finalizers = BitSet::new(3);
        for state in grouped_after_x {
            grouped_finalizers.union_with(grouped.dfa.finalizers(state));
        }
        assert_eq!(separate_finalizers, grouped_finalizers);
        assert!(grouped_finalizers.contains(1));
        assert!(grouped_finalizers.contains(2));
        assert!(grouped.dfa.possible_future_group_ids(0).contains(1));
        assert!(grouped.dfa.possible_future_group_ids(0).contains(2));

        let coordinates = grouped
            .terminal_residual_coordinates
            .as_ref()
            .expect("grouped install retains residual coordinates");
        assert_eq!(coordinates.len(), grouped.dfa.num_states());
        assert_eq!(coordinates.row(1), Some(&[(1, 0), (2, 0)][..]));
        assert_eq!(coordinates.row(2), Some(&[(1, 1), (2, 1)][..]));

        let (coordinate_partition, _, _) = grouped
            .terminal_residual_coordinate_observation_partition(1)
            .expect("retained residual coordinates should yield an observation partition");
        let (exact_partition, _, _) = grouped
            .exact_terminal_observation_partition(1, 1_024, 100_000)
            .expect("small deterministic fixture should admit the exact partition");
        let mut exact_by_coordinate = FxHashMap::<u32, u32>::default();
        for (&coordinate_class, &exact_class) in
            coordinate_partition.iter().zip(exact_partition.iter())
        {
            assert_eq!(coordinate_class == 0, exact_class == 0);
            if coordinate_class != 0 {
                if let Some(previous) = exact_by_coordinate.insert(coordinate_class, exact_class) {
                    assert_eq!(previous, exact_class);
                }
            }
        }
    }

    #[test]
    fn terminal_expr_projected_quotient_proves_exact_text_closure() {
        let tokenizer = tokenizer_from_exprs(vec![plus(bytes(b"a")), plus(bytes(b"b"))]);
        let source = tokenizer
            .terminal_scalar_dispatch_root(0)
            .expect("deterministic test tokenizer has a scalar terminal root");
        let quotient = tokenizer
            .terminal_expr_projected_quotient_from_root(0)
            .expect("retained terminal expression should admit an exact projected quotient");

        assert!(quotient.contains_source(source));
        assert!(quotient.byte_class_count() < 256);
        quotient
            .validate_exact_projection(&tokenizer, 0)
            .expect("freshly built projected quotient must validate against its tokenizer");

        let mut malformed = quotient.clone();
        malformed.byte_to_class[0] ^= 1;
        assert!(
            malformed.validate_exact_projection(&tokenizer, 0).is_err(),
            "wire validation must reject corrupted byte-class metadata"
        );

        let mut only_a = U8Set::empty();
        only_a.insert(b'a');
        assert_eq!(
            quotient.text_liveness_closed_bounded(source, only_a, 128, 4_096, false),
            Some(true),
        );

        let mut a_or_b = only_a;
        a_or_b.insert(b'b');
        assert_eq!(
            quotient.text_liveness_closed_bounded(source, a_or_b, 128, 4_096, false),
            Some(false),
            "a byte outside the projected terminal language must fail closed",
        );
    }

    fn serialized_roundtrip(tokenizer: &Tokenizer) -> Tokenizer {
        let bytes = bincode::serialize(tokenizer).unwrap();
        let loaded: Tokenizer = bincode::deserialize(&bytes).unwrap();
        assert!(loaded.exprs.is_none(), "Expr sidecars are intentionally not serialized");
        loaded
    }

    #[test]
    fn segment_wire_roundtrips_residual_and_compressed_transitions() {
        let mut dfa = DFA::new(4);
        dfa.ensure_group_capacity(1);
        let mut group = U8Set::empty();
        for byte in [b'a', b'b', b'x'] {
            group.insert(byte);
        }
        dfa.set_group_u8set(0, group);
        dfa.add_transition(0, b'x', 1);
        // Covered rows deliberately remain present in the compile-time DFA;
        // TKS2 must omit them and recover their behavior from the segment.
        dfa.add_transition(1, b'a', 2);
        dfa.add_transition(2, b'b', 3);
        dfa.add_epsilon_transition(0, 1);
        for state in 0..4u32 {
            let mut futures = BitSet::new(1);
            futures.set(0);
            let mut finalizers = BitSet::new(1);
            if state == 3 {
                finalizers.set(0);
            }
            dfa.overwrite_state_metadata(state, finalizers, futures);
        }
        let mut byte_to_class = vec![u8::MAX; 256];
        byte_to_class[b'a' as usize] = 0;
        byte_to_class[b'b' as usize] = 1;
        let segment = CompressedTransitionSegment {
            state_offset: 1,
            state_count: 3,
            byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
            class_members: Arc::from(
                vec![vec![b'a'].into_boxed_slice(), vec![b'b'].into_boxed_slice()]
                    .into_boxed_slice(),
            ),
            row_offsets: Arc::from(vec![0u32, 1, 2, 2].into_boxed_slice()),
            entries: CompressedTransitionEntries::from_parts(vec![0, 1], vec![1, 2]),
            expanded_transition_count: 2,
        };
        let original = Tokenizer::from_parts_with_compressed_transitions(
            dfa,
            1,
            None,
            vec![segment],
        );
        let wire = artifact_serde::to_segment_bytes(&original);
        assert!(wire.starts_with(b"TKS2"));
        let loaded = artifact_serde::from_fast_bytes(&wire).unwrap();
        assert_eq!(loaded.compressed_transition_segments.len(), 1);
        assert_eq!(loaded.transition_count(), original.transition_count());
        enumerate_bytes(b"abx", 3, |input| {
            for state in 0..original.num_states() {
                assert_eq!(
                    normalized_exec(&loaded, input, state),
                    normalized_exec(&original, input, state),
                    "TKS2 mismatch from state {state} on {input:?}",
                );
            }
        });

        let huge_wire = artifact_serde::build_huge_bytes(&original).expect("small TKS3 fixture");
        assert!(huge_wire.starts_with(b"TKS3"));
        let huge_loaded = artifact_serde::from_fast_bytes(&huge_wire).unwrap();
        assert!(huge_loaded.has_packed_runtime_metadata());
        assert_eq!(huge_loaded.num_states(), original.num_states());
        assert_eq!(huge_loaded.transition_count(), original.transition_count());
        enumerate_bytes(b"abx", 3, |input| {
            for state in 0..original.num_states() {
                assert_eq!(
                    normalized_exec(&huge_loaded, input, state),
                    normalized_exec(&original, input, state),
                    "TKS3 mismatch from state {state} on {input:?}",
                );
            }
        });

        // TKS2 is also the fallback for packed sources whose compressed
        // components are separated by ordinary states. It must not rely on
        // the loaded tokenizer retaining either owned rows or owned metadata.
        let fallback_wire = artifact_serde::to_segment_bytes(&huge_loaded);
        let fallback = artifact_serde::from_fast_bytes(&fallback_wire).unwrap();
        assert_eq!(fallback.num_states(), original.num_states());
        assert_eq!(fallback.compressed_transition_segments.len(), 1);
        enumerate_bytes(b"abx", 3, |input| {
            for state in 0..original.num_states() {
                assert_eq!(normalized_exec(&fallback, input, state),
                           normalized_exec(&original, input, state),
                           "packed-to-TKS2 mismatch state={state} input={input:?}");
            }
        });

    }

    #[test]
    fn segment_fallback_preserves_packed_overflow_deltas_and_ordinary_tail() {
        const COMPRESSED_STATES: u32 = 70_000;
        let mut dfa = DFA::new(COMPRESSED_STATES as usize + 1);
        dfa.ensure_group_capacity(1);
        let mut live = BitSet::new(1);
        live.set(0);
        for state in 0..=COMPRESSED_STATES {
            dfa.overwrite_state_metadata(state,
                if state == COMPRESSED_STATES { live.clone() } else { BitSet::new(1) },
                live.clone());
        }
        dfa.add_epsilon_transition(0, 1);
        let mut byte_to_class = vec![u8::MAX; 256];
        for byte in b'a'..=b'z' { byte_to_class[byte as usize] = 0; }
        let targets = (0..COMPRESSED_STATES).map(|state| {
            if state == 0 { COMPRESSED_STATES - 1 }
            else if state == COMPRESSED_STATES - 1 { 0 }
            else { state }
        }).collect();
        let original = Tokenizer::from_parts_with_compressed_transitions(dfa, 1, None,
            vec![CompressedTransitionSegment {
                state_offset: 1, state_count: COMPRESSED_STATES,
                byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
                class_members: Arc::from([Vec::from_iter(b'a'..=b'z').into_boxed_slice()]),
                row_offsets: Arc::from(Vec::from_iter(0..=COMPRESSED_STATES).into_boxed_slice()),
                entries: CompressedTransitionEntries::from_parts(vec![0; COMPRESSED_STATES as usize], targets),
                expanded_transition_count: COMPRESSED_STATES as usize * 26,
            }]);
        let wire = Arc::new(artifact_serde::build_huge_bytes(&original).unwrap());
        let mut packed = artifact_serde::from_fast_bytes_backed(&wire, Arc::clone(&wire), 0).unwrap();
        assert!(!packed.packed_compressed_transition_segments[0].overflow_indices.is_empty());
        packed.materialize_runtime_metadata_for_structural_mutation();
        let tail = packed.dfa.add_state();
        packed.dfa.overwrite_state_metadata(tail, live.clone(), live);
        packed.dfa.add_transition(tail, b'!', 0);
        assert!(artifact_serde::build_huge_bytes(&packed).is_none(),
            "an ordinary tail must exercise the non-suffix fallback");
        let fallback_wire = artifact_serde::to_segment_bytes(&packed);
        assert!(fallback_wire.len() < 40 * COMPRESSED_STATES as usize,
            "class rows must not expand into 26 byte edges each");
        let loaded = artifact_serde::from_fast_bytes(&fallback_wire).unwrap();
        assert_eq!(loaded.num_states(), packed.num_states());
        assert!(loaded.has_compressed_transition_segments());
        for state in [0, 1, 2, COMPRESSED_STATES, tail] {
            assert_eq!(loaded.possible_future_terminals(state), packed.possible_future_terminals(state));
            assert_eq!(loaded.matched_terminal_bitset(state), packed.matched_terminal_bitset(state));
            for byte in 0..=255 {
                assert_eq!(loaded.step(state, byte), packed.step(state, byte),
                    "packed overflow/tail mismatch state={state} byte={byte}");
            }
        }
    }

    #[test]
    fn huge_wire_roundtrips_compressed_transitions_with_wide_terminal_metadata() {
        const TERMINALS: usize = 300;
        const WIDE_TERMINAL: usize = 299;

        let mut dfa = DFA::new(4);
        dfa.ensure_group_capacity(TERMINALS);
        let mut group = U8Set::empty();
        group.insert(b'a');
        group.insert(b'b');
        dfa.set_group_u8set(WIDE_TERMINAL as u32, group);
        dfa.add_epsilon_transition(0, 1);

        for state in 0..4u32 {
            let mut futures = BitSet::new(TERMINALS);
            if state != 3 {
                futures.set(WIDE_TERMINAL);
            }
            let mut finalizers = BitSet::new(TERMINALS);
            if state == 3 {
                finalizers.set(WIDE_TERMINAL);
            }
            dfa.overwrite_state_metadata(state, finalizers, futures);
        }

        let mut byte_to_class = vec![u8::MAX; 256];
        byte_to_class[b'a' as usize] = 0;
        byte_to_class[b'b' as usize] = 1;
        let segment = CompressedTransitionSegment {
            state_offset: 1,
            state_count: 3,
            byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
            class_members: Arc::from(
                vec![vec![b'a'].into_boxed_slice(), vec![b'b'].into_boxed_slice()]
                    .into_boxed_slice(),
            ),
            row_offsets: Arc::from(vec![0u32, 1, 2, 2].into_boxed_slice()),
            entries: CompressedTransitionEntries::from_parts(vec![0, 1], vec![1, 2]),
            expanded_transition_count: 2,
        };
        let original = Tokenizer::from_parts_with_compressed_transitions(
            dfa,
            TERMINALS as u32,
            None,
            vec![segment],
        );

        let wire = artifact_serde::build_huge_bytes(&original).expect("wide TKS3 fixture");
        assert!(wire.starts_with(b"TKS3"));
        let backing = Arc::new(wire);
        let loaded = artifact_serde::from_fast_bytes_backed(
            backing.as_slice(),
            Arc::clone(&backing),
            0,
        )
        .expect("wide backed TKS3 roundtrip");

        assert_eq!(loaded.num_terminals(), TERMINALS as u32);
        assert_eq!(loaded.num_states(), original.num_states());
        assert!(loaded.has_packed_runtime_metadata());
        assert!(loaded.has_compressed_transition_segments());
        assert!(loaded.matched_terminal_bitset(3).contains(WIDE_TERMINAL));
        assert!(loaded.possible_future_terminals(1).contains(WIDE_TERMINAL));
        enumerate_bytes(b"abx", 3, |input| {
            for state in 0..original.num_states() {
                assert_eq!(
                    normalized_exec(&loaded, input, state),
                    normalized_exec(&original, input, state),
                    "wide TKS3 mismatch from state {state} on {input:?}",
                );
            }
        });

        let reencoded = artifact_serde::build_huge_bytes(&loaded)
            .expect("loaded wide TKS3 tokenizer must remain compactly serializable");
        assert!(reencoded.starts_with(b"TKS3"));
        let reloaded = artifact_serde::from_fast_bytes(&reencoded)
            .expect("re-encoded wide TKS3 roundtrip");
        assert_eq!(reloaded.num_terminals(), TERMINALS as u32);
        assert!(reloaded.matched_terminal_bitset(3).contains(WIDE_TERMINAL));
        assert!(reloaded.possible_future_terminals(1).contains(WIDE_TERMINAL));
    }

    #[test]
    fn fast_wire_with_packed_metadata_preserves_tkf2_transition_prefix() {
        let mut dfa = DFA::new(4);
        dfa.ensure_group_capacity(2);
        dfa.add_transition(0, b'a', 1);
        dfa.add_transition(0, b'b', 2);
        dfa.add_transition(1, b'b', 3);
        dfa.recompute_possible_futures();
        let tokenizer = Tokenizer::from_parts(dfa, 2, None);

        let tkf3 = artifact_serde::to_fast_bytes_with_packed_metadata(&tokenizer);
        let mut legacy_prefix = artifact_serde::to_fast_bytes(&tokenizer);
        let terminal_count = u32::from_le_bytes(legacy_prefix[4..8].try_into().unwrap()) as usize;
        let state_count = u32::from_le_bytes(legacy_prefix[8..12].try_into().unwrap()) as usize;
        let transition_count =
            u32::from_le_bytes(legacy_prefix[12..16].try_into().unwrap()) as usize;
        let state_id_width = legacy_prefix[28] as usize;
        let transition_end = 32
            + terminal_count * 32
            + (state_count + 1) * 4
            + transition_count
            + transition_count * state_id_width;
        legacy_prefix.truncate(transition_end);
        legacy_prefix[..4].copy_from_slice(b"TKF3");

        assert_eq!(&tkf3[..transition_end], legacy_prefix.as_slice());
    }

    #[test]
    fn fast_wire_with_packed_metadata_roundtrips_execution() {
        let mut dfa = DFA::new(4);
        dfa.ensure_group_capacity(2);
        dfa.add_transition(0, b'a', 1);
        dfa.add_transition(0, b'b', 2);
        dfa.add_transition(1, b'b', 3);
        let mut final_a = BitSet::new(2);
        final_a.set(0);
        dfa.overwrite_state_metadata(1, final_a, BitSet::new(2));
        let mut final_ab = BitSet::new(2);
        final_ab.set(1);
        dfa.overwrite_state_metadata(3, final_ab, BitSet::new(2));
        dfa.recompute_possible_futures();

        let original = Tokenizer::from_parts(dfa, 2, None);
        let wire = artifact_serde::to_fast_bytes_with_packed_metadata(&original);
        assert!(wire.starts_with(b"TKF3"));
        let loaded = artifact_serde::from_fast_bytes(&wire).expect("TKF3 tokenizer roundtrip");
        assert!(loaded.has_packed_runtime_transitions());
        assert!(loaded.has_packed_runtime_metadata());
        assert_eq!(loaded.num_states(), original.num_states());
        assert_eq!(loaded.transition_count(), original.transition_count());

        for input in [
            b"".as_slice(),
            b"a".as_slice(),
            b"b".as_slice(),
            b"ab".as_slice(),
            b"abb".as_slice(),
        ] {
            for state in 0..original.num_states() {
                assert_eq!(
                    normalized_exec(&loaded, input, state),
                    normalized_exec(&original, input, state),
                    "TKF3 mismatch from state {state} on {input:?}",
                );
            }
        }

        let mut unknown_flags = wire.clone();
        unknown_flags[artifact_serde::FAST_WIRE_FLAGS_OFFSET] |= 0x80;
        assert!(
            artifact_serde::from_fast_bytes(&unknown_flags).is_err(),
            "unknown TKF3 flags must fail closed",
        );

        let mut corrupted = wire;
        corrupted[8..12].copy_from_slice(&0u32.to_le_bytes());
        assert!(
            artifact_serde::from_fast_bytes(&corrupted).is_err(),
            "invalid TKF3 state count must fail closed",
        );
    }

    #[test]
    fn fast_wire_packed_transitions_preserve_multi_state_execution() {
        let mut dfa = DFA::new(3);
        dfa.ensure_group_capacity(2);
        dfa.add_transition(0, b'a', 1);
        dfa.add_transition(0, b'b', 2);
        let mut final_a = BitSet::new(2);
        final_a.set(0);
        dfa.overwrite_state_metadata(1, final_a, BitSet::new(2));
        let mut final_b = BitSet::new(2);
        final_b.set(1);
        dfa.overwrite_state_metadata(2, final_b, BitSet::new(2));
        dfa.recompute_possible_futures();

        let original = Tokenizer::from_parts(dfa, 2, None);
        let wire = artifact_serde::to_fast_bytes(&original);
        let loaded = artifact_serde::from_fast_bytes(&wire).expect("fast tokenizer roundtrip");
        assert!(loaded.has_packed_runtime_transitions());

        for input in [b"a".as_slice(), b"b".as_slice(), b"ab".as_slice()] {
            assert_eq!(
                normalized_exec(&loaded, input, loaded.initial_state()),
                normalized_exec(&original, input, original.initial_state()),
                "fast-wire execution mismatch on {input:?}",
            );
        }
    }

    #[test]
    fn disjoint_union_preserves_loaded_packed_transition_rows_without_materializing() {
        let mut dfa = DFA::new(3);
        dfa.ensure_group_capacity(2);
        dfa.add_transition(0, b'a', 1);
        dfa.add_transition(0, b'b', 2);
        let mut final_a = BitSet::new(2);
        final_a.set(0);
        dfa.overwrite_state_metadata(1, final_a, BitSet::new(2));
        let mut final_b = BitSet::new(2);
        final_b.set(1);
        dfa.overwrite_state_metadata(2, final_b, BitSet::new(2));
        dfa.recompute_possible_futures();

        let original = Tokenizer::from_parts(dfa, 2, None);
        let wire = artifact_serde::to_fast_bytes(&original);
        let loaded = artifact_serde::from_fast_bytes(&wire).expect("fast tokenizer roundtrip");
        assert!(loaded.packed_runtime_transitions.is_some());
        assert!(loaded.dfa.states()[0].transitions.is_empty());

        let (merged, offsets) = Tokenizer::disjoint_union_with_terminal_offsets(&[(&loaded, 0)]);
        let offset = offsets[0];
        assert!(merged.packed_runtime_transitions.is_none());
        assert_eq!(merged.packed_runtime_transition_segments.len(), 1);
        assert_eq!(merged.step(offset + loaded.initial_state(), b'a'), Some(offset + 1));
        assert_eq!(merged.step(offset + loaded.initial_state(), b'b'), Some(offset + 2));
        for input in [b"a".as_slice(), b"b".as_slice(), b"ab".as_slice()] {
            let (source_ends, source_matches) =
                normalized_exec(&loaded, input, loaded.initial_state());
            let expected = (
                source_ends
                    .into_iter()
                    .map(|state| offset + state)
                    .collect::<Vec<_>>(),
                source_matches
                    .into_iter()
                    .map(|(terminal, width, state)| (terminal, width, offset + state))
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                normalized_exec(&merged, input, offset + loaded.initial_state()),
                expected,
                "rebased packed execution mismatch on {input:?}",
            );
        }
    }

    #[test]
    fn fast_wire_roundtrip_preserves_nested_packed_transition_segments() {
        let mut dfa = DFA::new(3);
        dfa.ensure_group_capacity(2);
        dfa.add_transition(0, b'a', 1);
        dfa.add_transition(0, b'b', 2);
        let mut final_a = BitSet::new(2);
        final_a.set(0);
        dfa.overwrite_state_metadata(1, final_a, BitSet::new(2));
        let mut final_b = BitSet::new(2);
        final_b.set(1);
        dfa.overwrite_state_metadata(2, final_b, BitSet::new(2));
        dfa.recompute_possible_futures();

        let original = Tokenizer::from_parts(dfa, 2, None);
        let first_wire = artifact_serde::to_fast_bytes(&original);
        let loaded = artifact_serde::from_fast_bytes(&first_wire).expect("first fast tokenizer roundtrip");
        let (nested, offsets) = Tokenizer::disjoint_union_with_terminal_offsets(&[(&loaded, 0)]);
        let offset = offsets[0];
        assert_eq!(nested.packed_runtime_transition_segments.len(), 1);

        let layout = artifact_serde::fast_layout_for_write(&nested)
            .expect("ordinary packed segments should still support TKF2");
        let mut second_wire = vec![0u8; layout.len()];
        artifact_serde::write_fast_bytes_with_layout(&nested, layout, &mut second_wire)
            .expect("nested packed tokenizer should serialize");
        let roundtripped = artifact_serde::from_fast_bytes(&second_wire)
            .expect("nested fast tokenizer roundtrip");

        for input in [b"a".as_slice(), b"b".as_slice(), b"ab".as_slice()] {
            let (source_ends, source_matches) = normalized_exec(&loaded, input, loaded.initial_state());
            let expected = (
                source_ends.into_iter().map(|state| offset + state).collect::<Vec<_>>(),
                source_matches
                    .into_iter()
                    .map(|(terminal, width, state)| (terminal, width, offset + state))
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                normalized_exec(&roundtripped, input, offset + loaded.initial_state()),
                expected,
                "nested packed fast-wire mismatch on {input:?}",
            );
        }
    }

    #[test]
    fn serialized_terminal_language_equivalence_is_exact_without_exprs() {
        let left = serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ab")]));
        let equal = serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ab")]));
        let different_same_support =
            serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ba")]));

        assert_eq!(
            left.terminal_language_equivalent_bounded(0, &equal, 0, 64, 10_000),
            Some(true),
        );
        assert_eq!(
            left.terminal_language_equivalent_bounded(
                0,
                &different_same_support,
                0,
                64,
                10_000,
            ),
            Some(false),
        );
    }

    #[test]
    fn serialized_terminal_language_equivalence_budget_fails_closed() {
        let left = serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ab")]));
        let right = serialized_roundtrip(&tokenizer_from_exprs(vec![bytes(b"ab")]));
        assert_eq!(
            left.terminal_language_equivalent_bounded(0, &right, 0, 1, 10_000),
            None,
            "budget exhaustion must never be interpreted as equivalence",
        );
    }

    #[test]
    fn exact_terminal_observation_partition_matches_pairwise_bisimulation() {
        let tokenizers = vec![
            arbitrary_epsilon_l1_test_tokenizer(),
            tokenizer_from_exprs(vec![plus(bytes(b"a")), bytes(b"ab"), bytes(b"ba")]),
        ];

        for tokenizer in tokenizers {
            for terminal in 0..tokenizer.num_terminals() {
                let (classes, _, _) = tokenizer
                    .exact_terminal_observation_partition(terminal, 10_000, 1_000_000)
                    .expect("small test tokenizer must quotient within budget");
                let mut observed = BitSet::new(tokenizer.num_terminals() as usize);
                observed.set(terminal as usize);
                for left in 0..tokenizer.num_states() {
                    for right in 0..tokenizer.num_states() {
                        let exact = tokenizer
                            .state_sets_observation_equivalent_exact(
                                &[left],
                                &[right],
                                &observed,
                                10_000,
                            )
                            .expect("small pairwise proof must stay within budget");
                        assert_eq!(
                            classes[left as usize] == classes[right as usize],
                            exact,
                            "terminal {terminal}, raw states {left} and {right}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn scalar_exact_terminal_partition_never_aliases_uncovered_live_residuals() {
        let tokenizer = dispatch_prefix_tokenizer(true);
        assert!(tokenizer.has_scalar_deterministic_dispatch());
        let terminal = 0;
        let (classes, _, _) = tokenizer
            .exact_terminal_observation_partition(terminal, 10_000, 1_000_000)
            .expect("small scalar tokenizer must quotient within budget");
        let mut observed = BitSet::new(tokenizer.num_terminals() as usize);
        observed.set(terminal as usize);

        // State 4 is deliberately reset-unreachable but terminal-live.  It
        // must not be class zero merely because the fast scalar proof starts
        // from the certified dispatch component.
        assert!(tokenizer.state_live_for_terminal(4, terminal));
        assert_ne!(classes[4], 0);

        // The fast path may conservatively leave uncovered live residuals in
        // singleton classes, but every actual nonzero alias must still be an
        // exact arbitrary-continuation equivalence proof.
        for left in 0..tokenizer.num_states() {
            for right in 0..tokenizer.num_states() {
                if classes[left as usize] == 0
                    || classes[left as usize] != classes[right as usize]
                {
                    continue;
                }
                assert_eq!(
                    tokenizer
                        .state_sets_observation_equivalent_exact(
                            &[left],
                            &[right],
                            &observed,
                            10_000,
                        )
                        .expect("small pairwise proof must stay within budget"),
                    true,
                    "scalar alias for raw states {left} and {right} must be exact",
                );
            }
        }
    }

    fn normalized_exec(
        tokenizer: &Tokenizer,
        input: &[u8],
        start: u32,
    ) -> (Vec<u32>, Vec<(u32, usize, u32)>) {
        let result = tokenizer.execute_from_state_all_widths(input, start);
        let mut end_states = result.end_state.into_vec();
        end_states.sort_unstable();
        let mut matches = result
            .matches
            .into_iter()
            .map(|matched| (matched.id, matched.width, matched.end_state))
            .collect::<Vec<_>>();
        matches.sort_unstable();
        (end_states, matches)
    }

    fn enumerate_bytes(
        alphabet: &[u8],
        max_len: usize,
        mut visit: impl FnMut(&[u8]),
    ) {
        fn rec(
            alphabet: &[u8],
            remaining: usize,
            word: &mut Vec<u8>,
            visit: &mut impl FnMut(&[u8]),
        ) {
            visit(word);
            if remaining == 0 {
                return;
            }
            for &byte in alphabet {
                word.push(byte);
                rec(alphabet, remaining - 1, word, visit);
                word.pop();
            }
        }
        rec(alphabet, max_len, &mut Vec::new(), &mut visit);
    }

    fn semantic_exec(tokenizer: &Tokenizer, input: &[u8]) -> (bool, Vec<(u32, usize)>) {
        let result = tokenizer.execute_from_state_all_widths(input, tokenizer.initial_state());
        let mut matches = result
            .matches
            .into_iter()
            .map(|matched| (matched.id, matched.width))
            .collect::<Vec<_>>();
        matches.sort_unstable();
        matches.dedup();
        (!result.end_state.is_empty(), matches)
    }

    #[test]
    fn hybrid_virtual_unit_repeat_proxy_root_steps_exactly() {
        let mut tokenizer = tokenizer_from_exprs(vec![
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Class(U8Set::empty()),
        ]);
        tokenizer.isolate_start_state_and_drain_nullable_terminals();
        tokenizer
            .install_virtual_zero_min_unit_repeat_component(
                U8Set::single(b'a'),
                1_000_000_000,
                1,
            )
            .unwrap();
        let root = tokenizer
            .epsilon_closure_states(&[tokenizer.initial_state()])
            .into_iter()
            .find(|&state| {
                state != tokenizer.initial_state()
                    && tokenizer.possible_future_terminals(state).contains(1)
            })
            .expect("virtual repeat proxy root");
        let target = tokenizer.get_transition(root, b'a');
        assert_ne!(target, u32::MAX);
        assert_eq!(tokenizer.matched_terminals_iter(target).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn exact_terminal_observation_partition_declines_virtual_state_space() {
        let mut tokenizer = tokenizer_from_exprs(vec![
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Class(U8Set::empty()),
        ]);
        tokenizer.isolate_start_state_and_drain_nullable_terminals();
        tokenizer
            .install_virtual_zero_min_unit_repeat_component(
                U8Set::single(b'a'),
                1_000_000_000,
                1,
            )
            .unwrap();
        assert!(tokenizer.has_any_virtual_runtime());
        assert!(
            tokenizer
                .exact_terminal_observation_partition(1, 10_000, 1_000_000)
                .is_none(),
            "a raw-state observation quotient cannot certify lazily allocated virtual states",
        );
    }

    #[test]
    fn lazy_binary_repeat_intersection_matches_materialized_small_oracle() {
        use crate::automata::lexer::compile::compile_terminal_expr_dfa;
        use crate::automata::lexer::runtime_repeat_product::{
            VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
        };

        let left_body = Expr::Choice(vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"c".to_vec()),
        ]);
        let right_body = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"bc".to_vec()),
        ]);
        let left_max = 4usize;
        let right_max = 3usize;
        let expression = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(left_body.clone()),
                min: 0,
                max: Some(left_max),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(right_body.clone()),
                min: 0,
                max: Some(right_max),
            }),
        };

        let mut ordinary = tokenizer_from_exprs(vec![expression]);
        ordinary.isolate_start_state_and_drain_nullable_terminals();

        let left_dfa = Arc::new(compile_terminal_expr_dfa(&left_body));
        let right_dfa = Arc::new(compile_terminal_expr_dfa(&right_body));
        let mut virtualized = tokenizer_from_exprs(vec![Expr::U8Class(U8Set::empty())]);
        virtualized.isolate_start_state_and_drain_nullable_terminals();
        virtualized
            .install_virtual_binary_repeat_intersection_component(
                VirtualBinaryRepeatIntersectionDescriptor {
                    byte_support: U8Set::from_bytes(b"abc"),
                    left: VirtualBoundedRepeatSpec {
                        base_dfa: left_dfa,
                        min: 0,
                        max: left_max as u32,
                    },
                    right: VirtualBoundedRepeatSpec {
                        base_dfa: right_dfa,
                        min: 0,
                        max: right_max as u32,
                    },
                },
                0,
            )
            .unwrap();

        enumerate_bytes(b"abcx", 8, |input| {
            let (_, ordinary_matches) = semantic_exec(&ordinary, input);
            let (_, virtual_matches) = semantic_exec(&virtualized, input);
            assert_eq!(
                virtual_matches, ordinary_matches,
                "lazy repeat product changed exact matches on {input:?}",
            );
        });
        assert!(virtualized.virtual_binary_repeat_intersection_interned_state_count() < 128);
    }

    #[test]
    fn repeat_product_state_owner_index_dispatches_multiple_components_directly() {
        use crate::automata::lexer::compile::compile_terminal_expr_dfa;
        use crate::automata::lexer::runtime_repeat_product::{
            VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
        };

        let descriptor = |byte| {
            let base = Arc::new(compile_terminal_expr_dfa(&Expr::U8Seq(vec![byte])));
            VirtualBinaryRepeatIntersectionDescriptor {
                byte_support: U8Set::single(byte),
                left: VirtualBoundedRepeatSpec {
                    base_dfa: Arc::clone(&base),
                    min: 0,
                    max: 32,
                },
                right: VirtualBoundedRepeatSpec {
                    base_dfa: base,
                    min: 0,
                    max: 32,
                },
            }
        };

        let mut tokenizer = Tokenizer::from_parts(DFA::new(1), 2, None);
        tokenizer
            .install_virtual_binary_repeat_intersection_components(vec![
                (descriptor(b'a'), 0),
                (descriptor(b'b'), 1),
            ])
            .expect("two repeat-product components must install");

        let a_state = tokenizer.step(1, b'a').expect("first product must advance");
        let b_state = tokenizer.step(2, b'b').expect("second product must advance");
        assert_ne!(a_state, b_state);
        assert_eq!(
            tokenizer
                .virtual_repeat_runtime_for_state(a_state)
                .map(VirtualBinaryRepeatIntersectionRuntime::terminal),
            Some(0),
        );
        assert_eq!(
            tokenizer
                .virtual_repeat_runtime_for_state(b_state)
                .map(VirtualBinaryRepeatIntersectionRuntime::terminal),
            Some(1),
        );

        let aa_state = tokenizer
            .step(a_state, b'a')
            .expect("interleaved first product must remain owned");
        let bb_state = tokenizer
            .step(b_state, b'b')
            .expect("interleaved second product must remain owned");
        assert_eq!(
            tokenizer
                .virtual_repeat_runtime_for_state(aa_state)
                .map(VirtualBinaryRepeatIntersectionRuntime::terminal),
            Some(0),
        );
        assert_eq!(
            tokenizer
                .virtual_repeat_runtime_for_state(bb_state)
                .map(VirtualBinaryRepeatIntersectionRuntime::terminal),
            Some(1),
        );
    }

    #[test]
    fn lazy_binary_repeat_body_product_budget_fails_before_proxy_install() {
        use crate::automata::lexer::runtime_repeat_product::{
            VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
        };

        // 257^2 exceeds the exact body-product analysis budget. The installer
        // must reject this before appending any proxy root to the physical DFA.
        let descriptor = VirtualBinaryRepeatIntersectionDescriptor {
            byte_support: U8Set::from_bytes(b"a"),
            left: VirtualBoundedRepeatSpec {
                base_dfa: Arc::new(DFA::new(257)),
                min: 0,
                max: 10_000,
            },
            right: VirtualBoundedRepeatSpec {
                base_dfa: Arc::new(DFA::new(257)),
                min: 0,
                max: 10_000,
            },
        };
        let mut tokenizer = tokenizer_from_exprs(vec![Expr::U8Class(U8Set::empty())]);
        tokenizer.isolate_start_state_and_drain_nullable_terminals();
        let before_states = tokenizer.num_states();
        assert!(
            tokenizer
                .install_virtual_binary_repeat_intersection_component(descriptor, 0)
                .is_none(),
        );
        assert_eq!(tokenizer.num_states(), before_states);
        assert!(!tokenizer.has_virtual_binary_repeat_intersection());
    }

    #[test]
    fn nonzero_min_repeat_product_rejects_unsynchronized_manual_descriptor() {
        use crate::automata::lexer::compile::compile_terminal_expr_dfa;
        use crate::automata::lexer::runtime_repeat_product::{
            VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
        };

        let left = Arc::new(compile_terminal_expr_dfa(&Expr::U8Seq(b"a".to_vec())));
        let right = Arc::new(compile_terminal_expr_dfa(&Expr::U8Seq(b"aa".to_vec())));
        let descriptor = VirtualBinaryRepeatIntersectionDescriptor {
            byte_support: U8Set::single(b'a'),
            left: VirtualBoundedRepeatSpec {
                base_dfa: left,
                min: 4,
                max: 4,
            },
            right: VirtualBoundedRepeatSpec {
                base_dfa: right,
                min: 1,
                max: 1,
            },
        };
        let mut tokenizer = tokenizer_from_exprs(vec![Expr::U8Class(U8Set::empty())]);
        tokenizer.isolate_start_state_and_drain_nullable_terminals();
        let before_states = tokenizer.num_states();
        assert!(
            tokenizer
                .install_virtual_binary_repeat_intersection_component(descriptor, 0)
                .is_none()
        );
        assert_eq!(tokenizer.num_states(), before_states);
        assert!(!tokenizer.has_virtual_binary_repeat_intersection());
    }

    #[test]
    fn nonzero_min_repeat_product_rejects_empty_body_language() {
        use crate::automata::lexer::compile::compile_terminal_expr_dfa;
        use crate::automata::lexer::runtime_repeat_product::{
            VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
        };

        let empty_body = Expr::Seq(vec![
            Expr::U8Class(U8Set::empty()),
            Expr::U8Seq(b"a".to_vec()),
        ]);
        let base = Arc::new(compile_terminal_expr_dfa(&empty_body));
        assert!(!base.possible_future_group_ids(0).contains(0));
        let spec = VirtualBoundedRepeatSpec {
            base_dfa: Arc::clone(&base),
            min: 1,
            max: 10_000,
        };
        let descriptor = VirtualBinaryRepeatIntersectionDescriptor {
            byte_support: U8Set::single(b'a'),
            left: spec.clone(),
            right: spec,
        };
        let mut tokenizer = tokenizer_from_exprs(vec![Expr::U8Class(U8Set::empty())]);
        tokenizer.isolate_start_state_and_drain_nullable_terminals();
        let before_states = tokenizer.num_states();
        assert!(
            tokenizer
                .install_virtual_binary_repeat_intersection_component(descriptor, 0)
                .is_none()
        );
        assert_eq!(tokenizer.num_states(), before_states);
        assert!(!tokenizer.has_virtual_binary_repeat_intersection());
    }

    #[test]
    fn lazy_binary_repeat_mask_projection_is_exact_within_vocab_horizon() {
        use crate::automata::lexer::compile::compile_terminal_expr_dfa;
        use crate::automata::lexer::runtime_repeat_product::{
            VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
        };

        const HORIZON: usize = 3;
        let left_body = Expr::Choice(vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"c".to_vec()),
        ]);
        let right_body = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"bc".to_vec()),
        ]);
        let descriptor = VirtualBinaryRepeatIntersectionDescriptor {
            byte_support: U8Set::from_bytes(b"abc"),
            left: VirtualBoundedRepeatSpec {
                base_dfa: Arc::new(compile_terminal_expr_dfa(&left_body)),
                min: 0,
                max: 20,
            },
            right: VirtualBoundedRepeatSpec {
                base_dfa: Arc::new(compile_terminal_expr_dfa(&right_body)),
                min: 0,
                max: 17,
            },
        };
        let mut exact = tokenizer_from_exprs(vec![Expr::U8Class(U8Set::empty())]);
        exact.isolate_start_state_and_drain_nullable_terminals();
        exact
            .install_virtual_binary_repeat_intersection_component(descriptor, 0)
            .unwrap();
        let (mask, projection) = exact
            .virtual_binary_repeat_intersection_mask_tokenizer(HORIZON)
            .unwrap();

        let exact_root = exact
            .epsilon_closure_states(&[exact.initial_state()])
            .into_iter()
            .find(|&state| state != exact.initial_state())
            .unwrap();

        // Reach a representative collection of exact interior residuals. The
        // prefixes themselves may be longer than HORIZON; only each suffix
        // comparison is horizon-bounded.
        enumerate_bytes(b"abc", 8, |prefix| {
            let mut exact_state = exact_root;
            for &byte in prefix {
                let next = exact.get_transition(exact_state, byte);
                if next == u32::MAX {
                    return;
                }
                exact_state = next;
            }
            let mask_state = projection
                .project(exact_state)
                .expect("every exact lazy residual must have a static mask state");
            assert_eq!(
                exact.matched_terminal_bitset(exact_state),
                mask.matched_terminal_bitset(mask_state),
                "source finalizers differ after prefix {prefix:?}",
            );

            enumerate_bytes(b"abcx", HORIZON, |suffix| {
                let mut full = exact_state;
                let mut projected = mask_state;
                for (offset, &byte) in suffix.iter().enumerate() {
                    let full_next = exact.get_transition(full, byte);
                    let projected_next = mask.get_transition(projected, byte);
                    assert_eq!(
                        full_next == u32::MAX,
                        projected_next == u32::MAX,
                        "liveness differs after prefix {prefix:?}, suffix {suffix:?} at {offset}",
                    );
                    if full_next == u32::MAX {
                        break;
                    }
                    assert_eq!(
                        exact.matched_terminal_bitset(full_next),
                        mask.matched_terminal_bitset(projected_next),
                        "finalizers differ after prefix {prefix:?}, suffix {suffix:?} at {offset}",
                    );
                    assert_eq!(
                        exact.possible_future_terminals(full_next),
                        mask.possible_future_terminals(projected_next),
                        "futures differ after prefix {prefix:?}, suffix {suffix:?} at {offset}",
                    );
                    full = full_next;
                    projected = projected_next;
                }
            });
        });
    }

    #[test]
    fn synchronized_nonzero_min_repeat_projection_matches_materialized_oracle() {
        use crate::automata::lexer::compile::compile_terminal_expr_dfa;
        use crate::automata::lexer::runtime_repeat_product::{
            VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
        };

        const HORIZON: usize = 3;
        let body = Expr::Choice(vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"c".to_vec()),
        ]);
        let expression = Expr::Repeat {
            expr: Box::new(body.clone()),
            min: 6,
            max: Some(9),
        };
        let mut ordinary = tokenizer_from_exprs(vec![expression]);
        ordinary.isolate_start_state_and_drain_nullable_terminals();

        let base = Arc::new(compile_terminal_expr_dfa(&body));
        let spec = VirtualBoundedRepeatSpec {
            base_dfa: base,
            min: 6,
            max: 9,
        };
        let descriptor = VirtualBinaryRepeatIntersectionDescriptor {
            byte_support: U8Set::from_bytes(b"abc"),
            left: spec.clone(),
            right: spec,
        };
        let mut exact = tokenizer_from_exprs(vec![Expr::U8Class(U8Set::empty())]);
        exact.isolate_start_state_and_drain_nullable_terminals();
        exact
            .install_virtual_binary_repeat_intersection_component(descriptor, 0)
            .unwrap();
        let (mask, projection) = exact
            .virtual_binary_repeat_intersection_mask_tokenizer(HORIZON)
            .unwrap();

        enumerate_bytes(b"abcx", 10, |input| {
            let (_, ordinary_matches) = semantic_exec(&ordinary, input);
            let (_, exact_matches) = semantic_exec(&exact, input);
            assert_eq!(
                exact_matches, ordinary_matches,
                "nonzero-min lazy repeat changed exact matches on {input:?}",
            );
        });

        let exact_root = exact
            .epsilon_closure_states(&[exact.initial_state()])
            .into_iter()
            .find(|&state| state != exact.initial_state())
            .unwrap();
        enumerate_bytes(b"abc", 8, |prefix| {
            let mut exact_state = exact_root;
            for &byte in prefix {
                let next = exact.get_transition(exact_state, byte);
                if next == u32::MAX {
                    return;
                }
                exact_state = next;
            }
            let mask_state = projection
                .project(exact_state)
                .expect("every synchronized exact residual must have a finite projection");
            assert_eq!(
                exact.matched_terminal_bitset(exact_state),
                mask.matched_terminal_bitset(mask_state),
                "source finalizers differ after prefix {prefix:?}",
            );
            assert_eq!(
                exact.possible_future_terminals(exact_state),
                mask.possible_future_terminals(mask_state),
                "source futures differ after prefix {prefix:?}",
            );

            enumerate_bytes(b"abcx", HORIZON, |suffix| {
                let mut full = exact_state;
                let mut projected = mask_state;
                for (offset, &byte) in suffix.iter().enumerate() {
                    let full_next = exact.get_transition(full, byte);
                    let projected_next = mask.get_transition(projected, byte);
                    assert_eq!(
                        full_next == u32::MAX,
                        projected_next == u32::MAX,
                        "liveness differs after prefix {prefix:?}, suffix {suffix:?} at {offset}",
                    );
                    if full_next == u32::MAX {
                        break;
                    }
                    assert_eq!(
                        exact.matched_terminal_bitset(full_next),
                        mask.matched_terminal_bitset(projected_next),
                        "finalizers differ after prefix {prefix:?}, suffix {suffix:?} at {offset}",
                    );
                    assert_eq!(
                        exact.possible_future_terminals(full_next),
                        mask.possible_future_terminals(projected_next),
                        "futures differ after prefix {prefix:?}, suffix {suffix:?} at {offset}",
                    );
                    full = full_next;
                    projected = projected_next;
                }
            });
        });
    }

    #[test]
    fn lazy_binary_repeat_mask_projection_declines_quadratic_horizon_before_allocation() {
        use crate::automata::lexer::compile::compile_terminal_expr_dfa;
        use crate::automata::lexer::runtime_repeat_product::{
            VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
        };

        let body = Expr::U8Seq(b"a".to_vec());
        let base = Arc::new(compile_terminal_expr_dfa(&body));
        let descriptor = VirtualBinaryRepeatIntersectionDescriptor {
            byte_support: U8Set::from_bytes(b"a"),
            left: VirtualBoundedRepeatSpec {
                base_dfa: Arc::clone(&base),
                min: 0,
                max: 1_000_000_000,
            },
            right: VirtualBoundedRepeatSpec {
                base_dfa: base,
                min: 0,
                max: 900_000_000,
            },
        };
        let mut exact = tokenizer_from_exprs(vec![Expr::U8Class(U8Set::empty())]);
        exact.isolate_start_state_and_drain_nullable_terminals();
        exact
            .install_virtual_binary_repeat_intersection_component(descriptor, 0)
            .unwrap();

        assert!(
            exact
                .virtual_binary_repeat_intersection_mask_tokenizer(600)
                .is_none(),
            "a pathological finite-token horizon must decline the optional quotient instead of eagerly seeding its quadratic boundary grid",
        );
        assert!(
            exact.virtual_binary_repeat_intersection_interned_state_count() <= 1,
            "declining the quotient must not materialize exact residual states",
        );
    }

    #[test]
    fn virtual_unit_repeat_mask_projection_is_exact_for_every_bounded_word() {
        use crate::automata::lexer::compile::build_virtual_zero_min_unit_repeat_tokenizer;

        const FULL_MAX: usize = 20;
        const HORIZON: usize = 3;
        let expression = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
            min: 0,
            max: Some(FULL_MAX),
        };
        let full = build_virtual_zero_min_unit_repeat_tokenizer(&[expression]).unwrap();
        let (mask, projection) = full
            .virtual_zero_min_unit_repeat_mask_tokenizer(HORIZON)
            .unwrap();

        assert_eq!(full.num_states(), 1, "the exact billion-style counter is not materialized");
        assert_eq!(mask.num_states(), HORIZON as u32 + 3);

        for full_state in 0..=FULL_MAX as u32 {
            let mask_state = projection.project(full_state).unwrap();
            assert_eq!(
                full.matched_terminal_bitset(full_state),
                mask.matched_terminal_bitset(mask_state),
                "source finalizers differ at exact state {full_state}",
            );
            assert_eq!(
                full.possible_future_terminals(full_state),
                mask.possible_future_terminals(mask_state),
                "source futures differ at exact state {full_state}",
            );

            enumerate_bytes(b"ab", HORIZON, |input| {
                let mut full_cursor = full_state;
                let mut mask_cursor = mask_state;
                for (offset, &byte) in input.iter().enumerate() {
                    let full_next = full.get_transition(full_cursor, byte);
                    let mask_next = mask.get_transition(mask_cursor, byte);
                    assert_eq!(
                        full_next == u32::MAX,
                        mask_next == u32::MAX,
                        "liveness differs from exact state {full_state} on {input:?} at {offset}",
                    );
                    if full_next == u32::MAX {
                        break;
                    }
                    assert_eq!(
                        full.matched_terminal_bitset(full_next),
                        mask.matched_terminal_bitset(mask_next),
                        "finalizers differ from exact state {full_state} on {input:?} at {offset}",
                    );
                    assert_eq!(
                        full.possible_future_terminals(full_next),
                        mask.possible_future_terminals(mask_next),
                        "futures differ from exact state {full_state} on {input:?} at {offset}",
                    );
                    full_cursor = full_next;
                    mask_cursor = mask_next;
                }
            });
        }

        let billion = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
            min: 0,
            max: Some(1_000_000_000),
        };
        let billion = build_virtual_zero_min_unit_repeat_tokenizer(&[billion]).unwrap();
        let (billion_mask, _) = billion
            .virtual_zero_min_unit_repeat_mask_tokenizer(HORIZON)
            .unwrap();
        assert_eq!(billion.num_states(), 1);
        assert_eq!(billion_mask.num_states(), HORIZON as u32 + 3);
    }

    #[test]
    fn virtual_nonzero_min_unit_repeat_projection_matches_exact_counter() {
        use crate::automata::lexer::compile::build_virtual_unit_repeat_tokenizer;

        const FULL_MIN: usize = 8;
        const FULL_MAX: usize = 20;
        const HORIZON: usize = 3;
        let expression = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
            min: FULL_MIN,
            max: Some(FULL_MAX),
        };
        let full = build_virtual_unit_repeat_tokenizer(&[expression]).unwrap();
        let (mask, projection) = full.virtual_unit_repeat_mask_tokenizer(HORIZON).unwrap();

        assert_eq!(full.num_states(), 1, "the exact counter must stay arithmetic");
        assert!(projection.deep_lower_state().is_some());
        assert!(projection.interior_state().is_some());

        for full_state in 0..=FULL_MAX as u32 {
            let mask_state = projection.project(full_state).unwrap();
            assert_eq!(
                full.matched_terminal_bitset(full_state),
                mask.matched_terminal_bitset(mask_state),
                "source finalizers differ at exact state {full_state}",
            );
            assert_eq!(
                full.possible_future_terminals(full_state),
                mask.possible_future_terminals(mask_state),
                "source futures differ at exact state {full_state}",
            );

            enumerate_bytes(b"ab", HORIZON, |input| {
                let mut full_cursor = full_state;
                let mut mask_cursor = mask_state;
                for (offset, &byte) in input.iter().enumerate() {
                    let full_next = full.get_transition(full_cursor, byte);
                    let mask_next = mask.get_transition(mask_cursor, byte);
                    assert_eq!(
                        full_next == u32::MAX,
                        mask_next == u32::MAX,
                        "liveness differs from exact state {full_state} on {input:?} at {offset}",
                    );
                    if full_next == u32::MAX {
                        break;
                    }
                    assert_eq!(
                        full.matched_terminal_bitset(full_next),
                        mask.matched_terminal_bitset(mask_next),
                        "finalizers differ from exact state {full_state} on {input:?} at {offset}",
                    );
                    assert_eq!(
                        full.possible_future_terminals(full_next),
                        mask.possible_future_terminals(mask_next),
                        "futures differ from exact state {full_state} on {input:?} at {offset}",
                    );
                    full_cursor = full_next;
                    mask_cursor = mask_next;
                }
            });
        }

        let counts = projection.multiplicities();
        assert_eq!(counts.iter().sum::<usize>(), FULL_MAX + 1);
        let unique = projection.unique_full_states();
        for (mask_state, &multiplicity) in counts.iter().enumerate() {
            assert_eq!(
                unique[mask_state] != u32::MAX,
                multiplicity == 1,
                "unique-full-state metadata disagrees at mask state {mask_state}",
            );
        }
    }

    #[test]
    fn materialized_certified_bounded_code_terminal_does_not_require_virtual_owner_metadata() {
        let expression = Expr::Intersect {
            expr: Box::new(Expr::Seq(vec![
                bytes(b"\""),
                Expr::Repeat {
                    expr: Box::new(bytes(b"a")),
                    min: 0,
                    max: None,
                },
                bytes(b"\""),
            ])),
            intersect: Box::new(Expr::Seq(vec![
                bytes(b"\""),
                Expr::Repeat {
                    expr: Box::new(bytes(b"a")),
                    min: 0,
                    max: Some(128),
                },
                bytes(b"\""),
            ])),
        };
        assert!(
            crate::automata::lexer::compile::expression_supports_bounded_code_residual_runtime(
                &expression,
            ),
            "test expression must be eligible for the optional bounded-code residual representation",
        );
        let original = tokenizer_from_exprs(vec![expression.clone()]);
        assert!(!original.has_any_virtual_runtime());
        assert!(
            (0..original.num_states())
                .any(|state| original.state_finalizers(state).contains(0)),
            "the compatibility case must contain the terminal physically",
        );

        let mut loaded = serialized_roundtrip(&original);
        loaded
            .restore_terminal_exprs_with_virtual_runtime_metadata(
                Some(vec![expression]),
                &[],
                false,
            )
            .expect("an older fully-materialized certifiable terminal must not require a residual sidecar");
        assert!(!loaded.has_any_virtual_runtime());
    }

    #[test]
    fn exact_unit_runtime_restore_rejects_corrupt_serialized_futures() {
        use crate::automata::lexer::compile::build_virtual_zero_min_unit_repeat_tokenizer;

        let expression = Expr::Repeat {
            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
            min: 0,
            max: Some(10_000),
        };
        let original = build_virtual_zero_min_unit_repeat_tokenizer(&[expression.clone()])
            .expect("standalone unit repeat should use the arithmetic runtime");
        let metadata = original.virtual_runtime_metadata();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].kind, VirtualTokenizerRuntimeKind::UnitRepeat);

        let mut loaded = serialized_roundtrip(&original);
        // Simulate a decodable but inconsistent artifact. Exact restoration
        // must validate this physical row before attaching the sidecar, rather
        // than reading the reconstructed runtime's own future set.
        loaded.dfa.overwrite_state_metadata(
            loaded.start_state(),
            BitSet::new(1),
            BitSet::new(1),
        );
        let error = loaded
            .restore_terminal_exprs_with_virtual_runtime_metadata(
                Some(vec![expression]),
                &metadata,
                false,
            )
            .unwrap_err();
        assert!(
            error.contains("inconsistent future metadata"),
            "unexpected restoration error: {error}",
        );
        assert!(
            loaded.virtual_unit_repeat.is_none(),
            "failed exact restoration must not leave a reconstructed sidecar installed",
        );
    }

    #[test]
    fn residual_runtime_restore_migrates_legacy_exact_dead_root_futures() {
        let expression = Expr::Intersect {
            expr: Box::new(Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(bytes(b"a")),
                    min: 0,
                    max: Some(10_000),
                },
                bytes(b"b"),
            ])),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 0,
                max: None,
            }),
        };
        let mut original = Tokenizer::from_parts(DFA::new(1), 1, None);
        original
            .install_virtual_residual_components(vec![(expression.clone(), 0)])
            .expect("dead Boolean residual must install conservatively");
        let metadata = original.virtual_runtime_metadata();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].kind, VirtualTokenizerRuntimeKind::ResidualExpr);
        let root = metadata[0].root_state;
        assert!(original.state_futures(root).contains(0));
        assert_eq!(original.exact_dynamic_state_has_future(root).unwrap(), false);

        let mut loaded = serialized_roundtrip(&original);
        assert!(!loaded.has_packed_runtime_metadata());
        // Pre-conservative artifacts stored the exact-dead root as future=false
        // and consequently omitted that terminal from the reset-state future
        // set as well.
        let root_finalizers = loaded.state_finalizers(root).clone();
        loaded
            .dfa
            .overwrite_state_metadata(root, root_finalizers, BitSet::new(1));
        let start = loaded.start_state();
        let start_finalizers = loaded.state_finalizers(start).clone();
        loaded
            .dfa
            .overwrite_state_metadata(start, start_finalizers, BitSet::new(1));

        loaded
            .restore_terminal_exprs_with_virtual_runtime_metadata(
                Some(vec![expression]),
                &metadata,
                true,
            )
            .expect("exact-dead legacy residual metadata must migrate");
        assert!(loaded.has_virtual_residual_runtime());
        assert!(loaded.state_futures(root).contains(0));
        assert!(loaded.state_futures(start).contains(0));
        assert_eq!(loaded.exact_dynamic_state_has_future(root).unwrap(), false);
    }

    #[test]
    fn residual_runtime_restore_rejects_exact_dead_root_without_legacy_provenance() {
        let expression = Expr::Intersect {
            expr: Box::new(Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(bytes(b"a")),
                    min: 0,
                    max: Some(10_000),
                },
                bytes(b"b"),
            ])),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 0,
                max: None,
            }),
        };
        let mut original = Tokenizer::from_parts(DFA::new(1), 1, None);
        original
            .install_virtual_residual_components(vec![(expression.clone(), 0)])
            .expect("dead Boolean residual must install conservatively");
        let metadata = original.virtual_runtime_metadata();
        let root = metadata[0].root_state;

        let mut loaded = serialized_roundtrip(&original);
        let root_finalizers = loaded.state_finalizers(root).clone();
        loaded
            .dfa
            .overwrite_state_metadata(root, root_finalizers, BitSet::new(1));
        let start = loaded.start_state();
        let start_finalizers = loaded.state_finalizers(start).clone();
        loaded
            .dfa
            .overwrite_state_metadata(start, start_finalizers, BitSet::new(1));

        let error = loaded
            .restore_terminal_exprs_with_virtual_runtime_metadata(
                Some(vec![expression]),
                &metadata,
                false,
            )
            .unwrap_err();
        assert!(
            error.contains("inconsistent future metadata"),
            "current-version exact-dead corruption must not be mistaken for legacy metadata: {error}",
        );
        assert!(loaded.virtual_residuals.is_empty());
    }

    #[test]
    fn residual_runtime_restore_rejects_missing_future_for_live_root() {
        let expression = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::Choice(vec![bytes(b"a"), bytes(b"aa")])),
                min: 0,
                max: Some(10_000),
            },
            bytes(b"z"),
        ]);
        let mut original = Tokenizer::from_parts(DFA::new(1), 1, None);
        original
            .install_virtual_residual_components(vec![(expression.clone(), 0)])
            .expect("live residual must install");
        let metadata = original.virtual_runtime_metadata();
        let root = metadata[0].root_state;
        assert_eq!(original.exact_dynamic_state_has_future(root).unwrap(), true);

        let mut loaded = serialized_roundtrip(&original);
        let root_finalizers = loaded.state_finalizers(root).clone();
        loaded
            .dfa
            .overwrite_state_metadata(root, root_finalizers, BitSet::new(1));
        let start = loaded.start_state();
        let start_finalizers = loaded.state_finalizers(start).clone();
        loaded
            .dfa
            .overwrite_state_metadata(start, start_finalizers, BitSet::new(1));

        let error = loaded
            .restore_terminal_exprs_with_virtual_runtime_metadata(
                Some(vec![expression]),
                &metadata,
                false,
            )
            .unwrap_err();
        assert!(
            error.contains("inconsistent future metadata"),
            "unexpected restoration error: {error}",
        );
        assert!(loaded.virtual_residuals.is_empty());
    }

    #[test]
    fn grouped_summary_execution_matches_independent_scans() {
        let tokenizer = tokenizer_from_exprs(vec![
            bytes(b"ab"),
            plus(bytes(b"a")),
            bytes(b"ba"),
        ]);
        let starts = (0..tokenizer.num_states()).collect::<Vec<_>>();

        enumerate_bytes(b"abx", 4, |input| {
            let groups = tokenizer.execute_summary_groups_from_states(input, &starts);
            let mut grouped_by_start = std::collections::BTreeMap::new();
            for (end_states, matches, support) in groups {
                for start in support {
                    assert!(
                        grouped_by_start
                            .insert(start, (end_states.clone(), matches.clone()))
                            .is_none(),
                        "start state {start} appeared in more than one summary group for {input:?}",
                    );
                }
            }
            assert_eq!(grouped_by_start.len(), starts.len());
            for &start in &starts {
                assert_eq!(
                    grouped_by_start.get(&start),
                    Some(&tokenizer.execute_summary_from_state(input, start)),
                    "grouped residual scan differs for start state {start}, input {input:?}",
                );
            }
        });
    }

    #[test]
    fn disjoint_union_preserves_every_source_residual_and_unions_resets() {
        let left = tokenizer_from_exprs(vec![bytes(b"ab"), plus(bytes(b"x"))]);
        let right = tokenizer_from_exprs(vec![bytes(b"bc"), plus(bytes(b"y"))]);
        let right_terminal_offset = left.num_terminals();
        let (merged, state_offsets) = Tokenizer::disjoint_union_with_terminal_offsets(&[
            (&left, 0),
            (&right, right_terminal_offset),
        ]);
        assert_eq!(state_offsets.len(), 2);

        enumerate_bytes(b"abcxy", 3, |input| {
            for (source, terminal_offset, state_offset) in [
                (&left, 0u32, state_offsets[0]),
                (&right, right_terminal_offset, state_offsets[1]),
            ] {
                for source_state in 0..source.num_states() {
                    let (source_end, source_matches) =
                        normalized_exec(source, input, source_state);
                    let expected_end = source_end
                        .into_iter()
                        .map(|state| state_offset + state)
                        .collect::<Vec<_>>();
                    let expected_matches = source_matches
                        .into_iter()
                        .map(|(terminal, width, state)| {
                            (terminal_offset + terminal, width, state_offset + state)
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        normalized_exec(&merged, input, state_offset + source_state),
                        (expected_end, expected_matches),
                        "residual mismatch from source state {source_state} on {input:?}",
                    );
                }
            }

            let mut expected_end = Vec::new();
            let mut expected_matches = Vec::new();
            for (source, terminal_offset, state_offset) in [
                (&left, 0u32, state_offsets[0]),
                (&right, right_terminal_offset, state_offsets[1]),
            ] {
                let (end, matches) = normalized_exec(source, input, source.start_state());
                expected_end.extend(end.into_iter().map(|state| state_offset + state));
                expected_matches.extend(matches.into_iter().map(|(terminal, width, state)| {
                    (terminal_offset + terminal, width, state_offset + state)
                }));
            }
            if input.is_empty() {
                // The fresh epsilon dispatcher is a real physical state but
                // contributes no finalizer of its own. It remains in the empty
                // execution closure and disappears after the first byte.
                expected_end.push(merged.start_state());
            }
            expected_end.sort_unstable();
            expected_end.dedup();
            expected_matches.sort_unstable();
            expected_matches.dedup();
            assert_eq!(
                normalized_exec(&merged, input, merged.start_state()),
                (expected_end, expected_matches),
                "reset-union mismatch on {input:?}",
            );
        });
    }

    #[test]
    fn full_determinization_accepts_disjoint_union_local_metadata_widths() {
        let left = tokenizer_from_exprs(vec![bytes(b"ab"), bytes(b"ax")]);
        let right = tokenizer_from_exprs(vec![bytes(b"ab")]);
        let (merged, _) = Tokenizer::disjoint_union_with_terminal_offsets(&[
            (&left, 0),
            (&right, left.num_terminals()),
        ]);
        assert!(merged.has_epsilon_transitions());

        let built = merged
            .try_full_determinization(128, 4_096)
            .expect("small disjoint union must determinize across local metadata widths");
        assert!(!built.tokenizer.has_epsilon_transitions());

        enumerate_bytes(b"abx", 3, |input| {
            let source = merged.execute_from_state(input, merged.initial_state());
            let product = built
                .tokenizer
                .execute_from_state(input, built.tokenizer.initial_state());
            let normalize = |matches: Vec<TokenizerMatch>| {
                let mut matches = matches
                    .into_iter()
                    .map(|matched| (matched.id, matched.width))
                    .collect::<Vec<_>>();
                matches.sort_unstable();
                matches.dedup();
                matches
            };
            assert_eq!(normalize(product.matches), normalize(source.matches));
        });
    }

    #[test]
    fn full_determinization_refuses_virtual_residual_state_space() {
        let mut tokenizer = Tokenizer::from_parts(DFA::new(1), 1, None);
        let expression = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::Choice(vec![bytes(b"a"), bytes(b"aa")])),
                min: 0,
                max: Some(1_000_000_000),
            },
            bytes(b"z"),
        ]);
        tokenizer
            .install_virtual_residual_components(vec![(expression, 0)])
            .expect("general residual component must install");
        assert!(tokenizer.has_epsilon_transitions());
        assert!(tokenizer.has_virtual_residual_runtime());
        assert!(
            tokenizer.try_full_determinization(128, 4_096).is_none(),
            "physical subset construction must not discard virtual residual states",
        );
    }

    #[test]
    fn virtual_residual_mask_projection_matches_one_token_observations() {
        const HORIZON: usize = 4;
        const MAX: usize = 40;
        let body = Expr::U8Class(U8Set::from_bytes(b"ab"));
        let envelope = Expr::Seq(vec![
            bytes(b"\""),
            Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 1,
                max: Some(MAX),
            },
            bytes(b"\""),
        ]);
        let pattern = Expr::Seq(vec![
            bytes(b"\"a"),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: None,
            },
            bytes(b"\""),
        ]);
        let expression = Expr::Intersect {
            expr: Box::new(envelope),
            intersect: Box::new(pattern),
        };
        let mut exact = Tokenizer::from_parts(DFA::new(1), 1, None);
        exact
            .install_virtual_residual_components_preserving_oracle_coordinates(vec![(expression, 0)])
            .expect("bounded-code residual runtime must install");
        assert_eq!(exact.virtual_residual_bounded_code_liveness_oracle_count(), 1);
        assert_eq!(
            exact.virtual_residual_mask_projection_dense_state_work(HORIZON),
            // Includes the future acceptance witness after the token horizon.
            Some(112),
        );
        let (mask, projections) = exact
            .virtual_residuals_mask_tokenizer(HORIZON)
            .expect("bounded-code residual must admit a finite mask projection");
        assert_eq!(projections.len(), 1);
        assert!(mask.num_states() < 2_000, "small oracle should stay compact enough for the test");
        let projection = &projections[0];
        let root = exact
            .epsilon_closure_states(&[exact.start_state()])
            .into_iter()
            .find(|&state| state != exact.start_state())
            .unwrap();

        // Every exact state reachable in this finite oracle must have a
        // projection, not merely the hand-picked count samples below. This
        // protects the symbolic source-root construction against dropping a
        // valid deep token-boundary coordinate.
        let mut seen = BTreeSet::from([root]);
        let mut queue = VecDeque::from([root]);
        while let Some(state) = queue.pop_front() {
            assert!(
                projection.project(state).is_some(),
                "reachable exact residual state {state} has no finite projection",
            );
            for byte in 0u16..=255 {
                let next = exact.get_transition(state, byte as u8);
                if next != u32::MAX && seen.insert(next) {
                    queue.push_back(next);
                }
            }
        }

        let mut starting_states = vec![root];
        for count in [1usize, 2, 5, 12, 30, 35, 38, 39] {
            let mut state = root;
            state = exact.get_transition(state, b'\"');
            assert_ne!(state, u32::MAX);
            for index in 0..count {
                state = exact.get_transition(state, if index == 0 { b'a' } else { b'b' });
                assert_ne!(state, u32::MAX, "count={count} index={index}");
            }
            starting_states.push(state);
        }

        for full_start in starting_states {
            let mask_start = projection
                .project(full_start)
                .expect("every certified exact residual must project");
            assert_eq!(
                exact.matched_terminal_bitset(full_start),
                mask.matched_terminal_bitset(mask_start),
                "source finalizers differ at exact state {full_start}",
            );
            assert_eq!(
                exact.possible_future_terminals(full_start),
                mask.possible_future_terminals(mask_start),
                "source futures differ at exact state {full_start}",
            );

            enumerate_bytes(b"ab\"x", HORIZON, |suffix| {
                let mut full = full_start;
                let mut finite = mask_start;
                for (offset, &byte) in suffix.iter().enumerate() {
                    let full_next = exact.get_transition(full, byte);
                    let finite_next = mask.get_transition(finite, byte);
                    assert_eq!(
                        full_next == u32::MAX,
                        finite_next == u32::MAX,
                        "liveness differs from exact state {full_start} on {suffix:?} at {offset}",
                    );
                    if full_next == u32::MAX {
                        break;
                    }
                    assert_eq!(
                        exact.matched_terminal_bitset(full_next),
                        mask.matched_terminal_bitset(finite_next),
                        "finalizers differ from exact state {full_start} on {suffix:?} at {offset}",
                    );
                    assert_eq!(
                        exact.possible_future_terminals(full_next),
                        mask.possible_future_terminals(finite_next),
                        "futures differ from exact state {full_start} on {suffix:?} at {offset}",
                    );
                    // Runtime commit reprojects at token boundaries. The finite
                    // byte-walk endpoint need not be the same raw mask state,
                    // but it must carry the same observation as the independently
                    // projected exact endpoint.
                    let reprojected = projection.project(full_next).unwrap();
                    assert_eq!(
                        mask.matched_terminal_bitset(finite_next),
                        mask.matched_terminal_bitset(reprojected),
                    );
                    assert_eq!(
                        mask.possible_future_terminals(finite_next),
                        mask.possible_future_terminals(reprojected),
                    );
                    full = full_next;
                    finite = finite_next;
                }
            });
        }
    }

    #[test]
    fn prepared_virtual_residual_install_matches_direct_dynamic_install() {
        let body = Expr::U8Class(U8Set::from_bytes(b"ab"));
        let envelope = Expr::Seq(vec![
            bytes(b"\""),
            Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 0,
                max: Some(4),
            },
            bytes(b"\""),
        ]);
        let pattern = Expr::Seq(vec![
            bytes(b"\"a"),
            Expr::Repeat {
                expr: Box::new(body),
                min: 0,
                max: None,
            },
            bytes(b"\""),
        ]);
        let expression = Expr::Intersect {
            expr: Box::new(envelope),
            intersect: Box::new(pattern),
        };

        let mut direct = Tokenizer::from_parts(DFA::new(1), 1, None);
        direct
            .install_virtual_residual_components(vec![(expression.clone(), 0)])
            .expect("direct dynamic residual must install");

        let prepared = Tokenizer::prepare_virtual_residual_components(vec![(expression, 0)])
            .expect("bounded-code residual must prepare");
        let mut overlapped = Tokenizer::from_parts(DFA::new(1), 1, None);
        overlapped
            .install_prepared_virtual_residual_components(prepared)
            .expect("prepared dynamic residual must install");

        assert_eq!(direct.virtual_runtime_metadata(), overlapped.virtual_runtime_metadata());
        assert_eq!(
            direct.virtual_residual_bounded_code_liveness_oracle_count(),
            overlapped.virtual_residual_bounded_code_liveness_oracle_count(),
        );
        enumerate_bytes(b"ab\"", 4, |input| {
            assert_eq!(
                normalized_exec(&direct, input, direct.start_state()),
                normalized_exec(&overlapped, input, overlapped.start_state()),
                "prepared residual runtime differs on {input:?}",
            );
        });
    }

    #[test]
    fn virtual_residual_mask_projection_preserves_fast_loaded_physical_futures() {
        const HORIZON: usize = 4;
        let ordinary = bytes(b"<");
        let residual = Expr::Intersect {
            expr: Box::new(Expr::Seq(vec![
                bytes(b"\""),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                    min: 0,
                    max: None,
                },
                bytes(b"\""),
            ])),
            intersect: Box::new(Expr::Seq(vec![
                bytes(b"\""),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                    min: 1,
                    max: Some(128),
                },
                bytes(b"\""),
            ])),
        };
        let expressions = vec![ordinary.clone(), residual.clone()];

        // Build terminal 0 physically and reserve terminal 1 for the exact
        // residual runtime, matching the hybrid dynamic-tokenizer layout.
        let mut exact = tokenizer_from_exprs(vec![
            ordinary,
            Expr::U8Class(U8Set::empty()),
        ]);
        exact
            .restore_terminal_exprs_without_virtual_runtime(Some(expressions.clone()))
            .unwrap();
        exact
            .install_virtual_residual_components(vec![(residual, 1)])
            .expect("bounded-code residual runtime must install");
        let metadata = exact.virtual_runtime_metadata();
        assert_eq!(metadata.len(), 1);
        assert!(exact.possible_future_terminals(exact.start_state()).contains(0));

        // Current fast loads keep ordinary byte transitions and observation
        // metadata packed. Projection construction must structurally replace
        // the residual proxy without losing the packed physical language.
        let wire = artifact_serde::to_fast_bytes(&exact);
        let mut loaded = artifact_serde::from_fast_bytes(&wire).expect("fast tokenizer roundtrip");
        assert!(loaded.has_packed_runtime_transitions());
        assert!(loaded.has_packed_runtime_metadata());
        loaded
            .restore_terminal_exprs_with_virtual_runtime_metadata(
                Some(expressions),
                &metadata,
                false,
            )
            .expect("residual runtime must restore from explicit metadata");
        assert!(loaded.possible_future_terminals(loaded.start_state()).contains(0));

        let (mask, projections) = loaded
            .virtual_residuals_mask_tokenizer(HORIZON)
            .expect("fast-loaded residual tokenizer must admit a finite mask projection");
        assert_eq!(projections.len(), 1);
        assert!(
            mask.possible_future_terminals(mask.start_state()).contains(0),
            "projection must preserve ordinary physical futures stored in packed transition rows",
        );
        assert_eq!(
            normalized_exec(&mask, b"<", mask.start_state()),
            normalized_exec(&loaded, b"<", loaded.start_state()),
            "projection changed the ordinary physical terminal language",
        );
    }

    #[test]
    fn virtual_residual_mask_projection_accepts_full_bound_stencil() {
        const HORIZON: usize = 4;
        const MAX: usize = 3;
        let body = Expr::U8Class(U8Set::from_bytes(b"ab"));
        let envelope = Expr::Seq(vec![
            bytes(b"\""),
            Expr::Repeat {
                expr: Box::new(body),
                min: 0,
                max: Some(MAX),
            },
            bytes(b"\""),
        ]);
        let pattern = Expr::Seq(vec![
            bytes(b"\""),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: None,
            },
            bytes(b"\""),
        ]);
        let expression = Expr::Intersect {
            expr: Box::new(envelope),
            intersect: Box::new(pattern),
        };
        let mut exact = Tokenizer::from_parts(DFA::new(1), 1, None);
        exact
            .install_virtual_residual_components_preserving_oracle_coordinates(vec![(expression, 0)])
            .expect("bounded-code residual runtime must install");
        // The merged bounded-code oracle drops the redundant unbounded copy of
        // this same envelope from its pattern coordinate. That leaves a
        // one-state universal pattern coordinate, reducing the dense stencil
        // work from the old 30 states to 10 without changing the exact
        // one-token projection checked exhaustively below.
        assert_eq!(
            exact.virtual_residual_mask_projection_dense_state_work(HORIZON),
            Some(10),
        );
        let (mask, projections) = exact
            .virtual_residuals_mask_tokenizer(HORIZON)
            .expect("a cheap full-bound finite projection must remain available");
        let projection = &projections[0];
        let root = exact
            .epsilon_closure_states(&[exact.start_state()])
            .into_iter()
            .find(|&state| state != exact.start_state())
            .unwrap();

        let mask_root = projection.project(root).unwrap();
        enumerate_bytes(b"ab\"x", HORIZON, |suffix| {
            let mut full = root;
            let mut finite = mask_root;
            for &byte in suffix {
                let full_next = exact.get_transition(full, byte);
                let finite_next = mask.get_transition(finite, byte);
                assert_eq!(full_next == u32::MAX, finite_next == u32::MAX);
                if full_next == u32::MAX {
                    break;
                }
                assert_eq!(
                    exact.matched_terminal_bitset(full_next),
                    mask.matched_terminal_bitset(finite_next),
                );
                assert_eq!(
                    exact.possible_future_terminals(full_next),
                    mask.possible_future_terminals(finite_next),
                );
                full = full_next;
                finite = finite_next;
            }
        });
    }

    #[test]
    fn residual_state_owner_index_dispatches_multiple_components_directly() {
        let repeated_suffix = |byte, suffix| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(Expr::Choice(vec![
                        bytes(&[byte]),
                        bytes(&[byte, byte]),
                    ])),
                    min: 0,
                    max: Some(1_000_000_000),
                },
                bytes(&[suffix]),
            ])
        };
        let mut tokenizer = Tokenizer::from_parts(DFA::new(1), 2, None);
        tokenizer
            .install_virtual_residual_components(vec![
                (repeated_suffix(b'a', b'z'), 0),
                (repeated_suffix(b'b', b'y'), 1),
            ])
            .expect("two general residual components must install");

        let a_state = tokenizer.step(1, b'a').expect("first residual must advance");
        let b_state = tokenizer.step(2, b'b').expect("second residual must advance");
        assert_ne!(a_state, b_state);
        assert_eq!(
            tokenizer
                .virtual_residual_runtime_for_state(a_state)
                .map(VirtualResidualRuntime::terminal),
            Some(0),
        );
        assert_eq!(
            tokenizer
                .virtual_residual_runtime_for_state(b_state)
                .map(VirtualResidualRuntime::terminal),
            Some(1),
        );

        let aa_state = tokenizer
            .step(a_state, b'a')
            .expect("interleaved first residual must remain owned");
        let bb_state = tokenizer
            .step(b_state, b'b')
            .expect("interleaved second residual must remain owned");
        assert_eq!(
            tokenizer
                .virtual_residual_runtime_for_state(aa_state)
                .map(VirtualResidualRuntime::terminal),
            Some(0),
        );
        assert_eq!(
            tokenizer
                .virtual_residual_runtime_for_state(bb_state)
                .map(VirtualResidualRuntime::terminal),
            Some(1),
        );
    }

    use crate::automata::lexer::dfa::DFA;

    #[test]
    fn compressed_transition_entries_preserve_pair_sequence_wire_format() {
        let legacy = vec![(0u8, 7u32), (3, 11), (49, 782_231)];
        let entries = CompressedTransitionEntries::from_parts(
            legacy.iter().map(|&(class, _)| class).collect(),
            legacy.iter().map(|&(_, target)| target).collect(),
        );
        assert_eq!(
            bincode::serialize(&entries).unwrap(),
            bincode::serialize(&legacy).unwrap(),
        );
        let decoded: CompressedTransitionEntries =
            bincode::deserialize(&bincode::serialize(&legacy).unwrap()).unwrap();
        assert_eq!(decoded.iter_range(0, decoded.len()).collect::<Vec<_>>(), legacy);
    }

    fn dispatch_prefix_tokenizer(with_appended_residual: bool) -> Tokenizer {
        let mut dfa = DFA::new(if with_appended_residual { 5 } else { 4 });
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_epsilon_transition(0, 3);
        dfa.add_transition(1, b'a', 2);
        dfa.add_transition(2, b'a', 2);
        dfa.add_transition(3, b'x', 3);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(2, accepting.clone(), BitSet::new(1));
        if with_appended_residual {
            // This state is deliberately not reset-reachable. It models the
            // externally-entered residuals appended by structural synthesis.
            dfa.add_transition(4, b'a', 2);
            dfa.add_transition(4, b'b', 4);
            dfa.overwrite_state_metadata(4, accepting, BitSet::new(1));
        }
        dfa.recompute_possible_futures();
        Tokenizer::from_parts(dfa, 1, None)
    }

    #[test]
    fn deterministic_dispatch_execution_enters_roots_without_retaining_dispatcher() {
        let tokenizer = dispatch_prefix_tokenizer(false);
        assert_eq!(tokenizer.deterministic_dispatch_roots(), Some(&[1, 3][..]));
        assert!(tokenizer.has_scalar_deterministic_dispatch());

        let empty = tokenizer.execute_from_state_end_only(b"", tokenizer.initial_state_id());
        assert_eq!(empty.as_slice(), &[0, 1, 3]);

        let a = tokenizer.execute_from_state(b"a", tokenizer.initial_state_id());
        assert!(a.matches.iter().any(|matched| matched.id == 0 && matched.width == 1));
        assert_eq!(a.end_state.as_slice(), &[2]);

        let x = tokenizer.execute_from_state_end_only(b"x", tokenizer.initial_state_id());
        assert_eq!(x.as_slice(), &[3]);
    }

    #[test]
    fn huge_wire_preserves_preproven_scalar_dispatch_for_compressed_suffix() {
        let mut tokenizer = dispatch_prefix_tokenizer(false);
        let mut byte_to_class = vec![u8::MAX; 256];
        byte_to_class[b'a' as usize] = 0;
        byte_to_class[b'x' as usize] = 1;
        tokenizer.compressed_transition_segments = Arc::from([CompressedTransitionSegment {
            state_offset: 1,
            state_count: 3,
            byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
            class_members: Arc::from(
                vec![vec![b'a'].into_boxed_slice(), vec![b'x'].into_boxed_slice()]
                    .into_boxed_slice(),
            ),
            row_offsets: Arc::from([0u32, 1, 2, 3]),
            entries: CompressedTransitionEntries::from_parts(
                vec![0, 0, 1],
                vec![1, 1, 2],
            ),
            expanded_transition_count: 3,
        }]);
        for state in &mut tokenizer.dfa.states_mut()[1..] {
            state.transitions.clear();
        }
        tokenizer.invalidate_derived_caches();
        assert!(tokenizer.has_scalar_deterministic_dispatch());
        assert_eq!(tokenizer.scalar_deterministic_dispatch_cache.get(), Some(&true));

        let wire = artifact_serde::build_huge_bytes(&tokenizer).expect("scalar TKS3 fixture");
        let loaded = artifact_serde::from_fast_bytes(&wire).expect("scalar TKS3 roundtrip");
        assert_eq!(
            loaded.scalar_deterministic_dispatch_cache.get(),
            Some(&true),
            "TKS3 must restore the worker-side scalar-dispatch proof without rescanning",
        );
        assert!(loaded.has_scalar_deterministic_dispatch());
        enumerate_bytes(b"ax", 3, |input| {
            assert_eq!(
                normalized_exec(&loaded, input, loaded.initial_state_id()),
                normalized_exec(&tokenizer, input, tokenizer.initial_state_id()),
                "scalar TKS3 execution mismatch on {input:?}",
            );
        });

        let reencoded = artifact_serde::build_huge_bytes(&loaded)
            .expect("loaded scalar TKS3 tokenizer must remain compactly serializable");
        let reloaded = artifact_serde::from_fast_bytes(&reencoded)
            .expect("re-encoded scalar TKS3 roundtrip");
        assert_eq!(
            reloaded.scalar_deterministic_dispatch_cache.get(),
            Some(&true),
            "TKS3 re-encoding must preserve the preproven scalar-dispatch bit",
        );
    }

    #[test]
    fn fast_roundtrip_preserves_scalar_deterministic_dispatch() {
        let tokenizer = dispatch_prefix_tokenizer(false);
        assert!(tokenizer.has_scalar_deterministic_dispatch());
        assert_eq!(tokenizer.terminal_dispatch_root_candidate(0), Some(1));

        let wire = artifact_serde::to_fast_bytes_with_packed_metadata(&tokenizer);
        let loaded = artifact_serde::from_fast_bytes(&wire).expect("fast tokenizer roundtrip");

        assert_eq!(loaded.deterministic_dispatch_roots(), Some(&[1, 3][..]));
        assert_eq!(
            loaded.scalar_deterministic_dispatch_cache.get(),
            Some(&true),
            "TKF3 must restore the worker-side scalar-dispatch proof without rescanning",
        );
        assert!(
            loaded.has_scalar_deterministic_dispatch(),
            "packed runtime storage must not change the reset-dispatch topology proof",
        );
        assert_eq!(
            loaded.terminal_dispatch_root_candidate(0),
            Some(1),
            "packed runtime storage must preserve terminal-specific dispatch-root discovery",
        );
    }

    #[test]
    fn matched_terminal_iterator_does_not_materialize_all_state_cache() {
        let tokenizer = dispatch_prefix_tokenizer(false);
        assert!(tokenizer.matched_terminals_cache.get().is_none());

        assert_eq!(tokenizer.matched_terminals_iter(2).collect::<Vec<_>>(), vec![0]);
        assert!(
            tokenizer.matched_terminals_cache.get().is_none(),
            "per-state iteration must not build the all-state matched-terminal cache",
        );

        assert_eq!(tokenizer.matched_terminals_slice(2), &[0]);
        assert!(tokenizer.matched_terminals_cache.get().is_some());
    }

    #[test]
    fn whole_tokenizer_caches_are_reused_and_invalidated_after_mutation() {
        let source = dispatch_prefix_tokenizer(true);
        let rebuilt = dispatch_prefix_tokenizer(false);
        let mut local = rebuilt.clone();

        let loops_before = local.all_self_loop_bytes();
        let loops_before_again = local.all_self_loop_bytes();
        assert!(Arc::ptr_eq(&loops_before, &loops_before_again));
        assert!(loops_before[2].contains(b'a'));
        assert!(loops_before[3].contains(b'x'));
        let closures_before = local.all_singleton_epsilon_closures();
        let transition_count_before = local.transition_count();
        assert!(local.scalar_deterministic_dispatch_cache.get().is_none());
        let scalar_dispatch_before = local.has_scalar_deterministic_dispatch();
        assert_eq!(
            local.scalar_deterministic_dispatch_cache.get(),
            Some(&scalar_dispatch_before),
        );

        let rebuilt_to_local = (0..rebuilt.num_states()).collect::<Vec<_>>();
        local
            .augment_from_verified_component_prefixes(
                &source,
                &rebuilt,
                &rebuilt_to_local,
            )
            .expect("verified append-only component relation");

        assert!(local.scalar_deterministic_dispatch_cache.get().is_none());
        let scalar_dispatch_after = local.has_scalar_deterministic_dispatch();
        assert_eq!(
            local.scalar_deterministic_dispatch_cache.get(),
            Some(&scalar_dispatch_after),
        );
        let loops_after = local.all_self_loop_bytes();
        assert!(!Arc::ptr_eq(&loops_before, &loops_after));
        assert!(loops_after[4].contains(b'b'));
        assert!(!Arc::ptr_eq(
            &closures_before,
            &local.all_singleton_epsilon_closures(),
        ));
        assert!(local.transition_count() > transition_count_before);
    }

    #[test]
    fn precomputed_bounded_observation_sets_respect_16_and_64_horizons() {
        const BAD: u32 = 21;
        let mut dfa = DFA::new((BAD + 1) as usize);
        dfa.ensure_group_capacity(2);

        let stable_finalizers = BitSet::new(2);
        let mut stable_futures = BitSet::new(2);
        stable_futures.set(0);
        let mut bad_futures = BitSet::new(2);
        bad_futures.set(1);

        let string_bytes = (0x20u8..=0x7e)
            .filter(|&byte| !matches!(byte, b'"' | b'\\'))
            .collect::<Vec<_>>();
        for state in 0..BAD {
            dfa.overwrite_state_metadata(
                state,
                stable_finalizers.clone(),
                stable_futures.clone(),
            );
            for &byte in &string_bytes {
                if byte == b'0' {
                    dfa.add_transition(state, byte, state);
                } else {
                    dfa.add_transition(state, byte, state + 1);
                }
            }
        }
        dfa.overwrite_state_metadata(BAD, BitSet::new(2), bad_futures);
        dfa.add_transition(BAD, b'0', BAD);

        let mut tokenizer = Tokenizer::from_parts(dfa, 2, None);
        let advancing_bytes = string_bytes
            .iter()
            .copied()
            .filter(|&byte| byte != b'0')
            .collect::<Vec<_>>();
        // Deliberately split the advancing family across two byte classes.
        // The precompute must recover the larger semantic family by grouping
        // classes with the same destination rather than selecting one class.
        let advancing_a = advancing_bytes
            .iter()
            .copied()
            .filter(|byte| byte & 1 == 0)
            .collect::<Vec<_>>();
        let advancing_b = advancing_bytes
            .iter()
            .copied()
            .filter(|byte| byte & 1 != 0)
            .collect::<Vec<_>>();
        let mut byte_to_class = vec![3u8; 256];
        for &byte in &advancing_a {
            byte_to_class[byte as usize] = 0;
        }
        for &byte in &advancing_b {
            byte_to_class[byte as usize] = 1;
        }
        byte_to_class[b'0' as usize] = 2;
        let class_members = vec![
            advancing_a.into_boxed_slice(),
            advancing_b.into_boxed_slice(),
            vec![b'0'].into_boxed_slice(),
            (0u16..=255)
                .map(|byte| byte as u8)
                .filter(|byte| !advancing_bytes.contains(byte) && *byte != b'0')
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ];
        let mut row_offsets = Vec::<u32>::with_capacity((BAD + 2) as usize);
        let mut classes = Vec::<u8>::new();
        let mut targets = Vec::<u32>::new();
        row_offsets.push(0);
        for state in 0..BAD {
            classes.extend([0, 1, 2]);
            targets.extend([state + 1, state + 1, state]);
            row_offsets.push(classes.len() as u32);
        }
        classes.push(2);
        targets.push(BAD);
        row_offsets.push(classes.len() as u32);
        tokenizer.compressed_transition_segments = Arc::from([CompressedTransitionSegment {
            state_offset: 0,
            state_count: BAD + 1,
            byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
            class_members: Arc::from(class_members.into_boxed_slice()),
            row_offsets: Arc::from(row_offsets.into_boxed_slice()),
            entries: CompressedTransitionEntries::from_parts(classes, targets),
            expanded_transition_count: BAD as usize * 93 + 1,
        }]);
        let (safe16, safe64) = tokenizer.precompute_bounded_observation_safe_byte_sets();

        // The 92 advancing bytes all co-target the same state and are stable
        // for 16 steps from state 0, despite being split across two tokenizer
        // classes. They can reach the observation-changing state before 64
        // steps, so H64 falls back to the infinitely-safe literal self-loop.
        assert!(safe16[0].contains(b'a'));
        assert!(safe16[0].contains(b'Z'));
        assert!(safe16[0].contains(b'_'));
        assert!(!safe16[0].contains(b'0'));
        assert_eq!(safe16[0].len(), 92);
        assert!(!safe64[0].contains(b'a'));
        assert!(safe64[0].contains(b'0'));

        // Cross-check the advertised sets against the generic exact reference
        // proof using the complete terminal domain as the observation mask.
        let mut all_terminals = BitSet::new(2);
        all_terminals.set(0);
        all_terminals.set(1);
        assert_eq!(
            tokenizer.bounded_observation_safe_horizon_from_state(
                0,
                safe16[0],
                &all_terminals,
                16,
            ),
            16,
        );
        assert_eq!(
            tokenizer.bounded_observation_safe_horizon_from_state(
                0,
                safe64[0],
                &all_terminals,
                64,
            ),
            64,
        );
    }

    #[test]
    fn failed_or_noop_mutations_preserve_derived_caches() {
        let mut tokenizer = dispatch_prefix_tokenizer(false);
        let loops = tokenizer.all_self_loop_bytes();
        let closures = tokenizer.all_singleton_epsilon_closures();

        assert!(tokenizer
            .isolate_start_state_and_drain_nullable_terminals()
            .is_empty());
        assert!(Arc::ptr_eq(&loops, &tokenizer.all_self_loop_bytes()));
        assert!(Arc::ptr_eq(
            &closures,
            &tokenizer.all_singleton_epsilon_closures(),
        ));

        let incompatible = Tokenizer::from_parts(DFA::new(1), 2, None);
        assert!(tokenizer
            .augment_from_verified_component_prefixes(&incompatible, &tokenizer.clone(), &[])
            .is_none());
        assert!(Arc::ptr_eq(&loops, &tokenizer.all_self_loop_bytes()));
    }

    #[test]
    fn compressed_bounded_observation_certificate_matches_generic_reference() {
        let mut dfa = DFA::new(4);
        dfa.ensure_group_capacity(1);
        let mut live = BitSet::new(1);
        live.set(0);
        for state in 0..3u32 {
            dfa.overwrite_state_metadata(state, BitSet::new(1), live.clone());
        }
        dfa.overwrite_state_metadata(3, live.clone(), live.clone());

        let mut tokenizer = Tokenizer::from_parts(dfa, 1, None);
        let byte_to_class = vec![0u8; 256];
        let class_members = vec![(0u16..=255)
            .map(|byte| byte as u8)
            .collect::<Vec<_>>()
            .into_boxed_slice()];
        tokenizer.compressed_transition_segments = Arc::from([CompressedTransitionSegment {
            state_offset: 0,
            state_count: 4,
            byte_to_class: Arc::from(byte_to_class.into_boxed_slice()),
            class_members: Arc::from(class_members.into_boxed_slice()),
            row_offsets: Arc::from([0u32, 1, 2, 3, 4]),
            entries: CompressedTransitionEntries::from_parts(
                vec![0, 0, 0, 0],
                vec![1, 2, 3, 3],
            ),
            expanded_transition_count: 4 * 256,
        }]);

        let mut bytes = U8Set::empty();
        bytes.insert(b'a');
        bytes.insert(b'b');
        let mut active = BitSet::new(1);
        active.set(0);

        for max_horizon in 1..=8 {
            let optimized = tokenizer.bounded_observation_safe_horizon_from_state(
                0,
                bytes,
                &active,
                max_horizon,
            );
            let reference = <Tokenizer as Lexer>::bounded_observation_safe_horizon_from_state(
                &tokenizer,
                0,
                bytes,
                &active,
                max_horizon,
            );
            assert_eq!(optimized, reference, "max_horizon={max_horizon}");
        }
        assert_eq!(
            tokenizer.bounded_observation_safe_horizon_from_state(0, bytes, &active, 8),
            2,
        );

        let (source_horizon, witnesses) = tokenizer
            .bounded_observation_safe_horizon_with_witnesses(0, bytes, &active, 8);
        assert_eq!(source_horizon, 2);
        for (state, depth) in witnesses {
            if depth > source_horizon {
                continue;
            }
            let inherited = source_horizon - depth;
            let reference = <Tokenizer as Lexer>::bounded_observation_safe_horizon_from_state(
                &tokenizer,
                state,
                bytes,
                &active,
                inherited,
            );
            assert_eq!(
                reference, inherited,
                "witness state={state} depth={depth} did not inherit its conservative bound",
            );
        }
    }

    #[test]
    fn deterministic_reset_dispatch_does_not_certify_a_later_epsilon_frontier() {
        let mut dfa = DFA::new(6);
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_epsilon_transition(0, 4);
        dfa.add_transition(1, b'a', 2);
        dfa.add_epsilon_transition(2, 3);
        dfa.add_transition(3, b'b', 3);
        dfa.add_transition(4, b'x', 5);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(3, accepting.clone(), BitSet::new(1));
        dfa.overwrite_state_metadata(5, accepting, BitSet::new(1));
        dfa.recompute_possible_futures();

        let tokenizer = Tokenizer::from_parts(dfa, 1, None);
        assert_eq!(tokenizer.deterministic_dispatch_roots(), Some(&[1, 4][..]));
        assert!(tokenizer.has_deterministic_dispatch());
        assert!(!tokenizer.has_scalar_deterministic_dispatch());

        let result = tokenizer.execute_from_state_end_only(b"ab", tokenizer.initial_state_id());
        assert_eq!(result.as_slice(), &[3]);
    }

    #[test]
    fn structural_prefix_augmentation_rejects_a_non_homomorphic_target_prefix() {
        let source = dispatch_prefix_tokenizer(true);
        let rebuilt = dispatch_prefix_tokenizer(false);
        let mut local = rebuilt.clone();
        local
            .dfa
            .set_transitions_from_sorted_entries(2, vec![(b'a', 3)]);
        let rebuilt_to_local = (0..rebuilt.num_states()).collect::<Vec<_>>();

        assert!(local
            .augment_from_verified_component_prefixes(
                &source,
                &rebuilt,
                &rebuilt_to_local,
            )
            .is_none());
    }

    #[test]
    fn structural_prefix_augmentation_clones_only_appended_residuals() {
        let source = dispatch_prefix_tokenizer(true);
        let rebuilt = dispatch_prefix_tokenizer(false);
        let mut local = rebuilt.clone();
        let rebuilt_to_local = (0..rebuilt.num_states()).collect::<Vec<_>>();

        let source_to_local = local
            .augment_from_verified_component_prefixes(
                &source,
                &rebuilt,
                &rebuilt_to_local,
            )
            .expect("verified append-only component relation");

        assert_eq!(local.num_states(), source.num_states());
        assert_eq!(source_to_local, vec![0, 1, 2, 3, 4]);
        for input in [b"".as_slice(), b"a", b"b", b"ba", b"bba"] {
            let source_result = source.execute_from_state_all_widths(input, 4);
            let local_result = local.execute_from_state_all_widths(input, source_to_local[4]);
            assert_eq!(source_result.matches, local_result.matches, "input={input:?}");
            assert_eq!(source_result.end_state, local_result.end_state, "input={input:?}");
        }
    }

    #[test]
    fn matched_terminal_cache_is_invalidated_by_alias_canonicalization() {
        let mut dfa = DFA::new(1);
        dfa.ensure_group_capacity(2);
        let mut alias = BitSet::new(2);
        alias.set(1);
        dfa.overwrite_state_metadata(0, alias, BitSet::new(2));
        dfa.recompute_possible_futures();

        let mut tokenizer = Tokenizer::from_parts(dfa, 2, None);
        assert_eq!(tokenizer.matched_terminals_slice(0), &[1]);
        tokenizer.canonicalize_terminal_aliases(0, &[1]);
        assert_eq!(tokenizer.matched_terminals_slice(0), &[0]);
    }

    #[test]
    fn owned_parent_union_invalidates_preinitialized_runtime_caches() {
        fn one_byte(byte: u8) -> Tokenizer {
            let mut dfa = DFA::new(2);
            dfa.ensure_group_capacity(1);
            dfa.add_transition(0, byte, 1);
            let mut accepting = BitSet::new(1);
            accepting.set(0);
            dfa.overwrite_state_metadata(1, accepting, BitSet::new(1));
            dfa.recompute_possible_futures();
            Tokenizer::from_parts(dfa, 1, None)
        }

        let left = one_byte(b'a');
        let right = one_byte(b'b');
        assert_eq!(left.matched_terminals_slice(1), &[0]);
        assert!(left.initial_byte_frontiers()[b'a' as usize].contains(&1));

        let (merged, offsets) =
            Tokenizer::disjoint_union_with_owned_parent(left, 0, &[(&right, 1)]);
        let right_accept = offsets[1] + 1;
        assert_eq!(merged.matched_terminals_slice(right_accept), &[1]);
        assert!(merged.initial_byte_frontiers()[b'b' as usize].contains(&right_accept));
    }

    #[test]
    fn owned_parent_union_preserves_fast_loaded_parent_root_and_new_epsilon_child() {
        fn one_byte(byte: u8) -> Tokenizer {
            let mut dfa = DFA::new(2);
            dfa.ensure_group_capacity(1);
            dfa.add_transition(0, byte, 1);
            let mut accepting = BitSet::new(1);
            accepting.set(0);
            dfa.overwrite_state_metadata(1, accepting, BitSet::new(1));
            dfa.recompute_possible_futures();
            Tokenizer::from_parts(dfa, 1, None)
        }

        let parent = one_byte(b'<');
        let left = one_byte(b'a');
        let (half, _) = Tokenizer::disjoint_union_with_owned_parent(parent, 0, &[(&left, 1)]);
        let wire = artifact_serde::to_fast_bytes(&half);
        let loaded = artifact_serde::from_fast_bytes(&wire).expect("fast tokenizer roundtrip");
        assert!(loaded.packed_runtime_transitions.is_some());
        assert!(
            loaded.packed_runtime_metadata.is_some(),
            "current fast loads keep observation metadata packed until structural mutation",
        );
        assert_eq!(
            loaded.dfa.num_states(),
            1,
            "loaded structural DFA should remain a stub",
        );

        let right = one_byte(b'b');
        let (nested, offsets) =
            Tokenizer::disjoint_union_with_owned_parent(loaded, 0, &[(&right, 2)]);
        assert_eq!(offsets[0], 0);
        assert_eq!(nested.deterministic_reset_states().as_slice(), &[0]);
        assert!(
            nested.packed_runtime_transitions.is_some(),
            "structural mutation must not expand the loaded parent's packed byte rows",
        );

        for (input, terminal) in [(b"<".as_slice(), 0), (b"a", 1), (b"b", 2)] {
            let result = nested.execute_from_state(input, nested.initial_state_id());
            assert!(
                result
                    .matches
                    .iter()
                    .any(|matched| matched.id == terminal && matched.width == 1),
                "missing terminal {terminal} for {input:?}: {result:?}",
            );
        }
    }

    #[test]
    fn execution_handles_epsilon_edges_before_and_after_a_byte() {
        let mut dfa = DFA::new(6);
        dfa.ensure_group_capacity(2);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_epsilon_transition(1, 2);
        dfa.add_epsilon_transition(2, 1);
        dfa.add_transition(1, b'a', 3);
        dfa.add_transition(2, b'a', 4);
        dfa.add_epsilon_transition(3, 5);

        let mut terminal_zero = BitSet::new(2);
        terminal_zero.set(0);
        dfa.overwrite_state_metadata(5, terminal_zero, BitSet::new(2));
        let mut terminal_one = BitSet::new(2);
        terminal_one.set(1);
        dfa.overwrite_state_metadata(4, terminal_one, BitSet::new(2));
        dfa.recompute_possible_futures();

        let tokenizer = Tokenizer::from_parts(dfa, 2, None);
        let execution = tokenizer.execute_from_state_all_widths(b"a", 0);
        let mut matches = execution
            .matches
            .iter()
            .map(|matched| (matched.id, matched.width))
            .collect::<Vec<_>>();
        matches.sort_unstable();
        assert_eq!(matches, vec![(0, 1), (1, 1)]);
        assert!(execution.end_state.is_empty());
        let longest = tokenizer.execute_from_state(b"a", 0);
        assert_eq!(longest.end_state.as_slice(), &[3, 4, 5]);
        assert_eq!(tokenizer.matched_terminals(3), BTreeSet::from([0]));

        let interests = BitSet::all(2);
        let (matched, continuation) =
            tokenizer.scan_terminal_matches_from_state(b"a", 0, &interests);
        assert!(matched.contains(0));
        assert!(matched.contains(1));
        assert!(continuation.is_empty());
    }

    #[test]
    fn draining_nullable_initial_closure_preserves_later_root_matches() {
        let mut dfa = DFA::new(2);
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_transition(1, b'a', 1);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(1, accepting, BitSet::new(1));
        dfa.recompute_possible_futures();

        let mut tokenizer = Tokenizer::from_parts(dfa, 1, None);
        assert_eq!(tokenizer.matched_terminals(0), BTreeSet::from([0]));
        assert_eq!(
            tokenizer.isolate_start_state_and_drain_nullable_terminals(),
            BTreeSet::from([0]),
        );
        assert!(tokenizer.matched_terminals(0).is_empty());

        let one = tokenizer.execute_from_state(b"a", tokenizer.initial_state());
        assert!(one.matches.iter().any(|matched| matched.id == 0 && matched.width == 1));
        let two = tokenizer.execute_from_state(b"aa", tokenizer.initial_state());
        assert!(two.matches.iter().any(|matched| matched.id == 0 && matched.width == 2));
    }

    #[test]
    fn longest_match_preserves_every_accepting_end_state_for_one_terminal() {
        let mut dfa = DFA::new(5);
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_epsilon_transition(0, 2);
        dfa.add_transition(1, b'a', 3);
        dfa.add_transition(2, b'a', 4);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(3, accepting.clone(), BitSet::new(1));
        dfa.overwrite_state_metadata(4, accepting, BitSet::new(1));
        dfa.recompute_possible_futures();

        let tokenizer = Tokenizer::from_parts(dfa, 1, None);
        let mut end_states = tokenizer
            .execute_from_state(b"a", 0)
            .matches
            .into_iter()
            .filter(|matched| matched.id == 0 && matched.width == 1)
            .map(|matched| matched.end_state)
            .collect::<Vec<_>>();
        end_states.sort_unstable();
        assert_eq!(end_states, vec![3, 4]);
    }

    #[test]
    fn full_determinization_is_exact_subset_construction() {
        let source = arbitrary_epsilon_l1_test_tokenizer();
        let built = source
            .try_full_determinization(128, 4_096)
            .expect("small epsilon tokenizer must fully determinize");
        let deterministic = &built.tokenizer;

        assert!(!deterministic.has_epsilon_transitions());
        assert_eq!(
            built.source_subsets.len(),
            deterministic.num_states() as usize,
        );

        for state in 0..deterministic.num_states() {
            let subset = &built.source_subsets[state as usize];
            let mut finalizers = BitSet::new(source.num_terminals() as usize);
            let mut futures = BitSet::new(source.num_terminals() as usize);
            for &source_state in subset.iter() {
                finalizers.union_with(source.dfa.finalizers(source_state));
                futures.union_with(source.dfa.possible_future_group_ids(source_state));
            }
            assert_eq!(deterministic.dfa.finalizers(state), &finalizers);
            assert_eq!(
                deterministic.dfa.possible_future_group_ids(state),
                &futures,
            );

            for byte in 0u16..=255 {
                let mut expected = SmallVec::<[u32; 8]>::new();
                for &source_state in subset.iter() {
                    if let Some(target) = source.step(source_state, byte as u8) {
                        expected.push(target);
                    }
                }
                expected.sort_unstable();
                expected.dedup();
                let mut expected = source.dfa.epsilon_closure(&expected);
                expected.sort_unstable();
                expected.dedup();

                match deterministic.step(state, byte as u8) {
                    Some(target) => assert_eq!(
                        built.source_subsets[target as usize].as_ref(),
                        expected.as_slice(),
                        "state={state} byte={byte}",
                    ),
                    None => assert!(expected.is_empty(), "state={state} byte={byte}"),
                }
            }
        }

        for input in [
            b"".as_slice(),
            b"a",
            b"b",
            b"aa",
            b"ab",
            b"ba",
            b"aaa",
        ] {
            let source_result = source.execute_from_state(input, source.initial_state_id());
            let deterministic_result = deterministic
                .execute_from_state(input, deterministic.initial_state_id());
            let mut source_matches = source_result
                .matches
                .iter()
                .map(|matched| (matched.id, matched.width))
                .collect::<Vec<_>>();
            source_matches.sort_unstable();
            source_matches.dedup();
            let mut deterministic_matches = deterministic_result
                .matches
                .iter()
                .map(|matched| (matched.id, matched.width))
                .collect::<Vec<_>>();
            deterministic_matches.sort_unstable();
            deterministic_matches.dedup();
            assert_eq!(source_matches, deterministic_matches, "input={input:?}");

            let represented_end_states = deterministic_result
                .end_state
                .iter()
                .flat_map(|&state| built.source_subsets[state as usize].iter().copied())
                .collect::<BTreeSet<_>>();
            assert_eq!(
                represented_end_states,
                source_result.end_state.iter().copied().collect(),
                "input={input:?}",
            );
        }
    }

    #[test]
    fn full_determinization_all_starts_maps_every_raw_runtime_state_exactly() {
        let source = arbitrary_epsilon_l1_test_tokenizer();
        let (built, raw_to_determinized) = source
            .try_full_determinization_all_starts(256, 16_384)
            .expect("small epsilon tokenizer must determinize from all raw starts");
        let deterministic = &built.tokenizer;
        let closures = source.all_singleton_epsilon_closures();

        assert_eq!(raw_to_determinized.len(), source.num_states() as usize);
        assert!(raw_to_determinized
            .iter()
            .all(|&state| state < deterministic.num_states()));

        for raw_state in 0..source.num_states() {
            let deterministic_state = raw_to_determinized[raw_state as usize];
            assert_eq!(
                built.source_subsets[deterministic_state as usize].as_ref(),
                closures[raw_state as usize].as_ref(),
                "raw_state={raw_state}",
            );

            for byte in 0u16..=255 {
                let mut targets = SmallVec::<[u32; 8]>::new();
                for &source_state in closures[raw_state as usize].iter() {
                    if let Some(target) = source.step(source_state, byte as u8) {
                        targets.extend_from_slice(&closures[target as usize]);
                    }
                }
                targets.sort_unstable();
                targets.dedup();

                match deterministic.step(deterministic_state, byte as u8) {
                    Some(target) => assert_eq!(
                        built.source_subsets[target as usize].as_ref(),
                        targets.as_slice(),
                        "raw_state={raw_state} byte={byte}",
                    ),
                    None => assert!(
                        targets.is_empty(),
                        "raw_state={raw_state} byte={byte}",
                    ),
                }
            }
        }
    }

    #[test]
    fn all_starts_state_budget_rejects_acyclic_raw_state_lower_bound() {
        let mut dfa = DFA::new(2);
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        let source = Tokenizer::from_parts(dfa, 1, None);

        assert!(source
            .try_full_determinization_all_starts(1, 256)
            .is_none());
    }

    #[test]
    fn all_starts_state_budget_does_not_reject_epsilon_cycle_collapse() {
        let mut dfa = DFA::new(2);
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_epsilon_transition(1, 0);
        let source = Tokenizer::from_parts(dfa, 1, None);

        let (built, raw_to_determinized) = source
            .try_full_determinization_all_starts(1, 256)
            .expect("one epsilon SCC should fit in one determinized state");
        assert_eq!(built.tokenizer.num_states(), 1);
        assert_eq!(raw_to_determinized, vec![0, 0]);
    }
}

#[cfg(test)]
#[test]
fn induced_observation_view_preserves_direct_epsilon_and_packed_outputs(){
    use super::ast::{bytes,choice,plus,Expr};
    use super::compile::{build_regex_monolithic,build_regex_partitioned};
    let expressions=vec![bytes(b"longliteralend"),choice(vec![bytes(b"!x"),bytes(b"!yz")]),plus(bytes(b"q")),Expr::Epsilon];
    let words=vec![b"end!".to_vec(),b"q!xq".to_vec(),b"!y".to_vec(),b"qend".to_vec()];
    for fresh in [build_regex_monolithic(&expressions).into_tokenizer(4,Some(Arc::from(expressions.clone()))),
        build_regex_partitioned(&expressions,&[0,1,2,3]).into_tokenizer(4,Some(Arc::from(expressions.clone())))]{
        let wire=artifact_serde::to_fast_bytes(&fresh);
        let mut packed=artifact_serde::from_fast_bytes(&wire).unwrap();
        packed.restore_terminal_exprs_without_virtual_runtime(Some(expressions.clone())).unwrap();
        for tokenizer in [&fresh,&packed]{
            let mut keep=vec![false;tokenizer.num_states() as usize];
            let roots=tokenizer.deterministic_reset_states().into_vec();
            for word in &words{for start in 0..word.len(){
                let mut states=roots.clone();
                for &byte in &word[start..]{states=tokenizer.step_all(&states,byte).into_vec();
                    for &q in &states{keep[q as usize]=true;}
                }
            }}
            let view=tokenizer.induced_observation_view(&keep).unwrap();
            assert!(view.tokenizer.num_states()<tokenizer.num_states());
            assert!(view.tokenizer.terminal_exprs().is_some());
            for (q,&raw) in view.view_to_original.iter().enumerate(){
                assert_eq!(view.tokenizer.state_finalizers(q as u32),tokenizer.state_finalizers(raw));
                assert_eq!(view.tokenizer.state_futures(q as u32),tokenizer.state_futures(raw));
                let mut expected=Vec::new();tokenizer.for_each_epsilon_target(raw,|t|expected.push(view.original_to_view[t as usize]));
                let mut actual=Vec::new();view.tokenizer.for_each_epsilon_target(q as u32,|t|actual.push(t));
                expected.sort_unstable();actual.sort_unstable();assert_eq!(actual,expected);
            }
            for word in &words{for start in 0..word.len(){
                let mut a=roots.clone();let mut b=view.tokenizer.deterministic_reset_states().into_vec();
                for &byte in &word[start..]{
                    a=tokenizer.step_all(&a,byte).into_vec();b=view.tokenizer.step_all(&b,byte).into_vec();
                    let mut decoded=b.iter().map(|&q|view.view_to_original[q as usize]).collect::<Vec<_>>();
                    decoded.sort_unstable();decoded.dedup();assert_eq!(a,decoded);
                }
            }}
        }
    }
}
