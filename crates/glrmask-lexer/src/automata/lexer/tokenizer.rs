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
    #[inline]
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

    #[inline]
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
    #[inline]
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

    #[inline]
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

pub mod artifact_serde ;


/// Compact tokenizer wire form for dynamic artifacts.
///
/// The ordinary historical `Tokenizer` serializer stores one dense terminal
/// bitset for every lexer state.  That is appropriate for backwards-compatible
/// static artifacts, but source-specialized grammars can have hundreds of
/// thousands of states and thousands of terminals while setting only a handful
/// of metadata bits per state.  This wire form stores exactly the set bits and
/// the actual graph edges.
pub mod compact_artifact_serde ;

/// Packed tokenizer wire form used only by current static-constraint
/// artifacts. Runtime and compiler representations remain unchanged.
///
/// The byte labels of lexer transition rows are highly repetitive (large
/// schema tokenizers commonly have only ~100 distinct byte patterns across
/// thousands of states), while target ids are small. Store each byte pattern
/// once and encode the matching target sequence as varints. The old
/// `compact_artifact_serde` remains untouched because dynamic-constraint
/// artifacts use it as a persisted format.
mod packed_artifact_serde ;

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

    fn materialized_dfa(&self) -> DFA {
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
mod tests ;

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
