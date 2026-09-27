//! Compact parser, lexer, and commit-template transition tables.

use super::DenseWords;
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::weighted::dwa::DwaTransitionMap;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::ds::weight::Weight;
use glrmask_artifact::CommitTemplateDfas;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;
pub(super) const INLINE_DWA_TRANSITION_LIMIT: usize = 8;

#[derive(Debug, Clone)]
pub(crate) enum FastDwaTransitionRow {
    Inline(SmallVec<[(i32, (u32, Weight)); 4]>),
    Hash(FxHashMap<i32, (u32, Weight)>),
    Packed(DwaTransitionMap),
}

impl FastDwaTransitionRow {
    pub(crate) fn from_entries(entries: impl IntoIterator<Item = (i32, (u32, Weight))>) -> Self {
        let entries = entries.into_iter().collect::<SmallVec<[_; 4]>>();
        if entries.len() <= INLINE_DWA_TRANSITION_LIMIT {
            Self::Inline(entries)
        } else {
            Self::Hash(entries.into_iter().collect())
        }
    }

    pub(crate) fn from_exact_entries(
        len: usize,
        entries: impl IntoIterator<Item = (i32, (u32, Weight))>,
    ) -> Self {
        if len <= INLINE_DWA_TRANSITION_LIMIT {
            Self::Inline(entries.into_iter().collect())
        } else {
            let mut map = FxHashMap::default();
            map.reserve(len);
            map.extend(entries);
            Self::Hash(map)
        }
    }

    pub(crate) fn from_packed(row: DwaTransitionMap) -> Self {
        debug_assert!(row.is_packed());
        Self::Packed(row)
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Inline(entries) => entries.is_empty(),
            Self::Hash(entries) => entries.is_empty(),
            Self::Packed(row) => row.is_empty(),
        }
    }

    #[inline]
    pub(crate) fn get(&self, label: &i32) -> Option<(u32, &Weight)> {
        match self {
            Self::Inline(entries) => entries.iter().find_map(|(candidate, (target, weight))| {
                (candidate == label).then_some((*target, weight))
            }),
            Self::Hash(entries) => entries.get(label).map(|(target, weight)| (*target, weight)),
            Self::Packed(row) => row.get_entry(label),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum FastDwaTransitions {
    Direct(Vec<FastDwaTransitionRow>),
    Shared {
        rows: Vec<FastDwaTransitionRow>,
        state_rows: Vec<u32>,
    },
}

impl Default for FastDwaTransitions {
    fn default() -> Self {
        Self::Direct(Vec::new())
    }
}

impl FastDwaTransitions {
    pub(crate) fn direct(rows: Vec<FastDwaTransitionRow>) -> Self {
        Self::Direct(rows)
    }

    pub(crate) fn shared(rows: Vec<FastDwaTransitionRow>, state_rows: Vec<u32>) -> Self {
        Self::Shared { rows, state_rows }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Direct(rows) => rows.len(),
            Self::Shared { state_rows, .. } => state_rows.len(),
        }
    }

    #[inline]
    pub(crate) fn get(&self, state: usize) -> Option<&FastDwaTransitionRow> {
        match self {
            Self::Direct(rows) => rows.get(state),
            Self::Shared { rows, state_rows } => state_rows
                .get(state)
                .and_then(|&row| rows.get(row as usize)),
        }
    }
}

impl std::ops::Index<usize> for FastDwaTransitions {
    type Output = FastDwaTransitionRow;

    #[inline]
    fn index(&self, state: usize) -> &Self::Output {
        match self {
            Self::Direct(rows) => &rows[state],
            Self::Shared { rows, state_rows } => &rows[state_rows[state] as usize],
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum IndexedDagDenseMask {
    Full,
    Dense {
        words: DenseWords,
        start: usize,
        end: usize,
    },
    Empty,
}

#[derive(Debug, Clone)]
pub(crate) struct IndexedDagDenseTransition {
    pub(crate) target: u32,
    pub(crate) masks: IndexedDagDenseTransitionMasks,
}

pub(super) const INLINE_INDEXED_DAG_TSID_LIMIT: usize = 8;

#[derive(Debug, Clone)]
pub(crate) enum IndexedDagDenseTransitionMasks {
    Full,
    Inline(SmallVec<[(u32, IndexedDagDenseMask); 2]>),
    Hash(FxHashMap<u32, IndexedDagDenseMask>),
}

pub(super) static INDEXED_DAG_FULL_MASK: IndexedDagDenseMask = IndexedDagDenseMask::Full;

pub(super) static INDEXED_DAG_EMPTY_MASK: IndexedDagDenseMask = IndexedDagDenseMask::Empty;

impl IndexedDagDenseTransitionMasks {
    pub(crate) fn from_entries(
        entries: impl IntoIterator<Item = (u32, IndexedDagDenseMask)>,
    ) -> Self {
        let entries = entries.into_iter().collect::<SmallVec<[_; 2]>>();
        if entries.len() <= INLINE_INDEXED_DAG_TSID_LIMIT {
            Self::Inline(entries)
        } else {
            Self::Hash(entries.into_iter().collect())
        }
    }

    #[inline]
    pub(crate) fn get(&self, tsid: u32) -> &IndexedDagDenseMask {
        match self {
            Self::Full => &INDEXED_DAG_FULL_MASK,
            Self::Inline(entries) => entries
                .iter()
                .find_map(|(candidate, mask)| (*candidate == tsid).then_some(mask))
                .unwrap_or(&INDEXED_DAG_EMPTY_MASK),
            Self::Hash(entries) => entries.get(&tsid).unwrap_or(&INDEXED_DAG_EMPTY_MASK),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum IndexedDagDenseTransitionRow {
    Inline(SmallVec<[(i32, IndexedDagDenseTransition); 4]>),
    Hash(FxHashMap<i32, IndexedDagDenseTransition>),
}

impl IndexedDagDenseTransitionRow {
    pub(crate) fn from_entries(
        entries: impl IntoIterator<Item = (i32, IndexedDagDenseTransition)>,
    ) -> Self {
        let entries = entries.into_iter().collect::<SmallVec<[_; 4]>>();
        if entries.len() <= INLINE_DWA_TRANSITION_LIMIT {
            Self::Inline(entries)
        } else {
            Self::Hash(entries.into_iter().collect())
        }
    }

    #[inline]
    pub(crate) fn get(&self, label: &i32) -> Option<&IndexedDagDenseTransition> {
        match self {
            Self::Inline(entries) => entries
                .iter()
                .find_map(|(candidate, transition)| (candidate == label).then_some(transition)),
            Self::Hash(entries) => entries.get(label),
        }
    }
}

pub(crate) type IndexedDagDenseTransitions = Vec<IndexedDagDenseTransitionRow>;

#[derive(Debug, Clone)]
pub(crate) enum FastTokenizerTransitions {
    Dense(Vec<Box<[u32; 256]>>),
    Flat(Arc<[u32]>),
    /// Compact exact dense DFA rows for tokenizers with fewer than 32,768
    /// states. The high bit marks a target state with at least one finalizer;
    /// u16::MAX is the dead-transition sentinel.
    Flat16 {
        transitions: Arc<[u16]>,
        finalizer_code: Arc<[u32]>,
        single_finalizer_continues: Arc<[u8]>,
    },
    /// Exact dense DFA rows for the strict full-vocabulary walker when the
    /// tokenizer no longer fits the 15-bit Flat16 coordinate. The high bit
    /// marks a target state with at least one finalizer; u32::MAX is dead.
    Flat32 {
        transitions: Arc<[u32]>,
        finalizer_code: Arc<[u32]>,
        single_finalizer_continues: Arc<[u8]>,
    },
    /// Runtime tokenizer already owns an allocation-light exact transition
    /// table; call through instead of rebuilding a second dense table.
    Fallback(usize),
    Hybrid {
        state_to_dense_row: Vec<u32>,
        dense_rows: Vec<Box<[u32; 256]>>,
    },
}

impl Default for FastTokenizerTransitions {
    fn default() -> Self {
        Self::Dense(Vec::new())
    }
}

impl FastTokenizerTransitions {
    pub(super) fn full_walk_finalizer_metadata(tokenizer: &Tokenizer) -> (Arc<[u32]>, Arc<[u8]>) {
        const NONE: u32 = u32::MAX;
        const MULTI: u32 = u32::MAX - 1;
        let num_states = tokenizer.num_states();
        let state_metadata = |state: u32| {
            let finalizers = tokenizer.matched_terminals_slice(state);
            let code = match finalizers {
                [] => NONE,
                [terminal] => *terminal,
                _ => MULTI,
            };
            let continues = match finalizers {
                [terminal]
                    if tokenizer
                        .possible_future_terminals(state)
                        .contains(*terminal as usize) =>
                {
                    1u8
                }
                _ => 0u8,
            };
            (code, continues)
        };
        let metadata = if num_states >= 1_024 && rayon::current_num_threads() > 1 {
            (0..num_states)
                .into_par_iter()
                .map(state_metadata)
                .collect::<Vec<_>>()
        } else {
            (0..num_states).map(state_metadata).collect::<Vec<_>>()
        };
        let (finalizer_code, single_finalizer_continues): (Vec<_>, Vec<_>) =
            metadata.into_iter().unzip();
        (
            Arc::from(finalizer_code),
            Arc::from(single_finalizer_continues),
        )
    }

    pub(crate) fn flat16_for(tokenizer: &Tokenizer) -> Option<Self> {
        let num_states = tokenizer.num_states();
        if num_states >= 0x8000 {
            return None;
        }
        // Build finalizer metadata once per state before populating transition
        // cells. The old implementation called `matched_terminals_slice()` for
        // every edge merely to set the high-bit hint, then scanned every state
        // again below to construct these exact metadata arrays. Fresh compiler
        // DFAs make that per-edge lookup materially more expensive than loaded
        // packed tokenizers.
        let (finalizer_code, single_finalizer_continues) =
            Self::full_walk_finalizer_metadata(tokenizer);
        let mut flat = vec![u16::MAX; num_states as usize * 256];
        let fill_row = |(state, row): (usize, &mut [u16])| {
            for (byte, target) in tokenizer.transitions_from(state as u32) {
                let mut encoded = u16::try_from(target)
                    .expect("flat16 tokenizer target exceeds 15-bit state coordinate");
                if finalizer_code[target as usize] != u32::MAX {
                    encoded |= 0x8000;
                }
                row[byte as usize] = encoded;
            }
        };
        if num_states >= 1_024 && rayon::current_num_threads() > 1 {
            flat.par_chunks_mut(256).enumerate().for_each(fill_row);
        } else {
            flat.chunks_mut(256).enumerate().for_each(fill_row);
        }
        Some(Self::Flat16 {
            transitions: Arc::from(flat),
            finalizer_code,
            single_finalizer_continues,
        })
    }

    /// Dense u16 transition slab for ordinary tokenizer execution. Unlike the
    /// strict full-vocabulary walker, commit/scan only asks this object for the
    /// target state, so constructing per-state finalizer certificates (and
    /// probing finalizers for every transition target) is pure load-time waste.
    pub(crate) fn flat16_transitions_only_for(tokenizer: &Tokenizer) -> Option<Self> {
        let num_states = tokenizer.num_states();
        if num_states >= 0x8000 {
            return None;
        }
        let mut flat = vec![u16::MAX; num_states as usize * 256];
        let fill_row = |(state, row): (usize, &mut [u16])| {
            for (byte, target) in tokenizer.transitions_from(state as u32) {
                row[byte as usize] = u16::try_from(target)
                    .expect("flat16 tokenizer target exceeds 15-bit state coordinate");
            }
        };
        if num_states >= 1_024 && rayon::current_num_threads() > 1 {
            flat.par_chunks_mut(256).enumerate().for_each(fill_row);
        } else {
            flat.chunks_mut(256).enumerate().for_each(fill_row);
        }
        Some(Self::Flat16 {
            transitions: Arc::from(flat),
            finalizer_code: Arc::from([]),
            single_finalizer_continues: Arc::from([]),
        })
    }

    pub(crate) fn flat32_for(tokenizer: &Tokenizer) -> Option<Self> {
        let num_states = tokenizer.num_states();
        if num_states >= 0x8000_0000 {
            return None;
        }
        let (finalizer_code, single_finalizer_continues) =
            Self::full_walk_finalizer_metadata(tokenizer);
        let cell_count = (num_states as usize).checked_mul(256)?;
        let mut flat = vec![u32::MAX; cell_count];
        let fill_row = |(state, row): (usize, &mut [u32])| {
            for (byte, target) in tokenizer.transitions_from(state as u32) {
                debug_assert!(target < 0x8000_0000);
                let mut encoded = target;
                if finalizer_code[target as usize] != u32::MAX {
                    encoded |= 0x8000_0000;
                }
                row[byte as usize] = encoded;
            }
        };
        if num_states >= 1_024 && rayon::current_num_threads() > 1 {
            flat.par_chunks_mut(256).enumerate().for_each(fill_row);
        } else {
            flat.chunks_mut(256).enumerate().for_each(fill_row);
        }
        Some(Self::Flat32 {
            transitions: Arc::from(flat),
            finalizer_code,
            single_finalizer_continues,
        })
    }

    /// Build the dense transition representation used only by the strict full
    /// vocabulary walker. Flat16 is preferred whenever possible; Flat32 is
    /// available for larger deterministic mask coordinates. `max_bytes`
    /// bounds only transition-cell storage, not the small metadata arrays.
    pub(crate) fn full_walk_dense_for(tokenizer: &Tokenizer, max_bytes: usize) -> Option<Self> {
        let num_states = tokenizer.num_states() as usize;
        let flat16_bytes = num_states
            .checked_mul(256)?
            .checked_mul(std::mem::size_of::<u16>())?;
        if num_states < 0x8000 && flat16_bytes <= max_bytes {
            return Self::flat16_for(tokenizer);
        }
        let flat32_bytes = num_states
            .checked_mul(256)?
            .checked_mul(std::mem::size_of::<u32>())?;
        if num_states < 0x8000_0000 && flat32_bytes <= max_bytes {
            return Self::flat32_for(tokenizer);
        }
        None
    }

    #[inline]
    pub(crate) fn transition(&self, tokenizer: &Tokenizer, state: u32, byte: u8) -> u32 {
        match self {
            Self::Dense(rows) => rows.get(state as usize).map_or_else(
                || tokenizer.get_transition(state, byte),
                |row| row[byte as usize],
            ),
            Self::Flat(flat) => state
                .try_into()
                .ok()
                .and_then(|state: usize| state.checked_mul(256))
                .and_then(|offset| offset.checked_add(byte as usize))
                .and_then(|index| flat.get(index))
                .copied()
                .unwrap_or_else(|| tokenizer.get_transition(state, byte)),
            Self::Flat16 { transitions, .. } => state
                .try_into()
                .ok()
                .and_then(|state: usize| state.checked_mul(256))
                .and_then(|offset| offset.checked_add(byte as usize))
                .and_then(|index| transitions.get(index))
                .copied()
                .map(|target| {
                    if target == u16::MAX {
                        u32::MAX
                    } else {
                        u32::from(target & 0x7fff)
                    }
                })
                .unwrap_or_else(|| tokenizer.get_transition(state, byte)),
            Self::Flat32 { transitions, .. } => state
                .try_into()
                .ok()
                .and_then(|state: usize| state.checked_mul(256))
                .and_then(|offset| offset.checked_add(byte as usize))
                .and_then(|index| transitions.get(index))
                .copied()
                .map(|target| {
                    if target == u32::MAX {
                        u32::MAX
                    } else {
                        target & 0x7fff_ffff
                    }
                })
                .unwrap_or_else(|| tokenizer.get_transition(state, byte)),
            Self::Fallback(_) => tokenizer.get_transition(state, byte),
            Self::Hybrid {
                state_to_dense_row,
                dense_rows,
            } => {
                let dense = state_to_dense_row
                    .get(state as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                if dense == u32::MAX {
                    tokenizer.get_transition(state, byte)
                } else {
                    dense_rows[dense as usize][byte as usize]
                }
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Dense(rows) => rows.len(),
            Self::Flat(flat) => flat.len() / 256,
            Self::Flat16 { transitions, .. } => transitions.len() / 256,
            Self::Flat32 { transitions, .. } => transitions.len() / 256,
            Self::Fallback(len) => *len,
            Self::Hybrid {
                state_to_dense_row, ..
            } => state_to_dense_row.len(),
        }
    }

    /// Reuse the consumed parent's fast transition rows and append rebased
    /// child rows. Compressed child states remain sparse and fall back to the
    /// merged tokenizer, whose compressed segments have already been rebased.
    pub(crate) fn append_rebased_children(
        self,
        children: &[(&FastTokenizerTransitions, u32)],
    ) -> Option<Self> {
        fn flat_rows(flat: &[u32]) -> Option<Vec<Box<[u32; 256]>>> {
            let chunks = flat.chunks_exact(256);
            if !chunks.remainder().is_empty() {
                return None;
            }
            chunks
                .map(|chunk| {
                    let row: &[u32; 256] = chunk.try_into().ok()?;
                    Some(Box::new(*row))
                })
                .collect()
        }

        fn rebased_row(row: &[u32; 256], offset: u32) -> Box<[u32; 256]> {
            let mut rebased = Box::new(*row);
            for target in rebased.iter_mut() {
                if *target != u32::MAX {
                    *target = target
                        .checked_add(offset)
                        .expect("composed tokenizer fast-transition target overflow");
                }
            }
            rebased
        }

        let all_dense = children
            .iter()
            .all(|(child, _)| matches!(child, FastTokenizerTransitions::Dense(_)));
        match self {
            Self::Dense(mut rows) if all_dense => {
                for (child, offset) in children {
                    if *offset as usize != rows.len() {
                        return None;
                    }
                    let Self::Dense(child_rows) = child else {
                        unreachable!()
                    };
                    rows.extend(child_rows.iter().map(|row| rebased_row(row, *offset)));
                }
                Some(Self::Dense(rows))
            }
            parent => {
                let (mut state_to_dense_row, mut dense_rows) = match parent {
                    Self::Dense(rows) => {
                        let state_to_dense_row = (0..rows.len() as u32).collect::<Vec<_>>();
                        (state_to_dense_row, rows)
                    }
                    Self::Flat(flat) => {
                        let rows = flat_rows(&flat)?;
                        let state_to_dense_row = (0..rows.len() as u32).collect::<Vec<_>>();
                        (state_to_dense_row, rows)
                    }
                    Self::Flat16 { .. } | Self::Flat32 { .. } | Self::Fallback(_) => return None,
                    Self::Hybrid {
                        state_to_dense_row,
                        dense_rows,
                    } => (state_to_dense_row, dense_rows),
                };
                for (child, offset) in children {
                    if *offset as usize != state_to_dense_row.len() {
                        return None;
                    }
                    match child {
                        Self::Flat16 { .. } | Self::Flat32 { .. } | Self::Fallback(_) => {
                            return None;
                        }
                        Self::Dense(rows) => {
                            for row in rows {
                                let dense = dense_rows.len() as u32;
                                dense_rows.push(rebased_row(row, *offset));
                                state_to_dense_row.push(dense);
                            }
                        }
                        Self::Flat(flat) => {
                            let rows = flat_rows(flat)?;
                            for row in rows {
                                let dense = dense_rows.len() as u32;
                                dense_rows.push(rebased_row(&row, *offset));
                                state_to_dense_row.push(dense);
                            }
                        }
                        Self::Hybrid {
                            state_to_dense_row: child_mapping,
                            dense_rows: child_rows,
                        } => {
                            for &child_dense in child_mapping {
                                if child_dense == u32::MAX {
                                    state_to_dense_row.push(u32::MAX);
                                } else {
                                    let row = child_rows.get(child_dense as usize)?;
                                    let dense = dense_rows.len() as u32;
                                    dense_rows.push(rebased_row(row, *offset));
                                    state_to_dense_row.push(dense);
                                }
                            }
                        }
                    }
                }
                Some(Self::Hybrid {
                    state_to_dense_row,
                    dense_rows,
                })
            }
        }
    }
}

pub(crate) type TemplateDfasByTerminal = Vec<Option<Arc<CommitTemplateDfas>>>;

pub(crate) type FastTemplateDfasByTerminal = Vec<Option<Arc<FastCommitTemplateDfas>>>;

pub(super) const INLINE_TEMPLATE_TRANSITION_LIMIT: usize = 8;

#[derive(Debug, Clone, Default)]
pub(crate) enum FastTemplateTransitionRow {
    #[default]
    Empty,
    Inline(SmallVec<[(i32, u32); 4]>),
    Hash(FxHashMap<i32, u32>),
}

impl FastTemplateTransitionRow {
    pub(super) fn from_entries(entries: impl IntoIterator<Item = (i32, u32)>) -> Self {
        let entries = entries.into_iter().collect::<SmallVec<[_; 4]>>();
        match entries.len() {
            0 => Self::Empty,
            len if len <= INLINE_TEMPLATE_TRANSITION_LIMIT => Self::Inline(entries),
            _ => Self::Hash(entries.into_iter().collect()),
        }
    }

    #[inline]
    pub(crate) fn get(&self, label: i32) -> Option<u32> {
        match self {
            Self::Empty => None,
            Self::Inline(entries) => entries
                .iter()
                .find_map(|(candidate, target)| (*candidate == label).then_some(*target)),
            Self::Hash(entries) => entries.get(&label).copied(),
        }
    }

    #[inline]
    pub(crate) fn for_each(&self, mut f: impl FnMut(i32, u32)) {
        match self {
            Self::Empty => {}
            Self::Inline(entries) => {
                for &(label, target) in entries {
                    f(label, target);
                }
            }
            Self::Hash(entries) => {
                for (&label, &target) in entries {
                    f(label, target);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FastTemplateDfaState {
    pub(crate) is_accepting: bool,
    pub(crate) default_target: Option<u32>,
    pub(crate) transitions: FastTemplateTransitionRow,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FastTemplateDfa {
    pub(crate) states: Vec<FastTemplateDfaState>,
    pub(crate) start_state: u32,
}

impl FastTemplateDfa {
    pub(super) fn from_dfa(dfa: &UnweightedDfa) -> Self {
        Self {
            states: dfa
                .states
                .iter()
                .map(|state| FastTemplateDfaState {
                    is_accepting: state.is_accepting,
                    default_target: state.transitions.get(&DEFAULT_LABEL).copied(),
                    transitions: FastTemplateTransitionRow::from_entries(
                        state
                            .transitions
                            .iter()
                            .filter(|(label, _)| **label != DEFAULT_LABEL)
                            .map(|(&label, &target)| (label, target)),
                    ),
                })
                .collect(),
            start_state: dfa.start_state,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FastCommitTemplateDfas {
    pub(crate) pop: FastTemplateDfa,
    pub(crate) read: FastTemplateDfa,
    pub(crate) push: FastTemplateDfa,
    pub(crate) pop_to_read: Vec<Option<u32>>,
    pub(crate) pop_to_push: Vec<Option<u32>>,
    pub(crate) read_to_push: Vec<Option<u32>>,
}

impl FastCommitTemplateDfas {
    pub(crate) fn from_template(template: &CommitTemplateDfas) -> Self {
        Self {
            pop: FastTemplateDfa::from_dfa(&template.pop),
            read: FastTemplateDfa::from_dfa(&template.read),
            push: FastTemplateDfa::from_dfa(&template.push),
            pop_to_read: template.pop_to_read.clone(),
            pop_to_push: template.pop_to_push.clone(),
            read_to_push: template.read_to_push.clone(),
        }
    }
}
