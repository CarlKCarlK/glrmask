use std::collections::{BTreeMap, BTreeSet};
use std::cell::{Cell, RefCell};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, OnceLock};

use range_set_blaze::{
    CheckSortedDisjoint, CheckSortedDisjointMap, RangeMapBlaze, RangeSetBlaze,
};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHasher};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use super::nwa::Label;
use crate::ds::weight::{
    finalize_weight_map, finalize_weight_map_artifact_local, shared_rangeset,
    shared_rangeset_artifact_local, Weight,
};

#[derive(Debug)]
struct PackedDwaRowPool {
    labels: Vec<Vec<Label>>,
    targets: Vec<Vec<u32>>,
    weight_ids: Vec<Vec<u32>>,
    weights: Arc<[Weight]>,
}

#[derive(Debug)]
#[doc(hidden)]
pub struct PackedDwaTransitionRow {
    pool: Arc<PackedDwaRowPool>,
    label_id: usize,
    target_id: usize,
    weight_id: usize,
    materialized: OnceLock<BTreeMap<Label, (u32, Weight)>>,
}

impl PackedDwaTransitionRow {
    fn len(&self) -> usize {
        self.pool.labels[self.label_id].len()
    }

    fn materialized(&self) -> &BTreeMap<Label, (u32, Weight)> {
        self.materialized.get_or_init(|| {
            let labels = &self.pool.labels[self.label_id];
            let targets = &self.pool.targets[self.target_id];
            let weights = &self.pool.weight_ids[self.weight_id];
            labels
                .iter()
                .zip(targets)
                .zip(weights)
                .map(|((&label, &target), &weight)| {
                    (label, (target, self.pool.weights[weight as usize].clone()))
                })
                .collect()
        })
    }
}

pub struct DwaTransitionEntries<'a> {
    inner: DwaTransitionEntriesInner<'a>,
}

enum DwaTransitionEntriesInner<'a> {
    Tree(std::collections::btree_map::Iter<'a, Label, (u32, Weight)>),
    Packed {
        row: &'a PackedDwaTransitionRow,
        index: usize,
    },
}

impl<'a> Iterator for DwaTransitionEntries<'a> {
    type Item = (Label, u32, &'a Weight);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            DwaTransitionEntriesInner::Tree(entries) => entries
                .next()
                .map(|(&label, (target, weight))| (label, *target, weight)),
            DwaTransitionEntriesInner::Packed { row, index } => {
                let labels = &row.pool.labels[row.label_id];
                let position = *index;
                let &label = labels.get(position)?;
                let target = row.pool.targets[row.target_id][position];
                let weight_id = row.pool.weight_ids[row.weight_id][position] as usize;
                *index += 1;
                Some((label, target, &row.pool.weights[weight_id]))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match &self.inner {
            DwaTransitionEntriesInner::Tree(entries) => entries.len(),
            DwaTransitionEntriesInner::Packed { row, index } => row.len() - *index,
        };
        (len, Some(len))
    }
}

impl ExactSizeIterator for DwaTransitionEntries<'_> {}

#[derive(Debug, Clone)]
pub enum DwaTransitionMap {
    Owned(BTreeMap<Label, (u32, Weight)>),
    Shared(Arc<BTreeMap<Label, (u32, Weight)>>),
    Packed(Arc<PackedDwaTransitionRow>),
}

impl Default for DwaTransitionMap {
    fn default() -> Self {
        Self::Owned(BTreeMap::new())
    }
}

impl DwaTransitionMap {
    pub fn from_arc(transitions: Arc<BTreeMap<Label, (u32, Weight)>>) -> Self {
        Self::Shared(transitions)
    }

    pub fn ptr_key(&self) -> usize {
        match self {
            Self::Owned(transitions) => transitions as *const BTreeMap<_, _> as usize,
            Self::Shared(transitions) => Arc::as_ptr(transitions) as usize,
            Self::Packed(transitions) => Arc::as_ptr(transitions) as usize,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Owned(transitions) => transitions.len(),
            Self::Shared(transitions) => transitions.len(),
            Self::Packed(transitions) => transitions.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::Packed(_))
    }

    pub fn entries(&self) -> DwaTransitionEntries<'_> {
        let inner = match self {
            Self::Owned(transitions) => DwaTransitionEntriesInner::Tree(transitions.iter()),
            Self::Shared(transitions) => DwaTransitionEntriesInner::Tree(transitions.iter()),
            Self::Packed(row) => DwaTransitionEntriesInner::Packed { row, index: 0 },
        };
        DwaTransitionEntries { inner }
    }

    #[inline]
    pub fn get_entry(&self, label: &Label) -> Option<(u32, &Weight)> {
        match self {
            Self::Owned(transitions) => transitions
                .get(label)
                .map(|(target, weight)| (*target, weight)),
            Self::Shared(transitions) => transitions
                .get(label)
                .map(|(target, weight)| (*target, weight)),
            Self::Packed(row) => {
                let labels = &row.pool.labels[row.label_id];
                let index = labels.binary_search(label).ok()?;
                let target = row.pool.targets[row.target_id][index];
                let weight_id = row.pool.weight_ids[row.weight_id][index] as usize;
                Some((target, &row.pool.weights[weight_id]))
            }
        }
    }
}

impl PartialEq for DwaTransitionMap {
    fn eq(&self, other: &Self) -> bool {
        if let (Self::Packed(left), Self::Packed(right)) = (self, other)
            && Arc::ptr_eq(left, right)
        {
            return true;
        }
        self.deref() == other.deref()
    }
}

impl Deref for DwaTransitionMap {
    type Target = BTreeMap<Label, (u32, Weight)>;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(transitions) => transitions,
            Self::Shared(transitions) => transitions,
            Self::Packed(transitions) => transitions.materialized(),
        }
    }
}

impl DerefMut for DwaTransitionMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // `Shared` means the Arc identity is a canonical transition-row
        // identity.  Once a caller mutates the row that invariant no longer
        // holds, even if Arc::make_mut could give us unique storage.  Move or
        // clone it into `Owned` explicitly so serializers can continue to use
        // Shared pointer identity without missing equal rows after mutation.
        if matches!(self, Self::Shared(_) | Self::Packed(_)) {
            let current = std::mem::take(self);
            let owned = match current {
                Self::Shared(transitions) => Arc::try_unwrap(transitions)
                    .unwrap_or_else(|shared| shared.as_ref().clone()),
                Self::Packed(transitions) => transitions.materialized().clone(),
                Self::Owned(_) => unreachable!("checked shared representation above"),
            };
            *self = Self::Owned(owned);
        }
        match self {
            Self::Owned(transitions) => transitions,
            Self::Shared(_) | Self::Packed(_) => {
                unreachable!("shared row was converted to Owned")
            }
        }
    }
}

impl From<BTreeMap<Label, (u32, Weight)>> for DwaTransitionMap {
    fn from(transitions: BTreeMap<Label, (u32, Weight)>) -> Self {
        Self::Owned(transitions)
    }
}

impl FromIterator<(Label, (u32, Weight))> for DwaTransitionMap {
    fn from_iter<T: IntoIterator<Item = (Label, (u32, Weight))>>(iter: T) -> Self {
        BTreeMap::from_iter(iter).into()
    }
}

impl<'a> IntoIterator for &'a DwaTransitionMap {
    type Item = (&'a Label, &'a (u32, Weight));
    type IntoIter = std::collections::btree_map::Iter<'a, Label, (u32, Weight)>;

    fn into_iter(self) -> Self::IntoIter {
        self.deref().iter()
    }
}

impl<'a> IntoIterator for &'a mut DwaTransitionMap {
    type Item = (&'a Label, &'a mut (u32, Weight));
    type IntoIter = std::collections::btree_map::IterMut<'a, Label, (u32, Weight)>;

    fn into_iter(self) -> Self::IntoIter {
        self.deref_mut().iter_mut()
    }
}

#[derive(Debug, Clone, Default)]
pub struct DWAState {
    pub transitions: DwaTransitionMap,
    pub final_weight: Option<Weight>,
}

#[derive(Debug, Clone)]
pub struct DWA {
    states: Vec<DWAState>,
    start_state: u32,
    shared_transition_rows: bool,
    transition_count_cache: OnceLock<usize>,
    acyclic_cache: OnceLock<bool>,
}

#[derive(Debug, Clone, Copy)]
pub struct DwaStats {
    pub states: usize,
    pub transitions: usize,
    pub transition_pairs: usize,
    pub interned_ranges: usize,
}

/// Token-set identities observed cheaply while decoding the compact DWA.
/// Runtime finalization can consume this once instead of rescanning every
/// expanded transition solely to rebuild the same inventory.
#[derive(Default)]
pub struct PackedDwaTokenSetInventory {
    pub transition_sets: FxHashMap<usize, std::sync::Arc<RangeSetBlaze<u32>>>,
    pub final_sets: FxHashMap<usize, std::sync::Arc<RangeSetBlaze<u32>>>,
    pub transition_word_spans: FxHashMap<usize, u32>,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct PackedRuntimeTokenSetSpan {
    start: u32,
    len: u32,
    word_spans: u32,
}

#[derive(Debug)]
struct PackedRuntimeTokenSetChunk {
    ranges: Box<[[u32; 2]]>,
    spans: Box<[PackedRuntimeTokenSetSpan]>,
}

#[derive(Debug, Clone, Copy)]
struct PackedRuntimeTokenSetLocation {
    chunk: u32,
    local: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct PackedRuntimeWeight {
    geometry: u32,
    token_ids_start: u32,
    full: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct PackedRuntimeRow {
    labels: u32,
    targets: u32,
    weights: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct PackedRuntimeState {
    row: u32,
    final_weight: u32,
}

#[derive(Debug)]
struct PackedRuntimeSeqPool<T> {
    values: Box<[T]>,
    spans: Box<[[u32; 2]]>,
}

impl<T> PackedRuntimeSeqPool<T> {
    #[inline]
    fn len(&self) -> usize {
        self.spans.len()
    }

    #[inline]
    fn get(&self, id: u32) -> Option<&[T]> {
        let &[start, len] = self.spans.get(id as usize)?;
        let start = start as usize;
        self.values.get(start..start + len as usize)
    }
}

/// Allocation-light, read-only view of the current packed parser DWA.
///
/// This is intentionally separate from the compiler-facing `DWA`: current
/// artifacts can execute directly from canonical packed pools, while a cold
/// composition/mutation path may still materialize an ordinary `DWA` later.
#[derive(Debug)]
pub struct PackedRuntimeDwa {
    start_state: u32,
    /// Zero-copy DWF3 view used by current-format loads. Compiler-produced
    /// constraints keep the owned fields below and leave this unset.
    backed: Option<BackedPackedRuntimeDwa>,
    /// Owned narrow execution view for compiler-produced DWAs whose row/state/
    /// weight domains fit the DWF6 widths. Runtime transition/state/weight
    /// access prefers these arrays, so they are live execution storage rather
    /// than serialization-only preparation.
    owned_narrow: Option<OwnedNarrowPackedRuntimeDwa>,
    // DWF1 compatibility representation: one decoded range slab per wire
    // chunk plus token-id -> (chunk,local) indirection.
    token_set_chunks: Box<[PackedRuntimeTokenSetChunk]>,
    token_set_locations: Box<[PackedRuntimeTokenSetLocation]>,
    // DWF2 representation: one global slab and one span per token set. DWF2
    // carries per-chunk set/range counts, so the loader can allocate these
    // arrays once and decode all chunks directly into disjoint final slices.
    flat_token_ranges: Option<Box<[[u32; 2]]>>,
    /// Narrow runtime token-set slab used when every internal token id fits
    /// in u16. This is execution data, not a serialization cache: mask
    /// planning reads these ranges directly and DWF6 can persist the slab
    /// without re-encoding millions of RangeSetBlaze ranges.
    flat_token_ranges_u16: Option<Box<[[u16; 2]]>>,
    flat_token_spans: Option<Box<[PackedRuntimeTokenSetSpan]>>,
    // Freshly compiled constraints already own canonical interned token sets.
    // Keep those Arcs directly instead of immediately copying millions of
    // RangeSetBlaze ranges into a second flat slab. Loaded artifacts continue
    // to use the flat chunk/location representation above.
    materialized_token_sets: Option<Box<[Arc<RangeSetBlaze<u32>>]>>,
    materialized_token_word_spans: Option<Box<[u32]>>,
    // Freshly compiled large DWAs can build these while they already traverse
    // the canonical RangeSetBlaze token sets for runtime metadata.  This avoids
    // a second multi-million-range traversal on first save.  Loaded artifacts
    // do not need the cache because unchanged resave uses the whole-artifact
    // byte cache.
    fast_wire_token_chunks: Option<Box<[Box<[u8]>]>>,
    fast_wire_token_chunk_range_counts: Option<Box<[u32]>>,
    geometries: Box<[Vec<(u32, u32)>]>,
    weights: Box<[PackedRuntimeWeight]>,
    weight_token_ids: Box<[u32]>,
    label_pool: PackedRuntimeSeqPool<Label>,
    target_pool: PackedRuntimeSeqPool<u32>,
    weight_id_pool: PackedRuntimeSeqPool<u32>,
    rows: Box<[PackedRuntimeRow]>,
    states: Box<[PackedRuntimeState]>,
}

#[derive(Debug)]
struct OwnedNarrowPackedRuntimeDwa {
    /// Mixed live token-range slab. Compact sets use one u16 per range
    /// (13-bit start + 3-bit length code); selected large sets use two u16s per
    /// range (`start,end`) so masking can take the old flat16 fast path.
    /// `token_spans.start` is the byte offset into this slab; bit 0 marks a
    /// flat16 set (real byte offsets are always even). The exact same marked
    /// offsets and slab bytes are persisted by DWF8, so this is runtime state,
    /// not serialization preparation.
    token_ranges: Box<[u16]>,
    token_spans: Box<[PackedRuntimeTokenSetSpan]>,
    /// Per-token-set starting index into `token_range_overflows`.
    token_range_overflow_starts: Box<[u32]>,
    /// Overflow lengths for compact token ranges whose three-bit code is 7.
    /// DWF8 persists these as little-endian u16s.
    token_range_overflows: Box<[u16]>,
    /// Row-local [u16 label | u24 target | u16 weight-id] entries.
    transitions: Box<[u8]>,
    /// One packed (22-bit start, 10-bit len) word per transition row.
    spans: Box<[u32]>,
    /// One [u16 row | u16(final-weight+1)] record per state.
    states: Box<[u8]>,
    /// One [u16 geometry | u24 token-id-start | u8 full] record per weight.
    weights: Box<[u8]>,
    weight_token_ids: Box<[u16]>,
}

impl OwnedNarrowPackedRuntimeDwa {
    const TRANSITION_STRIDE: usize = 7;
    const STATE_STRIDE: usize = 4;
    const WEIGHT_STRIDE: usize = 6;

    #[inline]
    fn read_u16(bytes: &[u8], pos: usize) -> Option<u16> {
        Some(u16::from_le_bytes([
            *bytes.get(pos)?,
            *bytes.get(pos + 1)?,
        ]))
    }

    #[inline]
    fn read_u24(bytes: &[u8], pos: usize) -> Option<u32> {
        Some(
            *bytes.get(pos)? as u32
                | ((*bytes.get(pos + 1)? as u32) << 8)
                | ((*bytes.get(pos + 2)? as u32) << 16),
        )
    }

    #[inline]
    fn state(&self, id: u32) -> Option<PackedRuntimeState> {
        let pos = id as usize * Self::STATE_STRIDE;
        let row = Self::read_u16(&self.states, pos)? as u32;
        let final_plus_one = Self::read_u16(&self.states, pos + 2)?;
        Some(PackedRuntimeState {
            row,
            final_weight: if final_plus_one == 0 {
                u32::MAX
            } else {
                final_plus_one as u32 - 1
            },
        })
    }

    #[inline]
    fn span(&self, id: u32) -> Option<[u32; 2]> {
        const SPAN_LEN_BITS: u32 = 10;
        const SPAN_LEN_MASK: u32 = (1 << SPAN_LEN_BITS) - 1;
        let packed = *self.spans.get(id as usize)?;
        Some([packed >> SPAN_LEN_BITS, packed & SPAN_LEN_MASK])
    }

    #[inline]
    fn label(&self, index: usize) -> Option<Label> {
        let encoded = Self::read_u16(&self.transitions, index * Self::TRANSITION_STRIDE)?;
        Some(if encoded == u16::MAX {
            i32::MAX - 1
        } else {
            encoded as Label
        })
    }

    #[inline]
    fn target(&self, index: usize) -> Option<u32> {
        Self::read_u24(&self.transitions, index * Self::TRANSITION_STRIDE + 2)
    }

    #[inline]
    fn weight_id(&self, index: usize) -> Option<u32> {
        Self::read_u16(&self.transitions, index * Self::TRANSITION_STRIDE + 5).map(u32::from)
    }

    #[inline]
    fn weight(&self, id: u32) -> Option<PackedRuntimeWeight> {
        let pos = id as usize * Self::WEIGHT_STRIDE;
        Some(PackedRuntimeWeight {
            geometry: Self::read_u16(&self.weights, pos)? as u32,
            token_ids_start: Self::read_u24(&self.weights, pos + 2)?,
            full: *self.weights.get(pos + 5)? as u32,
        })
    }
}

#[derive(Debug)]
struct BackedPackedRuntimeDwa {
    backing: Arc<Vec<u8>>,
    section_start: usize,
    section_len: usize,
    /// DWF4/5 use a narrower directly-readable tail. DWF3 keeps the original
    /// fixed-width u32 representation.
    narrow: bool,
    /// DWF5/6 intern repeated label sequences separately from the transition
    /// stream. DWF4 keeps labels interleaved with target/weight.
    label_dedup: bool,
    /// DWF6 stores each transition as u24 target + u16 weight. DWF5 packs a
    /// 17-bit target and 15-bit weight into one u32; DWF4 uses that same packed
    /// u32 after each interleaved u16 label.
    split_target_weight: bool,
    /// DWF7 stores the owned runtime token-span table verbatim as
    /// (u32 start, u32 len, u32 word_spans) followed by the flat u16 range slab.
    /// Its remaining tail uses the ordinary wide runtime pools.
    direct_token_spans: bool,
    /// DWF8 uses the same narrow DWA tail as DWF6, but token ranges are the
    /// compact live execution stream rather than fixed u16 endpoint pairs.
    compact_token_ranges: bool,
    compact_overflow_locations_start: usize,
    compact_overflow_values_start: usize,
    compact_overflow_count: usize,
    token_set_count: usize,
    token_locations_start: usize,
    token_word_spans_start: usize,
    token_body_start: usize,
    token_body_end: usize,
    chunk_starts: Box<[u32]>,
    chunk_lens: Box<[u32]>,
    geometry_offsets: Box<[u32]>,
    /// DWF4/5/6/8 narrow geometries decoded once at load. These are tiny
    /// compared with the transition/token slabs and are hit repeatedly by
    /// mask-time weight lookup.
    narrow_geometry_spans: Box<[[u32; 2]]>,
    narrow_geometry_pairs: Box<[[u16; 2]]>,
    weights_start: usize,
    weight_count: usize,
    weight_token_ids_start: usize,
    weight_token_id_count: usize,
    label_values_start: usize,
    label_value_count: usize,
    label_spans_start: usize,
    label_span_count: usize,
    target_values_start: usize,
    target_value_count: usize,
    target_spans_start: usize,
    target_span_count: usize,
    weight_values_start: usize,
    weight_value_count: usize,
    weight_spans_start: usize,
    weight_span_count: usize,
    rows_start: usize,
    row_count: usize,
    states_start: usize,
    state_count: usize,
}

impl BackedPackedRuntimeDwa {
    fn parse(
        backing: Arc<Vec<u8>>,
        section_start: usize,
        section_len: usize,
    ) -> Result<(u32, Self), String> {
        let section_end = section_start
            .checked_add(section_len)
            .ok_or_else(|| "overflowing backed DWA range".to_owned())?;
        let input = backing
            .get(section_start..section_end)
            .ok_or_else(|| "backed DWA section is outside artifact backing".to_owned())?;
        if input.starts_with(b"DWF8") {
            return Self::parse_dwf8(backing, section_start, section_len);
        }
        if input.starts_with(b"DWF7") {
            return Self::parse_dwf7(backing, section_start, section_len);
        }
        if input.starts_with(b"DWF6") {
            return Self::parse_dwf6(backing, section_start, section_len);
        }
        if input.starts_with(b"DWF5") {
            return Self::parse_dwf5(backing, section_start, section_len);
        }
        if input.starts_with(b"DWF4") {
            return Self::parse_dwf4(backing, section_start, section_len);
        }
        Self::parse_dwf3(backing, section_start, section_len)
    }

    fn parse_dwf3(
        backing: Arc<Vec<u8>>,
        section_start: usize,
        section_len: usize,
    ) -> Result<(u32, Self), String> {
        let section_end = section_start
            .checked_add(section_len)
            .ok_or_else(|| "overflowing DWF3 backing range".to_owned())?;
        let input = backing
            .get(section_start..section_end)
            .ok_or_else(|| "DWF3 section is outside artifact backing".to_owned())?;
        if input.len() < 16 || !input.starts_with(b"DWF3") {
            return Err("invalid backed runtime DWA header".to_owned());
        }
        #[inline]
        fn read_u32(input: &[u8], pos: usize) -> Result<u32, String> {
            let bytes: [u8; 4] = input
                .get(pos..pos + 4)
                .ok_or_else(|| "truncated DWF3 u32".to_owned())?
                .try_into()
                .expect("four-byte slice");
            Ok(u32::from_le_bytes(bytes))
        }
        #[inline]
        fn skip(pos: &mut usize, bytes: usize, len: usize, what: &str) -> Result<usize, String> {
            let start = *pos;
            *pos = pos
                .checked_add(bytes)
                .ok_or_else(|| format!("overflowing DWF3 {what} range"))?;
            if *pos > len {
                return Err(format!("truncated DWF3 {what}"));
            }
            Ok(start)
        }

        let start_state = read_u32(input, 4)?;
        let chunk_count = read_u32(input, 8)? as usize;
        let token_set_count = read_u32(input, 12)? as usize;
        let mut pos = 16usize;
        let descriptor_bytes = chunk_count
            .checked_mul(8)
            .ok_or_else(|| "overflowing DWF3 chunk descriptors".to_owned())?;
        skip(&mut pos, descriptor_bytes, input.len(), "chunk descriptors")?;
        let token_locations_start = skip(
            &mut pos,
            token_set_count
                .checked_mul(4)
                .ok_or_else(|| "overflowing DWF3 token locations".to_owned())?,
            input.len(),
            "token locations",
        )?;
        let token_word_spans_start = skip(
            &mut pos,
            token_set_count
                .checked_mul(4)
                .ok_or_else(|| "overflowing DWF3 token word spans".to_owned())?,
            input.len(),
            "token word spans",
        )?;
        let mut chunk_starts = Vec::with_capacity(chunk_count);
        let mut chunk_lens = Vec::with_capacity(chunk_count);
        let mut declared_sets = 0usize;
        let mut body_pos = pos;
        for chunk in 0..chunk_count {
            let descriptor = 16 + chunk * 8;
            let body_len = read_u32(input, descriptor)? as usize;
            let set_count = read_u32(input, descriptor + 4)? as usize;
            declared_sets = declared_sets
                .checked_add(set_count)
                .ok_or_else(|| "overflowing DWF3 token-set count".to_owned())?;
            chunk_starts.push(u32::try_from(body_pos).map_err(|_| "DWF3 chunk offset exceeds u32".to_owned())?);
            chunk_lens.push(u32::try_from(body_len).map_err(|_| "DWF3 chunk length exceeds u32".to_owned())?);
            body_pos = body_pos
                .checked_add(body_len)
                .ok_or_else(|| "overflowing DWF3 chunk body range".to_owned())?;
            if body_pos > input.len() {
                return Err("truncated DWF3 chunk body".to_owned());
            }
        }
        if declared_sets != token_set_count {
            return Err("DWF3 token-set count mismatch".to_owned());
        }
        pos = body_pos;

        let geometry_count = read_u32(input, pos)? as usize;
        pos += 4;
        let mut geometry_offsets = Vec::with_capacity(geometry_count);
        for _ in 0..geometry_count {
            geometry_offsets.push(u32::try_from(pos).map_err(|_| "DWF3 geometry offset exceeds u32".to_owned())?);
            let count = read_u32(input, pos)? as usize;
            pos += 4;
            skip(
                &mut pos,
                count
                    .checked_mul(8)
                    .ok_or_else(|| "overflowing DWF3 geometry ranges".to_owned())?,
                input.len(),
                "geometry ranges",
            )?;
        }

        macro_rules! fixed_section {
            ($name:literal, $stride:expr) => {{
                let count = read_u32(input, pos)? as usize;
                pos += 4;
                let start = skip(
                    &mut pos,
                    count
                        .checked_mul($stride)
                        .ok_or_else(|| format!("overflowing DWF3 {}", $name))?,
                    input.len(),
                    $name,
                )?;
                (start, count)
            }};
        }
        let (weights_start, weight_count) = fixed_section!("weights", 12);
        let (weight_token_ids_start, weight_token_id_count) = fixed_section!("weight token ids", 4);
        let (label_values_start, label_value_count) = fixed_section!("label values", 4);
        let (label_spans_start, label_span_count) = fixed_section!("label spans", 8);
        let (target_values_start, target_value_count) = fixed_section!("target values", 4);
        let (target_spans_start, target_span_count) = fixed_section!("target spans", 8);
        let (weight_values_start, weight_value_count) = fixed_section!("weight-id values", 4);
        let (weight_spans_start, weight_span_count) = fixed_section!("weight-id spans", 8);
        if label_span_count != target_span_count || label_span_count != weight_span_count {
            return Err("DWF3 row-pool span counts do not match".to_owned());
        }
        let (rows_start, row_count) = fixed_section!("rows", 12);
        let (states_start, state_count) = fixed_section!("states", 8);
        if pos != input.len() {
            return Err("trailing bytes in DWF3 runtime DWA".to_owned());
        }
        if state_count != 0 && start_state as usize >= state_count {
            return Err("invalid DWF3 start state".to_owned());
        }

        Ok((
            start_state,
            Self {
                backing,
                section_start,
                section_len,
                narrow: false,
                label_dedup: false,
                split_target_weight: false,
                direct_token_spans: false,
                compact_token_ranges: false,
                compact_overflow_locations_start: 0,
                compact_overflow_values_start: 0,
                compact_overflow_count: 0,
                token_set_count,
                token_locations_start,
                token_word_spans_start,
                token_body_start: 0,
                token_body_end: 0,
                chunk_starts: chunk_starts.into_boxed_slice(),
                chunk_lens: chunk_lens.into_boxed_slice(),
                geometry_offsets: geometry_offsets.into_boxed_slice(),
                narrow_geometry_spans: Box::new([]),
                narrow_geometry_pairs: Box::new([]),
                weights_start,
                weight_count,
                weight_token_ids_start,
                weight_token_id_count,
                label_values_start,
                label_value_count,
                label_spans_start,
                label_span_count,
                target_values_start,
                target_value_count,
                target_spans_start,
                target_span_count,
                weight_values_start,
                weight_value_count,
                weight_spans_start,
                weight_span_count,
                rows_start,
                row_count,
                states_start,
                state_count,
            },
        ))
    }

    /// DWF7 is the runtime-shaped zero-copy layout used by compiler-created
    /// packed DWAs with a u16 token-range slab. Token spans and ranges are
    /// persisted exactly in their runtime widths; the remaining DWA tail keeps
    /// the ordinary wide fixed-width pools from DWF3. That makes fresh save a
    /// sequence of bulk copies rather than a narrowing/repacking pass.
    fn parse_dwf7(
        backing: Arc<Vec<u8>>,
        section_start: usize,
        section_len: usize,
    ) -> Result<(u32, Self), String> {
        let section_end = section_start
            .checked_add(section_len)
            .ok_or_else(|| "overflowing DWF7 backing range".to_owned())?;
        let input = backing
            .get(section_start..section_end)
            .ok_or_else(|| "DWF7 section is outside artifact backing".to_owned())?;
        if input.len() < 16 || !input.starts_with(b"DWF7") {
            return Err("invalid DWF7 runtime DWA header".to_owned());
        }
        #[inline]
        fn read_u32(input: &[u8], pos: usize) -> Result<u32, String> {
            let bytes: [u8; 4] = input
                .get(pos..pos + 4)
                .ok_or_else(|| "truncated DWF7 u32".to_owned())?
                .try_into()
                .expect("four-byte slice");
            Ok(u32::from_le_bytes(bytes))
        }
        #[inline]
        fn skip(pos: &mut usize, bytes: usize, len: usize, what: &str) -> Result<usize, String> {
            let start = *pos;
            *pos = pos
                .checked_add(bytes)
                .ok_or_else(|| format!("overflowing DWF7 {what} range"))?;
            if *pos > len {
                return Err(format!("truncated DWF7 {what}"));
            }
            Ok(start)
        }

        let start_state = read_u32(input, 4)?;
        let token_set_count = read_u32(input, 8)? as usize;
        let token_range_count = read_u32(input, 12)? as usize;
        let mut pos = 16usize;
        let token_locations_start = skip(
            &mut pos,
            token_set_count
                .checked_mul(12)
                .ok_or_else(|| "overflowing DWF7 token spans".to_owned())?,
            input.len(),
            "token spans",
        )?;
        let token_body_start = pos;
        skip(
            &mut pos,
            token_range_count
                .checked_mul(4)
                .ok_or_else(|| "overflowing DWF7 token ranges".to_owned())?,
            input.len(),
            "token ranges",
        )?;
        let token_body_end = pos;
        for index in 0..token_set_count {
            let span = token_locations_start + index * 12;
            let start = read_u32(input, span)? as usize;
            let len = read_u32(input, span + 4)? as usize;
            let end = start
                .checked_add(len)
                .ok_or_else(|| "overflowing DWF7 token span".to_owned())?;
            if end > token_range_count {
                return Err("DWF7 token span is outside range slab".to_owned());
            }
        }

        let geometry_count = read_u32(input, pos)? as usize;
        pos += 4;
        let mut geometry_offsets = Vec::with_capacity(geometry_count);
        for _ in 0..geometry_count {
            geometry_offsets.push(
                u32::try_from(pos).map_err(|_| "DWF7 geometry offset exceeds u32".to_owned())?,
            );
            let count = read_u32(input, pos)? as usize;
            pos += 4;
            skip(
                &mut pos,
                count
                    .checked_mul(8)
                    .ok_or_else(|| "overflowing DWF7 geometry ranges".to_owned())?,
                input.len(),
                "geometry ranges",
            )?;
        }

        macro_rules! fixed_section {
            ($name:literal, $stride:expr) => {{
                let count = read_u32(input, pos)? as usize;
                pos += 4;
                let start = skip(
                    &mut pos,
                    count
                        .checked_mul($stride)
                        .ok_or_else(|| format!("overflowing DWF7 {}", $name))?,
                    input.len(),
                    $name,
                )?;
                (start, count)
            }};
        }
        let (weights_start, weight_count) = fixed_section!("weights", 12);
        let (weight_token_ids_start, weight_token_id_count) =
            fixed_section!("weight token ids", 4);
        let (label_values_start, label_value_count) = fixed_section!("label values", 4);
        let (label_spans_start, label_span_count) = fixed_section!("label spans", 8);
        let (target_values_start, target_value_count) = fixed_section!("target values", 4);
        let (target_spans_start, target_span_count) = fixed_section!("target spans", 8);
        let (weight_values_start, weight_value_count) =
            fixed_section!("weight-id values", 4);
        let (weight_spans_start, weight_span_count) = fixed_section!("weight-id spans", 8);
        if label_span_count != target_span_count || label_span_count != weight_span_count {
            return Err("DWF7 row-pool span counts do not match".to_owned());
        }
        let (rows_start, row_count) = fixed_section!("rows", 12);
        let (states_start, state_count) = fixed_section!("states", 8);
        if pos != input.len() {
            return Err("trailing bytes in DWF7 runtime DWA".to_owned());
        }
        if state_count != 0 && start_state as usize >= state_count {
            return Err("invalid DWF7 start state".to_owned());
        }

        Ok((
            start_state,
            Self {
                backing,
                section_start,
                section_len,
                narrow: false,
                label_dedup: false,
                split_target_weight: false,
                direct_token_spans: true,
                compact_token_ranges: false,
                compact_overflow_locations_start: 0,
                compact_overflow_values_start: 0,
                compact_overflow_count: 0,
                token_set_count,
                token_locations_start,
                token_word_spans_start: 0,
                token_body_start,
                token_body_end,
                chunk_starts: Box::new([]),
                chunk_lens: Box::new([]),
                geometry_offsets: geometry_offsets.into_boxed_slice(),
                narrow_geometry_spans: Box::new([]),
                narrow_geometry_pairs: Box::new([]),
                weights_start,
                weight_count,
                weight_token_ids_start,
                weight_token_id_count,
                label_values_start,
                label_value_count,
                label_spans_start,
                label_span_count,
                target_values_start,
                target_value_count,
                target_spans_start,
                target_span_count,
                weight_values_start,
                weight_value_count,
                weight_spans_start,
                weight_span_count,
                rows_start,
                row_count,
                states_start,
                state_count,
            },
        ))
    }

    /// DWF4 is the narrow zero-copy wire layout. It deliberately has no
    /// variable-width schema: writers only select it when every field fits the
    /// fixed widths below, otherwise they fall back to DWF3. That keeps hot
    /// runtime access branch-light and makes malformed-artifact validation
    /// straightforward.
    fn parse_dwf4(
        backing: Arc<Vec<u8>>,
        section_start: usize,
        section_len: usize,
    ) -> Result<(u32, Self), String> {
        Self::parse_narrow(backing, section_start, section_len, false, false, false)
    }

    fn parse_dwf5(
        backing: Arc<Vec<u8>>,
        section_start: usize,
        section_len: usize,
    ) -> Result<(u32, Self), String> {
        Self::parse_narrow(backing, section_start, section_len, true, false, false)
    }

    fn parse_dwf6(
        backing: Arc<Vec<u8>>,
        section_start: usize,
        section_len: usize,
    ) -> Result<(u32, Self), String> {
        // DWF6 favors fresh-save speed over the DWF5 label dictionary: each
        // transition is one row-local u16 label + u24 target + u16 weight.
        // This keeps the wire directly readable while avoiding a whole-DWA
        // hash/intern pass during every independent save.
        Self::parse_narrow(backing, section_start, section_len, false, true, false)
    }

    fn parse_dwf8(
        backing: Arc<Vec<u8>>,
        section_start: usize,
        section_len: usize,
    ) -> Result<(u32, Self), String> {
        Self::parse_narrow(backing, section_start, section_len, false, true, true)
    }

    fn parse_narrow(
        backing: Arc<Vec<u8>>,
        section_start: usize,
        section_len: usize,
        label_dedup: bool,
        split_target_weight: bool,
        compact_token_ranges: bool,
    ) -> Result<(u32, Self), String> {
        let section_end = section_start
            .checked_add(section_len)
            .ok_or_else(|| "overflowing DWF4 backing range".to_owned())?;
        let input = backing
            .get(section_start..section_end)
            .ok_or_else(|| "DWF4 section is outside artifact backing".to_owned())?;
        let expected_magic = if compact_token_ranges {
            b"DWF8"
        } else if split_target_weight {
            b"DWF6"
        } else if label_dedup {
            b"DWF5"
        } else {
            b"DWF4"
        };
        if input.len() < 16 || !input.starts_with(expected_magic) {
            return Err("invalid backed narrow runtime DWA header".to_owned());
        }
        #[inline]
        fn read_u16(input: &[u8], pos: usize) -> Result<u16, String> {
            let bytes: [u8; 2] = input
                .get(pos..pos + 2)
                .ok_or_else(|| "truncated DWF4 u16".to_owned())?
                .try_into()
                .expect("two-byte slice");
            Ok(u16::from_le_bytes(bytes))
        }
        #[inline]
        fn read_u24(input: &[u8], pos: usize) -> Result<u32, String> {
            let bytes = input
                .get(pos..pos + 3)
                .ok_or_else(|| "truncated DWF4 u24".to_owned())?;
            Ok(bytes[0] as u32 | ((bytes[1] as u32) << 8) | ((bytes[2] as u32) << 16))
        }
        #[inline]
        fn read_u32(input: &[u8], pos: usize) -> Result<u32, String> {
            let bytes: [u8; 4] = input
                .get(pos..pos + 4)
                .ok_or_else(|| "truncated DWF4 u32".to_owned())?
                .try_into()
                .expect("four-byte slice");
            Ok(u32::from_le_bytes(bytes))
        }
        #[inline]
        fn skip(pos: &mut usize, bytes: usize, len: usize, what: &str) -> Result<usize, String> {
            let start = *pos;
            *pos = pos
                .checked_add(bytes)
                .ok_or_else(|| format!("overflowing DWF4 {what} range"))?;
            if *pos > len {
                return Err(format!("truncated DWF4 {what}"));
            }
            Ok(start)
        }

        let start_state = read_u32(input, 4)?;
        let token_set_count = read_u32(input, 8)? as usize;
        let token_body_len = read_u32(input, 12)? as usize;
        let mut pos = 16usize;
        let token_locations_start = skip(
            &mut pos,
            token_set_count
                .checked_mul(3)
                .ok_or_else(|| "overflowing DWF4 token locations".to_owned())?,
            input.len(),
            "token locations",
        )?;
        let compact_overflow_locations_start = if compact_token_ranges {
            skip(
                &mut pos,
                token_set_count
                    .checked_mul(3)
                    .ok_or_else(|| "overflowing DWF8 overflow locations".to_owned())?,
                input.len(),
                "compact overflow locations",
            )?
        } else {
            0
        };
        let token_word_spans_start = skip(
            &mut pos,
            token_set_count
                .checked_mul(2)
                .ok_or_else(|| "overflowing DWF4 token word spans".to_owned())?,
            input.len(),
            "token word spans",
        )?;
        let token_body_start = pos;
        skip(&mut pos, token_body_len, input.len(), "token body")?;
        let token_body_end = pos;

        let (compact_overflow_values_start, compact_overflow_count) = if compact_token_ranges {
            if token_body_len % 2 != 0 {
                return Err("DWF8 compact token body is not u16-aligned".to_owned());
            }
            let overflow_count = read_u32(input, pos)? as usize;
            pos += 4;
            let values_start = skip(
                &mut pos,
                overflow_count
                    .checked_mul(2)
                    .ok_or_else(|| "overflowing DWF8 compact overflow stream".to_owned())?,
                input.len(),
                "compact overflow stream",
            )?;

            // Validate only the tiny per-set offset tables. The multi-megabyte
            // range body remains zero-copy and is decoded lazily by masking.
            let mut previous_range = 0usize;
            let mut previous_overflow = 0usize;
            for index in 0..token_set_count {
                let range_location = read_u24(input, token_locations_start + index * 3)? as usize;
                let range_start = range_location & !1usize;
                let overflow_start =
                    read_u24(input, compact_overflow_locations_start + index * 3)? as usize;
                if range_start < previous_range
                    || range_start > token_body_len
                    || range_start % 2 != 0
                    || overflow_start < previous_overflow
                    || overflow_start > overflow_count
                {
                    return Err("invalid DWF8 compact token offsets".to_owned());
                }
                previous_range = range_start;
                previous_overflow = overflow_start;
            }
            (values_start, overflow_count)
        } else {
            (0, 0)
        };

        let geometry_count = read_u32(input, pos)? as usize;
        pos += 4;
        let mut geometry_offsets = Vec::with_capacity(geometry_count);
        let mut narrow_geometry_spans = Vec::<[u32; 2]>::with_capacity(geometry_count);
        let mut narrow_geometry_pairs = Vec::<[u16; 2]>::new();
        for _ in 0..geometry_count {
            geometry_offsets.push(
                u32::try_from(pos).map_err(|_| "DWF4 geometry offset exceeds u32".to_owned())?,
            );
            let count = read_u16(input, pos)? as usize;
            pos += 2;
            let pair_start = narrow_geometry_pairs.len();
            for index in 0..count {
                let pair = pos + index * 4;
                narrow_geometry_pairs.push([read_u16(input, pair)?, read_u16(input, pair + 2)?]);
            }
            narrow_geometry_spans.push([
                u32::try_from(pair_start)
                    .map_err(|_| "DWF4 geometry pair offset exceeds u32".to_owned())?,
                u32::try_from(count)
                    .map_err(|_| "DWF4 geometry pair count exceeds u32".to_owned())?,
            ]);
            skip(
                &mut pos,
                count
                    .checked_mul(4)
                    .ok_or_else(|| "overflowing DWF4 geometry ranges".to_owned())?,
                input.len(),
                "geometry ranges",
            )?;
        }

        macro_rules! fixed_section {
            ($name:literal, $stride:expr) => {{
                let count = read_u32(input, pos)? as usize;
                pos += 4;
                let start = skip(
                    &mut pos,
                    count
                        .checked_mul($stride)
                        .ok_or_else(|| format!("overflowing DWF4 {}", $name))?,
                    input.len(),
                    $name,
                )?;
                (start, count)
            }};
        }
        let (weights_start, weight_count) = fixed_section!("weights", 6);
        let (weight_token_ids_start, weight_token_id_count) =
            fixed_section!("weight token ids", 2);
        let (
            label_values_start,
            label_value_count,
            label_spans_start,
            label_span_count,
            target_values_start,
            target_value_count,
            target_spans_start,
            target_span_count,
            weight_values_start,
            weight_value_count,
            weight_spans_start,
            weight_span_count,
            rows_start,
            row_count,
        ) = if label_dedup {
            // DWF5 stores the heavily repeated sorted label vectors once. Each
            // row carries only a u16 label-vector id; target/weight entries keep
            // their row-local order and share one packed span table.
            let (label_values_start, label_value_count) = fixed_section!("label values", 2);
            let (label_spans_start, label_span_count) = fixed_section!("label spans", 4);
            let (rows_start, row_count) = fixed_section!("row label ids", 2);
            let transition_stride = if split_target_weight { 5 } else { 4 };
            let (target_values_start, target_value_count) =
                fixed_section!("target-weight entries", transition_stride);
            let (target_spans_start, target_span_count) = fixed_section!("transition spans", 4);
            if row_count != target_span_count {
                return Err("DWF5 row/span count mismatch".to_owned());
            }
            (
                label_values_start,
                label_value_count,
                label_spans_start,
                label_span_count,
                target_values_start,
                target_value_count,
                target_spans_start,
                target_span_count,
                target_values_start,
                target_value_count,
                target_spans_start,
                target_span_count,
                rows_start,
                row_count,
            )
        } else {
            // DWF4 interleaves u16 label + packed u32(target, weight). DWF6
            // widens that row-local record to u16 label + u24 target + u16
            // weight. All three compiler pools still share one identity
            // row/span mapping, so no label dictionary is required.
            let transition_stride = if split_target_weight { 7 } else { 6 };
            let (label_values_start, label_value_count) =
                fixed_section!("transition entries", transition_stride);
            let (label_spans_start, label_span_count) = fixed_section!("transition spans", 4);
            (
                label_values_start,
                label_value_count,
                label_spans_start,
                label_span_count,
                label_values_start,
                label_value_count,
                label_spans_start,
                label_span_count,
                label_values_start,
                label_value_count,
                label_spans_start,
                label_span_count,
                0,
                label_span_count,
            )
        };
        let (states_start, state_count) = fixed_section!("states", 4);
        if pos != input.len() {
            return Err("trailing bytes in DWF4 runtime DWA".to_owned());
        }
        if state_count != 0 && start_state as usize >= state_count {
            return Err("invalid DWF4 start state".to_owned());
        }

        Ok((
            start_state,
            Self {
                backing,
                section_start,
                section_len,
                narrow: true,
                label_dedup,
                split_target_weight,
                direct_token_spans: false,
                compact_token_ranges,
                compact_overflow_locations_start,
                compact_overflow_values_start,
                compact_overflow_count,
                token_set_count,
                token_locations_start,
                token_word_spans_start,
                token_body_start,
                token_body_end,
                chunk_starts: Box::new([]),
                chunk_lens: Box::new([]),
                geometry_offsets: geometry_offsets.into_boxed_slice(),
                narrow_geometry_spans: narrow_geometry_spans.into_boxed_slice(),
                narrow_geometry_pairs: narrow_geometry_pairs.into_boxed_slice(),
                weights_start,
                weight_count,
                weight_token_ids_start,
                weight_token_id_count,
                label_values_start,
                label_value_count,
                label_spans_start,
                label_span_count,
                target_values_start,
                target_value_count,
                target_spans_start,
                target_span_count,
                weight_values_start,
                weight_value_count,
                weight_spans_start,
                weight_span_count,
                rows_start,
                row_count,
                states_start,
                state_count,
            },
        ))
    }

    #[inline(always)]
    fn read_u32(&self, relative: usize) -> Option<u32> {
        if relative.checked_add(4)? > self.section_len {
            return None;
        }
        let ptr = unsafe {
            self.backing
                .as_ptr()
                .add(self.section_start + relative)
                .cast::<u32>()
        };
        Some(u32::from_le(unsafe { std::ptr::read_unaligned(ptr) }))
    }

    #[inline(always)]
    fn read_u16(&self, relative: usize) -> Option<u16> {
        if relative.checked_add(2)? > self.section_len {
            return None;
        }
        let ptr = unsafe {
            self.backing
                .as_ptr()
                .add(self.section_start + relative)
                .cast::<u16>()
        };
        Some(u16::from_le(unsafe { std::ptr::read_unaligned(ptr) }))
    }

    #[inline(always)]
    fn read_u24(&self, relative: usize) -> Option<u32> {
        if relative.checked_add(3)? > self.section_len {
            return None;
        }
        let ptr = unsafe { self.backing.as_ptr().add(self.section_start + relative) };
        let lo = u16::from_le(unsafe { std::ptr::read_unaligned(ptr.cast::<u16>()) }) as u32;
        let hi = unsafe { *ptr.add(2) } as u32;
        Some(lo | (hi << 16))
    }

    #[inline(always)]
    fn read_i32(&self, relative: usize) -> Option<i32> {
        self.read_u32(relative).map(|value| value as i32)
    }

    #[inline]
    fn weight(&self, id: u32) -> Option<PackedRuntimeWeight> {
        let index = id as usize;
        if index >= self.weight_count {
            return None;
        }
        if self.narrow {
            let pos = self.weights_start + index * 6;
            return Some(PackedRuntimeWeight {
                geometry: self.read_u16(pos)? as u32,
                token_ids_start: self.read_u24(pos + 2)?,
                full: *self.backing.get(self.section_start + pos + 5)? as u32,
            });
        }
        let pos = self.weights_start + index * 12;
        Some(PackedRuntimeWeight {
            geometry: self.read_u32(pos)?,
            token_ids_start: self.read_u32(pos + 4)?,
            full: self.read_u32(pos + 8)?,
        })
    }

    #[inline]
    fn row(&self, id: u32) -> Option<PackedRuntimeRow> {
        let index = id as usize;
        if index >= self.row_count {
            return None;
        }
        if self.narrow {
            let labels = if self.label_dedup {
                self.read_u16(self.rows_start + index * 2)? as u32
            } else {
                index as u32
            };
            return Some(PackedRuntimeRow {
                labels,
                targets: index as u32,
                weights: index as u32,
            });
        }
        let pos = self.rows_start + index * 12;
        Some(PackedRuntimeRow {
            labels: self.read_u32(pos)?,
            targets: self.read_u32(pos + 4)?,
            weights: self.read_u32(pos + 8)?,
        })
    }

    #[inline]
    fn state(&self, id: u32) -> Option<PackedRuntimeState> {
        let index = id as usize;
        if index >= self.state_count {
            return None;
        }
        if self.narrow {
            let pos = self.states_start + index * 4;
            let final_encoded = self.read_u16(pos + 2)? as u32;
            return Some(PackedRuntimeState {
                row: self.read_u16(pos)? as u32,
                final_weight: if final_encoded == 0 {
                    u32::MAX
                } else {
                    final_encoded - 1
                },
            });
        }
        let pos = self.states_start + index * 8;
        Some(PackedRuntimeState {
            row: self.read_u32(pos)?,
            final_weight: self.read_u32(pos + 4)?,
        })
    }

    #[inline]
    fn span(&self, start: usize, count: usize, id: u32) -> Option<[u32; 2]> {
        let index = id as usize;
        if index >= count {
            return None;
        }
        if self.narrow {
            let packed = self.read_u32(start + index * 4)?;
            return Some([packed >> 10, packed & 0x3ff]);
        }
        let pos = start + index * 8;
        Some([self.read_u32(pos)?, self.read_u32(pos + 4)?])
    }

    #[inline]
    fn geometry_len(&self, id: u32) -> Option<usize> {
        if self.narrow {
            return self
                .narrow_geometry_spans
                .get(id as usize)
                .map(|span| span[1] as usize);
        }
        let pos = *self.geometry_offsets.get(id as usize)? as usize;
        Some(self.read_u32(pos)? as usize)
    }

    #[inline]
    fn geometry_pair(&self, id: u32, index: usize) -> Option<(u32, u32)> {
        if self.narrow {
            let [start, len] = *self.narrow_geometry_spans.get(id as usize)?;
            if index >= len as usize {
                return None;
            }
            let pair = *self
                .narrow_geometry_pairs
                .get(start as usize + index)?;
            return Some((pair[0] as u32, pair[1] as u32));
        }
        let pos = *self.geometry_offsets.get(id as usize)? as usize;
        let len = self.read_u32(pos)? as usize;
        if index >= len {
            return None;
        }
        let pair = pos + 4 + index * 8;
        Some((self.read_u32(pair)?, self.read_u32(pair + 4)?))
    }

    #[inline]
    fn weight_token_id(&self, index: usize) -> Option<u32> {
        if self.narrow {
            return (index < self.weight_token_id_count)
                .then(|| self.read_u16(self.weight_token_ids_start + index * 2).map(u32::from))
                .flatten();
        }
        (index < self.weight_token_id_count)
            .then(|| self.read_u32(self.weight_token_ids_start + index * 4))
            .flatten()
    }

    #[inline]
    fn pool_u32(&self, start: usize, count: usize, index: usize) -> Option<u32> {
        (index < count)
            .then(|| self.read_u32(start + index * 4))
            .flatten()
    }

    #[inline]
    fn pool_i32(&self, start: usize, count: usize, index: usize) -> Option<i32> {
        (index < count)
            .then(|| self.read_i32(start + index * 4))
            .flatten()
    }

    #[inline]
    fn token_bytes(&self, id: u32) -> Option<(&[u8], u32)> {
        let id = id as usize;
        if id >= self.token_set_count {
            return None;
        }
        if self.direct_token_spans {
            let span = self.token_locations_start + id * 12;
            let start = self.read_u32(span)? as usize;
            let len = self.read_u32(span + 4)? as usize;
            let word_spans = self.read_u32(span + 8)?;
            let byte_start = start.checked_mul(4)?;
            let byte_end = start.checked_add(len)?.checked_mul(4)?;
            let body_len = self.token_body_end.checked_sub(self.token_body_start)?;
            if byte_end > body_len {
                return None;
            }
            let absolute_start = self.section_start + self.token_body_start + byte_start;
            let absolute_end = self.section_start + self.token_body_start + byte_end;
            return Some((self.backing.get(absolute_start..absolute_end)?, word_spans));
        }
        if self.narrow {
            let local = self.read_u24(self.token_locations_start + id * 3)? as usize;
            let body_len = self.token_body_end.checked_sub(self.token_body_start)?;
            if local >= body_len {
                return None;
            }
            let absolute = self.section_start + self.token_body_start + local;
            let body_end = self.section_start + self.token_body_end;
            let bytes = self.backing.get(absolute..body_end)?;
            let word_spans = self.read_u16(self.token_word_spans_start + id * 2)? as u32;
            return Some((bytes, word_spans));
        }
        let location = self.read_u32(self.token_locations_start + id * 4)?;
        let chunk = (location >> 24) as usize;
        let local = (location & 0x00ff_ffff) as usize;
        let chunk_start = *self.chunk_starts.get(chunk)? as usize;
        let chunk_len = *self.chunk_lens.get(chunk)? as usize;
        if local >= chunk_len {
            return None;
        }
        let absolute = self.section_start + chunk_start + local;
        let chunk_end = self.section_start + chunk_start + chunk_len;
        let bytes = self.backing.get(absolute..chunk_end)?;
        let word_spans = self.read_u32(self.token_word_spans_start + id * 4)?;
        Some((bytes, word_spans))
    }
}

#[derive(Clone, Copy)]
pub struct PackedRuntimeTokenSetRef<'a> {
    id: u32,
    storage: PackedRuntimeTokenSetStorageRef<'a>,
    word_spans: u32,
}

#[derive(Clone, Copy)]
enum PackedRuntimeTokenSetStorageRef<'a> {
    Flat(&'a [[u32; 2]]),
    Flat16(&'a [[u16; 2]]),
    BackedFlat16(&'a [u8]),
    Compact16 {
        ranges: &'a [u16],
        overflows: &'a [u16],
    },
    Compact {
        bytes: &'a [u8],
        range_count: u32,
        overflows: &'a [u8],
    },
    Varint {
        bytes: &'a [u8],
        range_count: u32,
    },
    Materialized(&'a Arc<RangeSetBlaze<u32>>),
}

impl<'a> PackedRuntimeTokenSetRef<'a> {
    #[inline]
    pub fn id(self) -> u32 {
        self.id
    }

    #[inline]
    pub fn range_count(self) -> usize {
        match self.storage {
            PackedRuntimeTokenSetStorageRef::Flat(ranges) => ranges.len(),
            PackedRuntimeTokenSetStorageRef::Flat16(ranges) => ranges.len(),
            PackedRuntimeTokenSetStorageRef::BackedFlat16(bytes) => bytes.len() / 4,
            PackedRuntimeTokenSetStorageRef::Compact16 { ranges, .. } => ranges.len(),
            PackedRuntimeTokenSetStorageRef::Compact { range_count, .. } => range_count as usize,
            PackedRuntimeTokenSetStorageRef::Varint { range_count, .. } => range_count as usize,
            PackedRuntimeTokenSetStorageRef::Materialized(tokens) => tokens.ranges().len(),
        }
    }

    #[inline]
    pub fn word_spans(self) -> u32 {
        self.word_spans
    }

    #[inline]
    pub fn materialized_arc(self) -> Option<&'a Arc<RangeSetBlaze<u32>>> {
        match self.storage {
            PackedRuntimeTokenSetStorageRef::Materialized(tokens) => Some(tokens),
            PackedRuntimeTokenSetStorageRef::Flat(_)
            | PackedRuntimeTokenSetStorageRef::Flat16(_)
            | PackedRuntimeTokenSetStorageRef::BackedFlat16(_)
            | PackedRuntimeTokenSetStorageRef::Compact16 { .. }
            | PackedRuntimeTokenSetStorageRef::Compact { .. }
            | PackedRuntimeTokenSetStorageRef::Varint { .. } => None,
        }
    }

    #[inline]
    pub fn for_each_range(self, mut f: impl FnMut(u32, u32)) {
        match self.storage {
            PackedRuntimeTokenSetStorageRef::Flat(ranges) => {
                for &[start, end] in ranges {
                    f(start, end);
                }
            }
            PackedRuntimeTokenSetStorageRef::Flat16(ranges) => {
                for &[start, end] in ranges {
                    f(start as u32, end as u32);
                }
            }
            PackedRuntimeTokenSetStorageRef::BackedFlat16(bytes) => {
                for pair in bytes.chunks_exact(4) {
                    let start = u16::from_le_bytes([pair[0], pair[1]]) as u32;
                    let end = u16::from_le_bytes([pair[2], pair[3]]) as u32;
                    f(start, end);
                }
            }
            PackedRuntimeTokenSetStorageRef::Compact16 { ranges, overflows } => {
                let mut overflow_pos = 0usize;
                for &packed in ranges {
                    let start = (packed & 0x1fff) as u32;
                    let code = packed >> 13;
                    let len = if code != 7 {
                        code as u32
                    } else {
                        let Some(&len) = overflows.get(overflow_pos) else {
                            return;
                        };
                        overflow_pos += 1;
                        len as u32
                    };
                    f(start, start + len);
                }
            }
            PackedRuntimeTokenSetStorageRef::Compact {
                bytes,
                range_count,
                overflows,
            } => {
                let range_count = range_count as usize;
                if bytes.len() < range_count.saturating_mul(2) {
                    return;
                }
                let mut overflow_pos = 0usize;
                let range_base = bytes.as_ptr();
                let overflow_base = overflows.as_ptr();
                let overflow_count = overflows.len() / 2;
                for index in 0..range_count {
                    // The compact wire is intentionally only two-byte aligned.
                    // `read_unaligned` avoids constructing a temporary byte pair
                    // for every hot-path range while remaining valid for backed
                    // artifact storage with arbitrary base alignment.
                    let packed = unsafe {
                        std::ptr::read_unaligned(range_base.add(index * 2).cast::<u16>())
                    };
                    let packed = u16::from_le(packed);
                    let start = (packed & 0x1fff) as u32;
                    let code = packed >> 13;
                    let len = if code != 7 {
                        code as u32
                    } else {
                        if overflow_pos >= overflow_count {
                            return;
                        }
                        let len = unsafe {
                            std::ptr::read_unaligned(
                                overflow_base.add(overflow_pos * 2).cast::<u16>(),
                            )
                        };
                        overflow_pos += 1;
                        u16::from_le(len) as u32
                    };
                    f(start, start + len);
                }
            }
            PackedRuntimeTokenSetStorageRef::Varint {
                bytes,
                range_count,
            } => {
                let mut pos = 0usize;
                let mut previous_end_plus_one = 0u64;
                for _ in 0..range_count {
                    let Ok(gap) = take_var_u64(bytes, &mut pos) else {
                        return;
                    };
                    let Ok(len) = take_var_u32(bytes, &mut pos) else {
                        return;
                    };
                    let Some(lo64) = previous_end_plus_one.checked_add(gap) else {
                        return;
                    };
                    let Ok(lo) = u32::try_from(lo64) else {
                        return;
                    };
                    let Some(hi) = lo.checked_add(len) else {
                        return;
                    };
                    f(lo, hi);
                    previous_end_plus_one = hi as u64 + 1;
                }
            }
            PackedRuntimeTokenSetStorageRef::Materialized(tokens) => {
                for range in tokens.ranges() {
                    f(*range.start(), *range.end());
                }
            }
        }
    }

}

#[derive(Clone, Copy)]
pub struct PackedRuntimeWeightRef<'a> {
    dwa: &'a PackedRuntimeDwa,
    id: u32,
}

pub struct PackedRuntimeWeightEntries<'a> {
    dwa: &'a PackedRuntimeDwa,
    weight: PackedRuntimeWeight,
    index: usize,
    len: usize,
}

impl<'a> Iterator for PackedRuntimeWeightEntries<'a> {
    type Item = ((u32, u32), PackedRuntimeTokenSetRef<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }
        let index = self.index;
        self.index += 1;
        let range = self.dwa.geometry_pair_at(self.weight.geometry, index)?;
        let token_id = self
            .dwa
            .weight_token_id_at(self.weight.token_ids_start as usize + index)?;
        Some((range, self.dwa.token_set(token_id)?))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len.saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a> PackedRuntimeWeightRef<'a> {
    #[inline]
    pub fn id(self) -> u32 {
        self.id
    }

    #[inline]
    pub fn is_full(self) -> bool {
        self.dwa
            .weight_record(self.id)
            .is_some_and(|weight| weight.full != 0)
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        let Some(weight) = self.dwa.weight_record(self.id) else {
            return true;
        };
        weight.full == 0 && self.dwa.geometry_len_at(weight.geometry).unwrap_or(0) == 0
    }

    pub fn token_set_for_tsid(self, tsid: u32) -> Option<PackedRuntimeTokenSetRef<'a>> {
        let weight = self.dwa.weight_record(self.id)?;
        if weight.full != 0 {
            return None;
        }
        let mut low = 0usize;
        let mut high = self.dwa.geometry_len_at(weight.geometry)?;
        let index = loop {
            if low >= high {
                return None;
            }
            let mid = (low + high) / 2;
            let (start, end) = self.dwa.geometry_pair_at(weight.geometry, mid)?;
            match (start, end) {
                (start, _) if tsid < start => high = mid,
                (_, end) if tsid > end => low = mid + 1,
                _ => break mid,
            }
        };
        let token_id = self
            .dwa
            .weight_token_id_at(weight.token_ids_start as usize + index)?;
        self.dwa.token_set(token_id)
    }

    pub fn entries(self) -> PackedRuntimeWeightEntries<'a> {
        let weight = self.dwa.weight_record(self.id).unwrap_or(PackedRuntimeWeight {
            geometry: 0,
            token_ids_start: 0,
            full: 1,
        });
        let len = if weight.full != 0 {
            0
        } else {
            self.dwa.geometry_len_at(weight.geometry).unwrap_or(0)
        };
        PackedRuntimeWeightEntries {
            dwa: self.dwa,
            weight,
            index: 0,
            len,
        }
    }
}

impl PackedRuntimeDwa {
    #[inline]
    fn weight_record(&self, id: u32) -> Option<PackedRuntimeWeight> {
        if let Some(narrow) = &self.owned_narrow {
            return narrow.weight(id);
        }
        self.backed
            .as_ref()
            .map_or_else(|| self.weights.get(id as usize).copied(), |backed| backed.weight(id))
    }

    #[inline]
    fn state_record(&self, id: u32) -> Option<PackedRuntimeState> {
        if let Some(narrow) = &self.owned_narrow {
            return narrow.state(id);
        }
        self.backed
            .as_ref()
            .map_or_else(|| self.states.get(id as usize).copied(), |backed| backed.state(id))
    }

    #[inline]
    fn row_record(&self, id: u32) -> Option<PackedRuntimeRow> {
        if let Some(narrow) = &self.owned_narrow {
            if id as usize >= narrow.spans.len() {
                return None;
            }
            return Some(PackedRuntimeRow {
                labels: id,
                targets: id,
                weights: id,
            });
        }
        self.backed
            .as_ref()
            .map_or_else(|| self.rows.get(id as usize).copied(), |backed| backed.row(id))
    }

    #[inline]
    fn geometry_len_at(&self, id: u32) -> Option<usize> {
        self.backed.as_ref().map_or_else(
            || self.geometries.get(id as usize).map(Vec::len),
            |backed| backed.geometry_len(id),
        )
    }

    #[inline]
    fn geometry_pair_at(&self, id: u32, index: usize) -> Option<(u32, u32)> {
        self.backed.as_ref().map_or_else(
            || self.geometries.get(id as usize)?.get(index).copied(),
            |backed| backed.geometry_pair(id, index),
        )
    }

    #[inline]
    fn weight_token_id_at(&self, index: usize) -> Option<u32> {
        if let Some(narrow) = &self.owned_narrow {
            return narrow.weight_token_ids.get(index).copied().map(u32::from);
        }
        self.backed.as_ref().map_or_else(
            || self.weight_token_ids.get(index).copied(),
            |backed| backed.weight_token_id(index),
        )
    }

    #[inline]
    fn pool_span_at(&self, kind: u8, id: u32) -> Option<[u32; 2]> {
        if let Some(narrow) = &self.owned_narrow {
            return (kind <= 2).then(|| narrow.span(id)).flatten();
        }
        if let Some(backed) = &self.backed {
            return match kind {
                0 => backed.span(backed.label_spans_start, backed.label_span_count, id),
                1 => backed.span(backed.target_spans_start, backed.target_span_count, id),
                2 => backed.span(backed.weight_spans_start, backed.weight_span_count, id),
                _ => None,
            };
        }
        match kind {
            0 => self.label_pool.spans.get(id as usize).copied(),
            1 => self.target_pool.spans.get(id as usize).copied(),
            2 => self.weight_id_pool.spans.get(id as usize).copied(),
            _ => None,
        }
    }

    #[inline]
    fn label_value_at(&self, index: usize) -> Option<i32> {
        if let Some(narrow) = &self.owned_narrow {
            return narrow.label(index);
        }
        self.backed.as_ref().map_or_else(
            || self.label_pool.values.get(index).copied(),
            |backed| {
                if backed.narrow {
                    if index >= backed.label_value_count {
                        return None;
                    }
                    let stride = if backed.label_dedup {
                        2
                    } else if backed.split_target_weight {
                        7
                    } else {
                        6
                    };
                    let encoded = backed.read_u16(backed.label_values_start + index * stride)?;
                    Some(if encoded == u16::MAX {
                        i32::MAX - 1
                    } else {
                        encoded as i32
                    })
                } else {
                    backed.pool_i32(backed.label_values_start, backed.label_value_count, index)
                }
            },
        )
    }

    #[inline]
    fn target_value_at(&self, index: usize) -> Option<u32> {
        if let Some(narrow) = &self.owned_narrow {
            return narrow.target(index);
        }
        self.backed.as_ref().map_or_else(
            || self.target_pool.values.get(index).copied(),
            |backed| {
                if backed.narrow {
                    if index >= backed.target_value_count {
                        return None;
                    }
                    if backed.split_target_weight {
                        let stride = if backed.label_dedup { 5 } else { 7 };
                        let offset = if backed.label_dedup { 0 } else { 2 };
                        return backed.read_u24(backed.target_values_start + index * stride + offset);
                    }
                    let pos = if backed.label_dedup {
                        backed.target_values_start + index * 4
                    } else {
                        backed.target_values_start + index * 6 + 2
                    };
                    Some(backed.read_u32(pos)? & 0x1ffff)
                } else {
                    backed.pool_u32(backed.target_values_start, backed.target_value_count, index)
                }
            },
        )
    }

    #[inline]
    fn weight_value_at(&self, index: usize) -> Option<u32> {
        if let Some(narrow) = &self.owned_narrow {
            return narrow.weight_id(index);
        }
        self.backed.as_ref().map_or_else(
            || self.weight_id_pool.values.get(index).copied(),
            |backed| {
                if backed.narrow {
                    if index >= backed.weight_value_count {
                        return None;
                    }
                    if backed.split_target_weight {
                        let stride = if backed.label_dedup { 5 } else { 7 };
                        let offset = if backed.label_dedup { 3 } else { 5 };
                        return backed
                            .read_u16(backed.weight_values_start + index * stride + offset)
                            .map(u32::from);
                    }
                    let pos = if backed.label_dedup {
                        backed.weight_values_start + index * 4
                    } else {
                        backed.weight_values_start + index * 6 + 2
                    };
                    Some(backed.read_u32(pos)? >> 17)
                } else {
                    backed.pool_u32(backed.weight_values_start, backed.weight_value_count, index)
                }
            },
        )
    }

    #[inline]
    fn find_label_in_row(&self, row: PackedRuntimeRow, label: Label) -> Option<usize> {
        let [start, len] = self.pool_span_at(0, row.labels)?;
        let start = start as usize;
        let mut low = 0usize;
        let mut high = len as usize;
        while low < high {
            let mid = (low + high) / 2;
            match self.label_value_at(start + mid)?.cmp(&label) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }


    #[inline]
    fn fast_wire_tail_len(&self) -> usize {
        self.fast_wire_len_for_chunks(&[]).saturating_sub(12)
    }

    fn owned_narrow_fast_wire_len(&self) -> Option<usize> {
        let narrow = self.owned_narrow.as_ref()?;
        let spans = narrow.token_spans.as_ref();
        if spans.len() != self.token_set_count()
            || narrow.token_range_overflow_starts.len() != spans.len()
        {
            return None;
        }
        let geometry_range_count = self.geometries.iter().map(Vec::len).sum::<usize>();
        Some(
            16usize
                .saturating_add(spans.len().saturating_mul(3))
                .saturating_add(spans.len().saturating_mul(3))
                .saturating_add(spans.len().saturating_mul(2))
                .saturating_add(narrow.token_ranges.len().saturating_mul(2))
                .saturating_add(4 + narrow.token_range_overflows.len().saturating_mul(2))
                .saturating_add(4 + self.geometries.len().saturating_mul(2))
                .saturating_add(geometry_range_count.saturating_mul(4))
                .saturating_add(4 + narrow.weights.len())
                .saturating_add(4 + narrow.weight_token_ids.len().saturating_mul(2))
                .saturating_add(4 + narrow.transitions.len())
                .saturating_add(4 + narrow.spans.len().saturating_mul(4))
                .saturating_add(4 + narrow.states.len()),
        )
    }

    fn write_owned_narrow_fast_wire_bytes(&self, dst: &mut [u8]) -> Option<()> {
        if !cfg!(target_endian = "little") {
            return None;
        }
        let narrow = self.owned_narrow.as_ref()?;
        let spans = narrow.token_spans.as_ref();
        let expected_len = self.owned_narrow_fast_wire_len()?;
        if dst.len() != expected_len
            || spans.len() != self.token_set_count()
            || narrow.token_range_overflow_starts.len() != spans.len()
        {
            return None;
        }

        #[inline]
        fn put_u16(dst: &mut [u8], pos: &mut usize, value: u16) {
            dst[*pos..*pos + 2].copy_from_slice(&value.to_le_bytes());
            *pos += 2;
        }
        #[inline]
        fn put_u24(dst: &mut [u8], pos: &mut usize, value: u32) {
            dst[*pos] = value as u8;
            dst[*pos + 1] = (value >> 8) as u8;
            dst[*pos + 2] = (value >> 16) as u8;
            *pos += 3;
        }
        #[inline]
        fn put_u32(dst: &mut [u8], pos: &mut usize, value: u32) {
            dst[*pos..*pos + 4].copy_from_slice(&value.to_le_bytes());
            *pos += 4;
        }
        #[inline]
        fn schedule_pod_copy<T>(
            jobs: &mut Vec<(usize, usize, usize)>,
            dst: &mut [u8],
            pos: &mut usize,
            values: &[T],
            split_chunks: usize,
        ) {
            let len = std::mem::size_of_val(values);
            if len != 0 {
                const SPLIT_MIN_BYTES: usize = 4 * 1024 * 1024;
                let destination = dst[*pos..*pos + len].as_mut_ptr() as usize;
                let source = values.as_ptr().cast::<u8>() as usize;
                if len >= SPLIT_MIN_BYTES && split_chunks > 1 {
                    let chunk_size = len.div_ceil(split_chunks).max(1024 * 1024);
                    let mut offset = 0usize;
                    while offset < len {
                        let count = chunk_size.min(len - offset);
                        jobs.push((destination + offset, source + offset, count));
                        offset += count;
                    }
                } else {
                    jobs.push((destination, source, len));
                }
            }
            *pos += len;
        }

        // The large parts of DWF6 are already the live execution arrays.
        // Reserve their final ranges while writing the small scalar metadata,
        // then copy the disjoint slabs concurrently. This is still fresh-save
        // work: no wire bytes are prepared or retained by compilation.
        let mut copy_jobs = Vec::<(usize, usize, usize)>::with_capacity(7);
        // Two-way slab copies were the best crossover in the canonical large
        // parser save benchmarks; more jobs increased scheduling/memory-bandwidth
        // overhead without improving wall time.
        let split_chunks = 2;
        let mut pos = 0usize;
        dst[pos..pos + 4].copy_from_slice(b"DWF8");
        pos += 4;
        put_u32(dst, &mut pos, self.start_state);
        put_u32(dst, &mut pos, spans.len() as u32);
        put_u32(
            dst,
            &mut pos,
            u32::try_from(narrow.token_ranges.len().saturating_mul(2)).ok()?,
        );
        for span in spans {
            put_u24(dst, &mut pos, span.start);
        }
        for &overflow_start in narrow.token_range_overflow_starts.iter() {
            put_u24(dst, &mut pos, overflow_start);
        }
        for span in spans {
            put_u16(dst, &mut pos, span.word_spans as u16);
        }
        schedule_pod_copy(
            &mut copy_jobs,
            dst,
            &mut pos,
            &narrow.token_ranges,
            split_chunks,
        );

        put_u32(
            dst,
            &mut pos,
            u32::try_from(narrow.token_range_overflows.len()).ok()?,
        );
        schedule_pod_copy(
            &mut copy_jobs,
            dst,
            &mut pos,
            &narrow.token_range_overflows,
            split_chunks,
        );

        put_u32(dst, &mut pos, self.geometries.len() as u32);
        for geometry in self.geometries.iter() {
            put_u16(dst, &mut pos, geometry.len() as u16);
            for &(start, end) in geometry {
                put_u16(dst, &mut pos, start as u16);
                put_u16(dst, &mut pos, end as u16);
            }
        }

        put_u32(
            dst,
            &mut pos,
            (narrow.weights.len() / OwnedNarrowPackedRuntimeDwa::WEIGHT_STRIDE) as u32,
        );
        schedule_pod_copy(&mut copy_jobs, dst, &mut pos, &narrow.weights, split_chunks);

        put_u32(dst, &mut pos, narrow.weight_token_ids.len() as u32);
        schedule_pod_copy(
            &mut copy_jobs,
            dst,
            &mut pos,
            &narrow.weight_token_ids,
            split_chunks,
        );

        put_u32(
            dst,
            &mut pos,
            (narrow.transitions.len() / OwnedNarrowPackedRuntimeDwa::TRANSITION_STRIDE) as u32,
        );
        schedule_pod_copy(
            &mut copy_jobs,
            dst,
            &mut pos,
            &narrow.transitions,
            split_chunks,
        );

        put_u32(dst, &mut pos, narrow.spans.len() as u32);
        schedule_pod_copy(&mut copy_jobs, dst, &mut pos, &narrow.spans, split_chunks);

        put_u32(
            dst,
            &mut pos,
            (narrow.states.len() / OwnedNarrowPackedRuntimeDwa::STATE_STRIDE) as u32,
        );
        schedule_pod_copy(&mut copy_jobs, dst, &mut pos, &narrow.states, split_chunks);
        debug_assert_eq!(pos, dst.len());

        let copy_one = |(destination, source, len): (usize, usize, usize)| unsafe {
            std::ptr::copy_nonoverlapping(source as *const u8, destination as *mut u8, len);
        };
        if rayon::current_num_threads() > 1 && copy_jobs.len() > 1 {
            copy_jobs.into_par_iter().for_each(copy_one);
        } else {
            copy_jobs.into_iter().for_each(copy_one);
        }
        Some(())
    }

    /// Exact directly-writable wire size for compiler-owned runtime storage.
    /// DWF6 is preferred when the hot execution view is narrow; DWF7 remains
    /// the wide runtime-shaped fallback.
    pub fn direct_fast_wire_len(&self) -> Option<usize> {
        self.owned_narrow_fast_wire_len()
            .or_else(|| self.runtime_shaped_fast_wire_len())
    }

    pub fn write_direct_fast_wire_bytes(&self, dst: &mut [u8]) -> Option<()> {
        if self.owned_narrow.is_some() {
            return self.write_owned_narrow_fast_wire_bytes(dst);
        }
        self.write_runtime_shaped_fast_wire_bytes(dst)
    }

    /// DWF3 keeps the compact per-token-set varint bodies from DWF2, but adds
    /// a direct token-id -> (chunk, byte-offset) index and word-span table.
    /// The fixed-width DWA tail remains byte-for-byte runtime-readable, so a
    /// current-format loader can retain the artifact backing instead of
    /// allocating/decoding millions of token ranges and pool entries.
    fn try_backed_fast_wire_bytes(&self) -> Option<Vec<u8>> {
        if let Some(len) = self.owned_narrow_fast_wire_len() {
            let mut out = vec![0u8; len];
            self.write_owned_narrow_fast_wire_bytes(&mut out)?;
            return Some(out);
        }
        if self.prefer_runtime_shaped_wire() {
            let mut out = Vec::with_capacity(self.runtime_shaped_fast_wire_len()?);
            self.try_append_runtime_shaped_fast_wire_bytes(&mut out)?;
            return Some(out);
        }
        self.try_narrow_backed_fast_wire_bytes()
            .or_else(|| self.try_backed_fast_wire_bytes_dwf3())
    }

    #[inline]
    fn prefer_runtime_shaped_wire(&self) -> bool {
        self.backed.is_none()
            && self.flat_token_ranges_u16.is_some()
            && self.flat_token_spans.is_some()
            && (self.states.len() >= 20_000
                || self
                    .flat_token_ranges_u16
                    .as_ref()
                    .is_some_and(|ranges| ranges.len() >= 262_144))
    }

    pub fn runtime_shaped_fast_wire_len(&self) -> Option<usize> {
        if !self.prefer_runtime_shaped_wire() {
            return None;
        }
        let ranges = self.flat_token_ranges_u16.as_deref()?;
        let spans = self.flat_token_spans.as_deref()?;
        if spans.len() != self.token_set_count() {
            return None;
        }
        let geometry_range_count = self.geometries.iter().map(Vec::len).sum::<usize>();
        Some(
            16usize
                .saturating_add(spans.len().saturating_mul(12))
                .saturating_add(ranges.len().saturating_mul(4))
                .saturating_add(4 + self.geometries.len().saturating_mul(4))
                .saturating_add(geometry_range_count.saturating_mul(8))
                .saturating_add(4 + self.weights.len().saturating_mul(12))
                .saturating_add(4 + self.weight_token_ids.len().saturating_mul(4))
                .saturating_add(4 + self.label_pool.values.len().saturating_mul(4))
                .saturating_add(4 + self.label_pool.spans.len().saturating_mul(8))
                .saturating_add(4 + self.target_pool.values.len().saturating_mul(4))
                .saturating_add(4 + self.target_pool.spans.len().saturating_mul(8))
                .saturating_add(4 + self.weight_id_pool.values.len().saturating_mul(4))
                .saturating_add(4 + self.weight_id_pool.spans.len().saturating_mul(8))
                .saturating_add(4 + self.rows.len().saturating_mul(12))
                .saturating_add(4 + self.states.len().saturating_mul(8)),
        )
    }

    /// Write the DWF7 runtime-shaped representation directly into an exact
    /// caller-provided destination. This avoids both the temporary DWA buffer
    /// and the zero-fill performed by `Vec::resize` before overwriting the
    /// section. The destination must have exactly
    /// [`Self::runtime_shaped_fast_wire_len`] bytes.
    pub fn write_runtime_shaped_fast_wire_bytes(&self, dst: &mut [u8]) -> Option<()> {
        if !cfg!(target_endian = "little") {
            return None;
        }
        let expected_len = self.runtime_shaped_fast_wire_len()?;
        if dst.len() != expected_len {
            return None;
        }
        let ranges = self.flat_token_ranges_u16.as_deref()?;
        let spans = self.flat_token_spans.as_deref()?;
        if spans.len() != self.token_set_count() {
            return None;
        }

        #[inline]
        fn put_u32_at(dst: &mut [u8], pos: &mut usize, value: u32) {
            dst[*pos..*pos + 4].copy_from_slice(&value.to_le_bytes());
            *pos += 4;
        }

        #[inline]
        fn put_pod_slice<T>(dst: &mut [u8], pos: &mut usize, values: &[T]) {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            };
            dst[*pos..*pos + bytes.len()].copy_from_slice(bytes);
            *pos += bytes.len();
        }

        debug_assert_eq!(std::mem::size_of::<PackedRuntimeTokenSetSpan>(), 12);
        debug_assert_eq!(std::mem::size_of::<PackedRuntimeWeight>(), 12);
        debug_assert_eq!(std::mem::size_of::<PackedRuntimeRow>(), 12);
        debug_assert_eq!(std::mem::size_of::<PackedRuntimeState>(), 8);

        dst[..4].copy_from_slice(b"DWF7");
        dst[4..8].copy_from_slice(&self.start_state.to_le_bytes());
        dst[8..12].copy_from_slice(&(spans.len() as u32).to_le_bytes());
        dst[12..16].copy_from_slice(&(ranges.len() as u32).to_le_bytes());

        // The runtime-shaped wire is almost entirely POD slabs. Queue those
        // slabs as disjoint copy jobs instead of grouping them into only three
        // coarse branches; on large parser DWAs the ~11 MiB token slab alone
        // otherwise becomes the critical path. Geometry metadata is small and
        // irregular, so keep that part scalar while the fixed-width pools are
        // copied in parallel below.
        let mut pos = 16usize;
        let mut copy_jobs = Vec::<(usize, usize, usize)>::with_capacity(64);

        #[inline]
        fn queue_pod_copy<T>(
            dst: &mut [u8],
            pos: &mut usize,
            values: &[T],
            jobs: &mut Vec<(usize, usize, usize)>,
        ) {
            let byte_len = std::mem::size_of_val(values);
            if byte_len == 0 {
                return;
            }
            let dst_base = unsafe { dst.as_mut_ptr().add(*pos) } as usize;
            let src_base = values.as_ptr().cast::<u8>() as usize;
            const COPY_CHUNK: usize = 1024 * 1024;
            let mut offset = 0usize;
            while offset < byte_len {
                let len = COPY_CHUNK.min(byte_len - offset);
                jobs.push((dst_base + offset, src_base + offset, len));
                offset += len;
            }
            *pos += byte_len;
        }

        queue_pod_copy(dst, &mut pos, spans, &mut copy_jobs);
        queue_pod_copy(dst, &mut pos, ranges, &mut copy_jobs);

        put_u32_at(dst, &mut pos, self.geometries.len() as u32);
        for geometry in self.geometries.iter() {
            put_u32_at(dst, &mut pos, geometry.len() as u32);
            for &(start, end) in geometry {
                put_u32_at(dst, &mut pos, start);
                put_u32_at(dst, &mut pos, end);
            }
        }

        put_u32_at(dst, &mut pos, self.weights.len() as u32);
        queue_pod_copy(dst, &mut pos, &self.weights, &mut copy_jobs);
        put_u32_at(dst, &mut pos, self.weight_token_ids.len() as u32);
        queue_pod_copy(dst, &mut pos, &self.weight_token_ids, &mut copy_jobs);
        put_u32_at(dst, &mut pos, self.label_pool.values.len() as u32);
        queue_pod_copy(dst, &mut pos, &self.label_pool.values, &mut copy_jobs);
        put_u32_at(dst, &mut pos, self.label_pool.spans.len() as u32);
        queue_pod_copy(dst, &mut pos, &self.label_pool.spans, &mut copy_jobs);
        put_u32_at(dst, &mut pos, self.target_pool.values.len() as u32);
        queue_pod_copy(dst, &mut pos, &self.target_pool.values, &mut copy_jobs);
        put_u32_at(dst, &mut pos, self.target_pool.spans.len() as u32);
        queue_pod_copy(dst, &mut pos, &self.target_pool.spans, &mut copy_jobs);
        put_u32_at(dst, &mut pos, self.weight_id_pool.values.len() as u32);
        queue_pod_copy(dst, &mut pos, &self.weight_id_pool.values, &mut copy_jobs);
        put_u32_at(dst, &mut pos, self.weight_id_pool.spans.len() as u32);
        queue_pod_copy(dst, &mut pos, &self.weight_id_pool.spans, &mut copy_jobs);
        put_u32_at(dst, &mut pos, self.rows.len() as u32);
        queue_pod_copy(dst, &mut pos, &self.rows, &mut copy_jobs);
        put_u32_at(dst, &mut pos, self.states.len() as u32);
        queue_pod_copy(dst, &mut pos, &self.states, &mut copy_jobs);

        debug_assert_eq!(pos, expected_len);
        if rayon::current_num_threads() > 1 && copy_jobs.len() > 1 {
            copy_jobs.into_par_iter().for_each(|(destination, source, len)| {
                // SAFETY: every job targets a disjoint byte range in `dst`,
                // and all source runtime slabs outlive this parallel copy.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        source as *const u8,
                        destination as *mut u8,
                        len,
                    );
                }
            });
        } else {
            for (destination, source, len) in copy_jobs {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        source as *const u8,
                        destination as *mut u8,
                        len,
                    );
                }
            }
        }
        Some(())
    }

    fn try_append_runtime_shaped_fast_wire_bytes(&self, out: &mut Vec<u8>) -> Option<usize> {
        if !self.prefer_runtime_shaped_wire() {
            return None;
        }
        let ranges = self.flat_token_ranges_u16.as_deref()?;
        let spans = self.flat_token_spans.as_deref()?;
        if spans.len() != self.token_set_count() {
            return None;
        }

        #[inline]
        fn put_u32(out: &mut Vec<u8>, value: u32) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        #[inline]
        fn put_u32s(out: &mut Vec<u8>, values: &[u32]) {
            if cfg!(target_endian = "little") {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        values.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(values),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for &value in values {
                    put_u32(out, value);
                }
            }
        }
        #[inline]
        fn put_i32s(out: &mut Vec<u8>, values: &[i32]) {
            if cfg!(target_endian = "little") {
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
        fn put_spans(out: &mut Vec<u8>, spans: &[[u32; 2]]) {
            if cfg!(target_endian = "little") {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        spans.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(spans),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for &[start, len] in spans {
                    put_u32(out, start);
                    put_u32(out, len);
                }
            }
        }
        #[inline]
        fn put_token_spans(out: &mut Vec<u8>, spans: &[PackedRuntimeTokenSetSpan]) {
            if cfg!(target_endian = "little") {
                debug_assert_eq!(std::mem::size_of::<PackedRuntimeTokenSetSpan>(), 12);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        spans.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(spans),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for span in spans {
                    put_u32(out, span.start);
                    put_u32(out, span.len);
                    put_u32(out, span.word_spans);
                }
            }
        }
        #[inline]
        fn put_ranges16(out: &mut Vec<u8>, ranges: &[[u16; 2]]) {
            if cfg!(target_endian = "little") {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        ranges.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(ranges),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for &[start, end] in ranges {
                    out.extend_from_slice(&start.to_le_bytes());
                    out.extend_from_slice(&end.to_le_bytes());
                }
            }
        }
        #[inline]
        fn put_weights(out: &mut Vec<u8>, weights: &[PackedRuntimeWeight]) {
            if cfg!(target_endian = "little") {
                debug_assert_eq!(std::mem::size_of::<PackedRuntimeWeight>(), 12);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        weights.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(weights),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for weight in weights {
                    put_u32(out, weight.geometry);
                    put_u32(out, weight.token_ids_start);
                    put_u32(out, weight.full);
                }
            }
        }
        #[inline]
        fn put_rows(out: &mut Vec<u8>, rows: &[PackedRuntimeRow]) {
            if cfg!(target_endian = "little") {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        rows.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(rows),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for row in rows {
                    put_u32(out, row.labels);
                    put_u32(out, row.targets);
                    put_u32(out, row.weights);
                }
            }
        }
        #[inline]
        fn put_states(out: &mut Vec<u8>, states: &[PackedRuntimeState]) {
            if cfg!(target_endian = "little") {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        states.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(states),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for state in states {
                    put_u32(out, state.row);
                    put_u32(out, state.final_weight);
                }
            }
        }

        let expected_len = self.runtime_shaped_fast_wire_len()?;

        // DWF7 mirrors the owned runtime widths, so on little-endian hosts the
        // large sections are already byte-for-byte wire data. Size the final
        // region once and copy the independent runtime slabs concurrently
        // instead of serially extending the Vec through ~20+ MiB of pools.
        if cfg!(target_endian = "little") {
            let output_start = out.len();
            out.resize(output_start + expected_len, 0);
            self.write_runtime_shaped_fast_wire_bytes(
                &mut out[output_start..output_start + expected_len],
            )?;
            return Some(expected_len);
        }

        let output_start = out.len();
        out.reserve(expected_len);
        out.extend_from_slice(b"DWF7");
        put_u32(out, self.start_state);
        put_u32(out, spans.len() as u32);
        put_u32(out, ranges.len() as u32);
        put_token_spans(out, spans);
        put_ranges16(out, ranges);

        put_u32(out, self.geometries.len() as u32);
        for geometry in self.geometries.iter() {
            put_u32(out, geometry.len() as u32);
            for &(start, end) in geometry {
                put_u32(out, start);
                put_u32(out, end);
            }
        }
        put_u32(out, self.weights.len() as u32);
        put_weights(out, &self.weights);
        put_u32(out, self.weight_token_ids.len() as u32);
        put_u32s(out, &self.weight_token_ids);
        put_u32(out, self.label_pool.values.len() as u32);
        put_i32s(out, &self.label_pool.values);
        put_u32(out, self.label_pool.spans.len() as u32);
        put_spans(out, &self.label_pool.spans);
        put_u32(out, self.target_pool.values.len() as u32);
        put_u32s(out, &self.target_pool.values);
        put_u32(out, self.target_pool.spans.len() as u32);
        put_spans(out, &self.target_pool.spans);
        put_u32(out, self.weight_id_pool.values.len() as u32);
        put_u32s(out, &self.weight_id_pool.values);
        put_u32(out, self.weight_id_pool.spans.len() as u32);
        put_spans(out, &self.weight_id_pool.spans);
        put_u32(out, self.rows.len() as u32);
        put_rows(out, &self.rows);
        put_u32(out, self.states.len() as u32);
        put_states(out, &self.states);

        let written = out.len() - output_start;
        debug_assert_eq!(written, expected_len);
        Some(written)
    }

    /// DWF5/6 narrow the directly-read fixed tail while preserving DWF3's
    /// zero-copy execution model. DWF5 packs a 17-bit target and 15-bit weight
    /// into one u32. DWF6 keeps the same compact pools/state records but stores
    /// each transition as u24 target + u16 weight, covering larger parser DWAs
    /// without falling all the way back to DWF3.
    fn try_narrow_backed_fast_wire_bytes(&self) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        self.try_append_narrow_backed_fast_wire_bytes(&mut out)?;
        Some(out)
    }

    fn try_append_narrow_backed_fast_wire_bytes(&self, mut out: &mut Vec<u8>) -> Option<usize> {
        let split_target_weight = self.states.len() > (1usize << 17)
            || self.target_pool.values.iter().any(|&value| value >= (1 << 17))
            || self.weight_id_pool.values.iter().any(|&value| value >= (1 << 15));
        let raw_u16_ranges = split_target_weight
            .then(|| self.flat_token_ranges_u16.as_deref())
            .flatten();
        let raw_u16_spans = split_target_weight
            .then(|| self.flat_token_spans.as_deref())
            .flatten();
        // DWF6's token body is the fixed-width u16 runtime slab. A large DWA
        // that does not use that runtime representation must fall back to
        // DWF3 rather than writing a varint body under DWF6 magic.
        if split_target_weight && (raw_u16_ranges.is_none() || raw_u16_spans.is_none()) {
            return None;
        }
        // Loaded artifacts may already carry their varint token chunks. Fresh
        // compiler-produced PackedRuntimeDwa values deliberately do not: those
        // chunks are a wire representation, not runtime state. Build them here
        // inside save() when needed so DWF5 remains available without moving
        // serialization work into compiler finalization.
        const TOKEN_SET_WIRE_CHUNK: usize = 1024;
        let owned_chunks;
        let chunks: &[Box<[u8]>] = if raw_u16_ranges.is_some() {
            &[]
        } else if let Some(chunks) = self.fast_wire_token_chunks.as_deref() {
            chunks
        } else {
            let token_set_count = self.token_set_count();
            let chunk_ranges = (0..token_set_count)
                .step_by(TOKEN_SET_WIRE_CHUNK)
                .map(|start| (start, (start + TOKEN_SET_WIRE_CHUNK).min(token_set_count)))
                .collect::<Vec<_>>();
            let encode_chunk = |&(start_id, end_id): &(usize, usize)| {
                let mut body = Vec::<u8>::new();
                put_var_u32(&mut body, (end_id - start_id) as u32);
                for id in start_id..end_id {
                    let token_set = self.token_set(id as u32)?;
                    put_var_u32(&mut body, token_set.range_count() as u32);
                    let mut previous_end_plus_one = 0u64;
                    token_set.for_each_range(|lo, hi| {
                        put_var_u64(&mut body, lo as u64 - previous_end_plus_one);
                        put_var_u32(&mut body, hi - lo);
                        previous_end_plus_one = hi as u64 + 1;
                    });
                }
                Some(body.into_boxed_slice())
            };
            owned_chunks = if chunk_ranges.len() >= 4 && rayon::current_num_threads() > 1 {
                chunk_ranges
                    .par_iter()
                    .map(encode_chunk)
                    .collect::<Option<Vec<_>>>()?
            } else {
                chunk_ranges
                    .iter()
                    .map(encode_chunk)
                    .collect::<Option<Vec<_>>>()?
            };
            &owned_chunks
        };
        let owned_word_spans;
        let word_spans: &[u32] = if let Some(spans) = raw_u16_spans {
            owned_word_spans = spans.iter().map(|span| span.word_spans).collect::<Vec<_>>();
            &owned_word_spans
        } else if let Some(spans) = self.materialized_token_word_spans.as_deref() {
            spans
        } else {
            owned_word_spans = (0..self.token_set_count())
                .map(|id| self.token_set(id as u32).map(|set| set.word_spans()))
                .collect::<Option<Vec<_>>>()?;
            &owned_word_spans
        };
        let token_set_count = self.token_set_count();
        if word_spans.len() != token_set_count
            || token_set_count > u16::MAX as usize + 1
            || word_spans.iter().any(|&value| value > u16::MAX as u32)
        {
            return None;
        }

        let token_bytes = raw_u16_ranges
            .map(|ranges| ranges.len().saturating_mul(4))
            .unwrap_or_else(|| chunks.iter().map(|chunk| chunk.len()).sum::<usize>());
        if token_bytes > (1usize << 24) {
            return None;
        }

        // DWF4 stores a single 24-bit offset into the concatenated token body.
        // The chunking remains an encoder implementation detail only.
        let mut locations = Vec::<u32>::with_capacity(token_set_count);
        if let Some(spans) = raw_u16_spans {
            if spans.len() != token_set_count || raw_u16_ranges.is_none() {
                return None;
            }
            for span in spans {
                let location = (span.start as usize).checked_mul(4)?;
                if location >= (1usize << 24) {
                    return None;
                }
                locations.push(location as u32);
            }
        } else {
            let mut body_base = 0usize;
            for body in chunks {
                let mut pos = 0usize;
                let set_count = take_var_u32(body, &mut pos).ok()? as usize;
                for _ in 0..set_count {
                    let location = body_base.checked_add(pos)?;
                    if location >= (1usize << 24) {
                        return None;
                    }
                    locations.push(location as u32);
                    let range_count = take_var_u32(body, &mut pos).ok()? as usize;
                    for _ in 0..range_count {
                        let _ = take_var_u64(body, &mut pos).ok()?;
                        let _ = take_var_u32(body, &mut pos).ok()?;
                    }
                }
                if pos != body.len() {
                    return None;
                }
                body_base = body_base.checked_add(body.len())?;
            }
            if body_base != token_bytes {
                return None;
            }
        }
        if locations.len() != token_set_count {
            return None;
        }

        const DEFAULT_LABEL_WIRE: i32 = i32::MAX - 1;
        const SPAN_LEN_BITS: u32 = 10;
        const SPAN_LEN_MAX: u32 = (1 << SPAN_LEN_BITS) - 1;
        const SPAN_START_MAX: u32 = (1 << (32 - SPAN_LEN_BITS)) - 1;
        let span_fits = |span: &[u32; 2]| span[0] <= SPAN_START_MAX && span[1] <= SPAN_LEN_MAX;

        if self.geometries.len() > u16::MAX as usize + 1
            || self.geometries.iter().any(|geometry| {
                geometry.len() > u16::MAX as usize
                    || geometry
                        .iter()
                        .any(|&(start, end)| start > u16::MAX as u32 || end > u16::MAX as u32)
            })
            || self.weights.len() > u16::MAX as usize
            || self.weights.iter().any(|weight| {
                weight.geometry > u16::MAX as u32
                    || weight.token_ids_start >= (1 << 24)
                    || weight.full > u8::MAX as u32
            })
            || self.weight_token_ids.iter().any(|&value| value > u16::MAX as u32)
            || self.label_pool.values.iter().any(|&label| {
                label != DEFAULT_LABEL_WIRE && !(0..u16::MAX as i32).contains(&label)
            })
            || self.label_pool.values.len() != self.target_pool.values.len()
            || self.label_pool.values.len() != self.weight_id_pool.values.len()
            || self.label_pool.spans.len() > u16::MAX as usize + 1
            || self.label_pool.spans.iter().any(|span| !span_fits(span))
            || self.target_pool.values.iter().any(|&value| {
                value
                    >= if split_target_weight {
                        1u32 << 24
                    } else {
                        1u32 << 17
                    }
            })
            || self.weight_id_pool.values.iter().any(|&value| {
                value
                    >= if split_target_weight {
                        1u32 << 16
                    } else {
                        1u32 << 15
                    }
            })
            || self.label_pool.spans.len() != self.target_pool.spans.len()
            || self.label_pool.spans.len() != self.weight_id_pool.spans.len()
            || self
                .label_pool
                .spans
                .iter()
                .zip(self.target_pool.spans.iter())
                .zip(self.weight_id_pool.spans.iter())
                .any(|((label, target), weight)| label != target || label != weight)
            || self.rows.len() != self.label_pool.spans.len()
            || self.rows.iter().enumerate().any(|(index, row)| {
                row.labels as usize != index
                    || row.targets as usize != index
                    || row.weights as usize != index
            })
            || self.states.len() > (1usize << 24)
            || self.states.iter().any(|state| {
                state.row > u16::MAX as u32
                    || (state.final_weight != u32::MAX && state.final_weight >= u16::MAX as u32)
            })
        {
            return None;
        }

        // DWF5 still interns repeated label vectors for compactness. DWF6 is
        // intentionally row-local instead: avoiding this whole-DWA hash pass
        // makes a genuinely fresh save cheap, and its u16/u24/u16 transition
        // record remains directly executable after load.
        let (unique_label_values, unique_label_spans, row_label_ids) = if split_target_weight {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            let mut label_ids = FxHashMap::<&[Label], u16>::default();
            label_ids.reserve(self.label_pool.spans.len().min(u16::MAX as usize));
            let mut unique_label_values = Vec::<Label>::new();
            let mut unique_label_spans = Vec::<[u32; 2]>::new();
            let mut row_label_ids = Vec::<u16>::with_capacity(self.label_pool.spans.len());
            for &[start, len] in self.label_pool.spans.iter() {
                let start_usize = start as usize;
                let Some(end) = start_usize.checked_add(len as usize) else {
                    return None;
                };
                let labels = self.label_pool.values.get(start_usize..end)?;
                let id = if let Some(&id) = label_ids.get(labels) {
                    id
                } else {
                    let id = u16::try_from(unique_label_spans.len()).ok()?;
                    let unique_start = u32::try_from(unique_label_values.len()).ok()?;
                    if unique_start > SPAN_START_MAX || len > SPAN_LEN_MAX {
                        return None;
                    }
                    unique_label_values.extend_from_slice(labels);
                    unique_label_spans.push([unique_start, len]);
                    label_ids.insert(labels, id);
                    id
                };
                row_label_ids.push(id);
            }
            (unique_label_values, unique_label_spans, row_label_ids)
        };

        #[inline]
        fn put_u16(out: &mut Vec<u8>, value: u16) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        #[inline]
        fn put_u24(out: &mut Vec<u8>, value: u32) {
            debug_assert!(value < (1 << 24));
            out.push(value as u8);
            out.push((value >> 8) as u8);
            out.push((value >> 16) as u8);
        }
        #[inline]
        fn put_u32(out: &mut Vec<u8>, value: u32) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        #[inline]
        fn put_span(out: &mut Vec<u8>, span: [u32; 2]) {
            debug_assert!(span[0] <= SPAN_START_MAX && span[1] <= SPAN_LEN_MAX);
            put_u32(out, (span[0] << SPAN_LEN_BITS) | span[1]);
        }

        let geometry_range_count = self.geometries.iter().map(Vec::len).sum::<usize>();
        let transition_section_len = if split_target_weight {
            4usize
                .saturating_add(self.label_pool.values.len().saturating_mul(7))
                .saturating_add(4 + self.label_pool.spans.len().saturating_mul(4))
        } else {
            4usize
                .saturating_add(unique_label_values.len().saturating_mul(2))
                .saturating_add(4 + unique_label_spans.len().saturating_mul(4))
                .saturating_add(4 + row_label_ids.len().saturating_mul(2))
                .saturating_add(4 + self.target_pool.values.len().saturating_mul(4))
                .saturating_add(4 + self.target_pool.spans.len().saturating_mul(4))
        };
        let expected_len = 16usize
            .saturating_add(token_set_count.saturating_mul(3))
            .saturating_add(token_set_count.saturating_mul(2))
            .saturating_add(token_bytes)
            .saturating_add(4)
            .saturating_add(self.geometries.len().saturating_mul(2))
            .saturating_add(geometry_range_count.saturating_mul(4))
            .saturating_add(4 + self.weights.len().saturating_mul(6))
            .saturating_add(4 + self.weight_token_ids.len().saturating_mul(2))
            .saturating_add(transition_section_len)
            .saturating_add(4 + self.states.len().saturating_mul(4));
        let output_start = out.len();
        out.reserve(expected_len);
        out.extend_from_slice(if split_target_weight { b"DWF6" } else { b"DWF5" });
        put_u32(&mut out, self.start_state);
        put_u32(&mut out, token_set_count as u32);
        put_u32(&mut out, token_bytes as u32);
        for location in locations {
            put_u24(&mut out, location);
        }
        for &word_span in word_spans {
            put_u16(&mut out, word_span as u16);
        }
        if let Some(ranges) = raw_u16_ranges {
            if cfg!(target_endian = "little") {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        ranges.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(ranges),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for &[start, end] in ranges {
                    put_u16(&mut out, start);
                    put_u16(&mut out, end);
                }
            }
        } else {
            for body in chunks {
                out.extend_from_slice(body);
            }
        }

        put_u32(&mut out, self.geometries.len() as u32);
        for geometry in self.geometries.iter() {
            put_u16(&mut out, geometry.len() as u16);
            for &(start, end) in geometry {
                put_u16(&mut out, start as u16);
                put_u16(&mut out, end as u16);
            }
        }
        put_u32(&mut out, self.weights.len() as u32);
        for weight in self.weights.iter() {
            put_u16(&mut out, weight.geometry as u16);
            put_u24(&mut out, weight.token_ids_start);
            out.push(weight.full as u8);
        }
        put_u32(&mut out, self.weight_token_ids.len() as u32);
        let weight_token_ids_start = out.len();
        out.resize(
            weight_token_ids_start + self.weight_token_ids.len().saturating_mul(2),
            0,
        );
        if self.weight_token_ids.len() >= 4096 && rayon::current_num_threads() > 1 {
            out[weight_token_ids_start..]
                .par_chunks_exact_mut(2)
                .zip(self.weight_token_ids.par_iter())
                .for_each(|(dst, &value)| dst.copy_from_slice(&(value as u16).to_le_bytes()));
        } else {
            for (dst, &value) in out[weight_token_ids_start..]
                .chunks_exact_mut(2)
                .zip(self.weight_token_ids.iter())
            {
                dst.copy_from_slice(&(value as u16).to_le_bytes());
            }
        }
        if split_target_weight {
            put_u32(&mut out, self.label_pool.values.len() as u32);
            let transitions_start = out.len();
            out.resize(
                transitions_start + self.label_pool.values.len().saturating_mul(7),
                0,
            );
            let write_transition =
                |dst: &mut [u8], ((&label, &target), &weight_id): ((&Label, &u32), &u32)| {
                    let label = if label == DEFAULT_LABEL_WIRE {
                        u16::MAX
                    } else {
                        label as u16
                    };
                    dst[..2].copy_from_slice(&label.to_le_bytes());
                    dst[2] = target as u8;
                    dst[3] = (target >> 8) as u8;
                    dst[4] = (target >> 16) as u8;
                    dst[5..7].copy_from_slice(&(weight_id as u16).to_le_bytes());
                };
            if self.label_pool.values.len() >= 4096 && rayon::current_num_threads() > 1 {
                out[transitions_start..]
                    .par_chunks_exact_mut(7)
                    .zip(
                        self.label_pool
                            .values
                            .par_iter()
                            .zip(self.target_pool.values.par_iter())
                            .zip(self.weight_id_pool.values.par_iter()),
                    )
                    .for_each(|(dst, entry)| write_transition(dst, entry));
            } else {
                for (dst, entry) in out[transitions_start..]
                    .chunks_exact_mut(7)
                    .zip(
                        self.label_pool
                            .values
                            .iter()
                            .zip(self.target_pool.values.iter())
                            .zip(self.weight_id_pool.values.iter()),
                    )
                {
                    write_transition(dst, entry);
                }
            }
            put_u32(&mut out, self.label_pool.spans.len() as u32);
            for &span in self.label_pool.spans.iter() {
                put_span(&mut out, span);
            }
        } else {
            put_u32(&mut out, unique_label_values.len() as u32);
            for &label in &unique_label_values {
                put_u16(
                    &mut out,
                    if label == DEFAULT_LABEL_WIRE {
                        u16::MAX
                    } else {
                        label as u16
                    },
                );
            }
            put_u32(&mut out, unique_label_spans.len() as u32);
            for &span in &unique_label_spans {
                put_span(&mut out, span);
            }
            put_u32(&mut out, row_label_ids.len() as u32);
            for &label_id in &row_label_ids {
                put_u16(&mut out, label_id);
            }
            put_u32(&mut out, self.target_pool.values.len() as u32);
            let transitions_start = out.len();
            out.resize(
                transitions_start + self.target_pool.values.len().saturating_mul(4),
                0,
            );
            let write_transition = |dst: &mut [u8], (&target, &weight_id): (&u32, &u32)| {
                dst.copy_from_slice(&(target | (weight_id << 17)).to_le_bytes());
            };
            if self.target_pool.values.len() >= 4096 && rayon::current_num_threads() > 1 {
                out[transitions_start..]
                    .par_chunks_exact_mut(4)
                    .zip(
                        self.target_pool
                            .values
                            .par_iter()
                            .zip(self.weight_id_pool.values.par_iter()),
                    )
                    .for_each(|(dst, pair)| write_transition(dst, pair));
            } else {
                for (dst, pair) in out[transitions_start..]
                    .chunks_exact_mut(4)
                    .zip(
                        self.target_pool
                            .values
                            .iter()
                            .zip(self.weight_id_pool.values.iter()),
                    )
                {
                    write_transition(dst, pair);
                }
            }
            put_u32(&mut out, self.target_pool.spans.len() as u32);
            for &span in self.target_pool.spans.iter() {
                put_span(&mut out, span);
            }
        }
        put_u32(&mut out, self.states.len() as u32);
        let states_start = out.len();
        out.resize(states_start + self.states.len().saturating_mul(4), 0);
        let write_state = |dst: &mut [u8], state: &PackedRuntimeState| {
            dst[..2].copy_from_slice(&(state.row as u16).to_le_bytes());
            let final_weight = if state.final_weight == u32::MAX {
                0
            } else {
                state.final_weight as u16 + 1
            };
            dst[2..4].copy_from_slice(&final_weight.to_le_bytes());
        };
        if self.states.len() >= 4096 && rayon::current_num_threads() > 1 {
            out[states_start..]
                .par_chunks_exact_mut(4)
                .zip(self.states.par_iter())
                .for_each(|(dst, state)| write_state(dst, state));
        } else {
            for (dst, state) in out[states_start..]
                .chunks_exact_mut(4)
                .zip(self.states.iter())
            {
                write_state(dst, state);
            }
        }
        let written = out.len() - output_start;
        debug_assert_eq!(written, expected_len);
        Some(written)
    }

    fn try_backed_fast_wire_bytes_dwf3(&self) -> Option<Vec<u8>> {
        let chunks = self.fast_wire_token_chunks.as_deref()?;
        let word_spans = self.materialized_token_word_spans.as_deref()?;
        if chunks.len() > u8::MAX as usize || word_spans.len() != self.token_set_count() {
            return None;
        }

        #[inline]
        fn put_u32(out: &mut Vec<u8>, value: u32) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        #[inline]
        fn put_u32s(out: &mut Vec<u8>, values: &[u32]) {
            if cfg!(target_endian = "little") {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        values.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(values),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for &value in values {
                    put_u32(out, value);
                }
            }
        }
        #[inline]
        fn put_i32s(out: &mut Vec<u8>, values: &[i32]) {
            if cfg!(target_endian = "little") {
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

        let token_set_count = word_spans.len();
        let mut locations = Vec::<u32>::with_capacity(token_set_count);
        let mut chunk_set_counts = Vec::<u32>::with_capacity(chunks.len());
        for (chunk_id, body) in chunks.iter().enumerate() {
            let mut pos = 0usize;
            let set_count = take_var_u32(body, &mut pos).ok()? as usize;
            chunk_set_counts.push(u32::try_from(set_count).ok()?);
            for _ in 0..set_count {
                if pos >= (1usize << 24) {
                    return None;
                }
                locations.push(((chunk_id as u32) << 24) | pos as u32);
                let range_count = take_var_u32(body, &mut pos).ok()? as usize;
                for _ in 0..range_count {
                    let _ = take_var_u64(body, &mut pos).ok()?;
                    let _ = take_var_u32(body, &mut pos).ok()?;
                }
            }
            if pos != body.len() {
                return None;
            }
        }
        if locations.len() != token_set_count {
            return None;
        }

        let token_bytes = chunks.iter().map(|chunk| chunk.len()).sum::<usize>();
        let mut out = Vec::with_capacity(
            16 + chunks.len() * 8 + token_set_count * 8 + token_bytes + self.fast_wire_tail_len(),
        );
        out.extend_from_slice(b"DWF3");
        put_u32(&mut out, self.start_state);
        put_u32(&mut out, chunks.len() as u32);
        put_u32(&mut out, token_set_count as u32);
        for (body, &set_count) in chunks.iter().zip(&chunk_set_counts) {
            put_u32(&mut out, body.len() as u32);
            put_u32(&mut out, set_count);
        }
        put_u32s(&mut out, &locations);
        put_u32s(&mut out, word_spans);
        for body in chunks {
            out.extend_from_slice(body);
        }

        put_u32(&mut out, self.geometries.len() as u32);
        for geometry in self.geometries.iter() {
            put_u32(&mut out, geometry.len() as u32);
            for &(start, end) in geometry {
                put_u32(&mut out, start);
                put_u32(&mut out, end);
            }
        }
        put_u32(&mut out, self.weights.len() as u32);
        for weight in self.weights.iter() {
            put_u32(&mut out, weight.geometry);
            put_u32(&mut out, weight.token_ids_start);
            put_u32(&mut out, weight.full);
        }
        put_u32(&mut out, self.weight_token_ids.len() as u32);
        put_u32s(&mut out, &self.weight_token_ids);
        put_u32(&mut out, self.label_pool.values.len() as u32);
        put_i32s(&mut out, &self.label_pool.values);
        put_u32(&mut out, self.label_pool.spans.len() as u32);
        for &[start, len] in self.label_pool.spans.iter() {
            put_u32(&mut out, start);
            put_u32(&mut out, len);
        }
        put_u32(&mut out, self.target_pool.values.len() as u32);
        put_u32s(&mut out, &self.target_pool.values);
        put_u32(&mut out, self.target_pool.spans.len() as u32);
        for &[start, len] in self.target_pool.spans.iter() {
            put_u32(&mut out, start);
            put_u32(&mut out, len);
        }
        put_u32(&mut out, self.weight_id_pool.values.len() as u32);
        put_u32s(&mut out, &self.weight_id_pool.values);
        put_u32(&mut out, self.weight_id_pool.spans.len() as u32);
        for &[start, len] in self.weight_id_pool.spans.iter() {
            put_u32(&mut out, start);
            put_u32(&mut out, len);
        }
        put_u32(&mut out, self.rows.len() as u32);
        for row in self.rows.iter() {
            put_u32(&mut out, row.labels);
            put_u32(&mut out, row.targets);
            put_u32(&mut out, row.weights);
        }
        put_u32(&mut out, self.states.len() as u32);
        for state in self.states.iter() {
            put_u32(&mut out, state.row);
            put_u32(&mut out, state.final_weight);
        }
        Some(out)
    }

    fn fast_wire_len_for_chunks(&self, chunk_bodies: &[Box<[u8]>]) -> usize {
        let token_bytes = chunk_bodies.iter().map(|body| body.len()).sum::<usize>();
        let geometry_range_count = self.geometries.iter().map(Vec::len).sum::<usize>();
        12usize
            .saturating_add(token_bytes)
            // DWF2 token frames carry body length, set count, and range count.
            .saturating_add(chunk_bodies.len().saturating_mul(12))
            .saturating_add(4)
            .saturating_add(self.geometries.len().saturating_mul(4))
            .saturating_add(geometry_range_count.saturating_mul(8))
            .saturating_add(4)
            .saturating_add(self.weights.len().saturating_mul(12))
            .saturating_add(4)
            .saturating_add(self.weight_token_ids.len().saturating_mul(4))
            .saturating_add(4)
            .saturating_add(self.label_pool.values.len().saturating_mul(4))
            .saturating_add(4)
            .saturating_add(self.label_pool.spans.len().saturating_mul(8))
            .saturating_add(4)
            .saturating_add(self.target_pool.values.len().saturating_mul(4))
            .saturating_add(4)
            .saturating_add(self.target_pool.spans.len().saturating_mul(8))
            .saturating_add(4)
            .saturating_add(self.weight_id_pool.values.len().saturating_mul(4))
            .saturating_add(4)
            .saturating_add(self.weight_id_pool.spans.len().saturating_mul(8))
            .saturating_add(4)
            .saturating_add(self.rows.len().saturating_mul(12))
            .saturating_add(4)
            .saturating_add(self.states.len().saturating_mul(8))
    }

    /// Exact DWF2 size when the token-set wire chunks have already been cached.
    /// Fresh compiler-produced packed DWAs satisfy this; loaded unchanged
    /// constraints return their cached whole artifact before reaching here.
    pub fn fast_wire_len(&self) -> Option<usize> {
        self.backed
            .as_ref()
            .map(|backed| backed.section_len)
            .or_else(|| self.owned_narrow_fast_wire_len())
            .or_else(|| self.runtime_shaped_fast_wire_len())
            .or_else(|| {
                // Compiler-owned JS-like DWAs use the row-local DWF6 layout.
                // Its exact byte length depends only on already-materialized
                // runtime array lengths, not on any serialization pass. Return
                // that size here so the outer Constraint serializer can reserve
                // the final artifact once instead of growing again after the
                // 10+ MiB DWA has already been appended.
                if self.states.len() <= (1usize << 17) {
                    return None;
                }
                let ranges = self.flat_token_ranges_u16.as_deref()?;
                let spans = self.flat_token_spans.as_deref()?;
                if spans.len() != self.token_set_count() {
                    return None;
                }
                let geometry_range_count = self.geometries.iter().map(Vec::len).sum::<usize>();
                let transition_section_len = 4usize
                    .saturating_add(self.label_pool.values.len().saturating_mul(7))
                    .saturating_add(4 + self.label_pool.spans.len().saturating_mul(4));
                Some(
                    16usize
                        .saturating_add(spans.len().saturating_mul(3))
                        .saturating_add(spans.len().saturating_mul(2))
                        .saturating_add(ranges.len().saturating_mul(4))
                        .saturating_add(4)
                        .saturating_add(self.geometries.len().saturating_mul(2))
                        .saturating_add(geometry_range_count.saturating_mul(4))
                        .saturating_add(4 + self.weights.len().saturating_mul(6))
                        .saturating_add(4 + self.weight_token_ids.len().saturating_mul(2))
                        .saturating_add(transition_section_len)
                        .saturating_add(4 + self.states.len().saturating_mul(4)),
                )
            })
            .or_else(|| {
                self.fast_wire_token_chunks
                    .as_deref()
                    .map(|chunks| self.fast_wire_len_for_chunks(chunks))
            })
    }

    /// Borrow the canonical runtime wire when this DWA already executes from
    /// a backed DWF3/4/5/6 section. Compiler-produced constraints are
    /// canonicalized into this representation after runtime caches are built,
    /// so save() can copy actual runtime state rather than re-encode it.
    pub fn backed_fast_wire_bytes(&self) -> Option<&[u8]> {
        let backed = self.backed.as_ref()?;
        let end = backed.section_start.checked_add(backed.section_len)?;
        backed.backing.get(backed.section_start..end)
    }

    /// Runtime wire format: compress only the enormous token-range
    /// slab, while keeping the already-flat Weight/row/state arrays in a form
    /// that can be copied or viewed directly on load.  Token-set chunks are
    /// encoded independently so the expensive millions-of-ranges pass scales
    /// across cores without requiring lexicographic set sorting.
    pub fn fast_wire_bytes(&self) -> Vec<u8> {
        if std::env::var_os("GLRMASK_PROFILE_DWA_LAYOUT").is_some() {
            let token_range_count = self
                .materialized_token_sets
                .as_ref()
                .map(|sets| sets.iter().map(|set| set.ranges().len()).sum::<usize>())
                .or_else(|| self.flat_token_ranges.as_ref().map(|ranges| ranges.len()))
                .unwrap_or_else(|| self.token_set_chunks.iter().map(|chunk| chunk.ranges.len()).sum());
            let token_max = self
                .materialized_token_sets
                .as_ref()
                .and_then(|sets| sets.iter().flat_map(|set| set.ranges()).map(|r| *r.end()).max())
                .or_else(|| {
                    self.flat_token_ranges
                        .as_ref()
                        .and_then(|ranges| ranges.iter().map(|range| range[1]).max())
                })
                .or_else(|| {
                    self.token_set_chunks
                        .iter()
                        .flat_map(|chunk| chunk.ranges.iter())
                        .map(|range| range[1])
                        .max()
                })
                .unwrap_or(0);
            let geometry_ranges = self.geometries.iter().map(Vec::len).sum::<usize>();
            let geometry_max = self
                .geometries
                .iter()
                .flat_map(|geometry| geometry.iter())
                .flat_map(|&(start, end)| [start, end])
                .max()
                .unwrap_or(0);
            let span_stats = |spans: &[[u32; 2]]| {
                (
                    spans.iter().map(|span| span[0]).max().unwrap_or(0),
                    spans.iter().map(|span| span[1]).max().unwrap_or(0),
                )
            };
            let (label_span_start, label_span_len) = span_stats(&self.label_pool.spans);
            let (target_span_start, target_span_len) = span_stats(&self.target_pool.spans);
            let (weight_span_start, weight_span_len) = span_stats(&self.weight_id_pool.spans);
            eprintln!(
                "[glrmask/profile][packed_runtime_dwa_layout] token_sets={} token_ranges={} token_max={} geometries={} geometry_ranges={} geometry_max={} weights={} weight_token_ids={} weight_token_id_max={} label_values={} label_min={} label_max={} label_spans={} label_span_start_max={} label_span_len_max={} target_values={} target_max={} target_spans={} target_span_start_max={} target_span_len_max={} weight_values={} weight_value_max={} weight_spans={} weight_span_start_max={} weight_span_len_max={} rows={} row_field_max={} states={} state_row_max={} state_final_max={}",
                self.token_set_count(),
                token_range_count,
                token_max,
                self.geometries.len(),
                geometry_ranges,
                geometry_max,
                self.weights.len(),
                self.weight_token_ids.len(),
                self.weight_token_ids.iter().copied().max().unwrap_or(0),
                self.label_pool.values.len(),
                self.label_pool.values.iter().copied().min().unwrap_or(0),
                self.label_pool.values.iter().copied().max().unwrap_or(0),
                self.label_pool.spans.len(),
                label_span_start,
                label_span_len,
                self.target_pool.values.len(),
                self.target_pool.values.iter().copied().max().unwrap_or(0),
                self.target_pool.spans.len(),
                target_span_start,
                target_span_len,
                self.weight_id_pool.values.len(),
                self.weight_id_pool.values.iter().copied().max().unwrap_or(0),
                self.weight_id_pool.spans.len(),
                weight_span_start,
                weight_span_len,
                self.rows.len(),
                self.rows.iter().flat_map(|row| [row.labels, row.targets, row.weights]).max().unwrap_or(0),
                self.states.len(),
                self.states.iter().map(|state| state.row).max().unwrap_or(0),
                self.states.iter().filter_map(|state| (state.final_weight != u32::MAX).then_some(state.final_weight)).max().unwrap_or(0),
            );
        }
        if let Some(out) = self.try_backed_fast_wire_bytes() {
            return out;
        }
        let mut out = Vec::new();
        self.append_fast_wire_bytes(&mut out);
        out
    }

    /// Append DWF2 directly to an existing artifact buffer. This is the same
    /// wire representation as [`Self::fast_wire_bytes`], but lets the outer
    /// constraint serializer avoid allocating and then copying a second
    /// multi-megabyte DWA buffer.
    pub fn append_fast_wire_bytes(&self, out: &mut Vec<u8>) {
        let profile = std::env::var_os("GLRMASK_PROFILE_DWA_FAST_WIRE").is_some();
        if std::env::var_os("GLRMASK_PROFILE_DWA_LAYOUT").is_some() {
            let token_ranges = self
                .flat_token_ranges_u16
                .as_ref()
                .map_or(0, |ranges| ranges.len());
            eprintln!(
                "[glrmask/profile][packed_runtime_dwa_append_layout] states={} token_sets={} token_ranges_u16={} prefer_runtime_shaped={} direct_len={:?}",
                self.states.len(),
                self.token_set_count(),
                token_ranges,
                self.prefer_runtime_shaped_wire(),
                self.direct_fast_wire_len(),
            );
        }
        let total_started = profile.then(std::time::Instant::now);
        let runtime_start = out.len();
        if let Some(written) = self.try_append_runtime_shaped_fast_wire_bytes(out) {
            if let Some(total_started) = total_started {
                eprintln!(
                    "[glrmask/profile][packed_runtime_dwa_fast_wire_emit] format=DWF7 total_ms={:.3} bytes={}",
                    total_started.elapsed().as_secs_f64() * 1000.0,
                    written,
                );
            }
            return;
        }
        debug_assert_eq!(out.len(), runtime_start);
        let narrow_start = out.len();
        if let Some(written) = self.try_append_narrow_backed_fast_wire_bytes(out) {
            if let Some(total_started) = total_started {
                let format = if out.get(narrow_start..narrow_start + 4) == Some(b"DWF6") {
                    "DWF6"
                } else {
                    "DWF5"
                };
                eprintln!(
                    "[glrmask/profile][packed_runtime_dwa_fast_wire_emit] format={} total_ms={:.3} bytes={}",
                    format,
                    total_started.elapsed().as_secs_f64() * 1000.0,
                    written,
                );
            }
            return;
        }
        debug_assert_eq!(out.len(), narrow_start);
        #[inline]
        fn put_u32(out: &mut Vec<u8>, value: u32) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        #[inline]
        fn put_u32s(out: &mut Vec<u8>, values: &[u32]) {
            if cfg!(target_endian = "little") {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        values.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(values),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for &value in values {
                    put_u32(out, value);
                }
            }
        }
        #[inline]
        fn put_i32s(out: &mut Vec<u8>, values: &[i32]) {
            if cfg!(target_endian = "little") {
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
        fn put_rows(out: &mut Vec<u8>, rows: &[PackedRuntimeRow]) {
            if cfg!(target_endian = "little") {
                debug_assert_eq!(std::mem::size_of::<PackedRuntimeRow>(), 12);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        rows.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(rows),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for row in rows {
                    put_u32(out, row.labels);
                    put_u32(out, row.targets);
                    put_u32(out, row.weights);
                }
            }
        }
        #[inline]
        fn put_states(out: &mut Vec<u8>, states: &[PackedRuntimeState]) {
            if cfg!(target_endian = "little") {
                debug_assert_eq!(std::mem::size_of::<PackedRuntimeState>(), 8);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        states.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(states),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for state in states {
                    put_u32(out, state.row);
                    put_u32(out, state.final_weight);
                }
            }
        }

        const TOKEN_SET_WIRE_CHUNK: usize = 1024;
        let token_set_count = self.token_set_count();
        let uncached_chunk_bodies;
        let computed_chunk_range_counts;
        let chunk_bodies: &[Box<[u8]>] = if let Some(cached) = &self.fast_wire_token_chunks {
            cached
        } else {
            let chunk_ranges = (0..token_set_count)
                .step_by(TOKEN_SET_WIRE_CHUNK)
                .map(|start| (start, (start + TOKEN_SET_WIRE_CHUNK).min(token_set_count)))
                .collect::<Vec<_>>();
            let encode_chunk = |&(start_id, end_id): &(usize, usize)| {
                let mut body = Vec::<u8>::new();
                put_var_u32(&mut body, (end_id - start_id) as u32);
                for id in start_id..end_id {
                    let token_set = self
                        .token_set(id as u32)
                        .expect("packed runtime token-set id is valid");
                    put_var_u32(&mut body, token_set.range_count() as u32);
                    let mut previous_end_plus_one = 0u64;
                    token_set.for_each_range(|lo, hi| {
                        put_var_u64(&mut body, lo as u64 - previous_end_plus_one);
                        put_var_u32(&mut body, hi - lo);
                        previous_end_plus_one = hi as u64 + 1;
                    });
                }
                body.into_boxed_slice()
            };
            uncached_chunk_bodies = if chunk_ranges.len() >= 4
                && rayon::current_num_threads() > 1
            {
                chunk_ranges
                    .par_iter()
                    .map(encode_chunk)
                    .collect::<Vec<_>>()
            } else {
                chunk_ranges
                    .iter()
                    .map(encode_chunk)
                    .collect::<Vec<_>>()
            };
            &uncached_chunk_bodies
        };
        let chunk_range_counts: &[u32] = if let Some(cached) = &self.fast_wire_token_chunk_range_counts {
            if cached.len() == chunk_bodies.len() {
                cached
            } else {
                return;
            }
        } else {
            computed_chunk_range_counts = (0..chunk_bodies.len())
                .map(|chunk| {
                    let start = chunk * TOKEN_SET_WIRE_CHUNK;
                    let end = (start + TOKEN_SET_WIRE_CHUNK).min(token_set_count);
                    let count = (start..end)
                        .map(|id| {
                            self.token_set(id as u32)
                                .expect("packed runtime token-set id is valid")
                                .range_count()
                        })
                        .sum::<usize>();
                    u32::try_from(count).expect("packed runtime token chunk range count fits u32")
                })
                .collect::<Vec<_>>();
            &computed_chunk_range_counts
        };

        // Reserve the exact DWF2 payload. The old approximation implicitly
        // assumed all three sequence pools had the same span count. On large
        // parser DWAs it undershot by enough that the final row/state append
        // reallocated and copied the entire ~18 MB buffer.
        let wire_len = self.fast_wire_len_for_chunks(chunk_bodies);
        out.reserve(wire_len);
        out.extend_from_slice(b"DWF2");
        put_u32(out, self.start_state);
        put_u32(out, chunk_bodies.len() as u32);
        let token_started = profile.then(std::time::Instant::now);
        let token_frame_bytes = chunk_bodies
            .iter()
            .map(|body| 12usize.saturating_add(body.len()))
            .sum::<usize>();
        let token_frames_start = out.len();
        out.resize(token_frames_start + token_frame_bytes, 0);
        if chunk_bodies.len() >= 4 && rayon::current_num_threads() > 1 {
            let mut remaining = &mut out[token_frames_start..];
            let mut copies = Vec::with_capacity(chunk_bodies.len());
            for (chunk, body) in chunk_bodies.iter().enumerate() {
                let frame_len = 12 + body.len();
                let (frame, rest) = remaining.split_at_mut(frame_len);
                let mut body_pos = 0usize;
                let set_count = take_var_u32(body, &mut body_pos)
                    .expect("cached packed runtime token chunk has valid set count");
                copies.push((chunk, set_count, frame, body.as_ref()));
                remaining = rest;
            }
            copies
                .into_par_iter()
                .for_each(|(chunk, set_count, frame, body)| {
                frame[..4].copy_from_slice(&(body.len() as u32).to_le_bytes());
                frame[4..8].copy_from_slice(&set_count.to_le_bytes());
                frame[8..12].copy_from_slice(&chunk_range_counts[chunk].to_le_bytes());
                frame[12..].copy_from_slice(body);
                });
        } else {
            let mut pos = token_frames_start;
            for (chunk, body) in chunk_bodies.iter().enumerate() {
                let mut body_pos = 0usize;
                let set_count = take_var_u32(body, &mut body_pos)
                    .expect("cached packed runtime token chunk has valid set count");
                let frame_end = pos + 12 + body.len();
                out[pos..pos + 4].copy_from_slice(&(body.len() as u32).to_le_bytes());
                out[pos + 4..pos + 8].copy_from_slice(&set_count.to_le_bytes());
                out[pos + 8..pos + 12]
                    .copy_from_slice(&chunk_range_counts[chunk].to_le_bytes());
                out[pos + 12..frame_end].copy_from_slice(body);
                pos = frame_end;
            }
        }
        let token_ms = token_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let geometry_started = profile.then(std::time::Instant::now);
        put_u32(out, self.geometries.len() as u32);
        for geometry in self.geometries.iter() {
            put_u32(out, geometry.len() as u32);
            for &(start, end) in geometry {
                put_u32(out, start);
                put_u32(out, end);
            }
        }
        put_u32(out, self.weights.len() as u32);
        for weight in self.weights.iter() {
            put_u32(out, weight.geometry);
            put_u32(out, weight.token_ids_start);
            put_u32(out, weight.full);
        }
        put_u32(out, self.weight_token_ids.len() as u32);
        put_u32s(out, &self.weight_token_ids);
        let geometry_ms = geometry_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let pools_started = profile.then(std::time::Instant::now);
        put_u32(out, self.label_pool.values.len() as u32);
        put_i32s(out, &self.label_pool.values);
        put_u32(out, self.label_pool.spans.len() as u32);
        for &[start, len] in self.label_pool.spans.iter() {
            put_u32(out, start);
            put_u32(out, len);
        }
        put_u32(out, self.target_pool.values.len() as u32);
        put_u32s(out, &self.target_pool.values);
        put_u32(out, self.target_pool.spans.len() as u32);
        for &[start, len] in self.target_pool.spans.iter() {
            put_u32(out, start);
            put_u32(out, len);
        }
        put_u32(out, self.weight_id_pool.values.len() as u32);
        put_u32s(out, &self.weight_id_pool.values);
        put_u32(out, self.weight_id_pool.spans.len() as u32);
        for &[start, len] in self.weight_id_pool.spans.iter() {
            put_u32(out, start);
            put_u32(out, len);
        }
        let pools_ms = pools_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let rows_started = profile.then(std::time::Instant::now);
        put_u32(out, self.rows.len() as u32);
        put_rows(out, &self.rows);
        put_u32(out, self.states.len() as u32);
        put_states(out, &self.states);
        let rows_ms = rows_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][packed_runtime_dwa_fast_wire_emit] token_ms={token_ms:.3} geometry_weight_ms={geometry_ms:.3} pools_ms={pools_ms:.3} rows_states_ms={rows_ms:.3} total_ms={:.3} bytes={}",
                total_started.elapsed().as_secs_f64() * 1000.0,
                out.len(),
            );
        }
    }

    pub fn from_fast_wire_bytes_backed(
        input: &[u8],
        backing: Arc<Vec<u8>>,
        section_start: usize,
    ) -> Result<Self, String> {
        let section_end = section_start
            .checked_add(input.len())
            .ok_or_else(|| "overflowing backed DWA section range".to_owned())?;
        let backed_slice = backing
            .get(section_start..section_end)
            .ok_or_else(|| "backed DWA section is outside artifact".to_owned())?;
        if backed_slice.as_ptr() != input.as_ptr() || backed_slice.len() != input.len() {
            return Err("backed DWA section does not match artifact backing".to_owned());
        }
        if !input.starts_with(b"DWF3")
            && !input.starts_with(b"DWF4")
            && !input.starts_with(b"DWF5")
            && !input.starts_with(b"DWF6")
            && !input.starts_with(b"DWF7")
            && !input.starts_with(b"DWF8")
        {
            return Self::from_fast_wire_bytes(input);
        }
        let (start_state, backed) =
            BackedPackedRuntimeDwa::parse(backing, section_start, input.len())?;
        Ok(Self {
            start_state,
            backed: Some(backed),
            owned_narrow: None,
            token_set_chunks: Box::new([]),
            token_set_locations: Box::new([]),
            flat_token_ranges: None,
            flat_token_ranges_u16: None,
            flat_token_spans: None,
            materialized_token_sets: None,
            materialized_token_word_spans: None,
            fast_wire_token_chunks: None,
            fast_wire_token_chunk_range_counts: None,
            geometries: Box::new([]),
            weights: Box::new([]),
            weight_token_ids: Box::new([]),
            label_pool: PackedRuntimeSeqPool {
                values: Box::new([]),
                spans: Box::new([]),
            },
            target_pool: PackedRuntimeSeqPool {
                values: Box::new([]),
                spans: Box::new([]),
            },
            weight_id_pool: PackedRuntimeSeqPool {
                values: Box::new([]),
                spans: Box::new([]),
            },
            rows: Box::new([]),
            states: Box::new([]),
        })
    }

    pub fn from_fast_wire_bytes(input: &[u8]) -> Result<Self, String> {
        if input.starts_with(b"DWF3")
            || input.starts_with(b"DWF4")
            || input.starts_with(b"DWF5")
            || input.starts_with(b"DWF6")
            || input.starts_with(b"DWF7")
            || input.starts_with(b"DWF8")
        {
            let backing = Arc::new(input.to_vec());
            let section = backing.as_slice();
            return Self::from_fast_wire_bytes_backed(section, Arc::clone(&backing), 0);
        }
        #[inline]
        fn take_fixed_u32(input: &[u8], pos: &mut usize) -> Result<u32, String> {
            let end = pos
                .checked_add(4)
                .ok_or_else(|| "overflowing fast DWA offset".to_owned())?;
            let bytes: [u8; 4] = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA u32".to_owned())?
                .try_into()
                .expect("four-byte slice");
            *pos = end;
            Ok(u32::from_le_bytes(bytes))
        }
        fn take_u32_vec(input: &[u8], pos: &mut usize, len: usize) -> Result<Vec<u32>, String> {
            let byte_len = len
                .checked_mul(4)
                .ok_or_else(|| "overflowing fast DWA vector length".to_owned())?;
            let end = pos
                .checked_add(byte_len)
                .ok_or_else(|| "overflowing fast DWA vector offset".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA u32 vector".to_owned())?;
            *pos = end;
            if cfg!(target_endian = "little") {
                let mut out = Vec::<u32>::with_capacity(len);
                unsafe {
                    out.set_len(len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr().cast::<u8>(),
                        byte_len,
                    );
                }
                Ok(out)
            } else {
                Ok(bytes
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                    .collect())
            }
        }
        fn take_i32_vec(input: &[u8], pos: &mut usize, len: usize) -> Result<Vec<i32>, String> {
            let byte_len = len
                .checked_mul(4)
                .ok_or_else(|| "overflowing fast DWA i32 vector length".to_owned())?;
            let end = pos
                .checked_add(byte_len)
                .ok_or_else(|| "overflowing fast DWA i32 vector offset".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA i32 vector".to_owned())?;
            *pos = end;
            if cfg!(target_endian = "little") {
                let mut out = Vec::<i32>::with_capacity(len);
                unsafe {
                    out.set_len(len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr().cast::<u8>(),
                        byte_len,
                    );
                }
                Ok(out)
            } else {
                Ok(bytes
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                    .collect())
            }
        }
        fn take_rows(
            input: &[u8],
            pos: &mut usize,
            len: usize,
        ) -> Result<Vec<PackedRuntimeRow>, String> {
            let byte_len = len
                .checked_mul(std::mem::size_of::<PackedRuntimeRow>())
                .ok_or_else(|| "overflowing fast DWA row vector length".to_owned())?;
            let end = pos
                .checked_add(byte_len)
                .ok_or_else(|| "overflowing fast DWA row vector offset".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA row vector".to_owned())?;
            *pos = end;
            if cfg!(target_endian = "little") {
                debug_assert_eq!(std::mem::size_of::<PackedRuntimeRow>(), 12);
                let mut out = Vec::<PackedRuntimeRow>::with_capacity(len);
                unsafe {
                    out.set_len(len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr().cast::<u8>(),
                        byte_len,
                    );
                }
                Ok(out)
            } else {
                let mut local = 0usize;
                let mut out = Vec::with_capacity(len);
                for _ in 0..len {
                    out.push(PackedRuntimeRow {
                        labels: take_fixed_u32(bytes, &mut local)?,
                        targets: take_fixed_u32(bytes, &mut local)?,
                        weights: take_fixed_u32(bytes, &mut local)?,
                    });
                }
                Ok(out)
            }
        }
        fn take_fixed_bytes<'a>(
            input: &'a [u8],
            pos: &mut usize,
            byte_len: usize,
        ) -> Result<&'a [u8], String> {
            let end = pos
                .checked_add(byte_len)
                .ok_or_else(|| "overflowing fast DWA fixed-section offset".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA fixed section".to_owned())?;
            *pos = end;
            Ok(bytes)
        }
        fn take_weights(
            input: &[u8],
            pos: &mut usize,
            len: usize,
        ) -> Result<Vec<PackedRuntimeWeight>, String> {
            let byte_len = len
                .checked_mul(std::mem::size_of::<PackedRuntimeWeight>())
                .ok_or_else(|| "overflowing fast DWA Weight vector length".to_owned())?;
            let end = pos
                .checked_add(byte_len)
                .ok_or_else(|| "overflowing fast DWA Weight vector offset".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA Weight vector".to_owned())?;
            *pos = end;
            if cfg!(target_endian = "little") {
                debug_assert_eq!(std::mem::size_of::<PackedRuntimeWeight>(), 12);
                let mut out = Vec::<PackedRuntimeWeight>::with_capacity(len);
                unsafe {
                    out.set_len(len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr().cast::<u8>(),
                        byte_len,
                    );
                }
                Ok(out)
            } else {
                let mut local = 0usize;
                let mut out = Vec::with_capacity(len);
                for _ in 0..len {
                    out.push(PackedRuntimeWeight {
                        geometry: take_fixed_u32(bytes, &mut local)?,
                        token_ids_start: take_fixed_u32(bytes, &mut local)?,
                        full: take_fixed_u32(bytes, &mut local)?,
                    });
                }
                Ok(out)
            }
        }
        fn take_u32_pairs(
            input: &[u8],
            pos: &mut usize,
            len: usize,
        ) -> Result<Vec<[u32; 2]>, String> {
            let byte_len = len
                .checked_mul(8)
                .ok_or_else(|| "overflowing fast DWA pair vector length".to_owned())?;
            let end = pos
                .checked_add(byte_len)
                .ok_or_else(|| "overflowing fast DWA pair vector offset".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA pair vector".to_owned())?;
            *pos = end;
            if cfg!(target_endian = "little") {
                let mut out = Vec::<[u32; 2]>::with_capacity(len);
                unsafe {
                    out.set_len(len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr().cast::<u8>(),
                        byte_len,
                    );
                }
                Ok(out)
            } else {
                Ok(bytes
                    .chunks_exact(8)
                    .map(|chunk| {
                        [
                            u32::from_le_bytes(chunk[..4].try_into().expect("four-byte chunk")),
                            u32::from_le_bytes(chunk[4..].try_into().expect("four-byte chunk")),
                        ]
                    })
                    .collect())
            }
        }
        fn take_states(
            input: &[u8],
            pos: &mut usize,
            len: usize,
        ) -> Result<Vec<PackedRuntimeState>, String> {
            let byte_len = len
                .checked_mul(std::mem::size_of::<PackedRuntimeState>())
                .ok_or_else(|| "overflowing fast DWA state vector length".to_owned())?;
            let end = pos
                .checked_add(byte_len)
                .ok_or_else(|| "overflowing fast DWA state vector offset".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA state vector".to_owned())?;
            *pos = end;
            if cfg!(target_endian = "little") {
                debug_assert_eq!(std::mem::size_of::<PackedRuntimeState>(), 8);
                let mut out = Vec::<PackedRuntimeState>::with_capacity(len);
                unsafe {
                    out.set_len(len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr().cast::<u8>(),
                        byte_len,
                    );
                }
                Ok(out)
            } else {
                let mut local = 0usize;
                let mut out = Vec::with_capacity(len);
                for _ in 0..len {
                    out.push(PackedRuntimeState {
                        row: take_fixed_u32(bytes, &mut local)?,
                        final_weight: take_fixed_u32(bytes, &mut local)?,
                    });
                }
                Ok(out)
            }
        }
        fn decode_token_chunk(body: &[u8]) -> Result<PackedRuntimeTokenSetChunk, String> {
            #[inline(always)]
            fn take_range_gap_len(body: &[u8], pos: &mut usize) -> Result<(u32, u32), String> {
                // JS-like packed token sets overwhelmingly encode both gap and
                // range length in one byte. Decode that pair together so the
                // hot 1.5M-range loop pays one bounds check rather than two
                // independent varint state machines.
                if let Some(pair) = body.get(*pos..pos.saturating_add(2))
                    && pair[0] < 0x80
                    && pair[1] < 0x80
                {
                    *pos += 2;
                    return Ok((pair[0] as u32, pair[1] as u32));
                }
                let gap = take_var_u32(body, pos)?;
                let len = take_var_u32(body, pos)?;
                Ok((gap, len))
            }
            let mut pos = 0usize;
            let set_count = take_var_u32(body, &mut pos)? as usize;
            // Every range consumes at least one gap byte and one length byte;
            // reserve from that hard upper bound so large JS chunks do not
            // repeatedly reallocate/copy tens of thousands of decoded ranges.
            let mut ranges = Vec::<[u32; 2]>::with_capacity(body.len() / 2);
            let mut spans = Vec::<PackedRuntimeTokenSetSpan>::with_capacity(set_count);
            for _ in 0..set_count {
                let range_count = take_var_u32(body, &mut pos)? as usize;
                let start_index = u32::try_from(ranges.len())
                    .map_err(|_| "fast DWA token chunk exceeds u32 offsets".to_owned())?;
                let mut previous_end_plus_one = 0u64;
                let mut word_spans = 0u32;
                for _ in 0..range_count {
                    let (gap, len) = take_range_gap_len(body, &mut pos)?;
                    let lo64 = previous_end_plus_one
                        .checked_add(gap as u64)
                        .ok_or_else(|| "overflowing fast DWA token range".to_owned())?;
                    let lo = u32::try_from(lo64)
                        .map_err(|_| "overflowing fast DWA token start".to_owned())?;
                    let hi = lo
                        .checked_add(len)
                        .ok_or_else(|| "overflowing fast DWA token end".to_owned())?;
                    ranges.push([lo, hi]);
                    word_spans = word_spans.saturating_add(hi / 64 - lo / 64 + 1);
                    previous_end_plus_one = hi as u64 + 1;
                }
                spans.push(PackedRuntimeTokenSetSpan {
                    start: start_index,
                    len: range_count as u32,
                    word_spans,
                });
            }
            if pos != body.len() {
                return Err("trailing bytes in fast DWA token chunk".to_owned());
            }
            Ok(PackedRuntimeTokenSetChunk {
                ranges: ranges.into_boxed_slice(),
                spans: spans.into_boxed_slice(),
            })
        }

        fn decode_token_chunk_into(
            body: &[u8],
            expected_set_count: usize,
            global_range_start: usize,
            ranges: &mut [std::mem::MaybeUninit<[u32; 2]>],
            spans: &mut [std::mem::MaybeUninit<PackedRuntimeTokenSetSpan>],
        ) -> Result<(), String> {
            #[inline(always)]
            fn take_range_gap_len(body: &[u8], pos: &mut usize) -> Result<(u32, u32), String> {
                if let Some(pair) = body.get(*pos..pos.saturating_add(2))
                    && pair[0] < 0x80
                    && pair[1] < 0x80
                {
                    *pos += 2;
                    return Ok((pair[0] as u32, pair[1] as u32));
                }
                let gap = take_var_u32(body, pos)?;
                let len = take_var_u32(body, pos)?;
                Ok((gap, len))
            }
            let mut pos = 0usize;
            let encoded_set_count = take_var_u32(body, &mut pos)? as usize;
            if encoded_set_count != expected_set_count || spans.len() != expected_set_count {
                return Err("fast DWA token chunk set-count mismatch".to_owned());
            }
            let mut range_index = 0usize;
            for span in spans.iter_mut() {
                let range_count = take_var_u32(body, &mut pos)? as usize;
                let range_end = range_index
                    .checked_add(range_count)
                    .ok_or_else(|| "fast DWA token range count overflow".to_owned())?;
                if range_end > ranges.len() {
                    return Err("fast DWA token chunk exceeds declared range count".to_owned());
                }
                let mut previous_end_plus_one = 0u64;
                let mut word_spans = 0u32;
                for slot in &mut ranges[range_index..range_end] {
                    let (gap, len) = take_range_gap_len(body, &mut pos)?;
                    let lo64 = previous_end_plus_one
                        .checked_add(gap as u64)
                        .ok_or_else(|| "overflowing fast DWA token range".to_owned())?;
                    let lo = u32::try_from(lo64)
                        .map_err(|_| "overflowing fast DWA token start".to_owned())?;
                    let hi = lo
                        .checked_add(len)
                        .ok_or_else(|| "overflowing fast DWA token end".to_owned())?;
                    slot.write([lo, hi]);
                    word_spans = word_spans.saturating_add(hi / 64 - lo / 64 + 1);
                    previous_end_plus_one = hi as u64 + 1;
                }
                span.write(PackedRuntimeTokenSetSpan {
                    start: u32::try_from(global_range_start + range_index)
                        .map_err(|_| "fast DWA token range slab exceeds u32".to_owned())?,
                    len: u32::try_from(range_count)
                        .map_err(|_| "fast DWA token set exceeds u32 ranges".to_owned())?,
                    word_spans,
                });
                range_index = range_end;
            }
            if range_index != ranges.len() {
                return Err("fast DWA token chunk used fewer ranges than declared".to_owned());
            }
            if pos != body.len() {
                return Err("trailing bytes in fast DWA token chunk".to_owned());
            }
            Ok(())
        }

        let profile = std::env::var_os("GLRMASK_PROFILE_DWA_RUNTIME").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let dwf2 = input.starts_with(b"DWF2");
        if !dwf2 && !input.starts_with(b"DWF1") {
            return Err("invalid fast runtime DWA header".to_owned());
        }
        let mut pos = 4usize;
        let start_state = take_fixed_u32(input, &mut pos)?;

        let scan_started = profile.then(std::time::Instant::now);
        let chunk_count = take_fixed_u32(input, &mut pos)? as usize;
        let mut chunk_bodies = Vec::<&[u8]>::with_capacity(chunk_count);
        let mut chunk_set_counts = Vec::<usize>::with_capacity(if dwf2 { chunk_count } else { 0 });
        let mut chunk_range_counts =
            Vec::<usize>::with_capacity(if dwf2 { chunk_count } else { 0 });
        for _ in 0..chunk_count {
            let len = take_fixed_u32(input, &mut pos)? as usize;
            if dwf2 {
                chunk_set_counts.push(take_fixed_u32(input, &mut pos)? as usize);
                chunk_range_counts.push(take_fixed_u32(input, &mut pos)? as usize);
            }
            let end = pos
                .checked_add(len)
                .ok_or_else(|| "overflowing fast DWA token-chunk length".to_owned())?;
            let body = input
                .get(pos..end)
                .ok_or_else(|| "truncated fast DWA token chunk".to_owned())?;
            chunk_bodies.push(body);
            pos = end;
        }
        let after_chunks = pos;
        enum DecodedFastTokenSets {
            Dwf1(Vec<PackedRuntimeTokenSetChunk>),
            Dwf2 {
                ranges: Box<[[u32; 2]]>,
                spans: Box<[PackedRuntimeTokenSetSpan]>,
            },
        }
        let decode_tokens = || -> Result<(DecodedFastTokenSets, usize, usize, f64), String> {
            let started = profile.then(std::time::Instant::now);
            if dwf2 {
                let total_sets = chunk_set_counts.iter().try_fold(0usize, |sum, &count| {
                    sum.checked_add(count)
                        .ok_or_else(|| "fast DWA token-set count overflow".to_owned())
                })?;
                let total_ranges = chunk_range_counts.iter().try_fold(0usize, |sum, &count| {
                    sum.checked_add(count)
                        .ok_or_else(|| "fast DWA token-range count overflow".to_owned())
                })?;
                let mut ranges = Vec::<std::mem::MaybeUninit<[u32; 2]>>::with_capacity(total_ranges);
                let mut spans =
                    Vec::<std::mem::MaybeUninit<PackedRuntimeTokenSetSpan>>::with_capacity(total_sets);
                // MaybeUninit elements are safe to leave uninitialized while
                // each chunk owns a disjoint final slice. Successful decoders
                // write every slot before the slabs are reinterpreted below.
                unsafe {
                    ranges.set_len(total_ranges);
                    spans.set_len(total_sets);
                }
                let mut range_remaining = ranges.as_mut_slice();
                let mut span_remaining = spans.as_mut_slice();
                let mut global_range_start = 0usize;
                let mut jobs = Vec::with_capacity(chunk_count);
                for chunk in 0..chunk_count {
                    let range_count = chunk_range_counts[chunk];
                    let set_count = chunk_set_counts[chunk];
                    let (range_slice, range_rest) = range_remaining.split_at_mut(range_count);
                    let (span_slice, span_rest) = span_remaining.split_at_mut(set_count);
                    jobs.push((
                        chunk_bodies[chunk],
                        set_count,
                        global_range_start,
                        range_slice,
                        span_slice,
                    ));
                    range_remaining = range_rest;
                    span_remaining = span_rest;
                    global_range_start += range_count;
                }
                let decode_job = |(body, set_count, range_start, range_slice, span_slice)| {
                    decode_token_chunk_into(
                        body,
                        set_count,
                        range_start,
                        range_slice,
                        span_slice,
                    )
                };
                if chunk_count >= 4 && rayon::current_num_threads() > 1 {
                    jobs.into_par_iter()
                        .map(decode_job)
                        .collect::<Result<Vec<_>, _>>()?;
                } else {
                    for job in jobs {
                        decode_job(job)?;
                    }
                }
                let ranges = ranges.into_boxed_slice();
                let spans = spans.into_boxed_slice();
                // SAFETY: every MaybeUninit slot belongs to exactly one job,
                // and decode_token_chunk_into returns success only after
                // writing exactly the declared number of range/span slots.
                let ranges = unsafe {
                    Box::from_raw(Box::into_raw(ranges) as *mut [[u32; 2]])
                };
                let spans = unsafe {
                    Box::from_raw(
                        Box::into_raw(spans) as *mut [PackedRuntimeTokenSetSpan],
                    )
                };
                let ms = started
                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                Ok((
                    DecodedFastTokenSets::Dwf2 { ranges, spans },
                    total_sets,
                    total_ranges,
                    ms,
                ))
            } else {
                let chunks = if chunk_count >= 4 && rayon::current_num_threads() > 1 {
                    chunk_bodies
                        .par_iter()
                        .map(|body| decode_token_chunk(body))
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    chunk_bodies
                        .iter()
                        .map(|body| decode_token_chunk(body))
                        .collect::<Result<Vec<_>, _>>()?
                };
                let total_sets = chunks.iter().map(|chunk| chunk.spans.len()).sum::<usize>();
                let total_ranges = chunks.iter().map(|chunk| chunk.ranges.len()).sum::<usize>();
                let ms = started
                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                Ok((DecodedFastTokenSets::Dwf1(chunks), total_sets, total_ranges, ms))
            }
        };
        let decode_other = || -> Result<_, String> {
            let started = profile.then(std::time::Instant::now);
            let mut pos = after_chunks;
            let geometry_started = profile.then(std::time::Instant::now);
            let geometry_count = take_fixed_u32(input, &mut pos)? as usize;
            let mut geometries = Vec::<Vec<(u32, u32)>>::with_capacity(geometry_count);
            for _ in 0..geometry_count {
                let len = take_fixed_u32(input, &mut pos)? as usize;
                let mut geometry = Vec::with_capacity(len);
                for _ in 0..len {
                    geometry.push((
                        take_fixed_u32(input, &mut pos)?,
                        take_fixed_u32(input, &mut pos)?,
                    ));
                }
                geometries.push(geometry);
            }
            let geometry_ms = geometry_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

            let weights_started = profile.then(std::time::Instant::now);
            let weight_count = take_fixed_u32(input, &mut pos)? as usize;
            let weights = take_weights(input, &mut pos, weight_count)?;
            if weights.iter().any(|weight| {
                weight.full == 0 && weight.geometry as usize >= geometries.len()
            }) {
                return Err("invalid fast DWA Weight geometry".to_owned());
            }
            let weight_token_id_count = take_fixed_u32(input, &mut pos)? as usize;
            let weight_token_ids = take_u32_vec(input, &mut pos, weight_token_id_count)?;
            let weights_ms = weights_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

            let pools_started = profile.then(std::time::Instant::now);
            let label_value_count = take_fixed_u32(input, &mut pos)? as usize;
            let label_value_bytes = take_fixed_bytes(
                input,
                &mut pos,
                label_value_count
                    .checked_mul(4)
                    .ok_or_else(|| "fast DWA label values overflow".to_owned())?,
            )?;
            let label_span_count = take_fixed_u32(input, &mut pos)? as usize;
            let label_span_bytes = take_fixed_bytes(
                input,
                &mut pos,
                label_span_count
                    .checked_mul(8)
                    .ok_or_else(|| "fast DWA label spans overflow".to_owned())?,
            )?;
            let target_value_count = take_fixed_u32(input, &mut pos)? as usize;
            let target_value_bytes = take_fixed_bytes(
                input,
                &mut pos,
                target_value_count
                    .checked_mul(4)
                    .ok_or_else(|| "fast DWA target values overflow".to_owned())?,
            )?;
            let target_span_count = take_fixed_u32(input, &mut pos)? as usize;
            let target_span_bytes = take_fixed_bytes(
                input,
                &mut pos,
                target_span_count
                    .checked_mul(8)
                    .ok_or_else(|| "fast DWA target spans overflow".to_owned())?,
            )?;
            let weight_value_count = take_fixed_u32(input, &mut pos)? as usize;
            let weight_value_bytes = take_fixed_bytes(
                input,
                &mut pos,
                weight_value_count
                    .checked_mul(4)
                    .ok_or_else(|| "fast DWA weight-id values overflow".to_owned())?,
            )?;
            let weight_span_count = take_fixed_u32(input, &mut pos)? as usize;
            let weight_span_bytes = take_fixed_bytes(
                input,
                &mut pos,
                weight_span_count
                    .checked_mul(8)
                    .ok_or_else(|| "fast DWA weight-id spans overflow".to_owned())?,
            )?;
            if label_span_count != target_span_count || label_span_count != weight_span_count {
                return Err("mismatched fast DWA row-pool spans".to_owned());
            }
            let decode_label_pool = || -> Result<(Vec<i32>, Vec<[u32; 2]>), String> {
                let mut values_pos = 0usize;
                let values = take_i32_vec(label_value_bytes, &mut values_pos, label_value_count)?;
                let mut spans_pos = 0usize;
                let spans = take_u32_pairs(label_span_bytes, &mut spans_pos, label_span_count)?;
                Ok((values, spans))
            };
            let decode_target_pool = || -> Result<(Vec<u32>, Vec<[u32; 2]>), String> {
                let mut values_pos = 0usize;
                let values = take_u32_vec(target_value_bytes, &mut values_pos, target_value_count)?;
                let mut spans_pos = 0usize;
                let spans = take_u32_pairs(target_span_bytes, &mut spans_pos, target_span_count)?;
                Ok((values, spans))
            };
            let decode_weight_pool = || -> Result<(Vec<u32>, Vec<[u32; 2]>), String> {
                let mut values_pos = 0usize;
                let values = take_u32_vec(weight_value_bytes, &mut values_pos, weight_value_count)?;
                let mut spans_pos = 0usize;
                let spans = take_u32_pairs(weight_span_bytes, &mut spans_pos, weight_span_count)?;
                Ok((values, spans))
            };
            // These copies run inside the "other DWA" half of an outer
            // `rayon::join` against token-set decoding. Spawning nested Rayon
            // work here just competes with that critical token branch on large
            // JS constraints and was neutral/slightly worse in practice.
            let label_pool = decode_label_pool();
            let target_pool = decode_target_pool();
            let weight_pool = decode_weight_pool();
            let (label_values, label_spans) = label_pool?;
            let (target_values, target_spans) = target_pool?;
            let (weight_values, weight_spans) = weight_pool?;
            let pools_ms = pools_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

            let rows_started = profile.then(std::time::Instant::now);
            let row_count = take_fixed_u32(input, &mut pos)? as usize;
            let rows = take_rows(input, &mut pos, row_count)?;
            let state_count = take_fixed_u32(input, &mut pos)? as usize;
            if state_count != 0 && start_state as usize >= state_count {
                return Err("invalid fast DWA start state".to_owned());
            }
            let states = take_states(input, &mut pos, state_count)?;
            if states.iter().any(|state| {
                state.row as usize >= row_count
                    || (state.final_weight != u32::MAX
                        && state.final_weight as usize >= weight_count)
            }) {
                return Err("invalid fast DWA state row/final Weight".to_owned());
            }
            if pos != input.len() {
                return Err("trailing bytes in fast runtime DWA".to_owned());
            }
            let rows_ms = rows_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            let ms = started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            if profile {
                eprintln!(
                    "[glrmask/profile][packed_runtime_dwa_fast_load_other] geometry_ms={geometry_ms:.3} weights_ms={weights_ms:.3} pools_ms={pools_ms:.3} rows_states_ms={rows_ms:.3} geometries={} geometry_ranges={} label_values={} spans={} weights={} states={}",
                    geometries.len(),
                    geometries.iter().map(Vec::len).sum::<usize>(),
                    label_values.len(),
                    label_spans.len(),
                    weights.len(),
                    states.len(),
                );
            }
            Ok((
                geometries,
                weights,
                weight_token_ids,
                label_values,
                label_spans,
                target_values,
                target_spans,
                weight_values,
                weight_spans,
                rows,
                states,
                ms,
            ))
        };
        let (token_result, other_result) = if rayon::current_num_threads() > 1 {
            rayon::join(decode_tokens, decode_other)
        } else {
            (decode_tokens(), decode_other())
        };
        let (decoded_token_sets, token_set_count, token_range_count, token_ms) = token_result?;
        let (
            geometries,
            weights,
            weight_token_ids,
            label_values,
            label_spans,
            target_values,
            target_spans,
            weight_values,
            weight_spans,
            rows,
            states,
            other_ms,
        ) = other_result?;
        let (token_set_chunks, token_set_locations, flat_token_ranges, flat_token_spans) =
            match decoded_token_sets {
                DecodedFastTokenSets::Dwf1(token_set_chunks) => {
                    let mut token_set_locations =
                        Vec::<PackedRuntimeTokenSetLocation>::with_capacity(token_set_count);
                    for (chunk, decoded) in token_set_chunks.iter().enumerate() {
                        for local in 0..decoded.spans.len() {
                            token_set_locations.push(PackedRuntimeTokenSetLocation {
                                chunk: chunk as u32,
                                local: local as u32,
                            });
                        }
                    }
                    (
                        token_set_chunks.into_boxed_slice(),
                        token_set_locations.into_boxed_slice(),
                        None,
                        None,
                    )
                }
                DecodedFastTokenSets::Dwf2 { ranges, spans } => {
                    (
                        Vec::<PackedRuntimeTokenSetChunk>::new().into_boxed_slice(),
                        Vec::<PackedRuntimeTokenSetLocation>::new().into_boxed_slice(),
                        Some(ranges),
                        Some(spans),
                    )
                }
            };
        let scan_ms = scan_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][packed_runtime_dwa_fast_load] token_ms={token_ms:.3} other_ms={other_ms:.3} scan_inclusive_ms={scan_ms:.3} total_ms={:.3} bytes={} states={} token_sets={} token_ranges={} weights={} rows={}",
                total_started.elapsed().as_secs_f64() * 1000.0,
                input.len(),
                states.len(),
                token_set_count,
                token_range_count,
                weights.len(),
                rows.len(),
            );
        }

        Ok(Self {
            start_state,
            backed: None,
            owned_narrow: None,
            token_set_chunks,
            token_set_locations,
            flat_token_ranges,
            flat_token_ranges_u16: None,
            flat_token_spans,
            materialized_token_sets: None,
            materialized_token_word_spans: None,
            fast_wire_token_chunks: None,
            fast_wire_token_chunk_range_counts: None,
            geometries: geometries.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
            weight_token_ids: weight_token_ids.into_boxed_slice(),
            label_pool: PackedRuntimeSeqPool {
                values: label_values.into_boxed_slice(),
                spans: label_spans.into_boxed_slice(),
            },
            target_pool: PackedRuntimeSeqPool {
                values: target_values.into_boxed_slice(),
                spans: target_spans.into_boxed_slice(),
            },
            weight_id_pool: PackedRuntimeSeqPool {
                values: weight_values.into_boxed_slice(),
                spans: weight_spans.into_boxed_slice(),
            },
            rows: rows.into_boxed_slice(),
            states: states.into_boxed_slice(),
        })
    }

    /// Build directly from a finalized DWA whose transition rows have already
    /// been canonicalized by identity.  This avoids the old temporary
    /// Vec<Vec<(label,target,weight-id)>> representation and retains canonical
    /// RangeSetBlaze token-set Arcs instead of copying every range immediately.
    fn from_shared_dwa_direct(dwa: &DWA) -> Result<Self, String> {
        let profile = std::env::var_os("GLRMASK_PROFILE_DWA_RUNTIME_BUILD").is_some();
        let total_started = profile.then(std::time::Instant::now);

        let row_classes_started = profile.then(std::time::Instant::now);
        let mut row_id_by_ptr = FxHashMap::<usize, u32>::default();
        row_id_by_ptr.reserve(dwa.states.len().min(32_768));
        let mut representative_states = Vec::<usize>::new();
        let mut state_row_ids = Vec::<u32>::with_capacity(dwa.states.len());
        for (state_index, state) in dwa.states.iter().enumerate() {
            let ptr = state.transitions.ptr_key();
            let row_id = if let Some(&row_id) = row_id_by_ptr.get(&ptr) {
                row_id
            } else {
                let row_id = representative_states.len() as u32;
                row_id_by_ptr.insert(ptr, row_id);
                representative_states.push(state_index);
                row_id
            };
            state_row_ids.push(row_id);
        }
        let row_classes_ms = row_classes_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let rows_started = profile.then(std::time::Instant::now);
        let mut local_weight_by_ptr = FxHashMap::<usize, u32>::default();
        local_weight_by_ptr.reserve(32_768);
        let mut weight_refs = Vec::<Weight>::new();
        let mut intern_weight = |weight: &Weight| -> u32 {
            let ptr = weight.ptr_key();
            *local_weight_by_ptr.entry(ptr).or_insert_with(|| {
                let id = weight_refs.len() as u32;
                weight_refs.push(weight.clone());
                id
            })
        };

        let total_row_entries = representative_states
            .iter()
            .map(|&state_index| dwa.states[state_index].transitions.len())
            .sum::<usize>();
        let mut label_values = Vec::<Label>::with_capacity(total_row_entries);
        let mut target_values = Vec::<u32>::with_capacity(total_row_entries);
        let mut weight_values = Vec::<u32>::with_capacity(total_row_entries);
        let mut label_spans = Vec::<[u32; 2]>::with_capacity(representative_states.len());
        let mut target_spans = Vec::<[u32; 2]>::with_capacity(representative_states.len());
        let mut weight_spans = Vec::<[u32; 2]>::with_capacity(representative_states.len());
        let mut rows = Vec::<PackedRuntimeRow>::with_capacity(representative_states.len());
        for (row_id, &state_index) in representative_states.iter().enumerate() {
            let row = &dwa.states[state_index].transitions;
            let start = u32::try_from(label_values.len())
                .map_err(|_| "packed runtime DWA row pool exceeds u32 offsets".to_owned())?;
            let len = u32::try_from(row.len())
                .map_err(|_| "packed runtime DWA row exceeds u32 entries".to_owned())?;
            for (label, target, weight) in row.entries() {
                label_values.push(label);
                target_values.push(target);
                weight_values.push(intern_weight(weight));
            }
            label_spans.push([start, len]);
            target_spans.push([start, len]);
            weight_spans.push([start, len]);
            let id = u32::try_from(row_id)
                .map_err(|_| "packed runtime DWA has too many rows".to_owned())?;
            rows.push(PackedRuntimeRow {
                labels: id,
                targets: id,
                weights: id,
            });
        }
        let rows_ms = rows_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let states_started = profile.then(std::time::Instant::now);
        let mut states = Vec::<PackedRuntimeState>::with_capacity(dwa.states.len());
        for (state, row) in dwa.states.iter().zip(state_row_ids) {
            let final_weight = state
                .final_weight
                .as_ref()
                .map(&mut intern_weight)
                .unwrap_or(u32::MAX);
            states.push(PackedRuntimeState { row, final_weight });
        }
        let states_ms = states_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let weights_started = profile.then(std::time::Instant::now);
        // Keep this identity interner growable: distinct token sets are not bounded by
        // the number of weights, because one weight can reference many different sets.
        let mut token_id_by_ptr = FxHashMap::<usize, u32>::default();
        token_id_by_ptr.reserve(weight_refs.len().saturating_mul(4).max(65_536));
        let mut token_sets = Vec::<Arc<RangeSetBlaze<u32>>>::new();
        let mut geometries = vec![Vec::<(u32, u32)>::new()];
        let mut geometry_ids = FxHashMap::<Vec<(u32, u32)>, u32>::default();
        geometry_ids.insert(Vec::new(), 0);
        let mut weights = Vec::<PackedRuntimeWeight>::with_capacity(weight_refs.len());
        let mut weight_token_ids = Vec::<u32>::with_capacity(weight_refs.len().saturating_mul(2));
        for weight in &weight_refs {
            if weight.is_full() {
                weights.push(PackedRuntimeWeight {
                    geometry: 0,
                    token_ids_start: weight_token_ids.len() as u32,
                    full: 1,
                });
                continue;
            }
            let token_ids_start = u32::try_from(weight_token_ids.len())
                .map_err(|_| "packed runtime DWA weight token ids exceed u32 offsets".to_owned())?;
            let mut geometry = Vec::<(u32, u32)>::new();
            for (range, tokens) in weight.raw_range_values() {
                geometry.push((*range.start(), *range.end()));
                let ptr = Arc::as_ptr(tokens) as usize;
                let token_id = if let Some(&id) = token_id_by_ptr.get(&ptr) {
                    id
                } else {
                    let id = u32::try_from(token_sets.len())
                        .map_err(|_| "packed runtime DWA has too many token sets".to_owned())?;
                    token_id_by_ptr.insert(ptr, id);
                    token_sets.push(Arc::clone(tokens));
                    id
                };
                weight_token_ids.push(token_id);
            }
            let geometry_id = if let Some(&id) = geometry_ids.get(&geometry) {
                id
            } else {
                let id = u32::try_from(geometries.len())
                    .map_err(|_| "packed runtime DWA has too many weight geometries".to_owned())?;
                geometry_ids.insert(geometry.clone(), id);
                geometries.push(geometry);
                id
            };
            weights.push(PackedRuntimeWeight {
                geometry: geometry_id,
                token_ids_start,
                full: 0,
            });
        }
        let weight_pool_ms = weights_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let token_range_count = token_sets
            .iter()
            .map(|tokens| tokens.ranges_len())
            .sum::<usize>();
        let token_stats_started = profile.then(std::time::Instant::now);
        // Token-set ranges are runtime data: masking walks them on every
        // relevant weight. Store the compact u16 slab directly when possible
        // instead of retaining RangeSetBlaze Arcs and later rebuilding a wire
        // representation in save(). The same pass computes word-span metadata
        // already required by mask planning.
        let mut flat_token_ranges_u16 = Vec::<[u16; 2]>::with_capacity(token_range_count);
        let mut flat_token_spans = Vec::<PackedRuntimeTokenSetSpan>::with_capacity(token_sets.len());
        let mut token_word_spans = Vec::<u32>::with_capacity(token_sets.len());
        let mut narrow_token_ranges = true;
        for tokens in &token_sets {
            let start = flat_token_ranges_u16.len();
            let mut word_spans = 0u32;
            for range in tokens.ranges() {
                let lo = *range.start();
                let hi = *range.end();
                word_spans = word_spans.saturating_add(hi / 64 - lo / 64 + 1);
                if hi <= u16::MAX as u32 {
                    flat_token_ranges_u16.push([lo as u16, hi as u16]);
                } else {
                    narrow_token_ranges = false;
                }
            }
            token_word_spans.push(word_spans);
            if narrow_token_ranges {
                flat_token_spans.push(PackedRuntimeTokenSetSpan {
                    start: u32::try_from(start)
                        .map_err(|_| "packed runtime token range slab exceeds u32".to_owned())?,
                    len: u32::try_from(flat_token_ranges_u16.len() - start)
                        .map_err(|_| "packed runtime token set exceeds u32 ranges".to_owned())?,
                    word_spans,
                });
            }
        }
        if !narrow_token_ranges {
            flat_token_ranges_u16.clear();
            flat_token_spans.clear();
        }
        let token_stats_ms = token_stats_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        // When the finalized parser fits DWF6's widths, build the same narrow
        // arrays as a live execution view. Runtime accessors below prefer this
        // representation, so the work is part of runtime packing rather than
        // serialization preparation. Keeping the wide compiler-facing pools
        // alongside it preserves cheap cold materialization/composition.
        const DEFAULT_LABEL_WIRE: i32 = i32::MAX - 1;
        const SPAN_LEN_BITS: u32 = 10;
        const SPAN_LEN_MAX: u32 = (1 << SPAN_LEN_BITS) - 1;
        const SPAN_START_MAX: u32 = (1 << (32 - SPAN_LEN_BITS)) - 1;
        let requires_dwf6 = states.len() > (1usize << 17)
            || target_values.iter().any(|&target| target >= (1 << 17))
            || weight_values.iter().any(|&weight| weight >= (1 << 15));
        let compact_tokens = if narrow_token_ranges
            && flat_token_ranges_u16
                .iter()
                .all(|&[start, _]| start < (1 << 13))
        {
            const LENGTH_ESCAPE: u16 = 7;
            // Very range-heavy token sets are materially faster to apply as
            // direct [start,end] pairs, while keeping the common sets compact
            // preserves most of DWF8's size/save advantage.
            const WIDE_RANGE_THRESHOLD: usize = 320;
            let mut ranges = Vec::<u16>::with_capacity(flat_token_ranges_u16.len());
            let mut spans = Vec::<PackedRuntimeTokenSetSpan>::with_capacity(flat_token_spans.len());
            let mut overflow_starts = Vec::<u32>::with_capacity(flat_token_spans.len());
            let mut overflows = Vec::<u16>::new();
            for span in &flat_token_spans {
                let byte_start = ranges.len().saturating_mul(2);
                let wide = span.len as usize > WIDE_RANGE_THRESHOLD;
                overflow_starts.push(
                    u32::try_from(overflows.len())
                        .map_err(|_| "compact runtime overflow stream exceeds u32".to_owned())?,
                );
                let range_start = span.start as usize;
                let range_end = range_start + span.len as usize;
                if wide {
                    for &[start, end] in &flat_token_ranges_u16[range_start..range_end] {
                        ranges.push(start);
                        ranges.push(end);
                    }
                } else {
                    for &[start, end] in &flat_token_ranges_u16[range_start..range_end] {
                        let len = end - start;
                        let encoded_len = len.min(LENGTH_ESCAPE);
                        let packed = start | (encoded_len << 13);
                        ranges.push(packed);
                        if len >= LENGTH_ESCAPE {
                            overflows.push(len);
                        }
                    }
                }
                spans.push(PackedRuntimeTokenSetSpan {
                    start: u32::try_from(byte_start)
                        .map_err(|_| "compact runtime token stream exceeds u32".to_owned())?
                        | u32::from(wide),
                    len: span.len,
                    word_spans: span.word_spans,
                });
            }
            (ranges.len().saturating_mul(2) < (1 << 24) && overflows.len() < (1 << 24))
                .then_some((ranges, spans, overflow_starts, overflows))
        } else {
            None
        };

        let owned_narrow = if requires_dwf6
            && compact_tokens.is_some()
            && flat_token_spans.len() <= u16::MAX as usize + 1
            && flat_token_spans
                .iter()
                .all(|span| span.word_spans <= u16::MAX as u32)
            && weights.len() < u16::MAX as usize
            && geometries.len() <= u16::MAX as usize + 1
            && geometries.iter().all(|geometry| {
                geometry.len() <= u16::MAX as usize
                    && geometry.iter().all(|&(start, end)| {
                        start <= u16::MAX as u32 && end <= u16::MAX as u32
                    })
            })
            && weight_token_ids.iter().all(|&id| id <= u16::MAX as u32)
            && weights.iter().all(|weight| {
                weight.geometry <= u16::MAX as u32
                    && weight.token_ids_start < (1 << 24)
                    && weight.full <= u8::MAX as u32
            })
            && label_values.len() == target_values.len()
            && label_values.len() == weight_values.len()
            && label_values.iter().all(|&label| {
                label == DEFAULT_LABEL_WIRE || (0..u16::MAX as i32).contains(&label)
            })
            && target_values.iter().all(|&target| target < (1 << 24))
            && weight_values.iter().all(|&weight| weight < (1 << 16))
            && label_spans.iter().all(|&[start, len]| {
                start <= SPAN_START_MAX && len <= SPAN_LEN_MAX
            })
            && states.iter().all(|state| {
                state.row <= u16::MAX as u32
                    && (state.final_weight == u32::MAX || state.final_weight < u16::MAX as u32)
            })
        {
            let (
                compact_token_ranges,
                compact_token_spans,
                compact_token_overflow_starts,
                compact_token_overflows,
            ) =
                compact_tokens.expect("checked compact token runtime");
            let mut transitions = Vec::<u8>::with_capacity(label_values.len() * 7);
            for ((&label, &target), &weight_id) in label_values
                .iter()
                .zip(target_values.iter())
                .zip(weight_values.iter())
            {
                let label = if label == DEFAULT_LABEL_WIRE {
                    u16::MAX
                } else {
                    label as u16
                };
                transitions.extend_from_slice(&label.to_le_bytes());
                transitions.push(target as u8);
                transitions.push((target >> 8) as u8);
                transitions.push((target >> 16) as u8);
                transitions.extend_from_slice(&(weight_id as u16).to_le_bytes());
            }

            let spans = label_spans
                .iter()
                .map(|&[start, len]| (start << SPAN_LEN_BITS) | len)
                .collect::<Vec<_>>()
                .into_boxed_slice();

            let mut narrow_states = Vec::<u8>::with_capacity(states.len() * 4);
            for state in &states {
                narrow_states.extend_from_slice(&(state.row as u16).to_le_bytes());
                let final_plus_one = if state.final_weight == u32::MAX {
                    0
                } else {
                    state.final_weight as u16 + 1
                };
                narrow_states.extend_from_slice(&final_plus_one.to_le_bytes());
            }

            let mut narrow_weights = Vec::<u8>::with_capacity(weights.len() * 6);
            for weight in &weights {
                narrow_weights.extend_from_slice(&(weight.geometry as u16).to_le_bytes());
                narrow_weights.push(weight.token_ids_start as u8);
                narrow_weights.push((weight.token_ids_start >> 8) as u8);
                narrow_weights.push((weight.token_ids_start >> 16) as u8);
                narrow_weights.push(weight.full as u8);
            }

            Some(OwnedNarrowPackedRuntimeDwa {
                token_ranges: compact_token_ranges.into_boxed_slice(),
                token_spans: compact_token_spans.into_boxed_slice(),
                token_range_overflow_starts: compact_token_overflow_starts.into_boxed_slice(),
                token_range_overflows: compact_token_overflows.into_boxed_slice(),
                transitions: transitions.into_boxed_slice(),
                spans,
                states: narrow_states.into_boxed_slice(),
                weights: narrow_weights.into_boxed_slice(),
                weight_token_ids: weight_token_ids
                    .iter()
                    .map(|&id| id as u16)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            })
        } else {
            None
        };
        let owned_narrow_present = owned_narrow.is_some();
        let keep_flat_token_ranges = narrow_token_ranges && !owned_narrow_present;

        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][packed_runtime_dwa_from_shared_dwa] row_classes_ms={row_classes_ms:.3} rows_ms={rows_ms:.3} states_ms={states_ms:.3} weight_pool_ms={weight_pool_ms:.3} token_stats_ms={token_stats_ms:.3} total_ms={:.3} token_sets={} token_ranges={} weight_entries={} weights={} geometries={} rows={} row_entries={} states={}",
                total_started.elapsed().as_secs_f64() * 1000.0,
                token_sets.len(),
                token_range_count,
                weight_token_ids.len(),
                weights.len(),
                geometries.len(),
                rows.len(),
                total_row_entries,
                states.len(),
            );
        }

        Ok(Self {
            start_state: dwa.start_state(),
            backed: None,
            owned_narrow,
            token_set_chunks: Box::new([]),
            token_set_locations: Box::new([]),
            flat_token_ranges: None,
            flat_token_ranges_u16: keep_flat_token_ranges
                .then(|| flat_token_ranges_u16.into_boxed_slice()),
            flat_token_spans: keep_flat_token_ranges.then(|| flat_token_spans.into_boxed_slice()),
            materialized_token_sets: (!narrow_token_ranges && !owned_narrow_present)
                .then(|| token_sets.into_boxed_slice()),
            materialized_token_word_spans: (!narrow_token_ranges && !owned_narrow_present)
                .then(|| token_word_spans.into_boxed_slice()),
            fast_wire_token_chunks: None,
            fast_wire_token_chunk_range_counts: None,
            geometries: geometries.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
            weight_token_ids: weight_token_ids.into_boxed_slice(),
            label_pool: PackedRuntimeSeqPool {
                values: label_values.into_boxed_slice(),
                spans: label_spans.into_boxed_slice(),
            },
            target_pool: PackedRuntimeSeqPool {
                values: target_values.into_boxed_slice(),
                spans: target_spans.into_boxed_slice(),
            },
            weight_id_pool: PackedRuntimeSeqPool {
                values: weight_values.into_boxed_slice(),
                spans: weight_spans.into_boxed_slice(),
            },
            rows: rows.into_boxed_slice(),
            states: states.into_boxed_slice(),
        })
    }

    /// Build the read-only runtime representation directly from a finalized
    /// compiler DWA.  Unlike `from_packed_bytes`, this does no wire-oriented
    /// canonical sorting, front coding, varint encoding, or reparsing.  It is
    /// intended to let compiler-created constraints use the same execution
    /// representation as loaded constraints without paying serialization work
    /// merely to construct it.
    pub fn from_dwa(dwa: &DWA) -> Result<Self, String> {
        if dwa.has_shared_transition_rows() {
            return Self::from_shared_dwa_direct(dwa);
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_DWA_RUNTIME_BUILD").is_some();
        let total_started = profile.then(std::time::Instant::now);

        let pooled_started = profile.then(std::time::Instant::now);
        let (token_sets, weight_pool, transition_rows, state_rows, _) =
            dwa.packed_pooled_parts();
        let pooled_ms = pooled_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        const TOKEN_SET_CHUNK: usize = 1024;
        let token_started = profile.then(std::time::Instant::now);
        let build_token_chunk = |sets: &[Arc<RangeSetBlaze<u32>>]| {
            let mut ranges = Vec::<[u32; 2]>::new();
            let mut spans = Vec::<PackedRuntimeTokenSetSpan>::with_capacity(sets.len());
            for tokens in sets {
                let start = u32::try_from(ranges.len())
                    .map_err(|_| "packed runtime token-set chunk exceeds u32 offsets".to_owned())?;
                let mut word_spans = 0u32;
                for range in tokens.ranges() {
                    let lo = *range.start();
                    let hi = *range.end();
                    ranges.push([lo, hi]);
                    word_spans = word_spans.saturating_add(hi / 64 - lo / 64 + 1);
                }
                let len = u32::try_from(ranges.len() - start as usize)
                    .map_err(|_| "packed runtime token set exceeds u32 ranges".to_owned())?;
                spans.push(PackedRuntimeTokenSetSpan {
                    start,
                    len,
                    word_spans,
                });
            }
            Ok::<_, String>(PackedRuntimeTokenSetChunk {
                ranges: ranges.into_boxed_slice(),
                spans: spans.into_boxed_slice(),
            })
        };
        let token_set_chunks = if token_sets.len() >= TOKEN_SET_CHUNK * 2
            && rayon::current_num_threads() > 1
        {
            token_sets
                .par_chunks(TOKEN_SET_CHUNK)
                .map(build_token_chunk)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            token_sets
                .chunks(TOKEN_SET_CHUNK)
                .map(build_token_chunk)
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut token_set_locations = Vec::with_capacity(token_sets.len());
        for (chunk, sets) in token_sets.chunks(TOKEN_SET_CHUNK).enumerate() {
            for local in 0..sets.len() {
                token_set_locations.push(PackedRuntimeTokenSetLocation {
                    chunk: chunk as u32,
                    local: local as u32,
                });
            }
        }
        let token_ms = token_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        // Profile-only diagnostic: isolate RangeSetBlaze range iteration from
        // allocation/copying into the packed slabs. This deliberately runs
        // after the real token build so it cannot inflate token_ms.
        let (range_iter_ms, range_iter_count, range_iter_checksum) = if profile {
            let started = std::time::Instant::now();
            let scan_one = |tokens: &Arc<RangeSetBlaze<u32>>| {
                tokens.ranges().fold((0usize, 0u64), |(count, checksum), range| {
                    (
                        count + 1,
                        checksum
                            .wrapping_add(*range.start() as u64)
                            .wrapping_add((*range.end() as u64).rotate_left(17)),
                    )
                })
            };
            let (count, checksum) = if token_sets.len() >= TOKEN_SET_CHUNK * 2
                && rayon::current_num_threads() > 1
            {
                token_sets
                    .par_iter()
                    .map(scan_one)
                    .reduce(|| (0usize, 0u64), |a, b| (a.0 + b.0, a.1.wrapping_add(b.1)))
            } else {
                token_sets
                    .iter()
                    .map(scan_one)
                    .fold((0usize, 0u64), |a, b| (a.0 + b.0, a.1.wrapping_add(b.1)))
            };
            (started.elapsed().as_secs_f64() * 1000.0, count, checksum)
        } else {
            (0.0, 0usize, 0u64)
        };

        let weights_started = profile.then(std::time::Instant::now);
        let mut geometries = vec![Vec::<(u32, u32)>::new()];
        let mut geometry_ids = FxHashMap::<Vec<(u32, u32)>, u32>::default();
        geometry_ids.insert(Vec::new(), 0);
        let total_weight_entries = weight_pool
            .iter()
            .map(|weight| weight.entries.len())
            .sum::<usize>();
        let mut weights = Vec::<PackedRuntimeWeight>::with_capacity(weight_pool.len());
        let mut weight_token_ids = Vec::<u32>::with_capacity(total_weight_entries);
        for weight in &weight_pool {
            if weight.all {
                weights.push(PackedRuntimeWeight {
                    geometry: 0,
                    token_ids_start: weight_token_ids.len() as u32,
                    full: 1,
                });
                continue;
            }
            let geometry = weight
                .entries
                .iter()
                .map(|&(start, end, _)| (start, end))
                .collect::<Vec<_>>();
            let geometry_id = if let Some(&id) = geometry_ids.get(&geometry) {
                id
            } else {
                let id = u32::try_from(geometries.len())
                    .map_err(|_| "packed runtime DWA has too many weight geometries".to_owned())?;
                geometry_ids.insert(geometry.clone(), id);
                geometries.push(geometry);
                id
            };
            let token_ids_start = u32::try_from(weight_token_ids.len())
                .map_err(|_| "packed runtime DWA weight token ids exceed u32 offsets".to_owned())?;
            for &(_, _, token_set) in &weight.entries {
                if token_set as usize >= token_sets.len() {
                    return Err("invalid direct packed-runtime token-set id".to_owned());
                }
                weight_token_ids.push(token_set);
            }
            weights.push(PackedRuntimeWeight {
                geometry: geometry_id,
                token_ids_start,
                full: 0,
            });
        }
        let weights_ms = weights_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let rows_started = profile.then(std::time::Instant::now);
        let total_row_entries = transition_rows.iter().map(Vec::len).sum::<usize>();
        let mut label_values = Vec::<Label>::with_capacity(total_row_entries);
        let mut target_values = Vec::<u32>::with_capacity(total_row_entries);
        let mut weight_values = Vec::<u32>::with_capacity(total_row_entries);
        let mut label_spans = Vec::<[u32; 2]>::with_capacity(transition_rows.len());
        let mut target_spans = Vec::<[u32; 2]>::with_capacity(transition_rows.len());
        let mut weight_spans = Vec::<[u32; 2]>::with_capacity(transition_rows.len());
        let mut rows = Vec::<PackedRuntimeRow>::with_capacity(transition_rows.len());
        for (row_id, row) in transition_rows.iter().enumerate() {
            let start = u32::try_from(label_values.len())
                .map_err(|_| "packed runtime DWA row pool exceeds u32 offsets".to_owned())?;
            let len = u32::try_from(row.len())
                .map_err(|_| "packed runtime DWA row exceeds u32 entries".to_owned())?;
            for &(label, target, weight) in row {
                if weight as usize >= weights.len() {
                    return Err("invalid direct packed-runtime Weight id".to_owned());
                }
                label_values.push(label);
                target_values.push(target);
                weight_values.push(weight);
            }
            label_spans.push([start, len]);
            target_spans.push([start, len]);
            weight_spans.push([start, len]);
            let id = u32::try_from(row_id)
                .map_err(|_| "packed runtime DWA has too many rows".to_owned())?;
            rows.push(PackedRuntimeRow {
                labels: id,
                targets: id,
                weights: id,
            });
        }
        let rows_ms = rows_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let states_started = profile.then(std::time::Instant::now);
        let mut states = Vec::<PackedRuntimeState>::with_capacity(state_rows.len());
        for &(row, final_weight) in &state_rows {
            if row as usize >= rows.len() {
                return Err("invalid direct packed-runtime row id".to_owned());
            }
            let final_weight = final_weight.unwrap_or(u32::MAX);
            if final_weight != u32::MAX && final_weight as usize >= weights.len() {
                return Err("invalid direct packed-runtime final Weight id".to_owned());
            }
            states.push(PackedRuntimeState { row, final_weight });
        }
        let states_ms = states_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][packed_runtime_dwa_from_dwa] pooled_ms={pooled_ms:.3} token_ms={token_ms:.3} weights_ms={weights_ms:.3} rows_ms={rows_ms:.3} states_ms={states_ms:.3} total_ms={:.3} token_sets={} token_ranges={} weights={} geometries={} rows={} row_entries={} states={}",
                total_started.elapsed().as_secs_f64() * 1000.0,
                token_sets.len(),
                token_set_chunks.iter().map(|chunk| chunk.ranges.len()).sum::<usize>(),
                weights.len(),
                geometries.len(),
                rows.len(),
                total_row_entries,
                states.len(),
            );
            eprintln!(
                "[glrmask/profile][packed_runtime_dwa_range_iter] ms={range_iter_ms:.3} ranges={range_iter_count} checksum={range_iter_checksum}",
            );
        }

        Ok(Self {
            start_state: dwa.start_state(),
            backed: None,
            owned_narrow: None,
            token_set_chunks: token_set_chunks.into_boxed_slice(),
            token_set_locations: token_set_locations.into_boxed_slice(),
            flat_token_ranges: None,
            flat_token_ranges_u16: None,
            flat_token_spans: None,
            materialized_token_sets: None,
            materialized_token_word_spans: None,
            fast_wire_token_chunks: None,
            fast_wire_token_chunk_range_counts: None,
            geometries: geometries.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
            weight_token_ids: weight_token_ids.into_boxed_slice(),
            label_pool: PackedRuntimeSeqPool {
                values: label_values.into_boxed_slice(),
                spans: label_spans.into_boxed_slice(),
            },
            target_pool: PackedRuntimeSeqPool {
                values: target_values.into_boxed_slice(),
                spans: target_spans.into_boxed_slice(),
            },
            weight_id_pool: PackedRuntimeSeqPool {
                values: weight_values.into_boxed_slice(),
                spans: weight_spans.into_boxed_slice(),
            },
            rows: rows.into_boxed_slice(),
            states: states.into_boxed_slice(),
        })
    }

    pub fn from_packed_bytes(input: &[u8]) -> Result<Self, String> {
        let profile_runtime = std::env::var_os("GLRMASK_PROFILE_DWA_RUNTIME").is_some();
        let total_started = profile_runtime.then(std::time::Instant::now);
        let scan_started = profile_runtime.then(std::time::Instant::now);
        if !input.starts_with(b"DWP6") {
            return Err("invalid packed runtime DWA header".to_owned());
        }
        let mut pos = 4usize;
        let start_state = take_var_u32(input, &mut pos)?;

        let token_set_count = take_var_u32(input, &mut pos)? as usize;
        let token_set_chunk_count = take_var_u32(input, &mut pos)? as usize;
        let token_set_bodies = take_length_prefixed_slices(
            input,
            &mut pos,
            token_set_chunk_count,
            "token-set chunk",
        )?;

        let geometry_count = take_var_u32(input, &mut pos)? as usize;
        let geometry_bodies = take_length_prefixed_slices(
            input,
            &mut pos,
            geometry_count,
            "weight geometry",
        )?;

        let weight_count = take_var_u32(input, &mut pos)? as usize;
        let weight_bodies =
            take_length_prefixed_slices(input, &mut pos, weight_count, "weight")?;

        enum RowSource<'a> {
            Direct(Vec<&'a [u8]>),
            Components {
                labels: Vec<&'a [u8]>,
                targets: Vec<&'a [u8]>,
                weights: Vec<&'a [u8]>,
                rows: Vec<(u32, u32, u32)>,
            },
        }

        let row_mode = *input
            .get(pos)
            .ok_or_else(|| "truncated packed DWA row mode".to_owned())?;
        pos += 1;
        let row_source = match row_mode {
            0 => {
                let row_count = take_var_u32(input, &mut pos)? as usize;
                RowSource::Direct(take_length_prefixed_slices(
                    input,
                    &mut pos,
                    row_count,
                    "transition row",
                )?)
            }
            1 => {
                let label_count = take_var_u32(input, &mut pos)? as usize;
                let labels = take_length_prefixed_slices(
                    input,
                    &mut pos,
                    label_count,
                    "label sequence",
                )?;
                let target_count = take_var_u32(input, &mut pos)? as usize;
                let targets = take_length_prefixed_slices(
                    input,
                    &mut pos,
                    target_count,
                    "target sequence",
                )?;
                let weight_id_count = take_var_u32(input, &mut pos)? as usize;
                let weights = take_length_prefixed_slices(
                    input,
                    &mut pos,
                    weight_id_count,
                    "weight-id sequence",
                )?;
                let row_count = take_var_u32(input, &mut pos)? as usize;
                let mut rows = Vec::with_capacity(row_count);
                for _ in 0..row_count {
                    rows.push((
                        take_var_u32(input, &mut pos)?,
                        take_var_u32(input, &mut pos)?,
                        take_var_u32(input, &mut pos)?,
                    ));
                }
                RowSource::Components {
                    labels,
                    targets,
                    weights,
                    rows,
                }
            }
            _ => return Err("invalid packed DWA row mode".to_owned()),
        };

        let state_count = take_var_u32(input, &mut pos)? as usize;
        if start_state as usize >= state_count && state_count != 0 {
            return Err("invalid packed DWA start state".to_owned());
        }
        let row_count = match &row_source {
            RowSource::Direct(rows) => rows.len(),
            RowSource::Components { rows, .. } => rows.len(),
        };
        let mut states = Vec::with_capacity(state_count);
        for _ in 0..state_count {
            let row = take_var_u32(input, &mut pos)?;
            if row as usize >= row_count {
                return Err("invalid packed DWA transition-row index".to_owned());
            }
            let final_encoded = take_var_u32(input, &mut pos)?;
            let final_weight = if final_encoded == 0 {
                u32::MAX
            } else {
                let weight = final_encoded - 1;
                if weight as usize >= weight_count {
                    return Err("invalid packed DWA final-weight index".to_owned());
                }
                weight
            };
            states.push(PackedRuntimeState { row, final_weight });
        }
        if pos != input.len() {
            return Err("trailing bytes in packed runtime DWA".to_owned());
        }
        let scan_ms = scan_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);

        let decode_token_sets = || -> Result<Vec<PackedRuntimeTokenSetChunk>, String> {
            let started = profile_runtime.then(std::time::Instant::now);
            if token_set_chunk_count >= 16 && rayon::current_num_threads() > 1 {
                let result = token_set_bodies
                    .par_iter()
                    .map(|body| decode_packed_runtime_token_set_chunk(body))
                    .collect();
                if let Some(started) = started {
                    eprintln!("[glrmask/profile][packed_runtime_dwa] token_sets_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                }
                result
            } else {
                let result = token_set_bodies
                    .iter()
                    .map(|body| decode_packed_runtime_token_set_chunk(body))
                    .collect();
                if let Some(started) = started {
                    eprintln!("[glrmask/profile][packed_runtime_dwa] token_sets_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                }
                result
            }
        };

        let decode_other = || -> Result<_, String> {
            let decode_geometry_weights = || -> Result<_, String> {
                let started = profile_runtime.then(std::time::Instant::now);
                let geometries: Vec<Vec<(u32, u32)>> =
                    if geometry_count >= 1024 && rayon::current_num_threads() > 1 {
                        geometry_bodies
                            .par_iter()
                            .map(|body| decode_packed_weight_geometry(body))
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        geometry_bodies
                            .iter()
                            .map(|body| decode_packed_weight_geometry(body))
                            .collect::<Result<Vec<_>, _>>()?
                    };
                let (weights, weight_token_ids) = decode_packed_runtime_weights_flat(
                    &weight_bodies,
                    &geometries,
                    token_set_count,
                )?;
                if let Some(started) = started {
                    eprintln!("[glrmask/profile][packed_runtime_dwa] geometry_weights_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                }
                Ok((geometries, weights, weight_token_ids))
            };

            let decode_rows = || -> Result<_, String> {
                let started = profile_runtime.then(std::time::Instant::now);
                match row_source {
                    RowSource::Direct(ref bodies) => {
                        let result = decode_packed_runtime_direct_row_pools(bodies, weight_count);
                        if let Some(started) = started {
                            eprintln!("[glrmask/profile][packed_runtime_dwa] rows_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                        }
                        result
                    }
                    RowSource::Components {
                        ref labels,
                        ref targets,
                        ref weights,
                        ref rows,
                    } => {
                        let decode_labels = || decode_packed_runtime_label_pool(labels);
                        let decode_targets = || decode_packed_runtime_u32_pool(targets, "target");
                        let decode_weights = || decode_packed_runtime_u32_pool(weights, "weight-id");
                        let (label_pool, (target_pool, weight_pool)) = rayon::join(
                            decode_labels,
                            || rayon::join(decode_targets, decode_weights),
                        );
                        let label_pool = label_pool?;
                        let target_pool = target_pool?;
                        let weight_pool = weight_pool?;
                        let mut packed_rows = Vec::with_capacity(rows.len());
                        for &(labels, targets, weights) in rows {
                            if labels as usize >= label_pool.len()
                                || targets as usize >= target_pool.len()
                                || weights as usize >= weight_pool.len()
                            {
                                return Err("invalid packed DWA row-component index".to_owned());
                            }
                            let len = label_pool
                                .get(labels)
                                .expect("validated packed DWA label pool id")
                                .len();
                            if target_pool
                                .get(targets)
                                .expect("validated packed DWA target pool id")
                                .len()
                                != len
                                || weight_pool
                                    .get(weights)
                                    .expect("validated packed DWA weight pool id")
                                    .len()
                                    != len
                            {
                                return Err("mismatched packed DWA row-component lengths".to_owned());
                            }
                            if weight_pool
                                .get(weights)
                                .expect("validated packed DWA weight pool id")
                                .iter()
                                .any(|&weight| weight as usize >= weight_count)
                            {
                                return Err("invalid packed DWA weight index".to_owned());
                            }
                            packed_rows.push(PackedRuntimeRow {
                                labels,
                                targets,
                                weights,
                            });
                        }
                        let result = Ok((label_pool, target_pool, weight_pool, packed_rows));
                        if let Some(started) = started {
                            eprintln!("[glrmask/profile][packed_runtime_dwa] rows_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                        }
                        result
                    }
                }
            };

            let ((geometries, weights, weight_token_ids), (labels, targets, weight_ids, rows)) =
                if rayon::current_num_threads() > 1 {
                    let (left, right) = rayon::join(decode_geometry_weights, decode_rows);
                    (left?, right?)
                } else {
                    (decode_geometry_weights()?, decode_rows()?)
                };
            Ok((
                geometries,
                weights,
                weight_token_ids,
                labels,
                targets,
                weight_ids,
                rows,
            ))
        };

        let (token_set_chunks, other) = if rayon::current_num_threads() > 1 {
            let (token_sets, other) = rayon::join(decode_token_sets, decode_other);
            (token_sets?, other?)
        } else {
            (decode_token_sets()?, decode_other()?)
        };
        let (geometries, weights, weight_token_ids, labels, targets, weight_ids, rows) = other;

        let mut token_set_locations = Vec::with_capacity(token_set_count);
        for (chunk_index, chunk) in token_set_chunks.iter().enumerate() {
            for local in 0..chunk.spans.len() {
                token_set_locations.push(PackedRuntimeTokenSetLocation {
                    chunk: chunk_index as u32,
                    local: local as u32,
                });
            }
        }
        if token_set_locations.len() != token_set_count {
            return Err("invalid packed DWA token-set count".to_owned());
        }
        for index in 0..targets.len() {
            if targets
                .get(index as u32)
                .expect("packed runtime target pool span exists")
                .iter()
                .any(|&target| target as usize >= state_count)
            {
                return Err("invalid packed DWA target state".to_owned());
            }
        }

        let result = Self {
            start_state,
            backed: None,
            owned_narrow: None,
            token_set_chunks: token_set_chunks.into_boxed_slice(),
            token_set_locations: token_set_locations.into_boxed_slice(),
            flat_token_ranges: None,
            flat_token_ranges_u16: None,
            flat_token_spans: None,
            materialized_token_sets: None,
            materialized_token_word_spans: None,
            fast_wire_token_chunks: None,
            fast_wire_token_chunk_range_counts: None,
            geometries: geometries.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
            weight_token_ids: weight_token_ids.into_boxed_slice(),
            label_pool: labels,
            target_pool: targets,
            weight_id_pool: weight_ids,
            rows: rows.into_boxed_slice(),
            states: states.into_boxed_slice(),
        };
        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][packed_runtime_dwa] scan_ms={scan_ms:.3} total_ms={:.3}",
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(result)
    }

    #[inline]
    pub fn start_state(&self) -> u32 {
        self.start_state
    }

    #[inline]
    pub fn state_count(&self) -> usize {
        if let Some(narrow) = &self.owned_narrow {
            return narrow.states.len() / OwnedNarrowPackedRuntimeDwa::STATE_STRIDE;
        }
        self.backed
            .as_ref()
            .map_or(self.states.len(), |backed| backed.state_count)
    }

    #[inline]
    fn row_count(&self) -> usize {
        if let Some(narrow) = &self.owned_narrow {
            return narrow.spans.len();
        }
        self.backed
            .as_ref()
            .map_or(self.rows.len(), |backed| backed.row_count)
    }

    #[inline]
    pub fn token_set_count(&self) -> usize {
        if let Some(narrow) = &self.owned_narrow {
            return narrow.token_spans.len();
        }
        if let Some(backed) = &self.backed {
            return backed.token_set_count;
        }
        self.materialized_token_sets.as_ref().map_or_else(
            || {
                self.flat_token_spans
                    .as_ref()
                    .map_or(self.token_set_locations.len(), |spans| spans.len())
            },
            |sets| sets.len(),
        )
    }

    /// Return the packed token-set ids referenced by transition weights.
    ///
    /// This scans only compact row/weight metadata; it does not decode token
    /// ranges or materialize Weight/RangeSet values. Runtime cache builders can
    /// therefore recover the same transition-only token-set domain used by the
    /// compiler without defeating packed-DWA loading.
    pub fn transition_token_set_ids(&self) -> Vec<u32> {
        let mut transition_weights = vec![false; self.weight_count()];
        for row_id in 0..self.row_count() {
            let Some(row_id) = u32::try_from(row_id).ok() else {
                break;
            };
            let Some(row) = self.row_record(row_id) else {
                continue;
            };
            let Some([start, len]) = self.pool_span_at(2, row.weights) else {
                continue;
            };
            for index in 0..len as usize {
                let Some(weight_id) = self.weight_value_at(start as usize + index) else {
                    continue;
                };
                if let Some(used) = transition_weights.get_mut(weight_id as usize) {
                    *used = true;
                }
            }
        }

        let mut transition_tokens = vec![false; self.token_set_count()];
        for (weight_id, used) in transition_weights.into_iter().enumerate() {
            if !used {
                continue;
            }
            let Some(weight_id) = u32::try_from(weight_id).ok() else {
                break;
            };
            let Some(weight) = self.weight_record(weight_id) else {
                continue;
            };
            if weight.full != 0 {
                continue;
            }
            let Some(len) = self.geometry_len_at(weight.geometry) else {
                continue;
            };
            for index in 0..len {
                let Some(token_id) = self.weight_token_id_at(weight.token_ids_start as usize + index)
                else {
                    continue;
                };
                if let Some(used) = transition_tokens.get_mut(token_id as usize) {
                    *used = true;
                }
            }
        }

        transition_tokens
            .into_iter()
            .enumerate()
            .filter_map(|(id, used)| used.then(|| u32::try_from(id).ok()).flatten())
            .collect()
    }
    #[inline]
    pub fn materialized_token_sets_with_word_spans(
        &self,
    ) -> Option<(&[Arc<RangeSetBlaze<u32>>], &[u32])> {
        Some((
            self.materialized_token_sets.as_deref()?,
            self.materialized_token_word_spans.as_deref()?,
        ))
    }

    #[inline]
    pub fn weight_count(&self) -> usize {
        if let Some(narrow) = &self.owned_narrow {
            return narrow.weights.len() / OwnedNarrowPackedRuntimeDwa::WEIGHT_STRIDE;
        }
        self.backed
            .as_ref()
            .map_or(self.weights.len(), |backed| backed.weight_count)
    }

    #[inline]
    pub fn token_set(&self, id: u32) -> Option<PackedRuntimeTokenSetRef<'_>> {
        if let Some(backed) = &self.backed {
            if backed.direct_token_spans {
                let index = id as usize;
                if index >= backed.token_set_count {
                    return None;
                }
                let span = backed.token_locations_start + index * 12;
                let start = backed.read_u32(span)? as usize;
                let len = backed.read_u32(span + 4)? as usize;
                let word_spans = backed.read_u32(span + 8)?;
                let byte_start = start.checked_mul(4)?;
                let byte_end = start.checked_add(len)?.checked_mul(4)?;
                let body_len = backed.token_body_end.checked_sub(backed.token_body_start)?;
                if byte_end > body_len {
                    return None;
                }
                let absolute_start = backed.section_start + backed.token_body_start + byte_start;
                let absolute_end = backed.section_start + backed.token_body_start + byte_end;
                return Some(PackedRuntimeTokenSetRef {
                    id,
                    storage: PackedRuntimeTokenSetStorageRef::BackedFlat16(
                        backed.backing.get(absolute_start..absolute_end)?,
                    ),
                    word_spans,
                });
            }
            if backed.split_target_weight {
                let index = id as usize;
                if index >= backed.token_set_count {
                    return None;
                }
                let start_location =
                    backed.read_u24(backed.token_locations_start + index * 3)? as usize;
                let wide_compact_set = backed.compact_token_ranges && (start_location & 1) != 0;
                let start = if backed.compact_token_ranges {
                    start_location & !1usize
                } else {
                    start_location
                };
                let body_len = backed.token_body_end.checked_sub(backed.token_body_start)?;
                let end = if index + 1 < backed.token_set_count {
                    let next = backed
                        .read_u24(backed.token_locations_start + (index + 1) * 3)?
                        as usize;
                    if backed.compact_token_ranges {
                        next & !1usize
                    } else {
                        next
                    }
                } else {
                    body_len
                };
                if start > end || end > body_len {
                    return None;
                }
                let absolute_start = backed.section_start + backed.token_body_start + start;
                let absolute_end = backed.section_start + backed.token_body_start + end;
                if backed.compact_token_ranges {
                    let word_spans =
                        backed.read_u16(backed.token_word_spans_start + index * 2)? as u32;
                    if wide_compact_set {
                        if (end - start) % 4 != 0 {
                            return None;
                        }
                        return Some(PackedRuntimeTokenSetRef {
                            id,
                            storage: PackedRuntimeTokenSetStorageRef::BackedFlat16(
                                backed.backing.get(absolute_start..absolute_end)?,
                            ),
                            word_spans,
                        });
                    }
                    if start % 2 != 0 || end % 2 != 0 {
                        return None;
                    }
                    let range_count = ((end - start) / 2) as u32;
                    let overflow_start = backed
                        .read_u24(backed.compact_overflow_locations_start + index * 3)?
                        as usize;
                    let overflow_end = if index + 1 < backed.token_set_count {
                        backed.read_u24(
                            backed.compact_overflow_locations_start + (index + 1) * 3,
                        )? as usize
                    } else {
                        backed.compact_overflow_count
                    };
                    if overflow_start > overflow_end || overflow_end > backed.compact_overflow_count {
                        return None;
                    }
                    let overflow_absolute_start = backed.section_start
                        + backed.compact_overflow_values_start
                        + overflow_start * 2;
                    let overflow_absolute_end = backed.section_start
                        + backed.compact_overflow_values_start
                        + overflow_end * 2;
                    return Some(PackedRuntimeTokenSetRef {
                        id,
                        storage: PackedRuntimeTokenSetStorageRef::Compact {
                            bytes: backed.backing.get(absolute_start..absolute_end)?,
                            range_count,
                            overflows: backed
                                .backing
                                .get(overflow_absolute_start..overflow_absolute_end)?,
                        },
                        word_spans,
                    });
                }
                if (end - start) % 4 != 0 {
                    return None;
                }
                let word_spans = backed.read_u16(backed.token_word_spans_start + index * 2)? as u32;
                return Some(PackedRuntimeTokenSetRef {
                    id,
                    storage: PackedRuntimeTokenSetStorageRef::BackedFlat16(
                        backed.backing.get(absolute_start..absolute_end)?,
                    ),
                    word_spans,
                });
            }
            let (bytes, word_spans) = backed.token_bytes(id)?;
            let mut pos = 0usize;
            let range_count = take_var_u32(bytes, &mut pos).ok()?;
            return Some(PackedRuntimeTokenSetRef {
                id,
                storage: PackedRuntimeTokenSetStorageRef::Varint {
                    bytes: bytes.get(pos..)?,
                    range_count,
                },
                word_spans,
            });
        }
        if let Some(narrow) = &self.owned_narrow {
            let span = *narrow.token_spans.get(id as usize)?;
            let wide = span.start & 1 != 0;
            let start_bytes = (span.start & !1) as usize;
            let end_bytes = if id as usize + 1 < narrow.token_spans.len() {
                (narrow.token_spans[id as usize + 1].start & !1) as usize
            } else {
                narrow.token_ranges.len().checked_mul(2)?
            };
            if start_bytes > end_bytes
                || end_bytes > narrow.token_ranges.len().checked_mul(2)?
                || start_bytes % 2 != 0
                || end_bytes % 2 != 0
            {
                return None;
            }
            let start = start_bytes / 2;
            let end = end_bytes / 2;
            if wide {
                if (end - start) % 2 != 0 || (end - start) / 2 != span.len as usize {
                    return None;
                }
                let words = narrow.token_ranges.get(start..end)?;
                // [u16; 2] has the same alignment as u16 and every u16 bit
                // pattern is valid, so an even-length u16 slice can be viewed
                // as start/end pairs without copying.
                let ranges = unsafe {
                    std::slice::from_raw_parts(
                        words.as_ptr().cast::<[u16; 2]>(),
                        words.len() / 2,
                    )
                };
                return Some(PackedRuntimeTokenSetRef {
                    id,
                    storage: PackedRuntimeTokenSetStorageRef::Flat16(ranges),
                    word_spans: span.word_spans,
                });
            }
            if end - start != span.len as usize {
                return None;
            }
            let overflow_start = *narrow.token_range_overflow_starts.get(id as usize)? as usize;
            let overflow_end = if id as usize + 1 < narrow.token_range_overflow_starts.len() {
                narrow.token_range_overflow_starts[id as usize + 1] as usize
            } else {
                narrow.token_range_overflows.len()
            };
            if overflow_start > overflow_end || overflow_end > narrow.token_range_overflows.len() {
                return None;
            }
            return Some(PackedRuntimeTokenSetRef {
                id,
                storage: PackedRuntimeTokenSetStorageRef::Compact16 {
                    ranges: narrow.token_ranges.get(start..end)?,
                    overflows: narrow.token_range_overflows.get(overflow_start..overflow_end)?,
                },
                word_spans: span.word_spans,
            });
        }
        if let Some(sets) = &self.materialized_token_sets {
            let tokens = sets.get(id as usize)?;
            let word_spans = *self
                .materialized_token_word_spans
                .as_ref()?
                .get(id as usize)?;
            return Some(PackedRuntimeTokenSetRef {
                id,
                storage: PackedRuntimeTokenSetStorageRef::Materialized(tokens),
                word_spans,
            });
        }
        if let (Some(ranges), Some(spans)) = (&self.flat_token_ranges_u16, &self.flat_token_spans) {
            let span = *spans.get(id as usize)?;
            let start = span.start as usize;
            let end = start.checked_add(span.len as usize)?;
            return Some(PackedRuntimeTokenSetRef {
                id,
                storage: PackedRuntimeTokenSetStorageRef::Flat16(ranges.get(start..end)?),
                word_spans: span.word_spans,
            });
        }
        if let (Some(ranges), Some(spans)) = (&self.flat_token_ranges, &self.flat_token_spans) {
            let span = *spans.get(id as usize)?;
            let start = span.start as usize;
            let end = start.checked_add(span.len as usize)?;
            return Some(PackedRuntimeTokenSetRef {
                id,
                storage: PackedRuntimeTokenSetStorageRef::Flat(ranges.get(start..end)?),
                word_spans: span.word_spans,
            });
        }
        let location = *self.token_set_locations.get(id as usize)?;
        let chunk = self.token_set_chunks.get(location.chunk as usize)?;
        let span = *chunk.spans.get(location.local as usize)?;
        let start = span.start as usize;
        let end = start + span.len as usize;
        Some(PackedRuntimeTokenSetRef {
            id,
            storage: PackedRuntimeTokenSetStorageRef::Flat(&chunk.ranges[start..end]),
            word_spans: span.word_spans,
        })
    }

    #[inline]
    pub fn weight(&self, id: u32) -> Option<PackedRuntimeWeightRef<'_>> {
        self.weight_record(id)
            .map(|_| PackedRuntimeWeightRef { dwa: self, id })
    }

    pub fn transition(&self, state: u32, label: Label) -> Option<(u32, PackedRuntimeWeightRef<'_>)> {
        let state = self.state_record(state)?;
        let row = self.row_record(state.row)?;
        let index = self.find_label_in_row(row, label)?;
        let [target_start, target_len] = self.pool_span_at(1, row.targets)?;
        let [weight_start, weight_len] = self.pool_span_at(2, row.weights)?;
        if index >= target_len as usize || index >= weight_len as usize {
            return None;
        }
        if target_start == weight_start
            && let Some(narrow) = &self.owned_narrow
        {
            let flat_index = target_start as usize + index;
            let target = narrow.target(flat_index)?;
            let weight = narrow.weight_id(flat_index)?;
            return Some((target, self.weight(weight)?));
        }
        if target_start == weight_start
            && let Some(backed) = &self.backed
            && backed.narrow
        {
            let flat_index = target_start as usize + index;
            if flat_index >= backed.target_value_count || flat_index >= backed.weight_value_count {
                return None;
            }
            if backed.split_target_weight {
                let stride = if backed.label_dedup { 5 } else { 7 };
                let pos = backed.target_values_start
                    + flat_index * stride
                    + if backed.label_dedup { 0 } else { 2 };
                let target = backed.read_u24(pos)?;
                let weight = backed.read_u16(pos + 3)? as u32;
                return Some((target, self.weight(weight)?));
            }
            let pos = if backed.label_dedup {
                backed.target_values_start + flat_index * 4
            } else {
                backed.target_values_start + flat_index * 6 + 2
            };
            let packed = backed.read_u32(pos)?;
            let target = packed & 0x1ffff;
            let weight = packed >> 17;
            return Some((target, self.weight(weight)?));
        }
        let target = self.target_value_at(target_start as usize + index)?;
        let weight = self.weight_value_at(weight_start as usize + index)?;
        Some((target, self.weight(weight)?))
    }

    pub fn final_weight(&self, state: u32) -> Option<PackedRuntimeWeightRef<'_>> {
        let state = self.state_record(state)?;
        (state.final_weight != u32::MAX)
            .then(|| self.weight(state.final_weight))
            .flatten()
    }

    /// Materialize the read-only packed runtime representation back into the
    /// ordinary compiler DWA representation. Runtime loads intentionally keep
    /// the packed form to avoid rebuilding allocation-heavy Weight/transition
    /// structures; compiler operations such as constraint composition can opt
    /// into this conversion only when they actually need to transform the DWA.
    pub fn to_dwa(&self) -> Result<DWA, String> {
        let mut weights = Vec::<Weight>::with_capacity(self.weight_count());
        for id in 0..self.weight_count() as u32 {
            let packed = self
                .weight(id)
                .ok_or_else(|| format!("packed parser DWA is missing weight {id}"))?;
            if packed.is_full() {
                weights.push(Weight::all());
                continue;
            }
            let mut runs = Vec::new();
            for ((start, end), token_set) in packed.entries() {
                let tokens = if let Some(tokens) = token_set.materialized_arc() {
                    Arc::clone(tokens)
                } else {
                    let mut ranges = Vec::with_capacity(token_set.range_count());
                    token_set.for_each_range(|lo, hi| ranges.push(lo..=hi));
                    shared_rangeset(RangeSetBlaze::from_iter(ranges))
                };
                runs.push((start, end, tokens));
            }
            weights.push(Weight::from_tsid_runs_shared(runs));
        }

        let mut states = Vec::with_capacity(self.state_count());
        for state_id in 0..self.state_count() as u32 {
            let packed_state = self
                .state_record(state_id)
                .ok_or_else(|| format!("packed parser DWA is missing state {state_id}"))?;
            let row = self
                .row_record(packed_state.row)
                .ok_or_else(|| format!("packed parser DWA state {state_id} has an invalid row"))?;
            let [label_start, label_len] = self
                .pool_span_at(0, row.labels)
                .ok_or_else(|| format!("packed parser DWA state {state_id} has invalid labels"))?;
            let [target_start, target_len] = self
                .pool_span_at(1, row.targets)
                .ok_or_else(|| format!("packed parser DWA state {state_id} has invalid targets"))?;
            let [weight_start, weight_len] = self
                .pool_span_at(2, row.weights)
                .ok_or_else(|| format!("packed parser DWA state {state_id} has invalid weights"))?;
            if label_len != target_len || label_len != weight_len {
                return Err(format!(
                    "packed parser DWA state {state_id} has mismatched row lengths"
                ));
            }

            let mut transitions = BTreeMap::new();
            for index in 0..label_len as usize {
                let label = self
                    .label_value_at(label_start as usize + index)
                    .ok_or_else(|| format!("packed parser DWA state {state_id} has invalid label"))?;
                let target = self
                    .target_value_at(target_start as usize + index)
                    .ok_or_else(|| format!("packed parser DWA state {state_id} has invalid target"))?;
                let weight_id = self
                    .weight_value_at(weight_start as usize + index)
                    .ok_or_else(|| format!("packed parser DWA state {state_id} has invalid weight id"))?;
                let weight = weights
                    .get(weight_id as usize)
                    .cloned()
                    .ok_or_else(|| format!("packed parser DWA references missing weight {weight_id}"))?;
                transitions.insert(label, (target, weight));
            }
            let final_weight = if packed_state.final_weight == u32::MAX {
                None
            } else {
                Some(
                    weights
                        .get(packed_state.final_weight as usize)
                        .cloned()
                        .ok_or_else(|| {
                            format!(
                                "packed parser DWA state {state_id} references missing final weight {}",
                                packed_state.final_weight
                            )
                        })?,
                )
            };
            states.push(DWAState {
                transitions: transitions.into(),
                final_weight,
            });
        }
        Ok(DWA::from_parts(states, self.start_state))
    }

    #[inline]
    pub fn row_is_empty(&self, state: u32) -> bool {
        let Some(state) = self.state_record(state) else {
            return true;
        };
        let Some(row) = self.row_record(state.row) else {
            return true;
        };
        self.pool_span_at(0, row.labels)
            .is_none_or(|span| span[1] == 0)
    }
}

// --- Two-level weight-pool serde for DWA ---
// Level 1: Pool structurally equal RangeSetBlaze<u32> token sets.
// Level 2: Pool structurally equal Weight values, referencing token-set indices.
//
// Both levels retain a pointer cache as the hot path, but pointer identity is
// deliberately not the serialization identity.  Equivalent DWAs can be built
// with different Arc-sharing layouts (especially when independent support
// operations are evaluated in parallel); the artifact bytes must not depend
// on allocator/interner timing.

/// Serialized token set: Vec of [start, end] range pairs
type EncodedTokenSet = Vec<[u32; 2]>;

/// A single entry in a pooled weight: (tsid_start, tsid_end, token_set_pool_index)
#[derive(Serialize, Deserialize)]
struct WeightPoolEntry {
    all: bool,
    /// Entries: (tsid_range_start, tsid_range_end, token_set_pool_index)
    entries: Vec<(u32, u32, u32)>,
}

#[derive(Serialize, Deserialize)]
struct DWAStateSerde {
    /// transitions: (label, target_state, weight_pool_index)
    transitions: Vec<(Label, u32, u32)>,
    /// final_weight: Some(weight_pool_index) or None
    final_weight: Option<u32>,
}

#[derive(Serialize, Deserialize)]
struct DWASerde {
    /// Pool of unique token sets (level 1)
    token_set_pool: Vec<EncodedTokenSet>,
    /// Pool of unique weights referencing token_set_pool indices (level 2)
    weight_pool: Vec<WeightPoolEntry>,
    states: Vec<DWAStateSerde>,
    start_state: u32,
}

thread_local! {
    /// Constraint artifact v13+ uses a packed, uncompressed DWA wire format.
    /// Older artifact versions leave this disabled and continue to decode the
    /// historical serde representation byte-for-byte.
    static PACKED_DWA_SERDE: Cell<bool> = const { Cell::new(false) };
    /// v14+ sectioned constraint artifacts serialize parser_dwa in its own
    /// section. The Constraint core therefore carries only a one-byte DWA
    /// placeholder, allowing DWA and core decoding to proceed independently.
    static EXTERNAL_DWA_SERDE: Cell<bool> = const { Cell::new(false) };
    static PACKED_DWA_TOKEN_SET_INVENTORY: RefCell<Option<PackedDwaTokenSetInventory>> =
        const { RefCell::new(None) };
}

/// Enable or disable the packed DWA wire format on this thread, returning the
/// previous setting. Constraint serialization uses this as a version-scoped
/// switch so old artifact formats remain readable.
pub fn set_packed_serde(enabled: bool) -> bool {
    PACKED_DWA_SERDE.with(|mode| mode.replace(enabled))
}

pub fn set_external_serde(enabled: bool) -> bool {
    EXTERNAL_DWA_SERDE.with(|mode| mode.replace(enabled))
}

fn packed_serde_enabled() -> bool {
    PACKED_DWA_SERDE.with(Cell::get)
}


fn external_serde_enabled() -> bool {
    EXTERNAL_DWA_SERDE.with(Cell::get)
}

pub fn take_packed_decode_token_set_inventory() -> Option<PackedDwaTokenSetInventory> {
    PACKED_DWA_TOKEN_SET_INVENTORY.with(|slot| slot.borrow_mut().take())
}

pub fn install_packed_decode_token_set_inventory(inventory: PackedDwaTokenSetInventory) {
    PACKED_DWA_TOKEN_SET_INVENTORY.with(|slot| *slot.borrow_mut() = Some(inventory));
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
fn put_var_u64(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

#[inline]
fn put_var_i64(out: &mut Vec<u8>, value: i64) {
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    put_var_u64(out, zigzag);
}

#[inline]
fn take_var_u32(input: &[u8], pos: &mut usize) -> Result<u32, String> {
    let start = *pos;
    let Some(&first) = input.get(start) else {
        return Err("truncated packed DWA varint".to_owned());
    };
    if first < 0x80 {
        *pos = start + 1;
        return Ok(first as u32);
    }
    let Some(&second) = input.get(start + 1) else {
        return Err("truncated packed DWA varint".to_owned());
    };
    let mut value = (first & 0x7f) as u32 | ((second & 0x7f) as u32) << 7;
    if second < 0x80 {
        *pos = start + 2;
        return Ok(value);
    }
    let mut cursor = start + 2;
    for shift in [14u32, 21, 28] {
        let byte = *input
            .get(cursor)
            .ok_or_else(|| "truncated packed DWA varint".to_owned())?;
        cursor += 1;
        if shift == 28 && byte > 0x0f {
            return Err("overflowing packed DWA u32 varint".to_owned());
        }
        value |= ((byte & 0x7f) as u32) << shift;
        if byte < 0x80 {
            *pos = cursor;
            return Ok(value);
        }
    }
    Err("overflowing packed DWA u32 varint".to_owned())
}

#[inline]
fn take_var_u64(input: &[u8], pos: &mut usize) -> Result<u64, String> {
    let start = *pos;
    let Some(&first) = input.get(start) else {
        return Err("truncated packed DWA varint".to_owned());
    };
    if first < 0x80 {
        *pos = start + 1;
        return Ok(first as u64);
    }
    let Some(&second) = input.get(start + 1) else {
        return Err("truncated packed DWA varint".to_owned());
    };
    let mut value = (first & 0x7f) as u64 | ((second & 0x7f) as u64) << 7;
    if second < 0x80 {
        *pos = start + 2;
        return Ok(value);
    }
    let mut cursor = start + 2;
    for index in 2..10 {
        let byte = *input
            .get(cursor)
            .ok_or_else(|| "truncated packed DWA varint".to_owned())?;
        cursor += 1;
        if index == 9 && byte > 1 {
            return Err("overflowing packed DWA u64 varint".to_owned());
        }
        value |= ((byte & 0x7f) as u64) << (index * 7);
        if byte < 0x80 {
            *pos = cursor;
            return Ok(value);
        }
    }
    Err("overflowing packed DWA u64 varint".to_owned())
}

#[inline]
fn take_var_i64(input: &[u8], pos: &mut usize) -> Result<i64, String> {
    let zigzag = take_var_u64(input, pos)?;
    Ok(((zigzag >> 1) as i64) ^ (-((zigzag & 1) as i64)))
}

fn take_length_prefixed_slices<'a>(
    input: &'a [u8],
    pos: &mut usize,
    count: usize,
    label: &str,
) -> Result<Vec<&'a [u8]>, String> {
    let mut slices = Vec::with_capacity(count);
    for _ in 0..count {
        let len = take_var_u32(input, pos)? as usize;
        let end = pos
            .checked_add(len)
            .ok_or_else(|| format!("overflowing packed DWA {label} length"))?;
        let body = input
            .get(*pos..end)
            .ok_or_else(|| format!("truncated packed DWA {label}"))?;
        slices.push(body);
        *pos = end;
    }
    Ok(slices)
}

#[inline]
fn var_u64_len(mut value: u64) -> usize {
    let mut len = 1usize;
    while value >= 0x80 {
        value >>= 7;
        len += 1;
    }
    len
}

#[inline]
fn zigzag_i64(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn decode_packed_label_sequence(body: &[u8]) -> Result<Vec<Label>, String> {
    let mut pos = 0usize;
    let count = take_var_u32(body, &mut pos)? as usize;
    let mut labels = Vec::with_capacity(count);
    let mut previous = 0i64;
    for _ in 0..count {
        let delta = take_var_i64(body, &mut pos)?;
        let label64 = previous
            .checked_add(delta)
            .ok_or_else(|| "overflowing packed DWA label sequence".to_owned())?;
        labels.push(
            i32::try_from(label64)
                .map_err(|_| "overflowing packed DWA label sequence".to_owned())?,
        );
        previous = label64;
    }
    if pos != body.len() {
        return Err("trailing bytes in packed DWA label sequence".to_owned());
    }
    Ok(labels)
}

fn decode_packed_u32_sequence(body: &[u8], label: &str) -> Result<Vec<u32>, String> {
    let mut pos = 0usize;
    let mode = *body
        .get(pos)
        .ok_or_else(|| format!("truncated packed DWA {label} sequence mode"))?;
    pos += 1;
    let count = take_var_u32(body, &mut pos)? as usize;
    let mut values = Vec::with_capacity(count);
    match mode {
        0 => {
            for _ in 0..count {
                values.push(take_var_u32(body, &mut pos)?);
            }
        }
        1 => {
            let mut previous = 0i64;
            for _ in 0..count {
                let delta = take_var_i64(body, &mut pos)?;
                let value = previous
                    .checked_add(delta)
                    .ok_or_else(|| format!("overflowing packed DWA {label} sequence"))?;
                values.push(
                    u32::try_from(value)
                        .map_err(|_| format!("overflowing packed DWA {label} sequence"))?,
                );
                previous = value;
            }
        }
        _ => return Err(format!("invalid packed DWA {label} sequence mode")),
    }
    if pos != body.len() {
        return Err(format!("trailing bytes in packed DWA {label} sequence"));
    }
    Ok(values)
}

fn decode_packed_runtime_label_pool(
    bodies: &[&[u8]],
) -> Result<PackedRuntimeSeqPool<Label>, String> {
    let mut values = Vec::<Label>::new();
    let mut spans = Vec::<[u32; 2]>::with_capacity(bodies.len());
    for body in bodies {
        let mut pos = 0usize;
        let count = take_var_u32(body, &mut pos)? as usize;
        let start = u32::try_from(values.len())
            .map_err(|_| "packed runtime label pool exceeds u32".to_owned())?;
        values.reserve(count);
        let mut previous = 0i64;
        for _ in 0..count {
            let delta = take_var_i64(body, &mut pos)?;
            let label64 = previous
                .checked_add(delta)
                .ok_or_else(|| "overflowing packed DWA label sequence".to_owned())?;
            values.push(
                i32::try_from(label64)
                    .map_err(|_| "overflowing packed DWA label sequence".to_owned())?,
            );
            previous = label64;
        }
        if pos != body.len() {
            return Err("trailing bytes in packed DWA label sequence".to_owned());
        }
        spans.push([start, count as u32]);
    }
    Ok(PackedRuntimeSeqPool {
        values: values.into_boxed_slice(),
        spans: spans.into_boxed_slice(),
    })
}

fn decode_packed_runtime_u32_pool(
    bodies: &[&[u8]],
    label: &str,
) -> Result<PackedRuntimeSeqPool<u32>, String> {
    let mut values = Vec::<u32>::new();
    let mut spans = Vec::<[u32; 2]>::with_capacity(bodies.len());
    for body in bodies {
        let mut pos = 0usize;
        let mode = *body
            .get(pos)
            .ok_or_else(|| format!("truncated packed DWA {label} sequence mode"))?;
        pos += 1;
        let count = take_var_u32(body, &mut pos)? as usize;
        let start = u32::try_from(values.len())
            .map_err(|_| format!("packed runtime {label} pool exceeds u32"))?;
        values.reserve(count);
        match mode {
            0 => {
                for _ in 0..count {
                    values.push(take_var_u32(body, &mut pos)?);
                }
            }
            1 => {
                let mut previous = 0i64;
                for _ in 0..count {
                    let delta = take_var_i64(body, &mut pos)?;
                    let value = previous
                        .checked_add(delta)
                        .ok_or_else(|| format!("overflowing packed DWA {label} sequence"))?;
                    values.push(
                        u32::try_from(value)
                            .map_err(|_| format!("overflowing packed DWA {label} sequence"))?,
                    );
                    previous = value;
                }
            }
            _ => return Err(format!("invalid packed DWA {label} sequence mode")),
        }
        if pos != body.len() {
            return Err(format!("trailing bytes in packed DWA {label} sequence"));
        }
        spans.push([start, count as u32]);
    }
    Ok(PackedRuntimeSeqPool {
        values: values.into_boxed_slice(),
        spans: spans.into_boxed_slice(),
    })
}

fn decode_packed_runtime_weights_flat(
    bodies: &[&[u8]],
    geometries: &[Vec<(u32, u32)>],
    token_set_count: usize,
) -> Result<(Vec<PackedRuntimeWeight>, Vec<u32>), String> {
    let token_id_capacity = bodies
        .iter()
        .filter_map(|body| {
            let mut pos = 1usize;
            (body.first().copied() == Some(0))
                .then(|| take_var_u32(body, &mut pos).ok())
                .flatten()
                .and_then(|geometry| geometries.get(geometry as usize))
                .map(Vec::len)
        })
        .sum::<usize>();
    let mut weights = Vec::with_capacity(bodies.len());
    let mut token_ids = Vec::with_capacity(token_id_capacity);
    for body in bodies {
        let mut pos = 0usize;
        let tag = *body
            .get(pos)
            .ok_or_else(|| "truncated packed DWA weight tag".to_owned())?;
        pos += 1;
        if tag == 1 {
            if pos != body.len() {
                return Err("trailing bytes in packed DWA full weight".to_owned());
            }
            weights.push(PackedRuntimeWeight {
                geometry: 0,
                token_ids_start: token_ids.len() as u32,
                full: 1,
            });
            continue;
        }
        if tag != 0 {
            return Err("invalid packed DWA weight tag".to_owned());
        }
        let geometry = take_var_u32(body, &mut pos)?;
        let geometry_ranges = geometries
            .get(geometry as usize)
            .ok_or_else(|| "invalid packed DWA weight-geometry index".to_owned())?;
        let token_ids_start = u32::try_from(token_ids.len())
            .map_err(|_| "packed runtime Weight token ids exceed u32".to_owned())?;
        for _ in geometry_ranges {
            let token_id = take_var_u32(body, &mut pos)?;
            if token_id as usize >= token_set_count {
                return Err("invalid packed DWA token-set index".to_owned());
            }
            token_ids.push(token_id);
        }
        if pos != body.len() {
            return Err("trailing bytes in packed DWA weight body".to_owned());
        }
        weights.push(PackedRuntimeWeight {
            geometry,
            token_ids_start,
            full: 0,
        });
    }
    Ok((weights, token_ids))
}

fn decode_packed_runtime_direct_row_pools(
    bodies: &[&[u8]],
    weight_count: usize,
) -> Result<(
    PackedRuntimeSeqPool<Label>,
    PackedRuntimeSeqPool<u32>,
    PackedRuntimeSeqPool<u32>,
    Vec<PackedRuntimeRow>,
), String> {
    let mut label_values = Vec::<Label>::new();
    let mut target_values = Vec::<u32>::new();
    let mut weight_values = Vec::<u32>::new();
    let mut label_spans = Vec::with_capacity(bodies.len());
    let mut target_spans = Vec::with_capacity(bodies.len());
    let mut weight_spans = Vec::with_capacity(bodies.len());
    let mut rows = Vec::with_capacity(bodies.len());
    for (row_id, body) in bodies.iter().enumerate() {
        let mut pos = 0usize;
        let count = take_var_u32(body, &mut pos)? as usize;
        let label_start = label_values.len() as u32;
        let target_start = target_values.len() as u32;
        let weight_start = weight_values.len() as u32;
        label_values.reserve(count);
        target_values.reserve(count);
        weight_values.reserve(count);
        let mut previous_label = 0i64;
        for index in 0..count {
            let delta = take_var_i64(body, &mut pos)?;
            let label64 = previous_label
                .checked_add(delta)
                .ok_or_else(|| "overflowing packed DWA label".to_owned())?;
            let label = i32::try_from(label64)
                .map_err(|_| "overflowing packed DWA label".to_owned())?;
            if index != 0 && label64 <= previous_label {
                return Err("packed DWA transition labels are not strictly increasing".to_owned());
            }
            previous_label = label64;
            let target = take_var_u32(body, &mut pos)?;
            let weight = take_var_u32(body, &mut pos)?;
            if weight as usize >= weight_count {
                return Err("invalid packed DWA weight index".to_owned());
            }
            label_values.push(label);
            target_values.push(target);
            weight_values.push(weight);
        }
        if pos != body.len() {
            return Err("trailing bytes in packed DWA transition-row body".to_owned());
        }
        label_spans.push([label_start, count as u32]);
        target_spans.push([target_start, count as u32]);
        weight_spans.push([weight_start, count as u32]);
        let id = row_id as u32;
        rows.push(PackedRuntimeRow {
            labels: id,
            targets: id,
            weights: id,
        });
    }
    Ok((
        PackedRuntimeSeqPool {
            values: label_values.into_boxed_slice(),
            spans: label_spans.into_boxed_slice(),
        },
        PackedRuntimeSeqPool {
            values: target_values.into_boxed_slice(),
            spans: target_spans.into_boxed_slice(),
        },
        PackedRuntimeSeqPool {
            values: weight_values.into_boxed_slice(),
            spans: weight_spans.into_boxed_slice(),
        },
        rows,
    ))
}

fn decode_packed_token_set_chunk(
    body: &[u8],
) -> Result<Vec<(std::sync::Arc<RangeSetBlaze<u32>>, u32)>, String> {
    let mut pos = 0usize;
    let token_set_count = take_var_u32(body, &mut pos)? as usize;
    let mut out = Vec::with_capacity(token_set_count);
    let mut previous = EncodedTokenSet::new();
    let mut prefix_word_spans = Vec::<u32>::new();
    for _ in 0..token_set_count {
        let prefix_len = take_var_u32(body, &mut pos)? as usize;
        if prefix_len > previous.len() {
            return Err("invalid packed DWA token-set prefix length".to_owned());
        }
        let suffix_len = take_var_u32(body, &mut pos)? as usize;
        // Front coding means the next token set is literally a prefix of the
        // previous decoded range vector plus a suffix.  Reuse that allocation
        // in place instead of allocating a new Vec and copying the common
        // prefix for every set in the chunk.
        previous.truncate(prefix_len);
        prefix_word_spans.truncate(prefix_len);
        previous.reserve(suffix_len);
        prefix_word_spans.reserve(suffix_len);
        let mut word_spans = prefix_word_spans.last().copied().unwrap_or(0);
        let mut previous_end_plus_one = previous
            .last()
            .map_or(0u64, |range| range[1] as u64 + 1);
        for _ in 0..suffix_len {
            let gap = take_var_u64(body, &mut pos)?;
            let start64 = previous_end_plus_one
                .checked_add(gap)
                .ok_or_else(|| "overflowing packed token-set range".to_owned())?;
            let start = u32::try_from(start64)
                .map_err(|_| "overflowing packed token-set start".to_owned())?;
            let len = take_var_u32(body, &mut pos)?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| "overflowing packed token-set end".to_owned())?;
            previous_end_plus_one = end as u64 + 1;
            previous.push([start, end]);
            word_spans = word_spans.saturating_add(end / 64 - start / 64 + 1);
            prefix_word_spans.push(word_spans);
        }
        let tokens = RangeSetBlaze::from_sorted_disjoint(CheckSortedDisjoint::new(
            previous.iter().map(|range| range[0]..=range[1]),
        ));
        out.push((shared_rangeset_artifact_local(tokens), word_spans));
    }
    if pos != body.len() {
        return Err("invalid packed DWA token-set chunk length".to_owned());
    }
    Ok(out)
}

fn decode_packed_runtime_token_set_chunk(
    body: &[u8],
) -> Result<PackedRuntimeTokenSetChunk, String> {
    let mut pos = 0usize;
    let token_set_count = take_var_u32(body, &mut pos)? as usize;
    let mut previous = EncodedTokenSet::new();
    let mut prefix_word_spans = Vec::<u32>::new();
    let mut ranges = Vec::<[u32; 2]>::new();
    let mut spans = Vec::<PackedRuntimeTokenSetSpan>::with_capacity(token_set_count);
    for _ in 0..token_set_count {
        let prefix_len = take_var_u32(body, &mut pos)? as usize;
        if prefix_len > previous.len() {
            return Err("invalid packed DWA token-set prefix length".to_owned());
        }
        let suffix_len = take_var_u32(body, &mut pos)? as usize;
        previous.truncate(prefix_len);
        prefix_word_spans.truncate(prefix_len);
        previous.reserve(suffix_len);
        prefix_word_spans.reserve(suffix_len);
        let mut word_spans = prefix_word_spans.last().copied().unwrap_or(0);
        let mut previous_end_plus_one = previous
            .last()
            .map_or(0u64, |range| range[1] as u64 + 1);
        for _ in 0..suffix_len {
            let gap = take_var_u64(body, &mut pos)?;
            let start64 = previous_end_plus_one
                .checked_add(gap)
                .ok_or_else(|| "overflowing packed token-set range".to_owned())?;
            let start = u32::try_from(start64)
                .map_err(|_| "overflowing packed token-set start".to_owned())?;
            let len = take_var_u32(body, &mut pos)?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| "overflowing packed token-set end".to_owned())?;
            previous_end_plus_one = end as u64 + 1;
            previous.push([start, end]);
            word_spans = word_spans.saturating_add(end / 64 - start / 64 + 1);
            prefix_word_spans.push(word_spans);
        }
        let start = u32::try_from(ranges.len())
            .map_err(|_| "packed runtime token-set chunk exceeds u32 offsets".to_owned())?;
        let len = u32::try_from(previous.len())
            .map_err(|_| "packed runtime token set exceeds u32 ranges".to_owned())?;
        ranges.extend_from_slice(&previous);
        spans.push(PackedRuntimeTokenSetSpan {
            start,
            len,
            word_spans,
        });
    }
    if pos != body.len() {
        return Err("invalid packed DWA token-set chunk length".to_owned());
    }
    Ok(PackedRuntimeTokenSetChunk {
        ranges: ranges.into_boxed_slice(),
        spans: spans.into_boxed_slice(),
    })
}

fn decode_packed_runtime_weight(
    body: &[u8],
    geometries: &[Vec<(u32, u32)>],
    token_set_count: usize,
) -> Result<(u32, bool, Vec<u32>), String> {
    let mut pos = 0usize;
    let tag = *body
        .get(pos)
        .ok_or_else(|| "truncated packed DWA weight tag".to_owned())?;
    pos += 1;
    if tag == 1 {
        if pos != body.len() {
            return Err("trailing bytes in packed DWA full weight".to_owned());
        }
        return Ok((0, true, Vec::new()));
    }
    if tag != 0 {
        return Err("invalid packed DWA weight tag".to_owned());
    }
    let geometry = take_var_u32(body, &mut pos)?;
    let Some(geometry_ranges) = geometries.get(geometry as usize) else {
        return Err("invalid packed DWA weight-geometry index".to_owned());
    };
    let mut token_ids = Vec::with_capacity(geometry_ranges.len());
    for _ in geometry_ranges {
        let token_id = take_var_u32(body, &mut pos)?;
        if token_id as usize >= token_set_count {
            return Err("invalid packed DWA token-set index".to_owned());
        }
        token_ids.push(token_id);
    }
    if pos != body.len() {
        return Err("trailing bytes in packed DWA weight body".to_owned());
    }
    Ok((geometry, false, token_ids))
}

fn decode_packed_runtime_row(
    body: &[u8],
    weight_count: usize,
) -> Result<(Vec<Label>, Vec<u32>, Vec<u32>), String> {
    let mut pos = 0usize;
    let transition_count = take_var_u32(body, &mut pos)? as usize;
    let mut labels = Vec::with_capacity(transition_count);
    let mut targets = Vec::with_capacity(transition_count);
    let mut weights = Vec::with_capacity(transition_count);
    let mut previous_label = 0i64;
    for index in 0..transition_count {
        let delta = take_var_i64(body, &mut pos)?;
        let label64 = previous_label
            .checked_add(delta)
            .ok_or_else(|| "overflowing packed DWA label".to_owned())?;
        let label = i32::try_from(label64)
            .map_err(|_| "overflowing packed DWA label".to_owned())?;
        if index != 0 && label64 <= previous_label {
            return Err("packed DWA transition labels are not strictly increasing".to_owned());
        }
        previous_label = label64;
        let target = take_var_u32(body, &mut pos)?;
        let weight = take_var_u32(body, &mut pos)?;
        if weight as usize >= weight_count {
            return Err("invalid packed DWA weight index".to_owned());
        }
        labels.push(label);
        targets.push(target);
        weights.push(weight);
    }
    if pos != body.len() {
        return Err("trailing bytes in packed DWA transition-row body".to_owned());
    }
    Ok((labels, targets, weights))
}

fn decode_packed_weight_geometry(body: &[u8]) -> Result<Vec<(u32, u32)>, String> {
    let mut pos = 0usize;
    let entry_count = take_var_u32(body, &mut pos)? as usize;
    let mut ranges = Vec::with_capacity(entry_count);
    let mut previous_end_plus_one = 0u64;
    for _ in 0..entry_count {
        let gap = take_var_u64(body, &mut pos)?;
        let start64 = previous_end_plus_one
            .checked_add(gap)
            .ok_or_else(|| "overflowing packed weight geometry".to_owned())?;
        let start = u32::try_from(start64)
            .map_err(|_| "overflowing packed weight geometry start".to_owned())?;
        let len = take_var_u32(body, &mut pos)?;
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

fn decode_packed_weight(
    body: &[u8],
    ts_pool: &[std::sync::Arc<RangeSetBlaze<u32>>],
    geometries: &[Vec<(u32, u32)>],
) -> Result<Weight, String> {
    let mut pos = 0usize;
    let tag = *body
        .get(pos)
        .ok_or_else(|| "truncated packed DWA weight tag".to_owned())?;
    pos += 1;
    if tag == 1 {
        if pos != body.len() {
            return Err("trailing bytes in packed DWA full weight".to_owned());
        }
        return Ok(Weight::all());
    }
    if tag != 0 {
        return Err("invalid packed DWA weight tag".to_owned());
    }
    let geometry_index = take_var_u32(body, &mut pos)? as usize;
    let geometry = geometries
        .get(geometry_index)
        .ok_or_else(|| "invalid packed DWA weight-geometry index".to_owned())?;
    if geometry.is_empty() {
        if pos != body.len() {
            return Err("trailing bytes in packed DWA empty weight".to_owned());
        }
        return Ok(Weight::empty());
    }
    // Most parser weights have only a small number of TSID ranges.  Keep just
    // their token-set indices inline while parsing, then stream the already
    // sorted shared geometry directly into RangeMapBlaze.  This avoids one
    // heap Vec of (RangeInclusive, Arc) entries per unique Weight.
    let mut token_set_ids = SmallVec::<[u32; 16]>::with_capacity(geometry.len());
    for _ in geometry {
        let token_set_idx = take_var_u32(body, &mut pos)?;
        if token_set_idx as usize >= ts_pool.len() {
            return Err("invalid packed DWA token-set index".to_owned());
        }
        token_set_ids.push(token_set_idx);
    }
    if pos != body.len() {
        return Err("trailing bytes in packed DWA weight body".to_owned());
    }
    let map = RangeMapBlaze::from_sorted_disjoint_map(CheckSortedDisjointMap::new(
        geometry
            .iter()
            .zip(token_set_ids.iter())
            .map(|(&(start, end), &token_set_idx)| {
                (start..=end, &ts_pool[token_set_idx as usize])
            }),
    ));
    Ok(finalize_weight_map_artifact_local(map))
}

fn decode_packed_transition_row(
    body: &[u8],
    w_pool: &[Weight],
    state_count: usize,
) -> Result<BTreeMap<Label, (u32, Weight)>, String> {
    let mut pos = 0usize;
    let transition_count = take_var_u32(body, &mut pos)? as usize;
    let mut entries = Vec::with_capacity(transition_count);
    let mut previous_label = 0i64;
    for index in 0..transition_count {
        let delta = take_var_i64(body, &mut pos)?;
        let label64 = previous_label
            .checked_add(delta)
            .ok_or_else(|| "overflowing packed DWA label".to_owned())?;
        let label = i32::try_from(label64)
            .map_err(|_| "overflowing packed DWA label".to_owned())?;
        if index != 0 && label64 <= previous_label {
            return Err("packed DWA transition labels are not strictly increasing".to_owned());
        }
        previous_label = label64;
        let target = take_var_u32(body, &mut pos)?;
        if target as usize >= state_count {
            return Err("invalid packed DWA target state".to_owned());
        }
        let weight_idx = take_var_u32(body, &mut pos)? as usize;
        let weight = w_pool
            .get(weight_idx)
            .cloned()
            .ok_or_else(|| "invalid packed DWA weight index".to_owned())?;
        entries.push((label, (target, weight)));
    }
    if pos != body.len() {
        return Err("trailing bytes in packed DWA transition-row body".to_owned());
    }
    Ok(entries.into_iter().collect())
}

fn hash_packed_transition_row(row: &[(Label, u32, u32)]) -> u64 {
    let mut hasher = FxHasher::default();
    row.hash(&mut hasher);
    hasher.finish()
}

impl DWA {
    /// Encode the compact parser-DWA section used by sectioned constraint
    /// artifacts, without an outer serde/bincode wrapper.
    pub fn artifact_packed_bytes(&self) -> Vec<u8> {
        self.to_packed_bytes()
    }

    /// Decode a compact parser-DWA section and return the token-set inventory
    /// observed during decode so a caller on another thread can hand it to
    /// runtime finalization without rescanning the expanded DWA.
    pub fn from_artifact_packed_bytes(
        input: &[u8],
    ) -> Result<(Self, Option<PackedDwaTokenSetInventory>), String> {
        let dwa = Self::from_packed_bytes(input)?;
        let inventory = take_packed_decode_token_set_inventory();
        Ok((dwa, inventory))
    }

    fn pooled_parts(&self) -> (Vec<EncodedTokenSet>, Vec<WeightPoolEntry>, Vec<DWAStateSerde>) {
        // These keys are already process-local interned pointers.  SipHash is
        // wasted work here: the pointer value itself is a perfectly adequate
        // input to FxHash, and exact identity is still checked by the map.
        let mut ts_ptr_to_idx: FxHashMap<usize, u32> = FxHashMap::default();
        ts_ptr_to_idx.reserve(32_768);
        // Arc pointer identity is the hot path, but structural identity is the
        // serialization identity. Independent parallel builders can produce
        // equal token sets with different Arc layouts.
        let mut ts_value_to_idx: FxHashMap<Arc<RangeSetBlaze<u32>>, u32> = FxHashMap::default();
        let mut token_set_pool: Vec<EncodedTokenSet> = Vec::new();
        let mut intern_token_set = |ts: &std::sync::Arc<RangeSetBlaze<u32>>| -> u32 {
            let ptr = std::sync::Arc::as_ptr(ts) as usize;
            if let Some(&idx) = ts_ptr_to_idx.get(&ptr) {
                return idx;
            }
            let idx = if let Some(&idx) = ts_value_to_idx.get(ts) {
                idx
            } else {
                let idx = token_set_pool.len() as u32;
                token_set_pool.push(ts.ranges().map(|r| [*r.start(), *r.end()]).collect());
                ts_value_to_idx.insert(std::sync::Arc::clone(ts), idx);
                idx
            };
            ts_ptr_to_idx.insert(ptr, idx);
            idx
        };

        let mut w_ptr_to_idx: FxHashMap<usize, u32> = FxHashMap::default();
        w_ptr_to_idx.reserve(32_768);
        // Level 2: same scheme for weights.  `Weight::Hash` is the cached full
        // structural hash and `Weight::Eq` is structural equality, so this
        // fallback is exact rather than probabilistic.
        let mut w_value_to_idx: FxHashMap<Weight, u32> = FxHashMap::default();
        let mut weight_pool: Vec<WeightPoolEntry> = Vec::new();
        let mut intern_weight = |w: &Weight| -> u32 {
            let ptr = w.ptr_key();
            if let Some(&idx) = w_ptr_to_idx.get(&ptr) {
                return idx;
            }
            let idx = if let Some(&idx) = w_value_to_idx.get(w) {
                idx
            } else {
                let idx = weight_pool.len() as u32;
                if w.is_full() {
                    weight_pool.push(WeightPoolEntry {
                        all: true,
                        entries: Vec::new(),
                    });
                } else {
                    let entries = w
                        .raw_range_values()
                        .map(|(range, tokens)| {
                            let ts_idx = intern_token_set(tokens);
                            (*range.start(), *range.end(), ts_idx)
                        })
                        .collect();
                    weight_pool.push(WeightPoolEntry {
                        all: false,
                        entries,
                    });
                }
                w_value_to_idx.insert(w.clone(), idx);
                idx
            };
            w_ptr_to_idx.insert(ptr, idx);
            idx
        };

        let states = self
            .states
            .iter()
            .map(|state| {
                let transitions = state
                    .transitions
                    .iter()
                    .map(|(&label, (target, weight))| (label, *target, intern_weight(weight)))
                    .collect();
                let final_weight = state.final_weight.as_ref().map(&mut intern_weight);
                DWAStateSerde {
                    transitions,
                    final_weight,
                }
            })
            .collect();
        (token_set_pool, weight_pool, states)
    }

    /// Build the pools needed by the compact artifact without first expanding
    /// every state into a second complete transition vector.  The compact wire
    /// format stores transition rows by identity anyway, so discover those row
    /// classes against the runtime DWA first and translate Weight pointers to
    /// pool indices only for the unique rows.
    fn packed_pooled_parts(
        &self,
    ) -> (
        Vec<Arc<RangeSetBlaze<u32>>>,
        Vec<WeightPoolEntry>,
        Vec<Vec<(Label, u32, u32)>>,
        Vec<(u32, Option<u32>)>,
        usize,
    ) {
        let profile = std::env::var_os("GLRMASK_PROFILE_DWA_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let phase_started = profile.then(std::time::Instant::now);
        let transition_count = self
            .states
            .iter()
            .map(|state| state.transitions.len())
            .sum::<usize>();

        let mut row_representatives = Vec::<usize>::new();
        let mut state_row_ids = Vec::<u32>::with_capacity(self.states.len());
        if self.shared_transition_rows {
            let mut row_id_by_ptr = FxHashMap::<usize, u32>::default();
            row_id_by_ptr.reserve(self.states.len().min(32_768));
            for (state_index, state) in self.states.iter().enumerate() {
                debug_assert!(matches!(
                    state.transitions,
                    DwaTransitionMap::Shared(_) | DwaTransitionMap::Packed(_)
                ));
                let ptr = state.transitions.ptr_key();
                let row_id = if let Some(&row_id) = row_id_by_ptr.get(&ptr) {
                    row_id
                } else {
                    let row_id = row_representatives.len() as u32;
                    row_representatives.push(state_index);
                    row_id_by_ptr.insert(ptr, row_id);
                    row_id
                };
                state_row_ids.push(row_id);
            }
        } else {
            let row_hash = |state: &DWAState| -> u64 {
            // Canonically shared rows can use pointer identity directly. Rows
            // that have been mutated are converted back to Owned by DerefMut,
            // so only those rows pay the structural hashing cost.
            if matches!(
                state.transitions,
                DwaTransitionMap::Shared(_) | DwaTransitionMap::Packed(_)
            ) {
                return state.transitions.ptr_key() as u64;
            }
            let mut hasher = FxHasher::default();
            state.transitions.len().hash(&mut hasher);
            for (label, (target, weight)) in state.transitions.iter() {
                label.hash(&mut hasher);
                target.hash(&mut hasher);
                weight.ptr_key().hash(&mut hasher);
            }
            hasher.finish()
            };
            let row_hashes = if self.states.len() >= 4_096 && rayon::current_num_threads() > 1 {
                self.states.par_iter().map(row_hash).collect::<Vec<_>>()
            } else {
                self.states.iter().map(row_hash).collect::<Vec<_>>()
            };
            let mut row_hash_to_indices = FxHashMap::<u64, Vec<u32>>::default();
            row_hash_to_indices.reserve(self.states.len().min(32_768));
            for (state_index, (&hash, state)) in row_hashes.iter().zip(&self.states).enumerate() {
                let bucket = row_hash_to_indices.entry(hash).or_default();
                let row_id = bucket
                    .iter()
                    .copied()
                    .find(|&candidate| {
                        let representative = row_representatives[candidate as usize];
                        self.states[representative].transitions == state.transitions
                    })
                    .unwrap_or_else(|| {
                        let row_id = row_representatives.len() as u32;
                        row_representatives.push(state_index);
                        bucket.push(row_id);
                        row_id
                    });
                state_row_ids.push(row_id);
            }
        }
        let row_classes_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        let mut weight_ptr_to_id: FxHashMap<usize, u32> = FxHashMap::default();
        // Small parser DWAs are common in ordinary schema constraints. A fixed
        // 32K reservation cost hundreds of microseconds even when the DWA had
        // only tens or hundreds of distinct weights.
        let weight_ref_count = transition_count.saturating_add(self.states.len());
        // Ordinary parser DWAs have far fewer distinct Weight objects than
        // transition/final references. Reserving for every reference touches
        // several times more hash-table memory than the whole packed wire.
        // Half the reference count avoids that fixed cost while still leaving
        // ample headroom for the observed ordinary-schema uniqueness ratio;
        // unusually high-entropy DWAs can grow normally.
        weight_ptr_to_id.reserve(weight_ref_count.div_ceil(2).clamp(64, 32_768));
        let mut weight_refs = Vec::<Weight>::new();
        let mut intern_weight = |w: &Weight| -> u32 {
            let ptr = w.ptr_key();
            *weight_ptr_to_id.entry(ptr).or_insert_with(|| {
                let idx = weight_refs.len() as u32;
                weight_refs.push(w.clone());
                idx
            })
        };

        let transition_rows_started = profile.then(std::time::Instant::now);
        let transition_rows = row_representatives
            .iter()
            .map(|&state_index| {
                self.states[state_index]
                    .transitions
                    .entries()
                    .map(|(label, target, weight)| (label, target, intern_weight(weight)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let transition_rows_ms = transition_rows_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let state_rows_started = profile.then(std::time::Instant::now);
        let state_rows = self
            .states
            .iter()
            .zip(state_row_ids)
            .map(|(state, row)| (row, state.final_weight.as_ref().map(&mut intern_weight)))
            .collect::<Vec<_>>();
        let state_rows_ms = state_rows_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let weight_discovery_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile {
            eprintln!(
                "[glrmask/profile][dwa_weight_discovery_split] transition_rows_ms={transition_rows_ms:.3} state_rows_ms={state_rows_ms:.3} transition_entries={} final_weights={}",
                transition_rows.iter().map(Vec::len).sum::<usize>(),
                state_rows.iter().filter(|(_, final_weight)| final_weight.is_some()).count(),
            );
        }
        let phase_started = profile.then(std::time::Instant::now);

        // Weight IDs above are assigned in stable first-encounter order. Walk
        // the unique Weight maps in parallel, then assign token-set IDs in that
        // same Weight/range order. This keeps the wire representation exactly
        // deterministic while avoiding a serial chase through tens of thousands
        // of scattered RangeMapBlaze allocations on first save after compile.
        struct RawWeightPoolEntry<'a> {
            all: bool,
            entries: Vec<(u32, u32, &'a Arc<RangeSetBlaze<u32>>)>,
        }
        fn materialize_weight(weight: &Weight) -> RawWeightPoolEntry<'_> {
            RawWeightPoolEntry {
                all: weight.is_full(),
                entries: if weight.is_full() {
                    Vec::new()
                } else {
                    weight
                        .raw_range_values()
                        .map(|(range, tokens)| (*range.start(), *range.end(), tokens))
                        .collect()
                },
            }
        }
        let raw_weight_pool = if weight_refs.len() >= 256 && rayon::current_num_threads() > 1 {
            weight_refs
                .par_iter()
                .map(materialize_weight)
                .collect::<Vec<_>>()
        } else {
            weight_refs
                .iter()
                .map(materialize_weight)
                .collect::<Vec<_>>()
        };
        let weight_materialize_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        let raw_token_set_refs = raw_weight_pool
            .iter()
            .map(|weight| weight.entries.len())
            .sum::<usize>();
        let mut ts_ptr_to_idx: FxHashMap<usize, u32> = FxHashMap::default();
        // Token sets are shared heavily across Weight ranges. Estimate from
        // the number of distinct weights instead of reserving for every range
        // reference; the map remains growable for unusual grammars.
        ts_ptr_to_idx.reserve(
            raw_token_set_refs
                .min(raw_weight_pool.len().saturating_mul(4).max(64))
                .min(32_768),
        );
        let mut token_set_pool: Vec<Arc<RangeSetBlaze<u32>>> = Vec::new();
        let mut weight_pool = Vec::<WeightPoolEntry>::with_capacity(raw_weight_pool.len());
        for raw in raw_weight_pool {
            if raw.all {
                weight_pool.push(WeightPoolEntry {
                    all: true,
                    entries: Vec::new(),
                });
                continue;
            }
            let entries = raw
                .entries
                .into_iter()
                .map(|(start, end, tokens)| {
                    let ptr = Arc::as_ptr(tokens) as usize;
                    let token_set = if let Some(&existing) = ts_ptr_to_idx.get(&ptr) {
                        existing
                    } else {
                        let idx = token_set_pool.len() as u32;
                        ts_ptr_to_idx.insert(ptr, idx);
                        token_set_pool.push(Arc::clone(tokens));
                        idx
                    };
                    (start, end, token_set)
                })
                .collect();
            weight_pool.push(WeightPoolEntry {
                all: false,
                entries,
            });
        }
        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][dwa_pooled_parts] row_classes_ms={row_classes_ms:.3} weight_discovery_ms={weight_discovery_ms:.3} weight_materialize_ms={weight_materialize_ms:.3} token_set_intern_ms={:.3} total_ms={:.3} unique_rows={} weights={} token_sets={}",
                phase_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                total_started.elapsed().as_secs_f64() * 1000.0,
                transition_rows.len(),
                weight_pool.len(),
                token_set_pool.len(),
            );
        }

        (
            token_set_pool,
            weight_pool,
            transition_rows,
            state_rows,
            transition_count,
        )
    }

    fn to_packed_bytes(&self) -> Vec<u8> {
        let profile = std::env::var_os("GLRMASK_PROFILE_DWA_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let phase_started = profile.then(std::time::Instant::now);
        let (token_set_arcs, mut weight_pool, transition_rows, state_rows, transition_count) =
            self.packed_pooled_parts();
        let pooled_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        // Adjacent token sets are often almost identical. Sort them
        // lexicographically, remap weight references, and front-code them in
        // independently decodable chunks. Chunk boundaries preserve parallel
        // load while sacrificing very little prefix sharing.
        let sort_token_sets = token_set_arcs.len() >= 1_024;
        let token_set_pool = if !sort_token_sets {
            Vec::new()
        } else if rayon::current_num_threads() > 1 {
            token_set_arcs
                .par_iter()
                .map(|ts| ts.ranges().map(|r| [*r.start(), *r.end()]).collect::<EncodedTokenSet>())
                .collect::<Vec<_>>()
        } else {
            token_set_arcs
                .iter()
                .map(|ts| ts.ranges().map(|r| [*r.start(), *r.end()]).collect::<EncodedTokenSet>())
                .collect::<Vec<_>>()
        };
        let token_set_materialize_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);
        // Lexicographic ordering improves front-coding for large token-set
        // pools, but for ordinary schema DWAs the O(n log n) sort/remap cost is
        // a material fraction of the entire save. Preserve stable discovery
        // order below 1K sets; it is already locality-friendly because sets are
        // discovered by Weight/range traversal.
        let mut token_set_order = (0..token_set_pool.len()).collect::<Vec<_>>();
        if sort_token_sets {
            let compare_token_sets = |left: &usize, right: &usize| {
                token_set_pool[*left].cmp(&token_set_pool[*right])
            };
            if token_set_order.len() >= 4_096 && rayon::current_num_threads() > 1 {
                token_set_order.par_sort_unstable_by(compare_token_sets);
            } else {
                token_set_order.sort_unstable_by(compare_token_sets);
            }
            let mut old_to_new_token_set = vec![0u32; token_set_pool.len()];
            for (new_index, &old_index) in token_set_order.iter().enumerate() {
                old_to_new_token_set[old_index] = new_index as u32;
            }
            for weight in &mut weight_pool {
                for (_, _, token_set) in &mut weight.entries {
                    *token_set = old_to_new_token_set[*token_set as usize];
                }
            }
        }
        let mut token_set_slots = token_set_pool.into_iter().map(Some).collect::<Vec<_>>();
        let token_set_sort_remap_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);
        let token_set_pool = token_set_order
            .into_iter()
            .map(|old_index| {
                token_set_slots[old_index]
                    .take()
                    .expect("token-set permutation contains each index once")
            })
            .collect::<Vec<_>>();

        // Pool the outer TSID range geometry independently from the token-set
        // values. Parser weights often differ only in which token set is
        // attached to an otherwise identical TSID partition.
        let mut geometry_hash_to_indices = FxHashMap::<u64, Vec<u32>>::default();
        let mut weight_geometries = Vec::<Vec<(u32, u32)>>::new();
        let mut weight_geometry_ids = Vec::<u32>::with_capacity(weight_pool.len());
        for weight in &weight_pool {
            if weight.all {
                weight_geometry_ids.push(u32::MAX);
                continue;
            }
            let geometry = weight
                .entries
                .iter()
                .map(|&(start, end, _)| (start, end))
                .collect::<Vec<_>>();
            let mut hasher = FxHasher::default();
            geometry.hash(&mut hasher);
            let hash = hasher.finish();
            let mut existing = None;
            if let Some(candidates) = geometry_hash_to_indices.get(&hash) {
                for &candidate in candidates {
                    if weight_geometries[candidate as usize] == geometry {
                        existing = Some(candidate);
                        break;
                    }
                }
            }
            let geometry_id = if let Some(existing) = existing {
                existing
            } else {
                let index = weight_geometries.len() as u32;
                weight_geometries.push(geometry);
                geometry_hash_to_indices.entry(hash).or_default().push(index);
                index
            };
            weight_geometry_ids.push(geometry_id);
        }
        let geometry_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        // Row pooling was fused into packed_pooled_parts(), before Weight-id
        // translation, so duplicate state rows never become duplicate serde
        // transition vectors.

        let mut out = Vec::new();
        let mut body = Vec::new();
        // Format tag, so accidental cross-version use fails cleanly rather
        // than silently constructing nonsense.
        out.extend_from_slice(b"DWP6");
        put_var_u32(&mut out, self.start_state);

        let token_set_count = token_set_arcs.len();
        put_var_u32(&mut out, token_set_count as u32);
        const TOKEN_SET_CHUNK_SIZE: usize = 64;
        let token_set_chunk_count = token_set_count.div_ceil(TOKEN_SET_CHUNK_SIZE);
        let token_set_range_count = if sort_token_sets {
            token_set_pool.iter().map(Vec::len).sum::<usize>()
        } else {
            token_set_arcs.iter().map(|set| set.ranges().len()).sum::<usize>()
        };
        put_var_u32(&mut out, token_set_chunk_count as u32);
        let encode_token_set_chunk = |chunk: &[EncodedTokenSet]| {
            let mut encoded = Vec::new();
            put_var_u32(&mut encoded, chunk.len() as u32);
            let mut previous: &[[u32; 2]] = &[];
            for token_set in chunk {
                let prefix_len = previous
                    .iter()
                    .zip(token_set)
                    .take_while(|(left, right)| left == right)
                    .count();
                put_var_u32(&mut encoded, prefix_len as u32);
                put_var_u32(&mut encoded, (token_set.len() - prefix_len) as u32);
                let mut previous_end_plus_one = if prefix_len == 0 {
                    0u64
                } else {
                    previous[prefix_len - 1][1] as u64 + 1
                };
                for &[start, end] in &token_set[prefix_len..] {
                    let start64 = start as u64;
                    let gap = start64
                        .checked_sub(previous_end_plus_one)
                        .expect("token-set ranges are sorted and disjoint");
                    put_var_u64(&mut encoded, gap);
                    put_var_u32(&mut encoded, end - start);
                    previous_end_plus_one = end as u64 + 1;
                }
                previous = token_set;
            }
            encoded
        };
        let encode_token_set_arc_chunk = |chunk: &[Arc<RangeSetBlaze<u32>>]| {
            let mut encoded = Vec::new();
            put_var_u32(&mut encoded, chunk.len() as u32);
            // The direct Arc path used to walk each RangeSetBlaze three or
            // four times (prefix comparison, len, nth, then skip). Ordinary
            // schema token sets are small enough that two reusable flat range
            // buffers are much cheaper: traverse each set exactly once, reuse
            // the allocations for the whole chunk, and front-code the slices.
            let mut previous = EncodedTokenSet::new();
            let mut current = EncodedTokenSet::new();
            for token_set in chunk {
                current.clear();
                current.extend(
                    token_set
                        .ranges()
                        .map(|range| [*range.start(), *range.end()]),
                );
                let prefix_len = previous
                    .iter()
                    .zip(&current)
                    .take_while(|(left, right)| left == right)
                    .count();
                put_var_u32(&mut encoded, prefix_len as u32);
                put_var_u32(&mut encoded, (current.len() - prefix_len) as u32);
                let mut previous_end_plus_one = if prefix_len == 0 {
                    0u64
                } else {
                    previous[prefix_len - 1][1] as u64 + 1
                };
                for &[start, end] in &current[prefix_len..] {
                    let gap = (start as u64)
                        .checked_sub(previous_end_plus_one)
                        .expect("token-set ranges are sorted and disjoint");
                    put_var_u64(&mut encoded, gap);
                    put_var_u32(&mut encoded, end - start);
                    previous_end_plus_one = end as u64 + 1;
                }
                std::mem::swap(&mut previous, &mut current);
            }
            encoded
        };
        if !sort_token_sets {
            // The p99-ish ordinary schemas have only hundreds of token sets
            // but tens of thousands of ranges. At that point range encoding,
            // not scheduling, dominates. Encode independent front-coding
            // chunks in parallel; smaller p50/p90 pools stay serial.
            if token_set_range_count >= 16_384 && rayon::current_num_threads() > 1 {
                let encoded_chunks = token_set_arcs
                    .par_chunks(TOKEN_SET_CHUNK_SIZE)
                    .map(encode_token_set_arc_chunk)
                    .collect::<Vec<_>>();
                for encoded in encoded_chunks {
                    put_var_u32(&mut out, encoded.len() as u32);
                    out.extend_from_slice(&encoded);
                }
            } else {
                for chunk in token_set_arcs.chunks(TOKEN_SET_CHUNK_SIZE) {
                    body = encode_token_set_arc_chunk(chunk);
                    put_var_u32(&mut out, body.len() as u32);
                    out.extend_from_slice(&body);
                }
            }
        } else if token_set_range_count >= 1_000_000 && rayon::current_num_threads() > 1 {
            let encoded_chunks = token_set_pool
                .par_chunks(TOKEN_SET_CHUNK_SIZE)
                .map(encode_token_set_chunk)
                .collect::<Vec<_>>();
            for encoded in encoded_chunks {
                put_var_u32(&mut out, encoded.len() as u32);
                out.extend_from_slice(&encoded);
            }
        } else {
            for chunk in token_set_pool.chunks(TOKEN_SET_CHUNK_SIZE) {
                body = encode_token_set_chunk(chunk);
                put_var_u32(&mut out, body.len() as u32);
                out.extend_from_slice(&body);
            }
        }
        let token_sets_end = out.len();
        let token_set_encode_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        put_var_u32(&mut out, weight_geometries.len() as u32);
        for geometry in &weight_geometries {
            body.clear();
            put_var_u32(&mut body, geometry.len() as u32);
            let mut previous_end_plus_one = 0u64;
            for &(start, end) in geometry {
                let gap = (start as u64)
                    .checked_sub(previous_end_plus_one)
                    .expect("weight geometry is sorted and disjoint");
                put_var_u64(&mut body, gap);
                put_var_u32(&mut body, end - start);
                previous_end_plus_one = end as u64 + 1;
            }
            put_var_u32(&mut out, body.len() as u32);
            out.extend_from_slice(&body);
        }
        let weight_geometries_end = out.len();
        let geometry_encode_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        put_var_u32(&mut out, weight_pool.len() as u32);
        for (weight_index, weight) in weight_pool.iter().enumerate() {
            body.clear();
            if weight.all {
                body.push(1);
            } else {
                body.push(0);
                put_var_u32(&mut body, weight_geometry_ids[weight_index]);
                for &(_, _, token_set) in &weight.entries {
                    put_var_u32(&mut body, token_set);
                }
            }
            put_var_u32(&mut out, body.len() as u32);
            out.extend_from_slice(&body);
        }
        let weights_end = out.len();
        let weight_encode_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        let unique_row_transition_count = transition_rows.iter().map(Vec::len).sum::<usize>();
        let use_component_rows = unique_row_transition_count >= 100_000
            && std::env::var_os("GLRMASK_DWA_DIRECT_ARTIFACT_ROWS").is_none();
        out.push(u8::from(use_component_rows));
        if use_component_rows {
            let mut label_ids = FxHashMap::<u64, Vec<u32>>::default();
            let mut target_ids = FxHashMap::<u64, Vec<u32>>::default();
            let mut weight_ids = FxHashMap::<u64, Vec<u32>>::default();
            label_ids.reserve(512);
            target_ids.reserve(transition_rows.len());
            weight_ids.reserve(transition_rows.len());
            let mut label_pool = Vec::<Vec<Label>>::new();
            let mut target_pool = Vec::<Vec<u32>>::new();
            let mut weight_id_pool = Vec::<Vec<u32>>::new();
            let mut row_components = Vec::<(u32, u32, u32)>::with_capacity(transition_rows.len());

            for row in &transition_rows {
                let mut label_hasher = FxHasher::default();
                let mut target_hasher = FxHasher::default();
                let mut weight_hasher = FxHasher::default();
                row.len().hash(&mut label_hasher);
                row.len().hash(&mut target_hasher);
                row.len().hash(&mut weight_hasher);
                for &(label, target, weight) in row {
                    label.hash(&mut label_hasher);
                    target.hash(&mut target_hasher);
                    weight.hash(&mut weight_hasher);
                }

                let label_hash = label_hasher.finish();
                let label_id = label_ids
                    .get(&label_hash)
                    .and_then(|candidates| {
                        candidates.iter().copied().find(|&candidate| {
                            let existing = &label_pool[candidate as usize];
                            existing.len() == row.len()
                                && existing
                                    .iter()
                                    .zip(row)
                                    .all(|(&value, &(label, _, _))| value == label)
                        })
                    })
                    .unwrap_or_else(|| {
                        let id = label_pool.len() as u32;
                        label_pool.push(row.iter().map(|&(label, _, _)| label).collect());
                        label_ids.entry(label_hash).or_default().push(id);
                        id
                    });

                let target_hash = target_hasher.finish();
                let target_id = target_ids
                    .get(&target_hash)
                    .and_then(|candidates| {
                        candidates.iter().copied().find(|&candidate| {
                            let existing = &target_pool[candidate as usize];
                            existing.len() == row.len()
                                && existing
                                    .iter()
                                    .zip(row)
                                    .all(|(&value, &(_, target, _))| value == target)
                        })
                    })
                    .unwrap_or_else(|| {
                        let id = target_pool.len() as u32;
                        target_pool.push(row.iter().map(|&(_, target, _)| target).collect());
                        target_ids.entry(target_hash).or_default().push(id);
                        id
                    });

                let weight_hash = weight_hasher.finish();
                let weight_id = weight_ids
                    .get(&weight_hash)
                    .and_then(|candidates| {
                        candidates.iter().copied().find(|&candidate| {
                            let existing = &weight_id_pool[candidate as usize];
                            existing.len() == row.len()
                                && existing
                                    .iter()
                                    .zip(row)
                                    .all(|(&value, &(_, _, weight))| value == weight)
                        })
                    })
                    .unwrap_or_else(|| {
                        let id = weight_id_pool.len() as u32;
                        weight_id_pool.push(row.iter().map(|&(_, _, weight)| weight).collect());
                        weight_ids.entry(weight_hash).or_default().push(id);
                        id
                    });
                row_components.push((label_id, target_id, weight_id));
            }

            put_var_u32(&mut out, label_pool.len() as u32);
            for labels in &label_pool {
                body.clear();
                put_var_u32(&mut body, labels.len() as u32);
                let mut previous = 0i64;
                for &label in labels {
                    let label64 = label as i64;
                    put_var_i64(&mut body, label64 - previous);
                    previous = label64;
                }
                put_var_u32(&mut out, body.len() as u32);
                out.extend_from_slice(&body);
            }

            let mut encode_u32_pool = |pool: &[Vec<u32>]| {
                put_var_u32(&mut out, pool.len() as u32);
                if pool.len() >= 1_024 && rayon::current_num_threads() > 1 {
                    let encoded = pool
                        .par_iter()
                        .map(|sequence| {
                            let absolute_len = sequence
                                .iter()
                                .map(|&value| var_u64_len(value as u64))
                                .sum::<usize>();
                            let mut previous = 0i64;
                            let delta_len = sequence
                                .iter()
                                .map(|&value| {
                                    let value = value as i64;
                                    let len = var_u64_len(zigzag_i64(value - previous));
                                    previous = value;
                                    len
                                })
                                .sum::<usize>();
                            let use_delta = delta_len < absolute_len;
                            let mut encoded = Vec::with_capacity(
                                1 + var_u64_len(sequence.len() as u64)
                                    + absolute_len.min(delta_len),
                            );
                            encoded.push(u8::from(use_delta));
                            put_var_u32(&mut encoded, sequence.len() as u32);
                            if use_delta {
                                let mut previous = 0i64;
                                for &value in sequence {
                                    let value = value as i64;
                                    put_var_i64(&mut encoded, value - previous);
                                    previous = value;
                                }
                            } else {
                                for &value in sequence {
                                    put_var_u32(&mut encoded, value);
                                }
                            }
                            encoded
                        })
                        .collect::<Vec<_>>();
                    for encoded in encoded {
                        put_var_u32(&mut out, encoded.len() as u32);
                        out.extend_from_slice(&encoded);
                    }
                    return;
                }
                for sequence in pool {
                    body.clear();
                    let absolute_len = sequence
                        .iter()
                        .map(|&value| var_u64_len(value as u64))
                        .sum::<usize>();
                    let mut previous = 0i64;
                    let delta_len = sequence
                        .iter()
                        .map(|&value| {
                            let value = value as i64;
                            let len = var_u64_len(zigzag_i64(value - previous));
                            previous = value;
                            len
                        })
                        .sum::<usize>();
                    let use_delta = delta_len < absolute_len;
                    body.push(u8::from(use_delta));
                    put_var_u32(&mut body, sequence.len() as u32);
                    if use_delta {
                        let mut previous = 0i64;
                        for &value in sequence {
                            let value = value as i64;
                            put_var_i64(&mut body, value - previous);
                            previous = value;
                        }
                    } else {
                        for &value in sequence {
                            put_var_u32(&mut body, value);
                        }
                    }
                    put_var_u32(&mut out, body.len() as u32);
                    out.extend_from_slice(&body);
                }
            };
            encode_u32_pool(&target_pool);
            encode_u32_pool(&weight_id_pool);

            put_var_u32(&mut out, row_components.len() as u32);
            for &(labels, targets, weights) in &row_components {
                put_var_u32(&mut out, labels);
                put_var_u32(&mut out, targets);
                put_var_u32(&mut out, weights);
            }
        } else {
            put_var_u32(&mut out, transition_rows.len() as u32);
            for row in &transition_rows {
                body.clear();
                put_var_u32(&mut body, row.len() as u32);
                let mut previous_label = 0i64;
                for &(label, target, weight) in row {
                    let label64 = label as i64;
                    put_var_i64(&mut body, label64 - previous_label);
                    previous_label = label64;
                    put_var_u32(&mut body, target);
                    put_var_u32(&mut body, weight);
                }
                put_var_u32(&mut out, body.len() as u32);
                out.extend_from_slice(&body);
            }
        }
        let rows_end = out.len();
        let row_encode_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        put_var_u32(&mut out, state_rows.len() as u32);
        for &(row, final_weight) in &state_rows {
            put_var_u32(&mut out, row);
            put_var_u32(
                &mut out,
                final_weight.map_or(0, |weight| weight.saturating_add(1)),
            );
        }
        let state_encode_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile {
            let weight_entries: usize = weight_pool.iter().map(|w| w.entries.len()).sum();
            let token_ranges = token_set_range_count;
            eprintln!(
                "[glrmask/profile][dwa_packed_phases] pooled_ms={pooled_ms:.3} token_set_materialize_ms={token_set_materialize_ms:.3} token_set_sort_remap_ms={token_set_sort_remap_ms:.3} geometry_ms={geometry_ms:.3} token_set_encode_ms={token_set_encode_ms:.3} geometry_encode_ms={geometry_encode_ms:.3} weight_encode_ms={weight_encode_ms:.3} row_encode_ms={row_encode_ms:.3} state_encode_ms={state_encode_ms:.3} total_ms={:.3}",
                total_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
            eprintln!(
                "[glrmask/profile][dwa_packed] bytes={} token_set_bytes={} weight_geometry_bytes={} weight_bytes={} row_bytes={} state_ref_bytes={} states={} unique_rows={} transitions={} weights={} weight_geometries={} weight_entries={} token_sets={} token_ranges={}",
                out.len(),
                token_sets_end - 4,
                weight_geometries_end - token_sets_end,
                weights_end - weight_geometries_end,
                rows_end - weights_end,
                out.len() - rows_end,
                state_rows.len(),
                transition_rows.len(),
                transition_count,
                weight_pool.len(),
                weight_geometries.len(),
                weight_entries,
                token_set_count,
                token_ranges,
            );
        }
        out
    }

    fn from_packed_bytes(input: &[u8]) -> Result<Self, String> {
        PACKED_DWA_TOKEN_SET_INVENTORY.with(|slot| *slot.borrow_mut() = None);
        if !input.starts_with(b"DWP6") {
            return Err("invalid packed DWA header".to_owned());
        }
        let mut pos = 4usize;
        let start_state = take_var_u32(input, &mut pos)?;
        let profile = std::env::var_os("GLRMASK_PROFILE_DWA_SERIALIZATION").is_some();
        let token_sets_started = profile.then(std::time::Instant::now);

        let token_set_count = take_var_u32(input, &mut pos)? as usize;
        let token_set_chunk_count = take_var_u32(input, &mut pos)? as usize;
        let token_set_chunks = take_length_prefixed_slices(
            input,
            &mut pos,
            token_set_chunk_count,
            "token-set chunk",
        )?;
        let decoded_chunks: Vec<_> = if token_set_count >= 1024 && rayon::current_num_threads() > 1 {
            token_set_chunks
                .par_iter()
                .map(|body| decode_packed_token_set_chunk(body))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            token_set_chunks
                .iter()
                .map(|body| decode_packed_token_set_chunk(body))
                .collect::<Result<Vec<_>, _>>()?
        };
        let decoded_token_sets = decoded_chunks.into_iter().flatten().collect::<Vec<_>>();
        let mut ts_pool = Vec::with_capacity(decoded_token_sets.len());
        let mut token_set_word_spans = Vec::with_capacity(decoded_token_sets.len());
        for (tokens, word_spans) in decoded_token_sets {
            ts_pool.push(tokens);
            token_set_word_spans.push(word_spans);
        }
        if ts_pool.len() != token_set_count {
            return Err("invalid packed DWA token-set count".to_owned());
        }
        let token_sets_ms = token_sets_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let geometries_started = profile.then(std::time::Instant::now);

        let geometry_count = take_var_u32(input, &mut pos)? as usize;
        let geometry_bodies =
            take_length_prefixed_slices(input, &mut pos, geometry_count, "weight geometry")?;
        let geometries: Vec<_> = if geometry_count >= 1024 && rayon::current_num_threads() > 1 {
            geometry_bodies
                .par_iter()
                .map(|body| decode_packed_weight_geometry(body))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            geometry_bodies
                .iter()
                .map(|body| decode_packed_weight_geometry(body))
                .collect::<Result<Vec<_>, _>>()?
        };
        let geometries_ms = geometries_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let weights_started = profile.then(std::time::Instant::now);

        let weight_count = take_var_u32(input, &mut pos)? as usize;
        let weight_bodies =
            take_length_prefixed_slices(input, &mut pos, weight_count, "weight")?;
        let w_pool: Vec<_> = if weight_count >= 1024 && rayon::current_num_threads() > 1 {
            weight_bodies
                .par_iter()
                .map(|body| decode_packed_weight(body, &ts_pool, &geometries))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            weight_bodies
                .iter()
                .map(|body| decode_packed_weight(body, &ts_pool, &geometries))
                .collect::<Result<Vec<_>, _>>()?
        };
        let w_pool = Arc::<[Weight]>::from(w_pool);
        let weights_ms = weights_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let rows_started = profile.then(std::time::Instant::now);
        let row_mode = *input
            .get(pos)
            .ok_or_else(|| "truncated packed DWA row mode".to_owned())?;
        pos += 1;
        let (transition_rows, state_count_pos, transition_weight_used) = match row_mode {
            0 => {
                let row_count = take_var_u32(input, &mut pos)? as usize;
                let row_bodies = take_length_prefixed_slices(
                    input,
                    &mut pos,
                    row_count,
                    "transition row",
                )?;
                let state_count_pos = pos;
                let state_count = take_var_u32(input, &mut pos)? as usize;
                let rows: Vec<_> =
                    if row_count >= 1024 && rayon::current_num_threads() > 1 {
                        row_bodies
                            .par_iter()
                            .map(|body| decode_packed_transition_row(body, &w_pool, state_count))
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        row_bodies
                            .iter()
                            .map(|body| decode_packed_transition_row(body, &w_pool, state_count))
                            .collect::<Result<Vec<_>, _>>()?
                    };
                (
                    rows.into_iter()
                        .map(|row| DwaTransitionMap::Shared(Arc::new(row)))
                        .collect(),
                    state_count_pos,
                    None,
                )
            }
            1 => {
                let label_count = take_var_u32(input, &mut pos)? as usize;
                let label_bodies = take_length_prefixed_slices(
                    input,
                    &mut pos,
                    label_count,
                    "label sequence",
                )?;
                let label_pool: Vec<_> = label_bodies
                    .iter()
                    .map(|body| decode_packed_label_sequence(body))
                    .collect::<Result<Vec<_>, _>>()?;

                let target_count = take_var_u32(input, &mut pos)? as usize;
                let target_bodies = take_length_prefixed_slices(
                    input,
                    &mut pos,
                    target_count,
                    "target sequence",
                )?;
                let weight_id_count = take_var_u32(input, &mut pos)? as usize;
                let weight_id_bodies = take_length_prefixed_slices(
                    input,
                    &mut pos,
                    weight_id_count,
                    "weight-id sequence",
                )?;
                let decode_targets = || {
                    if target_count >= 1024 && rayon::current_num_threads() > 1 {
                        target_bodies
                            .par_iter()
                            .map(|body| decode_packed_u32_sequence(body, "target"))
                            .collect::<Result<Vec<_>, _>>()
                    } else {
                        target_bodies
                            .iter()
                            .map(|body| decode_packed_u32_sequence(body, "target"))
                            .collect::<Result<Vec<_>, _>>()
                    }
                };
                let decode_weight_ids = || {
                    if weight_id_count >= 1024 && rayon::current_num_threads() > 1 {
                        weight_id_bodies
                            .par_iter()
                            .map(|body| decode_packed_u32_sequence(body, "weight-id"))
                            .collect::<Result<Vec<_>, _>>()
                    } else {
                        weight_id_bodies
                            .iter()
                            .map(|body| decode_packed_u32_sequence(body, "weight-id"))
                            .collect::<Result<Vec<_>, _>>()
                    }
                };
                let (target_pool, weight_id_pool) = if rayon::current_num_threads() > 1 {
                    let (targets, weights) = rayon::join(decode_targets, decode_weight_ids);
                    (targets?, weights?)
                } else {
                    (decode_targets()?, decode_weight_ids()?)
                };
                let mut transition_weight_used = vec![false; w_pool.len()];
                for sequence in &weight_id_pool {
                    for &weight in sequence {
                        let Some(used) = transition_weight_used.get_mut(weight as usize) else {
                            return Err("invalid packed DWA weight index".to_owned());
                        };
                        *used = true;
                    }
                }

                let row_count = take_var_u32(input, &mut pos)? as usize;
                let mut row_components = Vec::with_capacity(row_count);
                for _ in 0..row_count {
                    let labels = take_var_u32(input, &mut pos)? as usize;
                    let targets = take_var_u32(input, &mut pos)? as usize;
                    let weights = take_var_u32(input, &mut pos)? as usize;
                    if labels >= label_pool.len()
                        || targets >= target_pool.len()
                        || weights >= weight_id_pool.len()
                    {
                        return Err("invalid packed DWA row-component index".to_owned());
                    }
                    row_components.push((labels, targets, weights));
                }
                let state_count_pos = pos;
                let state_count = take_var_u32(input, &mut pos)? as usize;
                for labels in &label_pool {
                    if labels.windows(2).any(|pair| pair[0] >= pair[1]) {
                        return Err(
                            "packed DWA transition labels are not strictly increasing".to_owned(),
                        );
                    }
                }
                for targets in &target_pool {
                    if targets.iter().any(|&target| target as usize >= state_count) {
                        return Err("invalid packed DWA target state".to_owned());
                    }
                }
                for &(label_id, target_id, weight_id) in &row_components {
                    let labels = &label_pool[label_id];
                    let targets = &target_pool[target_id];
                    let weights = &weight_id_pool[weight_id];
                    if labels.len() != targets.len() || labels.len() != weights.len() {
                        return Err("mismatched packed DWA row-component lengths".to_owned());
                    }
                }
                let pool = Arc::new(PackedDwaRowPool {
                    labels: label_pool,
                    targets: target_pool,
                    weight_ids: weight_id_pool,
                    weights: Arc::clone(&w_pool),
                });
                let rows: Vec<DwaTransitionMap> = row_components
                    .into_iter()
                    .map(|(label_id, target_id, weight_id)| {
                        DwaTransitionMap::Packed(Arc::new(PackedDwaTransitionRow {
                            pool: Arc::clone(&pool),
                            label_id,
                            target_id,
                            weight_id,
                            materialized: OnceLock::new(),
                        }))
                    })
                    .collect();
                (rows, state_count_pos, Some(transition_weight_used))
            }
            _ => return Err("invalid packed DWA row mode".to_owned()),
        };
        let rows_ms = rows_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        // Rewind to the state-count varint and decode compact row/final refs.
        pos = state_count_pos;
        let state_count = take_var_u32(input, &mut pos)? as usize;
        let state_refs_started = profile.then(std::time::Instant::now);
        let mut state_refs = Vec::with_capacity(state_count);
        let mut final_weight_used = vec![false; w_pool.len()];
        for _ in 0..state_count {
            let row = take_var_u32(input, &mut pos)? as usize;
            if row >= transition_rows.len() {
                return Err("invalid packed DWA transition-row index".to_owned());
            }
            let final_encoded = take_var_u32(input, &mut pos)?;
            let final_weight = if final_encoded == 0 {
                None
            } else {
                let index = (final_encoded - 1) as usize;
                if index >= w_pool.len() {
                    return Err("invalid packed DWA final-weight index".to_owned());
                }
                final_weight_used[index] = true;
                Some(index)
            };
            state_refs.push((row, final_weight));
        }
        if pos != input.len() {
            return Err("trailing bytes in packed DWA".to_owned());
        }
        let build_inventory = || {
            let transition_weight_used = transition_weight_used.as_deref()?;
            let mut transition_token_used = vec![false; ts_pool.len()];
            let mut final_token_used = vec![false; ts_pool.len()];
            // The Weight bodies were already fully validated while w_pool was
            // decoded.  Rescan only their compact token-set IDs here instead
            // of retaining per-Weight ID vectors or walking reconstructed
            // RangeMapBlaze values and hashing the same pointers repeatedly.
            for (weight_index, body) in weight_bodies.iter().enumerate() {
                let mark_transition = transition_weight_used[weight_index];
                let mark_final = final_weight_used[weight_index];
                if !mark_transition && !mark_final {
                    continue;
                }
                let mut body_pos = 0usize;
                let tag = body[body_pos];
                body_pos += 1;
                if tag == 1 {
                    continue;
                }
                debug_assert_eq!(tag, 0);
                let geometry_index = take_var_u32(body, &mut body_pos)
                    .expect("packed Weight body was validated during decode")
                    as usize;
                for _ in &geometries[geometry_index] {
                    let token_set = take_var_u32(body, &mut body_pos)
                        .expect("packed Weight body was validated during decode")
                        as usize;
                    if mark_transition {
                        transition_token_used[token_set] = true;
                    }
                    if mark_final {
                        final_token_used[token_set] = true;
                    }
                }
            }
            let transition_set_count = transition_token_used.iter().filter(|&&used| used).count();
            let final_set_count = final_token_used.iter().filter(|&&used| used).count();
            let mut transition_sets = FxHashMap::default();
            transition_sets.reserve(transition_set_count);
            let mut transition_word_spans = FxHashMap::default();
            transition_word_spans.reserve(transition_set_count);
            let mut final_sets = FxHashMap::default();
            final_sets.reserve(final_set_count);
            for (index, token_set) in ts_pool.iter().enumerate() {
                let key = Arc::as_ptr(token_set) as usize;
                if transition_token_used[index] {
                    transition_sets.insert(key, Arc::clone(token_set));
                    transition_word_spans.insert(key, token_set_word_spans[index]);
                }
                if final_token_used[index] {
                    final_sets.insert(key, Arc::clone(token_set));
                }
            }
            Some(PackedDwaTokenSetInventory {
                transition_sets,
                final_sets,
                transition_word_spans,
            })
        };
        let build_state = |&(row, final_weight): &(usize, Option<usize>)| DWAState {
            transitions: transition_rows[row].clone(),
            final_weight: final_weight.map(|index| w_pool[index].clone()),
        };
        let build_states = || {
            if state_count >= 1024 && rayon::current_num_threads() > 1 {
                state_refs.par_iter().map(build_state).collect()
            } else {
                state_refs.iter().map(build_state).collect()
            }
        };
        let (states, inventory): (Vec<_>, _) = if transition_weight_used.is_some()
            && state_count >= 1024
            && rayon::current_num_threads() > 1
        {
            rayon::join(build_states, build_inventory)
        } else {
            (build_states(), build_inventory())
        };
        if let Some(inventory) = inventory {
            PACKED_DWA_TOKEN_SET_INVENTORY.with(|slot| {
                *slot.borrow_mut() = Some(inventory);
            });
        }
        let state_refs_ms = state_refs_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if start_state as usize >= states.len() && !states.is_empty() {
            return Err("invalid packed DWA start state".to_owned());
        }
        if profile {
            eprintln!(
                "[glrmask/profile][dwa_packed_load] bytes={} token_sets_ms={token_sets_ms:.3} geometries_ms={geometries_ms:.3} weights_ms={weights_ms:.3} rows_ms={rows_ms:.3} state_refs_ms={state_refs_ms:.3}",
                input.len(),
            );
        }
        Ok(Self {
            states,
            start_state,
            shared_transition_rows: true,
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        })
    }
}

impl Serialize for DWA {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if external_serde_enabled() {
            return 0u8.serialize(serializer);
        }
        if packed_serde_enabled() {
            return self.to_packed_bytes().serialize(serializer);
        }
        let (token_set_pool, weight_pool, states) = self.pooled_parts();

        let serde_repr = DWASerde {
            token_set_pool,
            weight_pool,
            states,
            start_state: self.start_state,
        };
        if std::env::var_os("GLRMASK_PROFILE_DWA_SERIALIZATION").is_some() {
            let transition_count: usize = serde_repr.states.iter().map(|s| s.transitions.len()).sum();
            let weight_entries: usize = serde_repr.weight_pool.iter().map(|w| w.entries.len()).sum();
            let token_ranges: usize = serde_repr.token_set_pool.iter().map(Vec::len).sum();
            eprintln!(
                "[glrmask/profile][dwa_serde] states={} transitions={} weights={} weight_entries={} token_sets={} token_ranges={}",
                serde_repr.states.len(),
                transition_count,
                serde_repr.weight_pool.len(),
                weight_entries,
                serde_repr.token_set_pool.len(),
                token_ranges,
            );
        }
        serde_repr.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for DWA {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if external_serde_enabled() {
            let marker = u8::deserialize(deserializer)?;
            if marker != 0 {
                return Err(serde::de::Error::custom("invalid external DWA placeholder"));
            }
            return Ok(Self::new(0, 0));
        }
        if packed_serde_enabled() {
            let bytes = Vec::<u8>::deserialize(deserializer)?;
            return Self::from_packed_bytes(&bytes).map_err(serde::de::Error::custom);
        }
        let serde_repr = DWASerde::deserialize(deserializer)?;

        // Reconstruct token set pool (shared Arcs)
        let ts_pool: Vec<std::sync::Arc<RangeSetBlaze<u32>>> = serde_repr
            .token_set_pool
            .into_iter()
            .map(|encoded| {
                // Serialized token sets come directly from RangeSetBlaze::ranges(),
                // hence they are already sorted and disjoint.  Bypass the generic
                // normalization path when rebuilding millions of ranges.
                let rs = RangeSetBlaze::from_sorted_disjoint(CheckSortedDisjoint::new(
                    encoded.into_iter().map(|[s, e]| s..=e),
                ));
                shared_rangeset(rs)
            })
            .collect();

        // Reconstruct weight pool
        let w_pool: Vec<Weight> = serde_repr
            .weight_pool
            .into_iter()
            .map(|entry| {
                if entry.all {
                    return Weight::all();
                }
                if entry.entries.is_empty() {
                    return Weight::empty();
                }
                let ranges: Vec<_> = entry.entries.into_iter().map(|(start, end, ts_idx)| {
                    let tokens = ts_pool
                        .get(ts_idx as usize)
                        .cloned()
                        .unwrap_or_else(|| std::sync::Arc::new(RangeSetBlaze::new()));
                    (start..=end, tokens)
                }).collect();
                let map = RangeMapBlaze::from_sorted_disjoint_map(
                    CheckSortedDisjointMap::new(
                        ranges.iter().map(|(range, tokens)| (range.clone(), tokens)),
                    ),
                );
                finalize_weight_map(map)
            })
            .collect();

        // Reconstruct DWA states
        let states = serde_repr
            .states
            .into_iter()
            .map(|s| {
                let transitions = s
                    .transitions
                    .into_iter()
                    .map(|(label, target, weight_idx)| {
                        let weight = w_pool
                            .get(weight_idx as usize)
                            .cloned()
                            .unwrap_or_else(Weight::empty);
                        (label, (target, weight))
                    })
                    .collect::<BTreeMap<_, _>>()
                    .into();
                let final_weight = s.final_weight.map(|idx| {
                    w_pool
                        .get(idx as usize)
                        .cloned()
                        .unwrap_or_else(Weight::empty)
                });
                DWAState {
                    transitions,
                    final_weight,
                }
            })
            .collect();

        Ok(DWA {
            states,
            start_state: serde_repr.start_state,
            shared_transition_rows: false,
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        })
    }
}

impl DWA {
    pub fn new(_num_tsids: u32, _max_token: u32) -> Self {
        Self {
            states: vec![DWAState::default()],
            start_state: 0,
            shared_transition_rows: false,
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        }
    }

    #[inline]
    fn invalidate_graph_caches(&mut self) {
        let _ = self.transition_count_cache.take();
        let _ = self.acyclic_cache.take();
    }

    #[inline]
    pub fn states(&self) -> &[DWAState] {
        &self.states
    }

    #[inline]
    pub fn states_mut(&mut self) -> &mut Vec<DWAState> {
        self.invalidate_graph_caches();
        &mut self.states
    }

    #[inline]
    pub fn start_state(&self) -> u32 {
        self.start_state
    }

    pub fn from_parts(states: Vec<DWAState>, start_state: u32) -> Self {
        Self {
            states,
            start_state,
            shared_transition_rows: false,
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        }
    }

    pub fn has_shared_transition_rows(&self) -> bool {
        self.shared_transition_rows
    }

    /// Canonicalize exact duplicate transition rows without changing state ids
    /// or graph topology.  Weight-coordinate remapping can legitimately turn
    /// previously shared rows back into `Owned` rows because `DerefMut` must
    /// break the Arc identity before mutating a Weight.  Once all remapping is
    /// finished, recovering row sharing is semantics-neutral and lets both the
    /// runtime fast-row builder and the artifact encoder operate once per
    /// unique row rather than once per state.
    pub fn share_exact_transition_rows_owned(mut self) -> Self {
        if self.shared_transition_rows || self.states.len() < 2 {
            return self;
        }

        let hash_row = |state: &DWAState| {
            let mut hasher = FxHasher::default();
            state.transitions.len().hash(&mut hasher);
            for (label, (target, weight)) in state.transitions.iter() {
                label.hash(&mut hasher);
                target.hash(&mut hasher);
                weight.ptr_key().hash(&mut hasher);
            }
            hasher.finish()
        };
        let hashes = if self.states.len() >= 4_096 && rayon::current_num_threads() > 1 {
            self.states.par_iter().map(hash_row).collect::<Vec<_>>()
        } else {
            self.states.iter().map(hash_row).collect::<Vec<_>>()
        };

        let mut buckets =
            FxHashMap::<u64, Vec<Arc<BTreeMap<Label, (u32, Weight)>>>>::default();
        buckets.reserve(self.states.len().min(32_768));
        for (state, hash) in self.states.iter_mut().zip(hashes) {
            let transitions = match std::mem::take(&mut state.transitions) {
                DwaTransitionMap::Owned(transitions) => transitions,
                DwaTransitionMap::Shared(transitions) => {
                    state.transitions = DwaTransitionMap::Shared(transitions);
                    continue;
                }
                DwaTransitionMap::Packed(transitions) => {
                    state.transitions = DwaTransitionMap::Packed(transitions);
                    continue;
                }
            };
            let bucket = buckets.entry(hash).or_default();
            let shared = bucket
                .iter()
                .find(|candidate| candidate.as_ref() == &transitions)
                .cloned()
                .unwrap_or_else(|| {
                    let shared = Arc::new(transitions);
                    bucket.push(Arc::clone(&shared));
                    shared
                });
            state.transitions = DwaTransitionMap::Shared(shared);
        }
        self.shared_transition_rows = true;
        self
    }

    /// Merge states whose complete local DWA representation is already
    /// identical: same final weight and the same label -> (target, weight)
    /// transition row. This is deliberately much cheaper than automaton
    /// minimization: it performs no weight algebra and no partition refinement.
    ///
    /// Replacing every reference to one exact duplicate by the other preserves
    /// the weighted language even for cyclic DWAs. A later invocation can expose
    /// additional duplicates after target ids have been remapped.
    pub fn merge_exact_duplicate_states_owned(self) -> Self {
        if self.states.len() < 2 {
            return self;
        }

        let hash_state = |state: &DWAState| {
            let mut hasher = FxHasher::default();
            // Compiler-produced Weights are globally interned, so pointer
            // identity is an extremely cheap structural fingerprint. Keep the
            // full DWAState equality check below as the authority.
            state
                .final_weight
                .as_ref()
                .map_or(0usize, Weight::ptr_key)
                .hash(&mut hasher);
            state.transitions.len().hash(&mut hasher);
            for (label, (target, weight)) in &state.transitions {
                label.hash(&mut hasher);
                target.hash(&mut hasher);
                weight.ptr_key().hash(&mut hasher);
            }
            hasher.finish()
        };
        let hashes = if self.states.len() >= 4_096 && rayon::current_num_threads() > 1 {
            self.states.par_iter().map(hash_state).collect::<Vec<_>>()
        } else {
            self.states.iter().map(hash_state).collect::<Vec<_>>()
        };

        let mut hash_buckets = FxHashMap::<u64, Vec<u32>>::default();
        let mut canonical_old = Vec::<u32>::with_capacity(self.states.len());
        let mut representatives = Vec::<u32>::new();

        for (state_id, (state, &hash)) in self.states.iter().zip(&hashes).enumerate() {
            let bucket = hash_buckets.entry(hash).or_default();
            let canonical = bucket
                .iter()
                .copied()
                .find(|&candidate| self.states[candidate as usize] == *state)
                .unwrap_or_else(|| {
                    let state_id = state_id as u32;
                    bucket.push(state_id);
                    representatives.push(state_id);
                    state_id
                });
            canonical_old.push(canonical);
        }

        if representatives.len() == self.states.len() {
            return self;
        }

        let mut old_to_new = vec![u32::MAX; self.states.len()];
        for (new_id, &old_id) in representatives.iter().enumerate() {
            old_to_new[old_id as usize] = new_id as u32;
        }
        for (old_id, &canonical) in canonical_old.iter().enumerate() {
            old_to_new[old_id] = old_to_new[canonical as usize];
        }

        let start_state = old_to_new[self.start_state as usize];
        let mut states = Vec::with_capacity(representatives.len());
        let mut row_buckets =
            FxHashMap::<u64, Vec<Arc<BTreeMap<Label, (u32, Weight)>>>>::default();
        for (old_id, mut state) in self.states.into_iter().enumerate() {
            if canonical_old[old_id] != old_id as u32 {
                continue;
            }

            // We already have to touch every surviving transition to remap its
            // target. Hash the remapped row in the same pass, then share exact
            // duplicate rows independently of final_weight. Large parser DWAs
            // commonly have far fewer unique outgoing rows than states.
            let mut row_hasher = FxHasher::default();
            state.transitions.len().hash(&mut row_hasher);
            for (label, (target, weight)) in state.transitions.iter_mut() {
                *target = old_to_new[*target as usize];
                label.hash(&mut row_hasher);
                target.hash(&mut row_hasher);
                weight.ptr_key().hash(&mut row_hasher);
            }
            let row_hash = row_hasher.finish();
            let transitions = match std::mem::take(&mut state.transitions) {
                DwaTransitionMap::Owned(transitions) => transitions,
                DwaTransitionMap::Shared(transitions) => Arc::try_unwrap(transitions)
                    .unwrap_or_else(|shared| shared.as_ref().clone()),
                DwaTransitionMap::Packed(transitions) => transitions.materialized().clone(),
            };
            let bucket = row_buckets.entry(row_hash).or_default();
            let shared = bucket
                .iter()
                .find(|candidate| candidate.as_ref() == &transitions)
                .cloned()
                .unwrap_or_else(|| {
                    let shared = Arc::new(transitions);
                    bucket.push(Arc::clone(&shared));
                    shared
                });
            state.transitions = DwaTransitionMap::Shared(shared);
            states.push(state);
        }

        Self {
            states,
            start_state,
            shared_transition_rows: true,
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        }
    }

    pub fn set_start_state(&mut self, state: u32) {
        self.start_state = state;
    }

    pub fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        self.states.push(DWAState::default());
        id
    }

    pub fn num_states(&self) -> u32 {
        self.states.len() as u32
    }

    pub fn num_transitions(&self) -> usize {
        *self.transition_count_cache.get_or_init(|| {
            self.states
                .iter()
                .map(|state| state.transitions.len())
                .sum()
        })
    }

    pub fn stats(&self) -> DwaStats {
        let mut transition_pairs = 0usize;
        let mut dsts = BTreeSet::new();
        for state in &self.states {
            dsts.clear();
            for (dst, _) in state.transitions.values() {
                dsts.insert(*dst);
            }
            transition_pairs += dsts.len();
        }

        let mut seen_weight_ptrs = BTreeSet::new();
        let mut seen_rangeset_ptrs = BTreeSet::new();
        let mut total_outer_ranges = 0usize;
        let mut total_inner_ranges = 0usize;

        let mut process_weight = |weight: &Weight| {
            let weight_ptr = weight.ptr_key();
            if seen_weight_ptrs.insert(weight_ptr) {
                total_outer_ranges += weight.raw_range_values().count();
            }
            for (_, tokens) in weight.raw_range_values() {
                let token_ptr = std::sync::Arc::as_ptr(tokens) as usize;
                if seen_rangeset_ptrs.insert(token_ptr) {
                    total_inner_ranges += tokens.ranges().count();
                }
            }
        };

        for state in &self.states {
            if let Some(final_weight) = &state.final_weight {
                process_weight(final_weight);
            }
            for (_, weight) in state.transitions.values() {
                process_weight(weight);
            }
        }

        DwaStats {
            states: self.states.len(),
            transitions: self.num_transitions(),
            transition_pairs,
            interned_ranges: total_outer_ranges + total_inner_ranges,
        }
    }

    pub fn set_final_weight(&mut self, state: u32, weight: Weight) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.final_weight = Some(weight);
        }
    }

    pub fn add_transition(&mut self, from: u32, label: Label, to: u32, weight: Weight) {
        self.invalidate_graph_caches();
        if let Some(entry) = self.states.get_mut(from as usize) {
            entry.transitions.insert(label, (to, weight));
        }
    }

    pub fn eval_word(&self, word: &[Label]) -> Weight {
        let mut state = self.start_state;
        let mut weight = Weight::all();
        for &label in word {
            let Some((next, edge_weight)) = self.states[state as usize].transitions.get(&label) else {
                return Weight::empty();
            };
            weight = weight.intersection(edge_weight);
            state = *next;
        }
        match self.states.get(state as usize).and_then(|state| state.final_weight.as_ref()) {
            Some(final_weight) => weight.intersection(final_weight),
            None => Weight::empty(),
        }
    }

    /// Clip all weights in the DWA so token sets contain only `0..=max_token`.
    pub fn clip_weights(&mut self, max_token: u32) {
        for state in &mut self.states {
            if let Some(fw) = &mut state.final_weight {
                fw.clip_tokens(max_token);
                if fw.is_empty() {
                    state.final_weight = None;
                }
            }
            for (_, (_, w)) in &mut state.transitions {
                w.clip_tokens(max_token);
            }
        }
    }

    pub fn labels(&self) -> Vec<Label> {
        self.states
            .iter()
            .flat_map(|state| state.transitions.keys().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn is_acyclic(&self) -> bool {
        *self.acyclic_cache.get_or_init(|| self.compute_is_acyclic())
    }

    fn compute_is_acyclic(&self) -> bool {
        let num_states = self.states.len();
        let mut indegree = vec![0u32; num_states];
        for state in &self.states {
            for &(target, _) in state.transitions.values() {
                if let Some(degree) = indegree.get_mut(target as usize) {
                    *degree += 1;
                }
            }
        }
        let mut queue = std::collections::VecDeque::new();
        for (state, &degree) in indegree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(state);
            }
        }
        let mut visited = 0usize;
        while let Some(state) = queue.pop_front() {
            visited += 1;
            for &(target, _) in self.states[state].transitions.values() {
                let Some(degree) = indegree.get_mut(target as usize) else {
                    continue;
                };
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(target as usize);
                }
            }
        }
        visited == num_states
    }

    /// Convert this DWA to an NWA representation.
    pub fn to_nwa(&self) -> super::nwa::NWA {
        use super::nwa::{NWA, NWAState};
        let mut nwa = NWA::from_parts(
            Vec::with_capacity(self.states.len()),
            vec![self.start_state],
        );
        for state in &self.states {
            let mut nwa_state = NWAState::default();
            nwa_state.final_weight = state.final_weight.clone();
            for (&label, (target, weight)) in &state.transitions {
                nwa_state
                    .transitions
                    .entry(label)
                    .or_default()
                    .push((*target, weight.clone()));
            }
            nwa.states_mut().push(nwa_state);
        }
        nwa
    }
}

fn fmt_dwa_states(
    dwa: &DWA,
    f: &mut std::fmt::Formatter<'_>,
    label_fn: &dyn Fn(Label) -> String,
    weight_fn: &dyn Fn(&Weight) -> String,
) -> std::fmt::Result {
    for (i, state) in dwa.states.iter().enumerate() {
        if state.transitions.is_empty() && state.final_weight.is_none() {
            continue;
        }

        let start_mark = if i as u32 == dwa.start_state { " [START]" } else { "" };
        writeln!(f, "  State {i}{start_mark}")?;

        if let Some(w) = &state.final_weight {
            writeln!(f, "    final: {}", weight_fn(w))?;
        }

        for (label, (tgt, w)) in &state.transitions {
            let lbl = label_fn(*label);
            writeln!(f, "    {lbl} → State {tgt}")?;
            writeln!(f, "      weight: {}", weight_fn(w))?;
        }
    }
    Ok(())
}

impl std::fmt::Display for DWA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DWA: {} states, start=State {}", self.states.len(), self.start_state)?;
        fmt_dwa_states(self, f, &|l| l.to_string(), &|w| format!("{w}"))
    }
}

impl PartialEq for DWA {
    fn eq(&self, other: &Self) -> bool {
        self.start_state == other.start_state && self.states == other.states
    }
}

impl PartialEq for DWAState {
    fn eq(&self, other: &Self) -> bool {
        self.transitions == other.transitions && self.final_weight == other.final_weight
    }
}

#[cfg(test)]
mod cache_tests ;
