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
