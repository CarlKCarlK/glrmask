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
