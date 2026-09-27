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
