use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::{SingletonEpsilonClosures, Tokenizer};
use crate::compiler::stages::equiv_types::ManyToOneIdMap;

use super::super::compat::{FlatDfa, FlatDfaState, TokenizerView};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RawPowersetTargetMode {
    Scalar,
    Direct,
}

#[derive(Debug, Clone, Copy)]
pub enum RefinementDepth {
    Stable,
    Bounded(usize),
}

pub struct BoundedAnalysisView {
    pub tokenizer_view: TokenizerView,
    raw_start_to_view: Vec<u32>,
}

pub struct RelevantPowersetView {
    pub states: Vec<FlatDfaState>,
    pub start_state: usize,
    pub bytes: Vec<u8>,
    pub edge_offsets: Vec<u32>,
    pub edges: Vec<(u8, u32)>,
    pub raw_start_to_view: Arc<[u32]>,
    pub configurations: Arc<[Box<[u32]>]>,
}

#[derive(Clone, Copy, Debug)]
pub struct RelevantPowersetWorkBudget {
    pub max_configurations: usize,
    pub max_edges: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelevantPowersetWork {
    pub configurations: usize,
    pub edges: usize,
}

pub struct PrebuiltSparsePowersetRefinement<'a> {
    pub raw_start_to_view: &'a [u32],
    pub configurations: &'a [Box<[u32]>],
    pub output_class_by_config: &'a [u32],
    pub edge_offsets: &'a [u32],
    pub edges: &'a [(u8, u32)],
}

impl PrebuiltSparsePowersetRefinement<'_> {
    pub fn compute_state_map(
        &self,
        tokenizer: &Tokenizer,
        initial_state_map: Option<&ManyToOneIdMap>,
        depth: RefinementDepth,
    ) -> ManyToOneIdMap {
        compute_state_map_from_prebuilt_sparse_powerset(
            tokenizer,
            initial_state_map,
            depth,
            self.raw_start_to_view,
            self.configurations,
            self.output_class_by_config,
            None,
            self.edge_offsets,
            self.edges,
        )
    }
}

pub fn powerset_output_class_ids(view: &RelevantPowersetView) -> Vec<u32> {
    let mut output_ids = FxHashMap::<(Vec<usize>, Vec<usize>), u32>::default();
    view.states
        .iter()
        .map(|state| {
            let key = (
                state.finalizers.clone(),
                state.possible_future_group_ids.clone(),
            );
            let next = output_ids.len() as u32;
            *output_ids.entry(key).or_insert(next)
        })
        .collect()
}

impl RelevantPowersetView {
    /// Return the powerset-view state corresponding to a raw lexer start.
    ///
    /// Unlike token-bounded analysis, a relevant-powerset view is total over
    /// the raw lexer-state domain. Keep that invariant behind a checked API so
    /// callers cannot accidentally cast the unmapped sentinel into a view ID.
    pub fn view_state_for_raw_start(&self, raw_state: usize) -> usize {
        let state = self
            .raw_start_to_view
            .get(raw_state)
            .copied()
            .expect("raw state is outside relevant-powerset domain");
        assert_ne!(
            state,
            u32::MAX,
            "relevant-powerset view must map every raw lexer state",
        );
        let state = state as usize;
        assert!(
            state < self.states.len(),
            "relevant-powerset raw-start map produced an invalid view state",
        );
        state
    }

    pub fn into_tokenizer_view(self) -> TokenizerView {
        let mut transitions = vec![u32::MAX; self.states.len() * 256];
        for state in 0..self.states.len() {
            let start = self.edge_offsets[state] as usize;
            let end = self.edge_offsets[state + 1] as usize;
            for &(byte, target) in &self.edges[start..end] {
                transitions[state * 256 + byte as usize] = target;
            }
        }
        TokenizerView {
            flat_dfa: FlatDfa {
                states: self.states,
                start_state: self.start_state,
                transitions: Arc::from(transitions),
            },
        }
    }
}

fn mark_relevant_powerset_trie_trajectories(
    view: &RelevantPowersetView,
    dense_transitions: &[u32],
    byte_to_slot: &[u16; 256],
    transition_stride: usize,
    start_states: &[u32],
    trie: &[ByteTrieNode],
    selected: &mut [bool],
    observed_bytes: &mut [[u64; 4]],
) {
    if trie.is_empty() || start_states.is_empty() {
        return;
    }

    let mut node_state_start = vec![0u32; trie.len()];
    let mut node_state_len = vec![0u32; trie.len()];
    let mut product_states = Vec::<u32>::with_capacity(start_states.len());
    product_states.extend_from_slice(start_states);
    node_state_len[0] = start_states.len() as u32;
    for &state in start_states {
        selected[state as usize] = true;
    }

    let mut state_marks = vec![0u32; view.states.len()];
    let mut generation = 0u32;
    for node_index in 0..trie.len() {
        let state_start = node_state_start[node_index] as usize;
        let state_len = node_state_len[node_index] as usize;
        if state_len == 0 {
            continue;
        }

        // The root frontier can contain tens of thousands of states. Route its
        // sparse outgoing edges to matching trie children once per source,
        // rather than probing every root byte from every source.
        if node_index == 0 && trie[node_index].children.len() > 1 {
            let mut child_slot_by_byte = [u16::MAX; 256];
            for (slot, &(byte, _)) in trie[node_index].children.iter().enumerate() {
                child_slot_by_byte[byte as usize] = slot as u16;
            }
            let mut child_frontiers =
                vec![Vec::<u32>::new(); trie[node_index].children.len()];
            for offset in 0..state_len {
                let source = product_states[state_start + offset];
                let edge_start = view.edge_offsets[source as usize] as usize;
                let edge_end = view.edge_offsets[source as usize + 1] as usize;
                for &(byte, target) in &view.edges[edge_start..edge_end] {
                    let slot = child_slot_by_byte[byte as usize];
                    if slot == u16::MAX {
                        continue;
                    }
                    let word = byte as usize / 64;
                    observed_bytes[source as usize][word] |= 1u64 << (byte as usize % 64);
                    selected[target as usize] = true;
                    child_frontiers[slot as usize].push(target);
                }
            }
            for ((_, child), frontier) in trie[node_index]
                .children
                .iter()
                .zip(child_frontiers.iter_mut())
            {
                frontier.sort_unstable();
                frontier.dedup();
                let child_start = product_states.len();
                product_states.extend_from_slice(frontier);
                node_state_start[*child] = child_start as u32;
                node_state_len[*child] = frontier.len() as u32;
            }
            continue;
        }

        for &(token_byte, child) in &trie[node_index].children {
            generation = generation.wrapping_add(1);
            if generation == 0 {
                state_marks.fill(0);
                generation = 1;
            }
            let byte = token_byte;
            let child_start = product_states.len();
            let byte_slot = byte_to_slot[byte as usize];
            if byte_slot == u16::MAX {
                node_state_start[child] = child_start as u32;
                node_state_len[child] = 0;
                continue;
            }
            for offset in 0..state_len {
                let source = product_states[state_start + offset];
                let target = dense_transitions
                    [source as usize * transition_stride + byte_slot as usize];
                if target == u32::MAX {
                    continue;
                }
                let word = byte as usize / 64;
                observed_bytes[source as usize][word] |= 1u64 << (byte as usize % 64);
                selected[target as usize] = true;
                if state_marks[target as usize] != generation {
                    state_marks[target as usize] = generation;
                    product_states.push(target);
                }
            }
            node_state_start[child] = child_start as u32;
            node_state_len[child] = (product_states.len() - child_start) as u32;
        }
    }
}

/// Restrict an exact relevant-byte powerset to precisely the whole-token and
/// reset-suffix trajectories needed by the bounded vocabulary analysis.
/// This preserves the bounded view's observation domain while reusing the
/// powerset configurations and transitions already constructed for the NFA
/// refinement prepass.
pub fn build_bounded_analysis_view_from_relevant_powerset(
    view: &RelevantPowersetView,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
) -> BoundedAnalysisView {
    let transition_stride = view.bytes.len();
    let mut byte_to_slot = [u16::MAX; 256];
    for (slot, &byte) in view.bytes.iter().enumerate() {
        byte_to_slot[byte as usize] = slot as u16;
    }
    let mut dense_transitions =
        vec![u32::MAX; view.states.len().saturating_mul(transition_stride)];
    for state in 0..view.states.len() {
        let edge_start = view.edge_offsets[state] as usize;
        let edge_end = view.edge_offsets[state + 1] as usize;
        for &(byte, target) in &view.edges[edge_start..edge_end] {
            let slot = byte_to_slot[byte as usize];
            debug_assert_ne!(slot, u16::MAX);
            dense_transitions[state * transition_stride + slot as usize] = target;
        }
    }
    let token_trie = build_byte_trie(tokens.iter().copied());
    let suffix_trie = build_byte_trie(
        tokens
            .iter()
            .flat_map(|token| (0..token.len()).map(move |offset| &token[offset..])),
    );
    let mut selected = vec![false; view.states.len()];
    let mut observed_bytes = vec![[0u64; 4]; view.states.len()];
    let mut raw_seed_states = raw_start_states
        .iter()
        .map(|&raw| view.view_state_for_raw_start(raw) as u32)
        .collect::<Vec<_>>();
    raw_seed_states.sort_unstable();
    raw_seed_states.dedup();
    mark_relevant_powerset_trie_trajectories(
        view,
        &dense_transitions,
        &byte_to_slot,
        transition_stride,
        &raw_seed_states,
        &token_trie,
        &mut selected,
        &mut observed_bytes,
    );
    mark_relevant_powerset_trie_trajectories(
        view,
        &dense_transitions,
        &byte_to_slot,
        transition_stride,
        std::slice::from_ref(&(view.start_state as u32)),
        &suffix_trie,
        &mut selected,
        &mut observed_bytes,
    );
    selected[view.start_state] = true;

    let mut old_to_new = vec![u32::MAX; view.states.len()];
    let mut states = Vec::<FlatDfaState>::new();
    for (old, &is_selected) in selected.iter().enumerate() {
        if is_selected {
            old_to_new[old] = states.len() as u32;
            states.push(view.states[old].clone());
        }
    }
    let mut transitions = vec![u32::MAX; states.len().saturating_mul(256)];
    for (old_source, &new_source) in old_to_new.iter().enumerate() {
        if new_source == u32::MAX {
            continue;
        }
        let edge_start = view.edge_offsets[old_source] as usize;
        let edge_end = view.edge_offsets[old_source + 1] as usize;
        for &(byte, old_target) in &view.edges[edge_start..edge_end] {
            let word = byte as usize / 64;
            if observed_bytes[old_source][word] & (1u64 << (byte as usize % 64)) == 0 {
                continue;
            }
            let new_target = old_to_new[old_target as usize];
            debug_assert_ne!(new_target, u32::MAX);
            transitions[new_source as usize * 256 + byte as usize] = new_target;
        }
    }

    let mut raw_start_to_view = vec![u32::MAX; view.raw_start_to_view.len()];
    for &raw in raw_start_states {
        let old = view.view_state_for_raw_start(raw);
        let new = old_to_new[old];
        assert_ne!(new, u32::MAX, "raw analysis seed was not retained");
        raw_start_to_view[raw] = new;
    }
    let start_state = old_to_new[view.start_state];
    assert_ne!(start_state, u32::MAX);
    BoundedAnalysisView {
        tokenizer_view: TokenizerView {
            flat_dfa: FlatDfa {
                states,
                start_state: start_state as usize,
                transitions: Arc::from(transitions),
            },
        },
        raw_start_to_view,
    }
}

pub fn build_relevant_powerset_analysis_view(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    active_groups: &[bool],
) -> BoundedAnalysisView {
    let view = build_relevant_powerset_view(
        tokenizer,
        relevant_bytes,
        Some(active_groups),
        None,
    );
    let raw_start_to_view = view.raw_start_to_view.to_vec();
    BoundedAnalysisView {
        tokenizer_view: view.into_tokenizer_view(),
        raw_start_to_view,
    }
}

enum RepresentativeClosure {
    Singleton(u32),
    Multi(Box<[u32]>),
}

/// Intern powerset configurations without storing every state vector twice.
/// The old `FxHashMap<Vec<u32>, u32>` kept one heap allocation as the map key
/// and cloned the same vector into `configs`. Large powersets therefore paid a
/// full configuration copy and roughly doubled configuration storage for every
/// newly discovered state. Keep the canonical vectors only in `configs`; the
/// hash table stores compact candidate IDs and resolves rare hash collisions by
/// exact slice comparison.
#[derive(Default)]
struct RelevantConfigInterner {
    ids_by_hash: FxHashMap<u64, SmallVec<[u32; 1]>>,
}

impl RelevantConfigInterner {
    #[inline]
    fn hash(states: &[u32]) -> u64 {
        let mut hasher = FxHasher::default();
        states.hash(&mut hasher);
        hasher.finish()
    }

    fn intern(&mut self, states: Vec<u32>, configs: &mut Vec<Box<[u32]>>) -> u32 {
        let hash = Self::hash(&states);
        if let Some(candidates) = self.ids_by_hash.get(&hash) {
            if let Some(id) = candidates
                .iter()
                .copied()
                .find(|&id| configs[id as usize].as_ref() == states.as_slice())
            {
                return id;
            }
        }
        let id = configs.len() as u32;
        configs.push(states.into_boxed_slice());
        self.ids_by_hash.entry(hash).or_default().push(id);
        id
    }
}

impl RepresentativeClosure {
    #[inline]
    fn singleton(&self) -> Option<u32> {
        match self {
            Self::Singleton(state) => Some(*state),
            Self::Multi(_) => None,
        }
    }

    #[inline]
    fn as_slice(&self) -> &[u32] {
        match self {
            Self::Singleton(state) => std::slice::from_ref(state),
            Self::Multi(states) => states,
        }
    }
}

pub fn raw_active_language_states(
    tokenizer: &Tokenizer,
    active_groups: Option<&[bool]>,
) -> Option<Vec<bool>> {
    let active_groups = active_groups?;
    Some(
        (0..tokenizer.num_states())
            .map(|state| {
                tokenizer
                    .matched_terminals_iter(state)
                    .chain(tokenizer.possible_future_terminals_iter(state))
                    .any(|group| {
                        active_groups
                            .get(group as usize)
                            .copied()
                            .unwrap_or(false)
                    })
            })
            .collect(),
    )
}

fn project_raw_config(mut states: Vec<u32>, active_language: Option<&[bool]>) -> Vec<u32> {
    if let Some(active_language) = active_language {
        states.retain(|&state| active_language[state as usize]);
    }
    states.sort_unstable();
    states.dedup();
    states
}

fn mapped_class_active_language(
    state_map: &ManyToOneIdMap,
    raw_active_language: Option<&[bool]>,
) -> Option<Vec<bool>> {
    raw_active_language.map(|raw_active_language| {
        state_map
            .internal_to_originals
            .iter()
            .map(|members| {
                let active = members
                    .iter()
                    .any(|&raw| raw_active_language[raw as usize]);
                debug_assert!(
                    members
                        .iter()
                        .all(|&raw| raw_active_language[raw as usize] == active),
                    "mapped powerset seed mixed active-live and active-dead scanner states",
                );
                active
            })
            .collect()
    })
}

fn mapped_target_config(
    targets: &[u32],
    state_map: &ManyToOneIdMap,
    class_active_language: Option<&[bool]>,
) -> Vec<u32> {
    let mut target_config = targets
        .iter()
        .map(|&raw_state| state_map.original_to_internal[raw_state as usize])
        .filter(|&class| {
            class_active_language.is_none_or(|active| active[class as usize])
        })
        .collect::<Vec<_>>();
    target_config.sort_unstable();
    target_config.dedup();
    target_config
}

fn intern_mapped_target_config(
    targets: &[u32],
    state_map: &ManyToOneIdMap,
    class_active_language: Option<&[bool]>,
    config_ids: &mut RelevantConfigInterner,
    configs: &mut Vec<Box<[u32]>>,
) -> u32 {
    let target_config = mapped_target_config(targets, state_map, class_active_language);
    config_ids.intern(target_config, configs)
}

impl BoundedAnalysisView {
    pub fn view_state_for_raw_start(&self, raw_state: usize) -> usize {
        let state = self
            .raw_start_to_view
            .get(raw_state)
            .copied()
            .expect("raw state is outside bounded NFA analysis domain");
        assert_ne!(
            state,
            u32::MAX,
            "raw state was not seeded into bounded NFA analysis",
        );
        let state = state as usize;
        assert!(
            state < self.tokenizer_view.dfa().states.len(),
            "bounded raw-start map produced an invalid view state",
        );
        state
    }
}

pub fn build_relevant_powerset_view(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    active_groups: Option<&[bool]>,
    state_map: Option<&ManyToOneIdMap>,
) -> RelevantPowersetView {
    try_build_relevant_powerset_view(
        tokenizer,
        relevant_bytes,
        active_groups,
        state_map,
        None,
        RawPowersetTargetMode::Direct,
    )
    .expect("unbounded relevant-powerset construction cannot abort")
}

pub fn build_relevant_powerset_view_budgeted(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    active_groups: Option<&[bool]>,
    state_map: Option<&ManyToOneIdMap>,
    budget: RelevantPowersetWorkBudget,
) -> Result<RelevantPowersetView, RelevantPowersetWork> {
    try_build_relevant_powerset_view(
        tokenizer,
        relevant_bytes,
        active_groups,
        state_map,
        Some(budget),
        RawPowersetTargetMode::Direct,
    )
}

fn try_build_relevant_powerset_view(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    active_groups: Option<&[bool]>,
    state_map: Option<&ManyToOneIdMap>,
    budget: Option<RelevantPowersetWorkBudget>,
    raw_target_mode: RawPowersetTargetMode,
) -> Result<RelevantPowersetView, RelevantPowersetWork> {
    #[inline]
    fn check_budget(
        configurations: usize,
        edges: usize,
        budget: Option<RelevantPowersetWorkBudget>,
    ) -> Result<(), RelevantPowersetWork> {
        if budget.is_some_and(|budget| {
            configurations > budget.max_configurations || edges > budget.max_edges
        }) {
            return Err(RelevantPowersetWork {
                configurations,
                edges,
            });
        }
        Ok(())
    }

    let raw_state_count = tokenizer.num_states() as usize;
    if let Some(state_map) = state_map {
        assert_eq!(state_map.original_to_internal.len(), raw_state_count);
        assert_eq!(
            state_map.internal_to_originals.len(),
            state_map.representative_original_ids.len(),
        );
        assert!(state_map.original_to_internal.iter().all(|&class| {
            class != u32::MAX
                && (class as usize) < state_map.representative_original_ids.len()
        }));
    }

    // Project epsilon configurations themselves, not only their visible output
    // metadata. A raw component with no active finalizer and no active future
    // terminal has empty active language, so retaining it in a mixed powerset
    // configuration would let inactive-only topology distinguish successors.
    let raw_active_language = raw_active_language_states(tokenizer, active_groups);
    let class_active_language = state_map.and_then(|state_map| {
        mapped_class_active_language(state_map, raw_active_language.as_deref())
    });

    let (mut config_ids, mut configs, raw_start_to_view, mut worklist, mut queued) =
        if let Some(state_map) = state_map {
            let class_count = state_map.representative_original_ids.len();
            let mut config_ids = RelevantConfigInterner::default();
            let mut configs = Vec::<Box<[u32]>>::new();
            let mut class_to_view = vec![u32::MAX; class_count];
            let mut worklist = VecDeque::<u32>::new();
            let mut queued = Vec::<bool>::new();
            for class in 0..class_count as u32 {
                let config = if class_active_language
                    .as_deref()
                    .is_none_or(|active| active[class as usize])
                {
                    vec![class]
                } else {
                    Vec::new()
                };
                let state = config_ids.intern(config, &mut configs);
                class_to_view[class as usize] = state;
                if queued.len() < configs.len() {
                    queued.resize(configs.len(), false);
                }
                if !queued[state as usize] {
                    queued[state as usize] = true;
                    worklist.push_back(state);
                }
            }
            let raw_start_to_view = state_map
                .original_to_internal
                .iter()
                .map(|&class| class_to_view[class as usize])
                .collect::<Vec<_>>();
            (
                config_ids,
                configs,
                raw_start_to_view,
                worklist,
                queued,
            )
        } else {
            let mut config_ids = RelevantConfigInterner::default();
            let mut configs = Vec::<Box<[u32]>>::new();
            let mut raw_start_to_view = vec![u32::MAX; raw_state_count];
            let mut worklist = VecDeque::<u32>::new();
            let mut queued = Vec::<bool>::new();
            for raw_state in 0..raw_state_count {
                let closure = project_raw_config(
                    tokenizer
                        .execute_from_state_end_only(&[], raw_state as u32)
                        .to_vec(),
                    raw_active_language.as_deref(),
                );
                let state = config_ids.intern(closure, &mut configs);
                raw_start_to_view[raw_state] = state;
                if queued.len() < configs.len() {
                    queued.resize(configs.len(), false);
                }
                if !queued[state as usize] {
                    queued[state as usize] = true;
                    worklist.push_back(state);
                }
            }
            (
                config_ids,
                configs,
                raw_start_to_view,
                worklist,
                queued,
            )
        };
    check_budget(configs.len(), 0, budget)?;

    let start_state = raw_start_to_view[tokenizer.initial_state_id() as usize] as usize;
    let bytes = relevant_bytes
        .iter()
        .enumerate()
        .filter_map(|(byte, &relevant)| relevant.then_some(byte as u8))
        .collect::<Vec<_>>();
    let closure_by_class = state_map.map(|state_map| {
        state_map
            .representative_original_ids
            .iter()
            .map(|&representative| {
                let closure = tokenizer.execute_from_state_end_only(&[], representative);
                if closure.len() == 1 {
                    RepresentativeClosure::Singleton(closure[0])
                } else {
                    RepresentativeClosure::Multi(closure.to_vec().into_boxed_slice())
                }
            })
            .collect::<Vec<_>>()
    });
    // For a mapped powerset, the one-byte NFA successor of a configuration is
    // the union of the one-byte successors of its member classes. Cache those
    // exact class/byte successor class-sets once. The old inner loop rebuilt a
    // raw representative vector, re-ran epsilon closure, remapped raw targets,
    // sorted, and deduplicated for every configuration × byte pair.
    let mapped_targets_by_class_byte = state_map.map(|state_map| {
        let closure_by_class = closure_by_class
            .as_ref()
            .expect("mapped powerset must retain representative closures");
        let class_count = state_map.representative_original_ids.len();
        let byte_count = bytes.len();
        let mut byte_slot = [usize::MAX; 256];
        for (slot, &byte) in bytes.iter().enumerate() {
            byte_slot[byte as usize] = slot;
        }
        let mut targets = (0..class_count.saturating_mul(byte_count))
            .map(|_| Box::<[u32]>::default())
            .collect::<Vec<_>>();
        for class in 0..class_count {
            if let Some(raw_source) = closure_by_class[class].singleton() {
                for (byte, raw_target) in tokenizer.transitions_from(raw_source) {
                    let slot = byte_slot[byte as usize];
                    if slot == usize::MAX {
                        continue;
                    }
                    let raw_targets = tokenizer.execute_from_state_end_only(&[], raw_target);
                    let mapped = mapped_target_config(
                        &raw_targets,
                        state_map,
                        class_active_language.as_deref(),
                    );
                    targets[class * byte_count + slot] = mapped.into_boxed_slice();
                }
            } else {
                let representative = state_map.representative_original_ids[class];
                for (slot, &byte) in bytes.iter().enumerate() {
                    let raw_targets = tokenizer.step_all(&[representative], byte);
                    if raw_targets.is_empty() {
                        continue;
                    }
                    let mapped = mapped_target_config(
                        &raw_targets,
                        state_map,
                        class_active_language.as_deref(),
                    );
                    targets[class * byte_count + slot] = mapped.into_boxed_slice();
                }
            }
        }
        targets
    });
    let mut edge_offsets = Vec::<u32>::with_capacity(configs.len() + 1);
    let mut edges = Vec::<(u8, u32)>::new();
    edge_offsets.push(0);
    if let Some(state_map) = state_map {
        let closure_by_class = closure_by_class
            .as_ref()
            .expect("mapped powerset must retain representative closures");
        let mapped_targets_by_class_byte = mapped_targets_by_class_byte
            .as_ref()
            .expect("mapped powerset must retain cached class transitions");
        let class_count = state_map.representative_original_ids.len();
        let byte_count = bytes.len();
        let mut target_marks = vec![0u32; class_count];
        let mut target_epoch = 0u32;
        while let Some(state) = worklist.pop_front() {
            assert_eq!(
                state as usize + 1,
                edge_offsets.len(),
                "powerset states must be processed in interning order",
            );
            let config_index = state as usize;
            let singleton_class = match configs[config_index].as_ref() {
                [class] => Some(*class),
                _ => None,
            };
            if let Some(class) = singleton_class
                && let Some(raw_source) = closure_by_class[class as usize].singleton()
            {
                for (byte, raw_target) in tokenizer.transitions_from(raw_source) {
                    if !relevant_bytes[byte as usize] {
                        continue;
                    }
                    let targets = tokenizer.execute_from_state_end_only(&[], raw_target);
                    debug_assert!(!targets.is_empty());
                    let target = intern_mapped_target_config(
                        &targets,
                        state_map,
                        class_active_language.as_deref(),
                        &mut config_ids,
                        &mut configs,
                    );
                    if queued.len() < configs.len() {
                        queued.resize(configs.len(), false);
                    }
                    // The canonical empty projected configuration is the
                    // implicit dead target, exactly like an absent edge.
                    if configs[target as usize].is_empty() {
                        continue;
                    }
                    edges.push((byte, target));
                    check_budget(configs.len(), edges.len(), budget)?;
                    if !queued[target as usize] {
                        queued[target as usize] = true;
                        worklist.push_back(target);
                    }
                }
            } else {
                for (byte_slot, &byte) in bytes.iter().enumerate() {
                    target_epoch = target_epoch.wrapping_add(1);
                    if target_epoch == 0 {
                        target_marks.fill(0);
                        target_epoch = 1;
                    }
                    let mut target_config = Vec::<u32>::new();
                    for &class in configs[config_index].iter() {
                        for &target_class in
                            &mapped_targets_by_class_byte[class as usize * byte_count + byte_slot]
                        {
                            let mark = &mut target_marks[target_class as usize];
                            if *mark != target_epoch {
                                *mark = target_epoch;
                                target_config.push(target_class);
                            }
                        }
                    }
                    if target_config.is_empty() {
                        continue;
                    }
                    target_config.sort_unstable();
                    let target = config_ids.intern(target_config, &mut configs);
                    if queued.len() < configs.len() {
                        queued.resize(configs.len(), false);
                    }
                    if configs[target as usize].is_empty() {
                        continue;
                    }
                    edges.push((byte, target));
                    check_budget(configs.len(), edges.len(), budget)?;
                    if !queued[target as usize] {
                        queued[target as usize] = true;
                        worklist.push_back(target);
                    }
                }
            }
            edge_offsets.push(edges.len() as u32);
        }
    } else {
        let direct_raw_target_views = raw_target_mode == RawPowersetTargetMode::Direct;
        // Moderate raw-NFA domains benefit from aggregating sparse transitions
        // directly into byte buckets. On very large bounded tokenizers the
        // powerset is usually dominated by preseeded singleton closures, and
        // retaining the old two-pass path avoids adding allocation/cache
        // pressure to the P100 compile path for little useful work.
        const ONE_PASS_RAW_TARGET_MAX_STATES: usize = 65_536;
        let one_pass_raw_targets =
            direct_raw_target_views && raw_state_count <= ONE_PASS_RAW_TARGET_MAX_STATES;
        let mut byte_marks = [0u32; 256];
        let mut byte_epoch = 0u32;
        let mut candidate_bytes = Vec::<u8>::new();
        let mut target_marks = (direct_raw_target_views && !one_pass_raw_targets)
            .then(|| vec![0u32; raw_state_count]);
        let mut target_epoch = 0u32;
        // In the one-pass lane, keep one reusable target bucket per byte. The
        // previous path first scanned every source row to discover candidate
        // bytes, then rescanned every source once per candidate byte via
        // `tokenizer.step`.
        let mut direct_target_buckets = one_pass_raw_targets.then(|| {
            (0..256).map(|_| Vec::<u32>::new()).collect::<Vec<_>>()
        });
        while let Some(state) = worklist.pop_front() {
            assert_eq!(
                state as usize + 1,
                edge_offsets.len(),
                "powerset states must be processed in interning order",
            );
            let config_index = state as usize;
            let singleton_source = match configs[config_index].as_ref() {
                [source] => Some(*source),
                _ => None,
            };
            if let Some(source) = singleton_source {
                // Raw starts were all seeded before traversal, so the exact
                // projected epsilon closure of every direct byte target is
                // already interned in `raw_start_to_view`.
                for (byte, raw_target) in tokenizer.transitions_from(source) {
                    if !relevant_bytes[byte as usize] {
                        continue;
                    }
                    let target = raw_start_to_view[raw_target as usize];
                    debug_assert_ne!(target, u32::MAX);
                    if configs[target as usize].is_empty() {
                        continue;
                    }
                    edges.push((byte, target));
                    check_budget(configs.len(), edges.len(), budget)?;
                    if !queued[target as usize] {
                        queued[target as usize] = true;
                        worklist.push_back(target);
                    }
                }
                edge_offsets.push(edges.len() as u32);
                continue;
            }

            byte_epoch = byte_epoch.wrapping_add(1);
            if byte_epoch == 0 {
                byte_marks.fill(0);
                byte_epoch = 1;
            }
            candidate_bytes.clear();
            if one_pass_raw_targets {
                let target_buckets = direct_target_buckets
                    .as_mut()
                    .expect("one-pass raw targets must retain target buckets");
                for &source in configs[config_index].iter() {
                    for (byte, raw_target) in tokenizer.transitions_from(source) {
                        let byte_index = byte as usize;
                        if !relevant_bytes[byte_index] {
                            continue;
                        }
                        if byte_marks[byte_index] != byte_epoch {
                            byte_marks[byte_index] = byte_epoch;
                            candidate_bytes.push(byte);
                        }
                        let target_view = raw_start_to_view[raw_target as usize];
                        target_buckets[byte_index]
                            .extend_from_slice(configs[target_view as usize].as_ref());
                    }
                }
            } else {
                for &source in configs[config_index].iter() {
                    for (byte, _) in tokenizer.transitions_from(source) {
                        let byte_index = byte as usize;
                        if relevant_bytes[byte_index] && byte_marks[byte_index] != byte_epoch {
                            byte_marks[byte_index] = byte_epoch;
                            candidate_bytes.push(byte);
                        }
                    }
                }
            }
            candidate_bytes.sort_unstable();
            for &byte in &candidate_bytes {
                let projected = if one_pass_raw_targets {
                    let target_buckets = direct_target_buckets
                        .as_mut()
                        .expect("one-pass raw targets must retain target buckets");
                    let mut projected = std::mem::take(&mut target_buckets[byte as usize]);
                    projected.sort_unstable();
                    projected.dedup();
                    projected
                } else if direct_raw_target_views {
                    target_epoch = target_epoch.wrapping_add(1);
                    let target_marks = target_marks
                        .as_mut()
                        .expect("two-pass raw targets must retain target marks");
                    if target_epoch == 0 {
                        target_marks.fill(0);
                        target_epoch = 1;
                    }
                    let mut projected = Vec::<u32>::new();
                    for &source in configs[config_index].iter() {
                        let Some(raw_target) = tokenizer.step(source, byte) else {
                            continue;
                        };
                        let target_view = raw_start_to_view[raw_target as usize];
                        for &target_state in configs[target_view as usize].iter() {
                            let mark = &mut target_marks[target_state as usize];
                            if *mark != target_epoch {
                                *mark = target_epoch;
                                projected.push(target_state);
                            }
                        }
                    }
                    projected.sort_unstable();
                    projected
                } else {
                    let targets = tokenizer.step_all(&configs[config_index], byte);
                    if targets.is_empty() {
                        continue;
                    }
                    project_raw_config(targets.to_vec(), raw_active_language.as_deref())
                };
                if projected.is_empty() {
                    continue;
                }
                let target = config_ids.intern(projected, &mut configs);
                if queued.len() < configs.len() {
                    queued.resize(configs.len(), false);
                }
                edges.push((byte, target));
                check_budget(configs.len(), edges.len(), budget)?;
                if !queued[target as usize] {
                    queued[target as usize] = true;
                    worklist.push_back(target);
                }
            }
            edge_offsets.push(edges.len() as u32);
        }
    }

    let states = if let Some(state_map) = state_map {
        let closure_by_class = closure_by_class
            .as_ref()
            .expect("mapped powerset must retain representative closures");
        let class_states = closure_by_class
            .iter()
            .map(|closure| {
                FlatDfaState {
                    finalizers: filtered_config_groups(
                        tokenizer,
                        closure.as_slice(),
                        active_groups,
                        true,
                    ),
                    possible_future_group_ids: filtered_config_groups(
                        tokenizer,
                        closure.as_slice(),
                        active_groups,
                        false,
                    ),
                }
            })
            .collect::<Vec<_>>();
        configs
            .iter()
            .map(|config| {
                if config.len() == 1 {
                    return class_states[config[0] as usize].clone();
                }
                let mut finalizers = Vec::<usize>::new();
                let mut possible_future_group_ids = Vec::<usize>::new();
                for &class in config.iter() {
                    let class_state = &class_states[class as usize];
                    finalizers.extend(class_state.finalizers.iter().copied());
                    possible_future_group_ids
                        .extend(class_state.possible_future_group_ids.iter().copied());
                }
                finalizers.sort_unstable();
                finalizers.dedup();
                possible_future_group_ids.sort_unstable();
                possible_future_group_ids.dedup();
                FlatDfaState {
                    finalizers,
                    possible_future_group_ids,
                }
            })
            .collect()
    } else {
        configs
            .iter()
            .map(|config| FlatDfaState {
                finalizers: filtered_config_groups(tokenizer, config, active_groups, true),
                possible_future_group_ids: filtered_config_groups(
                    tokenizer,
                    config,
                    active_groups,
                    false,
                ),
            })
            .collect()
    };
    Ok(RelevantPowersetView {
        states,
        start_state,
        bytes,
        edge_offsets,
        edges,
        raw_start_to_view: Arc::from(raw_start_to_view),
        configurations: Arc::from(configs),
    })
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ByteTrieNode {
    children: Vec<(u8, usize)>,
}

fn build_byte_trie<'a>(sequences: impl IntoIterator<Item = &'a [u8]>) -> Vec<ByteTrieNode> {
    let mut nodes = vec![ByteTrieNode::default()];
    let mut edges = FxHashMap::<u64, usize>::default();
    for sequence in sequences {
        let mut node = 0usize;
        for &byte in sequence {
            let key = ((node as u64) << 8) | byte as u64;
            let child = if let Some(&child) = edges.get(&key) {
                child
            } else {
                let child = nodes.len();
                nodes.push(ByteTrieNode::default());
                nodes[node].children.push((byte, child));
                edges.insert(key, child);
                child
            };
            node = child;
        }
    }
    nodes
}

/// Build the same trie as `build_byte_trie` when input sequences are already
/// in lexicographic byte order. Consecutive sequences share exactly their LCP,
/// so no edge hash table is needed: truncate the previous prefix path to the
/// LCP and append the new suffix once.
#[derive(Debug)]
pub struct TokenBoundedAnalysisTrie {
    nodes: Arc<[ByteTrieNode]>,
}

impl TokenBoundedAnalysisTrie {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_reasonable_superset_for(&self, token_count: usize) -> bool {
        const MAX_NODE_RATIO: usize = 8;
        self.len() <= token_count.saturating_mul(MAX_NODE_RATIO).max(256)
    }

    /// Raw states from which an event can be reached on a vocabulary prefix.
    ///
    /// All trie positions (including the root and token endpoints) may witness
    /// an event. The result is exact for this query; a caller may use a broad
    /// event predicate to obtain a conservative semantic support filter.
    /// `initial_events` optionally substitutes the event predicate only at the
    /// zero-byte root; every positive prefix still uses `events`.
    /// `epsilon_closures` supplies complete reflexive/transitive closures for
    /// nontrivial epsilon sources; omitted sources have the singleton closure.
    ///
    /// For trie node p, E_p = eps_pre(events OR union_b pre_b(E_child(p,b))).
    /// Induction on remaining trie height proves the recurrence. Nodes with
    /// identical outgoing byte/child signatures have identical E_p and can
    /// share one row. This quotients the *observation trie*, never lexer states,
    /// terminal labels, parser coordinates, or token identities.
    ///
    /// A bounded analysis may decline without changing the caller's domain.
    /// Unknown/virtual transitions also decline rather than become dead edges.
    pub fn prefix_event_sources(
        &self,
        flat_trans: &[u32],
        events: &[bool],
        epsilon_closures: &[(usize, Box<[u32]>)],
        initial_events: Option<&[bool]>,
    ) -> Option<(Vec<bool>, usize)> {
        let states = events.len();
        if flat_trans.len() != states.checked_mul(256)? || self.nodes.is_empty()
            || initial_events.is_some_and(|events| events.len() != states)
        {
            return None;
        }
        for (source, closure) in epsilon_closures {
            if *source >= states || closure.iter().any(|&q| q as usize >= states) {
                return None;
            }
        }

        // Reuse the ordinary sorted-vocabulary trie. Bottom-up suffix sharing
        // avoids evaluating repeated observation tails for every token prefix.
        let mut classes = vec![0usize; self.nodes.len()];
        let mut dag = Vec::<Vec<(u8, usize)>>::new();
        let mut intern = FxHashMap::<Vec<(u8, usize)>, usize>::default();
        for node in (0..self.nodes.len()).rev() {
            let mut signature = self.nodes[node].children.iter()
                .map(|&(byte, child)| (byte, classes[child])).collect::<Vec<_>>();
            signature.sort_unstable();
            classes[node] = if let Some(&id) = intern.get(&signature) {
                id
            } else {
                let id = dag.len();
                dag.push(signature.clone());
                intern.insert(signature, id);
                id
            };
        }
        let edges = dag.iter().map(Vec::len).sum::<usize>()
            + initial_events.map_or(0, |_| dag[classes[0]].len());
        let epsilon_volume = epsilon_closures.iter().map(|(_, closure)| closure.len()).sum::<usize>();
        let words = states.div_ceil(64);
        if states.checked_mul(edges)?.checked_add(epsilon_volume.checked_mul(dag.len())?)? > 120_000_000
            || words.checked_mul(dag.len())?.checked_mul(16)? > 64 * 1024 * 1024
        {
            return None;
        }
        let mut used_bytes = [false; 256];
        for &(byte, _) in dag.iter().flatten() { used_bytes[byte as usize] = true; }
        let byte_count = used_bytes.iter().filter(|&&used| used).count();
        if states.checked_mul(byte_count)?.checked_mul(4)? > 64 * 1024 * 1024 {
            return None;
        }
        let used_bytes=used_bytes.iter().enumerate().filter_map(|(b,&yes)|yes.then_some(b)).collect::<Vec<_>>();
        let mut columns=vec![Vec::<(u32,u32)>::new();256];
        let mut stored_pairs=0usize;
        // Scan each contiguous raw transition row once, not one strided full
        // lexer pass for every byte. This changes storage order only; each
        // byte column remains sorted by original source-state identity.
        for (source,row) in flat_trans.chunks_exact(256).enumerate(){
            for &byte in &used_bytes{
                let target=row[byte];
                if target==u32::MAX{continue;}
                if target as usize>=states{return None;}
                stored_pairs=stored_pairs.checked_add(1)?;
                if stored_pairs>8_000_000{return None;}
                columns[byte].push((source as u32,target));
            }
        }
        let mut event_words = vec![0u64; words];
        for (state, &event) in events.iter().enumerate() {
            if event { event_words[state / 64] |= 1u64 << (state % 64); }
        }
        let mut rows = Vec::<Vec<u64>>::with_capacity(dag.len());
        let mut row_ids=Vec::<usize>::with_capacity(dag.len());
        let mut interned_rows=FxHashMap::<Vec<u64>,usize>::default();
        let mut preimages=FxHashMap::<(u8,usize),Vec<(usize,u64)>>::default();
        let mut cache_words=0usize;
        for children in &dag {
            let mut live = event_words.clone();
            for &(byte, child) in children {
                let target_live = &rows[child];
                let key=(byte,row_ids[child]);
                if let Some(cached)=preimages.get(&key){
                    for &(word,bits) in cached{live[word]|=bits;}
                }else{
                    let mut observed=vec![0u64;words];
                    for &(source,target)in &columns[byte as usize]{
                        if target_live[target as usize/64]&(1u64<<(target%64))!=0{
                            observed[source as usize/64]|=1u64<<(source%64);
                        }
                    }
                    let nonzero=observed.into_iter().enumerate().filter(|(_,bits)|*bits!=0).collect::<Vec<_>>();
                    for &(word,bits)in &nonzero{live[word]|=bits;}
                    // Stop caching rather than approximating if the auxiliary
                    // index reaches its memory budget. This is only memoization.
                    if cache_words+nonzero.len()<=2_000_000 && preimages.len()<65_536{
                        cache_words+=nonzero.len();preimages.insert(key,nonzero);
                    }
                }
            }
            // Full closures make this order-independent, including cycles:
            // any newly observed member's closure is already contained in the
            // current source's transitive closure.
            for (source, closure) in epsilon_closures {
                if closure.iter().any(|&q| live[q as usize / 64] & (1u64 << (q % 64)) != 0) {
                    live[source / 64] |= 1u64 << (source % 64);
                }
            }
            let next_id=interned_rows.len();
            let id=*interned_rows.entry(live.clone()).or_insert(next_id);
            row_ids.push(id);
            rows.push(live);
        }
        // Some observers distinguish the token's zero-byte entry from later
        // positions (e.g. terminal completion requires consuming a byte, while
        // an already-visible foreign future can matter at entry). This changes
        // only the root event set; every child still uses ordinary events.
        let initial_row;
        let root = if let Some(initial_events) = initial_events {
            let mut live = vec![0u64; words];
            for (q, &event) in initial_events.iter().enumerate() {
                if event { live[q / 64] |= 1u64 << (q % 64); }
            }
            for &(byte, child) in &dag[classes[0]] {
                for &(source, target) in &columns[byte as usize] {
                    let source=source as usize;
                    if rows[child][target as usize / 64] & (1u64 << (target % 64)) != 0
                    { live[source / 64] |= 1u64 << (source % 64); }
                }
            }
            for (source, closure) in epsilon_closures {
                if closure.iter().any(|&q| live[q as usize / 64] & (1u64 << (q % 64)) != 0) {
                    live[source / 64] |= 1u64 << (source % 64);
                }
            }
            initial_row = live;
            &initial_row
        } else { &rows[classes[0]] };
        Some(((0..states).map(|q| root[q / 64] & (1u64 << (q % 64)) != 0).collect(), dag.len()))
    }
}

#[cfg(test)]
#[test]
fn prefix_event_sources_match_independent_forward_execution() {
    // Generated byte DFAs with nondeterministic epsilon closures, cycles,
    // duplicate spellings, empty tokens, and event-free graphs. Compare the
    // backward DAG recurrence with explicit forward execution of every token
    // from every raw source (no trie or quotient in that oracle).
    let mut rng = 7u64;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 32) as usize
    };
    for case in 0..512 {
        let n = 2 + next() % 13;
        let mut flat = vec![u32::MAX; n * 256];
        let mut eps = vec![Vec::<usize>::new(); n];
        for q in 0..n {
            for b in 0..3 {
                let target = next() % (n + 2);
                if target < n { flat[q * 256 + b] = target as u32; }
            }
            if next() % 3 == 0 { eps[q].push(next() % n); }
        }
        let mut closures = Vec::new();
        let mut full_closures = Vec::new();
        for q in 0..n {
            let mut seen = vec![false; n];
            let mut pending = vec![q];
            while let Some(s) = pending.pop() {
                if seen[s] { continue; }
                seen[s] = true;
                pending.extend(eps[s].iter().copied());
            }
            let closure = (0..n).filter(|&s| seen[s]).map(|s| s as u32).collect::<Vec<_>>();
            if closure.len() > 1 { closures.push((q, closure.clone().into_boxed_slice())); }
            full_closures.push(closure);
        }
        let events = (0..n).map(|_| next() % 4 == 0).collect::<Vec<_>>();
        let mut tokens = (0..12).map(|_| {
            let length = next() % 7;
            (0..length).map(|_| (next() % 3) as u8).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        tokens.sort();
        let borrowed = tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let trie = build_token_bounded_analysis_trie_sorted(&borrowed);
        let initial_events = (0..n).map(|_| next() % 5 == 0).collect::<Vec<_>>();
        for initial in [None, Some(initial_events.as_slice())] {
        let (actual, _) = trie.prefix_event_sources(&flat, &events, &closures, initial).unwrap();
        let expected = (0..n).map(|source| tokens.iter().any(|token| {
            let mut current = full_closures[source].clone();
            if current.iter().any(|&q| initial.unwrap_or(&events)[q as usize]) { return true; }
            for &byte in token {
                let mut target = Vec::new();
                for &q in &current {
                    let q = flat[q as usize * 256 + byte as usize];
                    if q != u32::MAX { target.extend(full_closures[q as usize].iter().copied()); }
                }
                target.sort_unstable(); target.dedup(); current = target;
                if current.iter().any(|&q| events[q as usize]) { return true; }
            }
            false
        })).collect::<Vec<_>>();
        assert_eq!(actual, expected, "generated prefix event case {case}");
        }
    }
}

#[cfg(test)]
#[test]
fn prefix_event_sources_declines_unknown_targets_and_bad_domains() {
    let trie = build_token_bounded_analysis_trie_sorted(&[b"ab"]);
    assert!(trie.prefix_event_sources(&[], &[false], &[], None).is_none());
    let mut flat = vec![u32::MAX; 256];
    flat[b'a' as usize] = 3;
    assert!(trie.prefix_event_sources(&flat, &[false], &[], None).is_none());
    flat[b'a' as usize] = 0;
    assert!(trie.prefix_event_sources(&flat, &[false], &[(0, vec![1].into_boxed_slice())], None).is_none());
    assert_eq!(trie.prefix_event_sources(&flat, &[false], &[], None).unwrap().0, vec![false]);
    assert_eq!(trie.prefix_event_sources(&flat, &[true], &[], None).unwrap().0, vec![true]);
    flat[b'a' as usize] = u32::MAX;
    assert_eq!(trie.prefix_event_sources(&flat, &[true], &[], Some(&[false])).unwrap().0, vec![false]);
    assert!(trie.prefix_event_sources(&flat, &[true], &[], Some(&[])).is_none());
}

pub fn build_token_bounded_analysis_trie_sorted(
    sequences: &[&[u8]],
) -> TokenBoundedAnalysisTrie {
    TokenBoundedAnalysisTrie {
        nodes: Arc::from(build_byte_trie_sorted(sequences)),
    }
}

fn build_byte_trie_sorted(sequences: &[&[u8]]) -> Vec<ByteTrieNode> {
    debug_assert!(sequences.windows(2).all(|pair| pair[0] <= pair[1]));

    let mut nodes = vec![ByteTrieNode::default()];
    let mut path = vec![0usize];
    let mut previous: &[u8] = &[];
    for &sequence in sequences {
        let lcp = previous
            .iter()
            .zip(sequence.iter())
            .take_while(|(left, right)| left == right)
            .count();
        path.truncate(lcp + 1);
        let mut node = *path.last().expect("root path must remain present");
        for &byte in &sequence[lcp..] {
            let child = nodes.len();
            nodes.push(ByteTrieNode::default());
            nodes[node].children.push((byte, child));
            path.push(child);
            node = child;
        }
        previous = sequence;
    }
    nodes
}

fn filtered_config_groups(
    tokenizer: &Tokenizer,
    config: &[u32],
    active_groups: Option<&[bool]>,
    finalizers: bool,
) -> Vec<usize> {
    let mut groups = Vec::<usize>::new();
    for &state in config {
        if finalizers {
            groups.extend(tokenizer.matched_terminals_iter(state).map(|group| group as usize));
        } else {
            groups.extend(
                tokenizer
                    .possible_future_terminals_iter(state)
                    .map(|group| group as usize),
            );
        }
    }
    groups.retain(|&group| {
        active_groups.is_none_or(|active| active.get(group).copied().unwrap_or(false))
    });
    groups.sort_unstable();
    groups.dedup();
    groups
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenBoundedAnalysisWork {
    pub configurations: usize,
    pub trie_visits: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TokenBoundedAnalysisWorkBudget {
    pub max_configurations: usize,
    pub max_trie_visits: usize,
}

fn ensure_config_transition(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&[u32]>,
    state: u32,
    byte: u8,
    configs: &mut Vec<Box<[u32]>>,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    transitions: &mut Vec<u32>,
    known_transitions: &mut Vec<u8>,
    preseeded_raw_closures: Option<&[u32]>,
    singleton_closures: &SingletonEpsilonClosures,
    active_language: Option<&[bool]>,
    target_marks: &mut [u32],
    target_generation: &mut u32,
) -> u32 {
    let slot = state as usize * 256 + byte as usize;
    if known_transitions[slot] != 0 {
        return transitions[slot];
    }
    known_transitions[slot] = 1;
    if let Some(raw_start_to_view) = preseeded_raw_closures
        && let [source] = configs[state as usize].as_ref()
    {
        let raw_target = raw_transitions.map_or_else(
            || tokenizer.get_transition(*source, byte),
            |transitions| transitions[*source as usize * 256 + byte as usize],
        );
        if raw_target == u32::MAX {
            return u32::MAX;
        }
        let target = raw_start_to_view[raw_target as usize];
        if target != u32::MAX {
            transitions[slot] = target;
            return target;
        }
    }
    let targets = step_epsilon_closed_config_cached(
        tokenizer,
        raw_transitions,
        configs[state as usize].as_ref(),
        byte,
        singleton_closures,
        active_language,
        target_marks,
        target_generation,
    );
    if targets.is_empty() {
        return u32::MAX;
    }
    let target = intern_config(targets, config_ids, configs);
    if transitions.len() < configs.len() * 256 {
        transitions.resize(configs.len() * 256, u32::MAX);
        known_transitions.resize(configs.len() * 256, 0);
    }
    transitions[slot] = target;
    target
}

/// Step a configuration that is already epsilon-closed (possibly after an
/// active-language projection) without closing its source again. The target is
/// exactly the union of the cached singleton epsilon closures reached by direct
/// byte transitions from the closed source configuration.
fn step_epsilon_closed_config_cached(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&[u32]>,
    config: &[u32],
    byte: u8,
    singleton_closures: &SingletonEpsilonClosures,
    active_language: Option<&[bool]>,
    target_marks: &mut [u32],
    target_generation: &mut u32,
) -> Vec<u32> {
    *target_generation = target_generation.wrapping_add(1);
    if *target_generation == 0 {
        target_marks.fill(0);
        *target_generation = 1;
    }
    let generation = *target_generation;
    let mut targets = Vec::<u32>::new();
    for &source in config {
        let direct_target = raw_transitions.map_or_else(
            || tokenizer.get_transition(source, byte),
            |transitions| transitions[source as usize * 256 + byte as usize],
        );
        if direct_target == u32::MAX {
            continue;
        }
        for &target in singleton_closures[direct_target as usize].iter() {
            if active_language.is_some_and(|active| !active[target as usize]) {
                continue;
            }
            let mark = &mut target_marks[target as usize];
            if *mark != generation {
                *mark = generation;
                targets.push(target);
            }
        }
    }
    targets.sort_unstable();
    targets
}

fn expand_trie_from_config(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&[u32]>,
    start_state: u32,
    trie: &[ByteTrieNode],
    configs: &mut Vec<Box<[u32]>>,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    transitions: &mut Vec<u32>,
    known_transitions: &mut Vec<u8>,
    visited: &mut rustc_hash::FxHashSet<(u32, usize)>,
    preseeded_raw_closures: Option<&[u32]>,
    singleton_closures: &SingletonEpsilonClosures,
    active_language: Option<&[bool]>,
    target_marks: &mut [u32],
    target_generation: &mut u32,
) {
    let mut stack = vec![(start_state, 0usize)];
    while let Some((state, node)) = stack.pop() {
        if !visited.insert((state, node)) {
            continue;
        }
        for &(byte, child) in &trie[node].children {
            let target = ensure_config_transition(
                tokenizer,
                raw_transitions,
                state,
                byte,
                configs,
                config_ids,
                transitions,
                known_transitions,
                preseeded_raw_closures,
                singleton_closures,
                active_language,
                target_marks,
                target_generation,
            );
            if target != u32::MAX {
                stack.push((target, child));
            }
        }
    }
}

fn expand_trie_from_config_budgeted(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&[u32]>,
    start_state: u32,
    trie: &[ByteTrieNode],
    configs: &mut Vec<Box<[u32]>>,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    transitions: &mut Vec<u32>,
    known_transitions: &mut Vec<u8>,
    visited: &mut rustc_hash::FxHashSet<(u32, usize)>,
    preseeded_raw_closures: Option<&[u32]>,
    singleton_closures: &SingletonEpsilonClosures,
    active_language: Option<&[bool]>,
    target_marks: &mut [u32],
    target_generation: &mut u32,
    budget: TokenBoundedAnalysisWorkBudget,
    prior_trie_visits: usize,
) -> Result<(), TokenBoundedAnalysisWork> {
    const CONFIG_CHECK_INTERVAL: usize = 4096;
    debug_assert!(CONFIG_CHECK_INTERVAL.is_power_of_two());
    let first_disallowed_trie_visit = budget.max_trie_visits.saturating_add(1);
    let mut stack = vec![(start_state, 0usize)];
    while let Some((state, node)) = stack.pop() {
        if !visited.insert((state, node)) {
            continue;
        }
        let trie_visits = prior_trie_visits + visited.len();
        if trie_visits == first_disallowed_trie_visit
            || (trie_visits & (CONFIG_CHECK_INTERVAL - 1) == 0
                && configs.len() > budget.max_configurations)
        {
            return Err(TokenBoundedAnalysisWork {
                configurations: configs.len(),
                trie_visits,
            });
        }
        for &(byte, child) in &trie[node].children {
            let target = ensure_config_transition(
                tokenizer,
                raw_transitions,
                state,
                byte,
                configs,
                config_ids,
                transitions,
                known_transitions,
                preseeded_raw_closures,
                singleton_closures,
                active_language,
                target_marks,
                target_generation,
            );
            if target != u32::MAX {
                stack.push((target, child));
            }
        }
    }
    if configs.len() > budget.max_configurations {
        return Err(TokenBoundedAnalysisWork {
            configurations: configs.len(),
            trie_visits: prior_trie_visits + visited.len(),
        });
    }
    Ok(())
}

/// Expand all reachable configuration × trie-node pairs by propagating a
/// unique configuration frontier from each trie node into each child. A trie
/// node has exactly one parent, so the child frontier can be deduplicated with
/// a reusable generation-mark table instead of a global hash set of
/// `(configuration, trie_node)` pairs.
fn expand_trie_frontiers(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&[u32]>,
    start_states: &[u32],
    trie: &[ByteTrieNode],
    configs: &mut Vec<Box<[u32]>>,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    transitions: &mut Vec<u32>,
    known_transitions: &mut Vec<u8>,
    preseeded_raw_closures: Option<&[u32]>,
    singleton_closures: &SingletonEpsilonClosures,
    active_language: Option<&[bool]>,
    target_marks: &mut [u32],
    target_generation: &mut u32,
    budget: Option<TokenBoundedAnalysisWorkBudget>,
    prior_trie_visits: usize,
) -> Result<usize, TokenBoundedAnalysisWork> {
    if trie.is_empty() || start_states.is_empty() {
        return Ok(0);
    }

    let mut node_state_start = vec![0u32; trie.len()];
    let mut node_state_len = vec![0u32; trie.len()];
    let mut product_states = Vec::<u32>::with_capacity(start_states.len());
    product_states.extend_from_slice(start_states);
    node_state_len[0] = start_states.len() as u32;

    let mut config_marks = vec![0u32; configs.len().max(1)];
    let mut config_generation = 0u32;
    let mut trie_visits = 0usize;
    for node_index in 0..trie.len() {
        let state_start = node_state_start[node_index] as usize;
        let state_len = node_state_len[node_index] as usize;
        if state_len == 0 {
            continue;
        }
        if let Some(budget) = budget {
            let next_total_visits = prior_trie_visits + trie_visits + state_len;
            if next_total_visits > budget.max_trie_visits {
                return Err(TokenBoundedAnalysisWork {
                    configurations: configs.len(),
                    trie_visits: budget.max_trie_visits.saturating_add(1),
                });
            }
            if configs.len() > budget.max_configurations {
                return Err(TokenBoundedAnalysisWork {
                    configurations: configs.len(),
                    trie_visits: prior_trie_visits + trie_visits,
                });
            }
        }
        trie_visits += state_len;

        // The root frontier is typically every raw tokenizer closure and may
        // have dozens or hundreds of first-byte children.  Scanning
        // `root_children × configurations` performs millions of absent-byte
        // lookups for L1 workloads.  When a configuration is a singleton
        // epsilon-closed physical state, iterate its actual sparse byte edges
        // once and route only edges represented at the trie root.  Non-singleton
        // configurations retain the generic exact path below.
        if node_index == 0
            && preseeded_raw_closures.is_some()
            && trie[node_index].children.len() > 1
        {
            let raw_start_to_view = preseeded_raw_closures.expect("checked above");
            let root_children = &trie[node_index].children;
            let mut child_slot_by_byte = [u16::MAX; 256];
            for (slot, &(byte, _)) in root_children.iter().enumerate() {
                child_slot_by_byte[byte as usize] = slot as u16;
            }
            let mut child_frontiers = vec![Vec::<u32>::new(); root_children.len()];

            for state_offset in 0..state_len {
                let state = product_states[state_start + state_offset];
                if let [source] = configs[state as usize].as_ref() {
                    for (byte, _) in tokenizer.transitions_from(*source) {
                        let child_slot = child_slot_by_byte[byte as usize];
                        if child_slot == u16::MAX {
                            continue;
                        }
                        // Preserve the sparse root scan, but use the exact
                        // configuration transition. A physical raw target may
                        // not have been preseeded as a raw start, so directly
                        // indexing `raw_start_to_view` can incorrectly turn a
                        // live transition into DEAD.
                        let target = ensure_config_transition(
                            tokenizer,
                            raw_transitions,
                            state,
                            byte,
                            configs,
                            config_ids,
                            transitions,
                            known_transitions,
                            None,
                            singleton_closures,
                            active_language,
                            target_marks,
                            target_generation,
                        );
                        if target == u32::MAX {
                            continue;
                        }
                        child_frontiers[child_slot as usize].push(target);
                    }
                    continue;
                }

                for (child_slot, &(byte, _)) in root_children.iter().enumerate() {
                    let target = ensure_config_transition(
                        tokenizer,
                        raw_transitions,
                        state,
                        byte,
                        configs,
                        config_ids,
                        transitions,
                        known_transitions,
                        preseeded_raw_closures,
                        singleton_closures,
                        active_language,
                        target_marks,
                        target_generation,
                    );
                    if target != u32::MAX {
                        child_frontiers[child_slot].push(target);
                    }
                }
            }

            for ((_, child), frontier) in root_children.iter().zip(child_frontiers.iter_mut()) {
                frontier.sort_unstable();
                frontier.dedup();
                let child_start = product_states.len();
                product_states.extend_from_slice(frontier);
                debug_assert_eq!(node_state_len[*child], 0, "trie child has multiple parents");
                node_state_start[*child] = child_start as u32;
                node_state_len[*child] = frontier.len() as u32;
            }
            continue;
        }

        for &(byte, child) in &trie[node_index].children {
            config_generation = config_generation.wrapping_add(1);
            if config_generation == 0 {
                config_marks.fill(0);
                config_generation = 1;
            }
            let child_start = product_states.len();
            for state_offset in 0..state_len {
                let state = product_states[state_start + state_offset];
                let target = ensure_config_transition(
                    tokenizer,
                    raw_transitions,
                    state,
                    byte,
                    configs,
                    config_ids,
                    transitions,
                    known_transitions,
                    preseeded_raw_closures,
                    singleton_closures,
                    active_language,
                    target_marks,
                    target_generation,
                );
                if target == u32::MAX {
                    continue;
                }
                if config_marks.len() < configs.len() {
                    config_marks.resize(configs.len(), 0);
                }
                let mark = &mut config_marks[target as usize];
                if *mark != config_generation {
                    *mark = config_generation;
                    product_states.push(target);
                }
            }
            debug_assert_eq!(node_state_len[child], 0, "trie child has multiple parents");
            node_state_start[child] = child_start as u32;
            node_state_len[child] = (product_states.len() - child_start) as u32;
        }
    }

    if let Some(budget) = budget
        && configs.len() > budget.max_configurations
    {
        return Err(TokenBoundedAnalysisWork {
            configurations: configs.len(),
            trie_visits: prior_trie_visits + trie_visits,
        });
    }
    Ok(trie_visits)
}

#[derive(Clone)]
pub struct TokenBoundedAnalysisTopology {
    configurations: Arc<[Box<[u32]>]>,
    start_state: usize,
    transitions: Arc<[u32]>,
    raw_start_to_view: Arc<[u32]>,
}

fn materialize_bounded_analysis_view(
    tokenizer: &Tokenizer,
    active_groups: Option<&[bool]>,
    configurations: &[Box<[u32]>],
    start_state: usize,
    transitions: Arc<[u32]>,
    raw_start_to_view: Vec<u32>,
) -> BoundedAnalysisView {
    let raw_state_count = tokenizer.num_states() as usize;
    let raw_state_outputs = (0..raw_state_count)
        .map(|state| FlatDfaState {
            finalizers: filtered_config_groups(
                tokenizer,
                std::slice::from_ref(&(state as u32)),
                active_groups,
                true,
            ),
            possible_future_group_ids: filtered_config_groups(
                tokenizer,
                std::slice::from_ref(&(state as u32)),
                active_groups,
                false,
            ),
        })
        .collect::<Vec<_>>();
    let states = configurations
        .iter()
        .map(|config| {
            if let [state] = config.as_ref() {
                return raw_state_outputs[*state as usize].clone();
            }
            let mut finalizers = Vec::<usize>::new();
            let mut possible_future_group_ids = Vec::<usize>::new();
            for &state in config.iter() {
                let output = &raw_state_outputs[state as usize];
                finalizers.extend_from_slice(&output.finalizers);
                possible_future_group_ids.extend_from_slice(&output.possible_future_group_ids);
            }
            finalizers.sort_unstable();
            finalizers.dedup();
            possible_future_group_ids.sort_unstable();
            possible_future_group_ids.dedup();
            FlatDfaState {
                finalizers,
                possible_future_group_ids,
            }
        })
        .collect();
    BoundedAnalysisView {
        tokenizer_view: TokenizerView {
            flat_dfa: FlatDfa {
                states,
                start_state,
                transitions,
            },
        },
        raw_start_to_view,
    }
}

impl TokenBoundedAnalysisTopology {
    pub fn state_count(&self) -> usize {
        self.configurations.len()
    }

    /// Materialize a topology that was already constructed with the same
    /// active-language projection. Re-projecting such a topology would be
    /// exact but wastefully scan and rebuild its full dense transition table.
    fn materialize_already_projected(
        &self,
        tokenizer: &Tokenizer,
        active_groups: &[bool],
    ) -> BoundedAnalysisView {
        materialize_bounded_analysis_view(
            tokenizer,
            Some(active_groups),
            &self.configurations,
            self.start_state,
            Arc::clone(&self.transitions),
            self.raw_start_to_view.to_vec(),
        )
    }

    pub fn materialize(
        &self,
        tokenizer: &Tokenizer,
        active_groups: Option<&[bool]>,
    ) -> BoundedAnalysisView {
        let Some(active_language) = raw_active_language_states(tokenizer, active_groups) else {
            return materialize_bounded_analysis_view(
                tokenizer,
                active_groups,
                &self.configurations,
                self.start_state,
                Arc::clone(&self.transitions),
                self.raw_start_to_view.to_vec(),
            );
        };

        // The token topology is intentionally built once for a superset of
        // terminal masks. Project raw NFA members when the topology is consumed
        // so inactive-only lexer structure cannot distinguish active-language
        // states. Since an inactive-language state has no active-language
        // successor, projecting after each prebuilt transition is equivalent to
        // projecting its source before taking that transition.
        let mut projected_ids = FxHashMap::<Vec<u32>, u32>::default();
        let mut projected_configs = Vec::<Box<[u32]>>::new();
        let projected_state_by_topology_state = self
            .configurations
            .iter()
            .map(|config| {
                let projected = config
                    .iter()
                    .copied()
                    .filter(|&raw| active_language[raw as usize])
                    .collect::<Vec<_>>();
                intern_config(projected, &mut projected_ids, &mut projected_configs)
            })
            .collect::<Vec<_>>();

        let mut projected_transitions =
            vec![u32::MAX; projected_configs.len().saturating_mul(256)];
        for topology_source in 0..self.configurations.len() {
            let source = projected_state_by_topology_state[topology_source] as usize;
            if projected_configs[source].is_empty() {
                continue;
            }
            let topology_row = topology_source * 256;
            for byte in 0..256 {
                let topology_target = self.transitions[topology_row + byte];
                if topology_target == u32::MAX {
                    continue;
                }
                let target = projected_state_by_topology_state[topology_target as usize];
                if projected_configs[target as usize].is_empty() {
                    continue;
                }
                let slot = source * 256 + byte;
                let previous = projected_transitions[slot];
                debug_assert!(
                    previous == u32::MAX || previous == target,
                    "equal projected NFA configs had different projected byte successors",
                );
                projected_transitions[slot] = target;
            }
        }

        let raw_start_to_view = self
            .raw_start_to_view
            .iter()
            .map(|&state| {
                (state != u32::MAX)
                    .then(|| projected_state_by_topology_state[state as usize])
                    .unwrap_or(u32::MAX)
            })
            .collect::<Vec<_>>();
        let start_state = projected_state_by_topology_state[self.start_state] as usize;
        materialize_bounded_analysis_view(
            tokenizer,
            active_groups,
            &projected_configs,
            start_state,
            Arc::from(projected_transitions),
            raw_start_to_view,
        )
    }
}

fn build_bounded_analysis_topology_impl(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&[u32]>,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    combine_start_states: bool,
    factor_common_first_byte: bool,
    include_reset_suffixes: bool,
    tokens_are_sorted: bool,
    active_groups: Option<&[bool]>,
    budget: Option<TokenBoundedAnalysisWorkBudget>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> Result<(TokenBoundedAnalysisTopology, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    let use_frontier_expansion =
        std::env::var_os("GLRMASK_DISABLE_TRIE_FRONTIER_EXPANSION").is_none();
    build_bounded_analysis_topology_impl_with_expansion(
        tokenizer,
        raw_transitions,
        raw_start_states,
        tokens,
        combine_start_states,
        factor_common_first_byte,
        include_reset_suffixes,
        tokens_are_sorted,
        active_groups,
        budget,
        prebuilt_token_trie,
        use_frontier_expansion,
    )
}

fn build_bounded_analysis_topology_impl_with_expansion(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&[u32]>,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    combine_start_states: bool,
    factor_common_first_byte: bool,
    include_reset_suffixes: bool,
    tokens_are_sorted: bool,
    active_groups: Option<&[bool]>,
    budget: Option<TokenBoundedAnalysisWorkBudget>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
    use_frontier_expansion: bool,
) -> Result<(TokenBoundedAnalysisTopology, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
    let total_started_at = profile_timing.then(std::time::Instant::now);
    let setup_started_at = profile_timing.then(std::time::Instant::now);
    let raw_state_count = tokenizer.num_states() as usize;
    let mut config_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut configs = Vec::<Box<[u32]>>::new();
    let mut raw_start_to_view = vec![u32::MAX; raw_state_count];
    let singleton_closures = tokenizer.all_singleton_epsilon_closures();
    let active_language = raw_active_language_states(tokenizer, active_groups);
    let project_closure = |closure: &[u32]| {
        if let Some(active_language) = active_language.as_deref() {
            closure
                .iter()
                .copied()
                .filter(|&state| active_language[state as usize])
                .collect::<Vec<_>>()
        } else {
            closure.to_vec()
        }
    };

    let start_closure = project_closure(&singleton_closures[tokenizer.initial_state_id() as usize]);
    let start_state = intern_config(start_closure, &mut config_ids, &mut configs);
    if combine_start_states {
        let mut closure = Vec::<u32>::new();
        for &raw_state in raw_start_states {
            assert!(raw_state < raw_state_count, "invalid raw NFA analysis seed");
            if let Some(active_language) = active_language.as_deref() {
                closure.extend(
                    singleton_closures[raw_state]
                        .iter()
                        .copied()
                        .filter(|&state| active_language[state as usize]),
                );
            } else {
                closure.extend_from_slice(&singleton_closures[raw_state]);
            }
        }
        closure.sort_unstable();
        closure.dedup();
        let state = intern_config(closure, &mut config_ids, &mut configs);
        for &raw_state in raw_start_states {
            raw_start_to_view[raw_state] = state;
        }
    } else {
        for &raw_state in raw_start_states {
            assert!(raw_state < raw_state_count, "invalid raw NFA analysis seed");
            let closure = project_closure(&singleton_closures[raw_state]);
            let state = intern_config(closure, &mut config_ids, &mut configs);
            raw_start_to_view[raw_state] = state;
        }
    }
    if budget.is_some_and(|budget| configs.len() > budget.max_configurations) {
        return Err(TokenBoundedAnalysisWork {
            configurations: configs.len(),
            trie_visits: 0,
        });
    }
    let mut transitions = vec![u32::MAX; configs.len() * 256];
    let mut known_transitions = vec![0u8; transitions.len()];
    let setup_ms = setup_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let trie_started_at = profile_timing.then(std::time::Instant::now);
    let common_first_byte = factor_common_first_byte
        .then(|| {
            let first = tokens.first()?.first().copied()?;
            tokens
                .iter()
                .all(|token| token.len() > 1 && token.first().copied() == Some(first))
                .then_some(first)
        })
        .flatten();
    let owned_token_trie;
    let token_trie: &[ByteTrieNode] = if common_first_byte.is_none()
        && let Some(prebuilt) = prebuilt_token_trie
    {
        prebuilt.nodes.as_ref()
    } else {
        owned_token_trie = if common_first_byte.is_some() {
            if tokens_are_sorted {
                let suffixes = tokens.iter().map(|token| &token[1..]).collect::<Vec<_>>();
                build_byte_trie_sorted(&suffixes)
            } else {
                build_byte_trie(tokens.iter().map(|token| &token[1..]))
            }
        } else if tokens_are_sorted {
            build_byte_trie_sorted(tokens)
        } else {
            build_byte_trie(tokens.iter().copied())
        };
        &owned_token_trie
    };
    let suffix_trie = include_reset_suffixes.then(|| {
        build_byte_trie(
            tokens
                .iter()
                .flat_map(|token| (0..token.len()).map(move |offset| &token[offset..])),
        )
    });
    let trie_ms = trie_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let expansion_started_at = profile_timing.then(std::time::Instant::now);
    let mut target_marks = vec![0u32; raw_state_count];
    let mut target_generation = 0u32;
    let mut seeded_configs = raw_start_states
        .iter()
        .map(|&raw| raw_start_to_view[raw])
        .collect::<Vec<_>>();
    seeded_configs.sort_unstable();
    seeded_configs.dedup();
    if let Some(first_byte) = common_first_byte {
        let mut advanced_seeded_configs = Vec::with_capacity(seeded_configs.len());
        for state in seeded_configs {
            let target = ensure_config_transition(
                tokenizer,
                raw_transitions,
                state,
                first_byte,
                &mut configs,
                &mut config_ids,
                &mut transitions,
                &mut known_transitions,
                (!combine_start_states).then_some(raw_start_to_view.as_slice()),
                singleton_closures.as_ref(),
                active_language.as_deref(),
                &mut target_marks,
                &mut target_generation,
            );
            if target != u32::MAX {
                advanced_seeded_configs.push(target);
            }
        }
        advanced_seeded_configs.sort_unstable();
        advanced_seeded_configs.dedup();
        seeded_configs = advanced_seeded_configs;
        if budget.is_some_and(|budget| configs.len() > budget.max_configurations) {
            return Err(TokenBoundedAnalysisWork {
                configurations: configs.len(),
                trie_visits: 0,
            });
        }
    }
    let preseeded_raw_closures = (!combine_start_states).then_some(raw_start_to_view.as_slice());
    let (token_trie_visits, suffix_trie_visits) = if use_frontier_expansion {
        let token_trie_visits = expand_trie_frontiers(
            tokenizer,
            raw_transitions,
            &seeded_configs,
            token_trie,
            &mut configs,
            &mut config_ids,
            &mut transitions,
            &mut known_transitions,
            preseeded_raw_closures,
            singleton_closures.as_ref(),
            active_language.as_deref(),
            &mut target_marks,
            &mut target_generation,
            budget,
            0,
        )?;
        let suffix_trie_visits = if let Some(suffix_trie) = suffix_trie.as_ref() {
            expand_trie_frontiers(
                tokenizer,
                raw_transitions,
                std::slice::from_ref(&start_state),
                suffix_trie,
                &mut configs,
                &mut config_ids,
                &mut transitions,
                &mut known_transitions,
                None,
                singleton_closures.as_ref(),
                active_language.as_deref(),
                &mut target_marks,
                &mut target_generation,
                budget,
                token_trie_visits,
            )?
        } else {
            0
        };
        (token_trie_visits, suffix_trie_visits)
    } else {
        let mut token_visited = rustc_hash::FxHashSet::<(u32, usize)>::default();
        for state in seeded_configs.iter().copied() {
            if let Some(budget) = budget {
                expand_trie_from_config_budgeted(
                    tokenizer,
                    raw_transitions,
                    state,
                    token_trie,
                    &mut configs,
                    &mut config_ids,
                    &mut transitions,
                    &mut known_transitions,
                    &mut token_visited,
                    preseeded_raw_closures,
                    singleton_closures.as_ref(),
                    active_language.as_deref(),
                    &mut target_marks,
                    &mut target_generation,
                    budget,
                    0,
                )?;
            } else {
                expand_trie_from_config(
                    tokenizer,
                    raw_transitions,
                    state,
                    token_trie,
                    &mut configs,
                    &mut config_ids,
                    &mut transitions,
                    &mut known_transitions,
                    &mut token_visited,
                    preseeded_raw_closures,
                    singleton_closures.as_ref(),
                    active_language.as_deref(),
                    &mut target_marks,
                    &mut target_generation,
                );
            }
        }
        let mut suffix_visited = rustc_hash::FxHashSet::<(u32, usize)>::default();
        if let Some(suffix_trie) = suffix_trie.as_ref() {
            if let Some(budget) = budget {
                expand_trie_from_config_budgeted(
                    tokenizer,
                    raw_transitions,
                    start_state,
                    suffix_trie,
                    &mut configs,
                    &mut config_ids,
                    &mut transitions,
                    &mut known_transitions,
                    &mut suffix_visited,
                    None,
                    singleton_closures.as_ref(),
                    active_language.as_deref(),
                    &mut target_marks,
                    &mut target_generation,
                    budget,
                    token_visited.len(),
                )?;
            } else {
                expand_trie_from_config(
                    tokenizer,
                    raw_transitions,
                    start_state,
                    suffix_trie,
                    &mut configs,
                    &mut config_ids,
                    &mut transitions,
                    &mut known_transitions,
                    &mut suffix_visited,
                    None,
                    singleton_closures.as_ref(),
                    active_language.as_deref(),
                    &mut target_marks,
                    &mut target_generation,
                );
            }
        }
        (token_visited.len(), suffix_visited.len())
    };

    let expansion_ms = expansion_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let work = TokenBoundedAnalysisWork {
        configurations: configs.len(),
        trie_visits: token_trie_visits + suffix_trie_visits,
    };
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][token_bounded_topology] raw_states={} tokens={} active_projection={} combine_starts={} common_first={} trie_nodes={} configs={} trie_visits={} setup_ms={:.3} trie_ms={:.3} expansion_ms={:.3} total_ms={:.3}",
            raw_state_count,
            tokens.len(),
            active_groups.is_some(),
            combine_start_states,
            common_first_byte.is_some(),
            token_trie.len(),
            work.configurations,
            work.trie_visits,
            setup_ms,
            trie_ms,
            expansion_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok((
        TokenBoundedAnalysisTopology {
            configurations: Arc::from(configs),
            start_state: start_state as usize,
            transitions: Arc::from(transitions),
            raw_start_to_view: Arc::from(raw_start_to_view),
        },
        work,
    ))
}

fn build_bounded_analysis_view_inner(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
    combine_start_states: bool,
) -> BoundedAnalysisView {
    build_bounded_analysis_topology_impl(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        combine_start_states,
        true,
        true,
        false,
        None,
        None,
        None,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize(tokenizer, active_groups)
}

fn build_bounded_analysis_view_impl(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
    combine_start_states: bool,
    factor_common_first_byte: bool,
    include_reset_suffixes: bool,
) -> BoundedAnalysisView {
    build_bounded_analysis_topology_impl(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        combine_start_states,
        factor_common_first_byte,
        include_reset_suffixes,
        false,
        None,
        None,
        None,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize(tokenizer, active_groups)
}

pub fn build_bounded_analysis_view(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
) -> BoundedAnalysisView {
    build_bounded_analysis_view_with_trie(
        tokenizer,
        raw_start_states,
        tokens,
        active_groups,
        None,
    )
}

pub fn build_bounded_analysis_view_with_trie(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> BoundedAnalysisView {
    // Invariant: analysis views used by vocabulary equivalence must materialize
    // all token suffix trajectories from start_state (whose epsilon closure covers
    // all component reset roots) so that restart/suffix evaluations encounter valid
    // transitions.
    build_bounded_analysis_topology_impl(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        false,
        true,
        true, // include_reset_suffixes
        false,
        None,
        None,
        prebuilt_token_trie,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize(tokenizer, active_groups)
}

pub fn build_bounded_analysis_view_from_combined_starts(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
) -> BoundedAnalysisView {
    build_bounded_analysis_view_from_combined_starts_with_trie(
        tokenizer,
        raw_start_states,
        tokens,
        active_groups,
        None,
    )
}

pub fn build_bounded_analysis_view_from_combined_starts_with_trie(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> BoundedAnalysisView {
    // Invariant: analysis views used by vocabulary equivalence must materialize
    // all token suffix trajectories from start_state (whose epsilon closure covers
    // all component reset roots) so that restart/suffix evaluations encounter valid
    // transitions.
    build_bounded_analysis_topology_impl(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        true,
        true,
        true, // include_reset_suffixes
        false,
        None,
        None,
        prebuilt_token_trie,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize(tokenizer, active_groups)
}

/// Build the combined-start bounded view needed for exact L2P prefix scans.
/// Unlike the general bounded view, this does not materialize reset-state
/// transitions for every token suffix; callers must evaluate reset suffixes
/// separately.
pub fn build_bounded_analysis_view_from_combined_starts_prefix_only_with_trie(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> BoundedAnalysisView {
    build_bounded_analysis_topology_impl(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        true,
        true,
        false,
        false,
        None,
        None,
        prebuilt_token_trie,
    )
    .expect("unbounded prefix-only token analysis topology build")
    .0
    .materialize(tokenizer, active_groups)
}

pub fn build_token_bounded_analysis_topology(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
) -> TokenBoundedAnalysisTopology {
    build_bounded_analysis_topology_impl(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        false,
        true,
        false,
        false,
        None,
        None,
        None,
    )
    .expect("unbounded token analysis topology build")
    .0
}

pub fn build_token_bounded_analysis_view_projected(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
) -> BoundedAnalysisView {
    build_token_bounded_analysis_view_projected_with_order(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        active_groups,
        false,
        None,
    )
}

pub fn build_token_bounded_analysis_view_projected_sorted(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
) -> BoundedAnalysisView {
    build_token_bounded_analysis_view_projected_with_order(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        active_groups,
        true,
        None,
    )
}

pub fn build_token_bounded_analysis_view_projected_sorted_with_raw_transitions(
    tokenizer: &Tokenizer,
    raw_transitions: &[u32],
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
) -> BoundedAnalysisView {
    build_token_bounded_analysis_view_projected_sorted_with_raw_transitions_and_trie(
        tokenizer,
        raw_transitions,
        raw_start_states,
        tokens,
        active_groups,
        None,
    )
}

pub fn build_token_bounded_analysis_view_projected_sorted_with_raw_transitions_and_trie(
    tokenizer: &Tokenizer,
    raw_transitions: &[u32],
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> BoundedAnalysisView {
    build_token_bounded_analysis_view_projected_with_order(
        tokenizer,
        Some(raw_transitions),
        raw_start_states,
        tokens,
        active_groups,
        true,
        prebuilt_token_trie,
    )
}

fn build_token_bounded_analysis_view_projected_with_order(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&[u32]>,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    tokens_are_sorted: bool,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> BoundedAnalysisView {
    build_bounded_analysis_topology_impl(
        tokenizer,
        raw_transitions,
        raw_start_states,
        tokens,
        false,
        true,
        false,
        tokens_are_sorted,
        Some(active_groups),
        None,
        prebuilt_token_trie,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize_already_projected(tokenizer, active_groups)
}

pub fn try_build_token_bounded_analysis_view_projected(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    budget: TokenBoundedAnalysisWorkBudget,
) -> Result<(BoundedAnalysisView, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    try_build_token_bounded_analysis_view_projected_with_order(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        active_groups,
        budget,
        false,
        None,
    )
}

pub fn try_build_token_bounded_analysis_view_projected_sorted(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    budget: TokenBoundedAnalysisWorkBudget,
) -> Result<(BoundedAnalysisView, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    try_build_token_bounded_analysis_view_projected_with_order(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        active_groups,
        budget,
        true,
        None,
    )
}

pub fn try_build_token_bounded_analysis_view_projected_sorted_with_raw_transitions(
    tokenizer: &Tokenizer,
    raw_transitions: &[u32],
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    budget: TokenBoundedAnalysisWorkBudget,
) -> Result<(BoundedAnalysisView, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    try_build_token_bounded_analysis_view_projected_sorted_with_raw_transitions_and_trie(
        tokenizer,
        raw_transitions,
        raw_start_states,
        tokens,
        active_groups,
        budget,
        None,
    )
}

pub fn try_build_token_bounded_analysis_view_projected_sorted_with_raw_transitions_and_trie(
    tokenizer: &Tokenizer,
    raw_transitions: &[u32],
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    budget: TokenBoundedAnalysisWorkBudget,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> Result<(BoundedAnalysisView, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    try_build_token_bounded_analysis_view_projected_with_order(
        tokenizer,
        Some(raw_transitions),
        raw_start_states,
        tokens,
        active_groups,
        budget,
        true,
        prebuilt_token_trie,
    )
}

fn try_build_token_bounded_analysis_view_projected_with_order(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&[u32]>,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: &[bool],
    budget: TokenBoundedAnalysisWorkBudget,
    tokens_are_sorted: bool,
    prebuilt_token_trie: Option<&TokenBoundedAnalysisTrie>,
) -> Result<(BoundedAnalysisView, TokenBoundedAnalysisWork), TokenBoundedAnalysisWork> {
    let (topology, work) = build_bounded_analysis_topology_impl(
        tokenizer,
        raw_transitions,
        raw_start_states,
        tokens,
        false,
        true,
        false,
        tokens_are_sorted,
        Some(active_groups),
        Some(budget),
        prebuilt_token_trie,
    )?;
    Ok((
        topology.materialize_already_projected(tokenizer, active_groups),
        work,
    ))
}

pub fn build_token_bounded_analysis_view(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
) -> BoundedAnalysisView {
    build_token_bounded_analysis_topology(tokenizer, raw_start_states, tokens)
        .materialize(tokenizer, active_groups)
}

pub fn build_token_bounded_analysis_view_from_combined_starts(
    tokenizer: &Tokenizer,
    raw_start_states: &[usize],
    tokens: &[&[u8]],
    active_groups: Option<&[bool]>,
) -> BoundedAnalysisView {
    build_bounded_analysis_topology_impl(
        tokenizer,
        None,
        raw_start_states,
        tokens,
        true,
        true,
        false,
        false,
        None,
        None,
        None,
    )
    .expect("unbounded token analysis topology build")
    .0
    .materialize(tokenizer, active_groups)
}

fn intern_config(
    states: Vec<u32>,
    config_ids: &mut FxHashMap<Vec<u32>, u32>,
    configs: &mut Vec<Box<[u32]>>,
) -> u32 {
    if let Some(&id) = config_ids.get(&states) {
        return id;
    }
    let id = configs.len() as u32;
    config_ids.insert(states.clone(), id);
    configs.push(states.into_boxed_slice());
    id
}

fn candidate_partition(
    num_states: usize,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> (Vec<Vec<u32>>, Vec<usize>, Vec<usize>) {
    let mut members = Vec::<Vec<u32>>::new();
    let mut representatives = Vec::<usize>::new();
    let mut raw_to_candidate = vec![usize::MAX; num_states];

    if let Some(map) = initial_state_map {
        for originals in &map.internal_to_originals {
            let mut candidate_members = Vec::with_capacity(originals.len());
            for &raw in originals {
                let raw = raw as usize;
                if raw < num_states && raw_to_candidate[raw] == usize::MAX {
                    candidate_members.push(raw as u32);
                }
            }
            if candidate_members.is_empty() {
                continue;
            }
            let candidate = members.len();
            let representative = candidate_members[0] as usize;
            for &raw in &candidate_members {
                raw_to_candidate[raw as usize] = candidate;
            }
            representatives.push(representative);
            members.push(candidate_members);
        }
    }

    for raw in 0..num_states {
        if raw_to_candidate[raw] != usize::MAX {
            continue;
        }
        let candidate = members.len();
        raw_to_candidate[raw] = candidate;
        representatives.push(raw);
        members.push(vec![raw as u32]);
    }

    (members, representatives, raw_to_candidate)
}

fn observation_words(
    tokenizer: &Tokenizer,
    states: &[u32],
    active_groups: Option<&[bool]>,
) -> Vec<u64> {
    let terminal_count = tokenizer.num_terminals() as usize;
    let word_count = terminal_count.div_ceil(64);
    let mut matched = vec![0u64; word_count];
    let mut future = vec![0u64; word_count];
    for &state in states {
        for terminal in tokenizer.matched_terminals_iter(state) {
            let terminal = terminal as usize;
            if active_groups.is_none_or(|active| active.get(terminal).copied().unwrap_or(false)) {
                matched[terminal >> 6] |= 1u64 << (terminal & 63);
            }
        }
        for terminal in tokenizer.possible_future_terminals_iter(state) {
            let terminal = terminal as usize;
            if active_groups.is_none_or(|active| active.get(terminal).copied().unwrap_or(false)) {
                future[terminal >> 6] |= 1u64 << (terminal & 63);
            }
        }
    }
    matched.extend(future);
    matched
}

fn same_partition(left: &[u32], right: &[u32]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut left_to_right = FxHashMap::<u32, u32>::default();
    let mut right_to_left = FxHashMap::<u32, u32>::default();
    for (&left, &right) in left.iter().zip(right) {
        if left_to_right.insert(left, right).is_some_and(|mapped| mapped != right) {
            return false;
        }
        if right_to_left.insert(right, left).is_some_and(|mapped| mapped != left) {
            return false;
        }
    }
    true
}

fn build_state_map(
    candidate_members: &[Vec<u32>],
    candidate_representatives: &[usize],
    candidate_classes: &[u32],
    num_states: usize,
) -> ManyToOneIdMap {
    let class_label_count = candidate_classes
        .iter()
        .copied()
        .max()
        .map_or(0, |class| class as usize + 1);
    let mut class_to_internal = vec![u32::MAX; class_label_count];
    let mut original_to_internal = vec![u32::MAX; num_states];
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representative_original_ids = Vec::<u32>::new();

    for ((members, &representative), &class) in candidate_members
        .iter()
        .zip(candidate_representatives)
        .zip(candidate_classes)
    {
        let class = class as usize;
        let internal = if class_to_internal[class] == u32::MAX {
            let internal = internal_to_originals.len() as u32;
            class_to_internal[class] = internal;
            internal_to_originals.push(Vec::new());
            representative_original_ids.push(representative as u32);
            internal
        } else {
            class_to_internal[class]
        };
        let bucket = &mut internal_to_originals[internal as usize];
        for &raw in members {
            original_to_internal[raw as usize] = internal;
            bucket.push(raw);
        }
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn prebuilt_candidate_target_sets(
    configurations: &[Box<[u32]>],
    raw_to_candidate: &[usize],
    raw_active_language: Option<&[bool]>,
) -> Vec<SmallVec<[u32; 4]>> {
    configurations
        .iter()
        .map(|config| {
            let mut candidates = SmallVec::<[u32; 4]>::new();
            candidates.extend(config.iter().filter_map(|&raw| {
                if raw_active_language.is_some_and(|active| !active[raw as usize]) {
                    None
                } else {
                    Some(raw_to_candidate[raw as usize] as u32)
                }
            }));
            candidates.sort_unstable();
            candidates.dedup();
            candidates
        })
        .collect()
}

fn prebuilt_candidate_signature(
    candidate: usize,
    class_for_candidate: &[u32],
    start_configs: &[u32],
    candidate_sets_by_config: &[SmallVec<[u32; 4]>],
    edge_offsets: &[u32],
    edges: &[(u8, u32)],
) -> SmallVec<[u32; 64]> {
    let config = start_configs[candidate] as usize;
    if candidate_sets_by_config[config].is_empty() {
        return SmallVec::new();
    }
    let edge_start = edge_offsets[config] as usize;
    let edge_end = edge_offsets[config + 1] as usize;
    let mut signature = SmallVec::<[u32; 64]>::new();
    let mut target_classes = SmallVec::<[u32; 8]>::new();
    for &(byte, target_config) in &edges[edge_start..edge_end] {
        let target_candidates = &candidate_sets_by_config[target_config as usize];
        if target_candidates.is_empty() {
            continue;
        }
        target_classes.clear();
        target_classes.extend(
            target_candidates
                .iter()
                .map(|&target| class_for_candidate[target as usize]),
        );
        target_classes.sort_unstable();
        target_classes.dedup();
        signature.push(byte as u32 + 1);
        signature.push(target_classes.len() as u32);
        signature.extend(target_classes.iter().map(|&class| class + 1));
    }
    signature
}

/// Compute the same stable set-valued NFA partition as the synchronous Moore
/// recurrence, but only revisit source blocks whose signatures can change.
///
/// A block split changes the class id of only the moved target candidates.
/// Therefore only source candidates with an edge to one of those moved targets
/// can acquire a different transition-class-set signature. Repartitioning just
/// those predecessor blocks reaches the same coarsest stable refinement while
/// avoiding one whole-graph pass per byte of a long lexer chain.
fn refine_prebuilt_sparse_powerset_worklist(
    initial_classes: &[u32],
    start_configs: &[u32],
    configurations: &[Box<[u32]>],
    raw_to_candidate: &[usize],
    raw_active_language: Option<&[bool]>,
    edge_offsets: &[u32],
    edges: &[(u8, u32)],
) -> Vec<u32> {
    let num_candidates = initial_classes.len();
    if num_candidates <= 1 {
        return initial_classes.to_vec();
    }

    let candidate_sets_by_config = prebuilt_candidate_target_sets(
        configurations,
        raw_to_candidate,
        raw_active_language,
    );

    let initial_class_count = initial_classes
        .iter()
        .copied()
        .max()
        .map_or(0usize, |class| class as usize + 1);
    let mut class_for_candidate = initial_classes.to_vec();
    let mut members_by_class = vec![Vec::<u32>::new(); initial_class_count];
    for (candidate, &class) in class_for_candidate.iter().enumerate() {
        members_by_class[class as usize].push(candidate as u32);
    }

    // Reverse dependency graph: target candidate -> source candidates whose
    // transition signature mentions that target. Byte labels are unnecessary
    // here because a queued source block recomputes its complete exact
    // signature. Duplicate source entries are harmless and are suppressed by
    // the per-block queued bit.
    let mut reverse_predecessors = vec![Vec::<u32>::new(); num_candidates];
    for source in 0..num_candidates {
        let config = start_configs[source] as usize;
        if candidate_sets_by_config[config].is_empty() {
            continue;
        }
        let edge_start = edge_offsets[config] as usize;
        let edge_end = edge_offsets[config + 1] as usize;
        for &(_, target_config) in &edges[edge_start..edge_end] {
            for &target in &candidate_sets_by_config[target_config as usize] {
                reverse_predecessors[target as usize].push(source as u32);
            }
        }
    }
    for predecessors in &mut reverse_predecessors {
        predecessors.sort_unstable();
        predecessors.dedup();
    }

    // Keep exact signatures for established blocks. Initially every output
    // block must be classified once. After that, a target split dirties only
    // predecessor *candidates*, and only those candidates are re-signatured.
    // This avoids repeatedly scanning a large mostly-unchanged source block.
    let mut block_signature = vec![None::<SmallVec<[u32; 64]>>; members_by_class.len()];
    let mut position_in_class = vec![0usize; num_candidates];
    for members in &members_by_class {
        for (position, &candidate) in members.iter().enumerate() {
            position_in_class[candidate as usize] = position;
        }
    }
    let mut dirty_by_class = vec![Vec::<u32>::new(); members_by_class.len()];
    let mut candidate_dirty = vec![false; num_candidates];
    let mut queue = VecDeque::<usize>::new();
    let mut queued = vec![false; members_by_class.len()];
    for class in 0..members_by_class.len() {
        if members_by_class[class].len() > 1 {
            queued[class] = true;
            queue.push_back(class);
        }
    }

    let mut moved_candidates = Vec::<u32>::new();

    while let Some(class) = queue.pop_front() {
        queued[class] = false;
        if members_by_class[class].len() <= 1 {
            dirty_by_class[class].clear();
            continue;
        }
        moved_candidates.clear();

        if block_signature[class].is_none() {
            // First exact classification of this output block. New blocks are
            // born with a known exact signature and thereafter use the dirty
            // candidate path below.
            let old_members = std::mem::take(&mut members_by_class[class]);
            let mut groups = Vec::<(SmallVec<[u32; 64]>, Vec<u32>)>::new();
            let mut group_by_signature =
                FxHashMap::<SmallVec<[u32; 64]>, usize>::default();
            for candidate in old_members {
                let signature = prebuilt_candidate_signature(
                    candidate as usize,
                    &class_for_candidate,
                    start_configs,
                    &candidate_sets_by_config,
                    edge_offsets,
                    edges,
                );
                let next_group = groups.len();
                let group = *group_by_signature.entry(signature.clone()).or_insert_with(|| {
                    groups.push((signature, Vec::new()));
                    next_group
                });
                groups[group].1.push(candidate);
            }

            let retained = groups
                .iter()
                .enumerate()
                .max_by_key(|(_, (_, members))| members.len())
                .map(|(index, _)| index)
                .expect("nonempty block has a signature group");
            groups.swap(0, retained);
            let (retained_signature, retained_members) = groups.remove(0);
            block_signature[class] = Some(retained_signature);
            members_by_class[class] = retained_members;
            for (position, &candidate) in members_by_class[class].iter().enumerate() {
                class_for_candidate[candidate as usize] = class as u32;
                position_in_class[candidate as usize] = position;
            }

            for (signature, group) in groups {
                let new_class = members_by_class.len();
                for (position, &candidate) in group.iter().enumerate() {
                    class_for_candidate[candidate as usize] = new_class as u32;
                    position_in_class[candidate as usize] = position;
                    moved_candidates.push(candidate);
                }
                members_by_class.push(group);
                block_signature.push(Some(signature));
                dirty_by_class.push(Vec::new());
                queued.push(false);
            }
        } else {
            // Only candidates explicitly invalidated by a target-class change
            // can differ from this block's stored exact signature.
            let old_signature = block_signature[class]
                .as_ref()
                .expect("initialized block has a signature")
                .clone();
            let dirty = std::mem::take(&mut dirty_by_class[class]);
            let mut changed_groups =
                FxHashMap::<SmallVec<[u32; 64]>, Vec<u32>>::default();
            for candidate in dirty {
                let candidate = candidate as usize;
                if !candidate_dirty[candidate]
                    || class_for_candidate[candidate] as usize != class
                {
                    continue;
                }
                candidate_dirty[candidate] = false;
                let signature = prebuilt_candidate_signature(
                    candidate,
                    &class_for_candidate,
                    start_configs,
                    &candidate_sets_by_config,
                    edge_offsets,
                    edges,
                );
                if signature != old_signature {
                    changed_groups
                        .entry(signature)
                        .or_default()
                        .push(candidate as u32);
                }
            }

            if !changed_groups.is_empty() {
                let changed_count = changed_groups.values().map(Vec::len).sum::<usize>();
                let old_signature_members = members_by_class[class].len() - changed_count;

                // If every member changed, no candidate still represents the
                // old block signature. Retain the largest new group under the
                // existing class id; only candidates moved to other ids need
                // predecessor invalidation.
                let retained_changed_signature = if old_signature_members == 0 {
                    changed_groups
                        .iter()
                        .max_by_key(|(_, members)| members.len())
                        .map(|(signature, _)| signature.clone())
                } else {
                    None
                };

                let retained_changed_group = retained_changed_signature.is_some();
                if let Some(retained_signature) = retained_changed_signature {
                    let retained = changed_groups
                        .remove(&retained_signature)
                        .expect("retained changed group exists");
                    block_signature[class] = Some(retained_signature);
                    members_by_class[class].clear();
                    members_by_class[class].extend_from_slice(&retained);
                    for (position, &candidate) in retained.iter().enumerate() {
                        class_for_candidate[candidate as usize] = class as u32;
                        position_in_class[candidate as usize] = position;
                    }
                }

                for (signature, group) in changed_groups {
                    let new_class = members_by_class.len();
                    // Remove only changed candidates. Unaffected members stay
                    // in place and keep the old exact block signature.
                    if !retained_changed_group {
                        for &candidate in &group {
                            let candidate_index = candidate as usize;
                            let position = position_in_class[candidate_index];
                            let removed = members_by_class[class].swap_remove(position);
                            debug_assert_eq!(removed, candidate);
                            if position < members_by_class[class].len() {
                                let swapped = members_by_class[class][position] as usize;
                                position_in_class[swapped] = position;
                            }
                        }
                    }
                    for (position, &candidate) in group.iter().enumerate() {
                        class_for_candidate[candidate as usize] = new_class as u32;
                        position_in_class[candidate as usize] = position;
                        moved_candidates.push(candidate);
                    }
                    members_by_class.push(group);
                    block_signature.push(Some(signature));
                    dirty_by_class.push(Vec::new());
                    queued.push(false);
                }
            }
        }

        // A moved target changes only the signatures of source candidates that
        // mention it. Mark those candidates, not their whole blocks. Blocks
        // still awaiting their first exact classification need no dirty mark:
        // that full classification will observe the current target classes.
        for &moved in &moved_candidates {
            for &source in &reverse_predecessors[moved as usize] {
                let source = source as usize;
                let source_class = class_for_candidate[source] as usize;
                if members_by_class[source_class].len() <= 1
                    || block_signature[source_class].is_none()
                    || candidate_dirty[source]
                {
                    continue;
                }
                candidate_dirty[source] = true;
                dirty_by_class[source_class].push(source as u32);
                if !queued[source_class] {
                    queued[source_class] = true;
                    queue.push_back(source_class);
                }
            }
        }
    }

    class_for_candidate
}

/// Refine raw scanner starts through a prebuilt sparse powerset topology.
/// When `raw_active_language` is supplied, inactive-only members are erased
/// from powerset class sets and edges into projected-empty configurations are
/// treated as the common implicit missing target. This permits a topology
/// built for a larger terminal mask to be reused after the mask shrinks.
pub fn compute_state_map_from_prebuilt_sparse_powerset(
    tokenizer: &Tokenizer,
    initial_state_map: Option<&ManyToOneIdMap>,
    depth: RefinementDepth,
    raw_start_to_view: &[u32],
    configurations: &[Box<[u32]>],
    output_class_by_config: &[u32],
    raw_active_language: Option<&[bool]>,
    edge_offsets: &[u32],
    edges: &[(u8, u32)],
) -> ManyToOneIdMap {
    let num_states = tokenizer.num_states() as usize;
    assert_eq!(raw_start_to_view.len(), num_states);
    assert!(output_class_by_config.len() >= configurations.len());
    assert!(edge_offsets.len() > configurations.len());
    if let Some(active_language) = raw_active_language {
        assert_eq!(active_language.len(), num_states);
    }

    let (candidate_members, candidate_representatives, raw_to_candidate) =
        candidate_partition(num_states, initial_state_map);
    let num_candidates = candidate_representatives.len();
    let start_configs = candidate_representatives
        .iter()
        .map(|&state| {
            let config = raw_start_to_view[state];
            assert_ne!(
                config,
                u32::MAX,
                "prebuilt powerset omitted candidate raw representative {state}",
            );
            assert!(
                (config as usize) < configurations.len(),
                "prebuilt powerset mapped raw representative {state} to invalid configuration {config}",
            );
            config
        })
        .collect::<Vec<_>>();
    let mut classes = start_configs
        .iter()
        .map(|&config| output_class_by_config[config as usize])
        .collect::<Vec<_>>();

    if matches!(depth, RefinementDepth::Stable) {
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = std::time::Instant::now();
        let refined = refine_prebuilt_sparse_powerset_worklist(
            &classes,
            &start_configs,
            configurations,
            &raw_to_candidate,
            raw_active_language,
            edge_offsets,
            edges,
        );
        if profile_timing {
            let class_count = refined
                .iter()
                .copied()
                .collect::<rustc_hash::FxHashSet<_>>()
                .len();
            eprintln!(
                "[glrmask/profile][nfa_restricted_worklist] states={} candidates={} configs={} edges={} classes={} total_ms={:.3}",
                num_states,
                num_candidates,
                configurations.len(),
                edges.len(),
                class_count,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return build_state_map(
            &candidate_members,
            &candidate_representatives,
            &refined,
            num_states,
        );
    }

    let projected_empty = configurations
        .iter()
        .map(|config| {
            raw_active_language.is_some_and(|active| {
                config.iter().all(|&raw| !active[raw as usize])
            })
        })
        .collect::<Vec<_>>();
    let round_limit = match depth {
        RefinementDepth::Stable => unreachable!("stable refinement returned above"),
        RefinementDepth::Bounded(rounds) => rounds,
    };
    let mut class_set_by_config = vec![0u32; configurations.len()];
    for _ in 0..round_limit {
        // Preserve singleton class-set IDs directly. Nearly every powerset
        // configuration in large epsilon lexers is a singleton closure, so
        // allocating, sorting, and hashing a SmallVec for each one is pure
        // overhead. Composite configurations can themselves collapse to a
        // singleton class set after refinement; map those to the same direct ID
        // so equality remains exactly identical to generic set interning.
        let singleton_id_limit = classes.iter().copied().max().map_or(0, |class| class + 1);
        let mut composite_class_set_ids =
            FxHashMap::<SmallVec<[u32; 4]>, u32>::default();
        for (config_id, config) in configurations.iter().enumerate() {
            if let [raw] = config.as_ref() {
                class_set_by_config[config_id] = classes[raw_to_candidate[*raw as usize]];
                continue;
            }

            let mut class_set = SmallVec::<[u32; 4]>::new();
            if let Some(active_language) = raw_active_language {
                class_set.extend(
                    config
                        .iter()
                        .filter(|&&raw| active_language[raw as usize])
                        .map(|&raw| classes[raw_to_candidate[raw as usize]]),
                );
            } else {
                class_set.extend(
                    config
                        .iter()
                        .map(|&raw| classes[raw_to_candidate[raw as usize]]),
                );
            }
            class_set.sort_unstable();
            class_set.dedup();
            class_set_by_config[config_id] = if let [class] = class_set.as_slice() {
                *class
            } else {
                let next = singleton_id_limit + composite_class_set_ids.len() as u32;
                *composite_class_set_ids.entry(class_set).or_insert(next)
            };
        }
        let mut zero_edge_signatures = FxHashMap::<u32, u32>::default();
        let mut one_edge_signatures = FxHashMap::<(u32, u8, u32), u32>::default();
        let mut larger_signatures = FxHashMap::<SmallVec<[u32; 8]>, u32>::default();
        let mut next_class = 0u32;
        let mut next_classes = vec![0u32; num_candidates];
        for candidate in 0..num_candidates {
            let config = start_configs[candidate] as usize;
            let edge_start = edge_offsets[config] as usize;
            let edge_end = edge_offsets[config + 1] as usize;
            let candidate_class = classes[candidate];
            next_classes[candidate] = if raw_active_language.is_none() {
                match &edges[edge_start..edge_end] {
                    [] => *zero_edge_signatures.entry(candidate_class).or_insert_with(|| {
                        let class = next_class;
                        next_class += 1;
                        class
                    }),
                    [(byte, target)] => {
                        let key = (
                            candidate_class,
                            *byte,
                            class_set_by_config[*target as usize],
                        );
                        *one_edge_signatures.entry(key).or_insert_with(|| {
                            let class = next_class;
                            next_class += 1;
                            class
                        })
                    }
                    direct_edges => {
                        let mut signature = SmallVec::<[u32; 8]>::new();
                        signature.push(candidate_class);
                        for &(byte, target) in direct_edges {
                            signature.push(byte as u32 + 1);
                            signature.push(class_set_by_config[target as usize] + 1);
                        }
                        *larger_signatures.entry(signature).or_insert_with(|| {
                            let class = next_class;
                            next_class += 1;
                            class
                        })
                    }
                }
            } else {
                let mut projected_edges = SmallVec::<[(u8, u32); 8]>::new();
                if !projected_empty[config] {
                    projected_edges.extend(
                        edges[edge_start..edge_end]
                            .iter()
                            .copied()
                            .filter(|&(_, target)| !projected_empty[target as usize]),
                    );
                }
                match projected_edges.as_slice() {
                    [] => *zero_edge_signatures.entry(candidate_class).or_insert_with(|| {
                        let class = next_class;
                        next_class += 1;
                        class
                    }),
                    [(byte, target)] => {
                        let key = (
                            candidate_class,
                            *byte,
                            class_set_by_config[*target as usize],
                        );
                        *one_edge_signatures.entry(key).or_insert_with(|| {
                            let class = next_class;
                            next_class += 1;
                            class
                        })
                    }
                    projected_edges => {
                        let mut signature = SmallVec::<[u32; 8]>::new();
                        signature.push(candidate_class);
                        for &(byte, target) in projected_edges {
                            signature.push(byte as u32 + 1);
                            signature.push(class_set_by_config[target as usize] + 1);
                        }
                        *larger_signatures.entry(signature).or_insert_with(|| {
                            let class = next_class;
                            next_class += 1;
                            class
                        })
                    }
                }
            };
        }
        let stable = same_partition(&classes, &next_classes);
        classes = next_classes;
        if stable {
            break;
        }
    }

    build_state_map(
        &candidate_members,
        &candidate_representatives,
        &classes,
        num_states,
    )
}

pub fn compute_state_map(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    active_groups: Option<&[bool]>,
    initial_state_map: Option<&ManyToOneIdMap>,
    depth: RefinementDepth,
) -> ManyToOneIdMap {
    let num_states = tokenizer.num_states() as usize;
    if num_states == 0 {
        return ManyToOneIdMap::from_original_to_internal_allowing_unmapped(Vec::new(), 0);
    }
    let active_bytes = relevant_bytes
        .iter()
        .enumerate()
        .filter_map(|(byte, &active)| active.then_some(byte as u8))
        .collect::<Vec<_>>();
    let (candidate_members, candidate_representatives, raw_to_candidate) =
        candidate_partition(num_states, initial_state_map);
    let num_candidates = candidate_representatives.len();
    let singleton_closures = tokenizer.all_singleton_epsilon_closures();
    let raw_active_language = raw_active_language_states(tokenizer, active_groups);

    let mut config_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut configs = Vec::<Box<[u32]>>::new();
    let start_configs = candidate_representatives
        .iter()
        .map(|&state| {
            let config = project_raw_config(
                singleton_closures[state].to_vec(),
                raw_active_language.as_deref(),
            );
            intern_config(config, &mut config_ids, &mut configs)
        })
        .collect::<Vec<_>>();

    let observations = start_configs
        .iter()
        .map(|&config| observation_words(tokenizer, &configs[config as usize], active_groups))
        .collect::<Vec<_>>();
    let mut initial_keys = FxHashMap::<Vec<u64>, u32>::default();
    let mut classes = vec![0u32; num_candidates];
    for candidate in 0..num_candidates {
        let key = observations[candidate].clone();
        let next = initial_keys.len() as u32;
        classes[candidate] = *initial_keys.entry(key).or_insert(next);
    }

    let mut target_configs = vec![u32::MAX; num_candidates * active_bytes.len()];
    let mut target_marks = vec![0u32; num_states];
    let mut target_generation = 0u32;
    let mut target_config = Vec::<u32>::new();
    for candidate in 0..num_candidates {
        let source = configs[start_configs[candidate] as usize].to_vec();
        for (slot, &byte) in active_bytes.iter().enumerate() {
            target_generation = target_generation.wrapping_add(1);
            if target_generation == 0 {
                target_marks.fill(0);
                target_generation = 1;
            }
            target_config.clear();
            // Every source config is already epsilon-closed.  Consume the byte
            // from each member and union cached singleton closures of the raw
            // targets.  This is exactly `step_all` without re-closing the
            // source, allocating a per-call seen vector, or sorting an
            // intermediate direct-target list.
            for &source_state in &source {
                let raw_target = tokenizer.get_transition(source_state, byte);
                if raw_target == u32::MAX {
                    continue;
                }
                for &reachable in singleton_closures[raw_target as usize].iter() {
                    if raw_active_language
                        .as_deref()
                        .is_some_and(|active| !active[reachable as usize])
                    {
                        continue;
                    }
                    let mark = &mut target_marks[reachable as usize];
                    if *mark != target_generation {
                        *mark = target_generation;
                        target_config.push(reachable);
                    }
                }
            }
            if !target_config.is_empty() {
                target_config.sort_unstable();
                target_configs[candidate * active_bytes.len() + slot] =
                    intern_config(target_config.clone(), &mut config_ids, &mut configs);
            }
        }
    }

    if matches!(depth, RefinementDepth::Stable) {
        // The direct NFA analysis has already materialized the exact one-byte
        // epsilon-closed successor configuration for every candidate and
        // relevant byte.  Reuse the same predecessor-driven stable refinement
        // as the sparse powerset path instead of rescanning every candidate in
        // synchronous Moore rounds.  Long deterministic chains can require one
        // logical refinement propagation per byte of distinguishing depth; the
        // worklist follows only affected predecessors rather than performing a
        // whole-graph pass for each propagation step.
        //
        // Multiple candidates can share the same epsilon-closed start
        // configuration. Their one-byte successor configurations are then
        // identical, so one candidate row is sufficient to materialize that
        // configuration's sparse outgoing edges.
        let mut source_candidate_by_config = vec![usize::MAX; configs.len()];
        for (candidate, &config) in start_configs.iter().enumerate() {
            let slot = &mut source_candidate_by_config[config as usize];
            if *slot == usize::MAX {
                *slot = candidate;
            } else {
                debug_assert!((0..active_bytes.len()).all(|byte_slot| {
                    target_configs[*slot * active_bytes.len() + byte_slot]
                        == target_configs[candidate * active_bytes.len() + byte_slot]
                }));
            }
        }

        let mut edge_offsets = Vec::<u32>::with_capacity(configs.len() + 1);
        let mut edges = Vec::<(u8, u32)>::new();
        edge_offsets.push(0);
        for source_candidate in source_candidate_by_config {
            if source_candidate != usize::MAX {
                let row_start = source_candidate * active_bytes.len();
                for (byte_slot, &byte) in active_bytes.iter().enumerate() {
                    let target = target_configs[row_start + byte_slot];
                    if target != u32::MAX {
                        edges.push((byte, target));
                    }
                }
            }
            edge_offsets.push(edges.len() as u32);
        }

        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = std::time::Instant::now();
        let refined = refine_prebuilt_sparse_powerset_worklist(
            &classes,
            &start_configs,
            &configs,
            &raw_to_candidate,
            None,
            &edge_offsets,
            &edges,
        );
        if profile_timing {
            let class_count = refined
                .iter()
                .copied()
                .collect::<rustc_hash::FxHashSet<_>>()
                .len();
            eprintln!(
                "[glrmask/profile][nfa_restricted_direct_worklist] states={} candidates={} configs={} edges={} classes={} total_ms={:.3}",
                num_states,
                num_candidates,
                configs.len(),
                edges.len(),
                class_count,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return build_state_map(
            &candidate_members,
            &candidate_representatives,
            &refined,
            num_states,
        );
    }

    let round_limit = match depth {
        RefinementDepth::Stable => unreachable!("stable refinement returned above"),
        RefinementDepth::Bounded(rounds) => rounds,
    };
    for _ in 0..round_limit {
        let mut signatures = FxHashMap::<Vec<u32>, u32>::default();
        let mut next_classes = vec![0u32; num_candidates];
        for candidate in 0..num_candidates {
            let mut signature = Vec::<u32>::with_capacity(1 + active_bytes.len() * 2);
            signature.push(classes[candidate]);
            for slot in 0..active_bytes.len() {
                let config = target_configs[candidate * active_bytes.len() + slot];
                if config == u32::MAX {
                    signature.push(0);
                    continue;
                }
                let mut target_classes = configs[config as usize]
                    .iter()
                    .map(|&target| classes[raw_to_candidate[target as usize]] + 1)
                    .collect::<Vec<_>>();
                target_classes.sort_unstable();
                target_classes.dedup();
                signature.push(target_classes.len() as u32 + 1);
                signature.extend(target_classes);
            }
            let next = signatures.len() as u32;
            next_classes[candidate] = *signatures.entry(signature).or_insert(next);
        }
        let stable = same_partition(&classes, &next_classes);
        classes = next_classes;
        if stable {
            break;
        }
    }

    build_state_map(
        &candidate_members,
        &candidate_representatives,
        &classes,
        num_states,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_map_compacts_sparse_worklist_class_labels() {
        let map = build_state_map(
            &[vec![0], vec![1], vec![2]],
            &[0, 1, 2],
            &[0, 2, 2],
            3,
        );

        assert_eq!(map.original_to_internal, vec![0, 1, 1]);
        assert_eq!(map.internal_to_originals, vec![vec![0], vec![1, 2]]);
        assert_eq!(map.representative_original_ids, vec![0, 1]);
    }

    #[test]
    fn prebuilt_superset_token_trie_matches_subset_analysis() {
        let tokenizer =
            crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
        let raw_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let flat_trans = crate::compiler::stages::id_map_and_terminal_dwa::l1::build_flat_transition_table(&tokenizer);
        let parent_tokens: Vec<&[u8]> = vec![b"", b"a", b"aa", b"ab", b"b", b"ba", b"x"];
        let subset_tokens: Vec<&[u8]> = vec![b"", b"a", b"aa", b"ab", b"b", b"ba"];
        let active = [true, true];
        let parent_trie = build_token_bounded_analysis_trie_sorted(&parent_tokens);
        let direct = build_token_bounded_analysis_view_projected_sorted_with_raw_transitions(
            &tokenizer,
            &flat_trans,
            &raw_states,
            &subset_tokens,
            &active,
        );
        let reused =
            build_token_bounded_analysis_view_projected_sorted_with_raw_transitions_and_trie(
                &tokenizer,
                &flat_trans,
                &raw_states,
                &subset_tokens,
                &active,
                Some(&parent_trie),
            );
        for &raw_state in &raw_states {
            let direct_start = direct.view_state_for_raw_start(raw_state);
            let reused_start = reused.view_state_for_raw_start(raw_state);
            for &token in &subset_tokens {
                assert_eq!(
                    view_trace(&direct.tokenizer_view, direct_start, token),
                    view_trace(&reused.tokenizer_view, reused_start, token),
                    "raw_state={raw_state} token={token:?}",
                );
            }
        }
    }

    #[test]
    fn prebuilt_superset_token_trie_matches_bounded_reset_suffix_analysis() {
        let tokenizer =
            crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
        let raw_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let parent_tokens: Vec<&[u8]> =
            vec![b"a", b"aa", b"ab", b"aba", b"b", b"ba", b"x"];
        let subset_tokens: Vec<&[u8]> = vec![b"aa", b"aba", b"ba"];
        let parent_trie = build_token_bounded_analysis_trie_sorted(&parent_tokens);
        let direct = build_bounded_analysis_view(
            &tokenizer,
            &raw_states,
            &subset_tokens,
            Some(&[true, true]),
        );
        let reused = build_bounded_analysis_view_with_trie(
            &tokenizer,
            &raw_states,
            &subset_tokens,
            Some(&[true, true]),
            Some(&parent_trie),
        );
        let observed = subset_tokens
            .iter()
            .flat_map(|token| (0..token.len()).map(move |offset| &token[offset..]))
            .collect::<Vec<_>>();
        for &raw_state in &raw_states {
            let direct_start = direct.view_state_for_raw_start(raw_state);
            let reused_start = reused.view_state_for_raw_start(raw_state);
            for &bytes in &observed {
                assert_eq!(
                    view_trace(&direct.tokenizer_view, direct_start, bytes),
                    view_trace(&reused.tokenizer_view, reused_start, bytes),
                    "raw_state={raw_state} bytes={bytes:?}",
                );
            }
        }
    }

    #[test]
    fn sorted_byte_trie_matches_generic_builder() {
        let sequences: Vec<&[u8]> = vec![
            b"",
            b"a",
            b"a",
            b"aa",
            b"ab",
            b"aba",
            b"b",
            b"ba",
            b"z",
            b"\xff",
        ];
        assert_eq!(
            build_byte_trie(sequences.iter().copied()),
            build_byte_trie_sorted(&sequences),
        );
    }
    use crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer;

    #[test]
    fn cached_closed_config_step_matches_scalar_step_all() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let singleton_closures = tokenizer.all_singleton_epsilon_closures();
        let mut raw_transitions = vec![u32::MAX; tokenizer.num_states() as usize * 256];
        for state in 0..tokenizer.num_states() {
            for (byte, target) in tokenizer.transitions_from(state) {
                raw_transitions[state as usize * 256 + byte as usize] = target;
            }
        }
        let active_group_masks: [Option<&[bool]>; 4] = [
            None,
            Some(&[true, false]),
            Some(&[false, true]),
            Some(&[true, true]),
        ];

        for active_groups in active_group_masks {
            let active_language = raw_active_language_states(&tokenizer, active_groups);
            let mut target_marks = vec![0u32; tokenizer.num_states() as usize];
            let mut target_generation = 0u32;
            for raw_state in 0..tokenizer.num_states() as usize {
                let config = singleton_closures[raw_state]
                    .iter()
                    .copied()
                    .filter(|&state| {
                        active_language
                            .as_deref()
                            .is_none_or(|active| active[state as usize])
                    })
                    .collect::<Vec<_>>();
                for byte in 0u8..=u8::MAX {
                    let scalar = tokenizer
                        .step_all(&config, byte)
                        .iter()
                        .copied()
                        .filter(|&state| {
                            active_language
                                .as_deref()
                                .is_none_or(|active| active[state as usize])
                        })
                        .collect::<Vec<_>>();
                    let cached = step_epsilon_closed_config_cached(
                        &tokenizer,
                        None,
                        &config,
                        byte,
                        singleton_closures.as_ref(),
                        active_language.as_deref(),
                        &mut target_marks,
                        &mut target_generation,
                    );
                    assert_eq!(cached, scalar, "raw_state={raw_state} byte={byte}");
                    let cached_from_raw = step_epsilon_closed_config_cached(
                        &tokenizer,
                        Some(&raw_transitions),
                        &config,
                        byte,
                        singleton_closures.as_ref(),
                        active_language.as_deref(),
                        &mut target_marks,
                        &mut target_generation,
                    );
                    assert_eq!(
                        cached_from_raw, scalar,
                        "raw transition table raw_state={raw_state} byte={byte}",
                    );
                }
            }
        }
    }

    #[test]
    fn projected_sorted_raw_transition_view_matches_tokenizer_step_view() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let active_groups = [true, false];
        let tokens: [&[u8]; 7] = [b"a", b"aa", b"ab", b"aba", b"b", b"ba", b"xyz"];
        let mut raw_transitions = vec![u32::MAX; tokenizer.num_states() as usize * 256];
        for state in 0..tokenizer.num_states() {
            for (byte, target) in tokenizer.transitions_from(state) {
                raw_transitions[state as usize * 256 + byte as usize] = target;
            }
        }

        let scalar = build_token_bounded_analysis_view_projected_sorted(
            &tokenizer,
            &raw_start_states,
            &tokens,
            &active_groups,
        );
        let dense = build_token_bounded_analysis_view_projected_sorted_with_raw_transitions(
            &tokenizer,
            &raw_transitions,
            &raw_start_states,
            &tokens,
            &active_groups,
        );

        assert_eq!(
            dense.tokenizer_view.dfa().states.len(),
            scalar.tokenizer_view.dfa().states.len(),
        );
        for &raw_state in &raw_start_states {
            let scalar_start = scalar.view_state_for_raw_start(raw_state);
            let dense_start = dense.view_state_for_raw_start(raw_state);
            for &token in &tokens {
                assert_eq!(
                    view_trace(&dense.tokenizer_view, dense_start, token),
                    view_trace(&scalar.tokenizer_view, scalar_start, token),
                    "raw_state={raw_state} token={token:?}",
                );
            }
        }
    }

    fn view_trace(
        view: &TokenizerView,
        start_state: usize,
        token: &[u8],
    ) -> Vec<(Vec<usize>, Vec<usize>, bool)> {
        let dfa = view.dfa();
        let mut state = start_state;
        let mut trace = vec![(
            dfa.states[state].finalizers.clone(),
            dfa.states[state].possible_future_group_ids.clone(),
            false,
        )];
        for &byte in token {
            let target = dfa.trans(state, byte as usize);
            if target == u32::MAX {
                trace.push((Vec::new(), Vec::new(), true));
                break;
            }
            state = target as usize;
            trace.push((
                dfa.states[state].finalizers.clone(),
                dfa.states[state].possible_future_group_ids.clone(),
                false,
            ));
        }
        trace
    }

    #[test]
    fn trie_frontier_expansion_matches_pair_hash_dfs() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let active_groups = [true, false];
        let token_sets: [&[&[u8]]; 2] = [
            &[b"a", b"aa", b"ab", b"aba"],
            &[b"a", b"ab", b"b", b"ba", b"x", b"xyz"],
        ];

        for tokens in token_sets {
            let (frontier, frontier_work) = build_bounded_analysis_topology_impl_with_expansion(
                &tokenizer,
                None,
                &raw_start_states,
                tokens,
                false,
                true,
                false,
                true,
                Some(&active_groups),
                None,
                None,
                true,
            )
            .expect("frontier topology build");
            let (dfs, dfs_work) = build_bounded_analysis_topology_impl_with_expansion(
                &tokenizer,
                None,
                &raw_start_states,
                tokens,
                false,
                true,
                false,
                true,
                Some(&active_groups),
                None,
                None,
                false,
            )
            .expect("DFS topology build");
            assert_eq!(frontier_work, dfs_work);

            let frontier = frontier.materialize_already_projected(&tokenizer, &active_groups);
            let dfs = dfs.materialize_already_projected(&tokenizer, &active_groups);
            for &raw_state in &raw_start_states {
                let frontier_start = frontier.view_state_for_raw_start(raw_state);
                let dfs_start = dfs.view_state_for_raw_start(raw_state);
                for &token in tokens {
                    assert_eq!(
                        view_trace(&frontier.tokenizer_view, frontier_start, token),
                        view_trace(&dfs.tokenizer_view, dfs_start, token),
                        "raw_state={raw_state} token={token:?}",
                    );
                }
            }
        }

        // A sparse-root edge can reach a raw state that was not itself one of
        // the preseeded start states. The sparse path must fall back to exact
        // configuration construction rather than treating the missing
        // raw_start_to_view entry as DEAD.
        let subset_start_states = vec![2usize, 4usize];
        let subset_tokens: &[&[u8]] = &[b"a", b"b"];
        let subset_active_groups = [true, true];
        let (frontier, frontier_work) = build_bounded_analysis_topology_impl_with_expansion(
            &tokenizer,
            None,
            &subset_start_states,
            subset_tokens,
            false,
            true,
            false,
            true,
            Some(&subset_active_groups),
            None,
            None,
            true,
        )
        .expect("frontier topology build with unpreseeded targets");
        let (dfs, dfs_work) = build_bounded_analysis_topology_impl_with_expansion(
            &tokenizer,
            None,
            &subset_start_states,
            subset_tokens,
            false,
            true,
            false,
            true,
            Some(&subset_active_groups),
            None,
            None,
            false,
        )
        .expect("DFS topology build with unpreseeded targets");
        assert_eq!(frontier_work, dfs_work);
        let frontier = frontier.materialize_already_projected(&tokenizer, &subset_active_groups);
        let dfs = dfs.materialize_already_projected(&tokenizer, &subset_active_groups);
        for &raw_state in &subset_start_states {
            let frontier_start = frontier.view_state_for_raw_start(raw_state);
            let dfs_start = dfs.view_state_for_raw_start(raw_state);
            for &token in subset_tokens {
                assert_eq!(
                    view_trace(&frontier.tokenizer_view, frontier_start, token),
                    view_trace(&dfs.tokenizer_view, dfs_start, token),
                    "unpreseeded target raw_state={raw_state} token={token:?}",
                );
            }
        }

        // Also cover the generic unsorted-token path, combined starts, and the
        // reset-suffix trie used by non-L1 callers.
        let unsorted_tokens: &[&[u8]] = &[b"xyz", b"a", b"ba", b"ab", b"x"];
        let (frontier, frontier_work) = build_bounded_analysis_topology_impl_with_expansion(
            &tokenizer,
            None,
            &raw_start_states,
            unsorted_tokens,
            true,
            false,
            true,
            false,
            None,
            None,
            None,
            true,
        )
        .expect("frontier topology build with reset suffixes");
        let (dfs, dfs_work) = build_bounded_analysis_topology_impl_with_expansion(
            &tokenizer,
            None,
            &raw_start_states,
            unsorted_tokens,
            true,
            false,
            true,
            false,
            None,
            None,
            None,
            false,
        )
        .expect("DFS topology build with reset suffixes");
        assert_eq!(frontier_work, dfs_work);
        let frontier = frontier.materialize(&tokenizer, None);
        let dfs = dfs.materialize(&tokenizer, None);
        for &raw_state in &raw_start_states {
            let frontier_start = frontier.view_state_for_raw_start(raw_state);
            let dfs_start = dfs.view_state_for_raw_start(raw_state);
            for &token in unsorted_tokens {
                assert_eq!(
                    view_trace(&frontier.tokenizer_view, frontier_start, token),
                    view_trace(&dfs.tokenizer_view, dfs_start, token),
                    "combined raw_state={raw_state} token={token:?}",
                );
            }
        }
    }

    #[test]
    fn active_filtered_powerset_drops_inactive_members_before_refinement() {
        use crate::automata::lexer::ast::Expr;
        use crate::automata::lexer::compile::build_regex_partitioned_with_adaptive;

        let expressions = vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"xq".to_vec()),
            Expr::U8Seq(b"yrr".to_vec()),
        ];
        let tokenizer = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1, 2],
            false,
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let view = build_relevant_powerset_view(
            &tokenizer,
            &[true; 256],
            Some(&[true, false, false]),
            None,
        );
        let start_config = &view.configurations[view.start_state];
        assert!(!start_config.is_empty());
        assert!(start_config.iter().all(|&state| {
            tokenizer.matched_terminals_iter(state).any(|terminal| terminal == 0)
                || tokenizer
                    .possible_future_terminals_iter(state)
                    .any(|terminal| terminal == 0)
        }));
        let edge_start = view.edge_offsets[view.start_state] as usize;
        let edge_end = view.edge_offsets[view.start_state + 1] as usize;
        let start_edges = &view.edges[edge_start..edge_end];
        assert!(!start_edges.iter().any(|&(byte, _)| byte == b'x' || byte == b'y'));
    }

    #[test]
    fn token_bounded_topology_materialization_drops_inactive_members() {
        use crate::automata::lexer::ast::Expr;
        use crate::automata::lexer::compile::build_regex_partitioned_with_adaptive;

        let expressions = vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"xq".to_vec()),
            Expr::U8Seq(b"yrr".to_vec()),
        ];
        let tokenizer = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1, 2],
            false,
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens = [
            b"a".as_slice(),
            b"ab".as_slice(),
            b"x".as_slice(),
            b"xq".as_slice(),
            b"y".as_slice(),
            b"yrr".as_slice(),
        ];
        let topology =
            build_token_bounded_analysis_topology(&tokenizer, &raw_start_states, &tokens);
        let projected = topology.materialize(&tokenizer, Some(&[true, false, false]));
        let dfa = projected.tokenizer_view.dfa();

        assert_ne!(dfa.trans(dfa.start_state, b'a' as usize), u32::MAX);
        assert_eq!(dfa.trans(dfa.start_state, b'x' as usize), u32::MAX);
        assert_eq!(dfa.trans(dfa.start_state, b'y' as usize), u32::MAX);
    }

    #[test]
    fn already_projected_token_topology_materialization_matches_reprojection() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens = [
            b"a".as_slice(),
            b"aa".as_slice(),
            b"ab".as_slice(),
            b"ba".as_slice(),
            b"bb".as_slice(),
        ];
        let active_groups = [true, false];
        let topology = build_bounded_analysis_topology_impl(
            &tokenizer,
            None,
            &raw_start_states,
            &tokens,
            false,
            true,
            false,
            false,
            Some(&active_groups),
            None,
            None,
        )
        .expect("projected topology build")
        .0;
        let direct = topology.materialize_already_projected(&tokenizer, &active_groups);
        let reprojected = topology.materialize(&tokenizer, Some(&active_groups));

        for &raw_state in &raw_start_states {
            let direct_start = direct.view_state_for_raw_start(raw_state);
            let reprojected_start = reprojected.view_state_for_raw_start(raw_state);
            for token in tokens {
                assert_eq!(
                    view_trace(&direct.tokenizer_view, direct_start, token),
                    view_trace(&reprojected.tokenizer_view, reprojected_start, token),
                );
            }
        }
    }

    #[test]
    fn token_bounded_projected_build_aborts_at_actual_work_budget() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens = [
            b"a".as_slice(),
            b"aa".as_slice(),
            b"ab".as_slice(),
            b"ba".as_slice(),
            b"bb".as_slice(),
        ];
        let active_groups = [true, true];
        let tiny_budget = TokenBoundedAnalysisWorkBudget {
            max_configurations: usize::MAX,
            max_trie_visits: 1,
        };

        let aborted = try_build_token_bounded_analysis_view_projected(
            &tokenizer,
            &raw_start_states,
            &tokens,
            &active_groups,
            tiny_budget,
        );
        let work = match aborted {
            Ok(_) => panic!("tiny actual-work budget unexpectedly completed"),
            Err(work) => work,
        };
        assert_eq!(work.trie_visits, tiny_budget.max_trie_visits + 1);

        let generous_budget = TokenBoundedAnalysisWorkBudget {
            max_configurations: 1_000,
            max_trie_visits: 10_000,
        };
        let (budgeted, _) = try_build_token_bounded_analysis_view_projected(
            &tokenizer,
            &raw_start_states,
            &tokens,
            &active_groups,
            generous_budget,
        )
        .unwrap_or_else(|work| panic!("generous budget aborted unexpectedly: {work:?}"));
        let reference = build_token_bounded_analysis_view_projected(
            &tokenizer,
            &raw_start_states,
            &tokens,
            &active_groups,
        );

        for &raw_state in &raw_start_states {
            let budgeted_start = budgeted.view_state_for_raw_start(raw_state);
            let reference_start = reference.view_state_for_raw_start(raw_state);
            for token in tokens {
                assert_eq!(
                    view_trace(&budgeted.tokenizer_view, budgeted_start, token),
                    view_trace(&reference.tokenizer_view, reference_start, token),
                );
            }
        }
    }

    #[test]
    fn bounded_nfa_common_first_factorization_preserves_observed_token_trajectories() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens = [b"aa".as_slice(), b"ab".as_slice(), b"aab".as_slice()];
        let factored = build_bounded_analysis_view_impl(
            &tokenizer,
            &raw_start_states,
            &tokens,
            None,
            false,
            true,
            true,
        );
        let reference = build_bounded_analysis_view_impl(
            &tokenizer,
            &raw_start_states,
            &tokens,
            None,
            false,
            false,
            true,
        );

        for &raw_state in &raw_start_states {
            let factored_start = factored.view_state_for_raw_start(raw_state);
            let reference_start = reference.view_state_for_raw_start(raw_state);
            for token in tokens {
                assert_eq!(
                    view_trace(&factored.tokenizer_view, factored_start, token),
                    view_trace(&reference.tokenizer_view, reference_start, token),
                );
            }
        }

        for token in tokens {
            for offset in 0..token.len() {
                assert_eq!(
                    view_trace(
                        &factored.tokenizer_view,
                        factored.tokenizer_view.dfa().start_state,
                        &token[offset..],
                    ),
                    view_trace(
                        &reference.tokenizer_view,
                        reference.tokenizer_view.dfa().start_state,
                        &token[offset..],
                    ),
                );
            }
        }
    }

    #[test]
    fn relevant_powerset_view_preserves_bounded_token_trajectories() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens = [
            b"".as_slice(),
            b"a".as_slice(),
            b"b".as_slice(),
            b"aa".as_slice(),
            b"ab".as_slice(),
            b"ba".as_slice(),
            b"aab".as_slice(),
        ];
        let active_groups = [true, true];
        let bounded = build_bounded_analysis_view(
            &tokenizer,
            &raw_start_states,
            &tokens,
            Some(&active_groups),
        );
        let mut relevant_bytes = [false; 256];
        for token in tokens {
            for &byte in token {
                relevant_bytes[byte as usize] = true;
            }
        }
        let powerset = build_relevant_powerset_view(
            &tokenizer,
            &relevant_bytes,
            Some(&active_groups),
            None,
        );
        let raw_start_to_powerset = Arc::clone(&powerset.raw_start_to_view);
        let powerset = powerset.into_tokenizer_view();

        for &raw_state in &raw_start_states {
            let bounded_start = bounded.view_state_for_raw_start(raw_state);
            let powerset_start = raw_start_to_powerset[raw_state] as usize;
            for token in tokens {
                assert_eq!(
                    view_trace(&bounded.tokenizer_view, bounded_start, token),
                    view_trace(&powerset, powerset_start, token),
                    "raw_state={raw_state} token={token:?}",
                );
            }
        }
    }

    #[test]
    fn raw_relevant_powerset_projected_targets_match_scalar_exactly() {
        fn assert_same(left: &RelevantPowersetView, right: &RelevantPowersetView) {
            assert_eq!(left.start_state, right.start_state);
            assert_eq!(left.bytes, right.bytes);
            assert_eq!(left.edge_offsets, right.edge_offsets);
            assert_eq!(left.edges, right.edges);
            assert_eq!(left.raw_start_to_view, right.raw_start_to_view);
            assert_eq!(left.configurations, right.configurations);
            assert_eq!(left.states.len(), right.states.len());
            for (left, right) in left.states.iter().zip(&right.states) {
                assert_eq!(left.finalizers, right.finalizers);
                assert_eq!(
                    left.possible_future_group_ids,
                    right.possible_future_group_ids,
                );
            }
        }

        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let mut relevant_bytes = [false; 256];
        relevant_bytes[b'a' as usize] = true;
        relevant_bytes[b'b' as usize] = true;

        for active_groups in [None, Some([true, true]), Some([true, false]), Some([false, true])] {
            let active_groups = active_groups.as_ref().map(<[_; 2]>::as_slice);
            let scalar = try_build_relevant_powerset_view(
                &tokenizer,
                &relevant_bytes,
                active_groups,
                None,
                None,
                RawPowersetTargetMode::Scalar,
            )
            .expect("unbounded scalar powerset construction");
            let direct = try_build_relevant_powerset_view(
                &tokenizer,
                &relevant_bytes,
                active_groups,
                None,
                None,
                RawPowersetTargetMode::Direct,
            )
            .expect("unbounded direct powerset construction");
            assert_same(&scalar, &direct);
        }
    }

    #[test]
    fn powerset_restricted_bounded_view_matches_direct_bounded_view() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens = [
            b"".as_slice(),
            b"a".as_slice(),
            b"b".as_slice(),
            b"aa".as_slice(),
            b"ab".as_slice(),
            b"ba".as_slice(),
            b"aab".as_slice(),
        ];
        let active_groups = [true, true];
        let direct = build_bounded_analysis_view(
            &tokenizer,
            &raw_start_states,
            &tokens,
            Some(&active_groups),
        );
        let mut relevant_bytes = [false; 256];
        for token in tokens {
            for &byte in token {
                relevant_bytes[byte as usize] = true;
            }
        }
        let powerset = build_relevant_powerset_view(
            &tokenizer,
            &relevant_bytes,
            Some(&active_groups),
            None,
        );
        let restricted = build_bounded_analysis_view_from_relevant_powerset(
            &powerset,
            &raw_start_states,
            &tokens,
        );

        for &raw_state in &raw_start_states {
            let direct_start = direct.view_state_for_raw_start(raw_state);
            let restricted_start = restricted.view_state_for_raw_start(raw_state);
            for token in tokens {
                assert_eq!(
                    view_trace(&restricted.tokenizer_view, restricted_start, token),
                    view_trace(&direct.tokenizer_view, direct_start, token),
                    "raw_state={raw_state} token={token:?}",
                );
            }
        }
        for token in tokens {
            for offset in 0..token.len() {
                assert_eq!(
                    view_trace(
                        &restricted.tokenizer_view,
                        restricted.tokenizer_view.dfa().start_state,
                        &token[offset..],
                    ),
                    view_trace(
                        &direct.tokenizer_view,
                        direct.tokenizer_view.dfa().start_state,
                        &token[offset..],
                    ),
                    "reset suffix={:?}",
                    &token[offset..],
                );
            }
        }
    }

    #[test]
    fn powerset_restricted_bounded_view_matches_direct_across_projections() {
        fn binary_inputs(max_len: usize) -> Vec<Vec<u8>> {
            let mut inputs = vec![Vec::new()];
            for len in 1..=max_len {
                let start = inputs.len();
                for bits in 0..(1usize << len) {
                    inputs.push(
                        (0..len)
                            .rev()
                            .map(|shift| if bits & (1 << shift) == 0 { b'a' } else { b'b' })
                            .collect(),
                    );
                }
                debug_assert_eq!(inputs.len() - start, 1usize << len);
            }
            inputs
        }

        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let exhaustive = binary_inputs(4);
        let sparse_cases = vec![
            vec![Vec::new(), b"ab".to_vec(), b"baa".to_vec(), b"bbbb".to_vec()],
            vec![b"a".to_vec(), b"bb".to_vec(), b"aaba".to_vec()],
        ];
        let mut token_cases = vec![exhaustive];
        token_cases.extend(sparse_cases);

        for active_groups in [[true, true], [true, false], [false, true]] {
            for owned_tokens in &token_cases {
                let tokens = owned_tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let direct = build_bounded_analysis_view(
                    &tokenizer,
                    &raw_start_states,
                    &tokens,
                    Some(&active_groups),
                );
                let mut relevant_bytes = [false; 256];
                for token in &tokens {
                    for &byte in *token {
                        relevant_bytes[byte as usize] = true;
                    }
                }
                let powerset = build_relevant_powerset_view(
                    &tokenizer,
                    &relevant_bytes,
                    Some(&active_groups),
                    None,
                );
                let restricted = build_bounded_analysis_view_from_relevant_powerset(
                    &powerset,
                    &raw_start_states,
                    &tokens,
                );

                for &raw_state in &raw_start_states {
                    let direct_start = direct.view_state_for_raw_start(raw_state);
                    let restricted_start = restricted.view_state_for_raw_start(raw_state);
                    for token in &tokens {
                        assert_eq!(
                            view_trace(&restricted.tokenizer_view, restricted_start, token),
                            view_trace(&direct.tokenizer_view, direct_start, token),
                            "active_groups={active_groups:?} raw_state={raw_state} token={token:?}",
                        );
                    }
                }
                for token in &tokens {
                    for offset in 0..token.len() {
                        assert_eq!(
                            view_trace(
                                &restricted.tokenizer_view,
                                restricted.tokenizer_view.dfa().start_state,
                                &token[offset..],
                            ),
                            view_trace(
                                &direct.tokenizer_view,
                                direct.tokenizer_view.dfa().start_state,
                                &token[offset..],
                            ),
                            "active_groups={active_groups:?} reset suffix={:?}",
                            &token[offset..],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn stable_restricted_observation_quotient_preserves_relevant_powerset_traces() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let active_groups = [true, true];
        let mut relevant_bytes = [false; 256];
        relevant_bytes[b'a' as usize] = true;
        relevant_bytes[b'b' as usize] = true;
        let stable_map = compute_state_map(
            &tokenizer,
            &relevant_bytes,
            Some(&active_groups),
            None,
            RefinementDepth::Stable,
        );
        let raw_powerset = build_relevant_powerset_view(
            &tokenizer,
            &relevant_bytes,
            Some(&active_groups),
            None,
        );
        let quotient_powerset = build_relevant_powerset_view(
            &tokenizer,
            &relevant_bytes,
            Some(&active_groups),
            Some(&stable_map),
        );
        let raw_start_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let owned_tokens = [
            Vec::new(),
            b"a".to_vec(),
            b"b".to_vec(),
            b"ab".to_vec(),
            b"ba".to_vec(),
            b"aab".to_vec(),
            b"bba".to_vec(),
            b"abba".to_vec(),
        ];
        let tokens = owned_tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let raw_bounded = build_bounded_analysis_view_from_relevant_powerset(
            &raw_powerset,
            &raw_start_states,
            &tokens,
        );
        let quotient_bounded = build_bounded_analysis_view_from_relevant_powerset(
            &quotient_powerset,
            &raw_start_states,
            &tokens,
        );

        for &raw_state in &raw_start_states {
            let raw_start = raw_bounded.view_state_for_raw_start(raw_state);
            let quotient_start = quotient_bounded.view_state_for_raw_start(raw_state);
            for token in &tokens {
                assert_eq!(
                    view_trace(&raw_bounded.tokenizer_view, raw_start, token),
                    view_trace(&quotient_bounded.tokenizer_view, quotient_start, token),
                    "raw_state={raw_state} token={token:?}"
                );
            }
        }
        for token in &tokens {
            for offset in 0..token.len() {
                assert_eq!(
                    view_trace(
                        &raw_bounded.tokenizer_view,
                        raw_bounded.tokenizer_view.dfa().start_state,
                        &token[offset..],
                    ),
                    view_trace(
                        &quotient_bounded.tokenizer_view,
                        quotient_bounded.tokenizer_view.dfa().start_state,
                        &token[offset..],
                    ),
                    "reset suffix={:?}",
                    &token[offset..],
                );
            }
        }
    }

    #[test]
    fn relevant_powerset_handles_fully_filtered_empty_configurations() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let relevant = [true; 256];
        let inactive_groups = [false, false];
        let view = build_relevant_powerset_view(
            &tokenizer,
            &relevant,
            Some(&inactive_groups),
            None,
        );

        assert!(view.configurations.iter().all(|config| config.is_empty()));
        assert!(view.edges.is_empty());
        assert!(view.raw_start_to_view.iter().all(|&state| state == view.start_state as u32));
    }

    #[test]
    fn relevant_powerset_budget_aborts_without_changing_successful_construction() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let relevant = [true; 256];
        let reference = build_relevant_powerset_view(&tokenizer, &relevant, None, None);

        let generous = build_relevant_powerset_view_budgeted(
            &tokenizer,
            &relevant,
            None,
            None,
            RelevantPowersetWorkBudget {
                max_configurations: reference.configurations.len(),
                max_edges: reference.edges.len(),
            },
        )
        .expect("budget equal to realized exact work must succeed");
        assert_eq!(generous.configurations.as_ref(), reference.configurations.as_ref());
        assert_eq!(generous.edge_offsets, reference.edge_offsets);
        assert_eq!(generous.edges, reference.edges);

        let too_small = build_relevant_powerset_view_budgeted(
            &tokenizer,
            &relevant,
            None,
            None,
            RelevantPowersetWorkBudget {
                max_configurations: reference.configurations.len(),
                max_edges: reference.edges.len().saturating_sub(1),
            },
        );
        let Err(work) = too_small else {
            panic!("undersized edge budget must abort powerset construction");
        };
        assert!(work.edges > reference.edges.len().saturating_sub(1));
    }

    #[test]
    fn set_valued_refinement_distinguishes_epsilon_successor_class_sets() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let mut relevant = [false; 256];
        relevant[b'a' as usize] = true;
        relevant[b'b' as usize] = true;
        let map = compute_state_map(
            &tokenizer,
            &relevant,
            None,
            None,
            RefinementDepth::Stable,
        );
        assert_ne!(map.original_to_internal[2], map.original_to_internal[4]);
        assert_ne!(map.original_to_internal[1], map.original_to_internal[2]);
    }

    #[test]
    fn identity_input_map_does_not_block_nfa_equivalence_merges() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let relevant = [true; 256];
        let identity = super::super::identity_state_map(tokenizer.num_states() as usize);
        let direct = compute_state_map(
            &tokenizer,
            &relevant,
            None,
            None,
            RefinementDepth::Stable,
        );
        let from_identity = compute_state_map(
            &tokenizer,
            &relevant,
            None,
            Some(&identity),
            RefinementDepth::Stable,
        );

        assert!(direct.num_internal_ids() < tokenizer.num_states());
        assert!(same_partition(
            &direct.original_to_internal,
            &from_identity.original_to_internal,
        ));
    }

    #[test]
    fn direct_nfa_worklist_matches_synchronous_refinement() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let mut relevant_ab = [false; 256];
        relevant_ab[b'a' as usize] = true;
        relevant_ab[b'b' as usize] = true;
        let relevant_all = [true; 256];
        let active_left = [true, false];
        let active_right = [false, true];
        let active_both = [true, true];

        for relevant in [&relevant_ab, &relevant_all] {
            for active_groups in [
                None,
                Some(active_left.as_slice()),
                Some(active_right.as_slice()),
                Some(active_both.as_slice()),
            ] {
                let synchronous = compute_state_map(
                    &tokenizer,
                    relevant,
                    active_groups,
                    None,
                    RefinementDepth::Bounded(tokenizer.num_states() as usize),
                );
                let worklist = compute_state_map(
                    &tokenizer,
                    relevant,
                    active_groups,
                    None,
                    RefinementDepth::Stable,
                );
                assert!(same_partition(
                    &synchronous.original_to_internal,
                    &worklist.original_to_internal,
                ));

                let seed = compute_state_map(
                    &tokenizer,
                    relevant,
                    active_groups,
                    None,
                    RefinementDepth::Bounded(1),
                );
                let seeded_synchronous = compute_state_map(
                    &tokenizer,
                    relevant,
                    active_groups,
                    Some(&seed),
                    RefinementDepth::Bounded(tokenizer.num_states() as usize),
                );
                let seeded_worklist = compute_state_map(
                    &tokenizer,
                    relevant,
                    active_groups,
                    Some(&seed),
                    RefinementDepth::Stable,
                );
                assert!(same_partition(
                    &seeded_synchronous.original_to_internal,
                    &seeded_worklist.original_to_internal,
                ));
            }
        }
    }

    #[test]
    fn prebuilt_sparse_powerset_refinement_matches_fresh_nfa_refinement() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let mut relevant = [false; 256];
        relevant[b'a' as usize] = true;
        relevant[b'b' as usize] = true;
        let direct = compute_state_map(
            &tokenizer,
            &relevant,
            None,
            None,
            RefinementDepth::Stable,
        );
        let view = build_relevant_powerset_view(&tokenizer, &relevant, None, None);
        let mut output_ids = FxHashMap::<Vec<u64>, u32>::default();
        let output_class_by_config = view
            .configurations
            .iter()
            .map(|config| {
                let observation = observation_words(&tokenizer, config, None);
                let next = output_ids.len() as u32;
                *output_ids.entry(observation).or_insert(next)
            })
            .collect::<Vec<_>>();
        let reused = compute_state_map_from_prebuilt_sparse_powerset(
            &tokenizer,
            None,
            RefinementDepth::Stable,
            &view.raw_start_to_view,
            &view.configurations,
            &output_class_by_config,
            None,
            &view.edge_offsets,
            &view.edges,
        );
        let synchronous = compute_state_map_from_prebuilt_sparse_powerset(
            &tokenizer,
            None,
            RefinementDepth::Bounded(tokenizer.num_states() as usize),
            &view.raw_start_to_view,
            &view.configurations,
            &output_class_by_config,
            None,
            &view.edge_offsets,
            &view.edges,
        );

        assert!(same_partition(
            &direct.original_to_internal,
            &reused.original_to_internal,
        ));
        assert!(same_partition(
            &synchronous.original_to_internal,
            &reused.original_to_internal,
        ));
    }

    #[test]
    fn prebuilt_sparse_worklist_matches_projected_synchronous_refinement() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let mut relevant = [false; 256];
        relevant[b'a' as usize] = true;
        relevant[b'b' as usize] = true;
        let active_groups = [true, false];
        let view = build_relevant_powerset_view(
            &tokenizer,
            &relevant,
            Some(&active_groups),
            None,
        );
        let output_class_by_config = powerset_output_class_ids(&view);
        let active_language = raw_active_language_states(&tokenizer, Some(&active_groups))
            .expect("active group projection");

        let seed = compute_state_map_from_prebuilt_sparse_powerset(
            &tokenizer,
            None,
            RefinementDepth::Bounded(1),
            &view.raw_start_to_view,
            &view.configurations,
            &output_class_by_config,
            Some(&active_language),
            &view.edge_offsets,
            &view.edges,
        );
        let worklist = compute_state_map_from_prebuilt_sparse_powerset(
            &tokenizer,
            Some(&seed),
            RefinementDepth::Stable,
            &view.raw_start_to_view,
            &view.configurations,
            &output_class_by_config,
            Some(&active_language),
            &view.edge_offsets,
            &view.edges,
        );
        let synchronous = compute_state_map_from_prebuilt_sparse_powerset(
            &tokenizer,
            Some(&seed),
            RefinementDepth::Bounded(tokenizer.num_states() as usize),
            &view.raw_start_to_view,
            &view.configurations,
            &output_class_by_config,
            Some(&active_language),
            &view.edge_offsets,
            &view.edges,
        );

        assert!(same_partition(
            &synchronous.original_to_internal,
            &worklist.original_to_internal,
        ));
    }

    #[test]
    fn prebuilt_sparse_worklist_matches_synchronous_on_random_topologies() {
        fn next_u32(state: &mut u64) -> u32 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*state >> 32) as u32
        }

        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let num_states = tokenizer.num_states() as usize;
        for seed in 0..128u64 {
            let mut rng = seed ^ 0x9e37_79b9_7f4a_7c15;
            let extra_configs = num_states * 3 + 1;
            let mut configurations = (0..num_states)
                .map(|state| vec![state as u32].into_boxed_slice())
                .collect::<Vec<_>>();
            for _ in 0..extra_configs {
                let mut config = Vec::<u32>::new();
                for raw in 0..num_states {
                    if next_u32(&mut rng).is_multiple_of(3) {
                        config.push(raw as u32);
                    }
                }
                if config.is_empty() && next_u32(&mut rng) & 1 != 0 {
                    config.push((next_u32(&mut rng) as usize % num_states) as u32);
                }
                configurations.push(config.into_boxed_slice());
            }

            let raw_start_to_view = (0..num_states as u32).collect::<Vec<_>>();
            let output_class_by_config = (0..configurations.len())
                .map(|_| next_u32(&mut rng) % 5)
                .collect::<Vec<_>>();
            let mut edge_offsets = Vec::<u32>::with_capacity(configurations.len() + 1);
            let mut edges = Vec::<(u8, u32)>::new();
            edge_offsets.push(0);
            for _ in 0..configurations.len() {
                for byte in 0..5u8 {
                    if next_u32(&mut rng) & 1 != 0 {
                        edges.push((
                            byte,
                            next_u32(&mut rng) % configurations.len() as u32,
                        ));
                    }
                }
                edge_offsets.push(edges.len() as u32);
            }
            let mut active_language = (0..num_states)
                .map(|_| next_u32(&mut rng) & 1 != 0)
                .collect::<Vec<_>>();
            if !active_language.iter().any(|&active| active) {
                active_language[next_u32(&mut rng) as usize % num_states] = true;
            }

            for active_language in [None, Some(active_language.as_slice())] {
                let synchronous = compute_state_map_from_prebuilt_sparse_powerset(
                    &tokenizer,
                    None,
                    RefinementDepth::Bounded(num_states),
                    &raw_start_to_view,
                    &configurations,
                    &output_class_by_config,
                    active_language,
                    &edge_offsets,
                    &edges,
                );
                let worklist = compute_state_map_from_prebuilt_sparse_powerset(
                    &tokenizer,
                    None,
                    RefinementDepth::Stable,
                    &raw_start_to_view,
                    &configurations,
                    &output_class_by_config,
                    active_language,
                    &edge_offsets,
                    &edges,
                );
                assert!(
                    same_partition(
                        &synchronous.original_to_internal,
                        &worklist.original_to_internal,
                    ),
                    "unseeded mismatch at seed={seed} active_projection={}",
                    active_language.is_some(),
                );

                let initial = compute_state_map_from_prebuilt_sparse_powerset(
                    &tokenizer,
                    None,
                    RefinementDepth::Bounded(1),
                    &raw_start_to_view,
                    &configurations,
                    &output_class_by_config,
                    active_language,
                    &edge_offsets,
                    &edges,
                );
                let seeded_synchronous = compute_state_map_from_prebuilt_sparse_powerset(
                    &tokenizer,
                    Some(&initial),
                    RefinementDepth::Bounded(num_states),
                    &raw_start_to_view,
                    &configurations,
                    &output_class_by_config,
                    active_language,
                    &edge_offsets,
                    &edges,
                );
                let seeded_worklist = compute_state_map_from_prebuilt_sparse_powerset(
                    &tokenizer,
                    Some(&initial),
                    RefinementDepth::Stable,
                    &raw_start_to_view,
                    &configurations,
                    &output_class_by_config,
                    active_language,
                    &edge_offsets,
                    &edges,
                );
                assert!(
                    same_partition(
                        &seeded_synchronous.original_to_internal,
                        &seeded_worklist.original_to_internal,
                    ),
                    "seeded mismatch at seed={seed} active_projection={}",
                    active_language.is_some(),
                );
            }
        }
    }

    #[test]
    fn bounded_analysis_view_materializes_reset_suffix_transitions_from_start_state() {
        let tokenizer =
            crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
        let raw_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens: Vec<&[u8]> = vec![b"xa", b"xaa", b"xb"];
        let bounded = build_bounded_analysis_view_with_trie(
            &tokenizer,
            &raw_states,
            &tokens,
            None,
            None,
        );
        let dfa = bounded.tokenizer_view.dfa();
        let start_state = dfa.start_state;
        let trans_a = dfa.trans(start_state, b'a' as usize);
        assert_ne!(
            trans_a,
            u32::MAX,
            "bounded view must materialize suffix transitions from start_state for byte 'a'",
        );
    }
}
