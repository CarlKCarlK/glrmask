//! Rooted terminal interchangeability for the L2+ terminal-DWA reference path.
//!
//! For one vocabulary partition, the tokenizer DFA itself is left unchanged.
//! The partition only chooses which byte transitions `characterize` traverses.
//! In particular, the tokenizer's stored finalizer and future-finalizer metadata
//! is frozen: it is neither recomputed nor minimized after restricting bytes.
//!
//! For terminals `left` and `right`, characterize the tagged state `(map, s)` by
//! hashing, for every enabled byte, the destination's previous-round
//! characterization together with its mapped finalizers and mapped frozen future
//! finalizers. The identity and `left <-> right` sides share the same hashes.
//! Their stable hash classes are the two state partitions of the rooted terminal
//! interchange map. The initial lexer state must occur in the same class on both
//! sides.

use std::collections::{BTreeMap, BTreeSet, VecDeque, hash_map::Entry};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use smallvec::SmallVec;
use super::nwa_builder::{TerminalNwaTransportMode, TransportScannerStateMap};
use super::equivalence_analysis::state_equivalence::nfa::{
    RefinementDepth, RelevantPowersetView, RelevantPowersetWork,
    RelevantPowersetWorkBudget, build_relevant_powerset_view,
    build_relevant_powerset_view_budgeted, compute_state_map_from_prebuilt_sparse_powerset,
    raw_active_language_states,
};
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
        use crate::automata::weighted::dwa::{DWAState, DWA};
use crate::compiler::stages::equiv_types::{GlobalScannerStateQuotient, ManyToOneIdMap};
use crate::ds::weight::{SharedTokenSet, Weight, shared_rangeset};
use crate::ds::bitset::BitSet;
use crate::grammar::flat::TerminalID;

const CHARACTERIZATION_DOMAIN: &[u8] =
    b"glrmask terminal interchangeability characterize v1\0";
const CHARACTERIZATION_SEED: &[u8] =
    b"glrmask terminal interchangeability characterize seed v1\0";

const TI_NFA_POWERSET_GROWTH_FACTOR: usize = 8;
const TI_NFA_POWERSET_MIN_CONFIGURATIONS: usize = 16_384;
const TI_NFA_POWERSET_MAX_CONFIGURATIONS: usize = 262_144;
const TI_NFA_POWERSET_MIN_EDGES: usize = 1_048_576;
const TI_NFA_POWERSET_MAX_EDGES: usize = 8_388_608;

fn ti_nfa_powerset_budget(
    tokenizer: &Tokenizer,
    relevant_bytes: &[bool; 256],
    global_state_quotient: Option<&GlobalScannerStateQuotient>,
) -> RelevantPowersetWorkBudget {
    fn env_limit(name: &str, computed: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(computed)
    }

    let seed_states = global_state_quotient.map_or(tokenizer.num_states() as usize, |quotient| {
        quotient.as_many_to_one().representative_original_ids.len()
    });
    let selected_bytes = relevant_bytes.iter().filter(|&&selected| selected).count();
    let computed_configurations = seed_states
        .saturating_mul(TI_NFA_POWERSET_GROWTH_FACTOR)
        .clamp(
            TI_NFA_POWERSET_MIN_CONFIGURATIONS,
            TI_NFA_POWERSET_MAX_CONFIGURATIONS,
        );
    let computed_edges = seed_states
        .saturating_mul(selected_bytes)
        .saturating_mul(TI_NFA_POWERSET_GROWTH_FACTOR)
        .clamp(TI_NFA_POWERSET_MIN_EDGES, TI_NFA_POWERSET_MAX_EDGES);

    RelevantPowersetWorkBudget {
        max_configurations: env_limit(
            "GLRMASK_TI_NFA_POWERSET_MAX_CONFIGURATIONS",
            computed_configurations,
        ),
        max_edges: env_limit(
            "GLRMASK_TI_NFA_POWERSET_MAX_EDGES",
            computed_edges,
        ),
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct CharacterizationHash([u8; blake3::OUT_LEN]);

impl CharacterizationHash {
    fn seed() -> Self {
        Self(*blake3::hash(CHARACTERIZATION_SEED).as_bytes())
    }
}

/// Sparse, sorted active-terminal labels for one frozen output family.
///
/// TI partitions typically observe only a handful of terminals at a lexer
/// state. Keeping a full active-terminal bitmap for every state therefore
/// spends most planning time allocating and copying zero words. Raw terminal
/// ids remain exact labels: disabled terminals are simply absent for the
/// current iterative round.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct OutputBits(SmallVec<[TerminalID; 4]>);

impl OutputBits {
    fn new(_words: usize) -> Self { Self(SmallVec::new()) }

    fn from_active(terminals: &[TerminalID], active_terminals: &[bool]) -> Self {
        let mut output = SmallVec::<[TerminalID; 4]>::new();
        for &terminal in terminals {
            if active_terminals
                .get(terminal as usize)
                .copied()
                .unwrap_or(false)
            {
                output.push(terminal);
            }
        }
        Self(output)
    }

    fn contains(&self, terminal: usize) -> bool {
        self.0.binary_search(&(terminal as TerminalID)).is_ok()
    }

    /// True iff swapping `left` and `right` changes this set, i.e. exactly one
    /// of them is present. When false, `mapped` would return an identical set.
    fn swap_changes(&self, left: TerminalID, right: TerminalID) -> bool {
        self.0.binary_search(&left).is_ok() != self.0.binary_search(&right).is_ok()
    }

    fn mapped(&self, swap: Option<(usize, usize)>) -> Self {
        let Some((left, right)) = swap else { return self.clone(); };
        if left == right { return self.clone(); }
        let left = left as TerminalID;
        let right = right as TerminalID;
        let left_present = self.0.binary_search(&left).ok();
        let right_present = self.0.binary_search(&right).ok();
        if left_present.is_some() == right_present.is_some() { return self.clone(); }
        let (source_index, replacement) = if let Some(index) = left_present {
            (index, right)
        } else {
            (right_present.expect("one swapped terminal must be present"), left)
        };
        let mut terminals = self.0.clone();
        terminals.remove(source_index);
        let insertion = terminals
            .binary_search(&replacement)
            .expect_err("the replacement terminal must be absent");
        terminals.insert(insertion, replacement);
        Self(terminals)
    }

    fn append_to(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&(self.0.len() as u32).to_le_bytes());
        for &terminal in &self.0 {
            output.extend_from_slice(&(terminal as u32).to_le_bytes());
        }
    }
}

/// One observed frozen-output label. The reference only reads these labels
/// at enabled-byte destinations, so this type is used solely for the global
/// destination-output closure prefilter below.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct OutputPair {
    finalizers: OutputBits,
    future_finalizers: OutputBits,
}

#[derive(Clone, Copy, Default)]
struct GenericOutputTrackStats {
    group_count: u32,
    outside_count: u32,
    outside_hash: u64,
}

#[derive(Clone, Copy, Default)]
struct GenericOutputPairStats {
    finalizers: GenericOutputTrackStats,
    future_finalizers: GenericOutputTrackStats,
}

/// One exact component of the sparse characterization tuple. The output id
/// names an immutable pair of frozen destination-output sets. Class ids are
/// only equality labels within one round; hash-map equality still compares the
/// complete tuple, so this avoids cryptographic hashing without relying on hash
/// collisions for semantics.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CanonicalComponent {
    byte: u8,
    previous_class: u32,
    output: u32,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CanonicalSignature(SmallVec<[CanonicalComponent; 8]>);

struct CanonicalRound {
    classes: Vec<u32>,
    /// One raw-state representative for every exact class in this round.
    /// Identity-round construction hashes the sparse row directly, then
    /// compares it against these representatives on every hash match.  This
    /// avoids allocating a `CanonicalSignature` for every raw state while
    /// retaining collision-free equality semantics.
    representative_by_class: Vec<u32>,
    classes_by_signature_hash: FxHashMap<u64, SmallVec<[u32; 1]>>,
}

/// Stable identity characterization collapsed to its exact state quotient.
/// The quotient does not merge tokenizer states; it is only a cached view used
/// to certify a terminal-swap automorphism before materializing its raw map.
struct CanonicalQuotient {
    class_for_state: Arc<[u32]>,
    representative_by_class: Arc<[u32]>,
    reverse_predecessors: Vec<Vec<u32>>,
    /// Identity class labels, projected onto stable quotient representatives,
    /// for rounds 0 through the identity fixed point.
    identity_classes_by_round: Vec<Vec<u32>>,
    /// Multiplicity of each identity class at the matching round. This lets a
    /// sparse swapped update prove class-set equality without scanning every
    /// quotient class.
    identity_class_counts_by_round: Vec<Vec<u32>>,
    /// At the identity fixed point, the preceding class partition maps
    /// bijectively to the next one. Sparse swapped rows must respect this map.
    stable_previous_to_next: Vec<u32>,
    stable_next_to_previous: Vec<u32>,
}

/// The stable identity quotient data needed by the common support-transport
/// proof.  It deliberately omits canonical row signatures and historical
/// round projections, which are required only by the slower generic fallback.
struct SupportQuotient {
    class_for_state: Arc<[u32]>,
    representative_by_class: Arc<[u32]>,
    reverse_predecessors: Vec<Vec<u32>>,
    /// Byte-labelled reverse edges over quotient representatives.  This lets
    /// sparse support permutations prove incoming-edge compatibility without
    /// rescanning every fixed predecessor's complete outgoing row.
    reverse_edges: Vec<Vec<(u8, u32)>>,
    // Deviation-independent scanner projection for this quotient, precomputed
    // once. Every accepted support-transposition map shares these two Arcs and
    // only attaches its own per-pair deviations, so building an accepted map is
    // O(deviations) instead of the O(raw_state_count) projection rebuild that
    // `scanner_map_from_internal_quotient` performs from scratch.
    scanner_state_count: usize,
    scanner_class_for_original: Arc<[u32]>,
    scanner_representative_for_class: Arc<[u32]>,
}

/// Exact stable support partition retained across iterative TI rounds. Removing
/// active terminals only projects labels away, so the next round's stable
/// partition can only coarsen this one. The active mask guards against using the
/// seed outside that monotone discovery sequence.
#[derive(Clone)]
struct SupportPartitionSeed {
    active_terminals: Arc<[bool]>,
    quotient: Arc<SupportQuotient>,
}

type SupportDeviationMap = SmallVec<[(u32, u32); 8]>;

/// Exact round-local active output alphabet retained for projection into the
/// next monotone active mask. Output-pair projection commutes with removing
/// inactive terminal labels, so remapping distinct pairs is enough; raw states
/// only replay their previous compact pair id.
#[derive(Clone)]
struct OutputProjectionSeed {
    active_terminals: Arc<[bool]>,
    output_pairs: Arc<[OutputPair]>,
    output_pair_by_state: Arc<[u32]>,
}


/// Per-swap output-label relabelling. The immutable base ids represent the
/// original frozen output pairs; ids allocated after `base_count` are local to
/// this swap and compare equal only to the same full mapped pair.
struct SwappedOutputIds<'a> {
    base_pairs: &'a [OutputPair],
    base_lookup: &'a FxHashMap<OutputPair, u32>,
    left: usize,
    right: usize,
    mapped: FxHashMap<u32, u32>,
    local: FxHashMap<OutputPair, u32>,
}

impl<'a> SwappedOutputIds<'a> {
    fn new(
        base_pairs: &'a [OutputPair],
        base_lookup: &'a FxHashMap<OutputPair, u32>,
        left: usize,
        right: usize,
    ) -> Self {
        Self {
            base_pairs,
            base_lookup,
            left,
            right,
            mapped: FxHashMap::default(),
            local: FxHashMap::default(),
        }
    }

    fn id(&mut self, output: u32) -> u32 {
        if let Some(&cached) = self.mapped.get(&output) {
            return cached;
        }
        let index = output as usize;
        let mapped = OutputPair {
            finalizers: self.base_pairs[index]
                .finalizers
                .mapped(Some((self.left, self.right))),
            future_finalizers: self.base_pairs[index]
                .future_finalizers
                .mapped(Some((self.left, self.right))),
        };
        let id = if let Some(&base) = self.base_lookup.get(&mapped) {
            base
        } else {
            let base_count = self.base_pairs.len() as u32;
            let next = base_count + self.local.len() as u32;
            *self.local.entry(mapped).or_insert(next)
        };
        self.mapped.insert(output, id);
        id
    }
}

/// The support-transposition proof touches only a handful of output-pair ids
/// per candidate. Keep its relabelling cache inline; unrestricted fallback
/// refinement retains `SwappedOutputIds` above.
struct SparseSwappedOutputIds<'a> {
    base_pairs: &'a [OutputPair],
    base_lookup: &'a FxHashMap<OutputPair, u32>,
    left: usize,
    right: usize,
    mapped: SmallVec<[(u32, u32); 16]>,
    local: SmallVec<[(OutputPair, u32); 4]>,
}

impl<'a> SparseSwappedOutputIds<'a> {
    fn new(
        base_pairs: &'a [OutputPair],
        base_lookup: &'a FxHashMap<OutputPair, u32>,
        left: usize,
        right: usize,
    ) -> Self {
        Self::with_mapped(base_pairs, base_lookup, left, right, SmallVec::new())
    }

    fn with_mapped(
        base_pairs: &'a [OutputPair],
        base_lookup: &'a FxHashMap<OutputPair, u32>,
        left: usize,
        right: usize,
        mapped: SmallVec<[(u32, u32); 16]>,
    ) -> Self {
        Self {
            base_pairs,
            base_lookup,
            left,
            right,
            mapped,
            local: SmallVec::new(),
        }
    }

    fn id(&mut self, output: u32) -> u32 {
        if let Some((_, id)) = self.mapped.iter().find(|&&(seen, _)| seen == output) {
            return *id;
        }
        let pair = &self.base_pairs[output as usize];
        let left = self.left as TerminalID;
        let right = self.right as TerminalID;
        if !pair.finalizers.swap_changes(left, right)
            && !pair.future_finalizers.swap_changes(left, right)
        {
            self.mapped.push((output, output));
            return output;
        }
        let mapped = OutputPair {
            finalizers: pair.finalizers.mapped(Some((self.left, self.right))),
            future_finalizers: pair
                .future_finalizers
                .mapped(Some((self.left, self.right))),
        };
        let id = if let Some(&base) = self.base_lookup.get(&mapped) {
            base
        } else if let Some((_, existing)) = self.local.iter().find(|(seen, _)| *seen == mapped) {
            *existing
        } else {
            let next = self.base_pairs.len() as u32 + self.local.len() as u32;
            self.local.push((mapped, next));
            next
        };
        self.mapped.push((output, id));
        id
    }
}

/// Sparse topology of the byte-restricted tokenizer DFA. Its real-state domain
/// is either the raw tokenizer state domain or a global scanner-state
/// quotient. A missing enabled transition has the same synthetic dead
/// destination for every real state. Keeping only real edges is exact: all
/// omitted bytes share that one default.
#[derive(Debug)]
struct RestrictedTopology {
    bytes: Vec<u8>,
    edge_offsets: Vec<u32>,
    edges: Vec<(u8, u32)>,
    reverse_predecessors: Arc<[Vec<u32>]>,
    observed_destinations: Arc<[bool]>,
    /// Number of raw tokenizer states. Transport witnesses must retain this
    /// external scanner coordinate even when discovery runs on a quotient.
    raw_state_count: usize,
    /// Raw scanner state selected for each discovery-domain state.
    raw_representative_by_state: Option<Arc<[u32]>>,
    /// Same representative vector, but with synthetic/unrepresented classes
    /// assigned an arbitrary in-range raw representative so it can be shared
    /// directly by transport quotient maps. This is immutable per topology;
    /// constructing it once avoids one allocation/copy for every accepted TI
    /// witness.
    transport_representative_by_state: Option<Arc<[u32]>>,
    /// Discovery-domain state for every raw scanner state.
    state_for_raw: Option<Arc<[u32]>>,
    /// Exact frozen output rows for a powerset discovery domain. Ordinary raw
    /// and quotient topologies derive these from raw-state metadata instead.
    nfa_output_rows: Option<(
        Arc<[Box<[TerminalID]>]>,
        Arc<[Box<[TerminalID]>]>,
    )>,
    /// Exact members of each powerset configuration. Only raw-coordinate NFA
    /// configurations may be reused by raw-state equivalence refinement.
    nfa_configurations: Option<Arc<[Box<[u32]>]>>,
    nfa_configurations_use_raw_states: bool,
    real_state_count: usize,
    initial_state: usize,
    max_outdegree: usize,
}

impl RestrictedTopology {
    fn new(tokenizer: &Tokenizer, relevant_bytes: &[bool; 256]) -> Self {
        Self::new_inner(tokenizer, relevant_bytes, None)
    }

    /// Build a restricted topology over a total scanner-state representative
    /// map. The map need not be a symmetric DFA equivalence: discovery follows
    /// the chosen representative for each raw state, so the caller must prove
    /// member-to-representative substitution for the observations reachable in
    /// this vocabulary partition. Partition C supplies that directional proof
    /// by positional subsumption and leaves the collapsed suffix tail exact.
    fn new_with_global_state_quotient(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        global_state_quotient: &GlobalScannerStateQuotient,
    ) -> Self {
        Self::new_inner(tokenizer, relevant_bytes, Some(global_state_quotient))
    }

    fn build_budgeted(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        global_state_quotient: Option<&GlobalScannerStateQuotient>,
        budget: RelevantPowersetWorkBudget,
    ) -> Result<Self, RelevantPowersetWork> {
        if tokenizer.has_epsilon_transitions() {
            return Self::new_nfa_budgeted(
                tokenizer,
                relevant_bytes,
                global_state_quotient,
                budget,
            );
        }
        Ok(Self::new_inner(
            tokenizer,
            relevant_bytes,
            global_state_quotient,
        ))
    }

    fn new_inner(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        global_state_quotient: Option<&GlobalScannerStateQuotient>,
    ) -> Self {
        if tokenizer.has_epsilon_transitions() {
            return Self::new_nfa(tokenizer, relevant_bytes, global_state_quotient);
        }
        let bytes = (0..=255u8)
            .filter(|&byte| relevant_bytes[byte as usize])
            .collect::<Vec<_>>();
        let raw_state_count = tokenizer.num_states() as usize;
        let (state_for_raw, raw_representative_by_state): (
            Option<Arc<[u32]>>,
            Option<Arc<[u32]>>,
        ) =
            match global_state_quotient {
                Some(global_state_quotient) => {
                    assert_eq!(
                        global_state_quotient.raw_state_count(),
                        raw_state_count,
                        "global scanner-state quotient must match the tokenizer state domain",
                    );
                    let map = global_state_quotient.as_many_to_one();
                    assert_eq!(
                        map.original_to_internal.len(),
                        raw_state_count,
                        "global scanner-state quotient must cover every tokenizer state",
                    );
                    assert!(
                        !map.representative_original_ids.is_empty() || raw_state_count == 0,
                        "global scanner-state quotient must retain one representative per class",
                    );
                    for (raw_state, &class) in map.original_to_internal.iter().enumerate() {
                        assert!(
                            class != u32::MAX
                                && (class as usize) < map.representative_original_ids.len(),
                            "global scanner-state quotient omitted raw tokenizer state {raw_state}",
                        );
                    }
                    for (class, &representative) in
                        map.representative_original_ids.iter().enumerate()
                    {
                        let representative = representative as usize;
                        assert!(
                            representative < raw_state_count
                                && map.original_to_internal[representative] == class as u32,
                            "global scanner-state quotient representative must belong to its class",
                        );
                    }

                    (
                        Some(Arc::from(map.original_to_internal.clone())),
                        Some(Arc::from(map.representative_original_ids.clone())),
                    )
                }
                None => (None, None),
            };
        let real_state_count = raw_representative_by_state
            .as_ref()
            .map_or(raw_state_count, |representatives| representatives.len());
        let mut edge_offsets = Vec::with_capacity(real_state_count + 2);
        let mut edges = Vec::new();
        let mut reverse_predecessors = vec![Vec::<u32>::new(); real_state_count];
        let mut observed_destinations = vec![false; real_state_count + 1];
        let mut max_outdegree = 0usize;
        edge_offsets.push(0);
        for state in 0..real_state_count {
            let start = edges.len();
            let raw_state = raw_representative_by_state
                .as_ref()
                .map_or(state as u32, |representatives| representatives[state]);
            for (byte, target) in tokenizer.transitions_from(raw_state) {
                if relevant_bytes[byte as usize] {
                    let target = state_for_raw
                        .as_ref()
                        .map_or(target, |classes| classes[target as usize]);
                    edges.push((byte, target));
                    reverse_predecessors[target as usize].push(state as u32);
                    observed_destinations[target as usize] = true;
                }
            }
            max_outdegree = max_outdegree.max(edges.len() - start);
            edge_offsets.push(edges.len() as u32);
        }
        // Synthetic dead has no real edges: every enabled byte loops to itself.
        edge_offsets.push(edges.len() as u32);
        let initial_state = state_for_raw
            .as_ref()
            .map_or(tokenizer.initial_state_id() as usize, |classes| {
                classes[tokenizer.initial_state_id() as usize] as usize
            });
        let transport_representative_by_state = raw_representative_by_state.as_ref().map(|raw| {
            if raw.iter().all(|&representative| representative != u32::MAX) {
                return Arc::clone(raw);
            }
            let fallback = raw
                .iter()
                .copied()
                .find(|&representative| representative != u32::MAX)
                .unwrap_or(0);
            raw.iter()
                .copied()
                .map(|representative| {
                    if representative == u32::MAX {
                        fallback
                    } else {
                        representative
                    }
                })
                .collect::<Vec<_>>()
                .into()
        });
        Self {
            bytes,
            edge_offsets,
            edges,
            reverse_predecessors: reverse_predecessors.into(),
            observed_destinations: observed_destinations.into(),
            raw_state_count,
            raw_representative_by_state,
            transport_representative_by_state,
            state_for_raw,
            nfa_output_rows: None,
            nfa_configurations: None,
            nfa_configurations_use_raw_states: false,
            real_state_count,
            initial_state,
            max_outdegree,
        }
    }

    fn new_nfa(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        global_state_quotient: Option<&GlobalScannerStateQuotient>,
    ) -> Self {
        let view = build_relevant_powerset_view(
            tokenizer,
            relevant_bytes,
            None,
            global_state_quotient.map(GlobalScannerStateQuotient::as_many_to_one),
        );
        Self::from_nfa_view(tokenizer, view, global_state_quotient.is_none())
    }

    fn new_nfa_budgeted(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        global_state_quotient: Option<&GlobalScannerStateQuotient>,
        budget: RelevantPowersetWorkBudget,
    ) -> Result<Self, RelevantPowersetWork> {
        let view = build_relevant_powerset_view_budgeted(
            tokenizer,
            relevant_bytes,
            None,
            global_state_quotient.map(GlobalScannerStateQuotient::as_many_to_one),
            budget,
        )?;
        Ok(Self::from_nfa_view(
            tokenizer,
            view,
            global_state_quotient.is_none(),
        ))
    }

    fn from_nfa_view(
        tokenizer: &Tokenizer,
        view: RelevantPowersetView,
        configurations_use_raw_states: bool,
    ) -> Self {
        let raw_state_count = tokenizer.num_states() as usize;
        let real_state_count = view.states.len();
        let nfa_configurations = Arc::clone(&view.configurations);
        let mut edge_offsets = view.edge_offsets;
        let mut reverse_predecessors = vec![Vec::<u32>::new(); real_state_count];
        let mut observed_destinations = vec![false; real_state_count + 1];
        let mut max_outdegree = 0usize;
        for state in 0..real_state_count {
            let start = edge_offsets[state] as usize;
            let end = edge_offsets[state + 1] as usize;
            for &(_, target) in &view.edges[start..end] {
                reverse_predecessors[target as usize].push(state as u32);
                observed_destinations[target as usize] = true;
            }
            max_outdegree = max_outdegree.max(end - start);
        }
        edge_offsets.push(view.edges.len() as u32);

        let mut raw_representative_by_state = vec![u32::MAX; real_state_count];
        for (raw_state, &state) in view.raw_start_to_view.iter().enumerate() {
            let representative = &mut raw_representative_by_state[state as usize];
            if *representative == u32::MAX {
                *representative = raw_state as u32;
            }
        }
        let finalizers = view
            .states
            .iter()
            .map(|state| {
                state
                    .finalizers
                    .iter()
                    .map(|&terminal| terminal as TerminalID)
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>();
        let future = view
            .states
            .iter()
            .map(|state| {
                state
                    .possible_future_group_ids
                    .iter()
                    .map(|&terminal| terminal as TerminalID)
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>();

        let raw_representative_by_state: Arc<[u32]> = raw_representative_by_state.into();
        let fallback = raw_representative_by_state
            .iter()
            .copied()
            .find(|&representative| representative != u32::MAX)
            .unwrap_or(0);
        let transport_representative_by_state: Arc<[u32]> = if raw_representative_by_state
            .iter()
            .all(|&representative| representative != u32::MAX)
        {
            Arc::clone(&raw_representative_by_state)
        } else {
            raw_representative_by_state
                .iter()
                .copied()
                .map(|representative| {
                    if representative == u32::MAX {
                        fallback
                    } else {
                        representative
                    }
                })
                .collect::<Vec<_>>()
                .into()
        };
        Self {
            bytes: view.bytes,
            edge_offsets,
            edges: view.edges,
            reverse_predecessors: reverse_predecessors.into(),
            observed_destinations: observed_destinations.into(),
            raw_state_count,
            raw_representative_by_state: Some(raw_representative_by_state),
            transport_representative_by_state: Some(transport_representative_by_state),
            state_for_raw: Some(Arc::clone(&view.raw_start_to_view)),
            nfa_output_rows: Some((finalizers.into(), future.into())),
            nfa_configurations: Some(nfa_configurations),
            nfa_configurations_use_raw_states: configurations_use_raw_states,
            real_state_count,
            initial_state: view.start_state,
            max_outdegree,
        }
    }

    fn state_count(&self) -> usize {
        self.real_state_count + 1
    }

    fn dead_state(&self) -> usize {
        self.real_state_count
    }

    fn edges_from(&self, state: usize) -> &[(u8, u32)] {
        let start = self.edge_offsets[state] as usize;
        let end = self.edge_offsets[state + 1] as usize;
        &self.edges[start..end]
    }

    fn destination_for_byte(&self, state: usize, byte: u8) -> usize {
        if state == self.dead_state() {
            return state;
        }
        self.edges_from(state)
            .binary_search_by_key(&byte, |(edge_byte, _)| *edge_byte)
            .ok()
            .map(|index| self.edges_from(state)[index].1 as usize)
            .unwrap_or_else(|| self.dead_state())
    }

    fn observed_destinations(&self) -> Arc<[bool]> {
        Arc::clone(&self.observed_destinations)
    }

    fn reverse_predecessors(&self) -> Arc<[Vec<u32>]> {
        Arc::clone(&self.reverse_predecessors)
    }

    #[inline]
    fn raw_state_for_state(&self, state: usize) -> u32 {
        self.raw_representative_by_state
            .as_ref()
            .map_or(state as u32, |representatives| representatives[state])
    }

    #[inline]
    fn state_has_raw_source(&self, state: usize) -> bool {
        self.raw_representative_by_state
            .as_ref()
            .map_or(true, |representatives| {
                representatives
                    .get(state)
                    .copied()
                    .unwrap_or(u32::MAX)
                    != u32::MAX
            })
    }

    fn scanner_map_from_state_targets(
        &self,
        state_targets: &[u32],
    ) -> Option<TransportScannerStateMap> {
        assert_eq!(state_targets.len(), self.real_state_count);
        if self.state_for_raw.is_none() {
            return Some(TransportScannerStateMap::Explicit(Arc::from(
                state_targets.to_vec(),
            )));
        }
        let state_for_raw = self
            .state_for_raw
            .as_ref()
            .expect("quotient topology must retain raw-to-state classes");
        let raw_representative_by_state = self
            .raw_representative_by_state
            .as_ref()
            .expect("quotient topology must retain raw representatives");
        let raw_targets = (0..self.raw_state_count)
            .map(|raw_state| {
                let state = state_for_raw[raw_state] as usize;
                let target_state = state_targets[state] as usize;
                raw_representative_by_state
                    .get(target_state)
                    .copied()
                    .filter(|&representative| representative != u32::MAX)
            })
            .collect::<Option<Vec<_>>>()?;
        Some(TransportScannerStateMap::Explicit(raw_targets.into()))
    }

    /// Deviation-independent part of `scanner_map_from_internal_quotient`:
    /// projects the internal quotient onto raw scanner states. The result
    /// depends only on the quotient (not on any per-pair deviations), so callers
    /// that certify many pairs against one quotient can compute it once and
    /// share the two Arcs.
    fn scanner_projection_from_internal_quotient(
        &self,
        class_for_state: Arc<[u32]>,
        representative_for_class: Arc<[u32]>,
    ) -> (usize, Arc<[u32]>, Arc<[u32]>) {
        if self.state_for_raw.is_none() {
            return (self.real_state_count, class_for_state, representative_for_class);
        }
        let state_for_raw = self
            .state_for_raw
            .as_ref()
            .expect("quotient topology must retain raw-to-state classes");
        let raw_representative_by_state = self
            .raw_representative_by_state
            .as_ref()
            .expect("quotient topology must retain raw representatives");
        let class_for_original: Arc<[u32]> = (0..self.raw_state_count)
            .map(|raw_state| class_for_state[state_for_raw[raw_state] as usize])
            .collect::<Vec<_>>()
            .into();
        let unused_dead_representative = raw_representative_by_state
            .iter()
            .copied()
            .find(|&representative| representative != u32::MAX)
            .unwrap_or(0);
        let raw_representative_for_class: Arc<[u32]> = representative_for_class
            .iter()
            .map(|&state| {
                raw_representative_by_state
                    .get(state as usize)
                    .copied()
                    .filter(|&representative| representative != u32::MAX)
                    // The synthetic dead class has no raw scanner source. Its
                    // representative is therefore never queried, but the
                    // public transport-map shape still needs one in-range ID.
                    .unwrap_or(unused_dead_representative)
            })
            .collect::<Vec<_>>()
            .into();
        (
            self.raw_state_count,
            class_for_original,
            raw_representative_for_class,
        )
    }

    fn scanner_map_from_internal_quotient(
        &self,
        class_for_state: Arc<[u32]>,
        representative_for_class: Arc<[u32]>,
        source_class_for_target_deviations: Box<[(u32, u32)]>,
    ) -> Option<TransportScannerStateMap> {
        if self.nfa_output_rows.is_some() {
            let state_for_raw = self
                .state_for_raw
                .as_ref()
                .expect("NFA TI topology must retain raw-to-config states");
            let class_count = class_for_state
                .iter()
                .copied()
                .max()
                .map_or(0usize, |class| class as usize + 1);
            let mut raw_representative_for_class = vec![u32::MAX; class_count];
            for raw_state in 0..self.raw_state_count {
                let state = state_for_raw[raw_state] as usize;
                let class = class_for_state[state] as usize;
                if raw_representative_for_class[class] == u32::MAX {
                    raw_representative_for_class[class] = raw_state as u32;
                }
            }
            let raw_targets = (0..self.raw_state_count)
                .map(|raw_state| {
                    let state = state_for_raw[raw_state] as usize;
                    let target_class = class_for_state[state];
                    let source_class = source_class_for_target_deviations
                        .binary_search_by_key(&target_class, |&(target, _)| target)
                        .map(|index| source_class_for_target_deviations[index].1)
                        .unwrap_or(target_class) as usize;
                    raw_representative_for_class
                        .get(source_class)
                        .copied()
                        .filter(|&representative| representative != u32::MAX)
                })
                .collect::<Option<Vec<_>>>()?;
            return Some(TransportScannerStateMap::Explicit(raw_targets.into()));
        }
        let (state_count, class_for_original, representative_for_class) =
            self.scanner_projection_from_internal_quotient(class_for_state, representative_for_class);
        Some(TransportScannerStateMap::Quotient {
            state_count,
            class_for_original,
            representative_for_class,
            source_class_for_target_deviations,
        })
    }
}

/// The frozen-output observation made by the root-reachable part of the
/// byte-restricted DFA for one terminal. State IDs are deliberately kept rather
/// than quotienting or transforming the tokenizer: terminal interchangeability
/// uses the original DFA and its original metadata.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RootOutputSignature {
    finalizer_states: Box<[u32]>,
    future_finalizer_states: Box<[u32]>,
}

/// Tokenizer output columns independent of vocabulary partition and enabled
/// bytes. Every L2P partition uses the same frozen lexer metadata, so build
/// these raw-state indexes once and share them across partition contexts.
struct TiTokenizerOutputData {
    finalizer_terminals_by_raw_state: Arc<[Box<[TerminalID]>]>,
    future_finalizer_terminals_by_raw_state: Arc<[Box<[TerminalID]>]>,
    finalizer_raw_states_by_terminal: Arc<[Vec<u32>]>,
    future_finalizer_raw_states_by_terminal: Arc<[Vec<u32>]>,
    full_output_projection: OnceLock<(Arc<[OutputPair]>, Arc<[u32]>)>,
}

impl TiTokenizerOutputData {
    fn from_state_rows(
        finalizer_terminals_by_state: Arc<[Box<[TerminalID]>]>,
        future_finalizer_terminals_by_state: Arc<[Box<[TerminalID]>]>,
        terminal_count: usize,
    ) -> Self {
        assert_eq!(
            finalizer_terminals_by_state.len(),
            future_finalizer_terminals_by_state.len(),
            "TI output rows must cover the same state domain",
        );
        let mut finalizer_states_by_terminal = vec![Vec::<u32>::new(); terminal_count];
        let mut future_finalizer_states_by_terminal = vec![Vec::<u32>::new(); terminal_count];
        for (state, terminals) in finalizer_terminals_by_state.iter().enumerate() {
            for &terminal in terminals.iter() {
                finalizer_states_by_terminal[terminal as usize].push(state as u32);
            }
        }
        for (state, terminals) in future_finalizer_terminals_by_state.iter().enumerate() {
            for &terminal in terminals.iter() {
                future_finalizer_states_by_terminal[terminal as usize].push(state as u32);
            }
        }
        Self {
            finalizer_terminals_by_raw_state: finalizer_terminals_by_state,
            future_finalizer_terminals_by_raw_state: future_finalizer_terminals_by_state,
            finalizer_raw_states_by_terminal: finalizer_states_by_terminal.into(),
            future_finalizer_raw_states_by_terminal: future_finalizer_states_by_terminal.into(),
            full_output_projection: OnceLock::new(),
        }
    }

    fn from_classify_bytesets(
        bytesets: &super::super::classify::SharedClassifyBytesets,
    ) -> Option<Self> {
        let (
            finalizer_terminals_by_raw_state,
            future_finalizer_terminals_by_raw_state,
            finalizer_raw_states_by_terminal,
            future_finalizer_raw_states_by_terminal,
        ) = bytesets.ti_output_index()?;
        Some(Self {
            finalizer_terminals_by_raw_state,
            future_finalizer_terminals_by_raw_state,
            finalizer_raw_states_by_terminal,
            future_finalizer_raw_states_by_terminal,
            full_output_projection: OnceLock::new(),
        })
    }

    fn full_output_pair_hash(
        finalizers: &[TerminalID],
        future_finalizers: &[TerminalID],
    ) -> u64 {
        let mut hasher = FxHasher::default();
        finalizers.hash(&mut hasher);
        future_finalizers.hash(&mut hasher);
        hasher.finish()
    }

    fn full_output_projection(&self) -> (Arc<[OutputPair]>, Arc<[u32]>) {
        let (pairs, by_state) = self.full_output_projection.get_or_init(|| {
            let empty = OutputPair {
                finalizers: OutputBits::new(0),
                future_finalizers: OutputBits::new(0),
            };
            let mut pairs = vec![empty];
            let mut buckets = FxHashMap::<u64, SmallVec<[u32; 1]>>::default();
            buckets
                .entry(Self::full_output_pair_hash(&[], &[]))
                .or_default()
                .push(0);
            let mut by_state = Vec::with_capacity(self.finalizer_terminals_by_raw_state.len());
            for (finalizers, future_finalizers) in self
                .finalizer_terminals_by_raw_state
                .iter()
                .zip(self.future_finalizer_terminals_by_raw_state.iter())
            {
                let hash = Self::full_output_pair_hash(finalizers, future_finalizers);
                let existing = buckets.get(&hash).and_then(|candidates| {
                    candidates.iter().copied().find(|&id| {
                        let pair = &pairs[id as usize];
                        pair.finalizers.0.as_slice() == finalizers.as_ref()
                            && pair.future_finalizers.0.as_slice()
                                == future_finalizers.as_ref()
                    })
                });
                let id = existing.unwrap_or_else(|| {
                    let id = pairs.len() as u32;
                    pairs.push(OutputPair {
                        finalizers: OutputBits(finalizers.iter().copied().collect()),
                        future_finalizers: OutputBits(
                            future_finalizers.iter().copied().collect(),
                        ),
                    });
                    buckets.entry(hash).or_default().push(id);
                    id
                });
                by_state.push(id);
            }
            (Arc::from(pairs), Arc::from(by_state))
        });
        (Arc::clone(pairs), Arc::clone(by_state))
    }

    fn new(tokenizer: &Tokenizer) -> Self {
        let state_count = tokenizer.num_states() as usize;
        let terminal_count = tokenizer.num_terminals() as usize;
        let parallel_output_min_states = std::env::var(
            "GLRMASK_TI_PARALLEL_OUTPUT_INDEX_MIN_STATES",
        )
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(4096);
        if state_count >= parallel_output_min_states {
            if let Some(pool) = ti_certification_pool() {
                let chunk_count = pool.current_num_threads().min(state_count).max(1);
                let build_chunks = || {
                    (0..chunk_count)
                        .into_par_iter()
                        .map(|chunk| {
                            let start = state_count * chunk / chunk_count;
                            let end = state_count * (chunk + 1) / chunk_count;
                            let mut finalizer_rows = Vec::with_capacity(end - start);
                            let mut future_rows = Vec::with_capacity(end - start);
                            let mut finalizer_states =
                                vec![Vec::<u32>::new(); terminal_count];
                            let mut future_states = vec![Vec::<u32>::new(); terminal_count];
                            for raw_state in start as u32..end as u32 {
                                let finalizers = tokenizer
                                    .matched_terminals_iter(raw_state)
                                    .collect::<Vec<_>>();
                                for &terminal in &finalizers {
                                    finalizer_states[terminal as usize].push(raw_state);
                                }
                                finalizer_rows.push(finalizers.into_boxed_slice());

                                let future_finalizers = tokenizer
                                    .possible_future_terminals_iter(raw_state)
                                    .collect::<Vec<_>>();
                                for &terminal in &future_finalizers {
                                    future_states[terminal as usize].push(raw_state);
                                }
                                future_rows.push(future_finalizers.into_boxed_slice());
                            }
                            (
                                finalizer_rows,
                                future_rows,
                                finalizer_states,
                                future_states,
                            )
                        })
                        .collect::<Vec<_>>()
                };
                let chunks = if pool.current_thread_index().is_some() {
                    build_chunks()
                } else {
                    pool.install(build_chunks)
                };
                let merge_terminal_columns = || {
                    rayon::join(
                        || {
                            (0..terminal_count)
                                .into_par_iter()
                                .map(|terminal| {
                                    let len = chunks
                                        .iter()
                                        .map(|chunk| chunk.2[terminal].len())
                                        .sum();
                                    let mut states = Vec::with_capacity(len);
                                    for chunk in &chunks {
                                        states.extend_from_slice(&chunk.2[terminal]);
                                    }
                                    states
                                })
                                .collect::<Vec<_>>()
                        },
                        || {
                            (0..terminal_count)
                                .into_par_iter()
                                .map(|terminal| {
                                    let len = chunks
                                        .iter()
                                        .map(|chunk| chunk.3[terminal].len())
                                        .sum();
                                    let mut states = Vec::with_capacity(len);
                                    for chunk in &chunks {
                                        states.extend_from_slice(&chunk.3[terminal]);
                                    }
                                    states
                                })
                                .collect::<Vec<_>>()
                        },
                    )
                };
                let (finalizer_raw_states_by_terminal, future_finalizer_raw_states_by_terminal) =
                    if pool.current_thread_index().is_some() {
                        merge_terminal_columns()
                    } else {
                        pool.install(merge_terminal_columns)
                    };
                let mut finalizer_terminals_by_raw_state = Vec::with_capacity(state_count);
                let mut future_finalizer_terminals_by_raw_state =
                    Vec::with_capacity(state_count);
                for (finalizer_rows, future_rows, _, _) in chunks {
                    finalizer_terminals_by_raw_state.extend(finalizer_rows);
                    future_finalizer_terminals_by_raw_state.extend(future_rows);
                }
                return Self {
                    finalizer_terminals_by_raw_state: finalizer_terminals_by_raw_state.into(),
                    future_finalizer_terminals_by_raw_state:
                        future_finalizer_terminals_by_raw_state.into(),
                    finalizer_raw_states_by_terminal: finalizer_raw_states_by_terminal.into(),
                    future_finalizer_raw_states_by_terminal:
                        future_finalizer_raw_states_by_terminal.into(),
                    full_output_projection: OnceLock::new(),
                };
            }
        }
        let mut finalizer_terminals_by_raw_state = Vec::with_capacity(state_count);
        let mut future_finalizer_terminals_by_raw_state = Vec::with_capacity(state_count);
        let mut finalizer_raw_states_by_terminal = vec![Vec::<u32>::new(); terminal_count];
        let mut future_finalizer_raw_states_by_terminal = vec![Vec::<u32>::new(); terminal_count];
        for raw_state in 0..state_count as u32 {
            let finalizers = tokenizer
                .matched_terminals_iter(raw_state)
                .collect::<Vec<_>>();
            for &terminal in &finalizers {
                finalizer_raw_states_by_terminal[terminal as usize].push(raw_state);
            }
            finalizer_terminals_by_raw_state.push(finalizers.into_boxed_slice());

            let future_finalizers = tokenizer
                .possible_future_terminals_iter(raw_state)
                .collect::<Vec<_>>();
            for &terminal in &future_finalizers {
                future_finalizer_raw_states_by_terminal[terminal as usize].push(raw_state);
            }
            future_finalizer_terminals_by_raw_state.push(future_finalizers.into_boxed_slice());
        }
        Self {
            finalizer_terminals_by_raw_state: finalizer_terminals_by_raw_state.into(),
            future_finalizer_terminals_by_raw_state: future_finalizer_terminals_by_raw_state.into(),
            finalizer_raw_states_by_terminal: finalizer_raw_states_by_terminal.into(),
            future_finalizer_raw_states_by_terminal: future_finalizer_raw_states_by_terminal.into(),
            full_output_projection: OnceLock::new(),
        }
    }
}

/// Thread-safe compile-local cache shared by all vocabulary partitions.
#[derive(Default)]
pub struct SharedTiTokenizerOutputCache {
    data: OnceLock<Arc<TiTokenizerOutputData>>,
}

impl SharedTiTokenizerOutputCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_classify_bytesets(
        bytesets: &super::super::classify::SharedClassifyBytesets,
    ) -> Option<Self> {
        let data = TiTokenizerOutputData::from_classify_bytesets(bytesets)?;
        let cache = Self::new();
        assert!(
            cache.data.set(Arc::new(data)).is_ok(),
            "new TI output cache must be empty",
        );
        Some(cache)
    }

    fn get(&self, tokenizer: &Tokenizer) -> Arc<TiTokenizerOutputData> {
        Arc::clone(
            self.data
                .get_or_init(|| Arc::new(TiTokenizerOutputData::new(tokenizer))),
        )
    }
}

/// Immutable raw lexer evidence that is independent of a historical TI round's
/// active-terminal mask.  The projected `OutputBits` remain round-local, but
/// their source columns and the reverse restricted topology are reusable.
struct TiRawDiscoveryData {
    finalizer_terminals_by_state: Arc<[Box<[TerminalID]>]>,
    future_finalizer_terminals_by_state: Arc<[Box<[TerminalID]>]>,
    finalizer_states_by_terminal: Arc<[Vec<u32>]>,
    future_finalizer_states_by_terminal: Arc<[Vec<u32>]>,
    reverse_predecessors: Arc<[Vec<u32>]>,
    observed_destinations: Arc<[bool]>,
    full_output_projection_seed: OutputProjectionSeed,
}

impl TiRawDiscoveryData {
    fn new(
        tokenizer: &Tokenizer,
        topology: &RestrictedTopology,
        shared_output_cache: Option<&SharedTiTokenizerOutputCache>,
    ) -> Self {
        let topology_outputs = topology.nfa_output_rows.as_ref().map(|(finalizers, future)| {
            Arc::new(TiTokenizerOutputData::from_state_rows(
                Arc::clone(finalizers),
                Arc::clone(future),
                tokenizer.num_terminals() as usize,
            ))
        });
        let outputs = topology_outputs.unwrap_or_else(|| {
            shared_output_cache
                .map(|cache| cache.get(tokenizer))
                .unwrap_or_else(|| Arc::new(TiTokenizerOutputData::new(tokenizer)))
        });
        let outputs_are_discovery_domain = topology.nfa_output_rows.is_some();

        let (
            finalizer_terminals_by_state,
            future_finalizer_terminals_by_state,
            finalizer_states_by_terminal,
            future_finalizer_states_by_terminal,
        ) = if outputs_are_discovery_domain || topology.raw_representative_by_state.is_none() {
            (
                Arc::clone(&outputs.finalizer_terminals_by_raw_state),
                Arc::clone(&outputs.future_finalizer_terminals_by_raw_state),
                Arc::clone(&outputs.finalizer_raw_states_by_terminal),
                Arc::clone(&outputs.future_finalizer_raw_states_by_terminal),
            )
        } else {
            let terminal_count = tokenizer.num_terminals() as usize;
            let mut finalizers_by_state = Vec::with_capacity(topology.real_state_count);
            let mut future_by_state = Vec::with_capacity(topology.real_state_count);
            let mut finalizer_states = vec![Vec::<u32>::new(); terminal_count];
            let mut future_states = vec![Vec::<u32>::new(); terminal_count];
            for state in 0..topology.real_state_count {
                let raw_state = topology.raw_state_for_state(state) as usize;
                let finalizers = outputs.finalizer_terminals_by_raw_state[raw_state].clone();
                for &terminal in finalizers.iter() {
                    finalizer_states[terminal as usize].push(state as u32);
                }
                finalizers_by_state.push(finalizers);
                let future = outputs.future_finalizer_terminals_by_raw_state[raw_state].clone();
                for &terminal in future.iter() {
                    future_states[terminal as usize].push(state as u32);
                }
                future_by_state.push(future);
            }
            (
                Arc::from(finalizers_by_state),
                Arc::from(future_by_state),
                Arc::from(finalizer_states),
                Arc::from(future_states),
            )
        };

        let reverse_predecessors = topology.reverse_predecessors();
        let (full_output_pairs, full_output_pair_by_raw_state) =
            outputs.full_output_projection();
        let mut full_output_pair_by_state = if outputs_are_discovery_domain {
            assert_eq!(
                full_output_pair_by_raw_state.len(),
                topology.real_state_count,
                "NFA TI output rows must align with powerset states",
            );
            full_output_pair_by_raw_state.to_vec()
        } else {
            let mut by_state = Vec::with_capacity(topology.real_state_count);
            for state in 0..topology.real_state_count {
                let raw_state = topology.raw_state_for_state(state) as usize;
                by_state.push(full_output_pair_by_raw_state[raw_state]);
            }
            by_state
        };
        full_output_pair_by_state.push(0);
        let full_output_projection_seed = OutputProjectionSeed {
            active_terminals: Arc::from(vec![true; tokenizer.num_terminals() as usize]),
            output_pairs: full_output_pairs,
            output_pair_by_state: Arc::from(full_output_pair_by_state),
        };

        Self {
            finalizer_terminals_by_state,
            future_finalizer_terminals_by_state,
            finalizer_states_by_terminal,
            future_finalizer_states_by_terminal,
            reverse_predecessors,
            observed_destinations: topology.observed_destinations(),
            full_output_projection_seed,
        }
    }
}


/// Static per-L2P-partition TI data.  The restricted raw lexer topology and
/// root observation depend only on vocabulary bytes, not on the historical
/// active-terminal mask of an iterative TI round.
pub struct TiDiscoveryContext {
    topology: Arc<RestrictedTopology>,
    /// Exact byte alphabet supplied by the L2P vocabulary partition.  The
    /// topology may internally compress equivalent bytes, so its stored edge
    /// labels are not a lossless reconstruction of this semantic input.
    relevant_bytes: [bool; 256],
    raw: Arc<TiRawDiscoveryData>,
    root_output_signatures: Vec<RootOutputSignature>,
    root_observed_states: usize,
    /// Stable support partition from the latest round that needed one. Later
    /// rounds only remove active terminals, so they may solve exactly on this
    /// quotient rather than over every discovery-domain state again.
    support_partition_seed: std::cell::RefCell<Option<SupportPartitionSeed>>,
    output_projection_seed: std::cell::RefCell<Option<OutputProjectionSeed>>,
    /// Positive observed-output closure proofs retained across a monotone
    /// shrinking active mask. Projection commutes with a terminal transposition,
    /// so the image of a swap-closed output alphabet remains swap-closed.
    output_pair_closed_memo: std::cell::RefCell<FxHashSet<(TerminalID, TerminalID)>>,
    output_pair_closed_active: std::cell::RefCell<Option<Vec<bool>>>,
    /// Cross-round memo of the first-round necessary-condition filter.
    /// `canonical_round_one_still_possible(a, b)` depends only on the fixed DFA
    /// topology, canonical Moore rounds, and per-terminal finalizer states -
    /// none of which change between discovery rounds - and is symmetric in its
    /// two terminals (it tests the transposition `(a b)`).  The fixed-point
    /// loop re-tests a barely-shrinking candidate set each round, so caching
    /// the verdict under the unordered pair key eliminates the redundant
    /// recomputation across rounds.
    first_round_memo: std::cell::RefCell<FxHashMap<(TerminalID, TerminalID), bool>>,
}

/// Exact stable restricted-observation quotient produced from the final TI
/// discovery round.  The mask and byte alphabet are retained with the map so
/// callers can reuse the fixed-point certificate only when their downstream
/// observation semantics are identical.
pub struct TiRestrictedObservationSeed {
    pub state_map: ManyToOneIdMap,
    pub active_terminals: Arc<[bool]>,
    pub relevant_bytes: [bool; 256],
}

impl TiDiscoveryContext {
    /// Keep the cached output projection synchronized with the current
    /// monotone TI mask even when a discovery round can return before building
    /// an `InterchangeabilityDfa`.  This occurs at the fixed point after an
    /// earlier round merged terminals: the root/candidate prefilters may prove
    /// there are no remaining pairs, but downstream restricted-observation
    /// reuse still needs the exact output projection for the final active mask.
    ///
    /// Projection is exact because later TI masks only remove terminals.  We
    /// deliberately do nothing when no prior seed exists, avoiding new work on
    /// partitions whose first round already has no candidates.
    fn refresh_output_projection_seed_if_present(&self, active_terminals: &[bool]) {
        let needs_refresh = self
            .output_projection_seed
            .borrow()
            .as_ref()
            .is_some_and(|seed| seed.active_terminals.as_ref() != active_terminals);
        if !needs_refresh {
            return;
        }

        let dfa = InterchangeabilityDfa::from_context(active_terminals, self);
        *self.output_projection_seed.borrow_mut() = Some(OutputProjectionSeed {
            active_terminals: Arc::from(active_terminals.to_vec()),
            output_pairs: Arc::clone(&dfa.output_pairs),
            output_pair_by_state: Arc::clone(&dfa.output_pair_by_state),
        });
    }

    fn from_topology(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        topology: RestrictedTopology,
        shared_output_cache: Option<&SharedTiTokenizerOutputCache>,
        profile_label: &'static str,
        started_at: Instant,
        topology_finished_at: Instant,
    ) -> Self {
        let profile = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let topology = Arc::new(topology);
        let raw = Arc::new(TiRawDiscoveryData::new(
            tokenizer,
            &topology,
            shared_output_cache,
        ));
        let raw_finished_at = Instant::now();
        let (root_output_signatures, root_observed_states) =
            root_output_signatures(&topology, &raw);
        let finished_at = Instant::now();
        if profile {
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] {} discovery_states={} sparse_edges={} selected_bytes={} topology_ms={:.3} raw_ms={:.3} root_sig_ms={:.3} total_ms={:.3}",
                profile_label,
                topology.real_state_count,
                topology.edges.len(),
                topology.bytes.len(),
                (topology_finished_at - started_at).as_secs_f64() * 1000.0,
                (raw_finished_at - topology_finished_at).as_secs_f64() * 1000.0,
                (finished_at - raw_finished_at).as_secs_f64() * 1000.0,
                (finished_at - started_at).as_secs_f64() * 1000.0,
            );
        }
        Self {
            topology,
            relevant_bytes: *relevant_bytes,
            raw,
            root_output_signatures,
            root_observed_states,
            support_partition_seed: std::cell::RefCell::new(None),
            output_projection_seed: std::cell::RefCell::new(None),
            output_pair_closed_memo: std::cell::RefCell::new(FxHashSet::default()),
            output_pair_closed_active: std::cell::RefCell::new(None),
            first_round_memo: std::cell::RefCell::new(FxHashMap::default()),
        }
    }

    pub fn new(tokenizer: &Tokenizer, relevant_bytes: &[bool; 256]) -> Self {
        Self::new_with_output_cache(tokenizer, relevant_bytes, None)
    }

    pub fn new_with_output_cache(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        shared_output_cache: Option<&SharedTiTokenizerOutputCache>,
    ) -> Self {
        let started_at = Instant::now();
        let topology = RestrictedTopology::new(tokenizer, relevant_bytes);
        let topology_finished_at = Instant::now();
        Self::from_topology(
            tokenizer,
            relevant_bytes,
            topology,
            shared_output_cache,
            "ti_context_build",
            started_at,
            topology_finished_at,
        )
    }

    pub fn try_new_with_output_cache(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        shared_output_cache: Option<&SharedTiTokenizerOutputCache>,
    ) -> Result<Self, RelevantPowersetWork> {
        let profile = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let budget = ti_nfa_powerset_budget(tokenizer, relevant_bytes, None);
        let t0 = Instant::now();
        let topology = match RestrictedTopology::build_budgeted(
            tokenizer,
            relevant_bytes,
            None,
            budget,
        ) {
            Ok(topology) => topology,
            Err(work) => {
                if profile {
                    eprintln!(
                        "[glrmask/profile][terminal_interchangeability] ti_context_build_aborted configurations={} edges={} max_configurations={} max_edges={} total_ms={:.3}",
                        work.configurations,
                        work.edges,
                        budget.max_configurations,
                        budget.max_edges,
                        t0.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                return Err(work);
            }
        };
        let topology_finished_at = Instant::now();
        Ok(Self::from_topology(
            tokenizer,
            relevant_bytes,
            topology,
            shared_output_cache,
            "ti_context_build",
            t0,
            topology_finished_at,
        ))
    }

    /// Build immutable discovery evidence over a total global scanner-state
    /// quotient. This is intentionally separate from an `initial_state_map`
    /// used to seed later id-map analysis.
    pub fn new_with_global_state_quotient(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        global_state_quotient: &GlobalScannerStateQuotient,
    ) -> Self {
        Self::new_with_global_state_quotient_and_output_cache(
            tokenizer,
            relevant_bytes,
            global_state_quotient,
            None,
        )
    }

    pub fn new_with_global_state_quotient_and_output_cache(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        global_state_quotient: &GlobalScannerStateQuotient,
        shared_output_cache: Option<&SharedTiTokenizerOutputCache>,
    ) -> Self {
        let started_at = Instant::now();
        let topology = RestrictedTopology::new_with_global_state_quotient(
            tokenizer,
            relevant_bytes,
            global_state_quotient,
        );
        let topology_finished_at = Instant::now();
        Self::from_topology(
            tokenizer,
            relevant_bytes,
            topology,
            shared_output_cache,
            "ti_context_build_c",
            started_at,
            topology_finished_at,
        )
    }

    pub fn try_new_with_global_state_quotient_and_output_cache(
        tokenizer: &Tokenizer,
        relevant_bytes: &[bool; 256],
        global_state_quotient: &GlobalScannerStateQuotient,
        shared_output_cache: Option<&SharedTiTokenizerOutputCache>,
    ) -> Result<Self, RelevantPowersetWork> {
        let profile = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let budget = ti_nfa_powerset_budget(
            tokenizer,
            relevant_bytes,
            Some(global_state_quotient),
        );
        let t0 = Instant::now();
        let topology = match RestrictedTopology::build_budgeted(
            tokenizer,
            relevant_bytes,
            Some(global_state_quotient),
            budget,
        ) {
            Ok(topology) => topology,
            Err(work) => {
                if profile {
                    eprintln!(
                        "[glrmask/profile][terminal_interchangeability] ti_context_build_c_aborted configurations={} edges={} max_configurations={} max_edges={} total_ms={:.3}",
                        work.configurations,
                        work.edges,
                        budget.max_configurations,
                        budget.max_edges,
                        t0.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                return Err(work);
            }
        };
        let topology_finished_at = Instant::now();
        Ok(Self::from_topology(
            tokenizer,
            relevant_bytes,
            topology,
            shared_output_cache,
            "ti_context_build_c",
            t0,
            topology_finished_at,
        ))
    }

    /// Exact finalizer/future observation IDs from the most recent discovery
    /// round, restricted and densified over real raw scanner states.
    pub fn final_raw_observation_ids(
        &self,
        raw_state_count: usize,
    ) -> Option<(Vec<u32>, Vec<u32>)> {
        if self.topology.raw_representative_by_state.is_some() {
            return None;
        }
        let seed = self.output_projection_seed.borrow();
        let by_state = &seed.as_ref()?.output_pair_by_state;
        if by_state.len() < raw_state_count {
            return None;
        }
        let mut dense_by_pair = FxHashMap::<u32, u32>::default();
        let mut ids = Vec::with_capacity(raw_state_count);
        let mut representatives = Vec::<u32>::new();
        for (state, &pair) in by_state[..raw_state_count].iter().enumerate() {
            let id = if let Some(&id) = dense_by_pair.get(&pair) {
                id
            } else {
                let id = dense_by_pair.len() as u32;
                dense_by_pair.insert(pair, id);
                representatives.push(state as u32);
                id
            };
            ids.push(id);
        }
        Some((ids, representatives))
    }

    pub fn reusable_nfa_restricted_observation_state_map(
        &self,
        tokenizer: &Tokenizer,
        initial_state_map: Option<&ManyToOneIdMap>,
    ) -> Option<TiRestrictedObservationSeed> {
        let seed = self.output_projection_seed.borrow();
        let seed = seed.as_ref()?;
        self.reusable_nfa_observation_state_map_from_seed(tokenizer, initial_state_map, seed)
    }

    /// Reuse the discovery topology but certify every original output label.
    /// A composed grammar may require global residual observations even when
    /// TI's final discovery round used only representative terminal labels.
    /// The full output seed is immutable and was retained before any folding.
    pub fn reusable_nfa_full_observation_state_map(
        &self,
        tokenizer: &Tokenizer,
        initial_state_map: Option<&ManyToOneIdMap>,
    ) -> Option<TiRestrictedObservationSeed> {
        self.reusable_nfa_observation_state_map_from_seed(
            tokenizer, initial_state_map, &self.raw.full_output_projection_seed,
        )
    }

    fn reusable_nfa_observation_state_map_from_seed(
        &self,
        tokenizer: &Tokenizer,
        initial_state_map: Option<&ManyToOneIdMap>,
        seed: &OutputProjectionSeed,
    ) -> Option<TiRestrictedObservationSeed> {
        if !tokenizer.has_epsilon_transitions() || !self.topology.nfa_configurations_use_raw_states {
            return None;
        }
        let configurations = self.topology.nfa_configurations.as_ref()?;
        let active_language = raw_active_language_states(
            tokenizer,
            Some(seed.active_terminals.as_ref()),
        )?;
        let state_map = compute_state_map_from_prebuilt_sparse_powerset(
            tokenizer,
            initial_state_map,
            RefinementDepth::Stable,
            self.topology.state_for_raw.as_ref()?.as_ref(),
            configurations.as_ref(),
            seed.output_pair_by_state.as_ref(),
            Some(&active_language),
            &self.topology.edge_offsets[..self.topology.real_state_count + 1],
            &self.topology.edges,
        );
        Some(TiRestrictedObservationSeed {
            state_map,
            active_terminals: Arc::clone(&seed.active_terminals),
            relevant_bytes: self.relevant_bytes,
        })
    }


}

/// Partition candidate terminals by a necessary-and-sufficient condition for
/// the root part of the pair characterization.
///
/// Let `R` be the states reachable from the lexer initial state using enabled
/// bytes, and let `D = δ(R, bytes)` be the observed destination states. The
/// identity and swapped sides start at the same root and have the same byte
/// transition function, so induction on enabled-byte words makes each state in
/// `D` compare with itself. At such a state, swapping `left` and `right` leaves
/// either frozen output set unchanged exactly when the two terminals have equal
/// membership in that set. Thus the root hashes can agree at every refinement
/// depth exactly when both terminals have equal finalizer and frozen-future
/// membership over every state in `D`.
fn root_output_signatures(
    topology: &RestrictedTopology,
    raw: &TiRawDiscoveryData,
) -> (Vec<RootOutputSignature>, usize) {
    let state_count = topology.real_state_count;
    let mut reachable = vec![false; state_count];
    reachable[topology.initial_state] = true;
    let mut worklist = vec![topology.initial_state];
    while let Some(state) = worklist.pop() {
        for &(_, destination) in topology.edges_from(state) {
            let destination = destination as usize;
            if !reachable[destination] {
                reachable[destination] = true;
                worklist.push(destination);
            }
        }
    }

    // `characterize` observes outputs on destinations, not current states.
    let mut observed = vec![false; state_count];
    for (state, &is_reachable) in reachable.iter().enumerate() {
        if is_reachable {
            for &(_, destination) in topology.edges_from(state) {
                observed[destination as usize] = true;
            }
        }
    }

    let terminal_count = raw.finalizer_states_by_terminal.len();
    let mut finalizer_states = vec![Vec::<u32>::new(); terminal_count];
    let mut future_finalizer_states = vec![Vec::<u32>::new(); terminal_count];
    for (state, &is_observed) in observed.iter().enumerate() {
        if !is_observed {
            continue;
        }
        for &terminal in raw.finalizer_terminals_by_state[state].iter() {
            finalizer_states[terminal as usize].push(state as u32);
        }
        for &terminal in raw.future_finalizer_terminals_by_state[state].iter() {
            future_finalizer_states[terminal as usize].push(state as u32);
        }
    }

    let signatures = finalizer_states
        .into_iter()
        .zip(future_finalizer_states)
        .map(|(finalizer_states, future_finalizer_states)| RootOutputSignature {
            finalizer_states: finalizer_states.into_boxed_slice(),
            future_finalizer_states: future_finalizer_states.into_boxed_slice(),
        })
        .collect::<Vec<_>>();
    (
        signatures,
        observed.into_iter().filter(|&value| value).count(),
    )
}

fn rooted_candidate_groups_from_signatures(
    candidates: &[TerminalID],
    signatures: &[RootOutputSignature],
) -> Vec<Vec<TerminalID>> {
    let mut groups = BTreeMap::<&RootOutputSignature, Vec<TerminalID>>::new();
    for &terminal in candidates {
        groups
            .entry(
                signatures
                    .get(terminal as usize)
                    .expect("TI candidate terminal must have a root signature"),
            )
            .or_default()
            .push(terminal);
    }
    groups.into_values().collect()
}

fn rooted_candidate_groups(
    tokenizer: &Tokenizer,
    candidates: &[TerminalID],
    topology: &RestrictedTopology,
) -> (Vec<Vec<TerminalID>>, usize) {
    let raw = TiRawDiscoveryData::new(tokenizer, topology, None);
    let (signatures, observed_states) = root_output_signatures(topology, &raw);
    (
        rooted_candidate_groups_from_signatures(candidates, &signatures),
        observed_states,
    )
}

/// A terminal's support across a terminal-name-independent structural
/// partition. It is only a rejection invariant; the full checker decides.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct StructuralOutputSignature {
    finalizer_support: Box<[u64]>,
    future_finalizer_support: Box<[u64]>,
}

/// Counts of quotient source classes whose successor observations mention a
/// terminal in finalizers only, future-finalizers only, or both. An accepted
/// terminal transposition maps these source classes bijectively, preserving
/// each category; unequal shapes are therefore an exact rejection invariant.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
struct SupportTrackShape {
    finalizer_only: u32,
    future_only: u32,
    both: u32,
}

const STRUCTURAL_REFINEMENT_ROUNDS: usize = 2;

/// Mix one invariant structural component into a deterministic 64-bit
/// fingerprint. Equal tuples always have equal fingerprints. A collision only
/// coarsens the prefilter and therefore cannot reject a valid pair.
#[inline]
fn mix_structural_fingerprint(mut state: u64, component: u64) -> u64 {
    state ^= component.wrapping_add(0x9e37_79b9_7f4a_7c15).rotate_left(17);
    state = state.wrapping_mul(0x517c_c1b7_2722_0a95);
    state ^ (state >> 29)
}

/// Compute structural support signatures using a sparse canonical form of the
/// reference tuple. At a given round, every missing byte has the common dead
/// component `(hash(dead), 0, 0)`. The full tuple is therefore determined by
/// the enabled-byte keys whose component differs from that default. Omitting
/// default entries preserves tuple equality exactly.
fn structural_candidate_signatures(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    candidates: &[TerminalID],
    topology: &RestrictedTopology,
    refinement_rounds: usize,
) -> (Vec<StructuralOutputSignature>, usize) {
    let state_count = topology.state_count();
    let dead_state = topology.dead_state();
    let mut finalizer_counts = vec![0u64; state_count];
    let mut future_finalizer_counts = vec![0u64; state_count];
    for state in 0..topology.real_state_count {
        let raw_state = topology.raw_state_for_state(state);
        finalizer_counts[state] = tokenizer
            .matched_terminals_iter(raw_state)
            .filter(|&terminal| active_terminals[terminal as usize])
            .count() as u64;
        future_finalizer_counts[state] = tokenizer
            .possible_future_terminals_iter(raw_state)
            .filter(|&terminal| active_terminals[terminal as usize])
            .count() as u64;
    }

    let mut fingerprints = vec![0x6a09_e667_f3bc_c909; state_count];
    for _ in 0..refinement_rounds {
        let default_fingerprint = fingerprints[dead_state];
        let mut next = Vec::with_capacity(state_count);
        for state in 0..state_count {
            let mut fingerprint = mix_structural_fingerprint(
                0xbb67_ae85_84ca_a73b,
                topology.bytes.len() as u64,
            );
            for &(byte, destination) in topology.edges_from(state) {
                let destination = destination as usize;
                if fingerprints[destination] == default_fingerprint
                    && finalizer_counts[destination] == 0
                    && future_finalizer_counts[destination] == 0
                {
                    continue;
                }
                fingerprint = mix_structural_fingerprint(fingerprint, byte as u64);
                fingerprint = mix_structural_fingerprint(fingerprint, fingerprints[destination]);
                fingerprint = mix_structural_fingerprint(fingerprint, finalizer_counts[destination]);
                fingerprint = mix_structural_fingerprint(
                    fingerprint,
                    future_finalizer_counts[destination],
                );
            }
            next.push(fingerprint);
        }
        fingerprints = next;
    }

    let mut color_ids = FxHashMap::<u64, u32>::default();
    let mut colors = Vec::with_capacity(state_count);
    for fingerprint in fingerprints {
        let next = color_ids.len() as u32;
        colors.push(*color_ids.entry(fingerprint).or_insert(next));
    }
    let color_count = color_ids.len();
    let words = color_count.div_ceil(64);

    let mut candidate_index_by_terminal = vec![usize::MAX; active_terminals.len()];
    for (candidate_index, &terminal) in candidates.iter().enumerate() {
        candidate_index_by_terminal[terminal as usize] = candidate_index;
    }
    let mut finalizer_support = vec![vec![0u64; words]; candidates.len()];
    let mut future_finalizer_support = vec![vec![0u64; words]; candidates.len()];
    for (state, &is_observed) in topology.observed_destinations().iter().enumerate() {
        if !is_observed || state == dead_state {
            continue;
        }
        let color = colors[state] as usize;
        let word = color / 64;
        let mask = 1u64 << (color % 64);
        let raw_state = topology.raw_state_for_state(state);
        for terminal in tokenizer.matched_terminals_iter(raw_state) {
            let candidate_index = candidate_index_by_terminal[terminal as usize];
            if candidate_index != usize::MAX {
                finalizer_support[candidate_index][word] |= mask;
            }
        }
        for terminal in tokenizer.possible_future_terminals_iter(raw_state) {
            let candidate_index = candidate_index_by_terminal[terminal as usize];
            if candidate_index != usize::MAX {
                future_finalizer_support[candidate_index][word] |= mask;
            }
        }
    }

    (
        finalizer_support
            .into_iter()
            .zip(future_finalizer_support)
            .map(|(finalizer_support, future_finalizer_support)| StructuralOutputSignature {
                finalizer_support: finalizer_support.into_boxed_slice(),
                future_finalizer_support: future_finalizer_support.into_boxed_slice(),
            })
            .collect(),
        color_count,
    )
}

/// Refine root candidates by the global structural invariant. Singletons need
/// no direct terminal-pair check.
fn refine_candidate_groups_by_structure(
    root_groups: Vec<Vec<TerminalID>>,
    candidates: &[TerminalID],
    signatures: &[StructuralOutputSignature],
) -> Vec<Vec<TerminalID>> {
    let terminal_count = candidates
        .iter()
        .copied()
        .max()
        .map_or(0, |terminal| terminal as usize + 1);
    let mut candidate_index_by_terminal = vec![usize::MAX; terminal_count];
    for (candidate_index, &terminal) in candidates.iter().enumerate() {
        candidate_index_by_terminal[terminal as usize] = candidate_index;
    }

    let mut groups = Vec::new();
    for root_group in root_groups {
        let mut by_signature = BTreeMap::<StructuralOutputSignature, Vec<TerminalID>>::new();
        for terminal in root_group {
            let candidate_index = candidate_index_by_terminal[terminal as usize];
            debug_assert_ne!(candidate_index, usize::MAX);
            by_signature
                .entry(signatures[candidate_index].clone())
                .or_default()
                .push(terminal);
        }
        groups.extend(by_signature.into_values().filter(|group| group.len() >= 2));
    }
    groups
}

/// Refine candidates by an exact necessary invariant of the observed frozen
/// output-pair relation.  A valid transposition permutes that relation, hence
/// it must preserve how many distinct observed pairs mention each terminal in
/// finalizer-only, future-finalizer-only, and both positions.  This is only a
/// rejection filter; all surviving pairs still pass the full exact checker.
fn refine_candidate_groups_by_observed_output_pair_shape(
    groups: Vec<Vec<TerminalID>>,
    shapes: &[SupportTrackShape],
) -> Vec<Vec<TerminalID>> {
    let mut refined = Vec::new();
    for group in groups {
        let mut by_shape = BTreeMap::<SupportTrackShape, Vec<TerminalID>>::new();
        for terminal in group {
            let shape = shapes
                .get(terminal as usize)
                .copied()
                .expect("TI candidate terminal must have an observed-output support shape");
            by_shape.entry(shape).or_default().push(terminal);
        }
        refined.extend(by_shape.into_values().filter(|group| group.len() >= 2));
    }
    refined
}

/// Sound terminal-side color refinement of the set of observed frozen output
/// pairs.  Treat each pair `(finalizers, future_finalizers)` as a directed
/// hyperedge, with a three-way incidence tag for final-only, future-only, and
/// both.  A valid terminal transposition maps this set of hyperedges to itself,
/// so it preserves every refinement color.  Different resulting colors can
/// therefore reject a pair without affecting exact TI semantics.
///
/// Terminals already excluded from a candidate group are fixed under every
/// remaining binary swap.  Giving them individual colors strengthens the
/// invariant: their raw labels are not accidentally treated as exchangeable
/// context.  The compact commutative fingerprints may collide only by merging
/// colors, which is conservative.
fn refine_candidate_groups_by_observed_output_hypergraph(
    groups: Vec<Vec<TerminalID>>,
    observed_pairs: &[OutputPair],
    terminal_count: usize,
) -> (Vec<Vec<TerminalID>>, usize) {
    if groups.is_empty() {
        return (groups, 0);
    }

    let mut candidate_group_by_terminal = vec![usize::MAX; terminal_count];
    for (group_index, group) in groups.iter().enumerate() {
        for &terminal in group {
            candidate_group_by_terminal[terminal as usize] = group_index;
        }
    }

    // Fixed terminals receive raw-id colors. Candidate terminals receive their
    // already sound prefilter group color, ensuring an accepted swap starts in
    // one common color class.
    let mut colors = (0..terminal_count as u32).collect::<Vec<_>>();
    for (group_index, group) in groups.iter().enumerate() {
        let color = terminal_count as u32 + group_index as u32;
        for &terminal in group {
            colors[terminal as usize] = color;
        }
    }

    let mut rounds = 0usize;
    loop {
        rounds += 1;
        let mut sum = vec![0u64; terminal_count];
        let mut xor = vec![0u64; terminal_count];
        let mut count = vec![0u32; terminal_count];

        for pair in observed_pairs {
            let color_multiset_fingerprint = |terminals: &OutputBits| {
                let mut total = 0u64;
                let mut parity = 0u64;
                for &terminal in &terminals.0 {
                    let component = mix_structural_fingerprint(
                        0x6d2b_79f5_aa99_5a71,
                        colors[terminal as usize] as u64,
                    );
                    total = total.wrapping_add(component);
                    parity ^= component.rotate_left((colors[terminal as usize] & 63) as u32);
                }
                let mut fingerprint = mix_structural_fingerprint(
                    0xa076_1d64_78bd_642f,
                    terminals.0.len() as u64,
                );
                fingerprint = mix_structural_fingerprint(fingerprint, total);
                mix_structural_fingerprint(fingerprint, parity)
            };
            let finalizer_fingerprint = color_multiset_fingerprint(&pair.finalizers);
            let future_fingerprint = color_multiset_fingerprint(&pair.future_finalizers);
            let mut edge_fingerprint = mix_structural_fingerprint(
                0xe703_7ed1_a0b4_28db,
                finalizer_fingerprint,
            );
            edge_fingerprint = mix_structural_fingerprint(edge_fingerprint, future_fingerprint);

            let mut finalizer_index = 0usize;
            let mut future_index = 0usize;
            while finalizer_index < pair.finalizers.0.len()
                || future_index < pair.future_finalizers.0.len()
            {
                let (terminal, category) = match (
                    pair.finalizers.0.get(finalizer_index),
                    pair.future_finalizers.0.get(future_index),
                ) {
                    (Some(&finalizer), Some(&future)) if finalizer == future => {
                        finalizer_index += 1;
                        future_index += 1;
                        (finalizer as usize, 2u64)
                    }
                    (Some(&finalizer), Some(&future)) if finalizer < future => {
                        finalizer_index += 1;
                        (finalizer as usize, 0u64)
                    }
                    (Some(_), Some(&future)) => {
                        future_index += 1;
                        (future as usize, 1u64)
                    }
                    (Some(&finalizer), None) => {
                        finalizer_index += 1;
                        (finalizer as usize, 0u64)
                    }
                    (None, Some(&future)) => {
                        future_index += 1;
                        (future as usize, 1u64)
                    }
                    (None, None) => unreachable!("nonempty observed-output hyperedge merge"),
                };
                let component = mix_structural_fingerprint(edge_fingerprint, category);
                sum[terminal] = sum[terminal].wrapping_add(component);
                xor[terminal] ^= component.rotate_left((category * 17) as u32);
                count[terminal] += 1;
            }
        }

        let mut class_for_signature = FxHashMap::<u64, u32>::default();
        let mut next_colors = colors.clone();
        for (terminal, &group_index) in candidate_group_by_terminal.iter().enumerate() {
            if group_index == usize::MAX {
                continue;
            }
            let mut fingerprint = mix_structural_fingerprint(
                0x8ebc_6af0_9c88_c6e3,
                colors[terminal] as u64,
            );
            fingerprint = mix_structural_fingerprint(fingerprint, sum[terminal]);
            fingerprint = mix_structural_fingerprint(fingerprint, xor[terminal]);
            fingerprint = mix_structural_fingerprint(fingerprint, count[terminal] as u64);
            let next = terminal_count as u32 + class_for_signature.len() as u32;
            next_colors[terminal] = *class_for_signature.entry(fingerprint).or_insert(next);
        }
        let stable = same_equality_partition_u32(&colors, &next_colors);
        colors = next_colors;
        if stable || rounds == terminal_count {
            break;
        }
    }

    let mut refined = Vec::new();
    for group in groups {
        let mut by_color = BTreeMap::<u32, Vec<TerminalID>>::new();
        for terminal in group {
            by_color
                .entry(colors[terminal as usize])
                .or_default()
                .push(terminal);
        }
        refined.extend(by_color.into_values().filter(|group| group.len() >= 2));
    }
    (refined, rounds)
}

/// The selected raw tokenizer-state representative for every source state
/// under one terminal swap. The exact characterization establishes a mapped
/// target class; all downstream consumers have always selected only that
/// class's first raw state, so retaining the whole class is redundant.
#[derive(Clone, Debug)]
struct InterchangeMap {
    scanner_state_map: TransportScannerStateMap,
}


/// One transient exact TI round.  This is deliberately build-local: the only
/// durable TI result remains the final flat terminal partition.  Keeping the
/// already-certified witnesses avoids re-running terminal discovery after the
/// representative core has been built.
#[derive(Clone, Debug)]
pub struct TiRoundTransportWitnesses {
    active_before_round: Vec<bool>,
    pub partition: BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    /// Certified maps for precisely the accepted representative/member pairs
    /// of this historical round.  The key is `(new_representative,
    /// old_representative)`.
    maps: BTreeMap<(TerminalID, TerminalID), Arc<TransportScannerStateMap>>,
}

impl TiRoundTransportWitnesses {
    fn singleton(active_terminals: &[bool]) -> Self {
        Self {
            active_before_round: active_terminals.to_vec(),
            partition: singleton_partition(active_terminals),
            maps: BTreeMap::new(),
        }
    }
}

impl PartialEq for InterchangeMap {
    fn eq(&self, other: &Self) -> bool {
        self.scanner_state_map.len() == other.scanner_state_map.len()
            && (0..self.scanner_state_map.len()).all(|state| {
                self.scanner_state_map.scanner_state(state as u32)
                    == other.scanner_state_map.scanner_state(state as u32)
            })
    }
}

impl Eq for InterchangeMap {}

impl InterchangeMap {
    fn scanner_state_map(&self) -> TransportScannerStateMap {
        self.scanner_state_map.clone()
    }

    #[cfg(test)]
    fn materialized_scanner_states(&self) -> Arc<[u32]> {
        self.scanner_state_map.materialized()
    }
}

struct PairCharacterization {
    identity_hashes: Vec<CharacterizationHash>,
    swapped_hashes: Vec<CharacterizationHash>,
}

struct InterchangeabilityDfa {
    topology: Arc<RestrictedTopology>,
    observed_output_pairs: Vec<OutputPair>,
    /// Canonical all-output-pair id for each entry in `observed_output_pairs`.
    observed_output_pair_canonical_ids: Vec<u32>,
    /// Membership of each canonical all-output-pair id in the enabled-byte
    /// destination observation set.
    observed_output_pair_is_observed: Vec<bool>,
    observed_output_pair_ids_by_terminal: Vec<Vec<u32>>,
    /// For each active terminal, its exact membership counts across the
    /// deduplicated observed frozen-output pairs. A valid terminal swap maps
    /// those pairs bijectively, preserving the final-only, future-only, and
    /// both categories. This is therefore an inexpensive rejection invariant
    /// for candidate pairs before the per-pair closure check.
    observed_output_pair_support_shapes_by_terminal: Vec<SupportTrackShape>,
    observed_output_pair_marks: Vec<u32>,
    observed_output_pair_mark_epoch: u32,
    /// Exact terminal columns over the frozen state outputs. When two columns
    /// coincide in both output families, swapping their terminal labels leaves
    /// every destination output unchanged and therefore needs no refinement.
    finalizer_states_by_terminal: Arc<[Vec<u32>]>,
    future_finalizer_states_by_terminal: Arc<[Vec<u32>]>,
    identity_rounds: Vec<Vec<CharacterizationHash>>,
    /// Canonical, collision-free characterization cache used by the hot path.
    /// It describes the same raw restricted topology as `identity_rounds` and
    /// never merges or rewrites tokenizer states.
    output_pairs: Arc<[OutputPair]>,
    output_pair_lookup: FxHashMap<OutputPair, u32>,
    output_pair_by_state: Arc<[u32]>,
    /// Reverse enabled-byte edges, used only by the exact first-round
    /// rejection prefilter. Each changed frozen output can affect only these
    /// source rows in the characterization tuple.
    reverse_predecessors: Arc<[Vec<u32>]>,
    canonical_rounds: Vec<CanonicalRound>,
    canonical_round_one_class_counts: Option<Vec<u32>>,
    canonical_round_one_source_marks: Vec<u32>,
    canonical_round_one_source_mark_epoch: u32,
    canonical_round_one_affected_sources: Vec<u32>,
    /// Cache of the left/representative-side affected sources, reused across
    /// the members sharing a fixed representative in the exact-verify loop.
    canonical_round_one_left_terminal: Option<TerminalID>,
    canonical_round_one_left_sources: Vec<u32>,
    canonical_round_one_changed_scratch: SmallVec<[(u32, u32); 16]>,
    canonical_round_one_added_scratch: SmallVec<[u32; 16]>,
    support_partition_seed: Option<SupportPartitionSeed>,
    support_quotient: Option<Arc<SupportQuotient>>,
    canonical_quotient: Option<CanonicalQuotient>,
    /// Per raw terminal, the canonical quotient classes whose representative
    /// frozen output mentions that terminal.  This is build-local discovery
    /// scratch, used to propose a tiny support-transposition witness before
    /// falling back to exact refinement.
    terminal_quotient_output_supports: Option<Vec<Option<Vec<(u32, u8)>>>>,
    support_cone_marks: Vec<u32>,
    support_cone_mark_epoch: u32,
    /// Thread-safe on-demand mirror of `terminal_quotient_output_supports` used
    /// only by the parallel certification path. Each terminal's support is
    /// pure given the (pre-built) global support quotient and immutable
    /// topology, so workers build exactly the entries they touch — the same
    /// set the sequential path would build lazily — with no `&mut self`.
    parallel_terminal_supports: OnceLock<Vec<OnceLock<Vec<(u32, u8)>>>>,
    quotient_certified: usize,
    support_transposition_certified: usize,
    support_transposition_no_template: usize,
    support_transposition_outside_cone: usize,
    support_transposition_root_rejected: usize,
    support_transposition_incoming_rejected: usize,
    support_transposition_signature_rejected: usize,
    support_transposition_support_setup_ns: u64,
    support_quotient_build_ns: u64,
    support_transposition_template_ns: u64,
    support_transposition_cone_ns: u64,
    support_transposition_verify_ns: u64,
    sparse_quotient_certified: usize,
    sparse_quotient_cone_classes_total: usize,
    sparse_quotient_cone_classes_max: usize,
    sparse_quotient_cone_ns: u64,
    sparse_quotient_refinement_ns: u64,
    sparse_quotient_map_ns: u64,
    canonical_stable_round: Option<usize>,
    canonical_identity_map: Option<InterchangeMap>,
    /// Per-state canonical identity signature hash as of the last built round.
    /// Reused across rounds so an incremental refinement step only rehashes
    /// states whose successor classes changed. Empty until the first round.
    canonical_identity_cached_hashes: Vec<u64>,
    signature_capacity: usize,
}

/// Per-worker scratch for the parallel per-pair certification path. The heavy
/// `InterchangeabilityDfa` state is immutable and lazily pre-built before the
/// certification loop, so each rayon worker only needs its own epoch-marked
/// scratch buffers plus counter accumulators. Allocated once per worker thread
/// via `map_init` and reused across the tasks that land on that worker.
struct CertScratch {
    source_marks: Vec<u32>,
    source_mark_epoch: u32,
    affected_sources: Vec<u32>,
    changed_scratch: SmallVec<[(u32, u32); 16]>,
    added_scratch: SmallVec<[u32; 16]>,
    output_pair_marks: Vec<u32>,
    output_pair_mark_epoch: u32,
}

impl CertScratch {
    fn new(dfa: &InterchangeabilityDfa) -> Self {
        CertScratch {
            source_marks: vec![0; dfa.canonical_round_one_source_marks.len()],
            source_mark_epoch: 0,
            affected_sources: Vec::new(),
            changed_scratch: SmallVec::new(),
            added_scratch: SmallVec::new(),
            output_pair_marks: vec![0; dfa.observed_output_pairs.len()],
            output_pair_mark_epoch: 0,
        }
    }
}

/// Outcome of certifying one member terminal against a fixed representative in
/// the parallel path. `NeedsExact` defers the rare exact fallback to the
/// sequential apply phase (direct_exact_checks is ~0 on the hot partitions).
enum MemberVerdict {
    RejectOutputPair,
    RejectFirstRound,
    AcceptIdentity(InterchangeMap),
    AcceptSupport(InterchangeMap),
    NeedsExact,
}

/// Per-member certification result carried back from the parallel workers.
/// Counter aggregation and the first-round memo merge happen sequentially in
/// the apply phase so the workers stay purely `&self`.
struct MemberResult {
    verdict: MemberVerdict,
    memo_update: Option<((TerminalID, TerminalID), bool)>,
    closed_update: Option<(TerminalID, TerminalID)>,
    used_cached_closed: bool,
}

/// Dedicated rayon thread pool for the per-pair TI certification loop. Built
/// once from `GLRMASK_TI_PARALLEL_THREADS`; the rest of the compiler can run
/// under `RAYON_NUM_THREADS=1` while this hot loop still fans out across cores.
fn ti_certification_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    let pool = POOL
        .get_or_init(|| {
            let threads = std::env::var("GLRMASK_TI_PARALLEL_THREADS")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(1);
            if threads <= 1 {
                return None;
            }
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("ti-cert-{index}"))
                .build()
                .ok()
        })
        .as_ref()?;
    // Optional keepalive: on macOS the first `pool.install` after the pool has
    // been idle (e.g. between partitions) pays a ~100-300ms worker-wake handoff
    // (thread park / E-core QoS demotion). A lightweight heartbeat that touches
    // every worker keeps them warm so the handoff stays ~microseconds. Gated by
    // env so it can be toggled for measurement; harmless on Linux.
    static KEEPALIVE: OnceLock<()> = OnceLock::new();
    KEEPALIVE.get_or_init(|| {
        let enabled = std::env::var("GLRMASK_TI_PARALLEL_KEEPALIVE")
            .ok()
            .map(|value| value.trim() != "0" && !value.trim().is_empty())
            .unwrap_or(false);
        if enabled {
            let threads = pool.current_num_threads();
            std::thread::Builder::new()
                .name("ti-cert-keepalive".to_string())
                .spawn(move || loop {
                    pool.install(|| {
                        (0..threads).into_par_iter().for_each(|_| {
                            std::hint::black_box(0u64);
                        });
                    });
                    std::thread::sleep(std::time::Duration::from_micros(200));
                })
                .ok();
        }
    });
    Some(pool)
}

/// Execute one parallel TI phase on the dedicated pool. When the caller is
/// already a member of that pool (the whole-compile mode), run the closure
/// directly so nested rayon work stays in the current registry without another
/// external-thread install handoff.
fn with_ti_parallel_pool<F, R>(f: F) -> Option<R>
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let pool = ti_certification_pool()?;
    Some(if pool.current_thread_index().is_some() {
        f()
    } else {
        pool.install(f)
    })
}

/// Build (and, if configured, start the keepalive for) the TI certification
/// thread pool ahead of first use. Intended to be called at Python module
/// import so the pool is already hot when discovery begins, avoiding the
/// first-use worker-wake handoff on macOS.
pub fn warm_ti_pool() {
    let _ = ti_certification_pool();
}

/// Run `f` inside the TI certification pool so that any nested parallel TI work
/// executes as cheap in-pool work-stealing rather than paying a per-call
/// external-thread block-and-wake handoff (a large, erratic latency on macOS).
///
/// When the pool is not configured (sequential mode) or the whole-compile wrap
/// is disabled, `f` runs directly on the calling thread.
pub fn with_ti_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let enabled = std::env::var("GLRMASK_TI_PARALLEL_WHOLE_COMPILE")
        .ok()
        .map(|value| value.trim() != "0" && !value.trim().is_empty())
        .unwrap_or(false);
    if enabled {
        if let Some(pool) = ti_certification_pool() {
            return pool.install(f);
        }
    }
    f()
}

impl InterchangeabilityDfa {
    fn new(
        tokenizer: &Tokenizer,
        observed_terminals: &[bool],
        relevant_bytes: &[bool; 256],
    ) -> Self {
        Self::from_topology(
            tokenizer,
            observed_terminals,
            Arc::new(RestrictedTopology::new(tokenizer, relevant_bytes)),
        )
    }

    fn from_topology(
        tokenizer: &Tokenizer,
        observed_terminals: &[bool],
        topology: Arc<RestrictedTopology>,
    ) -> Self {
        let raw = Arc::new(TiRawDiscoveryData::new(tokenizer, &topology, None));
        Self::from_raw_discovery_data(observed_terminals, topology, raw, None, None)
    }

    fn from_context(observed_terminals: &[bool], context: &TiDiscoveryContext) -> Self {
        let is_monotone_seed = |seed_active: &[bool]| {
            observed_terminals.len() == seed_active.len()
                && observed_terminals
                    .iter()
                    .zip(seed_active)
                    .all(|(&current, &old)| !current || old)
        };
        let support_partition_seed = context
            .support_partition_seed
            .borrow()
            .as_ref()
            .filter(|seed| {
                seed.quotient.class_for_state.len() == context.topology.state_count()
                    && is_monotone_seed(&seed.active_terminals)
            })
            .cloned();
        let output_projection_seed = context
            .output_projection_seed
            .borrow()
            .as_ref()
            .filter(|seed| {
                seed.output_pair_by_state.len() == context.topology.state_count()
                    && is_monotone_seed(&seed.active_terminals)
            })
            .cloned();
        Self::from_raw_discovery_data(
            observed_terminals,
            Arc::clone(&context.topology),
            Arc::clone(&context.raw),
            support_partition_seed,
            output_projection_seed,
        )
    }

    fn from_raw_discovery_data(
        observed_terminals: &[bool],
        topology: Arc<RestrictedTopology>,
        raw: Arc<TiRawDiscoveryData>,
        support_partition_seed: Option<SupportPartitionSeed>,
        output_projection_seed: Option<OutputProjectionSeed>,
    ) -> Self {
        let state_count = topology.state_count();
        let output_projection_seed =
            output_projection_seed.or_else(|| Some(raw.full_output_projection_seed.clone()));
        let empty_pair = OutputPair {
            finalizers: OutputBits::new(0),
            future_finalizers: OutputBits::new(0),
        };
        let (output_pairs, output_pair_lookup, output_pair_by_state): (
            Arc<[OutputPair]>,
            FxHashMap<OutputPair, u32>,
            Arc<[u32]>,
        ) = if let Some(seed) = output_projection_seed {
            let mut output_pairs = vec![empty_pair.clone()];
            let mut output_pair_lookup = FxHashMap::<OutputPair, u32>::default();
            output_pair_lookup.insert(empty_pair.clone(), 0);
            let parallel_dfa_min_states = std::env::var(
                "GLRMASK_TI_PARALLEL_DFA_MIN_STATES",
            )
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(4096);
            let project_pair = |pair: &OutputPair| OutputPair {
                finalizers: OutputBits::from_active(&pair.finalizers.0, observed_terminals),
                future_finalizers: OutputBits::from_active(
                    &pair.future_finalizers.0,
                    observed_terminals,
                ),
            };
            let projected_pairs = if seed.output_pairs.len() >= parallel_dfa_min_states {
                with_ti_parallel_pool(|| {
                    seed.output_pairs
                        .par_iter()
                        .map(project_pair)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| seed.output_pairs.iter().map(project_pair).collect())
            } else {
                seed.output_pairs.iter().map(project_pair).collect()
            };
            let mut projected_id = Vec::with_capacity(seed.output_pairs.len());
            for projected in projected_pairs {
                let id = match output_pair_lookup.entry(projected) {
                    Entry::Occupied(entry) => *entry.get(),
                    Entry::Vacant(entry) => {
                        let id = output_pairs.len() as u32;
                        output_pairs.push(entry.key().clone());
                        entry.insert(id);
                        id
                    }
                };
                projected_id.push(id);
            }
            let map_state = |&old_id: &u32| projected_id[old_id as usize];
            let output_pair_by_state = if seed.output_pair_by_state.len()
                >= parallel_dfa_min_states
            {
                with_ti_parallel_pool(|| {
                    seed.output_pair_by_state
                        .par_iter()
                        .map(map_state)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| seed.output_pair_by_state.iter().map(map_state).collect())
            } else {
                seed.output_pair_by_state.iter().map(map_state).collect()
            };
            (
                Arc::from(output_pairs),
                output_pair_lookup,
                Arc::from(output_pair_by_state),
            )
        } else {
            let parallel_dfa_min_states = std::env::var(
                "GLRMASK_TI_PARALLEL_DFA_MIN_STATES",
            )
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(4096);
            let project_state = |state: usize| OutputPair {
                finalizers: OutputBits::from_active(
                    &raw.finalizer_terminals_by_state[state],
                    observed_terminals,
                ),
                future_finalizers: OutputBits::from_active(
                    &raw.future_finalizer_terminals_by_state[state],
                    observed_terminals,
                ),
            };
            let projected_pairs = if topology.real_state_count >= parallel_dfa_min_states {
                with_ti_parallel_pool(|| {
                    (0..topology.real_state_count)
                        .into_par_iter()
                        .map(project_state)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| {
                    (0..topology.real_state_count)
                        .map(project_state)
                        .collect()
                })
            } else {
                (0..topology.real_state_count)
                    .map(project_state)
                    .collect()
            };
            let mut output_pairs = vec![empty_pair.clone()];
            let mut output_pair_lookup = FxHashMap::<OutputPair, u32>::default();
            output_pair_lookup.insert(empty_pair, 0);
            let mut output_pair_by_state = Vec::with_capacity(state_count);
            for pair in projected_pairs {
                let id = match output_pair_lookup.entry(pair) {
                    Entry::Occupied(entry) => *entry.get(),
                    Entry::Vacant(entry) => {
                        let id = output_pairs.len() as u32;
                        output_pairs.push(entry.key().clone());
                        entry.insert(id);
                        id
                    }
                };
                output_pair_by_state.push(id);
            }
            output_pair_by_state.push(0);
            (
                Arc::from(output_pairs),
                output_pair_lookup,
                Arc::from(output_pair_by_state),
            )
        };

        let observed_destinations = &raw.observed_destinations;
        let mut observed_output_pair_is_observed = vec![false; output_pairs.len()];
        let mut observed_output_pair_canonical_ids = Vec::<u32>::new();
        for state in 0..topology.real_state_count {
            if !observed_destinations[state] {
                continue;
            }
            let canonical_id = output_pair_by_state[state];
            if !observed_output_pair_is_observed[canonical_id as usize] {
                observed_output_pair_is_observed[canonical_id as usize] = true;
                observed_output_pair_canonical_ids.push(canonical_id);
            }
        }
        let observed_output_pairs = observed_output_pair_canonical_ids
            .iter()
            .map(|&id| output_pairs[id as usize].clone())
            .collect::<Vec<_>>();
        let mut observed_output_pair_ids_by_terminal =
            vec![Vec::<u32>::new(); observed_terminals.len()];
        let mut observed_output_pair_support_shapes_by_terminal =
            vec![SupportTrackShape::default(); observed_terminals.len()];
        for (id, pair) in observed_output_pairs.iter().enumerate() {
            let mut finalizer_index = 0usize;
            let mut future_index = 0usize;
            while finalizer_index < pair.finalizers.0.len()
                || future_index < pair.future_finalizers.0.len()
            {
                let (terminal, category) = match (
                    pair.finalizers.0.get(finalizer_index),
                    pair.future_finalizers.0.get(future_index),
                ) {
                    (Some(&finalizer), Some(&future)) if finalizer == future => {
                        finalizer_index += 1;
                        future_index += 1;
                        (finalizer as usize, 2u8)
                    }
                    (Some(&finalizer), Some(&future)) if finalizer < future => {
                        finalizer_index += 1;
                        (finalizer as usize, 0u8)
                    }
                    (Some(_), Some(&future)) => {
                        future_index += 1;
                        (future as usize, 1u8)
                    }
                    (Some(&finalizer), None) => {
                        finalizer_index += 1;
                        (finalizer as usize, 0u8)
                    }
                    (None, Some(&future)) => {
                        future_index += 1;
                        (future as usize, 1u8)
                    }
                    (None, None) => unreachable!("nonempty sparse output merge"),
                };
                observed_output_pair_ids_by_terminal[terminal].push(id as u32);
                let shape = &mut observed_output_pair_support_shapes_by_terminal[terminal];
                match category {
                    0 => shape.finalizer_only += 1,
                    1 => shape.future_only += 1,
                    2 => shape.both += 1,
                    _ => unreachable!("known observed-output support category"),
                }
            }
        }
        let signature_capacity = CHARACTERIZATION_DOMAIN.len()
            + 4
            + topology.max_outdegree
                * (1 + blake3::OUT_LEN + 2 * (size_of::<u32>() + 4 * size_of::<TerminalID>()));
        let seed = CharacterizationHash::seed();
        let observed_output_pair_count = observed_output_pairs.len();
        // A support quotient built for a larger active output alphabet remains
        // an exact (possibly finer) congruence after projecting terminals away.
        // Reusing it can only make the sufficient support certificate more
        // conservative; any missed certificate still falls through to the
        // ordinary exact checker.
        let support_quotient = std::env::var_os(
            "GLRMASK_TI_DISABLE_FINE_SUPPORT_QUOTIENT_REUSE",
        )
            .is_none()
            .then(|| {
                support_partition_seed
                    .as_ref()
                    .map(|seed| Arc::clone(&seed.quotient))
            })
            .flatten();
        Self {
            topology,
            observed_output_pairs,
            observed_output_pair_canonical_ids,
            observed_output_pair_is_observed,
            observed_output_pair_ids_by_terminal,
            observed_output_pair_support_shapes_by_terminal,
            observed_output_pair_marks: vec![0; observed_output_pair_count],
            observed_output_pair_mark_epoch: 0,
            finalizer_states_by_terminal: Arc::clone(&raw.finalizer_states_by_terminal),
            future_finalizer_states_by_terminal: Arc::clone(
                &raw.future_finalizer_states_by_terminal,
            ),
            identity_rounds: vec![vec![seed; state_count]],
            output_pairs,
            output_pair_lookup,
            output_pair_by_state,
            reverse_predecessors: Arc::clone(&raw.reverse_predecessors),
            canonical_rounds: vec![CanonicalRound {
                classes: vec![0; state_count],
                representative_by_class: vec![0],
                classes_by_signature_hash: FxHashMap::default(),
            }],
            canonical_round_one_class_counts: None,
            canonical_round_one_source_marks: vec![0; state_count - 1],
            canonical_round_one_source_mark_epoch: 0,
            canonical_round_one_affected_sources: Vec::new(),
            canonical_round_one_left_terminal: None,
            canonical_round_one_left_sources: Vec::new(),
            canonical_round_one_changed_scratch: SmallVec::new(),
            canonical_round_one_added_scratch: SmallVec::new(),
            support_partition_seed,
            support_quotient,
            canonical_quotient: None,
            terminal_quotient_output_supports: None,
            support_cone_marks: Vec::new(),
            support_cone_mark_epoch: 0,
            parallel_terminal_supports: OnceLock::new(),
            quotient_certified: 0,
            support_transposition_certified: 0,
            support_transposition_no_template: 0,
            support_transposition_outside_cone: 0,
            support_transposition_root_rejected: 0,
            support_transposition_incoming_rejected: 0,
            support_transposition_signature_rejected: 0,
            support_transposition_support_setup_ns: 0,
            support_quotient_build_ns: 0,
            support_transposition_template_ns: 0,
            support_transposition_cone_ns: 0,
            support_transposition_verify_ns: 0,
            sparse_quotient_certified: 0,
            sparse_quotient_cone_classes_total: 0,
            sparse_quotient_cone_classes_max: 0,
            sparse_quotient_cone_ns: 0,
            sparse_quotient_refinement_ns: 0,
            sparse_quotient_map_ns: 0,
            canonical_stable_round: None,
            canonical_identity_map: None,
            canonical_identity_cached_hashes: Vec::new(),
            signature_capacity,
        }
    }

    fn state_count(&self) -> usize {
        self.topology.state_count()
    }

    fn dead_state(&self) -> usize {
        self.topology.dead_state()
    }

    /// Compute only the final canonical identity partition needed by the
    /// support-transposition proof. Unlike the generic exact fallback, support
    /// certification never reads historical round projections or signature
    /// indexes, so retaining every intermediate `CanonicalRound` is redundant.
    fn support_identity_stable_partition_moore(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.ensure_canonical_identity_round(1);
        let mut prev_prev = self.canonical_rounds[0].classes.clone();
        let mut previous = self.canonical_rounds[1].classes.clone();
        let mut cached_hashes = self.canonical_identity_cached_hashes.clone();
        let state_count = self.state_count();
        for _ in 2..=state_count * 2 {
            let next = self.canonical_identity_round_incremental(
                &previous,
                Some(&prev_prev),
                &mut cached_hashes,
                false,
            );
            if same_equality_partition_u32(&previous, &next.classes) {
                if cfg!(debug_assertions) {
                    let stable_round = self.ensure_canonical_identity_stable_round();
                    debug_assert!(same_equality_partition_u32(
                        &next.classes,
                        &self.canonical_rounds[stable_round].classes,
                    ));
                }
                return (next.classes, next.representative_by_class);
            }
            prev_prev = previous;
            previous = next.classes;
        }
        panic!(
            "support terminal interchangeability characterization did not stabilize within {} rounds",
            state_count * 2,
        );
    }

    /// Exact sparse stabilization from canonical round one. Most round-one
    /// classes are already stable: compare each member directly with its class
    /// representative, and allocate signature-grouping machinery only for the
    /// classes that actually fail. Subsequent work is confined to predecessor
    /// classes of states moved by a split.
    fn support_identity_stable_partition_sparse_worklist(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.ensure_canonical_identity_round(1);
        let state_count = self.state_count();
        let dead_state = self.dead_state();
        let mut class_for_state = self.canonical_rounds[1].classes.clone();
        let initial_class_count = self.canonical_rounds[1].representative_by_class.len();
        let mut members_by_class = vec![Vec::<u32>::new(); initial_class_count];
        for (state, &class) in class_for_state.iter().enumerate() {
            members_by_class[class as usize].push(state as u32);
        }
        let mut queue = VecDeque::<usize>::new();
        let mut queued = vec![false; initial_class_count];
        for class in 0..initial_class_count {
            let members = &members_by_class[class];
            if members.len() <= 1 {
                continue;
            }
            let representative = members[0] as usize;
            if members[1..].iter().any(|&state| {
                !self.canonical_identity_signatures_equal(
                    representative,
                    state as usize,
                    &class_for_state,
                )
            }) {
                queued[class] = true;
                queue.push_back(class);
            }
        }

        while let Some(class) = queue.pop_front() {
            queued[class] = false;
            if members_by_class[class].len() <= 1 {
                continue;
            }
            let members = std::mem::take(&mut members_by_class[class]);
            let representative = if members.binary_search(&(dead_state as u32)).is_ok() {
                dead_state as u32
            } else {
                members[0]
            };
            let mut retained = Vec::<u32>::with_capacity(members.len());
            let mut deviants = Vec::<u32>::new();
            for &member in &members {
                if self.canonical_identity_signatures_equal(
                    representative as usize,
                    member as usize,
                    &class_for_state,
                ) {
                    retained.push(member);
                } else {
                    deviants.push(member);
                }
            }
            if deviants.is_empty() {
                members_by_class[class] = retained;
                continue;
            }

            // Most unstable classes differ from their representative by only
            // a tiny exceptional set. Group just those deviants instead of
            // rehashing every representative-equivalent member.
            let mut groups = Vec::<Vec<u32>>::new();
            let mut groups_by_hash = FxHashMap::<u64, SmallVec<[usize; 2]>>::default();
            for &member in &deviants {
                let state = member as usize;
                let hash = self.canonical_identity_signature_hash(state, &class_for_state);
                let matched = groups_by_hash.get(&hash).and_then(|candidates| {
                    candidates.iter().copied().find(|&group_index| {
                        self.canonical_identity_signatures_equal(
                            state,
                            groups[group_index][0] as usize,
                            &class_for_state,
                        )
                    })
                });
                let group_index = matched.unwrap_or_else(|| {
                    let group_index = groups.len();
                    groups.push(Vec::new());
                    groups_by_hash.entry(hash).or_default().push(group_index);
                    group_index
                });
                groups[group_index].push(member);
            }

            members_by_class[class] = retained;
            for &state in &members_by_class[class] {
                class_for_state[state as usize] = class as u32;
            }

            for group in groups {
                let new_class = members_by_class.len();
                queued.push(false);
                for &state in &group {
                    class_for_state[state as usize] = new_class as u32;
                }
                members_by_class.push(group);
                for &state in &members_by_class[new_class] {
                    let state = state as usize;
                    if state >= self.reverse_predecessors.len() {
                        continue;
                    }
                    for &source in &self.reverse_predecessors[state] {
                        let source_class = class_for_state[source as usize] as usize;
                        if members_by_class[source_class].len() > 1 && !queued[source_class] {
                            queued[source_class] = true;
                            queue.push_back(source_class);
                        }
                    }
                }
            }
        }

        let mut canonical_for_class = vec![u32::MAX; members_by_class.len()];
        let mut representative_by_class = Vec::<u32>::new();
        let mut canonical_classes = Vec::with_capacity(state_count);
        for state in 0..state_count {
            let class = class_for_state[state] as usize;
            let canonical = if canonical_for_class[class] == u32::MAX {
                let canonical = representative_by_class.len() as u32;
                canonical_for_class[class] = canonical;
                representative_by_class.push(state as u32);
                canonical
            } else {
                canonical_for_class[class]
            };
            canonical_classes.push(canonical);
        }
        (canonical_classes, representative_by_class)
    }

    /// Exact worklist refinement from canonical round one. The relation only
    /// splits: when a target class splits, only predecessor classes of states
    /// that moved can become unstable. This is the sparse counterpart of the
    /// global Moore ladder and is especially effective when stabilization adds
    /// only a handful of classes.
    fn support_identity_stable_partition_worklist(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.ensure_canonical_identity_round(1);
        let state_count = self.state_count();
        let dead_state = self.dead_state();
        let mut class_for_state = self.canonical_rounds[1].classes.clone();
        let initial_class_count = self.canonical_rounds[1].representative_by_class.len();
        let mut members_by_class = vec![Vec::<u32>::new(); initial_class_count];
        for (state, &class) in class_for_state.iter().enumerate() {
            members_by_class[class as usize].push(state as u32);
        }
        let mut queue = VecDeque::<usize>::new();
        let mut queued = vec![false; initial_class_count];
        for class in 0..initial_class_count {
            if members_by_class[class].len() > 1 {
                queue.push_back(class);
                queued[class] = true;
            }
        }

        while let Some(class) = queue.pop_front() {
            queued[class] = false;
            if members_by_class[class].len() <= 1 {
                continue;
            }
            let members = members_by_class[class].clone();
            let mut groups = Vec::<Vec<u32>>::new();
            let mut group_hashes = Vec::<u64>::new();
            let mut groups_by_hash = FxHashMap::<u64, SmallVec<[usize; 2]>>::default();
            for &member in &members {
                let state = member as usize;
                let hash = self.canonical_identity_signature_hash(state, &class_for_state);
                let matched = groups_by_hash.get(&hash).and_then(|candidates| {
                    candidates.iter().copied().find(|&group_index| {
                        self.canonical_identity_signatures_equal(
                            state,
                            groups[group_index][0] as usize,
                            &class_for_state,
                        )
                    })
                });
                let group_index = matched.unwrap_or_else(|| {
                    let group_index = groups.len();
                    groups.push(Vec::new());
                    group_hashes.push(hash);
                    groups_by_hash.entry(hash).or_default().push(group_index);
                    group_index
                });
                groups[group_index].push(member);
            }
            if groups.len() == 1 {
                continue;
            }

            // Keep the synthetic dead state's subgroup on the old id so the
            // implicit missing-edge target does not globally change identity.
            let retained = groups
                .iter()
                .position(|group| group.binary_search(&(dead_state as u32)).is_ok())
                .unwrap_or(0);
            groups.swap(0, retained);
            members_by_class[class] = std::mem::take(&mut groups[0]);
            for &state in &members_by_class[class] {
                class_for_state[state as usize] = class as u32;
            }

            for mut group in groups.into_iter().skip(1) {
                let new_class = members_by_class.len();
                queued.push(false);
                for &state in &group {
                    class_for_state[state as usize] = new_class as u32;
                }
                group.shrink_to_fit();
                members_by_class.push(group);
                // Only predecessors of moved states can observe the split.
                for &state in &members_by_class[new_class] {
                    let state = state as usize;
                    if state >= self.reverse_predecessors.len() {
                        continue;
                    }
                    for &source in &self.reverse_predecessors[state] {
                        let source_class = class_for_state[source as usize] as usize;
                        if members_by_class[source_class].len() > 1 && !queued[source_class] {
                            queued[source_class] = true;
                            queue.push_back(source_class);
                        }
                    }
                }
            }
        }

        // Canonical first-state numbering keeps downstream representatives
        // deterministic while discarding any empty historical ids.
        let mut canonical_for_class = vec![u32::MAX; members_by_class.len()];
        let mut representative_by_class = Vec::<u32>::new();
        let mut canonical_classes = Vec::with_capacity(state_count);
        for state in 0..state_count {
            let class = class_for_state[state] as usize;
            let canonical = if canonical_for_class[class] == u32::MAX {
                let canonical = representative_by_class.len() as u32;
                canonical_for_class[class] = canonical;
                representative_by_class.push(state as u32);
                canonical
            } else {
                canonical_for_class[class]
            };
            canonical_classes.push(canonical);
        }

        if cfg!(debug_assertions) {
            let moore = self.support_identity_stable_partition_moore();
            debug_assert!(same_equality_partition_u32(&canonical_classes, &moore.0));
        }
        (canonical_classes, representative_by_class)
    }

    /// Hopcroft-style exact refinement from canonical round one. Output labels
    /// are already part of round-one classes, so stabilization only needs to
    /// split source classes by destination classes. Processing every non-dead
    /// block as an incoming-edge splitter is complete; the dead preimage is the
    /// complement of all non-dead preimages for each byte and is therefore
    /// redundant.
    fn support_identity_stable_partition_full(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.ensure_canonical_identity_round(1);
        let state_count = self.state_count();
        let dead_state = self.dead_state();
        let mut class_for_state = self.canonical_rounds[1].classes.clone();
        let initial_class_count = self.canonical_rounds[1].representative_by_class.len();
        let mut members_by_class = vec![Vec::<u32>::new(); initial_class_count];
        for (state, &class) in class_for_state.iter().enumerate() {
            members_by_class[class as usize].push(state as u32);
        }
        let dead_class = class_for_state[dead_state] as usize;

        // Flat reverse-edge CSR retaining byte labels.
        let mut reverse_counts = vec![0usize; state_count];
        let mut edge_count = 0usize;
        for source in 0..state_count {
            for &(_, destination) in self.topology.edges_from(source) {
                reverse_counts[destination as usize] += 1;
                edge_count += 1;
            }
        }
        let mut reverse_offsets = Vec::with_capacity(state_count + 1);
        reverse_offsets.push(0usize);
        for &count in &reverse_counts {
            reverse_offsets.push(reverse_offsets.last().copied().unwrap() + count);
        }
        let mut next_offset = reverse_offsets[..state_count].to_vec();
        let mut reverse_edges = vec![(0u8, 0u32); edge_count];
        for source in 0..state_count {
            for &(byte, destination) in self.topology.edges_from(source) {
                let offset = &mut next_offset[destination as usize];
                reverse_edges[*offset] = (byte, source as u32);
                *offset += 1;
            }
        }

        let mut queue = VecDeque::<usize>::new();
        let mut queued = vec![false; initial_class_count];
        for class in 0..initial_class_count {
            if class != dead_class {
                queue.push_back(class);
                queued[class] = true;
            }
        }
        let mut incoming_by_byte = (0..256)
            .map(|_| Vec::<u32>::new())
            .collect::<Vec<_>>();
        let mut touched_bytes = Vec::<u8>::new();
        let mut source_marks = vec![0u32; state_count];
        let mut mark_epoch = 0u32;
        let mut class_counts = vec![0usize; initial_class_count];
        let mut touched_classes = Vec::<usize>::new();

        while let Some(splitter) = queue.pop_front() {
            queued[splitter] = false;
            if members_by_class[splitter].is_empty() {
                continue;
            }
            touched_bytes.clear();
            for &destination in &members_by_class[splitter] {
                let destination = destination as usize;
                for &(byte, source) in
                    &reverse_edges[reverse_offsets[destination]..reverse_offsets[destination + 1]]
                {
                    let bucket = &mut incoming_by_byte[byte as usize];
                    if bucket.is_empty() {
                        touched_bytes.push(byte);
                    }
                    bucket.push(source);
                }
            }

            for &byte in &touched_bytes {
                let sources = &incoming_by_byte[byte as usize];
                mark_epoch = mark_epoch.wrapping_add(1);
                if mark_epoch == 0 {
                    source_marks.fill(0);
                    mark_epoch = 1;
                }
                touched_classes.clear();
                for &source in sources {
                    let source = source as usize;
                    if source_marks[source] == mark_epoch {
                        continue;
                    }
                    source_marks[source] = mark_epoch;
                    let class = class_for_state[source] as usize;
                    if class_counts[class] == 0 {
                        touched_classes.push(class);
                    }
                    class_counts[class] += 1;
                }

                for &class in &touched_classes {
                    let marked_count = class_counts[class];
                    class_counts[class] = 0;
                    let class_len = members_by_class[class].len();
                    if marked_count == 0 || marked_count == class_len {
                        continue;
                    }
                    let old_members = std::mem::take(&mut members_by_class[class]);
                    let mut marked = Vec::with_capacity(marked_count);
                    let mut unmarked = Vec::with_capacity(class_len - marked_count);
                    for state in old_members {
                        if source_marks[state as usize] == mark_epoch {
                            marked.push(state);
                        } else {
                            unmarked.push(state);
                        }
                    }

                    let contains_dead_marked = marked.binary_search(&(dead_state as u32)).is_ok();
                    let contains_dead_unmarked =
                        unmarked.binary_search(&(dead_state as u32)).is_ok();
                    let was_queued = queued[class];
                    let (retained, moved) = if contains_dead_marked {
                        (marked, unmarked)
                    } else if contains_dead_unmarked {
                        (unmarked, marked)
                    } else if was_queued || marked.len() >= unmarked.len() {
                        (marked, unmarked)
                    } else {
                        (unmarked, marked)
                    };
                    members_by_class[class] = retained;
                    for &state in &members_by_class[class] {
                        class_for_state[state as usize] = class as u32;
                    }
                    let new_class = members_by_class.len();
                    for &state in &moved {
                        class_for_state[state as usize] = new_class as u32;
                    }
                    members_by_class.push(moved);
                    queued.push(false);
                    class_counts.push(0);

                    // If the old block was already pending, both halves must be
                    // pending. Otherwise only the smaller/non-dead half is
                    // needed by Hopcroft's splitter theorem. The retained block
                    // was chosen large unless it contains dead.
                    if was_queued {
                        if new_class != dead_class {
                            queued[new_class] = true;
                            queue.push_back(new_class);
                        }
                    } else if class == dead_class {
                        queued[new_class] = true;
                        queue.push_back(new_class);
                    } else {
                        queued[new_class] = true;
                        queue.push_back(new_class);
                    }
                }
                incoming_by_byte[byte as usize].clear();
            }
        }

        let mut canonical_for_class = vec![u32::MAX; members_by_class.len()];
        let mut representative_by_class = Vec::<u32>::new();
        let mut canonical_classes = Vec::with_capacity(state_count);
        for state in 0..state_count {
            let class = class_for_state[state] as usize;
            let canonical = if canonical_for_class[class] == u32::MAX {
                let canonical = representative_by_class.len() as u32;
                canonical_for_class[class] = canonical;
                representative_by_class.push(state as u32);
                canonical
            } else {
                canonical_for_class[class]
            };
            canonical_classes.push(canonical);
        }
        if cfg!(debug_assertions) {
            let moore = self.support_identity_stable_partition_moore();
            debug_assert!(same_equality_partition_u32(&canonical_classes, &moore.0));
        }
        (canonical_classes, representative_by_class)
    }

    /// Recompute the current stable partition on a previous round's exact
    /// quotient. For every old class and byte, old equivalence guarantees a
    /// unique old target class and old destination output; projection to the
    /// current active mask preserves that uniformity. Solving the same fixed
    /// point over old classes is therefore exactly equivalent to solving over
    /// all states, then lifting the resulting coarsening back.
    fn support_identity_stable_partition_from_seed(
        &self,
        seed: &SupportPartitionSeed,
    ) -> (Vec<u32>, Vec<u32>) {
        let old_class_count = seed.quotient.representative_by_class.len();
        let dead_old_class = seed.quotient.class_for_state[self.dead_state()] as usize;
        let mut previous = vec![0u32; old_class_count];
        for _ in 1..=old_class_count.saturating_mul(2).max(1) {
            let default_class = previous[dead_old_class];
            let mut classes_by_signature =
                FxHashMap::<CanonicalSignature, u32>::default();
            let mut next = Vec::with_capacity(old_class_count);
            let mut representative_old_class = Vec::<u32>::new();
            for old_class in 0..old_class_count {
                let state = seed.quotient.representative_by_class[old_class] as usize;
                let mut components = SmallVec::<[CanonicalComponent; 8]>::new();
                for &(byte, destination) in self.topology.edges_from(state) {
                    let destination = destination as usize;
                    let target_old_class = seed.quotient.class_for_state[destination] as usize;
                    let previous_class = previous[target_old_class];
                    let output = self.output_pair_by_state[destination];
                    if previous_class == default_class && output == 0 {
                        continue;
                    }
                    components.push(CanonicalComponent {
                        byte,
                        previous_class,
                        output,
                    });
                }
                let signature = CanonicalSignature(components);
                let next_id = classes_by_signature.len() as u32;
                let class = *classes_by_signature.entry(signature).or_insert_with(|| {
                    representative_old_class.push(old_class as u32);
                    next_id
                });
                next.push(class);
            }
            if same_equality_partition_u32(&previous, &next) {
                let class_for_state = seed
                    .quotient
                    .class_for_state
                    .iter()
                    .map(|&old_class| next[old_class as usize])
                    .collect::<Vec<_>>();
                let representative_by_class = representative_old_class
                    .into_iter()
                    .map(|old_class| seed.quotient.representative_by_class[old_class as usize])
                    .collect::<Vec<_>>();
                return (class_for_state, representative_by_class);
            }
            previous = next;
        }
        panic!(
            "seeded support terminal interchangeability characterization did not stabilize within {} rounds",
            old_class_count.saturating_mul(2).max(1),
        );
    }

    fn support_identity_stable_partition_inplace(&mut self) -> (Vec<u32>, Vec<u32>) {
        let profile = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let total_started_at = profile.then(Instant::now);
        self.ensure_canonical_identity_round(1);
        let round_one_finished_at = profile.then(Instant::now);
        let state_count = self.state_count();
        let dead_state = self.dead_state();
        let mut class_for_state = self.canonical_rounds[1].classes.clone();
        let initial_class_count = self.canonical_rounds[1].representative_by_class.len();

        let mut counts = vec![0usize; initial_class_count];
        for &class in &class_for_state {
            counts[class as usize] += 1;
        }
        let mut block_start = Vec::with_capacity(state_count);
        let mut block_end = Vec::with_capacity(state_count);
        let mut offset = 0usize;
        for &count in &counts {
            block_start.push(offset);
            offset += count;
            block_end.push(offset);
        }
        let mut next = block_start.clone();
        let mut states = vec![0u32; state_count];
        for (state, &class) in class_for_state.iter().enumerate() {
            let slot = &mut next[class as usize];
            states[*slot] = state as u32;
            *slot += 1;
        }
        let mut position = vec![0usize; state_count];
        for (pos, &state) in states.iter().enumerate() {
            position[state as usize] = pos;
        }
        let dead_class = class_for_state[dead_state] as usize;
        let partition_init_finished_at = profile.then(Instant::now);

        // Flat reverse-edge CSR retaining byte labels.
        let mut reverse_counts = vec![0usize; state_count];
        let mut edge_count = 0usize;
        for source in 0..state_count {
            for &(_, destination) in self.topology.edges_from(source) {
                reverse_counts[destination as usize] += 1;
                edge_count += 1;
            }
        }
        let reverse_finished_at = profile.then(Instant::now);
        let mut reverse_offsets = Vec::with_capacity(state_count + 1);
        reverse_offsets.push(0usize);
        for &count in &reverse_counts {
            reverse_offsets.push(reverse_offsets.last().copied().unwrap() + count);
        }
        let mut next_reverse = reverse_offsets[..state_count].to_vec();
        let mut reverse_edges = vec![(0u8, 0u32); edge_count];
        for source in 0..state_count {
            for &(byte, destination) in self.topology.edges_from(source) {
                let slot = &mut next_reverse[destination as usize];
                reverse_edges[*slot] = (byte, source as u32);
                *slot += 1;
            }
        }

        let mut queue = VecDeque::<usize>::new();
        let mut queued = vec![false; initial_class_count];
        for class in 0..initial_class_count {
            if class != dead_class {
                queue.push_back(class);
                queued[class] = true;
            }
        }
        let mut incoming_by_byte = (0..256)
            .map(|_| Vec::<u32>::new())
            .collect::<Vec<_>>();
        let mut touched_bytes = Vec::<u8>::new();
        let mut marked_count = vec![0usize; initial_class_count];
        let mut touched_blocks = Vec::<usize>::new();

        while let Some(splitter) = queue.pop_front() {
            queued[splitter] = false;
            let splitter_start = block_start[splitter];
            let splitter_end = block_end[splitter];
            if splitter_start == splitter_end {
                continue;
            }
            touched_bytes.clear();
            for pos in splitter_start..splitter_end {
                let destination = states[pos] as usize;
                for &(byte, source) in
                    &reverse_edges[reverse_offsets[destination]..reverse_offsets[destination + 1]]
                {
                    let bucket = &mut incoming_by_byte[byte as usize];
                    if bucket.is_empty() {
                        touched_bytes.push(byte);
                    }
                    // Determinism guarantees one source occurrence per byte.
                    bucket.push(source);
                }
            }

            for &byte in &touched_bytes {
                touched_blocks.clear();
                for &source in &incoming_by_byte[byte as usize] {
                    let source = source as usize;
                    let class = class_for_state[source] as usize;
                    if marked_count[class] == 0 {
                        touched_blocks.push(class);
                    }
                    let target_pos = block_end[class] - 1 - marked_count[class];
                    let source_pos = position[source];
                    if source_pos != target_pos {
                        let other = states[target_pos] as usize;
                        states.swap(source_pos, target_pos);
                        position[source] = target_pos;
                        position[other] = source_pos;
                    }
                    marked_count[class] += 1;
                }
                incoming_by_byte[byte as usize].clear();

                for &class in &touched_blocks {
                    let marked = marked_count[class];
                    marked_count[class] = 0;
                    let start = block_start[class];
                    let end = block_end[class];
                    let len = end - start;
                    if marked == 0 || marked == len {
                        continue;
                    }
                    let split = end - marked;
                    let unmarked = len - marked;
                    let was_queued = queued[class];
                    let new_class = block_start.len();

                    // Synthetic dead has no explicit outgoing edges and is
                    // therefore always in the unmarked prefix. Keep its block
                    // id fixed and queue only the separated non-dead suffix.
                    let new_is_marked = class == dead_class || marked <= unmarked;
                    let (new_start, new_end) = if new_is_marked {
                        block_end[class] = split;
                        (split, end)
                    } else {
                        block_start[class] = split;
                        (start, split)
                    };
                    block_start.push(new_start);
                    block_end.push(new_end);
                    queued.push(false);
                    marked_count.push(0);
                    for pos in new_start..new_end {
                        class_for_state[states[pos] as usize] = new_class as u32;
                    }

                    if was_queued {
                        queued[new_class] = true;
                        queue.push_back(new_class);
                    } else if class == dead_class {
                        queued[new_class] = true;
                        queue.push_back(new_class);
                    } else {
                        let old_len = block_end[class] - block_start[class];
                        let new_len = new_end - new_start;
                        if old_len <= new_len {
                            queued[class] = true;
                            queue.push_back(class);
                        } else {
                            queued[new_class] = true;
                            queue.push_back(new_class);
                        }
                    }
                }
            }
        }
        let refine_finished_at = profile.then(Instant::now);

        let mut canonical_for_class = vec![u32::MAX; block_start.len()];
        let mut representative_by_class = Vec::<u32>::new();
        let mut canonical_classes = Vec::with_capacity(state_count);
        for state in 0..state_count {
            let class = class_for_state[state] as usize;
            let canonical = if canonical_for_class[class] == u32::MAX {
                let canonical = representative_by_class.len() as u32;
                canonical_for_class[class] = canonical;
                representative_by_class.push(state as u32);
                canonical
            } else {
                canonical_for_class[class]
            };
            canonical_classes.push(canonical);
        }
        if let (
            Some(total_started_at),
            Some(round_one_finished_at),
            Some(partition_init_finished_at),
            Some(reverse_finished_at),
            Some(refine_finished_at),
        ) = (
            total_started_at,
            round_one_finished_at,
            partition_init_finished_at,
            reverse_finished_at,
            refine_finished_at,
        ) {
            eprintln!(
                "[glrmask/profile][ti_support_quotient_inplace] states={} edges={} initial_classes={} final_classes={} round1_ms={:.3} partition_init_ms={:.3} reverse_ms={:.3} refine_ms={:.3} canonicalize_ms={:.3} total_ms={:.3}",
                state_count,
                edge_count,
                initial_class_count,
                representative_by_class.len(),
                (round_one_finished_at - total_started_at).as_secs_f64() * 1000.0,
                (partition_init_finished_at - round_one_finished_at).as_secs_f64() * 1000.0,
                (reverse_finished_at - partition_init_finished_at).as_secs_f64() * 1000.0,
                (refine_finished_at - reverse_finished_at).as_secs_f64() * 1000.0,
                refine_finished_at.elapsed().as_secs_f64() * 1000.0,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        (canonical_classes, representative_by_class)
    }

    fn support_identity_stable_partition(&mut self) -> (Vec<u32>, Vec<u32>) {
        if let Some(seed) = self.support_partition_seed.clone() {
            let reduced = self.support_identity_stable_partition_from_seed(&seed);
            if cfg!(debug_assertions) {
                let full = self.support_identity_stable_partition_full();
                debug_assert!(same_equality_partition_u32(&reduced.0, &full.0));
            }
            reduced
        } else {
            let inplace = self.support_identity_stable_partition_inplace();
            if std::env::var_os("GLRMASK_TI_SPARSE_WORKLIST_QUOTIENT_VERIFY").is_some() {
                let full = self.support_identity_stable_partition_full();
                assert!(same_equality_partition_u32(&inplace.0, &full.0));
            }
            inplace
        }
    }

    fn ensure_support_quotient(&mut self) {
        if self.support_quotient.is_some() {
            return;
        }
        let quotient_build_started_at =
            std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some().then(Instant::now);
        let (class_for_state, representative_by_class) =
            self.support_identity_stable_partition();
        let class_count = representative_by_class.len();
        let mut reverse_predecessors = vec![Vec::<u32>::new(); class_count];
        let mut reverse_edges = vec![Vec::<(u8, u32)>::new(); class_count];
        for (class, &representative) in representative_by_class.iter().enumerate() {
            let state = representative as usize;
            for &(byte, destination) in self.topology.edges_from(state) {
                let destination_class = class_for_state[destination as usize] as usize;
                reverse_predecessors[destination_class].push(class as u32);
                reverse_edges[destination_class].push((byte, class as u32));
            }
        }
        for predecessors in &mut reverse_predecessors {
            predecessors.sort_unstable();
            predecessors.dedup();
        }
        for edges in &mut reverse_edges {
            edges.sort_unstable();
            edges.dedup();
        }
        let class_for_state: Arc<[u32]> = class_for_state.into();
        let representative_by_class: Arc<[u32]> = representative_by_class.into();
        let (scanner_state_count, scanner_class_for_original, scanner_representative_for_class) =
            self.topology.scanner_projection_from_internal_quotient(
                Arc::clone(&class_for_state),
                Arc::clone(&representative_by_class),
            );
        self.support_quotient = Some(Arc::new(SupportQuotient {
            class_for_state,
            representative_by_class,
            reverse_predecessors,
            reverse_edges,
            scanner_state_count,
            scanner_class_for_original,
            scanner_representative_for_class,
        }));
        if let Some(started_at) = quotient_build_started_at {
            self.support_quotient_build_ns += started_at.elapsed().as_nanos() as u64;
        }
    }

    fn ensure_terminal_quotient_output_support(&mut self, terminal: TerminalID) {
        let terminal = terminal as usize;
        if self
            .terminal_quotient_output_supports
            .as_ref()
            .is_some_and(|supports| supports.get(terminal).is_some_and(Option::is_some))
        {
            return;
        }
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        self.ensure_support_quotient();
        let class_for_state = Arc::clone(
            &self
                .support_quotient
                .as_ref()
                .expect("support quotient initialized")
                .class_for_state,
        );
        let mut support = Vec::<(u32, u8)>::new();
        for (destinations, mask) in [
            (&self.finalizer_states_by_terminal[terminal], 1u8),
            (&self.future_finalizer_states_by_terminal[terminal], 2u8),
        ] {
            for &destination in destinations {
                for &source in &self.reverse_predecessors[destination as usize] {
                    support.push((class_for_state[source as usize], mask));
                }
            }
        }
        support.sort_unstable_by_key(|&(class, _)| class);
        let mut write = 0usize;
        for read in 0..support.len() {
            if write > 0 && support[write - 1].0 == support[read].0 {
                support[write - 1].1 |= support[read].1;
            } else {
                support[write] = support[read];
                write += 1;
            }
        }
        support.truncate(write);
        let supports = self.terminal_quotient_output_supports.get_or_insert_with(|| {
            vec![None; self.finalizer_states_by_terminal.len()]
        });
        supports[terminal] = Some(support);
        if let Some(started_at) = started_at {
            self.support_transposition_support_setup_ns += started_at.elapsed().as_nanos() as u64;
        }
    }

    /// Materialize quotient output supports for a batch of terminals in one
    /// pass over quotient representative rows.  The lazy single-terminal path
    /// starts from raw destination states and repeatedly walks raw reverse
    /// predecessors before deduplicating them back to quotient classes.  For a
    /// candidate family that revisits dozens or hundreds of terminals, that
    /// repeats the same projection work many times.
    ///
    /// Stable quotient classes have identical labelled successor behaviour, so
    /// one representative row per class is exact.  Scanning those rows once
    /// directly produces the same `(source_class, final/future mask)` supports.
    fn prebuild_terminal_quotient_output_supports(&mut self, candidates: &[TerminalID]) {
        if candidates.is_empty() {
            return;
        }
        self.ensure_support_quotient();
        let terminal_count = self.finalizer_states_by_terminal.len();
        let supports = self
            .terminal_quotient_output_supports
            .get_or_insert_with(|| vec![None; terminal_count]);
        let missing = candidates
            .iter()
            .copied()
            .filter(|&terminal| supports[terminal as usize].is_none())
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return;
        }

        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        let mut candidate_mask = vec![false; terminal_count];
        for &terminal in &missing {
            candidate_mask[terminal as usize] = true;
        }
        let mut built = (0..terminal_count)
            .map(|_| Vec::<(u32, u8)>::new())
            .collect::<Vec<_>>();
        let quotient = self
            .support_quotient
            .as_ref()
            .expect("support quotient initialized");
        for (class, &representative) in quotient.representative_by_class.iter().enumerate() {
            let class = class as u32;
            for &(_, destination) in self.topology.edges_from(representative as usize) {
                let pair = &self.output_pairs[self.output_pair_by_state[destination as usize] as usize];
                for (terminals, mask) in [(&pair.finalizers.0, 1u8), (&pair.future_finalizers.0, 2u8)] {
                    for terminal in terminals.iter().copied() {
                        if !candidate_mask
                            .get(terminal as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            continue;
                        }
                        let support = &mut built[terminal as usize];
                        if let Some((last_class, last_mask)) = support.last_mut()
                            && *last_class == class
                        {
                            *last_mask |= mask;
                        } else {
                            support.push((class, mask));
                        }
                    }
                }
            }
        }
        for terminal in missing {
            supports[terminal as usize] = Some(std::mem::take(&mut built[terminal as usize]));
        }
        if let Some(started_at) = started_at {
            self.support_transposition_support_setup_ns +=
                started_at.elapsed().as_nanos() as u64;
        }
    }

    /// Propose the quotient action of swapping two terminal labels by pairing
    /// their terminal-specific observable support tracks.  It is only a
    /// candidate: `support_transposition_interchange_map` proves the complete
    /// local automorphism before returning it, otherwise discovery falls back
    /// to the ordinary exact checker.
    fn support_transposition_deviations_from_supports(
        left_support: &[(u32, u8)],
        right_support: &[(u32, u8)],
    ) -> Option<SupportDeviationMap> {
        // The old implementation filtered and rescanned both sorted support
        // lists once per mask.  Compute the same three sorted set differences
        // in one merge.  A class cancels only when both its class and mask are
        // identical; equal classes with different masks remain unmatched in
        // their respective mask buckets, exactly as in the filtered scans.
        let mut left_only: [SmallVec<[u32; 8]>; 3] = std::array::from_fn(|_| SmallVec::new());
        let mut right_only: [SmallVec<[u32; 8]>; 3] =
            std::array::from_fn(|_| SmallVec::new());
        let mut left_index = 0usize;
        let mut right_index = 0usize;
        while left_index < left_support.len() || right_index < right_support.len() {
            match (left_support.get(left_index), right_support.get(right_index)) {
                (Some(&(left_class, left_mask)), Some(&(right_class, right_mask)))
                    if left_class == right_class =>
                {
                    if left_mask != right_mask {
                        left_only[(left_mask - 1) as usize].push(left_class);
                        right_only[(right_mask - 1) as usize].push(right_class);
                    }
                    left_index += 1;
                    right_index += 1;
                }
                (Some(&(left_class, left_mask)), Some(&(right_class, _)))
                    if left_class < right_class =>
                {
                    left_only[(left_mask - 1) as usize].push(left_class);
                    left_index += 1;
                }
                (Some(_), Some(&(right_class, right_mask))) => {
                    right_only[(right_mask - 1) as usize].push(right_class);
                    right_index += 1;
                }
                (Some(&(left_class, left_mask)), None) => {
                    left_only[(left_mask - 1) as usize].push(left_class);
                    left_index += 1;
                }
                (None, Some(&(right_class, right_mask))) => {
                    right_only[(right_mask - 1) as usize].push(right_class);
                    right_index += 1;
                }
                (None, None) => break,
            }
        }

        let mut deviations = SupportDeviationMap::new();
        for mask_index in 0..3 {
            if left_only[mask_index].len() != right_only[mask_index].len() {
                return None;
            }
            for (&left_class, &right_class) in left_only[mask_index]
                .iter()
                .zip(&right_only[mask_index])
            {
                deviations.push((left_class, right_class));
                deviations.push((right_class, left_class));
            }
        }
        deviations.sort_unstable_by_key(|&(source, _)| source);
        deviations
            .windows(2)
            .all(|pair| pair[0].0 != pair[1].0)
            .then_some(deviations)
    }

    fn support_transposition_deviations(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<SupportDeviationMap> {
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        self.ensure_terminal_quotient_output_support(left);
        self.ensure_terminal_quotient_output_support(right);
        let supports = self
            .terminal_quotient_output_supports
            .as_ref()
            .expect("terminal quotient output supports initialized");
        let left_support = supports.get(left as usize)?.as_ref()?;
        let right_support = supports.get(right as usize)?.as_ref()?;
        let result = Self::support_transposition_deviations_from_supports(
            left_support,
            right_support,
        );
        if let Some(started_at) = started_at {
            self.support_transposition_template_ns += started_at.elapsed().as_nanos() as u64;
        }
        result
    }

    #[inline]
    fn mapped_support_class(deviations: &[(u32, u32)], class: usize) -> u32 {
        deviations
            .binary_search_by_key(&(class as u32), |&(source, _)| source)
            .ok()
            .map(|index| deviations[index].1)
            .unwrap_or(class as u32)
    }

    fn canonical_quotient_support_permuted_signature(
        &self,
        quotient: &CanonicalQuotient,
        class: usize,
        deviations: &[(u32, u32)],
        outputs: &mut SwappedOutputIds<'_>,
    ) -> CanonicalSignature {
        let state = quotient.representative_by_class[class] as usize;
        let default_class = Self::mapped_support_class(
            deviations,
            quotient.class_for_state[self.dead_state()] as usize,
        );
        let mut components = SmallVec::<[CanonicalComponent; 8]>::new();
        for &(byte, destination) in self.topology.edges_from(state) {
            let destination = destination as usize;
            let output = outputs.id(self.output_pair_by_state[destination]);
            let previous_class = Self::mapped_support_class(
                deviations,
                quotient.class_for_state[destination] as usize,
            );
            if previous_class == default_class && output == 0 {
                continue;
            }
            components.push(CanonicalComponent {
                byte,
                previous_class,
                output,
            });
        }
        CanonicalSignature(components.into())
    }

    fn support_mask_for_class(support: &[(u32, u8)], class: usize) -> u8 {
        support
            .binary_search_by_key(&(class as u32), |&(seen, _)| seen)
            .ok()
            .map_or(0, |index| support[index].1)
    }

    /// Necessary row condition for a quotient-class image under a terminal
    /// swap, ignoring only the recursive destination-class permutation. This
    /// sharply constrains the tiny exact cone search without accepting a map.
    fn support_output_rows_match_after_swap(
        &self,
        source_state: usize,
        target_state: usize,
        outputs: &mut SparseSwappedOutputIds<'_>,
    ) -> bool {
        let source_edges = self.topology.edges_from(source_state);
        let target_edges = self.topology.edges_from(target_state);
        let mut source_index = 0usize;
        let mut target_index = 0usize;
        while source_index < source_edges.len() || target_index < target_edges.len() {
            let (source_destination, target_destination) = match (
                source_edges.get(source_index),
                target_edges.get(target_index),
            ) {
                (Some(&(source_byte, source_destination)), Some(&(target_byte, target_destination)))
                    if source_byte == target_byte =>
                {
                    source_index += 1;
                    target_index += 1;
                    (source_destination as usize, target_destination as usize)
                }
                (Some(&(source_byte, source_destination)), Some(&(target_byte, _)))
                    if source_byte < target_byte =>
                {
                    source_index += 1;
                    (source_destination as usize, self.dead_state())
                }
                (Some(_), Some(&(_, target_destination))) => {
                    target_index += 1;
                    (self.dead_state(), target_destination as usize)
                }
                (Some(&(_, source_destination)), None) => {
                    source_index += 1;
                    (source_destination as usize, self.dead_state())
                }
                (None, Some(&(_, target_destination))) => {
                    target_index += 1;
                    (self.dead_state(), target_destination as usize)
                }
                (None, None) => unreachable!("sparse edge union loop is nonempty"),
            };
            if outputs.id(self.output_pair_by_state[source_destination])
                != self.output_pair_by_state[target_destination]
            {
                return false;
            }
        }
        true
    }

    /// Exact quotient-row check for a proposed sparse class permutation. The
    /// caller chooses an exact sufficient row set.
    fn support_deviations_are_valid(
        &self,
        quotient: &SupportQuotient,
        left: TerminalID,
        right: TerminalID,
        cone_classes: &[usize],
        deviations: &[(u32, u32)],
    ) -> bool {
        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        if Self::mapped_support_class(deviations, root_class) != root_class as u32 {
            return false;
        }
        let mut outputs = SparseSwappedOutputIds::new(
            &self.output_pairs,
            &self.output_pair_lookup,
            left as usize,
            right as usize,
        );
        for &class in cone_classes {
            let target = Self::mapped_support_class(deviations, class);
            let source_state = quotient.representative_by_class[class] as usize;
            let target_state = quotient.representative_by_class[target as usize] as usize;
            let source_edges = self.topology.edges_from(source_state);
            let target_edges = self.topology.edges_from(target_state);
            let mut source_index = 0usize;
            let mut target_index = 0usize;
            while source_index < source_edges.len() || target_index < target_edges.len() {
                let (source_destination, target_destination) = match (
                    source_edges.get(source_index),
                    target_edges.get(target_index),
                ) {
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, target_destination)))
                        if source_byte == target_byte =>
                    {
                        source_index += 1;
                        target_index += 1;
                        (source_destination as usize, target_destination as usize)
                    }
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, _)))
                        if source_byte < target_byte =>
                    {
                        source_index += 1;
                        (source_destination as usize, self.dead_state())
                    }
                    (Some(_), Some(&(_, target_destination))) => {
                        target_index += 1;
                        (self.dead_state(), target_destination as usize)
                    }
                    (Some(&(_, source_destination)), None) => {
                        source_index += 1;
                        (source_destination as usize, self.dead_state())
                    }
                    (None, Some(&(_, target_destination))) => {
                        target_index += 1;
                        (self.dead_state(), target_destination as usize)
                    }
                    (None, None) => unreachable!("sparse edge union loop is nonempty"),
                };
                let expected_destination = Self::mapped_support_class(
                    deviations,
                    quotient.class_for_state[source_destination] as usize,
                );
                if quotient.class_for_state[target_destination] != expected_destination {
                    return false;
                }
                if outputs.id(self.output_pair_by_state[source_destination])
                    != self.output_pair_by_state[target_destination]
                {
                    return false;
                }
            }
        }
        true
    }

    fn search_support_cone_permutation(
        &self,
        quotient: &SupportQuotient,
        left: TerminalID,
        right: TerminalID,
        cone_classes: &[usize],
        candidates: &[SmallVec<[usize; 8]>],
        order: &[usize],
        depth: usize,
        assignment: &mut [usize],
        used_targets: &mut [bool],
        search_nodes: &mut usize,
    ) -> Option<SupportDeviationMap> {
        const MAX_SEARCH_NODES: usize = 50_000;
        if *search_nodes >= MAX_SEARCH_NODES {
            return None;
        }
        if depth == order.len() {
            let mut deviations = cone_classes
                .iter()
                .enumerate()
                .filter_map(|(source_index, &source_class)| {
                    let target_class = cone_classes[assignment[source_index]];
                    (source_class != target_class)
                        .then_some((source_class as u32, target_class as u32))
                })
                .collect::<SupportDeviationMap>();
            deviations.sort_unstable_by_key(|&(source, _)| source);
            return self
                .support_deviations_are_valid(
                    quotient,
                    left,
                    right,
                    cone_classes,
                    &deviations,
                )
                .then_some(deviations);
        }
        let source_index = order[depth];
        for &target_index in &candidates[source_index] {
            if used_targets[target_index] {
                continue;
            }
            *search_nodes += 1;
            assignment[source_index] = target_index;
            used_targets[target_index] = true;
            if let Some(found) = self.search_support_cone_permutation(
                quotient,
                left,
                right,
                cone_classes,
                candidates,
                order,
                depth + 1,
                assignment,
                used_targets,
                search_nodes,
            ) {
                return Some(found);
            }
            used_targets[target_index] = false;
            assignment[source_index] = usize::MAX;
            if *search_nodes >= MAX_SEARCH_NODES {
                break;
            }
        }
        None
    }

    /// Exact small-cone fallback for support templates whose numeric-order
    /// pairing is wrong. Candidate images must swap the two terminal support
    /// masks and match the complete destination-output row; backtracking then
    /// verifies the full recursive quotient automorphism before acceptance.
    fn support_transposition_search_deviations(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<SupportDeviationMap> {
        self.ensure_terminal_quotient_output_support(left);
        self.ensure_terminal_quotient_output_support(right);
        let supports = self
            .terminal_quotient_output_supports
            .as_ref()
            .expect("terminal quotient output supports initialized");
        let left_support = supports[left as usize].as_ref()?;
        let right_support = supports[right as usize].as_ref()?;
        self.support_transposition_search_deviations_with_supports(
            left,
            right,
            left_support,
            right_support,
        )
    }

    fn support_transposition_search_deviations_with_supports(
        &self,
        left: TerminalID,
        right: TerminalID,
        left_support: &[(u32, u8)],
        right_support: &[(u32, u8)],
    ) -> Option<SupportDeviationMap> {
        let quotient = self
            .support_quotient
            .as_ref()
            .expect("support quotient initialized");
        let cone_classes = Self::support_quotient_affected_cone_from_supports_capped(
            quotient,
            left_support,
            right_support,
            12,
        )?;
        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        let dead_class = quotient.class_for_state[self.dead_state()] as usize;
        let mut outputs = SparseSwappedOutputIds::new(
            &self.output_pairs,
            &self.output_pair_lookup,
            left as usize,
            right as usize,
        );
        let mut candidates = vec![SmallVec::<[usize; 8]>::new(); cone_classes.len()];
        for (source_index, &source_class) in cone_classes.iter().enumerate() {
            let source_left = Self::support_mask_for_class(left_support, source_class);
            let source_right = Self::support_mask_for_class(right_support, source_class);
            let source_state = quotient.representative_by_class[source_class] as usize;
            for (target_index, &target_class) in cone_classes.iter().enumerate() {
                if (source_class == root_class && target_class != root_class)
                    || (source_class == dead_class && target_class != dead_class)
                {
                    continue;
                }
                let target_left = Self::support_mask_for_class(left_support, target_class);
                let target_right = Self::support_mask_for_class(right_support, target_class);
                if target_left != source_right || target_right != source_left {
                    continue;
                }
                let target_state = quotient.representative_by_class[target_class] as usize;
                if self.support_output_rows_match_after_swap(
                    source_state,
                    target_state,
                    &mut outputs,
                ) {
                    candidates[source_index].push(target_index);
                }
            }
            if candidates[source_index].is_empty() {
                return None;
            }
        }
        let mut order = (0..cone_classes.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|&source_index| candidates[source_index].len());
        let mut assignment = vec![usize::MAX; cone_classes.len()];
        let mut used_targets = vec![false; cone_classes.len()];
        let mut search_nodes = 0usize;
        self.search_support_cone_permutation(
            quotient,
            left,
            right,
            &cone_classes,
            &candidates,
            &order,
            0,
            &mut assignment,
            &mut used_targets,
            &mut search_nodes,
        )
    }

    fn support_transposition_interchange_map(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<InterchangeMap> {
        if let Some(deviations) = self.support_transposition_deviations(left, right) {
            if let Some(map) =
                self.support_transposition_interchange_map_from_deviations(left, right, deviations)
            {
                return Some(map);
            }
        } else {
            self.support_transposition_no_template += 1;
        }
        let deviations = self.support_transposition_search_deviations(left, right)?;
        self.support_transposition_interchange_map_from_deviations(left, right, deviations)
    }

    fn support_transposition_interchange_map_from_deviations(
        &mut self,
        left: TerminalID,
        right: TerminalID,
        deviations: SupportDeviationMap,
    ) -> Option<InterchangeMap> {
        let quotient = self
            .support_quotient
            .as_ref()
            .expect("support quotient initialized");
        let supports = self
            .terminal_quotient_output_supports
            .as_ref()
            .expect("terminal quotient output supports initialized");
        let left_support = supports[left as usize]
            .as_ref()
            .expect("left terminal support initialized");
        let right_support = supports[right as usize]
            .as_ref()
            .expect("right terminal support initialized");
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        if Self::mapped_support_class(&deviations, root_class) != root_class as u32 {
            self.support_transposition_root_rejected += 1;
            return None;
        }
        let verify_started_at = profile_timing.then(Instant::now);
        if !Self::support_deviation_incoming_edges_are_valid(quotient, &deviations) {
            debug_assert!({
                let full_cone = Self::support_quotient_affected_cone_from_supports_small(
                    quotient,
                    left_support,
                    right_support,
                );
                !self.support_deviations_are_valid(
                    quotient,
                    left,
                    right,
                    &full_cone,
                    &deviations,
                )
            });
            self.support_transposition_incoming_rejected += 1;
            if let Some(started_at) = verify_started_at {
                self.support_transposition_verify_ns += started_at.elapsed().as_nanos() as u64;
            }
            return None;
        }
        let cone_started_at = profile_timing.then(Instant::now);
        let affected_rows = Self::support_quotient_affected_rows_from_supports_marked(
            quotient,
            left_support,
            right_support,
            &deviations,
            &mut self.support_cone_marks,
            &mut self.support_cone_mark_epoch,
        );
        if let Some(started_at) = cone_started_at {
            self.support_transposition_cone_ns += started_at.elapsed().as_nanos() as u64;
        }
        debug_assert!(deviations.iter().all(|&(source, target)| {
            (source as usize) < quotient.representative_by_class.len()
                && (target as usize) < quotient.representative_by_class.len()
                && affected_rows.contains(&(source as usize))
                && affected_rows.contains(&(target as usize))
        }));
        let valid = self.support_deviations_are_valid(
            quotient,
            left,
            right,
            &affected_rows,
            &deviations,
        );
        debug_assert!({
            let full_cone = Self::support_quotient_affected_cone_from_supports_small(
                quotient,
                left_support,
                right_support,
            );
            self.support_deviations_are_valid(
                quotient,
                left,
                right,
                &full_cone,
                &deviations,
            ) == valid
        });
        if !valid {
            self.support_transposition_signature_rejected += 1;
            if let Some(started_at) = verify_started_at {
                self.support_transposition_verify_ns += started_at.elapsed().as_nanos() as u64;
            }
            return None;
        }
        if let Some(started_at) = verify_started_at {
            self.support_transposition_verify_ns += started_at.elapsed().as_nanos() as u64;
        }
        self.support_transposition_certified += 1;
        Some(InterchangeMap {
            scanner_state_map: TransportScannerStateMap::Quotient {
                state_count: quotient.scanner_state_count,
                class_for_original: Arc::clone(&quotient.scanner_class_for_original),
                representative_for_class: Arc::clone(&quotient.scanner_representative_for_class),
                source_class_for_target_deviations: deviations.into_boxed_slice(),
            },
        })
    }

    /// This does not transform the lexer. It supplies the absent destination
    /// while evaluating an enabled byte transition in `characterize`.
    fn destination_for_slot(&self, state: usize, byte_slot: usize) -> usize {
        self.topology
            .destination_for_byte(state, self.topology.bytes[byte_slot])
    }

    /// Hash the canonical sparse form of one reference characterization tuple.
    /// Missing byte transitions have the common dead component. Frozen outputs
    /// are read through the canonical per-state output-pair id, avoiding two
    /// eagerly materialized state-length output vectors on the hot path.
    fn characterize_round(
        &self,
        previous: &[CharacterizationHash],
        output_pairs: &[OutputPair],
    ) -> Vec<CharacterizationHash> {
        debug_assert_eq!(previous.len(), self.state_count());
        let mut next = Vec::with_capacity(self.state_count());
        let mut tuple = Vec::with_capacity(self.signature_capacity);
        let dead_state = self.dead_state();
        let default_hash = previous[dead_state];
        for state in 0..self.state_count() {
            tuple.clear();
            tuple.extend_from_slice(CHARACTERIZATION_DOMAIN);
            tuple.extend_from_slice(&(self.topology.bytes.len() as u32).to_le_bytes());
            for &(byte, destination) in self.topology.edges_from(state) {
                let destination = destination as usize;
                let output_id = self.output_pair_by_state[destination] as usize;
                if previous[destination] == default_hash && output_id == 0 {
                    continue;
                }
                let output = &output_pairs[output_id];
                tuple.push(byte);
                tuple.extend_from_slice(&previous[destination].0);
                output.finalizers.append_to(&mut tuple);
                output.future_finalizers.append_to(&mut tuple);
            }
            next.push(CharacterizationHash(*blake3::hash(&tuple).as_bytes()));
        }
        next
    }

    fn ensure_identity_round(&mut self, round: usize) {
        while self.identity_rounds.len() <= round {
            let previous_index = self.identity_rounds.len() - 1;
            let next = self.characterize_round(
                &self.identity_rounds[previous_index],
                &self.output_pairs,
            );
            self.identity_rounds.push(next);
        }
    }

    #[inline]
    fn canonical_signature_hash_component(
        mut hash: u64,
        byte: u8,
        previous_class: u32,
        output: u32,
    ) -> u64 {
        hash = mix_structural_fingerprint(hash, byte as u64);
        hash = mix_structural_fingerprint(hash, previous_class as u64);
        mix_structural_fingerprint(hash, output as u64)
    }

    fn canonical_signature_hash_components(components: &[CanonicalComponent]) -> u64 {
        let mut hash = 0x2d35_83d7_4f1a_6e9b;
        for component in components {
            hash = Self::canonical_signature_hash_component(
                hash,
                component.byte,
                component.previous_class,
                component.output,
            );
        }
        mix_structural_fingerprint(hash, components.len() as u64)
    }

    fn canonical_identity_signature_hash(&self, state: usize, previous: &[u32]) -> u64 {
        let default_class = previous[self.dead_state()];
        let mut hash = 0x2d35_83d7_4f1a_6e9b;
        let mut component_count = 0usize;
        for &(byte, destination) in self.topology.edges_from(state) {
            let destination = destination as usize;
            let output = self.output_pair_by_state[destination];
            let previous_class = previous[destination];
            if previous_class == default_class && output == 0 {
                continue;
            }
            hash = Self::canonical_signature_hash_component(hash, byte, previous_class, output);
            component_count += 1;
        }
        mix_structural_fingerprint(hash, component_count as u64)
    }

    fn canonical_identity_signatures_equal(
        &self,
        left_state: usize,
        right_state: usize,
        previous: &[u32],
    ) -> bool {
        let default_class = previous[self.dead_state()];
        let left_edges = self.topology.edges_from(left_state);
        let right_edges = self.topology.edges_from(right_state);
        let mut left_index = 0usize;
        let mut right_index = 0usize;

        loop {
            while let Some(&(_, destination)) = left_edges.get(left_index) {
                let destination = destination as usize;
                if previous[destination] != default_class
                    || self.output_pair_by_state[destination] != 0
                {
                    break;
                }
                left_index += 1;
            }
            while let Some(&(_, destination)) = right_edges.get(right_index) {
                let destination = destination as usize;
                if previous[destination] != default_class
                    || self.output_pair_by_state[destination] != 0
                {
                    break;
                }
                right_index += 1;
            }

            match (left_edges.get(left_index), right_edges.get(right_index)) {
                (None, None) => return true,
                (Some(_), None) | (None, Some(_)) => return false,
                (Some(&(left_byte, left_destination)), Some(&(right_byte, right_destination))) => {
                    let left_destination = left_destination as usize;
                    let right_destination = right_destination as usize;
                    if left_byte != right_byte
                        || previous[left_destination] != previous[right_destination]
                        || self.output_pair_by_state[left_destination]
                            != self.output_pair_by_state[right_destination]
                    {
                        return false;
                    }
                    left_index += 1;
                    right_index += 1;
                }
            }
        }
    }

    fn canonical_signature_matches_identity_state(
        &self,
        signature: &CanonicalSignature,
        state: usize,
        identity_previous: &[u32],
    ) -> bool {
        let default_class = identity_previous[self.dead_state()];
        let mut signature_index = 0usize;
        for &(byte, destination) in self.topology.edges_from(state) {
            let destination = destination as usize;
            let previous_class = identity_previous[destination];
            let output = self.output_pair_by_state[destination];
            if previous_class == default_class && output == 0 {
                continue;
            }
            let Some(component) = signature.0.get(signature_index) else {
                return false;
            };
            if component.byte != byte
                || component.previous_class != previous_class
                || component.output != output
            {
                return false;
            }
            signature_index += 1;
        }
        signature_index == signature.0.len()
    }

    fn canonical_round_identity_class_for_signature(
        &self,
        round: &CanonicalRound,
        identity_previous: &[u32],
        signature: &CanonicalSignature,
    ) -> Option<u32> {
        let hash = Self::canonical_signature_hash_components(&signature.0);
        round.classes_by_signature_hash.get(&hash)?.iter().copied().find(|&class| {
            let representative = round.representative_by_class[class as usize] as usize;
            self.canonical_signature_matches_identity_state(signature, representative, identity_previous)
        })
    }

    fn canonical_swapped_signature(
        &self,
        state: usize,
        previous: &[u32],
        outputs: &mut SwappedOutputIds<'_>,
    ) -> CanonicalSignature {
        let default_class = previous[self.dead_state()];
        let mut components = SmallVec::<[CanonicalComponent; 8]>::new();
        for &(byte, destination) in self.topology.edges_from(state) {
            let destination = destination as usize;
            let output = outputs.id(self.output_pair_by_state[destination]);
            if previous[destination] == default_class && output == 0 {
                continue;
            }
            components.push(CanonicalComponent {
                byte,
                previous_class: previous[destination],
                output,
            });
        }
        CanonicalSignature(components.into())
    }

    /// Sparse-output-cache variant for the first-round rejection filter. That
    /// filter usually touches only a handful of distinct output-pair ids per
    /// candidate, so inline linear lookup avoids constructing two hash maps for
    /// every tested terminal pair. The produced signature is byte-for-byte
    /// identical to `canonical_swapped_signature`.
    fn canonical_swapped_signature_sparse(
        &self,
        state: usize,
        previous: &[u32],
        outputs: &mut SparseSwappedOutputIds<'_>,
    ) -> CanonicalSignature {
        let default_class = previous[self.dead_state()];
        let mut components = SmallVec::<[CanonicalComponent; 8]>::new();
        for &(byte, destination) in self.topology.edges_from(state) {
            let destination = destination as usize;
            let output = outputs.id(self.output_pair_by_state[destination]);
            if previous[destination] == default_class && output == 0 {
                continue;
            }
            components.push(CanonicalComponent {
                byte,
                previous_class: previous[destination],
                output,
            });
        }
        CanonicalSignature(components.into())
    }

    fn canonical_identity_round(&self, previous: &[u32]) -> CanonicalRound {
        let mut representative_by_class = Vec::<u32>::new();
        let mut classes_by_signature_hash = FxHashMap::<u64, SmallVec<[u32; 1]>>::default();
        let mut classes = Vec::with_capacity(self.state_count());
        for state in 0..self.state_count() {
            let hash = self.canonical_identity_signature_hash(state, previous);
            let existing = classes_by_signature_hash.get(&hash).and_then(|candidates| {
                candidates.iter().copied().find(|&class| {
                    self.canonical_identity_signatures_equal(
                        state,
                        representative_by_class[class as usize] as usize,
                        previous,
                    )
                })
            });
            let class = existing.unwrap_or_else(|| {
                let class = representative_by_class.len() as u32;
                representative_by_class.push(state as u32);
                classes_by_signature_hash.entry(hash).or_default().push(class);
                class
            });
            classes.push(class);
        }
        CanonicalRound {
            classes,
            representative_by_class,
            classes_by_signature_hash,
        }
    }

    /// Evaluate one exact identity-refinement round on a stable quotient from
    /// an earlier, larger active terminal alphabet. Projecting terminal labels
    /// away preserves that quotient as a congruence, so every current-round
    /// signature is constant on each old class. Iterating old classes in their
    /// canonical first-state order preserves the raw-state builder's class ids
    /// and representatives exactly; only the final class vector is lifted.
    fn canonical_identity_round_from_support_seed(
        &self,
        previous: &[u32],
        seed: &SupportPartitionSeed,
    ) -> CanonicalRound {
        debug_assert_eq!(previous.len(), self.state_count());
        debug_assert_eq!(seed.quotient.class_for_state.len(), self.state_count());
        debug_assert!(seed
            .quotient
            .representative_by_class
            .windows(2)
            .all(|pair| pair[0] < pair[1]));

        let mut representative_by_class = Vec::<u32>::new();
        let mut classes_by_signature_hash =
            FxHashMap::<u64, SmallVec<[u32; 1]>>::default();
        let mut class_for_old =
            Vec::<u32>::with_capacity(seed.quotient.representative_by_class.len());
        for &state in seed.quotient.representative_by_class.iter() {
            let state = state as usize;
            let hash = self.canonical_identity_signature_hash(state, previous);
            let existing = classes_by_signature_hash.get(&hash).and_then(|candidates| {
                candidates.iter().copied().find(|&class| {
                    self.canonical_identity_signatures_equal(
                        state,
                        representative_by_class[class as usize] as usize,
                        previous,
                    )
                })
            });
            let class = existing.unwrap_or_else(|| {
                let class = representative_by_class.len() as u32;
                representative_by_class.push(state as u32);
                classes_by_signature_hash.entry(hash).or_default().push(class);
                class
            });
            class_for_old.push(class);
        }
        let classes = seed
            .quotient
            .class_for_state
            .iter()
            .map(|&old_class| class_for_old[old_class as usize])
            .collect();
        CanonicalRound {
            classes,
            representative_by_class,
            classes_by_signature_hash,
        }
    }

    /// Exact incremental variant of [`Self::canonical_identity_round`].
    ///
    /// The full recompute rehashes every state each round. In Moore
    /// refinement a state's signature can only change when one of its
    /// successors changed class in the previous step, so this reuses the
    /// per-state hashes from the last round and only rehashes the states that
    /// have an edge into a class that changed between `prev_prev` and
    /// `previous`. The grouping still runs over every state in index order,
    /// so the resulting `classes`, `representative_by_class`, and
    /// `classes_by_signature_hash` are byte-for-byte identical to the full
    /// recompute. `cached_hashes` is updated in place to hold this round's
    /// hashes for the next call.
    fn canonical_identity_round_incremental(
        &self,
        previous: &[u32],
        prev_prev: Option<&[u32]>,
        cached_hashes: &mut Vec<u64>,
        retain_signature_index: bool,
    ) -> CanonicalRound {
        let state_count = self.state_count();
        let dead = self.dead_state();
        // A full pass is required whenever there is no usable cache, or when the
        // dead-state class moved (it sets the skip threshold for every state).
        let full = match prev_prev {
            Some(prev_prev)
                if cached_hashes.len() == state_count && previous[dead] == prev_prev[dead] =>
            {
                false
            }
            _ => true,
        };
        if cached_hashes.len() != state_count {
            cached_hashes.clear();
            cached_hashes.resize(state_count, 0);
        }
        if full {
            let mut representative_by_class = Vec::<u32>::new();
            let mut classes_by_signature_hash = FxHashMap::<u64, SmallVec<[u32; 1]>>::default();
            let mut classes = Vec::with_capacity(state_count);
            for state in 0..state_count {
                let hash = self.canonical_identity_signature_hash(state, previous);
                cached_hashes[state] = hash;
                let existing = classes_by_signature_hash.get(&hash).and_then(|candidates| {
                    candidates.iter().copied().find(|&class| {
                        self.canonical_identity_signatures_equal(
                            state,
                            representative_by_class[class as usize] as usize,
                            previous,
                        )
                    })
                });
                let class = existing.unwrap_or_else(|| {
                    let class = representative_by_class.len() as u32;
                    representative_by_class.push(state as u32);
                    classes_by_signature_hash.entry(hash).or_default().push(class);
                    class
                });
                classes.push(class);
            }
            return CanonicalRound {
                classes,
                representative_by_class,
                classes_by_signature_hash: if retain_signature_index {
                    classes_by_signature_hash
                } else {
                    FxHashMap::default()
                },
            };
        }

        // Incremental (non-full) refinement. Moore refinement only ever splits
        // classes, so a round-(r-1) class stays whole unless one of its members
        // is dirty (i.e. has a successor that changed class between prev_prev
        // and previous). Clean members keep their unchanged signature, hence
        // their class; only dirty members are regrouped within their class.
        // This avoids the per-state signature-equality edge walk for the
        // overwhelming majority of (clean) states, while still producing the
        // exact same `classes`, `representative_by_class`, and
        // `classes_by_signature_hash` as the full recompute (verified by the
        // debug-only assertion in `ensure_canonical_identity_round`).
        let prev_prev = prev_prev.expect("non-full pass has a previous partition");
        let mut dirty = vec![false; state_count];
        for destination in 0..self.reverse_predecessors.len() {
            if previous[destination] != prev_prev[destination] {
                for &source in &self.reverse_predecessors[destination] {
                    dirty[source as usize] = true;
                }
            }
        }
        let prev_class_count = previous
            .iter()
            .copied()
            .max()
            .map_or(0, |class| class as usize + 1);
        let mut class_clean_rep = vec![u32::MAX; prev_class_count];
        let mut class_dirty_members = vec![Vec::<u32>::new(); prev_class_count];
        for state in 0..state_count {
            let class = previous[state] as usize;
            if dirty[state] {
                cached_hashes[state] = self.canonical_identity_signature_hash(state, previous);
                class_dirty_members[class].push(state as u32);
            } else if class_clean_rep[class] == u32::MAX {
                class_clean_rep[class] = state as u32;
            }
        }
        // Group id per state: initialised to the old class id (correct for every
        // clean state and every state of an untouched class). Dirty members are
        // regrouped below; split-off sub-classes receive fresh ids.
        let mut group_of_state = previous.to_vec();
        let mut next_split_group = prev_class_count as u32;
        for class in 0..prev_class_count {
            if class_dirty_members[class].is_empty() {
                continue;
            }
            let mut sub_reps: SmallVec<[(u32, u32); 4]> = SmallVec::new();
            if class_clean_rep[class] != u32::MAX {
                sub_reps.push((class_clean_rep[class], class as u32));
            }
            for &member in &class_dirty_members[class] {
                let member_state = member as usize;
                let matched = sub_reps.iter().copied().find(|&(rep, _)| {
                    self.canonical_identity_signatures_equal(member_state, rep as usize, previous)
                });
                let group = match matched {
                    Some((_, group)) => group,
                    None => {
                        let group = if sub_reps.is_empty() {
                            class as u32
                        } else {
                            let g = next_split_group;
                            next_split_group += 1;
                            g
                        };
                        sub_reps.push((member, group));
                        group
                    }
                };
                group_of_state[member_state] = group;
            }
        }
        // Canonical index-order renumbering. Two states share a group iff they
        // share a round-r signature, and (by monotone refinement) states in
        // different old classes never share a new signature, so first-appearance
        // order of groups equals first-appearance order of signatures: the class
        // ids, representatives, and hash buckets match the full recompute byte
        // for byte.
        let mut final_of_group = vec![u32::MAX; next_split_group as usize];
        let mut representative_by_class = Vec::<u32>::new();
        let mut classes_by_signature_hash = FxHashMap::<u64, SmallVec<[u32; 1]>>::default();
        let mut classes = Vec::with_capacity(state_count);
        for state in 0..state_count {
            let group = group_of_state[state] as usize;
            let class = if final_of_group[group] == u32::MAX {
                let class = representative_by_class.len() as u32;
                final_of_group[group] = class;
                representative_by_class.push(state as u32);
                if retain_signature_index {
                    classes_by_signature_hash
                        .entry(cached_hashes[state])
                        .or_default()
                        .push(class);
                }
                class
            } else {
                final_of_group[group]
            };
            classes.push(class);
        }
        CanonicalRound {
            classes,
            representative_by_class,
            classes_by_signature_hash,
        }
    }

    fn ensure_canonical_identity_round(&mut self, round: usize) {
        while self.canonical_rounds.len() <= round {
            let started_at = Instant::now();
            let current = self.canonical_rounds.len();
            let previous = self.canonical_rounds[current - 1].classes.clone();
            let prev_prev = if current >= 2 {
                Some(self.canonical_rounds[current - 2].classes.clone())
            } else {
                None
            };
            let mut cached_hashes = std::mem::take(&mut self.canonical_identity_cached_hashes);
            let next = self
                .support_partition_seed
                .as_ref()
                .map(|seed| self.canonical_identity_round_from_support_seed(&previous, seed))
                .unwrap_or_else(|| {
                    self.canonical_identity_round_incremental(
                        &previous,
                        prev_prev.as_deref(),
                        &mut cached_hashes,
                        true,
                    )
                });
            self.canonical_identity_cached_hashes = cached_hashes;
            if cfg!(debug_assertions) {
                // The incremental refinement must be byte-for-byte identical to
                // the full recompute; verify it in debug builds (this branch is
                // never taken in release).
                let full = self.canonical_identity_round(&previous);
                debug_assert_eq!(
                    next.classes, full.classes,
                    "incremental identity round diverged from full recompute",
                );
                debug_assert_eq!(
                    next.representative_by_class, full.representative_by_class,
                    "incremental identity representatives diverged from full recompute",
                );
            }
            if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
                eprintln!(
                    "[glrmask/profile][terminal_interchangeability] canonical_identity_round={} classes={} elapsed_ms={:.3}",
                    self.canonical_rounds.len(),
                    next.representative_by_class.len(),
                    started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            self.canonical_rounds.push(next);
        }
    }

    fn ensure_canonical_identity_stable_round(&mut self) -> usize {
        if let Some(round) = self.canonical_stable_round {
            return round;
        }
        let state_count = self.state_count();
        for round in 1..=state_count * 2 {
            self.ensure_canonical_identity_round(round);
            if same_equality_partition_u32(
                &self.canonical_rounds[round - 1].classes,
                &self.canonical_rounds[round].classes,
            ) {
                self.canonical_stable_round = Some(round);
                return round;
            }
        }
        panic!(
            "canonical terminal interchangeability characterization did not stabilize within {} rounds",
            state_count * 2,
        );
    }

    fn ensure_canonical_quotient(&mut self) {
        if self.canonical_quotient.is_some() {
            return;
        }
        let stable_round = self.ensure_canonical_identity_stable_round();
        let class_for_state = self.canonical_rounds[stable_round].classes.clone();
        let class_count = self.canonical_rounds[stable_round]
            .representative_by_class
            .len();
        debug_assert_eq!(
            class_count,
            class_for_state.iter().copied().max().map_or(0, |class| class as usize + 1),
        );

        let representative_by_class = self.canonical_rounds[stable_round]
            .representative_by_class
            .clone();

        let mut reverse_predecessors = vec![Vec::<u32>::new(); class_count];
        for (class, &representative) in representative_by_class.iter().enumerate() {
            let state = representative as usize;
            for &(_, destination) in self.topology.edges_from(state) {
                let destination_class = class_for_state[destination as usize] as usize;
                reverse_predecessors[destination_class].push(class as u32);
            }
        }
        for predecessors in &mut reverse_predecessors {
            predecessors.sort_unstable();
            predecessors.dedup();
        }

        let mut identity_classes_by_round = Vec::with_capacity(stable_round + 1);
        let mut identity_class_counts_by_round = Vec::with_capacity(stable_round + 1);
        for round in 0..=stable_round {
            let identity_classes = representative_by_class
                .iter()
                .map(|&state| self.canonical_rounds[round].classes[state as usize])
                .collect::<Vec<_>>();
            let class_count_at_round = self.canonical_rounds[round]
                .classes
                .iter()
                .copied()
                .max()
                .map_or(0, |class| class as usize + 1);
            let mut counts = vec![0u32; class_count_at_round];
            for (quotient_class, &class) in identity_classes.iter().enumerate() {
                if (representative_by_class[quotient_class] as usize) < self.topology.real_state_count {
                    counts[class as usize] += 1;
                }
            }
            identity_classes_by_round.push(identity_classes);
            identity_class_counts_by_round.push(counts);
        }

        let stable_previous = &identity_classes_by_round[stable_round - 1];
        let stable_next = &identity_classes_by_round[stable_round];
        let stable_previous_class_count = self.canonical_rounds[stable_round - 1]
            .classes
            .iter()
            .copied()
            .max()
            .map_or(0, |class| class as usize + 1);
        let stable_next_class_count = self.canonical_rounds[stable_round]
            .classes
            .iter()
            .copied()
            .max()
            .map_or(0, |class| class as usize + 1);
        let mut stable_previous_to_next = vec![u32::MAX; stable_previous_class_count];
        let mut stable_next_to_previous = vec![u32::MAX; stable_next_class_count];
        for (&previous, &next) in stable_previous.iter().zip(stable_next) {
            let previous_target = &mut stable_previous_to_next[previous as usize];
            if *previous_target == u32::MAX {
                *previous_target = next;
            } else {
                assert_eq!(*previous_target, next, "stable TI quotient split an identity class");
            }
            let next_source = &mut stable_next_to_previous[next as usize];
            if *next_source == u32::MAX {
                *next_source = previous;
            } else {
                assert_eq!(*next_source, previous, "stable TI quotient merged identity classes");
            }
        }
        assert!(
            stable_previous_to_next.iter().all(|&target| target != u32::MAX)
                && stable_next_to_previous.iter().all(|&source| source != u32::MAX),
            "stable TI quotient omitted an identity class",
        );

        self.canonical_quotient = Some(CanonicalQuotient {
            class_for_state: class_for_state.into(),
            representative_by_class: representative_by_class.into(),
            reverse_predecessors,
            identity_classes_by_round,
            identity_class_counts_by_round,
            stable_previous_to_next,
            stable_next_to_previous,
        });
    }

    fn canonical_quotient_swapped_signature(
        &self,
        quotient: &CanonicalQuotient,
        class: usize,
        previous: &[u32],
        outputs: &mut SwappedOutputIds<'_>,
    ) -> CanonicalSignature {
        let state = quotient.representative_by_class[class] as usize;
        let default_class = previous[quotient.class_for_state[self.dead_state()] as usize];
        let mut components = SmallVec::<[CanonicalComponent; 8]>::new();
        for &(byte, destination) in self.topology.edges_from(state) {
            let destination = destination as usize;
            let output = outputs.id(self.output_pair_by_state[destination]);
            let previous_class = previous[quotient.class_for_state[destination] as usize];
            if previous_class == default_class && output == 0 {
                continue;
            }
            components.push(CanonicalComponent {
                byte,
                previous_class,
                output,
            });
        }
        CanonicalSignature(components.into())
    }

    fn quotient_identity_classes_at_round<'a>(
        &self,
        quotient: &'a CanonicalQuotient,
        round: usize,
    ) -> &'a [u32] {
        quotient
            .identity_classes_by_round
            .get(round)
            .expect("identity quotient round must be cached")
    }

    fn quotient_rooted_class_set_still_possible(
        &self,
        quotient: &CanonicalQuotient,
        identity: &[u32],
        swapped: &[u32],
    ) -> bool {
        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        if identity[root_class] != swapped[root_class] {
            return false;
        }
        let mut identity_set = FxHashSet::<u32>::default();
        let mut swapped_set = FxHashSet::<u32>::default();
        for (class, &representative) in quotient.representative_by_class.iter().enumerate() {
            if representative as usize >= self.topology.real_state_count {
                continue;
            }
            identity_set.insert(identity[class]);
            swapped_set.insert(swapped[class]);
        }
        identity_set == swapped_set
    }

    /// Non-cone quotient classes retain their identity labels, so equality of
    /// complete identity and swapped class sets reduces to labels whose final
    /// occurrence lies inside the cone.
    fn sparse_quotient_rooted_class_set_still_possible(
        &self,
        quotient: &CanonicalQuotient,
        identity: &[u32],
        identity_counts: &[u32],
        swapped_cone: &[u32],
        cone_classes: &[usize],
        cone_index_by_class: &FxHashMap<usize, usize>,
    ) -> bool {
        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        let swapped_root_class = cone_index_by_class
            .get(&root_class)
            .map(|&index| swapped_cone[index])
            .unwrap_or(identity[root_class]);
        if identity[root_class] != swapped_root_class {
            return false;
        }

        let mut changed_counts = FxHashMap::<u32, u32>::default();
        let mut introduced_or_retained = FxHashSet::<u32>::default();
        for (cone_index, &class) in cone_classes.iter().enumerate() {
            let before = identity[class];
            let after = swapped_cone[cone_index];
            if after as usize >= identity_counts.len() {
                return false;
            }
            *changed_counts.entry(before).or_default() += 1;
            introduced_or_retained.insert(after);
        }
        changed_counts.into_iter().all(|(class, changed)| {
            changed < identity_counts[class as usize]
                || introduced_or_retained.contains(&class)
        })
    }

    /// At the stable identity round the old-to-new class map is bijective.
    /// Outside the cone the paired rows are identity rows, so only the changed
    /// cone rows need checking to prove paired partition stability.
    fn sparse_quotient_pair_is_stable(
        &self,
        quotient: &CanonicalQuotient,
        swapped_previous_cone: &[u32],
        swapped_next_cone: &[u32],
    ) -> bool {
        swapped_previous_cone
            .iter()
            .zip(swapped_next_cone)
            .all(|(&previous, &next)| {
                let previous = previous as usize;
                let next = next as usize;
            previous < quotient.stable_previous_to_next.len()
                && next < quotient.stable_next_to_previous.len()
                && quotient.stable_previous_to_next[previous] == next as u32
                && quotient.stable_next_to_previous[next] == previous as u32
            })
    }

    fn quotient_interchange_map_from_classes(
        &self,
        quotient: &CanonicalQuotient,
        swapped_classes: &[u32],
    ) -> Option<InterchangeMap> {
        let class_count = quotient.representative_by_class.len();
        if swapped_classes.len() != class_count {
            return None;
        }
        // Invert the stable quotient-level class permutation rather than
        // scanning every raw state once merely to rediscover its first class
        // representative. The later raw-state expansion remains exact.
        let mut source_class_for_target_class = vec![u32::MAX; class_count];
        for source_class in 0..class_count {
            // The synthetic dead state may have its own quotient class. The
            // old raw-state scan never visited it, so keep that same domain.
            if quotient.representative_by_class[source_class] as usize
                >= self.topology.real_state_count
            {
                continue;
            }
            let target_class = swapped_classes[source_class] as usize;
            if target_class >= class_count
                || quotient.representative_by_class[target_class] as usize
                    >= self.topology.real_state_count
                || source_class_for_target_class[target_class] != u32::MAX
            {
                return None;
            }
            source_class_for_target_class[target_class] = source_class as u32;
        }
        let source_class_for_target_deviations = source_class_for_target_class
            .iter()
            .enumerate()
            .filter_map(|(target_class, &source_class)| {
                (source_class != u32::MAX && source_class != target_class as u32)
                    .then_some((target_class as u32, source_class))
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Some(InterchangeMap {
            scanner_state_map: self.topology.scanner_map_from_internal_quotient(
                Arc::clone(&quotient.class_for_state),
                Arc::clone(&quotient.representative_by_class),
                source_class_for_target_deviations,
            )?,
        })
    }

    #[inline]
    fn quotient_identity_map(&self, quotient: &CanonicalQuotient) -> InterchangeMap {
        InterchangeMap {
            scanner_state_map: self.topology.scanner_map_from_internal_quotient(
                Arc::clone(&quotient.class_for_state),
                Arc::clone(&quotient.representative_by_class),
                Box::default(),
            )
            .expect("identity TI quotient must retain a raw coordinate for every raw source class"),
        }
    }

    fn sparse_quotient_interchange_map_from_cone(
        &self,
        quotient: &CanonicalQuotient,
        cone_classes: &[usize],
        swapped_cone: &[u32],
        cone_index_by_class: &FxHashMap<usize, usize>,
    ) -> Option<InterchangeMap> {
        if cone_classes.len() != swapped_cone.len() {
            return None;
        }
        let mut targets = FxHashSet::<usize>::default();
        let mut deviations = Vec::new();
        for (cone_index, &source_class) in cone_classes.iter().enumerate() {
            let target_class = swapped_cone[cone_index] as usize;
            if source_class >= quotient.representative_by_class.len()
                || target_class >= quotient.representative_by_class.len()
                || quotient.representative_by_class[source_class] as usize
                    >= self.topology.real_state_count
                || quotient.representative_by_class[target_class] as usize
                    >= self.topology.real_state_count
                || !cone_index_by_class.contains_key(&target_class)
                || !targets.insert(target_class)
            {
                return None;
            }
            if source_class != target_class {
                deviations.push((target_class as u32, source_class as u32));
            }
        }
        if targets.len() != cone_classes.len() {
            return None;
        }
        deviations.sort_unstable_by_key(|&(target, _)| target);
        Some(InterchangeMap {
            scanner_state_map: self.topology.scanner_map_from_internal_quotient(
                Arc::clone(&quotient.class_for_state),
                Arc::clone(&quotient.representative_by_class),
                deviations.into_boxed_slice(),
            )?,
        })
    }

    /// Evaluate the exact paired characterization on the already-stable
    /// identity quotient.  Stable quotient classes are congruent for every
    /// later swapped round, so one representative per class produces the same
    /// partition relation as the raw computation.  A failed quotient proof is
    /// inconclusive and falls back to the raw exact refinement below.
    fn canonical_quotient_affected_cone(
        &self,
        quotient: &CanonicalQuotient,
        left: TerminalID,
        right: TerminalID,
    ) -> (Vec<bool>, Vec<usize>) {
        let mut in_cone = vec![false; quotient.representative_by_class.len()];
        let mut cone_classes = Vec::<usize>::new();
        let mut worklist = Vec::<u32>::new();
        for destinations in [
            &self.finalizer_states_by_terminal[left as usize],
            &self.future_finalizer_states_by_terminal[left as usize],
            &self.finalizer_states_by_terminal[right as usize],
            &self.future_finalizer_states_by_terminal[right as usize],
        ] {
            for &destination in destinations {
                for &source in &self.reverse_predecessors[destination as usize] {
                    let class = quotient.class_for_state[source as usize] as usize;
                    if !in_cone[class] {
                        in_cone[class] = true;
                        cone_classes.push(class);
                        worklist.push(class as u32);
                    }
                }
            }
        }
        while let Some(class) = worklist.pop() {
            for &predecessor in &quotient.reverse_predecessors[class as usize] {
                let predecessor = predecessor as usize;
                if !in_cone[predecessor] {
                    in_cone[predecessor] = true;
                    cone_classes.push(predecessor);
                    worklist.push(predecessor as u32);
                }
            }
        }
        (in_cone, cone_classes)
    }

    /// Allocation-free exact reverse closure for the support-transposition
    /// proof. The normal sparse fallback also needs a membership bit-vector;
    /// this fast path only iterates the resulting cone, which is tiny for the
    /// proposed transport actions.
    fn support_quotient_affected_cone_small(
        &self,
        quotient: &SupportQuotient,
        left: TerminalID,
        right: TerminalID,
    ) -> SmallVec<[usize; 16]> {
        let mut cone = SmallVec::<[usize; 16]>::new();
        for destinations in [
            &self.finalizer_states_by_terminal[left as usize],
            &self.future_finalizer_states_by_terminal[left as usize],
            &self.finalizer_states_by_terminal[right as usize],
            &self.future_finalizer_states_by_terminal[right as usize],
        ] {
            for &destination in destinations {
                for &source in &self.reverse_predecessors[destination as usize] {
                    let class = quotient.class_for_state[source as usize] as usize;
                    if !cone.contains(&class) {
                        cone.push(class);
                    }
                }
            }
        }
        let mut next = 0usize;
        while next < cone.len() {
            let class = cone[next];
            next += 1;
            for &predecessor in &quotient.reverse_predecessors[class] {
                let predecessor = predecessor as usize;
                if !cone.contains(&predecessor) {
                    cone.push(predecessor);
                }
            }
        }
        cone
    }

    fn support_quotient_affected_cone_from_supports_capped(
        quotient: &SupportQuotient,
        left_support: &[(u32, u8)],
        right_support: &[(u32, u8)],
        max_classes: usize,
    ) -> Option<SmallVec<[usize; 16]>> {
        let mut cone = SmallVec::<[usize; 16]>::new();
        let push_new = |class: usize, cone: &mut SmallVec<[usize; 16]>| {
            if cone.contains(&class) {
                return true;
            }
            if cone.len() == max_classes {
                return false;
            }
            cone.push(class);
            true
        };
        for &(class, _) in left_support.iter().chain(right_support) {
            if !push_new(class as usize, &mut cone) {
                return None;
            }
        }
        let mut next = 0usize;
        while next < cone.len() {
            let class = cone[next];
            next += 1;
            for &predecessor in &quotient.reverse_predecessors[class] {
                if !push_new(predecessor as usize, &mut cone) {
                    return None;
                }
            }
        }
        Some(cone)
    }

    fn support_quotient_affected_cone_from_supports_small(
        quotient: &SupportQuotient,
        left_support: &[(u32, u8)],
        right_support: &[(u32, u8)],
    ) -> SmallVec<[usize; 16]> {
        let mut cone = SmallVec::<[usize; 16]>::new();
        for &(class, _) in left_support.iter().chain(right_support) {
            let class = class as usize;
            if !cone.contains(&class) {
                cone.push(class);
            }
        }
        let mut next = 0usize;
        while next < cone.len() {
            let class = cone[next];
            next += 1;
            for &predecessor in &quotient.reverse_predecessors[class] {
                let predecessor = predecessor as usize;
                if !cone.contains(&predecessor) {
                    cone.push(predecessor);
                }
            }
        }
        cone
    }

    fn support_quotient_affected_cone_from_supports_marked(
        quotient: &SupportQuotient,
        left_support: &[(u32, u8)],
        right_support: &[(u32, u8)],
        marks: &mut Vec<u32>,
        mark_epoch: &mut u32,
    ) -> SmallVec<[usize; 16]> {
        let class_count = quotient.representative_by_class.len();
        if marks.len() != class_count {
            marks.resize(class_count, 0);
            *mark_epoch = 0;
        }
        *mark_epoch = mark_epoch.wrapping_add(1);
        if *mark_epoch == 0 {
            marks.fill(0);
            *mark_epoch = 1;
        }
        let epoch = *mark_epoch;
        let mut cone = SmallVec::<[usize; 16]>::new();
        for &(class, _) in left_support.iter().chain(right_support) {
            let class = class as usize;
            if marks[class] != epoch {
                marks[class] = epoch;
                cone.push(class);
            }
        }
        let mut next = 0usize;
        while next < cone.len() {
            let class = cone[next];
            next += 1;
            for &predecessor in &quotient.reverse_predecessors[class] {
                let predecessor = predecessor as usize;
                if marks[predecessor] != epoch {
                    marks[predecessor] = epoch;
                    cone.push(predecessor);
                }
            }
        }
        cone
    }

    /// Exact incoming-edge condition for a proposed sparse quotient-class
    /// permutation. For every moved class `source -> target`, applying the same
    /// permutation to the predecessor endpoint of each byte-labelled incoming
    /// edge must produce exactly the incoming edges of `target`.
    fn support_deviation_incoming_edges_are_valid(
        quotient: &SupportQuotient,
        deviations: &[(u32, u32)],
    ) -> bool {
        for &(source, target) in deviations {
            let source_edges = &quotient.reverse_edges[source as usize];
            let target_edges = &quotient.reverse_edges[target as usize];
            if source_edges.len() != target_edges.len() {
                return false;
            }
            for &(byte, predecessor) in source_edges {
                let mapped_predecessor =
                    Self::mapped_support_class(deviations, predecessor as usize);
                if target_edges
                    .binary_search(&(byte, mapped_predecessor))
                    .is_err()
                {
                    return false;
                }
            }
        }
        true
    }

    /// Exact set of quotient rows whose local automorphism equation can still
    /// change after incoming-edge compatibility of moved classes has already
    /// been proved.
    ///
    /// A fixed class whose outgoing destinations are all fixed and whose
    /// destination outputs mention neither swapped terminal has an identical
    /// row on both sides of the automorphism equation, regardless of changes
    /// deeper in the graph. Therefore verification needs only:
    ///
    /// * rows whose destination outputs mention either swapped terminal;
    /// * moved classes themselves.
    ///
    /// Direct predecessors of moved classes no longer need full row scans: the
    /// byte-labelled incoming-edge check above proves precisely the destination
    /// permutation condition for those rows. The old transitive reverse cone is
    /// sufficient but much larger.
    fn support_quotient_affected_rows_from_supports_marked(
        quotient: &SupportQuotient,
        left_support: &[(u32, u8)],
        right_support: &[(u32, u8)],
        deviations: &[(u32, u32)],
        marks: &mut Vec<u32>,
        mark_epoch: &mut u32,
    ) -> SmallVec<[usize; 16]> {
        let class_count = quotient.representative_by_class.len();
        if marks.len() != class_count {
            marks.resize(class_count, 0);
            *mark_epoch = 0;
        }
        *mark_epoch = mark_epoch.wrapping_add(1);
        if *mark_epoch == 0 {
            marks.fill(0);
            *mark_epoch = 1;
        }
        let epoch = *mark_epoch;
        let mut rows = SmallVec::<[usize; 16]>::new();
        let mut add = |class: usize| {
            if marks[class] != epoch {
                marks[class] = epoch;
                rows.push(class);
            }
        };
        for &(class, _) in left_support.iter().chain(right_support) {
            add(class as usize);
        }
        for &(source, target) in deviations {
            for class in [source as usize, target as usize] {
                add(class);
            }
        }
        rows
    }

    fn raw_terminal_cone_capped(
        &self,
        terminal: TerminalID,
        max_states: usize,
        marks: &mut Vec<u32>,
        mark_epoch: &mut u32,
    ) -> Option<Vec<u32>> {
        let state_count = self.state_count();
        if marks.len() != state_count {
            marks.resize(state_count, 0);
            *mark_epoch = 0;
        }
        *mark_epoch = mark_epoch.wrapping_add(1);
        if *mark_epoch == 0 {
            marks.fill(0);
            *mark_epoch = 1;
        }
        let epoch = *mark_epoch;
        let mut cone = Vec::<u32>::new();
        let push_new = |state: u32, cone: &mut Vec<u32>, marks: &mut [u32]| {
            let state_index = state as usize;
            if marks[state_index] == epoch {
                return true;
            }
            if cone.len() == max_states {
                return false;
            }
            marks[state_index] = epoch;
            cone.push(state);
            true
        };
        for destinations in [
            &self.finalizer_states_by_terminal[terminal as usize],
            &self.future_finalizer_states_by_terminal[terminal as usize],
        ] {
            for &destination in destinations {
                for &source in &self.reverse_predecessors[destination as usize] {
                    if !push_new(source, &mut cone, marks) {
                        return None;
                    }
                }
            }
        }
        let mut next = 0usize;
        while next < cone.len() {
            let state = cone[next] as usize;
            next += 1;
            for &predecessor in &self.reverse_predecessors[state] {
                if !push_new(predecessor, &mut cone, marks) {
                    return None;
                }
            }
        }
        cone.sort_unstable();
        Some(cone)
    }

    fn output_pair_matches_swap(
        &self,
        source_output_id: u32,
        target_output_id: u32,
        left: TerminalID,
        right: TerminalID,
    ) -> bool {
        let source = &self.output_pairs[source_output_id as usize];
        let target = &self.output_pairs[target_output_id as usize];
        let swap = Some((left as usize, right as usize));
        source.finalizers.mapped(swap) == target.finalizers
            && source.future_finalizers.mapped(swap) == target.future_finalizers
    }

    fn raw_identity_states_equivalent(
        &self,
        left: u32,
        right: u32,
        memo: &mut FxHashMap<(u32, u32), bool>,
    ) -> bool {
        if left == right {
            return true;
        }
        let key = if left <= right { (left, right) } else { (right, left) };
        if let Some(&cached) = memo.get(&key) {
            return cached;
        }
        let mut queue = VecDeque::<(u32, u32)>::new();
        let mut seen = FxHashSet::<(u32, u32)>::default();
        queue.push_back((left, right));
        while let Some((left_state, right_state)) = queue.pop_front() {
            if left_state == right_state {
                continue;
            }
            let pair_key = if left_state <= right_state {
                (left_state, right_state)
            } else {
                (right_state, left_state)
            };
            if !seen.insert(pair_key) {
                continue;
            }
            if let Some(&cached) = memo.get(&pair_key) {
                if cached {
                    continue;
                }
                memo.insert(key, false);
                return false;
            }
            for &byte in &self.topology.bytes {
                let left_destination = self
                    .topology
                    .destination_for_byte(left_state as usize, byte) as u32;
                let right_destination = self
                    .topology
                    .destination_for_byte(right_state as usize, byte) as u32;
                if self.output_pair_by_state[left_destination as usize]
                    != self.output_pair_by_state[right_destination as usize]
                {
                    memo.insert(key, false);
                    return false;
                }
                if left_destination != right_destination {
                    queue.push_back((left_destination, right_destination));
                }
            }
        }
        for pair in seen {
            memo.insert(pair, true);
        }
        memo.insert(key, true);
        true
    }

    fn raw_small_cone_interchange_map(
        &self,
        left: TerminalID,
        right: TerminalID,
        left_cone: &[u32],
        right_cone: &[u32],
        identity_equivalence_memo: &mut FxHashMap<(u32, u32), bool>,
    ) -> Option<InterchangeMap> {
        if left_cone.is_empty() || right_cone.is_empty() {
            return None;
        }
        let mut common = Vec::<u32>::new();
        let mut left_petal = Vec::<u32>::new();
        let mut right_petal = Vec::<u32>::new();
        let mut left_cursor = 0usize;
        let mut right_cursor = 0usize;
        while left_cursor < left_cone.len() || right_cursor < right_cone.len() {
            match (left_cone.get(left_cursor), right_cone.get(right_cursor)) {
                (Some(&left_state), Some(&right_state)) if left_state == right_state => {
                    common.push(left_state);
                    left_cursor += 1;
                    right_cursor += 1;
                }
                (Some(&left_state), Some(&right_state)) if left_state < right_state => {
                    left_petal.push(left_state);
                    left_cursor += 1;
                }
                (Some(_), Some(&right_state)) => {
                    right_petal.push(right_state);
                    right_cursor += 1;
                }
                (Some(&left_state), None) => {
                    left_petal.push(left_state);
                    left_cursor += 1;
                }
                (None, Some(&right_state)) => {
                    right_petal.push(right_state);
                    right_cursor += 1;
                }
                (None, None) => break,
            }
        }
        if left_petal.len() != right_petal.len() || left_petal.is_empty() {
            return None;
        }
        let cone_len = left_petal.len();
        let left_index = left_petal
            .iter()
            .enumerate()
            .map(|(index, &state)| (state, index))
            .collect::<FxHashMap<_, _>>();
        let right_index = right_petal
            .iter()
            .enumerate()
            .map(|(index, &state)| (state, index))
            .collect::<FxHashMap<_, _>>();
        let common_set = common.iter().copied().collect::<FxHashSet<_>>();

        let rows_locally_compatible = |left_state: u32, right_state: u32| {
            let left_state = left_state as usize;
            let right_state = right_state as usize;
            if (left_state == self.topology.initial_state)
                != (right_state == self.topology.initial_state)
                || self.topology.state_has_raw_source(left_state)
                    != self.topology.state_has_raw_source(right_state)
            {
                return false;
            }
            for &byte in &self.topology.bytes {
                let left_destination =
                    self.topology.destination_for_byte(left_state, byte) as u32;
                let right_destination =
                    self.topology.destination_for_byte(right_state, byte) as u32;
                if !self.output_pair_matches_swap(
                        self.output_pair_by_state[left_destination as usize],
                        self.output_pair_by_state[right_destination as usize],
                        left,
                        right,
                    )
                {
                    return false;
                }
                let left_internal = left_index.contains_key(&left_destination);
                let right_internal = right_index.contains_key(&right_destination);
                if left_internal != right_internal {
                    return false;
                }
                let left_common = common_set.contains(&left_destination);
                let right_common = common_set.contains(&right_destination);
                if left_common != right_common
                    || (left_common && left_destination != right_destination)
                    || right_index.contains_key(&left_destination)
                    || left_index.contains_key(&right_destination)
                {
                    return false;
                }
            }
            true
        };

        let mut candidates = vec![Vec::<usize>::new(); cone_len];
        for (left_slot, &left_state) in left_petal.iter().enumerate() {
            for (right_slot, &right_state) in right_petal.iter().enumerate() {
                if rows_locally_compatible(left_state, right_state) {
                    candidates[left_slot].push(right_slot);
                }
            }
            if candidates[left_slot].is_empty() {
                return None;
            }
        }
        let mut order = (0..cone_len).collect::<Vec<_>>();
        order.sort_unstable_by_key(|&slot| candidates[slot].len());
        let mut assignment = vec![usize::MAX; cone_len];
        let mut used = vec![false; cone_len];

        fn search(
            depth: usize,
            order: &[usize],
            candidates: &[Vec<usize>],
            assignment: &mut [usize],
            used: &mut [bool],
            verify: &mut dyn FnMut(&[usize]) -> bool,
        ) -> bool {
            if depth == order.len() {
                return verify(assignment);
            }
            let left_slot = order[depth];
            for &right_slot in &candidates[left_slot] {
                if used[right_slot] {
                    continue;
                }
                assignment[left_slot] = right_slot;
                used[right_slot] = true;
                if search(depth + 1, order, candidates, assignment, used, verify) {
                    return true;
                }
                used[right_slot] = false;
                assignment[left_slot] = usize::MAX;
            }
            false
        }

        let mut winning_assignment = None::<Vec<usize>>;
        let mut verify = |assignment: &[usize]| {
            let mut inverse = vec![usize::MAX; cone_len];
            for (left_slot, &right_slot) in assignment.iter().enumerate() {
                if right_slot >= cone_len || inverse[right_slot] != usize::MAX {
                    return false;
                }
                inverse[right_slot] = left_slot;
            }
            let mapped_state = |state: u32| -> u32 {
                if let Some(&slot) = left_index.get(&state) {
                    right_petal[assignment[slot]]
                } else if let Some(&slot) = right_index.get(&state) {
                    left_petal[inverse[slot]]
                } else {
                    state
                }
            };
            if mapped_state(self.topology.initial_state as u32)
                != self.topology.initial_state as u32
            {
                return false;
            }
            for &source in left_petal
                .iter()
                .chain(&right_petal)
                .chain(&common)
            {
                let target_source = mapped_state(source) as usize;
                for &byte in &self.topology.bytes {
                    let source_destination = self
                        .topology
                        .destination_for_byte(source as usize, byte) as u32;
                    let target_destination = self
                        .topology
                        .destination_for_byte(target_source, byte) as u32;
                    let destination_matches = if left_index.contains_key(&source_destination)
                        || right_index.contains_key(&source_destination)
                        || common_set.contains(&source_destination)
                    {
                        mapped_state(source_destination) == target_destination
                    } else {
                        !left_index.contains_key(&target_destination)
                            && !right_index.contains_key(&target_destination)
                            && !common_set.contains(&target_destination)
                            && self.raw_identity_states_equivalent(
                                source_destination,
                                target_destination,
                                identity_equivalence_memo,
                            )
                    };
                    if !destination_matches
                        || !self.output_pair_matches_swap(
                            self.output_pair_by_state[source_destination as usize],
                            self.output_pair_by_state[target_destination as usize],
                            left,
                            right,
                        )
                    {
                        return false;
                    }
                }
            }
            winning_assignment = Some(assignment.to_vec());
            true
        };
        if !search(
            0,
            &order,
            &candidates,
            &mut assignment,
            &mut used,
            &mut verify,
        ) {
            return None;
        }
        let assignment = winning_assignment.expect("successful cone search records assignment");
        let mut deviations = Vec::<(u32, u32)>::with_capacity(cone_len * 2);
        for (left_slot, &right_slot) in assignment.iter().enumerate() {
            let left_state = left_petal[left_slot];
            let right_state = right_petal[right_slot];
            deviations.push((left_state, right_state));
            deviations.push((right_state, left_state));
        }
        deviations.sort_unstable_by_key(|&(target, _)| target);

        let scanner_state_map = if let (Some(state_for_raw), Some(representative_for_class)) = (
            self.topology.state_for_raw.as_ref(),
            self.topology.transport_representative_by_state.as_ref(),
        ) {
            TransportScannerStateMap::Quotient {
                state_count: self.topology.raw_state_count,
                class_for_original: Arc::clone(state_for_raw),
                representative_for_class: Arc::clone(representative_for_class),
                source_class_for_target_deviations: deviations.into_boxed_slice(),
            }
        } else {
            let mut targets = (0..self.topology.real_state_count as u32).collect::<Vec<_>>();
            for &(target, source) in &deviations {
                targets[target as usize] = source;
            }
            TransportScannerStateMap::Explicit(targets.into())
        };
        Some(InterchangeMap { scanner_state_map })
    }

    fn support_quotient_terminal_cone(
        &self,
        quotient: &SupportQuotient,
        terminal: TerminalID,
    ) -> SmallVec<[usize; 8]> {
        let mut cone = SmallVec::<[usize; 8]>::new();
        for destinations in [
            &self.finalizer_states_by_terminal[terminal as usize],
            &self.future_finalizer_states_by_terminal[terminal as usize],
        ] {
            for &destination in destinations {
                for &source in &self.reverse_predecessors[destination as usize] {
                    let class = quotient.class_for_state[source as usize] as usize;
                    if !cone.contains(&class) {
                        cone.push(class);
                    }
                }
            }
        }
        let mut next = 0usize;
        while next < cone.len() {
            let class = cone[next];
            next += 1;
            for &predecessor in &quotient.reverse_predecessors[class] {
                let predecessor = predecessor as usize;
                if !cone.contains(&predecessor) {
                    cone.push(predecessor);
                }
            }
        }
        cone.sort_unstable();
        cone
    }

    fn normalized_group_output_code(
        pair: &OutputPair,
        self_terminal: TerminalID,
        group_membership: &[bool],
        group_size: usize,
        out: &mut Vec<u64>,
    ) -> bool {
        for track in [&pair.finalizers, &pair.future_finalizers] {
            let mut group_count = 0usize;
            let mut self_present = false;
            let mut outside = SmallVec::<[TerminalID; 4]>::new();
            for &terminal in &track.0 {
                if group_membership
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(false)
                {
                    group_count += 1;
                    self_present |= terminal == self_terminal;
                } else {
                    outside.push(terminal);
                }
            }
            let tag = if group_count == 0 {
                0u64
            } else if group_count == group_size {
                1u64
            } else if group_count == 1 && self_present {
                2u64
            } else {
                return false;
            };
            out.push(tag);
            out.push(outside.len() as u64);
            out.extend(outside.into_iter().map(|terminal| terminal as u64));
        }
        true
    }

    fn support_petal_code_for_order(
        &self,
        quotient: &SupportQuotient,
        petal: &[usize],
        order: &[usize],
        self_terminal: TerminalID,
        group_membership: &[bool],
        group_size: usize,
        union_petals: &[bool],
    ) -> Option<Vec<u64>> {
        let mut canonical_index = FxHashMap::<usize, usize>::default();
        for (index, &petal_index) in order.iter().enumerate() {
            canonical_index.insert(petal[petal_index], index);
        }
        let mut code = Vec::<u64>::new();
        code.push(order.len() as u64);
        for &petal_index in order {
            let class = petal[petal_index];
            let state = quotient.representative_by_class[class] as usize;
            code.push((state == self.topology.initial_state) as u64);
            let edges = self.topology.edges_from(state);
            code.push(edges.len() as u64);
            for &(byte, destination) in edges {
                let destination = destination as usize;
                let target_class = quotient.class_for_state[destination] as usize;
                code.push(byte as u64);
                if let Some(&target_index) = canonical_index.get(&target_class) {
                    code.push(0);
                    code.push(target_index as u64);
                } else {
                    if union_petals[target_class] {
                        return None;
                    }
                    code.push(1);
                    code.push(target_class as u64);
                }
                let pair = &self.output_pairs[self.output_pair_by_state[destination] as usize];
                if !Self::normalized_group_output_code(
                    pair,
                    self_terminal,
                    group_membership,
                    group_size,
                    &mut code,
                ) {
                    return None;
                }
            }
        }
        Some(code)
    }

    fn support_petal_code_for_class_order(
        &self,
        quotient: &SupportQuotient,
        class_order: &[usize],
        self_terminal: TerminalID,
        group_membership: &[bool],
        group_size: usize,
        union_petals: &[bool],
    ) -> Option<Vec<u64>> {
        let mut canonical_index = SmallVec::<[(usize, usize); 8]>::new();
        for (index, &class) in class_order.iter().enumerate() {
            canonical_index.push((class, index));
        }
        let mut code = Vec::<u64>::new();
        code.push(class_order.len() as u64);
        for &class in class_order {
            let state = quotient.representative_by_class[class] as usize;
            code.push((state == self.topology.initial_state) as u64);
            let edges = self.topology.edges_from(state);
            code.push(edges.len() as u64);
            for &(byte, destination) in edges {
                let destination = destination as usize;
                let target_class = quotient.class_for_state[destination] as usize;
                code.push(byte as u64);
                if let Some((_, target_index)) = canonical_index
                    .iter()
                    .find(|(seen_class, _)| *seen_class == target_class)
                {
                    code.push(0);
                    code.push(*target_index as u64);
                } else {
                    if union_petals[target_class] {
                        return None;
                    }
                    code.push(1);
                    code.push(target_class as u64);
                }
                let pair = &self.output_pairs[self.output_pair_by_state[destination] as usize];
                if !Self::normalized_group_output_code(
                    pair,
                    self_terminal,
                    group_membership,
                    group_size,
                    &mut code,
                ) {
                    return None;
                }
            }
        }
        Some(code)
    }

    fn support_petal_canonical_form(
        &self,
        quotient: &SupportQuotient,
        petal: &[usize],
        self_terminal: TerminalID,
        group_membership: &[bool],
        group_size: usize,
        union_petals: &[bool],
    ) -> Option<(Vec<u64>, Vec<usize>)> {
        let mut index_by_class = FxHashMap::<usize, usize>::default();
        for (index, &class) in petal.iter().enumerate() {
            index_by_class.insert(class, index);
        }
        let mut colors = vec![0u32; petal.len()];
        let mut final_signatures = Vec::<Vec<u64>>::new();
        for _ in 0..=petal.len() {
            let mut signatures = Vec::with_capacity(petal.len());
            for &class in petal {
                let state = quotient.representative_by_class[class] as usize;
                let mut signature = Vec::<u64>::new();
                signature.push((state == self.topology.initial_state) as u64);
                let edges = self.topology.edges_from(state);
                signature.push(edges.len() as u64);
                for &(byte, destination) in edges {
                    let destination = destination as usize;
                    let target_class = quotient.class_for_state[destination] as usize;
                    signature.push(byte as u64);
                    if let Some(&target_index) = index_by_class.get(&target_class) {
                        signature.push(0);
                        signature.push(colors[target_index] as u64);
                    } else {
                        if union_petals[target_class] {
                            return None;
                        }
                        signature.push(1);
                        signature.push(target_class as u64);
                    }
                    let pair =
                        &self.output_pairs[self.output_pair_by_state[destination] as usize];
                    if !Self::normalized_group_output_code(
                        pair,
                        self_terminal,
                        group_membership,
                        group_size,
                        &mut signature,
                    ) {
                        return None;
                    }
                }
                signatures.push(signature);
            }
            let mut unique = signatures.clone();
            unique.sort_unstable();
            unique.dedup();
            let next = signatures
                .iter()
                .map(|signature| unique.binary_search(signature).unwrap() as u32)
                .collect::<Vec<_>>();
            final_signatures = signatures;
            if next == colors {
                break;
            }
            colors = next;
        }
        let mut seen = FxHashSet::<u32>::default();
        if colors.iter().any(|&color| !seen.insert(color)) {
            return None;
        }
        let mut canonical_order = (0..petal.len()).collect::<Vec<_>>();
        canonical_order.sort_unstable_by_key(|&index| colors[index]);
        let mut code = Vec::<u64>::new();
        code.push(petal.len() as u64);
        for &index in &canonical_order {
            code.push(final_signatures[index].len() as u64);
            code.extend_from_slice(&final_signatures[index]);
        }
        Some((
            code,
            canonical_order
                .into_iter()
                .map(|index| petal[index])
                .collect(),
        ))
    }

    fn generic_terminal_output_code(
        pair: &OutputPair,
        self_terminal: TerminalID,
        candidate_membership: &[bool],
        out: &mut Vec<u64>,
    ) {
        for track in [&pair.finalizers, &pair.future_finalizers] {
            let mut self_present = false;
            let mut peer_count = 0usize;
            let mut outside = SmallVec::<[TerminalID; 4]>::new();
            for &terminal in &track.0 {
                if terminal == self_terminal {
                    self_present = true;
                } else if candidate_membership
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(false)
                {
                    peer_count += 1;
                } else {
                    outside.push(terminal);
                }
            }
            out.push(self_present as u64);
            out.push(peer_count as u64);
            out.push(outside.len() as u64);
            out.extend(outside.into_iter().map(|terminal| terminal as u64));
        }
    }

    fn generic_output_pair_stats(
        &self,
        candidate_membership: &[bool],
    ) -> Vec<GenericOutputPairStats> {
        self.output_pairs
            .iter()
            .map(|pair| {
                let track_stats = |track: &OutputBits| {
                    let mut stats = GenericOutputTrackStats {
                        outside_hash: 0xbb67_ae85_84ca_a73b,
                        ..GenericOutputTrackStats::default()
                    };
                    for &terminal in &track.0 {
                        if candidate_membership
                            .get(terminal as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            stats.group_count += 1;
                        } else {
                            stats.outside_count += 1;
                            stats.outside_hash = mix_structural_fingerprint(
                                stats.outside_hash,
                                terminal as u64 + 1,
                            );
                        }
                    }
                    stats
                };
                GenericOutputPairStats {
                    finalizers: track_stats(&pair.finalizers),
                    future_finalizers: track_stats(&pair.future_finalizers),
                }
            })
            .collect()
    }

    fn generic_terminal_output_fingerprint_direct(
        pair: &OutputPair,
        self_terminal: TerminalID,
        candidate_membership: &[bool],
    ) -> u64 {
        let mut fingerprint = 0x6a09_e667_f3bc_c909u64;
        for (track_index, track) in [&pair.finalizers, &pair.future_finalizers]
            .into_iter()
            .enumerate()
        {
            let mut self_present = false;
            let mut peer_count = 0usize;
            let mut outside_count = 0usize;
            let mut outside_hash = 0xbb67_ae85_84ca_a73bu64;
            for &terminal in &track.0 {
                if terminal == self_terminal {
                    self_present = true;
                } else if candidate_membership
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(false)
                {
                    peer_count += 1;
                } else {
                    outside_count += 1;
                    outside_hash = mix_structural_fingerprint(
                        outside_hash,
                        terminal as u64 + 1,
                    );
                }
            }
            fingerprint = mix_structural_fingerprint(fingerprint, track_index as u64);
            fingerprint = mix_structural_fingerprint(fingerprint, self_present as u64);
            fingerprint = mix_structural_fingerprint(fingerprint, peer_count as u64);
            fingerprint = mix_structural_fingerprint(fingerprint, outside_count as u64);
            fingerprint = mix_structural_fingerprint(fingerprint, outside_hash);
        }
        fingerprint
    }

    fn generic_terminal_output_fingerprint(
        pair: &OutputPair,
        stats: GenericOutputPairStats,
        self_terminal: TerminalID,
    ) -> u64 {
        let mut fingerprint = 0x6a09_e667_f3bc_c909u64;
        for (track_index, (track, track_stats)) in [
            (&pair.finalizers, stats.finalizers),
            (&pair.future_finalizers, stats.future_finalizers),
        ]
        .into_iter()
        .enumerate()
        {
            let self_present = track.0.binary_search(&self_terminal).is_ok();
            let peer_count = track_stats.group_count - u32::from(self_present);
            fingerprint = mix_structural_fingerprint(fingerprint, track_index as u64);
            fingerprint = mix_structural_fingerprint(fingerprint, self_present as u64);
            fingerprint = mix_structural_fingerprint(fingerprint, peer_count as u64);
            fingerprint =
                mix_structural_fingerprint(fingerprint, track_stats.outside_count as u64);
            fingerprint = mix_structural_fingerprint(fingerprint, track_stats.outside_hash);
        }
        fingerprint
    }

    fn support_terminal_cone_generic_form(
        &self,
        quotient: &SupportQuotient,
        cone: &[usize],
        terminal: TerminalID,
        candidate_membership: &[bool],
        pair_stats: Option<&[GenericOutputPairStats]>,
    ) -> (SmallVec<[u64; 8]>, Option<SmallVec<[usize; 8]>>) {
        // A one-class cone has a forced canonical order. Evaluate the same two
        // fingerprint rounds directly, without allocating an index, row table,
        // color vectors, or sorting a singleton. The exact batch certificate
        // remains the sole accepting proof.
        if std::env::var_os("GLRMASK_TI_DISABLE_SINGLETON_CONE_FAST").is_none()
            && let [class] = cone
        {
            let state = quotient.representative_by_class[*class] as usize;
            let root = state == self.topology.initial_state;
            let mut color = 0u64;
            for round in 0..2usize {
                let mut fingerprint =
                    mix_structural_fingerprint(0x510e_527f_ade6_82d1, round as u64);
                fingerprint = mix_structural_fingerprint(fingerprint, root as u64);
                for &(byte, destination) in self.topology.edges_from(state) {
                    let destination = destination as usize;
                    let target_class = quotient.class_for_state[destination] as usize;
                    fingerprint = mix_structural_fingerprint(fingerprint, byte as u64 + 1);
                    if target_class == *class {
                        fingerprint = mix_structural_fingerprint(fingerprint, 0);
                        fingerprint = mix_structural_fingerprint(fingerprint, color);
                    } else {
                        fingerprint = mix_structural_fingerprint(fingerprint, 1);
                        fingerprint =
                            mix_structural_fingerprint(fingerprint, target_class as u64 + 1);
                    }
                    let output_id = self.output_pair_by_state[destination] as usize;
                    let pair = &self.output_pairs[output_id];
                    let output = match pair_stats {
                        Some(pair_stats) => Self::generic_terminal_output_fingerprint(
                            pair,
                            pair_stats[output_id],
                            terminal,
                        ),
                        None => Self::generic_terminal_output_fingerprint_direct(
                            pair,
                            terminal,
                            candidate_membership,
                        ),
                    };
                    fingerprint = mix_structural_fingerprint(fingerprint, output);
                }
                color = fingerprint;
            }
            return (
                SmallVec::from_slice(&[1, color]),
                Some(SmallVec::from_slice(&[*class])),
            );
        }

        #[derive(Clone, Copy)]
        struct GenericEdge {
            byte: u8,
            internal_target: Option<usize>,
            external_target: usize,
            output: u64,
        }

        let mut index_by_class = SmallVec::<[(usize, usize); 8]>::new();
        for (index, &class) in cone.iter().enumerate() {
            index_by_class.push((class, index));
        }
        let mut rows = SmallVec::<[SmallVec<[GenericEdge; 8]>; 8]>::new();
        let mut root_flags = SmallVec::<[bool; 8]>::new();
        rows.reserve(cone.len());
        root_flags.reserve(cone.len());
        for &class in cone {
            let state = quotient.representative_by_class[class] as usize;
            root_flags.push(state == self.topology.initial_state);
            let mut row = SmallVec::<[GenericEdge; 8]>::new();
            for &(byte, destination) in self.topology.edges_from(state) {
                let destination = destination as usize;
                let target_class = quotient.class_for_state[destination] as usize;
                let internal_target = index_by_class
                    .iter()
                    .find(|(seen_class, _)| *seen_class == target_class)
                    .map(|(_, index)| *index);
                let pair = &self.output_pairs[self.output_pair_by_state[destination] as usize];
                row.push(GenericEdge {
                    byte,
                    internal_target,
                    external_target: target_class,
                    output: match pair_stats {
                        Some(pair_stats) => Self::generic_terminal_output_fingerprint(
                            pair,
                            pair_stats[self.output_pair_by_state[destination] as usize],
                            terminal,
                        ),
                        None => Self::generic_terminal_output_fingerprint_direct(
                            pair,
                            terminal,
                            candidate_membership,
                        ),
                    },
                });
            }
            rows.push(row);
        }

        let mut colors = SmallVec::<[u64; 8]>::new();
        let mut next = SmallVec::<[u64; 8]>::new();
        colors.resize(cone.len(), 0);
        next.resize(cone.len(), 0);
        // Fixed-depth refinement is an isomorphism invariant. Hash collisions
        // only coarsen candidate groups; the exact petal certificate below is
        // the sole accepting proof.
        for round in 0..2usize {
            for index in 0..cone.len() {
                let mut fingerprint = mix_structural_fingerprint(
                    0x510e_527f_ade6_82d1,
                    round as u64,
                );
                fingerprint = mix_structural_fingerprint(
                    fingerprint,
                    root_flags[index] as u64,
                );
                for edge in &rows[index] {
                    fingerprint =
                        mix_structural_fingerprint(fingerprint, edge.byte as u64 + 1);
                    if let Some(target_index) = edge.internal_target {
                        fingerprint = mix_structural_fingerprint(fingerprint, 0);
                        fingerprint =
                            mix_structural_fingerprint(fingerprint, colors[target_index]);
                    } else {
                        fingerprint = mix_structural_fingerprint(fingerprint, 1);
                        fingerprint = mix_structural_fingerprint(
                            fingerprint,
                            edge.external_target as u64 + 1,
                        );
                    }
                    fingerprint = mix_structural_fingerprint(fingerprint, edge.output);
                }
                next[index] = fingerprint;
            }
            std::mem::swap(&mut colors, &mut next);
        }
        let mut order = (0..colors.len()).collect::<SmallVec<[usize; 8]>>();
        order.sort_unstable_by_key(|&index| colors[index]);
        let unique = order
            .windows(2)
            .all(|pair| colors[pair[0]] != colors[pair[1]]);
        let mut sorted_colors = colors.clone();
        sorted_colors.sort_unstable();
        let mut code = SmallVec::<[u64; 8]>::new();
        code.reserve(sorted_colors.len() + 1);
        code.push(cone.len() as u64);
        code.extend(sorted_colors);
        let canonical_order = unique.then(|| {
            order
                .into_iter()
                .map(|index| cone[index])
                .collect::<SmallVec<[usize; 8]>>()
        });
        (code, canonical_order)
    }

    fn normalized_group_track_equal(
        left: &OutputBits,
        left_self: TerminalID,
        right: &OutputBits,
        right_self: TerminalID,
        group_membership: &[bool],
        group_size: usize,
    ) -> bool {
        let classify = |track: &OutputBits, self_terminal: TerminalID| {
            let mut group_count = 0usize;
            let mut self_present = false;
            for &terminal in &track.0 {
                if group_membership
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(false)
                {
                    group_count += 1;
                    self_present |= terminal == self_terminal;
                }
            }
            if group_count == 0 {
                Some(0u8)
            } else if group_count == group_size {
                Some(1u8)
            } else if group_count == 1 && self_present {
                Some(2u8)
            } else {
                None
            }
        };
        if classify(left, left_self) != classify(right, right_self) {
            return false;
        }
        left.0
            .iter()
            .copied()
            .filter(|&terminal| {
                !group_membership
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(false)
            })
            .eq(right.0.iter().copied().filter(|&terminal| {
                !group_membership
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(false)
            }))
    }

    fn normalized_group_output_equal(
        left: &OutputPair,
        left_self: TerminalID,
        right: &OutputPair,
        right_self: TerminalID,
        group_membership: &[bool],
        group_size: usize,
    ) -> bool {
        Self::normalized_group_track_equal(
            &left.finalizers,
            left_self,
            &right.finalizers,
            right_self,
            group_membership,
            group_size,
        ) && Self::normalized_group_track_equal(
            &left.future_finalizers,
            left_self,
            &right.future_finalizers,
            right_self,
            group_membership,
            group_size,
        )
    }

    fn singleton_petal_rows_match(
        &self,
        quotient: &SupportQuotient,
        left_class: usize,
        left_terminal: TerminalID,
        right_class: usize,
        right_terminal: TerminalID,
        union_petals: &[bool],
        group_membership: &[bool],
        group_size: usize,
    ) -> bool {
        let left_state = quotient.representative_by_class[left_class] as usize;
        let right_state = quotient.representative_by_class[right_class] as usize;
        if (left_state == self.topology.initial_state)
            != (right_state == self.topology.initial_state)
        {
            return false;
        }
        let left_edges = self.topology.edges_from(left_state);
        let right_edges = self.topology.edges_from(right_state);
        if left_edges.len() != right_edges.len() {
            return false;
        }
        for (&(left_byte, left_destination), &(right_byte, right_destination)) in
            left_edges.iter().zip(right_edges)
        {
            if left_byte != right_byte {
                return false;
            }
            let left_destination = left_destination as usize;
            let right_destination = right_destination as usize;
            let left_target = quotient.class_for_state[left_destination] as usize;
            let right_target = quotient.class_for_state[right_destination] as usize;
            let left_internal = left_target == left_class;
            let right_internal = right_target == right_class;
            if left_internal != right_internal {
                return false;
            }
            if !left_internal {
                if union_petals[left_target]
                    || union_petals[right_target]
                    || left_target != right_target
                {
                    return false;
                }
            }
            let left_output =
                &self.output_pairs[self.output_pair_by_state[left_destination] as usize];
            let right_output =
                &self.output_pairs[self.output_pair_by_state[right_destination] as usize];
            if !Self::normalized_group_output_equal(
                left_output,
                left_terminal,
                right_output,
                right_terminal,
                group_membership,
                group_size,
            ) {
                return false;
            }
        }
        true
    }

    /// Fast exact batch certificate. The fixed-depth cone colors only propose
    /// one class order per terminal. Hash collisions or insufficient color
    /// refinement are harmless: every resulting sparse permutation is checked
    /// by `support_deviations_are_valid` before acceptance.
    fn support_petal_batch_certificate_from_orders(
        &self,
        quotient: &SupportQuotient,
        members: &[TerminalID],
        cones: &[&[usize]],
        proposed_orders: &[&[usize]],
    ) -> Option<Vec<Vec<usize>>> {
        if members.len() < 2
            || members.len() != cones.len()
            || members.len() != proposed_orders.len()
        {
            return None;
        }
        let common = cones.iter().skip(1).fold(cones[0].to_vec(), |common, cone| {
            let mut intersection = Vec::new();
            let mut i = 0usize;
            let mut j = 0usize;
            while i < common.len() && j < cone.len() {
                match common[i].cmp(&cone[j]) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => {
                        intersection.push(common[i]);
                        i += 1;
                        j += 1;
                    }
                }
            }
            intersection
        });
        let class_count = quotient.representative_by_class.len();
        let mut owner = vec![usize::MAX; class_count];
        let mut canonical_orders = Vec::with_capacity(members.len());
        for (terminal_index, (cone, proposed_order)) in
            cones.iter().zip(proposed_orders).enumerate()
        {
            if proposed_order.len() != cone.len() {
                return None;
            }
            let mut petal_order = Vec::new();
            for &class in *proposed_order {
                if cone.binary_search(&class).is_err() {
                    return None;
                }
                if common.binary_search(&class).is_ok() {
                    continue;
                }
                if owner[class] != usize::MAX {
                    return None;
                }
                owner[class] = terminal_index;
                petal_order.push(class);
            }
            canonical_orders.push(petal_order);
        }
        if canonical_orders
            .iter()
            .skip(1)
            .any(|order| order.len() != canonical_orders[0].len())
        {
            return None;
        }

        let union_petals = owner
            .iter()
            .map(|&owner| owner != usize::MAX)
            .collect::<Vec<_>>();
        let mut group_membership = vec![false; self.finalizer_states_by_terminal.len()];
        for &terminal in members {
            group_membership[terminal as usize] = true;
        }
        for &class in &common {
            let state = quotient.representative_by_class[class] as usize;
            for &(_, destination) in self.topology.edges_from(state) {
                let destination = destination as usize;
                let target_class = quotient.class_for_state[destination] as usize;
                if union_petals[target_class] {
                    return None;
                }
                let pair = &self.output_pairs[self.output_pair_by_state[destination] as usize];
                let mut ignored = Vec::new();
                if !Self::normalized_group_output_code(
                    pair,
                    u32::MAX,
                    &group_membership,
                    members.len(),
                    &mut ignored,
                ) {
                    return None;
                }
            }
        }

        // The overwhelmingly common case is one moved quotient class per
        // terminal. Compare those rows directly, normalizing SELF and the one
        // internal petal target. This is the same exact certificate as the
        // canonical code below without allocating one Vec<u64> per member.
        if canonical_orders.iter().all(|order| order.len() == 1) {
            let representative_class = canonical_orders[0][0];
            let representative_terminal = members[0];
            for (&terminal, order) in members[1..].iter().zip(&canonical_orders[1..]) {
                if !self.singleton_petal_rows_match(
                    quotient,
                    representative_class,
                    representative_terminal,
                    order[0],
                    terminal,
                    &union_petals,
                    &group_membership,
                    members.len(),
                ) {
                    return None;
                }
            }
            return Some(canonical_orders);
        }

        // The color order is only a proposal. Compare the complete labelled
        // quotient rows in that order, with the current terminal normalized to
        // SELF. Equality proves every petal bijection at once; together with
        // reverse closure, disjointness, and the fixed-core check above, this
        // is a complete group-level automorphism certificate. No per-member
        // whole-cone verification is then necessary.
        let mut exact_code: Option<Vec<u64>> = None;
        for ((&terminal, class_order), terminal_index) in members
            .iter()
            .zip(&canonical_orders)
            .zip(0usize..)
        {
            let code = self.support_petal_code_for_class_order(
                quotient,
                class_order,
                terminal,
                &group_membership,
                members.len(),
                &union_petals,
            )?;
            if exact_code.as_ref().is_some_and(|expected| *expected != code) {
                return None;
            }
            exact_code.get_or_insert(code);
            debug_assert_eq!(terminal_index + 1, canonical_orders[..=terminal_index].len());
        }
        Some(canonical_orders)
    }

    /// Sufficient whole-class certificate: all mutable behavior is contained in
    /// pairwise-disjoint reverse-closed petals with one common canonical form;
    /// the complement is fixed and invariant under every permutation of the
    /// terminal group. Then the canonical petal bijections extend by identity
    /// to an action of the full symmetric group.
    fn support_petal_batch_certificate(
        &self,
        quotient: &SupportQuotient,
        members: &[TerminalID],
    ) -> Option<Vec<Vec<usize>>> {
        let owned_cones = members
            .iter()
            .map(|&terminal| self.support_quotient_terminal_cone(quotient, terminal))
            .collect::<Vec<_>>();
        let cones = owned_cones
            .iter()
            .map(|cone| cone.as_slice())
            .collect::<Vec<_>>();
        self.support_petal_batch_certificate_with_cones(quotient, members, &cones)
    }

    fn support_petal_batch_certificate_with_cones(
        &self,
        quotient: &SupportQuotient,
        members: &[TerminalID],
        cones: &[&[usize]],
    ) -> Option<Vec<Vec<usize>>> {
        if members.len() < 2 || members.len() != cones.len() {
            return None;
        }
        let common = cones.iter().skip(1).fold(cones[0].to_vec(), |common, cone| {
            let mut intersection = Vec::new();
            let mut i = 0usize;
            let mut j = 0usize;
            while i < common.len() && j < cone.len() {
                match common[i].cmp(&cone[j]) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => {
                        intersection.push(common[i]);
                        i += 1;
                        j += 1;
                    }
                }
            }
            intersection
        });
        let petals = cones
            .iter()
            .map(|cone| {
                cone.iter()
                    .copied()
                    .filter(|class| common.binary_search(class).is_err())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let class_count = quotient.representative_by_class.len();
        let mut owner = vec![usize::MAX; class_count];
        for (terminal_index, petal) in petals.iter().enumerate() {
            for &class in petal {
                if owner[class] != usize::MAX {
                    return None;
                }
                owner[class] = terminal_index;
            }
        }
        let union_petals = owner
            .iter()
            .map(|&owner| owner != usize::MAX)
            .collect::<Vec<_>>();
        let terminal_count = self.finalizer_states_by_terminal.len();
        let mut group_membership = vec![false; terminal_count];
        for &terminal in members {
            group_membership[terminal as usize] = true;
        }

        // Every C_t is reverse-closed, hence their union is reverse-closed:
        // no class outside the union can enter a moved petal or observe a
        // group-terminal output. Only the intersection K remains inside every
        // cone while fixed by the petal action, so only K needs an invariance
        // check.
        for &class in &common {
            let state = quotient.representative_by_class[class] as usize;
            for &(_, destination) in self.topology.edges_from(state) {
                let destination = destination as usize;
                let target_class = quotient.class_for_state[destination] as usize;
                if union_petals[target_class] {
                    return None;
                }
                let pair = &self.output_pairs[self.output_pair_by_state[destination] as usize];
                let mut ignored = Vec::new();
                // Use a sentinel outside the group: only empty/all-group tracks
                // are accepted in the fixed common core.
                if !Self::normalized_group_output_code(
                    pair,
                    u32::MAX,
                    &group_membership,
                    members.len(),
                    &mut ignored,
                ) {
                    return None;
                }
            }
        }

        let mut canonical_code: Option<Vec<u64>> = None;
        let mut canonical_orders = Vec::with_capacity(members.len());
        for (&terminal, petal) in members.iter().zip(&petals) {
            let (code, order) = self.support_petal_canonical_form(
                quotient,
                petal,
                terminal,
                &group_membership,
                members.len(),
                &union_petals,
            )?;
            if canonical_code.as_ref().is_some_and(|expected| *expected != code) {
                return None;
            }
            canonical_code.get_or_insert(code);
            canonical_orders.push(order);
        }
        Some(canonical_orders)
    }

    fn support_petal_batch_maps(
        &self,
        quotient: &SupportQuotient,
        members: &[TerminalID],
        canonical_orders: &[Vec<usize>],
    ) -> Option<Vec<InterchangeMap>> {
        if members.len() != canonical_orders.len() || members.len() < 2 {
            return None;
        }
        let representative_order = &canonical_orders[0];
        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        let mut maps = Vec::with_capacity(members.len() - 1);
        for member_order in &canonical_orders[1..] {
            if member_order.len() != representative_order.len() {
                return None;
            }
            let mut deviations = Vec::<(u32, u32)>::with_capacity(member_order.len() * 2);
            for (&representative_class, &member_class) in
                representative_order.iter().zip(member_order)
            {
                if representative_class == member_class {
                    continue;
                }
                if representative_class == root_class || member_class == root_class {
                    return None;
                }
                deviations.push((representative_class as u32, member_class as u32));
                deviations.push((member_class as u32, representative_class as u32));
            }
            deviations.sort_unstable_by_key(|&(source, _)| source);
            if deviations.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return None;
            }
            if cfg!(debug_assertions) {
                let member_index = maps.len() + 1;
                let cone_classes = self.support_quotient_affected_cone_small(
                    quotient,
                    members[0],
                    members[member_index],
                );
                debug_assert!(self.support_deviations_are_valid(
                    quotient,
                    members[0],
                    members[member_index],
                    &cone_classes,
                    &deviations,
                ));
            }
            maps.push(InterchangeMap {
                scanner_state_map: TransportScannerStateMap::Quotient {
                    state_count: quotient.scanner_state_count,
                    class_for_original: Arc::clone(&quotient.scanner_class_for_original),
                    representative_for_class: Arc::clone(
                        &quotient.scanner_representative_for_class,
                    ),
                    source_class_for_target_deviations: deviations.into_boxed_slice(),
                },
            });
        }
        Some(maps)
    }

    fn canonical_quotient_affected_cone_small(
        &self,
        quotient: &CanonicalQuotient,
        left: TerminalID,
        right: TerminalID,
    ) -> SmallVec<[usize; 16]> {
        let mut cone = SmallVec::<[usize; 16]>::new();
        for destinations in [
            &self.finalizer_states_by_terminal[left as usize],
            &self.future_finalizer_states_by_terminal[left as usize],
            &self.finalizer_states_by_terminal[right as usize],
            &self.future_finalizer_states_by_terminal[right as usize],
        ] {
            for &destination in destinations {
                for &source in &self.reverse_predecessors[destination as usize] {
                    let class = quotient.class_for_state[source as usize] as usize;
                    if !cone.contains(&class) {
                        cone.push(class);
                    }
                }
            }
        }
        let mut next = 0usize;
        while next < cone.len() {
            let class = cone[next];
            next += 1;
            for &predecessor in &quotient.reverse_predecessors[class] {
                let predecessor = predecessor as usize;
                if !cone.contains(&predecessor) {
                    cone.push(predecessor);
                }
            }
        }
        cone
    }

    fn canonical_sparse_quotient_swapped_signature(
        &self,
        quotient: &CanonicalQuotient,
        class: usize,
        identity_previous: &[u32],
        cone_index_by_class: &FxHashMap<usize, usize>,
        swapped_previous_cone: &[u32],
        outputs: &mut SwappedOutputIds<'_>,
    ) -> CanonicalSignature {
        let state = quotient.representative_by_class[class] as usize;
        let previous_class_for = |quotient_class: usize| {
            cone_index_by_class
                .get(&quotient_class)
                .map(|&index| swapped_previous_cone[index])
                .unwrap_or(identity_previous[quotient_class])
        };
        let default_class = previous_class_for(
            quotient.class_for_state[self.dead_state()] as usize,
        );
        let mut components = SmallVec::<[CanonicalComponent; 8]>::new();
        for &(byte, destination) in self.topology.edges_from(state) {
            let destination = destination as usize;
            let output = outputs.id(self.output_pair_by_state[destination]);
            let previous_class = previous_class_for(
                quotient.class_for_state[destination] as usize,
            );
            if previous_class == default_class && output == 0 {
                continue;
            }
            components.push(CanonicalComponent {
                byte,
                previous_class,
                output,
            });
        }
        CanonicalSignature(components.into())
    }

    /// Exact sparse quotient evaluation. Outside the backward cone of every
    /// relabelled frozen output, the swapped characterization is identical to
    /// the identity at every round. Only the cone needs recomputation.
    fn canonical_sparse_quotient_interchange_map(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<InterchangeMap> {
        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let stable_round = self.ensure_canonical_identity_stable_round();
        self.ensure_canonical_quotient();
        let pair_started_at = profile_timing.then(Instant::now);
        let result = {
            let quotient = self
                .canonical_quotient
                .as_ref()
                .expect("canonical quotient initialized");
            let cone_started_at = profile_timing.then(Instant::now);
            let (_in_cone, cone_classes) =
                self.canonical_quotient_affected_cone(quotient, left, right);
            let cone_size = cone_classes.len();
            if let Some(started_at) = cone_started_at {
                self.sparse_quotient_cone_ns += started_at.elapsed().as_nanos() as u64;
            }
            if cone_size == 0 {
                Some((self.quotient_identity_map(quotient), cone_size))
            } else {
                let cone_index_by_class = cone_classes
                    .iter()
                    .enumerate()
                    .map(|(index, &class)| (class, index))
                    .collect::<FxHashMap<_, _>>();
                let mut swapped_previous_cone = cone_classes
                    .iter()
                    .map(|&class| self.quotient_identity_classes_at_round(quotient, 0)[class])
                    .collect::<Vec<_>>();
                let mut outputs = SwappedOutputIds::new(
                    &self.output_pairs,
                    &self.output_pair_lookup,
                    left as usize,
                    right as usize,
                );
                let mut found = None;
                for round in 1..=stable_round {
                    let identity_previous =
                        self.quotient_identity_classes_at_round(quotient, round - 1);
                    let identity_next = self.quotient_identity_classes_at_round(quotient, round);
                    let identity_counts = &quotient.identity_class_counts_by_round[round];
                    let identity = &self.canonical_rounds[round];
                    let identity_raw_previous = &self.canonical_rounds[round - 1].classes;
                    let mut local = FxHashMap::<CanonicalSignature, u32>::default();
                    let local_base = identity.representative_by_class.len() as u32;
                    let mut swapped_next_cone = Vec::with_capacity(cone_size);
                    for &source_class in &cone_classes {
                        let signature = self.canonical_sparse_quotient_swapped_signature(
                            quotient,
                            source_class,
                            identity_previous,
                            &cone_index_by_class,
                            &swapped_previous_cone,
                            &mut outputs,
                        );
                        let swapped_class = if let Some(identity_class) = self
                            .canonical_round_identity_class_for_signature(
                                identity,
                                identity_raw_previous,
                                &signature,
                            )
                        {
                            identity_class
                        } else {
                                let next = local_base + local.len() as u32;
                                *local.entry(signature).or_insert(next)
                            };
                        swapped_next_cone.push(swapped_class);
                    }
                    if !self.sparse_quotient_rooted_class_set_still_possible(
                        quotient,
                        identity_next,
                        identity_counts,
                        &swapped_next_cone,
                        &cone_classes,
                        &cone_index_by_class,
                    ) {
                        return None;
                    }
                    // Materialization uses stable identity class labels, so
                    // accept only at the identity partition fixed point.
                    if round == stable_round
                        && self.sparse_quotient_pair_is_stable(
                            quotient,
                            &swapped_previous_cone,
                            &swapped_next_cone,
                        )
                    {
                        let map_started_at = profile_timing.then(Instant::now);
                        found = self
                            .sparse_quotient_interchange_map_from_cone(
                                quotient,
                                &cone_classes,
                                &swapped_next_cone,
                                &cone_index_by_class,
                            )
                            .map(|map| (map, cone_size));
                        if let Some(started_at) = map_started_at {
                            self.sparse_quotient_map_ns += started_at.elapsed().as_nanos() as u64;
                        }
                        break;
                    }
                    swapped_previous_cone = swapped_next_cone;
                }
                found
            }
        };
        if profile_timing {
            if let Some(started_at) = pair_started_at {
                self.sparse_quotient_refinement_ns += started_at.elapsed().as_nanos() as u64;
            }
        }
        if let Some((map, cone_size)) = result {
            self.sparse_quotient_certified += 1;
            self.sparse_quotient_cone_classes_total += cone_size;
            self.sparse_quotient_cone_classes_max =
                self.sparse_quotient_cone_classes_max.max(cone_size);
            return Some(map);
        }
        None
    }

    fn canonical_quotient_interchange_map(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<InterchangeMap> {
        self.canonical_sparse_quotient_interchange_map(left, right)
            .or_else(|| self.canonical_full_quotient_interchange_map(left, right))
    }

    fn canonical_full_quotient_interchange_map(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<InterchangeMap> {
        let stable_round = self.ensure_canonical_identity_stable_round();
        self.ensure_canonical_quotient();
        let quotient = self
            .canonical_quotient
            .as_ref()
            .expect("canonical quotient initialized");
        let mut swapped_previous = vec![0u32; quotient.representative_by_class.len()];
        let mut outputs = SwappedOutputIds::new(
            &self.output_pairs,
            &self.output_pair_lookup,
            left as usize,
            right as usize,
        );
        for round in 1..=stable_round {
            let identity_previous = self.quotient_identity_classes_at_round(quotient, round - 1);
            let identity_next = self.quotient_identity_classes_at_round(quotient, round);
            let identity = &self.canonical_rounds[round];
            let identity_raw_previous = &self.canonical_rounds[round - 1].classes;
            let mut local = FxHashMap::<CanonicalSignature, u32>::default();
            let local_base = identity.representative_by_class.len() as u32;
            let mut swapped_next = Vec::with_capacity(quotient.representative_by_class.len());
            for class in 0..quotient.representative_by_class.len() {
                let signature = self.canonical_quotient_swapped_signature(
                    quotient,
                    class,
                    &swapped_previous,
                    &mut outputs,
                );
                let class = if let Some(identity_class) = self
                    .canonical_round_identity_class_for_signature(
                        identity,
                        identity_raw_previous,
                        &signature,
                    )
                {
                    identity_class
                } else {
                    let next = local_base + local.len() as u32;
                    *local.entry(signature).or_insert(next)
                };
                swapped_next.push(class);
            }
            if !self.quotient_rooted_class_set_still_possible(
                quotient,
                &identity_next,
                &swapped_next,
            ) {
                return None;
            }
            if same_equality_partition_pair_u32(
                &identity_previous,
                &swapped_previous,
                &identity_next,
                &swapped_next,
            ) {
                let map = self.quotient_interchange_map_from_classes(quotient, &swapped_next)?;
                self.quotient_certified += 1;
                return Some(map);
            }
            swapped_previous = swapped_next;
        }
        None
    }

    fn canonical_swapped_round(
        &self,
        previous: &[u32],
        identity_previous: &[u32],
        identity: &CanonicalRound,
        outputs: &mut SwappedOutputIds<'_>,
    ) -> Vec<u32> {
        let mut local = FxHashMap::<CanonicalSignature, u32>::default();
        let local_base = identity.representative_by_class.len() as u32;
        let mut classes = Vec::with_capacity(self.state_count());
        for state in 0..self.state_count() {
            let signature = self.canonical_swapped_signature(state, previous, outputs);
            let class = if let Some(identity_class) = self
                .canonical_round_identity_class_for_signature(
                    identity,
                    identity_previous,
                    &signature,
                )
            {
                identity_class
            } else {
                let next = local_base + local.len() as u32;
                *local.entry(signature).or_insert(next)
            };
            classes.push(class);
        }
        classes
    }

    /// Exact necessary first-round condition for a terminal swap. Round one
    /// depends only on frozen destination outputs, so a swap can change only
    /// source rows with an enabled edge into a state mentioning either terminal.
    /// All other rows keep their cached identity class. This never accepts a
    /// pair by itself; it only rejects a pair whose first characterization
    /// partition cannot be rooted-bijective.
    fn canonical_round_one_still_possible(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> bool {
        self.canonical_round_one_still_possible_with_mapped(
            left,
            right,
            SmallVec::new(),
        )
    }

    fn canonical_round_one_still_possible_with_mapped(
        &mut self,
        left: TerminalID,
        right: TerminalID,
        mapped_output_ids: SmallVec<[(u32, u32); 16]>,
    ) -> bool {
        self.ensure_canonical_identity_round(1);
        if self.canonical_round_one_class_counts.is_none() {
            let class_count = self.canonical_rounds[1].representative_by_class.len();
            let mut counts = vec![0u32; class_count];
            for &class in &self.canonical_rounds[1].classes[..self.topology.real_state_count] {
                counts[class as usize] += 1;
            }
            self.canonical_round_one_class_counts = Some(counts);
        }

        self.canonical_round_one_source_mark_epoch =
            self.canonical_round_one_source_mark_epoch.wrapping_add(1);
        if self.canonical_round_one_source_mark_epoch == 0 {
            self.canonical_round_one_source_marks.fill(0);
            self.canonical_round_one_source_mark_epoch = 1;
        }
        let epoch = self.canonical_round_one_source_mark_epoch;

        // The left/representative-side affected sources are identical for every
        // member sharing this representative in the exact-verify loop, so gather
        // them once and reuse them (re-marked under the fresh epoch) thereafter.
        // The resulting affected-source SET is identical to gathering all four
        // destination lists together; iteration order differs (left sources
        // first) but the accept/reject result is order-independent.
        if self.canonical_round_one_left_terminal != Some(left) {
            self.canonical_round_one_left_sources.clear();
            for destinations in [
                &self.finalizer_states_by_terminal[left as usize],
                &self.future_finalizer_states_by_terminal[left as usize],
            ] {
                for &destination in destinations {
                    for &source in &self.reverse_predecessors[destination as usize] {
                        let source = source as usize;
                        if self.canonical_round_one_source_marks[source] != epoch {
                            self.canonical_round_one_source_marks[source] = epoch;
                            self.canonical_round_one_left_sources.push(source as u32);
                        }
                    }
                }
            }
            self.canonical_round_one_left_terminal = Some(left);
        } else {
            for &source in &self.canonical_round_one_left_sources {
                self.canonical_round_one_source_marks[source as usize] = epoch;
            }
        }

        self.canonical_round_one_affected_sources.clear();
        self.canonical_round_one_affected_sources
            .extend_from_slice(&self.canonical_round_one_left_sources);
        for destinations in [
            &self.finalizer_states_by_terminal[right as usize],
            &self.future_finalizer_states_by_terminal[right as usize],
        ] {
            for &destination in destinations {
                for &source in &self.reverse_predecessors[destination as usize] {
                    let source = source as usize;
                    if self.canonical_round_one_source_marks[source] != epoch {
                        self.canonical_round_one_source_marks[source] = epoch;
                        self.canonical_round_one_affected_sources.push(source as u32);
                    }
                }
            }
        }

        // Move the scratch list out while comparing exact cached signatures so
        // immutable borrows of the DFA do not conflict with scratch reuse.
        let affected_sources = std::mem::take(&mut self.canonical_round_one_affected_sources);
        let mut changed_by_identity_class =
            std::mem::take(&mut self.canonical_round_one_changed_scratch);
        changed_by_identity_class.clear();
        let mut added_identity_classes =
            std::mem::take(&mut self.canonical_round_one_added_scratch);
        added_identity_classes.clear();
        let identity = &self.canonical_rounds[1];
        let identity_counts = self
            .canonical_round_one_class_counts
            .as_ref()
            .expect("first-round counts initialized");
        let mut swapped_root_class = identity.classes[self.topology.initial_state];
        let mut outputs = SparseSwappedOutputIds::with_mapped(
            &self.output_pairs,
            &self.output_pair_lookup,
            left as usize,
            right as usize,
            mapped_output_ids,
        );
        for &source in &affected_sources {
            let source = source as usize;
            let identity_class = identity.classes[source];
            if let Some((_, changed)) = changed_by_identity_class
                .iter_mut()
                .find(|(class, _)| *class == identity_class)
            {
                *changed += 1;
            } else {
                changed_by_identity_class.push((identity_class, 1));
            }
            let signature = self.canonical_swapped_signature_sparse(
                source,
                &self.canonical_rounds[0].classes,
                &mut outputs,
            );
            let Some(swapped_class) = self.canonical_round_identity_class_for_signature(
                identity,
                &self.canonical_rounds[0].classes,
                &signature,
            ) else {
                self.canonical_round_one_affected_sources = affected_sources;
                self.canonical_round_one_changed_scratch = changed_by_identity_class;
                self.canonical_round_one_added_scratch = added_identity_classes;
                return false;
            };
            if !added_identity_classes.contains(&swapped_class) {
                added_identity_classes.push(swapped_class);
            }
            if source == self.topology.initial_state {
                swapped_root_class = swapped_class;
            }
        }
        self.canonical_round_one_affected_sources = affected_sources;

        let root_matches = swapped_root_class == identity.classes[self.topology.initial_state];
        let accepted = root_matches
            && changed_by_identity_class.iter().all(|&(class, changed)| {
                changed < identity_counts[class as usize]
                    || added_identity_classes.contains(&class)
            });
        self.canonical_round_one_changed_scratch = changed_by_identity_class;
        self.canonical_round_one_added_scratch = added_identity_classes;
        accepted
    }

    /// Exact paired refinement over the disjoint union of the identity and
    /// relabelled restricted scanners.
    ///
    /// Canonical round one already contains every frozen destination-output
    /// observation. After that, the synchronous characterization recurrence is
    /// ordinary deterministic partition refinement: a block may split only by
    /// the destination block reached for one enabled byte. Running the same
    /// byte-labelled Hopcroft refinement used by the support quotient on both
    /// tagged copies therefore computes the exact combined fixed point without
    /// propagating one byte of a long bounded chain per global Moore round.
    fn canonical_paired_hopcroft_interchange_map(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<InterchangeMap> {
        self.ensure_canonical_identity_round(1);
        let state_count = self.state_count();
        let combined_state_count = state_count * 2;
        let dead_state = self.dead_state();

        let identity_round_one = &self.canonical_rounds[1];
        let identity_seed = &self.canonical_rounds[0].classes;
        let mut outputs = SwappedOutputIds::new(
            &self.output_pairs,
            &self.output_pair_lookup,
            left as usize,
            right as usize,
        );
        let swapped_round_one = self.canonical_swapped_round(
            identity_seed,
            identity_seed,
            identity_round_one,
            &mut outputs,
        );
        drop(outputs);

        let mut class_for_state = Vec::with_capacity(combined_state_count);
        class_for_state.extend_from_slice(&identity_round_one.classes);
        class_for_state.extend_from_slice(&swapped_round_one);
        let initial_class_count = class_for_state
            .iter()
            .copied()
            .max()
            .map_or(0, |class| class as usize + 1);
        let mut members_by_class = vec![Vec::<u32>::new(); initial_class_count];
        for (state, &class) in class_for_state.iter().enumerate() {
            members_by_class[class as usize].push(state as u32);
        }
        let dead_class = class_for_state[dead_state] as usize;
        debug_assert_eq!(
            class_for_state[dead_state],
            class_for_state[state_count + dead_state],
            "identity and swapped synthetic dead states must share round-one output behavior",
        );

        // Flat reverse-edge CSR over the two tagged copies. Missing enabled-byte
        // transitions go to the copy-local synthetic dead state. As in the
        // identity Hopcroft pass, explicit non-dead preimages are sufficient:
        // for each byte the dead preimage is their complement.
        let mut reverse_counts = vec![0usize; combined_state_count];
        let mut edge_count = 0usize;
        for side in 0..2 {
            let base = side * state_count;
            for source in 0..state_count {
                for &(_, destination) in self.topology.edges_from(source) {
                    reverse_counts[base + destination as usize] += 1;
                    edge_count += 1;
                }
            }
        }
        let mut reverse_offsets = Vec::with_capacity(combined_state_count + 1);
        reverse_offsets.push(0usize);
        for &count in &reverse_counts {
            reverse_offsets.push(reverse_offsets.last().copied().unwrap() + count);
        }
        let mut next_offset = reverse_offsets[..combined_state_count].to_vec();
        let mut reverse_edges = vec![(0u8, 0u32); edge_count];
        for side in 0..2 {
            let base = side * state_count;
            for source in 0..state_count {
                for &(byte, destination) in self.topology.edges_from(source) {
                    let destination = base + destination as usize;
                    let offset = &mut next_offset[destination];
                    reverse_edges[*offset] = (byte, (base + source) as u32);
                    *offset += 1;
                }
            }
        }

        let mut queue = VecDeque::<usize>::new();
        let mut queued = vec![false; initial_class_count];
        for class in 0..initial_class_count {
            if class != dead_class {
                queue.push_back(class);
                queued[class] = true;
            }
        }
        let mut incoming_by_byte = (0..256)
            .map(|_| Vec::<u32>::new())
            .collect::<Vec<_>>();
        let mut touched_bytes = Vec::<u8>::new();
        let mut source_marks = vec![0u32; combined_state_count];
        let mut mark_epoch = 0u32;
        let mut class_counts = vec![0usize; initial_class_count];
        let mut touched_classes = Vec::<usize>::new();

        while let Some(splitter) = queue.pop_front() {
            queued[splitter] = false;
            if members_by_class[splitter].is_empty() {
                continue;
            }
            touched_bytes.clear();
            for &destination in &members_by_class[splitter] {
                let destination = destination as usize;
                for &(byte, source) in
                    &reverse_edges[reverse_offsets[destination]..reverse_offsets[destination + 1]]
                {
                    let bucket = &mut incoming_by_byte[byte as usize];
                    if bucket.is_empty() {
                        touched_bytes.push(byte);
                    }
                    bucket.push(source);
                }
            }

            for &byte in &touched_bytes {
                let sources = &incoming_by_byte[byte as usize];
                mark_epoch = mark_epoch.wrapping_add(1);
                if mark_epoch == 0 {
                    source_marks.fill(0);
                    mark_epoch = 1;
                }
                touched_classes.clear();
                for &source in sources {
                    let source = source as usize;
                    if source_marks[source] == mark_epoch {
                        continue;
                    }
                    source_marks[source] = mark_epoch;
                    let class = class_for_state[source] as usize;
                    if class_counts[class] == 0 {
                        touched_classes.push(class);
                    }
                    class_counts[class] += 1;
                }

                for &class in &touched_classes {
                    let marked_count = class_counts[class];
                    class_counts[class] = 0;
                    let class_len = members_by_class[class].len();
                    if marked_count == 0 || marked_count == class_len {
                        continue;
                    }
                    let old_members = std::mem::take(&mut members_by_class[class]);
                    let mut marked = Vec::with_capacity(marked_count);
                    let mut unmarked = Vec::with_capacity(class_len - marked_count);
                    for state in old_members {
                        if source_marks[state as usize] == mark_epoch {
                            marked.push(state);
                        } else {
                            unmarked.push(state);
                        }
                    }

                    let contains_dead_marked = marked.binary_search(&(dead_state as u32)).is_ok();
                    let contains_dead_unmarked =
                        unmarked.binary_search(&(dead_state as u32)).is_ok();
                    let was_queued = queued[class];
                    let (retained, moved) = if contains_dead_marked {
                        (marked, unmarked)
                    } else if contains_dead_unmarked {
                        (unmarked, marked)
                    } else if was_queued || marked.len() >= unmarked.len() {
                        (marked, unmarked)
                    } else {
                        (unmarked, marked)
                    };
                    members_by_class[class] = retained;
                    for &state in &members_by_class[class] {
                        class_for_state[state as usize] = class as u32;
                    }
                    let new_class = members_by_class.len();
                    for &state in &moved {
                        class_for_state[state as usize] = new_class as u32;
                    }
                    members_by_class.push(moved);
                    queued.push(false);
                    class_counts.push(0);

                    if was_queued {
                        if new_class != dead_class {
                            queued[new_class] = true;
                            queue.push_back(new_class);
                        }
                    } else if class == dead_class {
                        queued[new_class] = true;
                        queue.push_back(new_class);
                    } else {
                        queued[new_class] = true;
                        queue.push_back(new_class);
                    }
                }
                incoming_by_byte[byte as usize].clear();
            }
        }

        let mut canonical_for_class = vec![u32::MAX; members_by_class.len()];
        let mut canonical_classes = Vec::with_capacity(combined_state_count);
        let mut next_canonical = 0u32;
        for state in 0..combined_state_count {
            let class = class_for_state[state] as usize;
            let canonical = if canonical_for_class[class] == u32::MAX {
                let canonical = next_canonical;
                next_canonical += 1;
                canonical_for_class[class] = canonical;
                canonical
            } else {
                canonical_for_class[class]
            };
            canonical_classes.push(canonical);
        }
        debug_assert_eq!(
            canonical_classes[dead_state],
            canonical_classes[state_count + dead_state],
            "paired Hopcroft refinement split equivalent synthetic dead states",
        );

        let (identity_classes, swapped_classes) = canonical_classes.split_at(state_count);
        self.interchange_map_from_classes(identity_classes, swapped_classes)
    }

    /// Collision-free exact refinement. The paired Hopcroft pass computes the
    /// same shared identity/relabelled characterization fixed point as the
    /// synchronous Moore reference, but avoids a depth-proportional number of
    /// full-state rounds on long bounded lexer chains.
    fn canonical_interchange_map(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<InterchangeMap> {
        self.canonical_paired_hopcroft_interchange_map(left, right)
    }

    fn canonical_identity_map_via_moore_history(&mut self) -> InterchangeMap {
        self.ensure_canonical_identity_stable_round();
        self.ensure_canonical_quotient();
        let quotient = self
            .canonical_quotient
            .as_ref()
            .expect("canonical quotient initialized");
        self.quotient_identity_map(quotient)
    }

    fn canonical_identity_map(&mut self) -> InterchangeMap {
        if let Some(map) = &self.canonical_identity_map {
            return map.clone();
        }
        let use_support_identity = std::env::var("GLRMASK_TI_IDENTITY_FROM_SUPPORT_QUOTIENT")
            .map(|value| {
                let value = value.trim();
                value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
        let map = if use_support_identity && self.support_partition_seed.is_none() {
            self.ensure_support_quotient();
            let support_map = self.identity_map_from_unseeded_support_quotient();
            if std::env::var_os("GLRMASK_TI_VALIDATE_SUPPORT_IDENTITY_MAP").is_some() {
                let reference = self.canonical_identity_map_via_moore_history();
                assert_eq!(
                    support_map.scanner_state_map.materialized(),
                    reference.scanner_state_map.materialized(),
                    "support-quotient identity transport diverged from canonical Moore identity transport",
                );
            }
            support_map
        } else {
            self.canonical_identity_map_via_moore_history()
        };
        self.canonical_identity_map = Some(map.clone());
        map
    }

    fn interchange_map_from_classes(
        &self,
        identity_classes: &[u32],
        swapped_classes: &[u32],
    ) -> Option<InterchangeMap> {
        debug_assert_eq!(identity_classes.len(), self.state_count());
        debug_assert_eq!(swapped_classes.len(), self.state_count());
        if identity_classes[self.topology.initial_state]
            != swapped_classes[self.topology.initial_state]
        {
            return None;
        }

        let mut target_representative_by_class = BTreeMap::<u32, u32>::new();
        let mut source_classes = BTreeMap::<u32, ()>::new();
        for state in 0..self.topology.real_state_count {
            source_classes.insert(identity_classes[state], ());
            match target_representative_by_class.entry(swapped_classes[state]) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(state as u32);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if !self.topology.state_has_raw_source(*entry.get() as usize)
                        && self.topology.state_has_raw_source(state)
                    {
                        *entry.get_mut() = state as u32;
                    }
                }
            }
        }
        if source_classes.len() != target_representative_by_class.len()
            || source_classes
                .keys()
                .any(|class| !target_representative_by_class.contains_key(class))
        {
            return None;
        }

        let target_representative_for_source_state = (0..self.topology.real_state_count)
            .map(|source| target_representative_by_class.get(&identity_classes[source]).copied())
            .collect::<Option<Vec<_>>>()?;
        Some(InterchangeMap {
            scanner_state_map: self
                .topology
                .scanner_map_from_state_targets(&target_representative_for_source_state)?,
        })
    }

    /// The set of output pairs visible on enabled-byte destinations is closed
    /// under every valid interchange. This filter is exact as a rejection
    /// condition: it does not accept a pair, it only avoids an impossible full
    /// characterization.
    fn observed_output_pair_set_is_swap_closed(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> bool {
        let mut mapped_output_ids = SmallVec::new();
        self.observed_output_pair_set_is_swap_closed_with_mapping(
            left,
            right,
            &mut mapped_output_ids,
        )
    }

    fn observed_output_pair_set_is_swap_closed_with_mapping(
        &mut self,
        left: TerminalID,
        right: TerminalID,
        mapped_output_ids: &mut SmallVec<[(u32, u32); 16]>,
    ) -> bool {
        mapped_output_ids.clear();
        let left_terminal = left as usize;
        let right_terminal = right as usize;
        self.observed_output_pair_mark_epoch = self.observed_output_pair_mark_epoch.wrapping_add(1);
        if self.observed_output_pair_mark_epoch == 0 {
            self.observed_output_pair_marks.fill(0);
            self.observed_output_pair_mark_epoch = 1;
        }
        let epoch = self.observed_output_pair_mark_epoch;
        let swap = Some((
            left as usize,
            right as usize,
        ));
        for ids in [
            &self.observed_output_pair_ids_by_terminal[left_terminal],
            &self.observed_output_pair_ids_by_terminal[right_terminal],
        ] {
            for &id in ids {
                let id = id as usize;
                if self.observed_output_pair_marks[id] == epoch {
                    continue;
                }
                self.observed_output_pair_marks[id] = epoch;
                let pair = &self.observed_output_pairs[id];
                // If neither component changes under the swap, the pair maps to
                // itself. Since it is an observed pair it is trivially in the
                // lookup, so skip the allocation and hash lookup entirely.
                let finalizers_change =
                    pair.finalizers.swap_changes(left as TerminalID, right as TerminalID);
                let future_change = pair
                    .future_finalizers
                    .swap_changes(left as TerminalID, right as TerminalID);
                if !finalizers_change && !future_change {
                    continue;
                }
                let swapped = OutputPair {
                    finalizers: if finalizers_change {
                        pair.finalizers.mapped(swap)
                    } else {
                        pair.finalizers.clone()
                    },
                    future_finalizers: if future_change {
                        pair.future_finalizers.mapped(swap)
                    } else {
                        pair.future_finalizers.clone()
                    },
                };
                let Some(&mapped_id) = self.output_pair_lookup.get(&swapped) else {
                    return false;
                };
                if !self
                    .observed_output_pair_is_observed
                    .get(mapped_id as usize)
                    .copied()
                    .unwrap_or(false)
                {
                    return false;
                }
                let canonical_id = self.observed_output_pair_canonical_ids[id];
                mapped_output_ids.push((canonical_id, mapped_id));
            }
        }
        true
    }

    fn swap_preserves_all_frozen_outputs(&self, left: TerminalID, right: TerminalID) -> bool {
        let left = left as usize;
        let right = right as usize;
        self.finalizer_states_by_terminal[left] == self.finalizer_states_by_terminal[right]
            && self.future_finalizer_states_by_terminal[left]
                == self.future_finalizer_states_by_terminal[right]
    }

    // ----- Parallel per-pair certification path -------------------------------
    //
    // The methods below are `&self` mirrors of the hot per-pair checks, taking a
    // per-worker `CertScratch` instead of the DFA's own scratch fields and
    // profiling counters. They rely on every lazy build (canonical round 1,
    // class counts, support quotient, per-terminal supports, canonical identity
    // map) having been forced before the loop by `prebuild_certification_state`,
    // so they perform no `&mut self` mutation and can run concurrently over an
    // immutable `&InterchangeabilityDfa`.

    /// Build the identity scanner transport directly from an *unseeded* exact
    /// stable support quotient.  In that case `support_identity_stable_partition`
    /// and the canonical Moore ladder solve the same deterministic-output
    /// bisimulation fixed point from the same round-one output partition. Both
    /// canonicalize classes by first state, so they choose the same scanner
    /// representatives as well as the same equivalence relation.
    ///
    /// A support quotient projected from a finer previous TI round is allowed to
    /// remain a strict refinement, which is sufficient for the support proof but
    /// is not the canonical current-round identity quotient. Never use this
    /// shortcut when `support_partition_seed` is present.
    fn identity_map_from_unseeded_support_quotient(&self) -> InterchangeMap {
        assert!(
            self.support_partition_seed.is_none(),
            "seeded support quotient is not necessarily the canonical current-round identity partition",
        );
        let quotient = self
            .support_quotient
            .as_ref()
            .expect("support quotient pre-built before identity transport");
        InterchangeMap {
            scanner_state_map: self
                .topology
                .scanner_map_from_internal_quotient(
                    Arc::clone(&quotient.class_for_state),
                    Arc::clone(&quotient.representative_by_class),
                    Box::default(),
                )
                .expect("exact support identity quotient must retain a raw representative"),
        }
    }

    /// Force every lazily built structure the certification path reads, so the
    /// `&self` variants never trigger a build. `candidates` is the flattened set
    /// of terminals that enter the certification loop.
    fn prebuild_certification_state(&mut self, candidates: &[TerminalID]) {
        self.ensure_canonical_identity_round(1);
        if self.canonical_round_one_class_counts.is_none() {
            let class_count = self.canonical_rounds[1].representative_by_class.len();
            let mut counts = vec![0u32; class_count];
            for &class in &self.canonical_rounds[1].classes[..self.topology.real_state_count] {
                counts[class as usize] += 1;
            }
            self.canonical_round_one_class_counts = Some(counts);
        }
        self.ensure_support_quotient();
        // Size the thread-safe on-demand support slots once; workers fill only
        // the terminals they actually reach at the support stage.
        let terminal_count = self.finalizer_states_by_terminal.len();
        let _ = self.parallel_terminal_supports.get_or_init(|| {
            (0..terminal_count).map(|_| OnceLock::new()).collect()
        });
        let _ = candidates;
        // Force the identity transport used by the frozen-preserve branch.
        let _ = self.canonical_identity_map();
    }

    /// Thread-safe, `&self` lazy build of one terminal's quotient output
    /// support. Idempotent and pure given the pre-built global support
    /// quotient, so racing workers converge on identical contents.
    fn terminal_support_parallel(&self, terminal: TerminalID) -> &[(u32, u8)] {
        if let Some(support) = self
            .terminal_quotient_output_supports
            .as_ref()
            .and_then(|supports| supports.get(terminal as usize))
            .and_then(Option::as_ref)
        {
            return support;
        }
        let slots = self
            .parallel_terminal_supports
            .get()
            .expect("parallel terminal supports sized before certification loop");
        slots[terminal as usize].get_or_init(|| {
            let class_for_state = &self
                .support_quotient
                .as_ref()
                .expect("support quotient pre-built")
                .class_for_state;
            let terminal = terminal as usize;
            let mut support = Vec::<(u32, u8)>::new();
            for (destinations, mask) in [
                (&self.finalizer_states_by_terminal[terminal], 1u8),
                (&self.future_finalizer_states_by_terminal[terminal], 2u8),
            ] {
                for &destination in destinations {
                    for &source in &self.reverse_predecessors[destination as usize] {
                        support.push((class_for_state[source as usize], mask));
                    }
                }
            }
            support.sort_unstable_by_key(|&(class, _)| class);
            let mut write = 0usize;
            for read in 0..support.len() {
                if write > 0 && support[write - 1].0 == support[read].0 {
                    support[write - 1].1 |= support[read].1;
                } else {
                    support[write] = support[read];
                    write += 1;
                }
            }
            support.truncate(write);
            support
        })
    }

    /// Deduplicated set of characterization source rows affected by swapping in
    /// the representative terminal. Computed once per representative and shared
    /// read-only across all its members in the parallel loop.
    fn representative_affected_sources(&self, representative: TerminalID, out: &mut Vec<u32>) {
        out.clear();
        for destinations in [
            &self.finalizer_states_by_terminal[representative as usize],
            &self.future_finalizer_states_by_terminal[representative as usize],
        ] {
            for &destination in destinations {
                for &source in &self.reverse_predecessors[destination as usize] {
                    out.push(source);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    /// `&self` mirror of `canonical_identity_map`, reading the pre-built cache.
    fn canonical_identity_map_prebuilt(&self) -> InterchangeMap {
        self.canonical_identity_map
            .as_ref()
            .expect("canonical identity map pre-built before certification loop")
            .clone()
    }

    /// `&self` + scratch mirror of `observed_output_pair_set_is_swap_closed`.
    fn observed_output_pair_set_is_swap_closed_scratch(
        &self,
        scratch: &mut CertScratch,
        left: TerminalID,
        right: TerminalID,
    ) -> bool {
        let left_terminal = left as usize;
        let right_terminal = right as usize;
        scratch.output_pair_mark_epoch = scratch.output_pair_mark_epoch.wrapping_add(1);
        if scratch.output_pair_mark_epoch == 0 {
            scratch.output_pair_marks.fill(0);
            scratch.output_pair_mark_epoch = 1;
        }
        let epoch = scratch.output_pair_mark_epoch;
        let swap = Some((left as usize, right as usize));
        for ids in [
            &self.observed_output_pair_ids_by_terminal[left_terminal],
            &self.observed_output_pair_ids_by_terminal[right_terminal],
        ] {
            for &id in ids {
                let id = id as usize;
                if scratch.output_pair_marks[id] == epoch {
                    continue;
                }
                scratch.output_pair_marks[id] = epoch;
                let pair = &self.observed_output_pairs[id];
                let finalizers_change =
                    pair.finalizers.swap_changes(left as TerminalID, right as TerminalID);
                let future_change = pair
                    .future_finalizers
                    .swap_changes(left as TerminalID, right as TerminalID);
                if !finalizers_change && !future_change {
                    continue;
                }
                let swapped = OutputPair {
                    finalizers: if finalizers_change {
                        pair.finalizers.mapped(swap)
                    } else {
                        pair.finalizers.clone()
                    },
                    future_finalizers: if future_change {
                        pair.future_finalizers.mapped(swap)
                    } else {
                        pair.future_finalizers.clone()
                    },
                };
                let Some(&mapped_id) = self.output_pair_lookup.get(&swapped) else {
                    return false;
                };
                if !self
                    .observed_output_pair_is_observed
                    .get(mapped_id as usize)
                    .copied()
                    .unwrap_or(false)
                {
                    return false;
                }
            }
        }
        true
    }

    /// `&self` + scratch mirror of `canonical_round_one_still_possible`. The
    /// representative-side affected sources are pre-computed and passed in as
    /// `left_sources` (shared read-only across all members of the pivot).
    fn canonical_round_one_still_possible_scratch(
        &self,
        scratch: &mut CertScratch,
        left: TerminalID,
        left_sources: &[u32],
        right: TerminalID,
    ) -> bool {
        scratch.source_mark_epoch = scratch.source_mark_epoch.wrapping_add(1);
        if scratch.source_mark_epoch == 0 {
            scratch.source_marks.fill(0);
            scratch.source_mark_epoch = 1;
        }
        let epoch = scratch.source_mark_epoch;

        scratch.affected_sources.clear();
        for &source in left_sources {
            scratch.source_marks[source as usize] = epoch;
            scratch.affected_sources.push(source);
        }
        for destinations in [
            &self.finalizer_states_by_terminal[right as usize],
            &self.future_finalizer_states_by_terminal[right as usize],
        ] {
            for &destination in destinations {
                for &source in &self.reverse_predecessors[destination as usize] {
                    let source = source as usize;
                    if scratch.source_marks[source] != epoch {
                        scratch.source_marks[source] = epoch;
                        scratch.affected_sources.push(source as u32);
                    }
                }
            }
        }

        let changed_by_identity_class = &mut scratch.changed_scratch;
        changed_by_identity_class.clear();
        let added_identity_classes = &mut scratch.added_scratch;
        added_identity_classes.clear();
        let identity = &self.canonical_rounds[1];
        let identity_counts = self
            .canonical_round_one_class_counts
            .as_ref()
            .expect("first-round counts pre-built");
        let mut swapped_root_class = identity.classes[self.topology.initial_state];
        let mut outputs = SparseSwappedOutputIds::new(
            &self.output_pairs,
            &self.output_pair_lookup,
            left as usize,
            right as usize,
        );
        for &source in &scratch.affected_sources {
            let source = source as usize;
            let identity_class = identity.classes[source];
            if let Some((_, changed)) = changed_by_identity_class
                .iter_mut()
                .find(|(class, _)| *class == identity_class)
            {
                *changed += 1;
            } else {
                changed_by_identity_class.push((identity_class, 1));
            }
            let signature = self.canonical_swapped_signature_sparse(
                source,
                &self.canonical_rounds[0].classes,
                &mut outputs,
            );
            let Some(swapped_class) = self.canonical_round_identity_class_for_signature(
                identity,
                &self.canonical_rounds[0].classes,
                &signature,
            ) else {
                return false;
            };
            if !added_identity_classes.contains(&swapped_class) {
                added_identity_classes.push(swapped_class);
            }
            if source == self.topology.initial_state {
                swapped_root_class = swapped_class;
            }
        }

        let root_matches = swapped_root_class == identity.classes[self.topology.initial_state];
        root_matches
            && changed_by_identity_class.iter().all(|&(class, changed)| {
                changed < identity_counts[class as usize]
                    || added_identity_classes.contains(&class)
            })
    }

    /// `&self` + scratch mirror of `support_transposition_interchange_map`.
    fn support_transposition_interchange_map_scratch(
        &self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<InterchangeMap> {
        if let Some(deviations) = self.support_transposition_deviations_scratch(left, right) {
            if let Some(map) = self.support_transposition_interchange_map_from_deviations_scratch(
                left,
                right,
                deviations,
            ) {
                return Some(map);
            }
        }
        let left_support = self.terminal_support_parallel(left);
        let right_support = self.terminal_support_parallel(right);
        let deviations = self.support_transposition_search_deviations_with_supports(
            left,
            right,
            left_support,
            right_support,
        )?;
        self.support_transposition_interchange_map_from_deviations_scratch(left, right, deviations)
    }

    /// `&self` mirror of `support_transposition_deviations` (reads pre-built
    /// per-terminal supports; performs no lazy build and records no timing).
    fn support_transposition_deviations_scratch(
        &self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<SupportDeviationMap> {
        let left_support = self.terminal_support_parallel(left);
        let right_support = self.terminal_support_parallel(right);
        Self::support_transposition_deviations_from_supports(left_support, right_support)
    }

    /// `&self` + scratch mirror of
    /// `support_transposition_interchange_map_from_deviations`.
    fn support_transposition_interchange_map_from_deviations_scratch(
        &self,
        left: TerminalID,
        right: TerminalID,
        deviations: SupportDeviationMap,
    ) -> Option<InterchangeMap> {
        let quotient = self
            .support_quotient
            .as_ref()
            .expect("support quotient pre-built");
        let left_support = self.terminal_support_parallel(left);
        let right_support = self.terminal_support_parallel(right);
        let cone_classes = Self::support_quotient_affected_cone_from_supports_small(
            quotient,
            left_support,
            right_support,
        );
        debug_assert!({
            let mut raw = self
                .support_quotient_affected_cone_small(quotient, left, right)
                .into_vec();
            let mut cached = cone_classes.clone().into_vec();
            raw.sort_unstable();
            cached.sort_unstable();
            raw == cached
        });
        debug_assert!(deviations.iter().all(|&(source, target)| {
            (source as usize) < quotient.representative_by_class.len()
                && (target as usize) < quotient.representative_by_class.len()
                && cone_classes.contains(&(source as usize))
                && cone_classes.contains(&(target as usize))
        }));
        let root_class = quotient.class_for_state[self.topology.initial_state] as usize;
        if Self::mapped_support_class(&deviations, root_class) != root_class as u32 {
            return None;
        }
        let mut outputs = SparseSwappedOutputIds::new(
            &self.output_pairs,
            &self.output_pair_lookup,
            left as usize,
            right as usize,
        );
        for &class in &cone_classes {
            let target = Self::mapped_support_class(&deviations, class);
            let source_state = quotient.representative_by_class[class] as usize;
            let target_state = quotient.representative_by_class[target as usize] as usize;
            let source_edges = self.topology.edges_from(source_state);
            let target_edges = self.topology.edges_from(target_state);
            let mut source_index = 0usize;
            let mut target_index = 0usize;
            while source_index < source_edges.len() || target_index < target_edges.len() {
                let (_byte, source_destination, target_destination) = match (
                    source_edges.get(source_index),
                    target_edges.get(target_index),
                ) {
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, target_destination)))
                        if source_byte == target_byte =>
                    {
                        source_index += 1;
                        target_index += 1;
                        (source_byte, source_destination as usize, target_destination as usize)
                    }
                    (Some(&(source_byte, source_destination)), Some(&(target_byte, _)))
                        if source_byte < target_byte =>
                    {
                        source_index += 1;
                        (source_byte, source_destination as usize, self.dead_state())
                    }
                    (Some(_), Some(&(target_byte, target_destination))) => {
                        target_index += 1;
                        (target_byte, self.dead_state(), target_destination as usize)
                    }
                    (Some(&(source_byte, source_destination)), None) => {
                        source_index += 1;
                        (source_byte, source_destination as usize, self.dead_state())
                    }
                    (None, Some(&(target_byte, target_destination))) => {
                        target_index += 1;
                        (target_byte, self.dead_state(), target_destination as usize)
                    }
                    (None, None) => unreachable!("sparse edge union loop is nonempty"),
                };
                let expected_destination = Self::mapped_support_class(
                    &deviations,
                    quotient.class_for_state[source_destination] as usize,
                );
                if quotient.class_for_state[target_destination] != expected_destination
                    || outputs.id(self.output_pair_by_state[source_destination])
                        != self.output_pair_by_state[target_destination]
                {
                    return None;
                }
            }
        }
        Some(InterchangeMap {
            scanner_state_map: TransportScannerStateMap::Quotient {
                state_count: quotient.scanner_state_count,
                class_for_original: Arc::clone(&quotient.scanner_class_for_original),
                representative_for_class: Arc::clone(&quotient.scanner_representative_for_class),
                source_class_for_target_deviations: deviations.into_vec().into_boxed_slice(),
            },
        })
    }

    /// Certify one member terminal against a fixed representative using only
    /// `&self` reads and per-worker scratch. `left_sources` is the pre-computed
    /// representative-side affected source set. `memo_snapshot` is an immutable
    /// snapshot of the cross-round first-round memo; misses are recorded in
    /// `scratch.memo_updates` for a post-loop merge.
    fn certify_member(
        &self,
        scratch: &mut CertScratch,
        representative: TerminalID,
        left_sources: &[u32],
        terminal: TerminalID,
        memo_snapshot: &FxHashMap<(TerminalID, TerminalID), bool>,
        closed_snapshot: &FxHashSet<(TerminalID, TerminalID)>,
    ) -> MemberResult {
        let memo_key = if representative <= terminal {
            (representative, terminal)
        } else {
            (terminal, representative)
        };
        let used_cached_closed = closed_snapshot.contains(&memo_key);
        let outpair_ok = used_cached_closed
            || self.observed_output_pair_set_is_swap_closed_scratch(
                scratch,
                representative,
                terminal,
            );
        if !outpair_ok {
            return MemberResult {
                verdict: MemberVerdict::RejectOutputPair,
                memo_update: None,
                closed_update: None,
                used_cached_closed,
            };
        }
        let closed_update = (!used_cached_closed).then_some(memo_key);
        let frozen_preserved = self.swap_preserves_all_frozen_outputs(representative, terminal);
        if frozen_preserved {
            return MemberResult {
                verdict: MemberVerdict::AcceptIdentity(self.canonical_identity_map_prebuilt()),
                memo_update: None,
                closed_update,
                used_cached_closed,
            };
        }
        if !self.exact_pair_fallback_enabled() {
            let support =
                self.support_transposition_interchange_map_scratch(representative, terminal);
            return MemberResult {
                verdict: match support {
                    Some(map) => MemberVerdict::AcceptSupport(map),
                    None => MemberVerdict::NeedsExact,
                },
                memo_update: None,
                closed_update,
                used_cached_closed,
            };
        }
        let mut memo_update = None;
        let first_round_possible = match memo_snapshot.get(&memo_key).copied() {
            Some(value) => value,
            None => {
                let value = self.canonical_round_one_still_possible_scratch(
                    scratch,
                    representative,
                    left_sources,
                    terminal,
                );
                memo_update = Some((memo_key, value));
                value
            }
        };
        if !first_round_possible {
            return MemberResult {
                verdict: MemberVerdict::RejectFirstRound,
                memo_update,
                closed_update,
                used_cached_closed,
            };
        }
        let support = self.support_transposition_interchange_map_scratch(representative, terminal);
        let verdict = match support {
            Some(map) => MemberVerdict::AcceptSupport(map),
            None => MemberVerdict::NeedsExact,
        };
        MemberResult {
            verdict,
            memo_update,
            closed_update,
            used_cached_closed,
        }
    }

    /// Pairwise exact refinement is a correctness-preserving *merge finder*,
    /// not a correctness requirement.  Every path before it is a positive
    /// certificate or a sound rejection filter; treating an uncertified pair
    /// as non-interchangeable can only miss an optimization.
    ///
    /// The raw/canonical pair refinement is intentionally retained for small
    /// discovery automata and for explicit reference runs.  On large lexer
    /// products its cost is quadratic in candidate pairs times tens of Moore
    /// rounds, while the support certificates already find the useful literal
    /// classes.  Keep production discovery bounded by a deterministic state
    /// threshold instead of a wall-clock budget.
    fn exact_pair_fallback_enabled(&self) -> bool {
        if let Ok(value) = std::env::var("GLRMASK_TI_EXACT_PAIR_FALLBACK") {
            let value = value.trim();
            return value.is_empty() || value == "1" || value.eq_ignore_ascii_case("true");
        }
        self.state_count() <= 1024
    }


    fn support_quotient_fallback_enabled(&self) -> bool {
        if let Ok(value) = std::env::var("GLRMASK_TI_SUPPORT_QUOTIENT_FALLBACK") {
            let value = value.trim();
            return value.is_empty() || value == "1" || value.eq_ignore_ascii_case("true");
        }
        let max_states = std::env::var("GLRMASK_TI_SUPPORT_QUOTIENT_MAX_STATES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(16_384);
        let max_edges = std::env::var("GLRMASK_TI_SUPPORT_QUOTIENT_MAX_EDGES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(300_000);
        self.state_count() <= max_states && self.topology.edges.len() <= max_edges
    }

    fn characterize_pair(&mut self, left: TerminalID, right: TerminalID) -> PairCharacterization {
        let state_count = self.state_count();
        let swap = Some((left as usize, right as usize));
        let swapped_output_pairs = self
            .output_pairs
            .iter()
            .map(|pair| OutputPair {
                finalizers: pair.finalizers.mapped(swap),
                future_finalizers: pair.future_finalizers.mapped(swap),
            })
            .collect::<Vec<_>>();
        let mut swapped_previous = self.identity_rounds[0].clone();

        for rounds in 1..=state_count * 2 {
            self.ensure_identity_round(rounds);
            if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() && rounds % 256 == 0 {
                eprintln!(
                    "[glrmask/profile][terminal_interchangeability] exact_pair={}<> {} rounds={} identity_rounds={}",
                    left,
                    right,
                    rounds,
                    self.identity_rounds.len(),
                );
            }
            let swapped_next =
                self.characterize_round(&swapped_previous, &swapped_output_pairs);
            if !rooted_class_bijection_still_possible(
                &self.identity_rounds[rounds],
                &swapped_next,
                self.topology.initial_state,
                self.topology.real_state_count,
            ) {
                return PairCharacterization {
                    identity_hashes: self.identity_rounds[rounds].clone(),
                    swapped_hashes: swapped_next,
                };
            }
            if same_equality_partition_pair(
                &self.identity_rounds[rounds - 1],
                &swapped_previous,
                &self.identity_rounds[rounds],
                &swapped_next,
            ) {
                return PairCharacterization {
                    identity_hashes: self.identity_rounds[rounds].clone(),
                    swapped_hashes: swapped_next,
                };
            }
            swapped_previous = swapped_next;
        }
        panic!(
            "terminal interchangeability characterization did not stabilize within {} rounds",
            state_count * 2,
        );
    }

    fn interchange_map(&mut self, left: TerminalID, right: TerminalID) -> Option<InterchangeMap> {
        if self.swap_preserves_all_frozen_outputs(left, right) {
            return Some(self.canonical_identity_map());
        }
        self.canonical_interchange_map(left, right)
    }

    fn reference_interchange_map(
        &mut self,
        left: TerminalID,
        right: TerminalID,
    ) -> Option<InterchangeMap> {
        let characterization = self.characterize_pair(left, right);
        self.interchange_map_from_characterization(&characterization)
    }

    fn interchange_map_from_characterization(
        &self,
        characterization: &PairCharacterization,
    ) -> Option<InterchangeMap> {
        if characterization.identity_hashes[self.topology.initial_state]
            != characterization.swapped_hashes[self.topology.initial_state]
        {
            return None;
        }

        let mut source_classes = BTreeMap::<CharacterizationHash, ()>::new();
        let mut target_representative_by_class = BTreeMap::<CharacterizationHash, u32>::new();
        for state in 0..self.topology.real_state_count {
            source_classes.insert(characterization.identity_hashes[state], ());
            match target_representative_by_class.entry(characterization.swapped_hashes[state]) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(state as u32);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if !self.topology.state_has_raw_source(*entry.get() as usize)
                        && self.topology.state_has_raw_source(state)
                    {
                        *entry.get_mut() = state as u32;
                    }
                }
            }
        }
        if source_classes.len() != target_representative_by_class.len()
            || source_classes
                .keys()
                .any(|hash| !target_representative_by_class.contains_key(hash))
        {
            return None;
        }

        let target_representative_for_source_state = (0..self.topology.real_state_count)
            .map(|source| {
                target_representative_by_class
                    .get(&characterization.identity_hashes[source])
                    .copied()
            })
            .collect::<Option<Vec<_>>>()?;
        Some(InterchangeMap {
            scanner_state_map: self
                .topology
                .scanner_map_from_state_targets(&target_representative_for_source_state)?,
        })
    }
}

/// Equality of characterization digests represents a partition, not a required
/// fixed digest value. The partition is stable exactly when the two tagged sides
/// induce the same equivalence relation in consecutive rounds.
fn same_equality_partition_pair(
    identity_previous: &[CharacterizationHash],
    swapped_previous: &[CharacterizationHash],
    identity_next: &[CharacterizationHash],
    swapped_next: &[CharacterizationHash],
) -> bool {
    debug_assert_eq!(identity_previous.len(), swapped_previous.len());
    debug_assert_eq!(identity_previous.len(), identity_next.len());
    debug_assert_eq!(identity_previous.len(), swapped_next.len());
    let mut previous_to_next = FxHashMap::<CharacterizationHash, CharacterizationHash>::default();
    let mut next_to_previous = FxHashMap::<CharacterizationHash, CharacterizationHash>::default();
    for (&old, &new) in identity_previous
        .iter()
        .zip(identity_next)
        .chain(swapped_previous.iter().zip(swapped_next))
    {
        if previous_to_next
            .get(&old)
            .is_some_and(|&existing| existing != new)
            || next_to_previous
                .get(&new)
                .is_some_and(|&existing| existing != old)
        {
            return false;
        }
        previous_to_next.insert(old, new);
        next_to_previous.insert(new, old);
    }
    true
}

/// Equality of integer class ids denotes a partition within a single
/// refinement side. The concrete ids may change between rounds; only the
/// induced equality relation matters.
fn same_equality_partition_u32(previous: &[u32], next: &[u32]) -> bool {
    debug_assert_eq!(previous.len(), next.len());
    let mut previous_to_next = FxHashMap::<u32, u32>::default();
    let mut next_to_previous = FxHashMap::<u32, u32>::default();
    for (&old, &new) in previous.iter().zip(next) {
        if previous_to_next
            .get(&old)
            .is_some_and(|&existing| existing != new)
            || next_to_previous
                .get(&new)
                .is_some_and(|&existing| existing != old)
        {
            return false;
        }
        previous_to_next.insert(old, new);
        next_to_previous.insert(new, old);
    }
    true
}

/// Integer-class counterpart of `same_equality_partition_pair`. Class ids are
/// shared across the identity and relabelled sides exactly when their complete
/// sparse signatures are equal, so the combined refinement must induce one
/// coherent old-to-new partition map across both sides.
fn same_equality_partition_pair_u32(
    identity_previous: &[u32],
    swapped_previous: &[u32],
    identity_next: &[u32],
    swapped_next: &[u32],
) -> bool {
    debug_assert_eq!(identity_previous.len(), swapped_previous.len());
    debug_assert_eq!(identity_previous.len(), identity_next.len());
    debug_assert_eq!(identity_previous.len(), swapped_next.len());
    let mut previous_to_next = FxHashMap::<u32, u32>::default();
    let mut next_to_previous = FxHashMap::<u32, u32>::default();
    for (&old, &new) in identity_previous
        .iter()
        .zip(identity_next)
        .chain(swapped_previous.iter().zip(swapped_next))
    {
        if previous_to_next
            .get(&old)
            .is_some_and(|&existing| existing != new)
            || next_to_previous
                .get(&new)
                .is_some_and(|&existing| existing != old)
        {
            return false;
        }
        previous_to_next.insert(old, new);
        next_to_previous.insert(new, old);
    }
    true
}

/// Canonical class ids are shared with the identity side exactly when their
/// full sparse signatures are equal. A missing root match or class-set member
/// can never be restored by a later refinement because partitions only split.
fn rooted_class_set_still_possible_u32(
    identity: &[u32],
    swapped: &[u32],
    initial_state: usize,
    real_state_count: usize,
) -> bool {
    if identity[initial_state] != swapped[initial_state] {
        return false;
    }
    let mut identity_classes = FxHashSet::<u32>::default();
    let mut swapped_classes = FxHashSet::<u32>::default();
    identity_classes.extend(identity[..real_state_count].iter().copied());
    swapped_classes.extend(swapped[..real_state_count].iter().copied());
    identity_classes == swapped_classes
}

/// A valid eventual map needs its root class and every current left class to
/// have a matching right class. Since characterization starts from the common
/// seed and only refines, a failure here can never be repaired later.
fn rooted_class_bijection_still_possible(
    identity: &[CharacterizationHash],
    swapped: &[CharacterizationHash],
    initial_state: usize,
    real_state_count: usize,
) -> bool {
    if identity[initial_state] != swapped[initial_state] {
        return false;
    }
    let mut identity_classes = FxHashSet::<CharacterizationHash>::default();
    let mut swapped_classes = FxHashSet::<CharacterizationHash>::default();
    identity_classes.extend(identity[..real_state_count].iter().copied());
    swapped_classes.extend(swapped[..real_state_count].iter().copied());
    identity_classes == swapped_classes
}

pub fn discover_one_round(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    relevant_bytes: &[bool; 256],
    ignore_terminal: Option<TerminalID>,
) -> BTreeMap<TerminalID, BTreeSet<TerminalID>> {
    discover_one_round_with_transport_witnesses(
        tokenizer,
        active_terminals,
        relevant_bytes,
        ignore_terminal,
    )
    .partition
}

/// Discover one exact TI round and retain only the accepted binary witnesses
/// for immediate build-local transport reconstruction.  Each witness is
/// derived under this round's exact pre-merge active mask; no later final-mask
/// re-derivation is permitted.
pub fn discover_one_round_with_transport_witnesses(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    relevant_bytes: &[bool; 256],
    ignore_terminal: Option<TerminalID>,
) -> TiRoundTransportWitnesses {
    let context = TiDiscoveryContext::new(tokenizer, relevant_bytes);
    discover_one_round_with_transport_witnesses_in_context(
        tokenizer,
        active_terminals,
        &context,
        ignore_terminal,
    )
}

/// As `discover_one_round_with_transport_witnesses`, but evaluates the exact
/// TI relation over a total global scanner-state quotient.
pub fn discover_one_round_with_transport_witnesses_with_global_state_quotient(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    relevant_bytes: &[bool; 256],
    ignore_terminal: Option<TerminalID>,
    global_state_quotient: &GlobalScannerStateQuotient,
) -> TiRoundTransportWitnesses {
    let context = TiDiscoveryContext::new_with_global_state_quotient(
        tokenizer,
        relevant_bytes,
        global_state_quotient,
    );
    discover_one_round_with_transport_witnesses_in_context(
        tokenizer,
        active_terminals,
        &context,
        ignore_terminal,
    )
}

/// As `discover_one_round_with_transport_witnesses`, but reuses the immutable
/// topology/root-observation context across iterative historical masks.
pub fn discover_one_round_with_transport_witnesses_in_context(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    context: &TiDiscoveryContext,
    ignore_terminal: Option<TerminalID>,
) -> TiRoundTransportWitnesses {
    discover_one_round_with_transport_witnesses_in_context_permitted(
        tokenizer,
        active_terminals,
        context,
        ignore_terminal,
        None,
    )
}

/// Exact TI discovery restricted to caller-certified candidate families.
///
/// `permitted_groups` is only a *pair-discovery* restriction: every accepted
/// pair still passes the same full tokenizer automorphism/witness proof as the
/// unrestricted path.  Terminals from different groups are never considered
/// interchangeable.  On iterative rounds the groups are projected through the
/// current active representatives, so historical members cannot create a new
/// cross-family merge later.
pub fn discover_one_round_with_transport_witnesses_in_context_permitted(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
    context: &TiDiscoveryContext,
    ignore_terminal: Option<TerminalID>,
    permitted_groups: Option<&[Vec<TerminalID>]>,
) -> TiRoundTransportWitnesses {
        let candidates = active_terminals
            .iter()
            .enumerate()
            .filter_map(|(terminal, &active)| active.then_some(terminal as TerminalID))
            .filter(|&terminal| Some(terminal) != ignore_terminal)
            .collect::<Vec<_>>();
        if candidates.len() < 2 {
            context.refresh_output_projection_seed_if_present(active_terminals);
            return TiRoundTransportWitnesses::singleton(active_terminals);
        }

        {
            let mut previous_active = context.output_pair_closed_active.borrow_mut();
            let monotone_subset = previous_active.as_ref().is_some_and(|previous| {
                previous.len() == active_terminals.len()
                    && active_terminals
                        .iter()
                        .zip(previous)
                        .all(|(&current, &old)| !current || old)
            });
            if !monotone_subset {
                context.output_pair_closed_memo.borrow_mut().clear();
            }
            *previous_active = Some(active_terminals.to_vec());
        }

        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        let topology = Arc::clone(&context.topology);
        let topology_ms = 0.0;
        let topology_edge_count = topology.edges.len();
        let topology_max_outdegree = topology.max_outdegree;
        let topology_byte_count = topology.bytes.len();
        let candidate_filter_started_at = profile_timing.then(Instant::now);
        let root_candidate_groups = if let Some(permitted_groups) = permitted_groups {
            let candidate_set = candidates.iter().copied().collect::<BTreeSet<_>>();
            permitted_groups
                .iter()
                .flat_map(|group| {
                    let active = group
                        .iter()
                        .copied()
                        .filter(|terminal| candidate_set.contains(terminal))
                        .collect::<Vec<_>>();
                    rooted_candidate_groups_from_signatures(
                        &active,
                        &context.root_output_signatures,
                    )
                })
                .collect::<Vec<_>>()
        } else {
            rooted_candidate_groups_from_signatures(
                &candidates,
                &context.root_output_signatures,
            )
        };
        let root_observed_states = context.root_observed_states;
        let root_candidate_pairs = root_candidate_groups
            .iter()
            .map(|group| group.len() * group.len().saturating_sub(1) / 2)
            .sum::<usize>();
        if root_candidate_pairs == 0 {
            context.refresh_output_projection_seed_if_present(active_terminals);
            return TiRoundTransportWitnesses::singleton(active_terminals);
        }

        // The old structural support pass is only a rejection prefilter. Once
        // the output-pair hypergraph refinement is in place it costs more than
        // the additional exact checks it avoids on p0/p1, so production goes
        // straight from rooted candidates to the exact-safe output filters.
        let structural_filter_started_at = profile_timing.then(Instant::now);
        let structural_candidate_groups = root_candidate_groups
            .into_iter()
            .filter(|group| group.len() >= 2)
            .collect::<Vec<_>>();
        let structural_color_count = 0usize;
        let structural_candidate_pairs = structural_candidate_groups
            .iter()
            .map(|group| group.len() * group.len().saturating_sub(1) / 2)
            .sum::<usize>();
        let structural_candidate_group_count = structural_candidate_groups.len();
        let structural_filter_ms = structural_filter_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        if structural_candidate_pairs == 0 {
            context.refresh_output_projection_seed_if_present(active_terminals);
            return TiRoundTransportWitnesses::singleton(active_terminals);
        }

        let dfa_setup_started_at = profile_timing.then(Instant::now);
        let mut dfa = InterchangeabilityDfa::from_context(active_terminals, context);
        let dfa_setup_ms = dfa_setup_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let output_shape_filter_started_at = profile_timing.then(Instant::now);
        let output_shape_candidate_groups = refine_candidate_groups_by_observed_output_pair_shape(
            structural_candidate_groups,
            &dfa.observed_output_pair_support_shapes_by_terminal,
        );
        let output_shape_filter_ms = output_shape_filter_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let output_hypergraph_filter_started_at = profile_timing.then(Instant::now);
        let (candidate_groups, output_hypergraph_rounds) =
            refine_candidate_groups_by_observed_output_hypergraph(
                output_shape_candidate_groups,
                &dfa.observed_output_pairs,
                active_terminals.len(),
            );
        let exact_candidate_pairs = candidate_groups
            .iter()
            .map(|group| group.len() * group.len().saturating_sub(1) / 2)
            .sum::<usize>();
        let diagnostic_candidate_groups = std::env::var_os("GLRMASK_DUMP_TI_CONE_COMPONENTS")
            .is_some()
            .then(|| candidate_groups.clone());
        let output_hypergraph_filter_ms = output_hypergraph_filter_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let candidate_filter_ms = candidate_filter_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        if profile_timing {
            let mut group_size_histogram = BTreeMap::<usize, usize>::new();
            for group in &candidate_groups {
                *group_size_histogram.entry(group.len()).or_default() += 1;
            }
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] active={} raw_states={} discovery_states={} selected_bytes={} sparse_edges={} max_outdegree={} root_observed_states={} root_candidate_pairs={} structural_colors={} structural_candidate_groups={} structural_candidate_pairs={} observed_output_candidate_groups={} output_hypergraph_rounds={} exact_candidate_pairs={} group_size_histogram={:?} topology_ms={:.3} candidate_filter_total_ms={:.3} structural_filter_ms={:.3} dfa_setup_ms={:.3} output_shape_filter_ms={:.3} output_hypergraph_filter_ms={:.3}",
                candidates.len(),
                topology.raw_state_count,
                topology.real_state_count,
                topology_byte_count,
                topology_edge_count,
                topology_max_outdegree,
                root_observed_states,
                root_candidate_pairs,
                structural_color_count,
                structural_candidate_group_count,
                structural_candidate_pairs,
                candidate_groups.len(),
                output_hypergraph_rounds,
                exact_candidate_pairs,
                group_size_histogram,
                topology_ms,
                candidate_filter_ms,
                structural_filter_ms,
                dfa_setup_ms,
                output_shape_filter_ms,
                output_hypergraph_filter_ms,
            );
        }
        if exact_candidate_pairs == 0 {
            context.refresh_output_projection_seed_if_present(active_terminals);
            return TiRoundTransportWitnesses::singleton(active_terminals);
        }
        let mut result = singleton_partition(active_terminals);
        let mut accepted_maps = BTreeMap::<(TerminalID, TerminalID), Arc<TransportScannerStateMap>>::new();
        let mut output_pair_rejections = 0usize;
        let mut output_pair_cached_closed = 0usize;
        let mut output_invariant_checks = 0usize;
        let mut first_round_rejections = 0usize;
        let mut support_transposition_checks = 0usize;
        let mut direct_exact_checks = 0usize;
        let mut exact_fallback_skips = 0usize;
        let mut accepted_representative_members = 0usize;
        let mut output_pair_filter_ns = 0u64;
        let mut frozen_output_ns = 0u64;
        let mut first_round_ns = 0u64;
        let mut support_transposition_ns = 0u64;
        let mut exact_map_ns = 0u64;
        let mut accepted_map_storage_ns = 0u64;

        if let Some(groups) = diagnostic_candidate_groups.as_ref() {
            dfa.ensure_support_quotient();
            let quotient = dfa
                .support_quotient
                .as_ref()
                .expect("support quotient initialized for candidate batch diagnostics");
            let mut certified_groups = 0usize;
            let mut certified_members = 0usize;
            let mut failed_sizes = Vec::new();
            for group in groups {
                if dfa.support_petal_batch_certificate(quotient, group).is_some() {
                    certified_groups += 1;
                    certified_members += group.len();
                } else {
                    failed_sizes.push(group.len());
                }
            }
            failed_sizes.sort_unstable_by(|a, b| b.cmp(a));
            eprintln!(
                "[glrmask/dump][ti_candidate_batches] groups={} members={} certified_groups={} certified_members={} failed_sizes={:?}",
                groups.len(),
                groups.iter().map(Vec::len).sum::<usize>(),
                certified_groups,
                certified_members,
                failed_sizes,
            );
        }

        let petal_batch_enabled = std::env::var("GLRMASK_TI_PETAL_BATCH")
            .ok()
            .map(|value| {
                let value = value.trim();
                value.is_empty() || value == "1" || value.eq_ignore_ascii_case("true")
            })
            .unwrap_or(true);
        let mut petal_batch_groups = 0usize;
        let mut petal_batch_members = 0usize;
        let mut petal_batch_grouping_ns = 0u64;
        let mut petal_batch_certificate_ns = 0u64;
        let mut petal_batch_map_ns = 0u64;
        // Literal-like terminal families commonly differ only inside a tiny
        // reverse-closed scanner cone. Prove those symmetries directly before
        // constructing a stable quotient of the entire discovery topology.
        // This path is exact: the common cone is fixed, the two disjoint petals
        // are exhaustively matched, and every selected-byte row/output is
        // verified under the terminal transposition.
        let raw_small_cone_max_states = std::env::var("GLRMASK_TI_RAW_SMALL_CONE_MAX_STATES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(8);
        let candidate_groups = if petal_batch_enabled
            && std::env::var_os("GLRMASK_DISABLE_TI_RAW_SMALL_BATCH").is_none()
        {
            let grouping_started_at = Instant::now();
            let mut cone_marks = Vec::<u32>::new();
            let mut cone_mark_epoch = 0u32;
            let mut identity_equivalence_memo = FxHashMap::<(u32, u32), bool>::default();
            let mut residual_groups = Vec::<Vec<TerminalID>>::new();

            for candidate_group in candidate_groups {
                let mut by_cone_size =
                    BTreeMap::<usize, Vec<(TerminalID, Vec<u32>)>>::new();
                let mut unresolved = Vec::<TerminalID>::new();
                for &terminal in &candidate_group {
                    if let Some(cone) = dfa.raw_terminal_cone_capped(
                        terminal,
                        raw_small_cone_max_states,
                        &mut cone_marks,
                        &mut cone_mark_epoch,
                    ) {
                        by_cone_size
                            .entry(cone.len())
                            .or_default()
                            .push((terminal, cone));
                    } else {
                        unresolved.push(terminal);
                    }
                }

                struct RawConeClass {
                    representative: TerminalID,
                    cone: Vec<u32>,
                    members: Vec<(TerminalID, InterchangeMap)>,
                }

                for bucket in by_cone_size.into_values() {
                    let mut classes = Vec::<RawConeClass>::new();
                    for (terminal, cone) in bucket {
                        let mut accepted = None::<(usize, InterchangeMap)>;
                        for (class_index, class) in classes.iter().enumerate() {
                            if let Some(map) = dfa.raw_small_cone_interchange_map(
                                class.representative,
                                terminal,
                                &class.cone,
                                &cone,
                                &mut identity_equivalence_memo,
                            ) {
                                accepted = Some((class_index, map));
                                break;
                            }
                        }
                        if let Some((class_index, map)) = accepted {
                            classes[class_index].members.push((terminal, map));
                        } else {
                            classes.push(RawConeClass {
                                representative: terminal,
                                cone,
                                members: Vec::new(),
                            });
                        }
                    }

                    for class in classes {
                        if class.members.is_empty() {
                            unresolved.push(class.representative);
                            continue;
                        }
                        let representative = class.representative;
                        petal_batch_groups += 1;
                        petal_batch_members += class.members.len();
                        support_transposition_checks += class.members.len();
                        accepted_representative_members += class.members.len();
                        for (terminal, map) in class.members {
                            result
                                .get_mut(&representative)
                                .expect("raw TI batch representative must remain present")
                                .insert(terminal);
                            let removed = result.remove(&terminal);
                            debug_assert!(
                                removed.is_some(),
                                "raw TI batch member must remain a singleton until merged",
                            );
                            let replaced = accepted_maps.insert(
                                (representative, terminal),
                                Arc::new(map.scanner_state_map()),
                            );
                            debug_assert!(replaced.is_none());
                        }
                    }
                }

                unresolved.sort_unstable();
                unresolved.dedup();
                if unresolved.len() >= 2 {
                    residual_groups.push(unresolved);
                }
            }
            petal_batch_grouping_ns += grouping_started_at.elapsed().as_nanos() as u64;
            residual_groups
        } else {
            candidate_groups
        };

        // The full stable support quotient is another optional merge finder.
        // On large discovery products it can cost far more than the downstream
        // optimization is worth. Exact raw-cone merges above are retained;
        // remaining candidates simply stay distinct when this structural work
        // bound is exceeded.
        let support_quotient_fallback_enabled = dfa.support_quotient_fallback_enabled();
        let support_quotient_skipped_candidates = if support_quotient_fallback_enabled {
            0usize
        } else {
            candidate_groups.iter().map(Vec::len).sum::<usize>()
        };
        let candidate_groups = if support_quotient_fallback_enabled {
            candidate_groups
        } else {
            Vec::new()
        };

        // Large rooted candidate groups are often already exact symmetric
        // petal families.  Try the whole-group certificate directly before
        // doing either per-terminal generic-form classification or pairwise
        // pivot certification.  This is especially important for generic NFA
        // topologies, where literal families can contain hundreds of terminals
        // but each terminal moves only a tiny, disjoint quotient petal.
        let direct_batch_min_terminals = std::env::var(
            "GLRMASK_TI_DIRECT_BATCH_MIN_TERMINALS",
        )
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(64);
        let candidate_groups = if petal_batch_enabled
            && candidate_groups
                .iter()
                .any(|group| group.len() >= direct_batch_min_terminals)
        {
            dfa.ensure_support_quotient();
            let quotient = dfa
                .support_quotient
                .as_ref()
                .expect("support quotient initialized for direct petal batching");
            let mut residual_groups = Vec::<Vec<TerminalID>>::new();
            for group in candidate_groups {
                if group.len() < direct_batch_min_terminals {
                    residual_groups.push(group);
                    continue;
                }
                let certificate_started_at = Instant::now();
                let Some(canonical_orders) =
                    dfa.support_petal_batch_certificate(quotient, &group)
                else {
                    petal_batch_certificate_ns +=
                        certificate_started_at.elapsed().as_nanos() as u64;
                    residual_groups.push(group);
                    continue;
                };
                petal_batch_certificate_ns +=
                    certificate_started_at.elapsed().as_nanos() as u64;
                let map_started_at = Instant::now();
                let Some(maps) = dfa.support_petal_batch_maps(
                    quotient,
                    &group,
                    &canonical_orders,
                ) else {
                    petal_batch_map_ns += map_started_at.elapsed().as_nanos() as u64;
                    residual_groups.push(group);
                    continue;
                };
                petal_batch_map_ns += map_started_at.elapsed().as_nanos() as u64;
                let representative = group[0];
                petal_batch_groups += 1;
                petal_batch_members += group.len() - 1;
                support_transposition_checks += group.len() - 1;
                accepted_representative_members += group.len() - 1;
                for (&terminal, map) in group[1..].iter().zip(maps) {
                    result
                        .get_mut(&representative)
                        .expect("TI direct batch representative must retain its singleton partition entry")
                        .insert(terminal);
                    let removed = result.remove(&terminal);
                    debug_assert!(
                        removed.is_some(),
                        "TI direct batch member must retain its singleton partition entry"
                    );
                    let replaced = accepted_maps.insert(
                        (representative, terminal),
                        Arc::new(map.scanner_state_map()),
                    );
                    debug_assert!(
                        replaced.is_none(),
                        "TI direct batch must accept each pair once"
                    );
                }
            }
            residual_groups
        } else {
            candidate_groups
        };
        let candidate_groups = if petal_batch_enabled
            && dfa.topology.raw_representative_by_state.is_none()
        {
            dfa.ensure_support_quotient();
            let quotient = dfa
                .support_quotient
                .as_ref()
                .expect("support quotient initialized for petal batching");
            let mut residual_groups = Vec::<Vec<TerminalID>>::new();
            let mut cone_by_terminal =
                vec![None::<SmallVec<[usize; 8]>>; active_terminals.len()];
            let mut order_by_terminal =
                vec![None::<SmallVec<[usize; 8]>>; active_terminals.len()];
            let parallel_petal_min_terminals = std::env::var(
                "GLRMASK_TI_PARALLEL_PETAL_MIN_TERMINALS",
            )
            .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(64);
            for candidate_group in candidate_groups {
                let grouping_started_at = Instant::now();
                let mut membership = vec![false; active_terminals.len()];
                for &terminal in &candidate_group {
                    membership[terminal as usize] = true;
                }
                let pair_stats = (candidate_group.len() >= 64)
                    .then(|| dfa.generic_output_pair_stats(&membership));
                let mut by_code =
                    BTreeMap::<SmallVec<[u64; 8]>, Vec<TerminalID>>::new();
                let classify_terminal = |terminal: TerminalID| {
                    let cone = dfa.support_quotient_terminal_cone(quotient, terminal);
                    let (code, order) = dfa.support_terminal_cone_generic_form(
                        quotient,
                        &cone,
                        terminal,
                        &membership,
                        pair_stats.as_deref(),
                    );
                    (terminal, cone, code, order)
                };
                let classified = if candidate_group.len() >= parallel_petal_min_terminals {
                    if let Some(classified) = with_ti_parallel_pool(|| {
                            candidate_group
                                .par_iter()
                                .copied()
                                .map(classify_terminal)
                                .collect::<Vec<_>>()
                        })
                    {
                        classified
                    } else {
                        candidate_group
                            .iter()
                            .copied()
                            .map(classify_terminal)
                            .collect::<Vec<_>>()
                    }
                } else {
                    candidate_group
                        .iter()
                        .copied()
                        .map(classify_terminal)
                        .collect::<Vec<_>>()
                };
                // `par_iter` over a slice is indexed, so collection preserves
                // the original terminal order. Apply cache updates and BTree
                // grouping sequentially to keep witness ordering byte-stable.
                for (terminal, cone, code, order) in classified {
                    cone_by_terminal[terminal as usize] = Some(cone);
                    order_by_terminal[terminal as usize] = order;
                    by_code.entry(code).or_default().push(terminal);
                }
                petal_batch_grouping_ns += grouping_started_at.elapsed().as_nanos() as u64;
                for group in by_code.into_values().filter(|group| group.len() >= 2) {
                    let certificate_started_at = Instant::now();
                    let group_cones = group
                        .iter()
                        .map(|&terminal| {
                            cone_by_terminal[terminal as usize]
                                .as_deref()
                                .expect("petal batch terminal cone cached during grouping")
                        })
                        .collect::<Vec<_>>();
                    let group_orders = group
                        .iter()
                        .map(|&terminal| {
                            order_by_terminal[terminal as usize]
                                .as_deref()
                                .ok_or(())
                        })
                        .collect::<Result<Vec<_>, _>>();
                    let Some(canonical_orders) = group_orders.ok().and_then(|group_orders| {
                        dfa.support_petal_batch_certificate_from_orders(
                            quotient,
                            &group,
                            &group_cones,
                            &group_orders,
                        )
                    }) else {
                        petal_batch_certificate_ns +=
                            certificate_started_at.elapsed().as_nanos() as u64;
                        residual_groups.push(group);
                        continue;
                    };
                    petal_batch_certificate_ns +=
                        certificate_started_at.elapsed().as_nanos() as u64;
                    let map_started_at = Instant::now();
                    let Some(maps) = dfa.support_petal_batch_maps(
                        quotient,
                        &group,
                        &canonical_orders,
                    ) else {
                        petal_batch_map_ns += map_started_at.elapsed().as_nanos() as u64;
                        residual_groups.push(group);
                        continue;
                    };
                    petal_batch_map_ns += map_started_at.elapsed().as_nanos() as u64;
                    let representative = group[0];
                    petal_batch_groups += 1;
                    petal_batch_members += group.len() - 1;
                    support_transposition_checks += group.len() - 1;
                    accepted_representative_members += group.len() - 1;
                    for (&terminal, map) in group[1..].iter().zip(maps) {
                        result
                            .get_mut(&representative)
                            .expect("TI batch representative must retain its singleton partition entry")
                            .insert(terminal);
                        let removed = result.remove(&terminal);
                        debug_assert!(
                            removed.is_some(),
                            "TI batch member must retain its singleton partition entry"
                        );
                        let replaced = accepted_maps.insert(
                            (representative, terminal),
                            Arc::new(map.scanner_state_map()),
                        );
                        debug_assert!(replaced.is_none(), "TI batch must accept each pair once");
                    }
                }
            }
            residual_groups
        } else {
            candidate_groups
        };

        let support_candidates = candidate_groups
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        dfa.prebuild_terminal_quotient_output_supports(&support_candidates);

        // Accepted terminal swaps are automorphisms. Therefore (a b) and
        // (b c) imply (a c) by conjugation, so interchangeability is an
        // equivalence relation. Partition each candidate group by pivots,
        // keeping only the representative-to-member maps transport requires.
        if let Some(pool) = ti_certification_pool() {
            // Parallel per-pair certification. For a fixed representative every
            // member of `unresolved[1..]` is tested independently against that
            // representative, so the verdicts fan out across `pool`; they are
            // then applied sequentially in the original member order, which
            // preserves the pivot sequence and hence the exact partition.
            let all_candidates: Vec<TerminalID> =
                candidate_groups.iter().flatten().copied().collect();
            dfa.prebuild_certification_state(&all_candidates);
            // Within a single discover-one-round call the same unordered pair is
            // never certified twice (a member accepted by a pivot is removed;
            // rejected members form fresh pairs against later pivots). The
            // first-round memo therefore only serves reuse *across* rounds, so we
            // snapshot it once here (immutable, read by all parallel workers) and
            // merge every new entry back in a single sweep after the round.
            let memo_snapshot = context.first_round_memo.borrow().clone();
            let closed_snapshot = context.output_pair_closed_memo.borrow().clone();
            let mut memo_additions: Vec<((TerminalID, TerminalID), bool)> = Vec::new();
            let mut closed_additions: Vec<(TerminalID, TerminalID)> = Vec::new();
            let mut rep_sources = Vec::<u32>::new();
            // Per-item certification work is only microseconds, so rayon's
            // fork/join and pool-wake latency dwarfs it for small pivots. Only
            // fan out when a pivot has enough members to amortize that overhead;
            // otherwise certify sequentially reusing a single scratch (which
            // matches the sequential path's cost exactly, no pool involved).
            let min_members = std::env::var("GLRMASK_TI_PARALLEL_MIN_MEMBERS")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(256);
            let mut seq_scratch = CertScratch::new(&dfa);
            // Enter the pool exactly ONCE for the whole pivot loop. Every nested
            // `par_iter` then runs on the already-hot pool, avoiding the ~2.7ms
            // cold pool-entry (worker wake) latency that a per-pivot
            // `pool.install` was paying (111 forks * 2.7ms = ~300ms on the p0
            // heavy round). `install` runs this closure on a pool worker and
            // blocks the caller, so the captured `&mut` state is used single
            // threaded here; only the `par_iter` bodies fan out.
            pool.install(|| {
            for initial_group in candidate_groups {
                let mut unresolved = initial_group;
                while !unresolved.is_empty() {
                    let representative = unresolved[0];
                    dfa.representative_affected_sources(representative, &mut rep_sources);
                    let member_count = unresolved.len() - 1;
                    let results: Vec<MemberResult> = if member_count >= min_members {
                        let dfa_ref = &dfa;
                        let rep_sources_ref = &rep_sources;
                        let memo_snapshot_ref = &memo_snapshot;
                        let closed_snapshot_ref = &closed_snapshot;
                        // NOTE: no per-pivot `pool.install` here. The whole pivot
                        // loop runs inside a single `pool.install` below, so these
                        // nested `par_iter`s reuse the already-hot pool instead of
                        // paying a cold pool-entry (worker wake) latency per pivot.
                        let forked: Vec<MemberResult> = unresolved[1..]
                            .par_iter()
                            .map_init(
                                || CertScratch::new(dfa_ref),
                                |scratch, &terminal| {
                                    dfa_ref.certify_member(
                                        scratch,
                                        representative,
                                        rep_sources_ref,
                                        terminal,
                                        memo_snapshot_ref,
                                        closed_snapshot_ref,
                                    )
                                },
                            )
                            .collect();
                        forked
                    } else {
                        unresolved[1..]
                            .iter()
                            .map(|&terminal| {
                                dfa.certify_member(
                                    &mut seq_scratch,
                                    representative,
                                    &rep_sources,
                                    terminal,
                                    &memo_snapshot,
                                    &closed_snapshot,
                                )
                            })
                            .collect()
                    };
                    for member_result in &results {
                        if let Some(update) = member_result.memo_update {
                            memo_additions.push(update);
                        }
                        if let Some(update) = member_result.closed_update {
                            closed_additions.push(update);
                        }
                        if member_result.used_cached_closed {
                            output_pair_cached_closed += 1;
                        }
                    }
                    let members: Vec<TerminalID> = unresolved[1..].to_vec();
                    let mut next_unresolved =
                        Vec::with_capacity(unresolved.len().saturating_sub(1));
                    let mut accepted_this_pivot = 0usize;
                    for (terminal, member_result) in members.into_iter().zip(results) {
                        let map = match member_result.verdict {
                            MemberVerdict::RejectOutputPair => {
                                output_pair_rejections += 1;
                                None
                            }
                            MemberVerdict::RejectFirstRound => {
                                first_round_rejections += 1;
                                None
                            }
                            MemberVerdict::AcceptIdentity(map) => {
                                output_invariant_checks += 1;
                                Some(map)
                            }
                            MemberVerdict::AcceptSupport(map) => {
                                support_transposition_checks += 1;
                                Some(map)
                            }
                            MemberVerdict::NeedsExact => {
                                support_transposition_checks += 1;
                                if dfa.exact_pair_fallback_enabled() {
                                    direct_exact_checks += 1;
                                    dfa.interchange_map(representative, terminal)
                                } else {
                                    exact_fallback_skips += 1;
                                    None
                                }
                            }
                        };
                        if let Some(map) = map {
                            accepted_this_pivot += 1;
                            accepted_representative_members += 1;
                            result
                                .get_mut(&representative)
                                .expect("TI representative must retain its singleton partition entry")
                                .insert(terminal);
                            let removed = result.remove(&terminal);
                            debug_assert!(removed.is_some(), "TI member must retain its singleton partition entry");
                            let replaced = accepted_maps.insert(
                                (representative, terminal),
                                Arc::new(map.scanner_state_map()),
                            );
                            debug_assert!(replaced.is_none(), "TI round must accept each pair once");
                        } else {
                            next_unresolved.push(terminal);
                        }
                    }

                    // A failed pivot in a moderate candidate group is the
                    // pathological case for sequential pivoting: if the group
                    // contains mostly singletons, we otherwise pay one
                    // fork/join (or one serial scan) per remaining pivot and
                    // eventually certify nearly every unordered pair. When the
                    // exact fallback is unavailable, every pair verdict is a
                    // pure independent support/output certificate. Evaluate all
                    // remaining pairs in one hot-pool batch, then replay the
                    // same pivot algorithm from the completed matrix. This
                    // preserves the exact partition and the exact
                    // representative-to-member maps that sequential pivoting
                    // would retain; it only changes scheduling.
                    let all_pairs_batch_min = std::env::var(
                        "GLRMASK_TI_ALL_PAIRS_BATCH_MIN",
                    )
                    .ok()
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(16);
                    let all_pairs_batch_max = std::env::var(
                        "GLRMASK_TI_ALL_PAIRS_BATCH_MAX",
                    )
                    .ok()
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(96);
                    if accepted_this_pivot == 0
                        && !dfa.exact_pair_fallback_enabled()
                        && (all_pairs_batch_min..=all_pairs_batch_max)
                            .contains(&next_unresolved.len())
                    {
                        let batch_members = next_unresolved;
                        let batch_len = batch_members.len();
                        let mut affected_sources_by_member =
                            Vec::<Vec<u32>>::with_capacity(batch_len);
                        for &terminal in &batch_members {
                            let mut sources = Vec::new();
                            dfa.representative_affected_sources(terminal, &mut sources);
                            affected_sources_by_member.push(sources);
                        }
                        let pair_indices = (0..batch_len)
                            .flat_map(|left| {
                                (left + 1..batch_len).map(move |right| (left, right))
                            })
                            .collect::<Vec<_>>();
                        let dfa_ref = &dfa;
                        let memo_snapshot_ref = &memo_snapshot;
                        let closed_snapshot_ref = &closed_snapshot;
                        let pair_results = pair_indices
                            .par_iter()
                            .map_init(
                                || CertScratch::new(dfa_ref),
                                |scratch, &(left, right)| {
                                    dfa_ref.certify_member(
                                        scratch,
                                        batch_members[left],
                                        &affected_sources_by_member[left],
                                        batch_members[right],
                                        memo_snapshot_ref,
                                        closed_snapshot_ref,
                                    )
                                },
                            )
                            .collect::<Vec<_>>();
                        let mut pair_maps = (0..batch_len * batch_len)
                            .map(|_| None::<InterchangeMap>)
                            .collect::<Vec<_>>();
                        for ((left, right), member_result) in
                            pair_indices.into_iter().zip(pair_results)
                        {
                            if let Some(update) = member_result.memo_update {
                                memo_additions.push(update);
                            }
                            if let Some(update) = member_result.closed_update {
                                closed_additions.push(update);
                            }
                            if member_result.used_cached_closed {
                                output_pair_cached_closed += 1;
                            }
                            let map = match member_result.verdict {
                                MemberVerdict::RejectOutputPair => {
                                    output_pair_rejections += 1;
                                    None
                                }
                                MemberVerdict::RejectFirstRound => {
                                    first_round_rejections += 1;
                                    None
                                }
                                MemberVerdict::AcceptIdentity(map) => {
                                    output_invariant_checks += 1;
                                    Some(map)
                                }
                                MemberVerdict::AcceptSupport(map) => {
                                    support_transposition_checks += 1;
                                    Some(map)
                                }
                                MemberVerdict::NeedsExact => {
                                    support_transposition_checks += 1;
                                    exact_fallback_skips += 1;
                                    None
                                }
                            };
                            pair_maps[left * batch_len + right] = map;
                        }

                        let mut remaining = (0..batch_len).collect::<Vec<_>>();
                        while !remaining.is_empty() {
                            let representative_index = remaining[0];
                            let representative = batch_members[representative_index];
                            let mut next_remaining =
                                Vec::with_capacity(remaining.len().saturating_sub(1));
                            for &member_index in &remaining[1..] {
                                let map = pair_maps
                                    [representative_index * batch_len + member_index]
                                    .take();
                                if let Some(map) = map {
                                    let terminal = batch_members[member_index];
                                    accepted_representative_members += 1;
                                    result
                                        .get_mut(&representative)
                                        .expect("TI batch replay representative must retain its singleton partition entry")
                                        .insert(terminal);
                                    let removed = result.remove(&terminal);
                                    debug_assert!(
                                        removed.is_some(),
                                        "TI batch replay member must retain its singleton partition entry"
                                    );
                                    let replaced = accepted_maps.insert(
                                        (representative, terminal),
                                        Arc::new(map.scanner_state_map()),
                                    );
                                    debug_assert!(
                                        replaced.is_none(),
                                        "TI batch replay must accept each retained pair once"
                                    );
                                } else {
                                    next_remaining.push(member_index);
                                }
                            }
                            remaining = next_remaining;
                        }
                        unresolved.clear();
                        continue;
                    }
                    unresolved = next_unresolved;
                }
            }
            });
            if !memo_additions.is_empty() {
                let mut memo = context.first_round_memo.borrow_mut();
                for (key, value) in memo_additions {
                    memo.insert(key, value);
                }
            }
            if !closed_additions.is_empty() {
                context
                    .output_pair_closed_memo
                    .borrow_mut()
                    .extend(closed_additions);
            }
        } else {
        // Sequential discovery owns this partition-local context for the whole
        // round. Hold the memo borrow once instead of entering the RefCell for
        // every candidate pair (a get plus a possible insert per miss).
        let mut first_round_memo = context.first_round_memo.borrow_mut();
        for initial_group in candidate_groups {
            let mut unresolved = initial_group;
            while !unresolved.is_empty() {
                let representative = unresolved[0];
                let mut next_unresolved = Vec::with_capacity(unresolved.len().saturating_sub(1));
                for &terminal in &unresolved[1..] {
                    let output_pair_started_at = profile_timing.then(Instant::now);
                    let memo_key = if representative <= terminal {
                        (representative, terminal)
                    } else {
                        (terminal, representative)
                    };
                    let mut mapped_output_ids = SmallVec::new();
                    let cached_closed = context
                        .output_pair_closed_memo
                        .borrow()
                        .contains(&memo_key);
                    let output_pair_is_closed = if cached_closed {
                        output_pair_cached_closed += 1;
                        true
                    } else {
                        dfa.observed_output_pair_set_is_swap_closed_with_mapping(
                            representative,
                            terminal,
                            &mut mapped_output_ids,
                        )
                    };
                    if let Some(started_at) = output_pair_started_at {
                        output_pair_filter_ns += started_at.elapsed().as_nanos() as u64;
                    }
                    if !output_pair_is_closed {
                        output_pair_rejections += 1;
                        next_unresolved.push(terminal);
                        continue;
                    }
                    if !cached_closed {
                        context.output_pair_closed_memo.borrow_mut().insert(memo_key);
                    }
                    let frozen_output_started_at = profile_timing.then(Instant::now);
                    let preserves_frozen_outputs =
                        dfa.swap_preserves_all_frozen_outputs(representative, terminal);
                    if let Some(started_at) = frozen_output_started_at {
                        frozen_output_ns += started_at.elapsed().as_nanos() as u64;
                    }
                    let map = if preserves_frozen_outputs {
                        output_invariant_checks += 1;
                        Some(dfa.canonical_identity_map())
                    } else if !dfa.exact_pair_fallback_enabled() {
                        support_transposition_checks += 1;
                        let support_transposition_started_at = profile_timing.then(Instant::now);
                        let support_map =
                            dfa.support_transposition_interchange_map(representative, terminal);
                        if let Some(started_at) = support_transposition_started_at {
                            support_transposition_ns += started_at.elapsed().as_nanos() as u64;
                        }
                        if support_map.is_none() {
                            exact_fallback_skips += 1;
                        }
                        support_map
                    } else {
                        let first_round_started_at = profile_timing.then(Instant::now);
                        let first_round_possible = match first_round_memo.get(&memo_key).copied() {
                            Some(value) => value,
                            None => {
                                let value = dfa
                                    .canonical_round_one_still_possible_with_mapped(
                                        representative,
                                        terminal,
                                        mapped_output_ids,
                                    );
                                first_round_memo.insert(memo_key, value);
                                value
                            }
                        };
                        if let Some(started_at) = first_round_started_at {
                            first_round_ns += started_at.elapsed().as_nanos() as u64;
                        }
                        if !first_round_possible {
                            first_round_rejections += 1;
                            None
                        } else {
                            support_transposition_checks += 1;
                            let support_transposition_started_at = profile_timing.then(Instant::now);
                            let support_map = dfa
                                .support_transposition_interchange_map(representative, terminal);
                            if let Some(started_at) = support_transposition_started_at {
                                support_transposition_ns += started_at.elapsed().as_nanos() as u64;
                            }
                            if let Some(support_map) = support_map {
                                Some(support_map)
                            } else if !dfa.exact_pair_fallback_enabled() {
                                exact_fallback_skips += 1;
                                None
                            } else {
                                direct_exact_checks += 1;
                                let exact_map_started_at = profile_timing.then(Instant::now);
                                let map = dfa.interchange_map(representative, terminal);
                                if let Some(started_at) = exact_map_started_at {
                                    exact_map_ns += started_at.elapsed().as_nanos() as u64;
                                }
                                map
                            }
                        }
                    };
                    if let Some(map) = map {
                        let storage_started_at = profile_timing.then(Instant::now);
                        accepted_representative_members += 1;
                        result
                            .get_mut(&representative)
                            .expect("TI representative must retain its singleton partition entry")
                            .insert(terminal);
                        let removed = result.remove(&terminal);
                        debug_assert!(removed.is_some(), "TI member must retain its singleton partition entry");
                        let replaced = accepted_maps.insert(
                            (representative, terminal),
                            Arc::new(map.scanner_state_map()),
                        );
                        debug_assert!(replaced.is_none(), "TI round must accept each pair once");
                        if let Some(started_at) = storage_started_at {
                            accepted_map_storage_ns += started_at.elapsed().as_nanos() as u64;
                        }
                    } else {
                        next_unresolved.push(terminal);
                    }
                }
                unresolved = next_unresolved;
            }
        }
        }

        if std::env::var_os("GLRMASK_DUMP_TI_CONE_COMPONENTS").is_some() {
            dfa.ensure_support_quotient();
            let quotient = dfa
                .support_quotient
                .as_ref()
                .expect("support quotient initialized for cone diagnostics");
            if let Some(candidate_groups) = diagnostic_candidate_groups.as_ref() {
                let mut final_class_by_terminal = vec![u32::MAX; active_terminals.len()];
                for (&representative, members) in &result {
                    for &member in members {
                        final_class_by_terminal[member as usize] = representative;
                    }
                }
                let mut predicted_groups = 0usize;
                let mut predicted_members = 0usize;
                let mut certified_groups = 0usize;
                let mut certified_members = 0usize;
                let mut impure_groups = 0usize;
                let mut exact_final_groups = 0usize;
                for candidate_group in candidate_groups {
                    let mut membership = vec![false; active_terminals.len()];
                    for &terminal in candidate_group {
                        membership[terminal as usize] = true;
                    }
                    let pair_stats = (candidate_group.len() >= 64)
                        .then(|| dfa.generic_output_pair_stats(&membership));
                    let mut by_code =
                        BTreeMap::<SmallVec<[u64; 8]>, Vec<TerminalID>>::new();
                    for &terminal in candidate_group {
                        let cone = dfa.support_quotient_terminal_cone(quotient, terminal);
                        let (code, _) = dfa.support_terminal_cone_generic_form(
                            quotient,
                            &cone,
                            terminal,
                            &membership,
                            pair_stats.as_deref(),
                        );
                        by_code.entry(code).or_default().push(terminal);
                    }
                    for predicted in by_code.into_values().filter(|group| group.len() >= 2) {
                        predicted_groups += 1;
                        predicted_members += predicted.len();
                        let first_class = final_class_by_terminal[predicted[0] as usize];
                        let pure = predicted.iter().all(|&terminal| {
                            final_class_by_terminal[terminal as usize] == first_class
                        });
                        impure_groups += usize::from(!pure);
                        if pure {
                            let final_size = result
                                .get(&first_class)
                                .map_or(0, BTreeSet::len);
                            exact_final_groups += usize::from(final_size == predicted.len());
                        }
                        if dfa
                            .support_petal_batch_certificate(quotient, &predicted)
                            .is_some()
                        {
                            certified_groups += 1;
                            certified_members += predicted.len();
                        }
                    }
                }
                eprintln!(
                    "[glrmask/dump][ti_generic_cone_groups] predicted_groups={} predicted_members={} certified_groups={} certified_members={} impure_groups={} exact_final_groups={}",
                    predicted_groups,
                    predicted_members,
                    certified_groups,
                    certified_members,
                    impure_groups,
                    exact_final_groups,
                );
            }
            for (&representative, members) in &result {
                if members.len() < 2 {
                    continue;
                }
                let members = members.iter().copied().collect::<Vec<_>>();
                let cones = members
                    .iter()
                    .map(|&terminal| dfa.support_quotient_terminal_cone(quotient, terminal))
                    .collect::<Vec<_>>();
                let common_cone = cones.iter().skip(1).fold(
                    cones.first().cloned().unwrap_or_default(),
                    |common, cone| {
                        let mut intersection = Vec::new();
                        let mut i = 0usize;
                        let mut j = 0usize;
                        while i < common.len() && j < cone.len() {
                            match common[i].cmp(&cone[j]) {
                                std::cmp::Ordering::Less => i += 1,
                                std::cmp::Ordering::Greater => j += 1,
                                std::cmp::Ordering::Equal => {
                                    intersection.push(common[i]);
                                    i += 1;
                                    j += 1;
                                }
                            }
                        }
                        intersection.into()
                    },
                );
                let petals = cones
                    .iter()
                    .map(|cone| {
                        let mut petal = Vec::new();
                        let mut common_index = 0usize;
                        for &class in cone {
                            while common_index < common_cone.len()
                                && common_cone[common_index] < class
                            {
                                common_index += 1;
                            }
                            if common_index == common_cone.len()
                                || common_cone[common_index] != class
                            {
                                petal.push(class);
                            }
                        }
                        petal
                    })
                    .collect::<Vec<_>>();
                let mut parent = (0..members.len()).collect::<Vec<_>>();
                fn find(parent: &mut [usize], mut x: usize) -> usize {
                    while parent[x] != x {
                        parent[x] = parent[parent[x]];
                        x = parent[x];
                    }
                    x
                }
                for left in 0..members.len() {
                    for right in left + 1..members.len() {
                        let mut i = 0usize;
                        let mut j = 0usize;
                        let mut overlap = false;
                        while i < petals[left].len() && j < petals[right].len() {
                            match petals[left][i].cmp(&petals[right][j]) {
                                std::cmp::Ordering::Less => i += 1,
                                std::cmp::Ordering::Greater => j += 1,
                                std::cmp::Ordering::Equal => {
                                    overlap = true;
                                    break;
                                }
                            }
                        }
                        if overlap {
                            let a = find(&mut parent, left);
                            let b = find(&mut parent, right);
                            if a != b {
                                parent[b] = a;
                            }
                        }
                    }
                }
                let mut component_sizes = FxHashMap::<usize, usize>::default();
                for index in 0..members.len() {
                    let root = find(&mut parent, index);
                    *component_sizes.entry(root).or_default() += 1;
                }
                let mut component_sizes = component_sizes.into_values().collect::<Vec<_>>();
                component_sizes.sort_unstable_by(|a, b| b.cmp(a));
                let mut cone_sizes = cones.iter().map(|cone| cone.len()).collect::<Vec<_>>();
                cone_sizes.sort_unstable();
                let batch_certified = dfa
                    .support_petal_batch_certificate(quotient, &members)
                    .is_some();
                eprintln!(
                    "[glrmask/dump][ti_cones] representative={} members={} cone_min={} cone_median={} cone_max={} common={} petal_overlap_components={:?} batch_certified={}",
                    representative,
                    members.len(),
                    cone_sizes.first().copied().unwrap_or(0),
                    cone_sizes.get(cone_sizes.len() / 2).copied().unwrap_or(0),
                    cone_sizes.last().copied().unwrap_or(0),
                    common_cone.len(),
                    component_sizes,
                    batch_certified,
                );
            }
        }

        if profile_timing {
            eprintln!(
                "[glrmask/profile][terminal_interchangeability] output_pair_rejections={} output_pair_cached_closed={} output_invariant_checks={} first_round_rejections={} support_transposition_checks={} petal_batch_groups={} petal_batch_members={} petal_batch_grouping_ms={:.3} petal_batch_certificate_ms={:.3} petal_batch_map_ms={:.3} support_transposition_certified={} support_transposition_no_template={} support_transposition_outside_cone={} support_transposition_root_rejected={} support_transposition_incoming_rejected={} support_transposition_signature_rejected={} direct_exact_checks={} exact_fallback_skips={} support_quotient_skipped_candidates={} output_pair_filter_ms={:.3} frozen_output_ms={:.3} first_round_ms={:.3} support_transposition_ms={:.3} support_setup_ms={:.3} support_quotient_build_ms={:.3} support_template_ms={:.3} support_cone_ms={:.3} support_verify_ms={:.3} exact_map_ms={:.3} accepted_map_storage_ms={:.3} quotient_certified={} sparse_quotient_certified={} sparse_cone_avg={:.1} sparse_cone_max={} sparse_cone_ms={:.3} sparse_refinement_ms={:.3} sparse_map_ms={:.3} accepted_representative_members={} total_ms={:.3}",
                output_pair_rejections,
                output_pair_cached_closed,
                output_invariant_checks,
                first_round_rejections,
                support_transposition_checks,
                petal_batch_groups,
                petal_batch_members,
                petal_batch_grouping_ns as f64 / 1_000_000.0,
                petal_batch_certificate_ns as f64 / 1_000_000.0,
                petal_batch_map_ns as f64 / 1_000_000.0,
                dfa.support_transposition_certified,
                dfa.support_transposition_no_template,
                dfa.support_transposition_outside_cone,
                dfa.support_transposition_root_rejected,
                dfa.support_transposition_incoming_rejected,
                dfa.support_transposition_signature_rejected,
                direct_exact_checks,
                exact_fallback_skips,
                support_quotient_skipped_candidates,
                output_pair_filter_ns as f64 / 1_000_000.0,
                frozen_output_ns as f64 / 1_000_000.0,
                first_round_ns as f64 / 1_000_000.0,
                support_transposition_ns as f64 / 1_000_000.0,
                dfa.support_transposition_support_setup_ns as f64 / 1_000_000.0,
                dfa.support_quotient_build_ns as f64 / 1_000_000.0,
                dfa.support_transposition_template_ns as f64 / 1_000_000.0,
                dfa.support_transposition_cone_ns as f64 / 1_000_000.0,
                dfa.support_transposition_verify_ns as f64 / 1_000_000.0,
                exact_map_ns as f64 / 1_000_000.0,
                accepted_map_storage_ns as f64 / 1_000_000.0,
                dfa.quotient_certified,
                dfa.sparse_quotient_certified,
                dfa.sparse_quotient_cone_classes_total as f64
                    / dfa.sparse_quotient_certified.max(1) as f64,
                dfa.sparse_quotient_cone_classes_max,
                dfa.sparse_quotient_cone_ns as f64 / 1_000_000.0,
                dfa.sparse_quotient_refinement_ns as f64 / 1_000_000.0,
                dfa.sparse_quotient_map_ns as f64 / 1_000_000.0,
                accepted_representative_members,
                started_at
                    .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0),
            );
        }
        assert_partition_invariants(&result, active_terminals);
        if let Some(quotient) = dfa.support_quotient.as_ref() {
            *context.support_partition_seed.borrow_mut() = Some(SupportPartitionSeed {
                active_terminals: Arc::from(active_terminals.to_vec()),
                quotient: Arc::clone(quotient),
            });
        }
        *context.output_projection_seed.borrow_mut() = Some(OutputProjectionSeed {
            active_terminals: Arc::from(active_terminals.to_vec()),
            output_pairs: Arc::clone(&dfa.output_pairs),
            output_pair_by_state: Arc::clone(&dfa.output_pair_by_state),
        });
        TiRoundTransportWitnesses {
            active_before_round: active_terminals.to_vec(),
            partition: result,
            maps: accepted_maps,
        }
    }


/// Return the singleton-inclusive one-round partition for the current active
/// terminal mask. Inactive terminals are deliberately absent: this same mask
/// controls both candidates and frozen F/U observation for the round.
pub fn singleton_partition(
    active_terminals: &[bool],
) -> BTreeMap<TerminalID, BTreeSet<TerminalID>> {
    active_terminals
        .iter()
        .enumerate()
        .filter_map(|(terminal, &active)| {
            active.then_some((terminal as TerminalID, BTreeSet::from([terminal as TerminalID])))
        })
        .collect()
}

fn assert_partition_invariants(
    partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    active_terminals: &[bool],
) {
    let mut seen = vec![false; active_terminals.len()];
    for (&representative, members) in partition {
        assert!(members.contains(&representative), "TI partition class must contain its key");
        for &member in members {
            assert!(
                active_terminals.get(member as usize).copied().unwrap_or(false),
                "TI partition contains an inactive terminal",
            );
            assert!(
                !std::mem::replace(&mut seen[member as usize], true),
                "TI partition contains a terminal in multiple classes",
            );
        }
    }
    assert_eq!(seen, active_terminals, "TI partition must cover exactly active terminals");
}

/// Fold one transient discovery round into a flat original-terminal
/// partition. `round` names only current visible representatives; its values
/// are expanded through the existing flat classes. No history is retained.
pub fn fold_one_round_partition(
    classes: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    round: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
) -> BTreeMap<TerminalID, BTreeSet<TerminalID>> {
    let mut next_classes = BTreeMap::new();
    for (&new_representative, old_representatives) in round {
        let mut members = BTreeSet::new();
        for &old_representative in old_representatives {
            let old_members = classes
                .get(&old_representative)
                .expect("TI round must refer only to visible representatives");
            members.extend(old_members.iter().copied());
        }
        assert!(members.contains(&new_representative));
        next_classes.insert(new_representative, members);
    }
    next_classes
}

pub fn active_terminals_for_partition(
    partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    terminal_count: usize,
) -> Vec<bool> {
    let mut active = vec![false; terminal_count];
    for &representative in partition.keys() {
        active[representative as usize] = true;
    }
    active
}

pub fn partition_has_merges(partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>) -> bool {
    partition.values().any(|members| members.len() > 1)
}

pub fn visible_output_raw_labels(
    partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    active_terminals: &[bool],
) -> Vec<bool> {
    // TI narrows the L2P-active alphabet; it must never reactivate terminals
    // owned by another terminal family. Start from the caller's L2P mask, then
    // hide only nonrepresentative TI members. Expansion restores those members.
    let mut visible = active_terminals.to_vec();
    for (&representative, members) in partition {
        for &member in members {
            if member != representative {
                visible[member as usize] = false;
            }
        }
    }
    visible
}

fn representatives_for_partition(
    partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    terminal_count: usize,
) -> Vec<TerminalID> {
    let mut representative_for = (0..terminal_count as TerminalID).collect::<Vec<_>>();
    for (&representative, members) in partition {
        for &member in members {
            representative_for[member as usize] = representative;
        }
    }
    representative_for
}

/// Coalesce grammar follows for a compact representative alphabet. A class pair
/// is disallowed only when every concrete member pair is disallowed; original
/// follows are retained for the later raw-member transport construction.
pub fn coalesced_disallowed_follows(
    partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    original: &BTreeMap<u32, BitSet>,
    num_terminals: usize,
) -> BTreeMap<u32, BitSet> {
    // A compact predecessor class is disallowed from a compact successor class
    // precisely when every raw predecessor member forbids every raw successor
    // member.  For one predecessor class, first intersect its raw rows; the
    // resulting bitset contains exactly the raw successors forbidden by all
    // of its members.  Testing each successor class against that intersection
    // avoids a hash-map update for every raw edge, while retaining the exact
    // all-member-pair relation.
    let classes = partition
        .iter()
        .map(|(&representative, members)| (representative, members))
        .collect::<Vec<_>>();
    let empty = BitSet::new(num_terminals);
    let mut coalesced = BTreeMap::new();

    for &(representative, members) in &classes {
        let mut members = members.iter();
        let Some(&first_member) = members.next() else {
            continue;
        };
        let mut common_successors = original
            .get(&(first_member as u32))
            .cloned()
            .unwrap_or_else(|| empty.clone());
        for &member in members {
            let Some(successors) = original.get(&(member as u32)) else {
                common_successors.clear_all();
                break;
            };
            common_successors.intersect_with(successors);
            if common_successors.is_empty() {
                break;
            }
        }

        let mut row = BitSet::new(num_terminals);
        for &(successor_representative, successor_members) in &classes {
            if successor_members
                .iter()
                .all(|&member| common_successors.contains(member as usize))
            {
                row.set(successor_representative as usize);
            }
        }
        coalesced.insert(representative as u32, row);
    }

    coalesced
}

/// Re-run the same accepted-pair decision used by discovery for one replayed
/// round. Replay must derive the scanner coordinate under that round's
/// pre-merge active-terminal mask.
fn replay_accepted_interchange_map(
    dfa: &mut InterchangeabilityDfa,
    representative: TerminalID,
    member: TerminalID,
) -> Option<InterchangeMap> {
    if !dfa.observed_output_pair_set_is_swap_closed(representative, member) {
        return None;
    }
    if dfa.swap_preserves_all_frozen_outputs(representative, member) {
        return Some(dfa.canonical_identity_map());
    }
    if !dfa.canonical_round_one_still_possible(representative, member) {
        return None;
    }
    dfa.interchange_map(representative, member)
}

/// Replay deterministic TI discovery while materializing temporary transport
/// witnesses. The stored result remains only the flat final partition.
///
/// Each accepted pair uses the active mask that existed before its round
/// merged anything. When a later round folds `old_representative` into
/// `representative`, it composes that round-local map outside the prior map for
/// every original member. Reconstructing final pairs under only final
/// representatives, or final representatives plus one member, is unsound.
pub fn binary_transport_modes(
    tokenizer: &Tokenizer,
    original_active_terminals: &[bool],
    partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    relevant_bytes: &[bool; 256],
    ignore_terminal: Option<TerminalID>,
) -> Vec<TerminalNwaTransportMode> {
    let terminal_count = tokenizer.num_terminals() as usize;
    assert_eq!(original_active_terminals.len(), terminal_count);

    let mut active = original_active_terminals.to_vec();
    let mut classes = singleton_partition(&active);
    let mut rounds = Vec::new();

    loop {
        let round = discover_one_round_with_transport_witnesses(
            tokenizer,
            &active,
            relevant_bytes,
            ignore_terminal,
        );
        let next_classes = fold_one_round_partition(&classes, &round.partition);
        let next_active = active_terminals_for_partition(&round.partition, active.len());
        rounds.push(round);
        if next_active == active {
            assert_eq!(
                &next_classes,
                partition,
                "deterministic TI replay must reproduce the retained final partition",
            );
            break;
        }
        classes = next_classes;
        active = next_active;
    }

    binary_transport_modes_from_witnesses(tokenizer, original_active_terminals, partition, &rounds)
}

/// Build temporary member reconstruction modes from the witnesses produced by
/// the original discovery loop.  This is the exact iterative replay relation,
/// but it never repeats discovery: each historical active mask and its
/// accepted pair map were captured at the instant that round was certified.
pub fn binary_transport_modes_from_witnesses(
    tokenizer: &Tokenizer,
    original_active_terminals: &[bool],
    partition: &BTreeMap<TerminalID, BTreeSet<TerminalID>>,
    rounds: &[TiRoundTransportWitnesses],
) -> Vec<TerminalNwaTransportMode> {
    let state_count = tokenizer.num_states() as usize;
    let terminal_count = tokenizer.num_terminals() as usize;
    assert_eq!(original_active_terminals.len(), terminal_count);

    let mut active = original_active_terminals.to_vec();
    let mut classes = singleton_partition(&active);
    let mut map_for_original = vec![None::<Arc<TransportScannerStateMap>>; terminal_count];

    for round in rounds {
        assert_eq!(
            round.active_before_round, active,
            "TI transport witness must use its historical pre-merge active mask",
        );
        for (&representative, old_representatives) in &round.partition {
            for &old_representative in old_representatives {
                if old_representative == representative {
                    continue;
                }
                let round_map = round
                    .maps
                    .get(&(representative, old_representative))
                    .unwrap_or_else(|| {
                        panic!(
                            "TI replay lost an accepted round-local transport witness: rep={} member={} active_before_round={:?}",
                            representative,
                            old_representative,
                            active,
                        )
                    });
                let original_members = classes
                    .get(&old_representative)
                    .expect("TI replay round must refer only to current representatives");
                for &original_member in original_members {
                    let original_member = original_member as usize;
                    map_for_original[original_member] = Some(
                        map_for_original[original_member]
                            .take()
                            .map(|prior_map| {
                                TransportScannerStateMap::compose(
                                    Arc::clone(round_map),
                                    prior_map,
                                )
                            })
                            .unwrap_or_else(|| Arc::clone(round_map)),
                    );
                }
            }
        }

        let next_classes = fold_one_round_partition(&classes, &round.partition);
        let next_active = active_terminals_for_partition(&round.partition, active.len());
        classes = next_classes;
        active = next_active;
    }

    assert_eq!(
        &classes, partition,
        "captured TI replay rounds must reproduce the retained final partition",
    );

    let mut modes = vec![TerminalNwaTransportMode::ordinary(state_count)];
    for (&representative, members) in partition {
        for &member in members {
            if member == representative {
                continue;
            }
            let map = map_for_original[member as usize].as_ref().unwrap_or_else(|| {
                panic!(
                    "TI replay produced no composed transport witness: rep={} member={}",
                    representative, member,
                )
            });
            modes.push(TerminalNwaTransportMode::member(
                map.as_ref().clone(),
                representative,
                member,
            ));
        }
    }
    modes
}

/// Validate the common raw scanner domain of temporary transport modes.
///
/// Older code eagerly rewrote every mode into an explicit raw-state vector of
/// ordinary quotient representatives here.  That does not change either later
/// consumer: both need exactly `Q(mode(source))`, and applying `Q` directly is
/// equivalent to first replacing the raw target by its `Q` representative.
/// Keeping the maps lazy avoids `raw lexer states × TI members` work.
pub fn canonicalize_transport_mode_states(
    modes: &[TerminalNwaTransportMode],
    ordinary_state_map: &ManyToOneIdMap,
) {
    let state_count = ordinary_state_map.original_to_internal.len();
    for mode in modes {
        assert_eq!(
            mode.scanner_state_for_original.len(),
            state_count,
            "transport mode state domain must match ordinary state quotient",
        );
    }
}

/// Refine an ordinary terminal-DWA quotient by the scanner destinations of
/// every temporary transport mode. A source raw state `s` is assigned a compact
/// transport TSID from the exact signature
/// `(Q(m_0(s)), Q(m_1(s)), …)`, where `m_0` is the ordinary mode and each later
/// mode is one target-only member reconstruction coordinate.
///
/// The full states-by-modes signature matrix is deliberately never
/// materialized.  A composed transport is constant on the source quotient of
/// its innermost round-local map.  Therefore every mode sharing that source
/// quotient is evaluated once per source class, then each raw state is refined
/// only by the small tuple of its per-round group ids.  This is the same exact
/// signature vector while avoiding the normal `raw states × TI members` path.
pub fn transport_coordinate_quotient(
    ordinary_state_map: &ManyToOneIdMap,
    modes: &[TerminalNwaTransportMode],
) -> ManyToOneIdMap {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
    let total_started_at = profile_timing.then(Instant::now);
    assert!(
        !modes.is_empty(),
        "transport coordinate quotient needs the ordinary mode",
    );
    let state_count = ordinary_state_map.original_to_internal.len();
    let ordinary_coordinate_key = |target_state: usize| {
        let mapped = ordinary_state_map
            .original_to_internal
            .get(target_state)
            .copied()
            .unwrap_or(u32::MAX);
        // A target outside the ordinary proof domain must retain its raw
        // identity; merging all unmapped targets would be unsound.
        if mapped == u32::MAX {
            (1u64 << 32) | target_state as u64
        } else {
            mapped as u64
        }
    };

    struct ModeGroup {
        domain_key: usize,
        domain_mode: usize,
        modes: Vec<usize>,
        component_for_source_class: Vec<u32>,
    }

    #[derive(Eq, Hash, PartialEq)]
    struct SparseModeSignature {
        /// Sparse feature overrides from the ordinary coordinate. Real mode
        /// ids occupy `[0, modes.len())`; tail-default features use ids above
        /// that range. Absence means the ordinary coordinate already present
        /// in the final state signature.
        deviations: SmallVec<[(u32, u64); 8]>,
    }

    struct TailGroup<'a> {
        tail: Vec<&'a TransportScannerStateMap>,
        modes: Vec<(usize, &'a TransportScannerStateMap)>,
        /// Sorted sparse overrides from the ordinary coordinate, keyed by the
        /// current group's source class.
        default_deviations: Vec<(u32, u64)>,
    }

    // The ordinary mode is represented directly by the first coordinate below.
    // Group every target-only reconstruction by the partition on which its
    // complete composed transform is constant.
    let mut group_index_by_domain = FxHashMap::<usize, usize>::default();
    let mut groups = Vec::<ModeGroup>::new();
    for (mode_index, mode) in modes.iter().enumerate().skip(1) {
        assert_eq!(
            mode.scanner_state_for_original.len(),
            state_count,
            "transport mode state domain must match ordinary state quotient",
        );
        let domain_key = mode.scanner_state_for_original.innermost_source_domain_key();
        let group_index = match group_index_by_domain.entry(domain_key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let index = groups.len();
                groups.push(ModeGroup {
                    domain_key,
                    domain_mode: mode_index,
                    modes: Vec::new(),
                    component_for_source_class: Vec::new(),
                });
                entry.insert(index);
                index
            }
        };
        groups[group_index].modes.push(mode_index);
    }

    let components_started_at = profile_timing.then(Instant::now);
    let mut source_class_mode_evaluations = 0usize;
    let mut macro_transport_groups = 0usize;
    let mut sparse_tail_groups = 0usize;
    let mut sparse_tail_atoms = 0usize;
    let mut sparse_tail_default_cells = 0usize;
    let mut sparse_tail_default_changes = 0usize;
    // A transport mode with no quotient deviations is a pure structural
    // relabeling that preserves the ordinary coordinate, so its component is a
    // function of the ordinary coordinate and cannot refine the ordinary
    // partition. Such groups are redundant: we skip building their component
    // table and omit them from the final per-state signature. Because the
    // absolute final-class numbering never reaches the digest (proven: the
    // digest is invariant under class renumbering), dropping redundant groups
    // is exact.
    let mut group_is_redundant = vec![false; groups.len()];
    for (group_index, group) in groups.iter_mut().enumerate() {
        let domain = &modes[group.domain_mode].scanner_state_for_original;
        let source_class_count = domain.innermost_source_class_count();

        let mut all_modes_are_direct_quotients = true;
        let group_quotient_deviations: usize = group
            .modes
            .iter()
            .map(|&mode_index| {
                let deviations = modes[mode_index]
                    .scanner_state_for_original
                    .quotient_deviations();
                all_modes_are_direct_quotients &= deviations.is_some();
                deviations.map_or(0, |deviations| deviations.len())
            })
            .sum();
        // Only a direct quotient with no deviations is known structurally to
        // preserve the ordinary coordinate. `None` does not mean identity: it
        // also covers composed, macro-quotient, and explicit transport maps.
        // Treating those shapes as zero-deviation groups can erase a genuine
        // transport signature component and merge raw states whose member-mode
        // destinations land in different core coordinates.
        if all_modes_are_direct_quotients && group_quotient_deviations == 0 {
            group_is_redundant[group_index] = true;
            continue;
        }

        // A direct macro quotient is constant on the shared macro class of a
        // source C class. Classify those macro classes once, then project the
        // component id back to C instead of scanning every C class per mode.
        if let Some(first_parts) = domain.direct_macro_quotient_parts() {
            let shared_macro_classes = first_parts.macro_class_for_source_class;
            let shared_macro_representatives = first_parts.representative_for_macro_class;
            let mut mode_deviations = Vec::with_capacity(group.modes.len());
            let mut all_direct_macro = true;
            for &mode_index in &group.modes {
                let Some(parts) = modes[mode_index]
                    .scanner_state_for_original
                    .direct_macro_quotient_parts()
                else {
                    all_direct_macro = false;
                    break;
                };
                if parts.macro_class_for_source_class.as_ptr() != shared_macro_classes.as_ptr()
                    || parts.representative_for_macro_class.as_ptr()
                        != shared_macro_representatives.as_ptr()
                {
                    all_direct_macro = false;
                    break;
                }
                mode_deviations.push(parts.source_macro_for_target_deviations);
            }
            if all_direct_macro {
                assert_eq!(shared_macro_classes.len(), source_class_count);
                let macro_coordinates = shared_macro_representatives
                    .iter()
                    .map(|&state| ordinary_coordinate_key(state as usize))
                    .collect::<Vec<_>>();
                let mut component_by_signature = FxHashMap::<Vec<u64>, u32>::default();
                let mut component_for_macro = Vec::with_capacity(macro_coordinates.len());
                for source_macro in 0..macro_coordinates.len() {
                    let signature = mode_deviations
                        .iter()
                        .map(|deviations| {
                            let target_macro = deviations
                                .binary_search_by_key(&(source_macro as u32), |&(source, _)| source)
                                .map(|index| deviations[index].1 as usize)
                                .unwrap_or(source_macro);
                            macro_coordinates[target_macro]
                        })
                        .collect::<Vec<_>>();
                    let next_component = component_by_signature.len() as u32;
                    let component = *component_by_signature
                        .entry(signature)
                        .or_insert(next_component);
                    component_for_macro.push(component);
                }
                source_class_mode_evaluations += macro_coordinates.len() * group.modes.len();
                group.component_for_source_class = shared_macro_classes
                    .iter()
                    .map(|&source_macro| component_for_macro[source_macro as usize])
                    .collect();
                macro_transport_groups += 1;
                continue;
            }
        }

        let mut programs = Vec::<(usize, Vec<&TransportScannerStateMap>)>::new();
        let mut supports_sparse_program = true;
        for &mode_index in &group.modes {
            let mut parts = Vec::new();
            modes[mode_index]
                .scanner_state_for_original
                .append_atomic_transforms(&mut parts);
            if parts
                .first()
                .and_then(|inner| inner.quotient_deviations())
                .is_none()
            {
                supports_sparse_program = false;
                break;
            }
            programs.push((mode_index, parts));
        }

        if !supports_sparse_program {
            let mut component_by_signature = FxHashMap::<Vec<u64>, u32>::default();
            group.component_for_source_class = Vec::with_capacity(source_class_count);
            for source_class in 0..source_class_count {
                let source_state = domain.innermost_source_representative(source_class);
                let mut signature = Vec::with_capacity(group.modes.len());
                for &mode_index in &group.modes {
                    let mode = &modes[mode_index].scanner_state_for_original;
                    let target_state = mode.scanner_state(source_state) as usize;
                    signature.push(ordinary_coordinate_key(target_state));
                }
                source_class_mode_evaluations += group.modes.len();
                let next_component = component_by_signature.len() as u32;
                let component = *component_by_signature
                    .entry(signature)
                    .or_insert(next_component);
                group.component_for_source_class.push(component);
            }
            continue;
        }

        let mut tail_group_index_by_key = FxHashMap::<Vec<usize>, usize>::default();
        let mut tail_groups = Vec::<TailGroup>::new();
        for (mode_index, parts) in programs {
            let inner = parts[0];
            debug_assert_eq!(
                inner.innermost_source_domain_key(),
                group.domain_key,
                "the first transport program atom must own this source domain",
            );
            let tail = parts[1..].to_vec();
            let tail_key = tail
                .iter()
                .map(|part| std::ptr::from_ref(*part) as usize)
                .collect::<Vec<_>>();
            let tail_group_index = match tail_group_index_by_key.entry(tail_key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let index = tail_groups.len();
                    tail_groups.push(TailGroup {
                        tail,
                        modes: Vec::new(),
                        default_deviations: Vec::new(),
                    });
                    entry.insert(index);
                    index
                }
            };
            tail_groups[tail_group_index].modes.push((mode_index, inner));
        }

        sparse_tail_groups += tail_groups.len();
        sparse_tail_atoms += tail_groups.iter().map(|group| group.tail.len()).sum::<usize>();
        let ordinary_for_source_class = (0..source_class_count)
            .map(|source_class| {
                ordinary_coordinate_key(
                    domain.innermost_source_representative(source_class) as usize,
                )
            })
            .collect::<Vec<_>>();
        let mut deviations_by_source_class =
            vec![SmallVec::<[(u32, u64); 8]>::new(); source_class_count];
        let mut current_classes_by_tail_domain =
            FxHashMap::<usize, Vec<Vec<u32>>>::default();

        for (tail_group_index, tail_group) in tail_groups.iter_mut().enumerate() {
            let tail_feature = modes.len() as u32 + tail_group_index as u32;
            if tail_group.tail.is_empty() {
                // Identity tail: every default is already represented by the
                // ordinary coordinate in the final quotient signature.
            } else if tail_group.tail.len() == 1
                && tail_group.tail[0].quotient_deviations().is_some()
            {
                let transform = tail_group.tail[0];
                let tail_domain_key = transform.innermost_source_domain_key();
                let classes_by_tail_class = current_classes_by_tail_domain
                    .entry(tail_domain_key)
                    .or_insert_with(|| {
                        let mut inverse =
                            vec![Vec::<u32>::new(); transform.innermost_source_class_count()];
                        for source_class in 0..source_class_count {
                            let raw_state =
                                domain.innermost_source_representative(source_class);
                            let tail_class =
                                transform.innermost_source_class(raw_state);
                            inverse[tail_class].push(source_class as u32);
                        }
                        inverse
                    });
                for &(input_class, output_class) in transform
                    .quotient_deviations()
                    .expect("one-atom quotient tail was checked above")
                {
                    let coordinate = ordinary_coordinate_key(
                        transform.innermost_source_representative(output_class as usize)
                            as usize,
                    );
                    for &source_class in &classes_by_tail_class[input_class as usize] {
                        let source_class = source_class as usize;
                        sparse_tail_default_cells += 1;
                        source_class_mode_evaluations += 1;
                        if coordinate != ordinary_for_source_class[source_class] {
                            sparse_tail_default_changes += 1;
                            tail_group
                                .default_deviations
                                .push((source_class as u32, coordinate));
                            deviations_by_source_class[source_class]
                                .push((tail_feature, coordinate));
                        }
                    }
                }
                tail_group.default_deviations.sort_unstable_by_key(|&(class, _)| class);
            } else {
                // General exact fallback. This is absent in the current p0/p1
                // workload, but retains arbitrary composed-tail semantics.
                for source_class in 0..source_class_count {
                    let mut target_state =
                        domain.innermost_source_representative(source_class);
                    for transform in &tail_group.tail {
                        target_state = transform.scanner_state(target_state);
                    }
                    let coordinate = ordinary_coordinate_key(target_state as usize);
                    sparse_tail_default_cells += 1;
                    source_class_mode_evaluations += 1 + tail_group.tail.len();
                    if coordinate != ordinary_for_source_class[source_class] {
                        sparse_tail_default_changes += 1;
                        tail_group
                            .default_deviations
                            .push((source_class as u32, coordinate));
                        deviations_by_source_class[source_class]
                            .push((tail_feature, coordinate));
                    }
                }
            }

            let default_for = |source_class: usize| {
                tail_group
                    .default_deviations
                    .binary_search_by_key(&(source_class as u32), |&(class, _)| class)
                    .map(|index| tail_group.default_deviations[index].1)
                    .unwrap_or(ordinary_for_source_class[source_class])
            };
            for &(mode_index, inner) in &tail_group.modes {
                let inner_deviations = inner
                    .quotient_deviations()
                    .expect("sparse transport program was checked above");
                for &(input_class, output_class) in inner_deviations {
                    let input_class = input_class as usize;
                    let output_class = output_class as usize;
                    assert!(
                        input_class < source_class_count && output_class < source_class_count,
                        "TI transport deviation must stay within its source quotient",
                    );
                    let coordinate = default_for(output_class);
                    if coordinate != default_for(input_class) {
                        deviations_by_source_class[input_class]
                            .push((mode_index as u32, coordinate));
                    }
                }
            }
        }

        let mut component_by_signature = FxHashMap::<SparseModeSignature, u32>::default();
        group.component_for_source_class = Vec::with_capacity(source_class_count);
        for source_class in 0..source_class_count {
            let mut deviations = std::mem::take(&mut deviations_by_source_class[source_class]);
            deviations.sort_unstable_by_key(|&(feature, _)| feature);
            let signature = SparseModeSignature { deviations };
            let next_component = component_by_signature.len() as u32;
            let component = *component_by_signature
                .entry(signature)
                .or_insert(next_component);
            group.component_for_source_class.push(component);
        }
    }
    let component_build_ms = components_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let final_refinement_started_at = profile_timing.then(Instant::now);
    let refining_groups: Vec<&ModeGroup> = groups
        .iter()
        .enumerate()
        .filter(|&(group_index, _)| !group_is_redundant[group_index])
        // A group whose `component_for_source_class` is constant induces a
        // constant column (every state maps to the same component), which
        // refines nothing. The state-order loop below already drops constant
        // columns via `next_canon <= 1`, but only after materializing all
        // `state_count` entries. Checking the far smaller source-class vector
        // here skips that materialization entirely — decisive when there are
        // many single-mode groups over a large state space (each such group
        // otherwise costs a full `state_count` scan plus a full-column hash).
        .filter(|&(_, group)| {
            let components = &group.component_for_source_class;
            match components.first() {
                Some(&first) => components.iter().any(|&c| c != first),
                None => false,
            }
        })
        .map(|(_, group)| group)
        .collect();
    // Resolve each refining group's innermost class slice once. `None` means an
    // `Explicit` innermost map where the source class equals the raw state, so
    // the per-state signature assembly below is a plain array index instead of
    // re-walking any `Composed` transport chain.
    let refining_slices: Vec<(Option<&[u32]>, &[u32])> = refining_groups
        .iter()
        .map(|group| {
            let domain = &modes[group.domain_mode].scanner_state_for_original;
            (
                domain.innermost_class_for_original(),
                group.component_for_source_class.as_slice(),
            )
        })
        .collect();
    // Component ids are exact equality labels within each group. Their numeric
    // names do not matter: including the raw labels directly in the joint
    // signature yields the same partition intersection. This avoids building,
    // relabelling, hashing, and then rereading one full raw-state column per
    // group.
    let mut class_for_signature = FxHashMap::<SmallVec<[u64; 8]>, u32>::default();
    let mut class_for_state = Vec::with_capacity(state_count);
    for source_state in 0..state_count {
        let mut signature = SmallVec::<[u64; 8]>::with_capacity(refining_slices.len() + 1);
        signature.push(ordinary_coordinate_key(source_state));
        for &(class_slice, components) in &refining_slices {
            let source_class = match class_slice {
                Some(slice) => slice[source_state] as usize,
                None => source_state,
            };
            signature.push(components[source_class] as u64);
        }
        let next_class = class_for_signature.len() as u32;
        let class = *class_for_signature.entry(signature).or_insert(next_class);
        class_for_state.push(class);
    }
    let final_refinement_ms = final_refinement_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let class_count = class_for_state
        .iter()
        .copied()
        .max()
        .map_or(0usize, |class| class as usize + 1);
    let mut representatives = vec![u32::MAX; class_count];
    for (state, &class) in class_for_state.iter().enumerate() {
        let representative = &mut representatives[class as usize];
        if *representative == u32::MAX {
            *representative = state as u32;
        }
    }

    let quotient = ManyToOneIdMap::from_original_to_internal_with_representatives(
        class_for_state,
        class_count as u32,
        representatives,
    );
    if profile_timing {
        let group_source_class_counts = groups
            .iter()
            .map(|group| {
                modes[group.domain_mode]
                    .scanner_state_for_original
                    .innermost_source_class_count()
            })
            .collect::<Vec<_>>();
        let group_mode_counts = groups
            .iter()
            .map(|group| group.modes.len())
            .collect::<Vec<_>>();
        eprintln!(
            "[glrmask/profile][transport_coordinate_quotient] modes={} groups={} macro_transport_groups={} sparse_tail_groups={} sparse_tail_atoms={} sparse_tail_default_cells={} sparse_tail_default_changes={} group_source_class_counts={:?} group_mode_counts={:?} source_class_mode_evaluations={} raw_state_group_lookups={} component_build_ms={:.3} final_refinement_ms={:.3} total_ms={:.3}",
            modes.len(),
            groups.len(),
            macro_transport_groups,
            sparse_tail_groups,
            sparse_tail_atoms,
            sparse_tail_default_cells,
            sparse_tail_default_changes,
            group_source_class_counts,
            group_mode_counts,
            source_class_mode_evaluations,
            state_count * groups.len(),
            component_build_ms,
            final_refinement_ms,
            total_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0),
        );
    }
    quotient
}

/// Sparse final-coordinate lifting for post-DWA TI expansion.
///
/// The ordinary coordinate lift is shared by every member mode. A mode differs
/// only at the final transport-coordinate classes whose representative scanner
/// state lands in a different ordinary core class. BFCL p0 has thousands of
/// members but only a handful of such deviations per mode, so lifting the full
/// final-coordinate weight for every `(weight, mode)` pair is the wrong shape.
struct PostDwaWeightLifter {
    ordinary_coordinates: Vec<u32>,
    // Maximal `(start_tsid, end_tsid, coordinate)` runs of consecutive final
    // TSIDs sharing one non-MAX core coordinate. Precomputed once so `base_lift`
    // can emit one range per run instead of iterating every final TSID.
    coordinate_runs: Vec<(u32, u32, u32)>,
    mode_deviations: Vec<Vec<(u32, u32)>>,
    base_lifts: FxHashMap<usize, Weight>,
    mode_lifts: FxHashMap<(usize, usize), Weight>,
    // (core-weight body, exact mode-set id, entry-domain body).
    group_lifts: FxHashMap<(usize, usize, usize), Weight>,
    // Coordinate plans depend only on the exact transport-mode set, not on
    // the core state whose entry domain happens to reach that set.
    group_coordinate_plans: FxHashMap<Vec<usize>, GroupCoordinatePlan>,
    profile_base_lift_ms: f64,
    profile_plan_ms: f64,
    profile_signature_ms: f64,
    profile_override_ms: f64,
    profile_apply_ms: f64,
}

#[derive(Clone)]
struct GroupCoordinateSignature {
    base_coordinate_index: usize,
    alternate_coordinate_indices: Box<[usize]>,
}

struct GroupCoordinatePlan {
    #[cfg(debug_assertions)]
    reference_overrides: Vec<(u32, u32)>,
    signatures: Vec<GroupCoordinateSignature>,
    // Final TSIDs are grouped by the exact `(ordinary coordinate, alternate
    // coordinates)` signature that determines both their base and transported
    // token sets for every lifted core weight.
    final_tsid_runs_by_signature: Vec<Box<[(u32, u32)]>>,
    coordinates: Vec<u32>,
}

fn finite_weight_token_cardinality(weight: &Weight) -> Option<u128> {
    if weight.is_full() {
        return None;
    }
    Some(
        weight
            .range_entries()
            .map(|(start, end, tokens)| {
                let tsid_count = u128::from(end) - u128::from(start) + 1;
                let token_count: u128 = tokens
                    .ranges()
                    .map(|range| u128::from(*range.end()) - u128::from(*range.start()) + 1)
                    .sum();
                tsid_count * token_count
            })
            .sum(),
    )
}

impl PostDwaWeightLifter {
    fn new(
        core_state_map: &ManyToOneIdMap,
        final_state_map: &ManyToOneIdMap,
        modes: &[TerminalNwaTransportMode],
        active_mode_indices: &[usize],
    ) -> Self {
        let final_sources: Vec<u32> = final_state_map.iter_representative_ids().collect();
        let ordinary_coordinates: Vec<u32> = final_sources
            .iter()
            .map(|&source| Self::core_coordinate(core_state_map, source))
            .collect();
        // Precompute maximal contiguous runs of equal, non-MAX coordinates so
        // `base_lift` emits one range per run instead of iterating every final
        // TSID. MAX coordinates break runs (they are skipped when lifting).
        let mut coordinate_runs: Vec<(u32, u32, u32)> = Vec::new();
        {
            let mut idx = 0usize;
            let n = ordinary_coordinates.len();
            while idx < n {
                let coordinate = ordinary_coordinates[idx];
                if coordinate == u32::MAX {
                    idx += 1;
                    continue;
                }
                let start = idx;
                idx += 1;
                while idx < n && ordinary_coordinates[idx] == coordinate {
                    idx += 1;
                }
                coordinate_runs.push((start as u32, (idx - 1) as u32, coordinate));
            }
        }
        if std::env::var("GLRMASK_PROFILE_L2P_TIMING").is_ok() {
            eprintln!(
                "ti_post_dwa_lifter_setup final_sources={} coordinate_runs={}",
                final_sources.len(),
                coordinate_runs.len()
            );
        }
        let mut mode_deviations = vec![Vec::new(); modes.len()];
        let mut sparse_mode_ready = vec![false; modes.len()];
        let mut direct_modes_by_default = FxHashMap::<(usize, usize), Vec<usize>>::default();
        for &mode_index in active_mode_indices {
            if mode_index == 0 {
                continue;
            }
            if let Some(default_key) = modes[mode_index]
                .scanner_state_for_original
                .direct_quotient_default_key()
            {
                direct_modes_by_default
                    .entry(default_key)
                    .or_default()
                    .push(mode_index);
            }
        }

        // A direct quotient has one shared default representative per source
        // class and only sparse member-specific class deviations. Verify that
        // the shared default preserves every final coordinate before using the
        // deviation-only path; all other transport shapes retain the full scan.
        for mode_indices in direct_modes_by_default.into_values() {
            let domain = &modes[mode_indices[0]].scanner_state_for_original;
            let default_preserves_ordinary = final_sources
                .iter()
                .enumerate()
                .all(|(final_tsid, &source)| {
                    let source_class = domain.innermost_source_class(source);
                    Self::core_coordinate(
                        core_state_map,
                        domain.innermost_source_representative(source_class),
                    ) == ordinary_coordinates[final_tsid]
                });
            if !default_preserves_ordinary {
                continue;
            }

            let mut final_entries_by_source_class =
                vec![Vec::<(u32, u32)>::new(); domain.innermost_source_class_count()];
            for (final_tsid, &source) in final_sources.iter().enumerate() {
                final_entries_by_source_class[domain.innermost_source_class(source)].push((
                    final_tsid as u32,
                    ordinary_coordinates[final_tsid],
                ));
            }
            for mode_index in mode_indices {
                let deviations = &mut mode_deviations[mode_index];
                for &(input_class, output_class) in modes[mode_index]
                    .scanner_state_for_original
                    .quotient_deviations()
                    .expect("direct quotient transport must expose deviations")
                {
                    let Some(entries) = final_entries_by_source_class.get(input_class as usize) else {
                        continue;
                    };
                    let coordinate = Self::core_coordinate(
                        core_state_map,
                        domain.innermost_source_representative(output_class as usize),
                    );
                    deviations.extend(entries.iter().filter_map(
                        |&(final_tsid, ordinary_coordinate)| {
                            (coordinate != ordinary_coordinate)
                                .then_some((final_tsid, coordinate))
                        },
                    ));
                }
                deviations.sort_unstable_by_key(|&(final_tsid, _)| final_tsid);
                sparse_mode_ready[mode_index] = true;
            }
        }

        for &mode_index in active_mode_indices {
            if mode_index == 0 || sparse_mode_ready[mode_index] {
                continue;
            }
            let mode = &modes[mode_index];
            let deviations = &mut mode_deviations[mode_index];
            for (final_tsid, &source) in final_sources.iter().enumerate() {
                let coordinate = Self::core_coordinate(
                    core_state_map,
                    mode.scanner_state_for_original.scanner_state(source),
                );
                if coordinate != ordinary_coordinates[final_tsid] {
                    deviations.push((final_tsid as u32, coordinate));
                }
            }
        }

        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            let mut nonempty_modes = 0usize;
            let mut total_deviations = 0usize;
            let mut max_deviations = 0usize;
            for &mode_index in active_mode_indices {
                if mode_index == 0 {
                    continue;
                }
                let count = mode_deviations[mode_index].len();
                nonempty_modes += usize::from(count != 0);
                total_deviations += count;
                max_deviations = max_deviations.max(count);
            }
            eprintln!(
                "[glrmask/profile][ti_post_dwa_lifter] final_tsids={} coordinate_runs={} active_modes={} nonempty_modes={} total_deviations={} max_deviations={}",
                final_sources.len(),
                coordinate_runs.len(),
                active_mode_indices.len(),
                nonempty_modes,
                total_deviations,
                max_deviations,
            );
        }

        Self {
            ordinary_coordinates,
            coordinate_runs,
            mode_deviations,
            base_lifts: FxHashMap::default(),
            mode_lifts: FxHashMap::default(),
            group_lifts: FxHashMap::default(),
            group_coordinate_plans: FxHashMap::default(),
            profile_base_lift_ms: 0.0,
            profile_plan_ms: 0.0,
            profile_signature_ms: 0.0,
            profile_override_ms: 0.0,
            profile_apply_ms: 0.0,
        }
    }

    #[inline]
    fn core_coordinate(core_state_map: &ManyToOneIdMap, raw_state: u32) -> u32 {
        core_state_map
            .original_to_internal
            .get(raw_state as usize)
            .copied()
            .unwrap_or(u32::MAX)
    }

    #[inline]
    fn tokens_for_coordinate(weight: &Weight, coordinate: u32) -> SharedTokenSet {
        if coordinate == u32::MAX {
            shared_rangeset(range_set_blaze::RangeSetBlaze::new())
        } else {
            weight.shared_tokens_for_tsid(coordinate)
        }
    }

    fn base_lift(&mut self, weight: &Weight) -> Weight {
        if weight.is_empty() || weight.is_full() {
            return weight.clone();
        }
        let key = weight.ptr_key();
        if let Some(existing) = self.base_lifts.get(&key) {
            return existing.clone();
        }

        // `ordinary_coordinates[final_tsid]` is precomputed and identical to
        // `core_coordinate(core_state_map, final_sources[final_tsid])`. Thousands
        // of final TSIDs collapse onto only a few core coordinates, so iterate
        // the precomputed contiguous coordinate runs and emit one range per run.
        // The runs are sorted by coordinate (post-clustering relabel), so lift
        // them with a single linear merge against the weight's ranges instead
        // of one binary-search `get` per run.
        let lifted = weight.lift_sorted_coordinate_runs(&self.coordinate_runs);
        self.base_lifts.insert(key, lifted.clone());
        lifted
    }

    fn lift_for_mode(&mut self, weight: &Weight, mode_index: usize) -> Weight {
        if weight.is_empty() || weight.is_full() || mode_index == 0 {
            return self.base_lift(weight);
        }
        let key = (weight.ptr_key(), mode_index);
        if let Some(existing) = self.mode_lifts.get(&key) {
            return existing.clone();
        }

        let base = self.base_lift(weight);
        let overrides: Vec<(u32, SharedTokenSet)> = self.mode_deviations[mode_index]
            .iter()
            .map(|&(final_tsid, coordinate)| {
                (final_tsid, Self::tokens_for_coordinate(weight, coordinate))
            })
            .collect();
        let lifted = base.with_sparse_tsid_overrides(&overrides);
        self.mode_lifts.insert(key, lifted.clone());
        lifted
    }

    fn prepare_group_coordinate_plan(&mut self, mode_indices: &[usize]) {
        if mode_indices.len() <= 1 || self.group_coordinate_plans.contains_key(mode_indices) {
            return;
        }

        let mut alternates_by_final_tsid = vec![Vec::<u32>::new(); self.ordinary_coordinates.len()];
        for &mode_index in mode_indices {
            for &(final_tsid, coordinate) in &self.mode_deviations[mode_index] {
                let alternates = &mut alternates_by_final_tsid[final_tsid as usize];
                if !alternates.contains(&coordinate) {
                    alternates.push(coordinate);
                }
            }
        }

        let mut signature_for_key = FxHashMap::<(u32, Vec<u32>), u32>::default();
        let mut signatures = Vec::<GroupCoordinateSignature>::new();
        let mut overrides = Vec::<(u32, u32)>::new();
        for (final_tsid, alternates) in alternates_by_final_tsid.iter_mut().enumerate() {
            if alternates.is_empty() {
                continue;
            }
            alternates.sort_unstable();
            alternates.dedup();
            let base_coordinate = self.ordinary_coordinates[final_tsid];
            let key = (base_coordinate, alternates.clone());
            let signature = match signature_for_key.entry(key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let signature = signatures.len() as u32;
                    let (base_coordinate, alternate_coordinates) = entry.key();
                    signatures.push(GroupCoordinateSignature {
                        base_coordinate_index: *base_coordinate as usize,
                        alternate_coordinate_indices: alternate_coordinates.iter().map(|&coordinate| coordinate as usize).collect::<Vec<_>>().into_boxed_slice(),
                    });
                    entry.insert(signature);
                    signature
                }
            };
            overrides.push((final_tsid as u32, signature));
        }

        let mut coordinates = signatures
            .iter()
            .flat_map(|signature| {
                std::iter::once(signature.base_coordinate_index as u32).chain(
                    signature
                        .alternate_coordinate_indices
                        .iter()
                        .map(|&coordinate| coordinate as u32),
                )
            })
            .collect::<Vec<_>>();
        coordinates.sort_unstable();
        coordinates.dedup();
        for signature in &mut signatures {
            let raw_base = signature.base_coordinate_index as u32;
            signature.base_coordinate_index = coordinates
                .binary_search(&raw_base)
                .expect("group-plan base coordinate must be retained");
            for coordinate_index in signature.alternate_coordinate_indices.iter_mut() {
                let raw_coordinate = *coordinate_index as u32;
                *coordinate_index = coordinates
                    .binary_search(&raw_coordinate)
                    .expect("group-plan alternate coordinate must be retained");
            }
        }

        let mut final_tsids_by_signature = vec![Vec::<u32>::new(); signatures.len()];
        for &(final_tsid, signature) in &overrides {
            final_tsids_by_signature[signature as usize].push(final_tsid);
        }
        let final_tsid_runs_by_signature = final_tsids_by_signature
            .into_iter()
            .map(|final_tsids| {
                let mut runs = Vec::<(u32, u32)>::new();
                for final_tsid in final_tsids {
                    if let Some((_, end)) = runs.last_mut()
                        && end.checked_add(1) == Some(final_tsid)
                    {
                        *end = final_tsid;
                    } else {
                        runs.push((final_tsid, final_tsid));
                    }
                }
                runs.into_boxed_slice()
            })
            .collect();

        self.group_coordinate_plans.insert(
            mode_indices.to_vec(),
            GroupCoordinatePlan {
                #[cfg(debug_assertions)]
                reference_overrides: overrides,
                signatures,
                final_tsid_runs_by_signature,
                coordinates,
            },
        );
    }

    /// Union the exact lifted weight over one proven-disjoint transport group.
    fn lift_over_disjoint_group(
        &mut self,
        weight: &Weight,
        mode_set_id: usize,
        mode_indices: &[usize],
        entry_domain: &Weight,
    ) -> Weight {
        if weight.is_empty() || weight.is_full() {
            return weight.intersection(entry_domain);
        }
        if mode_indices.len() == 1 {
            return self
                .lift_for_mode(weight, mode_indices[0])
                .intersection(entry_domain);
        }
        let key = (weight.ptr_key(), mode_set_id, entry_domain.ptr_key());
        if let Some(existing) = self.group_lifts.get(&key) {
            return existing.clone();
        }

        let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
        let started_at = profile_timing.then(Instant::now);
        let base = self.base_lift(weight);
        if let Some(started_at) = started_at {
            self.profile_base_lift_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profile_timing.then(Instant::now);
        self.prepare_group_coordinate_plan(mode_indices);
        if let Some(started_at) = started_at {
            self.profile_plan_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profile_timing.then(Instant::now);
        let overrides = {
            let plan = self
                .group_coordinate_plans
                .get(mode_indices)
                .expect("prepared group coordinate plan must be retained");
            let mut union_for_token_signature = FxHashMap::<Vec<usize>, SharedTokenSet>::default();
            let coordinate_tokens = weight.shared_tokens_for_sorted_tsids(&plan.coordinates);
            let transformed_tokens: Vec<SharedTokenSet> = plan
                .signatures
                .iter()
                .map(|signature| {
                    let mut token_sets = Vec::with_capacity(signature.alternate_coordinate_indices.len() + 1);
                    token_sets.push(Arc::clone(&coordinate_tokens[signature.base_coordinate_index]));
                    for &coordinate_index in &signature.alternate_coordinate_indices {
                        token_sets.push(Arc::clone(&coordinate_tokens[coordinate_index]));
                    }
                    let mut token_signature: Vec<usize> = token_sets
                        .iter()
                        .filter(|tokens| !tokens.is_empty())
                        .map(|tokens| Arc::as_ptr(tokens) as usize)
                        .collect();
                    token_signature.sort_unstable();
                    token_signature.dedup();
                    if let Some(tokens) = union_for_token_signature.get(&token_signature) {
                        return Arc::clone(tokens);
                    }

                    let mut tokens = shared_rangeset(range_set_blaze::RangeSetBlaze::new());
                    for candidate in token_sets {
                        if candidate.is_empty() || candidate.as_ref() == tokens.as_ref() {
                            continue;
                        }
                        tokens = shared_rangeset(tokens.as_ref() | candidate.as_ref());
                    }
                    union_for_token_signature.insert(token_signature, Arc::clone(&tokens));
                    tokens
                })
                .collect();
            let mut overrides = Vec::<(u32, u32, SharedTokenSet)>::new();
            for (signature_index, signature) in plan.signatures.iter().enumerate() {
                let tokens = &transformed_tokens[signature_index];
                let base_tokens = &coordinate_tokens[signature.base_coordinate_index];
                if Arc::ptr_eq(tokens, base_tokens) || tokens.as_ref() == base_tokens.as_ref() {
                    continue;
                }
                overrides.extend(
                    plan.final_tsid_runs_by_signature[signature_index]
                        .iter()
                        .map(|&(start, end)| (start, end, Arc::clone(tokens))),
                );
            }
            // The ranged sparse-override helper consumes a TSID stream in
            // ascending coordinate order. Grouping by signature above changes
            // only construction order, so restore canonical final-TSID order
            // before applying the exact same overrides.
            overrides.sort_unstable_by_key(|&(start, _, _)| start);

            #[cfg(debug_assertions)]
            {
                let reference = plan
                    .reference_overrides
                    .iter()
                    .zip(base.shared_tokens_for_sorted_tsids(
                        &plan
                            .reference_overrides
                            .iter()
                            .map(|&(final_tsid, _)| final_tsid)
                            .collect::<Vec<_>>(),
                    ))
                    .filter_map(|(&(final_tsid, signature), base_tokens)| {
                        let tokens = &transformed_tokens[signature as usize];
                        let differs = !Arc::ptr_eq(tokens, &base_tokens)
                            && tokens.as_ref() != base_tokens.as_ref();
                        differs.then(|| (final_tsid, Arc::clone(tokens)))
                    })
                    .collect::<Vec<_>>();
                let expanded = overrides
                    .iter()
                    .flat_map(|(start, end, tokens)| {
                        (*start..=*end).map(move |final_tsid| {
                            (final_tsid, Arc::clone(tokens))
                        })
                    })
                    .collect::<Vec<_>>();
                debug_assert_eq!(expanded, reference);
            }

            overrides
        };
        if let Some(started_at) = started_at {
            self.profile_signature_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }

        let started_at = profile_timing.then(Instant::now);
        let lifted =
            base.with_sparse_tsid_range_overrides_intersection(&overrides, entry_domain);
        if let Some(started_at) = started_at {
            self.profile_apply_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        self.group_lifts.insert(key, lifted.clone());
        lifted
    }
}

/// Expand a minimized representative DWA directly into a raw-terminal DWA.
///
/// The previous implementation allocated one full suffix graph per replayed
/// member mode and asked generic DWA minimization to rediscover the common
/// representative topology. The direct construction shares that topology from
/// the start. It restores raw member labels at the dispatcher, then lifts each
/// shared suffix weight to the exact union of its reachable transported
/// final-coordinate behaviors.
pub fn expand_representative_dwa_after_minimization(
    core_dwa: &DWA,
    core_state_map: &ManyToOneIdMap,
    final_state_map: &ManyToOneIdMap,
    modes: &[TerminalNwaTransportMode],
) -> DWA {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some();
    let coordinate_setup_started_at = profile_timing.then(Instant::now);
    assert!(
        !modes.is_empty(),
        "post-DWA TI expansion needs the ordinary transport mode",
    );
    assert_eq!(
        modes[0].member_reconstruction(),
        None,
        "the first TI transport mode must be ordinary",
    );

    let mut member_modes_by_representative = BTreeMap::<TerminalID, Vec<(TerminalID, usize)>>::new();
    for (mode_index, mode) in modes.iter().enumerate().skip(1) {
        let (representative, member) = mode
            .member_reconstruction()
            .expect("non-ordinary TI transport mode must reconstruct one member");
        member_modes_by_representative
            .entry(representative)
            .or_default()
            .push((member, mode_index));
    }

    let core_start = core_dwa.start_state() as usize;
    let core_states = core_dwa.states();
    let core_start_transitions = &core_states[core_start].transitions;

    let mut candidate_mode_indices = vec![0usize];
    for (mode_index, mode) in modes.iter().enumerate().skip(1) {
        let (representative, _) = mode
            .member_reconstruction()
            .expect("non-ordinary TI transport mode must reconstruct one member");
        if core_start_transitions.contains_key(&(representative as i32)) {
            candidate_mode_indices.push(mode_index);
        }
    }

    // Drop modes whose first raw-member edge has no viable token flow. These
    // clones were unreachable in the old graph-copy construction and therefore
    // must not participate in a shared suffix-weight union.
    let mut lifter = PostDwaWeightLifter::new(
        core_state_map,
        final_state_map,
        modes,
        &candidate_mode_indices,
    );
    let coordinate_setup_ms = coordinate_setup_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let active_filter_started_at = profile_timing.then(Instant::now);
    let active_mode_indices: Vec<usize> = candidate_mode_indices
        .iter()
        .copied()
        .filter(|&mode_index| {
            if mode_index == 0 {
                return true;
            }
            let (representative, _) = modes[mode_index]
                .member_reconstruction()
                .expect("non-ordinary TI transport mode must reconstruct one member");
            let (_, weight) = core_start_transitions
                .get(&(representative as i32))
                .expect("active TI member mode must have its representative start edge");
            !lifter.lift_for_mode(weight, mode_index).is_empty()
        })
        .collect();
    if active_mode_indices != candidate_mode_indices {
        lifter = PostDwaWeightLifter::new(
            core_state_map,
            final_state_map,
            modes,
            &active_mode_indices,
        );
    }
    let active_filter_ms = active_filter_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let grouping_started_at = profile_timing.then(Instant::now);
    // Modes can share one suffix graph only when their first-edge domains are
    // disjoint. That domain is retained by DWA weight intersection after the
    // dispatcher, so merging their suffix weights by union cannot leak one
    // member's transported behavior into another member's execution.
    // The ordinary representative mode remains a singleton deliberately: it
    // has multiple possible first labels rather than one stable gate.
    let mut mode_groups = vec![vec![0usize]];
    // Exact entry unions computed while proving disjointness can be reused
    // later for suffix states with the same complete member-mode set.
    let mut known_member_entry_domain_by_mode_set = FxHashMap::<Vec<usize>, Weight>::default();
    let mut group_for_mode = vec![None::<usize>; modes.len()];
    group_for_mode[0] = Some(0);
    let member_entry_weights: Vec<(usize, u32, Weight)> = active_mode_indices
        .iter()
        .copied()
        .skip(1)
        .map(|mode_index| {
            let (representative, _) = modes[mode_index]
                .member_reconstruction()
                .expect("non-ordinary TI transport mode must reconstruct one member");
            let (target, start_weight) = core_start_transitions
                .get(&(representative as i32))
                .expect("active TI member mode must have its representative start edge");
            (mode_index, *target, lifter.lift_for_mode(start_weight, mode_index))
        })
        .collect();

    let all_member_entry_union =
        Weight::union_all(member_entry_weights.iter().map(|(_, _, weight)| weight));
    let member_entries_are_pairwise_disjoint = finite_weight_token_cardinality(&all_member_entry_union)
        .zip(
            member_entry_weights
                .iter()
                .map(|(_, _, weight)| finite_weight_token_cardinality(weight))
                .collect::<Option<Vec<_>>>()
                .map(|counts| counts.into_iter().sum()),
        )
        .is_some_and(|(union_count, member_count)| union_count == member_count);
    let ordinary_entry_weights: Vec<Weight> = core_start_transitions
        .values()
        .map(|(_, weight)| lifter.lift_for_mode(weight, 0))
        .collect();

    // The ordinary union itself is never needed. Ordinary/member disjointness
    // is exactly equivalent to every ordinary entry weight being disjoint from
    // the member union, so avoid materializing another potentially large weight.
    let ordinary_and_members_are_disjoint = member_entries_are_pairwise_disjoint
        && ordinary_entry_weights
            .iter()
            .all(|weight| weight.is_disjoint(&all_member_entry_union));
    if ordinary_and_members_are_disjoint {
        let modes_in_group: Vec<usize> = member_entry_weights
            .iter()
            .map(|(mode_index, _, _)| *mode_index)
            .collect();
        for &mode_index in &modes_in_group {
            group_for_mode[mode_index] = Some(0);
        }
        mode_groups[0].extend(modes_in_group);
    } else if member_entries_are_pairwise_disjoint {
        let group_index = mode_groups.len();
        let modes_in_group: Vec<usize> = member_entry_weights
            .iter()
            .map(|(mode_index, _, _)| *mode_index)
            .collect();
        for &mode_index in &modes_in_group {
            group_for_mode[mode_index] = Some(group_index);
        }
        known_member_entry_domain_by_mode_set
            .insert(modes_in_group.clone(), all_member_entry_union.clone());
        mode_groups.push(modes_in_group);
    } else {
        // First-fit coloring only needs to know whether the incoming entry
        // domain overlaps anything already assigned to a group. The old path
        // materialized a new immutable `Weight` union after every insertion;
        // on o21328 p0 those 239 unions account for essentially the entire
        // grouping cost, even though the resulting unions are never consumed
        // after coloring.
        //
        // Keep the exact occupied token sets directly in the already-compact
        // final TSID coordinate domain instead. Since every member inside one
        // group is pairwise disjoint, storing its component token sets is
        // sufficient: a candidate conflicts with a group iff one of its token
        // sets intersects one occupied component at the same TSID. Shared token
        // sets are interned, so cache the small number of structural overlap
        // queries by Arc identity.
        let final_tsid_count = final_state_map.representative_original_ids.len();
        let mut occupied_by_tsid =
            vec![Vec::<(usize, SharedTokenSet)>::new(); final_tsid_count];
        let mut groups_with_full_domain = Vec::<bool>::with_capacity(modes.len() + 1);
        groups_with_full_domain.push(false); // ordinary group
        let mut forbidden_epoch = vec![0u32; modes.len() + 1];
        let mut epoch = 0u32;
        let mut token_overlap_cache = FxHashMap::<(usize, usize), bool>::default();

        for (mode_index, _, entry_weight) in &member_entry_weights {
            epoch = epoch.wrapping_add(1);
            if epoch == 0 {
                forbidden_epoch.fill(0);
                epoch = 1;
            }

            if entry_weight.is_full() {
                for group in 1..mode_groups.len() {
                    forbidden_epoch[group] = epoch;
                }
            } else {
                for (start, end, tokens) in entry_weight.range_entries() {
                    debug_assert!((end as usize) < final_tsid_count);
                    for tsid in start..=end {
                        for (group, occupied_tokens) in &occupied_by_tsid[tsid as usize] {
                            if forbidden_epoch[*group] == epoch {
                                continue;
                            }
                            let left = Arc::as_ptr(tokens) as usize;
                            let right = Arc::as_ptr(occupied_tokens) as usize;
                            let key = if left <= right {
                                (left, right)
                            } else {
                                (right, left)
                            };
                            let overlaps = *token_overlap_cache.entry(key).or_insert_with(|| {
                                !tokens.as_ref().is_disjoint(occupied_tokens.as_ref())
                            });
                            if overlaps {
                                forbidden_epoch[*group] = epoch;
                            }
                        }
                    }
                }
                for (group, &full) in groups_with_full_domain.iter().enumerate().skip(1) {
                    if full {
                        forbidden_epoch[group] = epoch;
                    }
                }
            }

            let group_index = (1..mode_groups.len())
                .find(|&group| forbidden_epoch[group] != epoch)
                .unwrap_or_else(|| {
                    let group = mode_groups.len();
                    mode_groups.push(Vec::new());
                    groups_with_full_domain.push(false);
                    group
                });

            if entry_weight.is_full() {
                groups_with_full_domain[group_index] = true;
            } else {
                for (start, end, tokens) in entry_weight.range_entries() {
                    for tsid in start..=end {
                        occupied_by_tsid[tsid as usize]
                            .push((group_index, Arc::clone(tokens)));
                    }
                }
            }
            mode_groups[group_index].push(*mode_index);
            group_for_mode[*mode_index] = Some(group_index);
        }
    }

    let mut core_reachable_from = vec![vec![false; core_states.len()]; core_states.len()];
    let grouping_pre_reach_ms = grouping_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    for source in 0..core_states.len() {
        let mut stack = vec![source];
        while let Some(state) = stack.pop() {
            if core_reachable_from[source][state] {
                continue;
            }
            core_reachable_from[source][state] = true;
            for (target, _) in core_states[state].transitions.values() {
                if (*target as usize) < core_states.len() {
                    stack.push(*target as usize);
                }
            }
        }
    }

    let mut mode_indices_at_core_state =
        vec![vec![Vec::<usize>::new(); core_states.len()]; mode_groups.len()];
    let grouping_reach_ms = grouping_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let mut entry_weights_at_core_state =
        vec![vec![Vec::<Weight>::new(); core_states.len()]; mode_groups.len()];

    for (_, (target, weight)) in core_start_transitions {
        let entry_weight = lifter.lift_for_mode(weight, 0);
        if entry_weight.is_empty() {
            continue;
        }
        for (core_state, reachable) in core_reachable_from[*target as usize].iter().enumerate() {
            if *reachable {
                entry_weights_at_core_state[0][core_state].push(entry_weight.clone());
            }
        }
    }
    for (mode_index, target, entry_weight) in &member_entry_weights {
        let group_index = group_for_mode[*mode_index]
            .expect("active TI member mode must belong to one direct suffix group");
        for (core_state, reachable) in core_reachable_from[*target as usize].iter().enumerate() {
            if *reachable {
                mode_indices_at_core_state[group_index][core_state].push(*mode_index);
                entry_weights_at_core_state[group_index][core_state].push(entry_weight.clone());
            }
        }
    }
    for core_state in 0..core_states.len() {
        if !entry_weights_at_core_state[0][core_state].is_empty() {
            mode_indices_at_core_state[0][core_state].push(0);
        }
    }
    // Member entry domains are determined solely by the exact active mode
    // set. Several p0 suffix states share the complete 1,609-member set, so
    // retain one exact union rather than rebuilding it per core state.
    let mut member_entry_domain_by_mode_set = known_member_entry_domain_by_mode_set;
    let entry_domain_at_core_state: Vec<Vec<Weight>> = entry_weights_at_core_state
        .iter()
        .enumerate()
        .map(|(group_index, by_core_state)| {
            by_core_state
                .iter()
                .enumerate()
                .map(|(core_state, weights)| {
                    if group_index == 0 {
                        return Weight::union_all(weights.iter());
                    }
                    let mode_set = &mode_indices_at_core_state[group_index][core_state];
                    if mode_set.is_empty() {
                        return Weight::empty();
                    }
                    member_entry_domain_by_mode_set
                        .entry(mode_set.clone())
                        .or_insert_with(|| Weight::union_all(weights.iter()))
                        .clone()
                })
                .collect()
        })
        .collect();

    let mut mode_set_id_by_contents = FxHashMap::<Vec<usize>, usize>::default();
    let mode_set_id_at_core_state: Vec<Vec<Option<usize>>> = mode_indices_at_core_state
        .iter()
        .map(|by_core_state| {
            by_core_state
                .iter()
                .map(|mode_set| {
                    if mode_set.is_empty() {
                        return None;
                    }
                    let next = mode_set_id_by_contents.len();
                    Some(*mode_set_id_by_contents
                        .entry(mode_set.clone())
                        .or_insert(next))
                })
                .collect()
        })
        .collect();

    let grouping_ms = grouping_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    if profile_timing {
        let grouping_entryw_ms = grouping_ms - grouping_reach_ms;
        let grouping_reach_only_ms = grouping_reach_ms - grouping_pre_reach_ms;
        eprintln!(
            "[glrmask/profile][ti_post_dwa_grouping_split] pre_reach_ms={:.3} reach_ms={:.3} entryw_ms={:.3} total_ms={:.3}",
            grouping_pre_reach_ms,
            grouping_reach_only_ms,
            grouping_entryw_ms,
            grouping_ms,
        );
    }

    if profile_timing {
        let member_group_count = mode_groups.len().saturating_sub(1);
        let largest_member_group = mode_groups
            .iter()
            .skip(1)
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        eprintln!(
            "[glrmask/profile][ti_post_dwa_direct_groups] core_states={} active_modes={} member_groups={} largest_member_group={} direct_states_before_follow={}",
            core_states.len(),
            active_mode_indices.len(),
            member_group_count,
            largest_member_group,
            1 + mode_groups.len() * core_states.len(),
        );
    }

    let state_for = |group_index: usize, core_state_index: usize| -> u32 {
        (1 + group_index * core_states.len() + core_state_index) as u32
    };
    let shared_build_started_at = profile_timing.then(Instant::now);
    let mut final_lift_ms = 0.0;
    let mut transition_lift_ms = 0.0;
    let mut raw_label_insert_ms = 0.0;
    let mut final_lift_calls = 0usize;
    let mut transition_lift_calls = 0usize;
    let mut raw_label_inserts = 0usize;
    let mut states = vec![DWAState::default(); 1 + mode_groups.len() * core_states.len()];
    for group_index in 0..mode_groups.len() {
        for (core_state_index, core_state) in core_states.iter().enumerate() {
            let mode_indices = &mode_indices_at_core_state[group_index][core_state_index];
            let entry_domain = &entry_domain_at_core_state[group_index][core_state_index];
            if mode_indices.is_empty() || entry_domain.is_empty() {
                continue;
            }
            let mode_set_id = mode_set_id_at_core_state[group_index][core_state_index]
                .expect("active suffix state must have a mode-set id");
            let final_weight = core_state.final_weight.as_ref().map(|weight| {
                let started_at = profile_timing.then(Instant::now);
                let lifted = lifter.lift_over_disjoint_group(weight, mode_set_id, mode_indices, entry_domain);
                if let Some(started_at) = started_at {
                    final_lift_ms += started_at.elapsed().as_secs_f64() * 1000.0;
                }
                final_lift_calls += 1;
                lifted
            });
            let mut transitions = BTreeMap::new();
            for (&label, (target, weight)) in &core_state.transitions {
                let lift_started_at = profile_timing.then(Instant::now);
                let lifted_weight = lifter.lift_over_disjoint_group(
                    weight,
                    mode_set_id,
                    mode_indices,
                    entry_domain,
                );
                if let Some(started_at) = lift_started_at {
                    transition_lift_ms += started_at.elapsed().as_secs_f64() * 1000.0;
                }
                transition_lift_calls += 1;
                if lifted_weight.is_empty() {
                    continue;
                }
                let destination = state_for(group_index, *target as usize);
                assert!(
                    transitions
                        .insert(label, (destination, lifted_weight.clone()))
                        .is_none(),
                    "representative DWA must be deterministic before TI expansion",
                );

                // First terminal selection is handled only by the dispatcher:
                // its member-specific transport determines the entire suffix.
                if core_state_index == core_start || label < 0 {
                    continue;
                }
                let representative = label as TerminalID;
                for &(member, _) in member_modes_by_representative
                    .get(&representative)
                    .into_iter()
                    .flatten()
                {
                    let raw_label_started_at = profile_timing.then(Instant::now);
                    assert!(
                        transitions
                            .insert(member as i32, (destination, lifted_weight.clone()))
                            .is_none(),
                        "one raw member must belong to exactly one TI representative class",
                    );
                    if let Some(started_at) = raw_label_started_at {
                        raw_label_insert_ms += started_at.elapsed().as_secs_f64() * 1000.0;
                    }
                    raw_label_inserts += 1;
                }
            }
            states[state_for(group_index, core_state_index) as usize] = DWAState {
                transitions: transitions.into(),
                final_weight,
            };
        }
    }

    let ordinary_start = &core_states[core_start];
    let mut dispatcher_transitions = BTreeMap::new();
    for (&label, (target, weight)) in &ordinary_start.transitions {
        let ordinary_weight = lifter.lift_for_mode(weight, 0);
        if ordinary_weight.is_empty() {
            continue;
        }
        assert!(
            dispatcher_transitions
                .insert(label, (state_for(0, *target as usize), ordinary_weight))
                .is_none(),
            "representative initial DWA edge labels must be unique",
        );
    }
    for (representative, member_modes) in &member_modes_by_representative {
        let Some((target, weight)) = core_start_transitions.get(&(*representative as i32)) else {
            continue;
        };
        for &(member, mode_index) in member_modes {
            let Some(group_index) = group_for_mode[mode_index] else {
                continue;
            };
            let member_weight = lifter.lift_for_mode(weight, mode_index);
            if member_weight.is_empty() {
                continue;
            }
            assert!(
                dispatcher_transitions
                    .insert(
                        member as i32,
                        (state_for(group_index, *target as usize), member_weight),
                    )
                    .is_none(),
                "one raw member must belong to exactly one TI representative class",
            );
        }
    }
    states[0] = DWAState {
        transitions: dispatcher_transitions.into(),
        final_weight: ordinary_start
            .final_weight
            .as_ref()
            .map(|weight| lifter.lift_for_mode(weight, 0)),
    };

    if let Some(started_at) = shared_build_started_at {
        eprintln!(
            "[glrmask/profile][ti_post_dwa_direct_detail] coordinate_setup_ms={:.3} active_filter_ms={:.3} grouping_ms={:.3} shared_build_ms={:.3} final_lift_calls={} final_lift_ms={:.3} transition_lift_calls={} transition_lift_ms={:.3} raw_label_inserts={} raw_label_insert_ms={:.3} base_lifts={} mode_lifts={} group_lifts={} coordinate_plans={} group_base_ms={:.3} group_plan_ms={:.3} group_signature_ms={:.3} group_apply_ms={:.3}",
            coordinate_setup_ms,
            active_filter_ms,
            grouping_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
            final_lift_calls,
            final_lift_ms,
            transition_lift_calls,
            transition_lift_ms,
            raw_label_inserts,
            raw_label_insert_ms,
            lifter.base_lifts.len(),
            lifter.mode_lifts.len(),
            lifter.group_lifts.len(),
            lifter.group_coordinate_plans.len(),
            lifter.profile_base_lift_ms,
            lifter.profile_plan_ms,
            lifter.profile_signature_ms,
            lifter.profile_apply_ms,
        );
    }

    DWA::from_parts(states, 0)
}

/// Restore the original raw-terminal follow relation after building and
/// minimizing a representative-only core. This is the deterministic product of
/// the expanded DWA with the same one-previous-terminal guard used by the NWA
/// postprocess. It deliberately performs no further NWA construction,
/// determinization, or minimization.
pub struct RawFollowRestoration {
    pub dwa: DWA,
    pub used_follow_row_quotient: bool,
}

pub fn restore_raw_follow_constraints_after_expansion(
    expanded_dwa: &DWA,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: usize,
    ignore_terminal: Option<TerminalID>,
) -> RawFollowRestoration {
    restore_raw_follow_constraints_after_expansion_with_transparent(
        expanded_dwa, disallowed_follows, num_terminals, ignore_terminal, None,
    )
}

/// The same predecessor-follow product with additional scoped-transparent
/// terminal labels. A transparent label neither checks nor replaces the
/// predecessor; parser component ownership is enforced later by its transfer.
/// This matches the ordinary NWA follow product used before TI expansion.
pub fn restore_raw_follow_constraints_after_expansion_with_transparent(
    expanded_dwa: &DWA,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: usize,
    ignore_terminal: Option<TerminalID>,
    follow_transparent: Option<&BitSet>,
) -> RawFollowRestoration {
    restore_raw_follow_constraints_impl(expanded_dwa,disallowed_follows,num_terminals,
        ignore_terminal,follow_transparent,false)
}

/// Exact predecessor-row quotient for a boundary-only follow product. The
/// previous symbol is observed only via its disallowed-successor row; grouping
/// equal rows is a congruence. This changes neither labels nor edge weights.
/// Keep the ordinary caller's historical shape policy separate.
#[cfg(any(test,feature="internal-api"))]
pub fn restore_boundary_follow_constraints(
    expanded_dwa:&DWA,disallowed_follows:&BTreeMap<u32,BitSet>,num_terminals:usize,
    ignore_terminal:Option<TerminalID>,
)->RawFollowRestoration{
    restore_raw_follow_constraints_impl(expanded_dwa,disallowed_follows,num_terminals,
        ignore_terminal,None,true)
}

fn restore_raw_follow_constraints_impl(
    expanded_dwa:&DWA,disallowed_follows:&BTreeMap<u32,BitSet>,num_terminals:usize,
    ignore_terminal:Option<TerminalID>,follow_transparent:Option<&BitSet>,
    boundary_row_quotient:bool,
)->RawFollowRestoration{
    // Every query below is for an in-range raw terminal. Borrow the original
    // rows instead of normalizing by cloning `num_terminals` full bitsets. The
    // former normalization was several milliseconds by itself on p0/p1 even
    // when the ensuing product was tiny.
    let rows_for_terminal: Vec<Option<&BitSet>> = (0..num_terminals)
        .map(|terminal| {
            disallowed_follows
                .get(&(terminal as u32))
                .filter(|row| !row.is_zero())
        })
        .collect();
    if !rows_for_terminal.iter().flatten().any(|row| {
        row.iter().any(|disallowed| disallowed < num_terminals)
    }) {
        return RawFollowRestoration {
            dwa: expanded_dwa.clone(),
            used_follow_row_quotient: false,
        };
    }

    // A previous terminal is observed only through its follow row. On the
    // tiny direct suffix shapes this exact quotient avoids constructing one
    // otherwise identical product state per raw terminal. Retain raw
    // predecessor identity on larger shapes because pointwise minimization's
    // canonical representative choice is intentionally shape-sensitive there.
    let follow_row_class_count = rows_for_terminal
        .iter()
        .copied()
        .collect::<FxHashSet<_>>()
        .len();
    // This exact quotient is particularly valuable for compact direct suffix
    // graphs.  Bound the product shape so large general TI artifacts retain
    // their historical canonicalization path.
    let use_follow_row_quotient = if boundary_row_quotient {
        expanded_dwa.states().len().checked_mul(follow_row_class_count)
            .is_some_and(|states|states <= 2_000_000)
    }else{
        expanded_dwa.states().len() <= 24 && follow_row_class_count <= 64
            && expanded_dwa.states().len()*follow_row_class_count <= 1200
    };
    let (previous_key_for_terminal, follow_rows) = if use_follow_row_quotient {
        let mut class_for_row = FxHashMap::<Option<&BitSet>, u32>::default();
        let mut rows = Vec::<Option<&BitSet>>::new();
        let keys = rows_for_terminal
            .iter()
            .copied()
            .map(|row| {
                let next = rows.len() as u32;
                *class_for_row.entry(row).or_insert_with(|| {
                    rows.push(row);
                    next
                })
            })
            .collect::<Vec<_>>();
        (keys, rows)
    } else {
        ((0..num_terminals as u32).collect(), rows_for_terminal)
    };

    type ProductState = (u32, Option<u32>);
    let mut state_ids = FxHashMap::<ProductState, u32>::default();
    let mut worklist = VecDeque::<ProductState>::new();
    let mut states = Vec::<DWAState>::new();

    let get_or_create = |product: ProductState,
                         state_ids: &mut FxHashMap<ProductState, u32>,
                         worklist: &mut VecDeque<ProductState>,
                         states: &mut Vec<DWAState>| {
        if let Some(&id) = state_ids.get(&product) {
            return id;
        }
        let id = states.len() as u32;
        state_ids.insert(product, id);
        worklist.push_back(product);
        states.push(DWAState::default());
        id
    };

    let start = get_or_create(
        (expanded_dwa.start_state(), None),
        &mut state_ids,
        &mut worklist,
        &mut states,
    );
    while let Some((dwa_state, previous_key)) = worklist.pop_front() {
        let result_state = state_ids[&(dwa_state, previous_key)] as usize;
        let source = &expanded_dwa.states()[dwa_state as usize];
        states[result_state].final_weight = source.final_weight.clone();

        for (&label, (target, weight)) in &source.transitions {
            let next_previous_key = if label < 0
                || ignore_terminal.is_some_and(|ignore| label as TerminalID == ignore)
                || follow_transparent.is_some_and(|bits| bits.get(label as usize))
            {
                previous_key
            } else if (label as usize) < previous_key_for_terminal.len() {
                let terminal = label as usize;
                if previous_key.is_some_and(|previous| {
                    follow_rows[previous as usize].is_some_and(|row| row.contains(terminal))
                }) {
                    continue;
                }
                Some(previous_key_for_terminal[terminal])
            } else {
                None
            };
            let destination = get_or_create(
                (*target, next_previous_key),
                &mut state_ids,
                &mut worklist,
                &mut states,
            );
            let previous = states[result_state]
                .transitions
                .insert(label, (destination, weight.clone()));
            assert!(
                previous.is_none(),
                "expanded DWA and raw follow product must remain deterministic",
            );
        }
    }

    if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][ti_post_dwa_raw_follow] source_states={} follow_row_classes={} used_row_quotient={} result_states={}",
            expanded_dwa.states().len(),
            follow_row_class_count,
            use_follow_row_quotient,
            states.len(),
        );
    }
    RawFollowRestoration {
        dwa: DWA::from_parts(states, start),
        used_follow_row_quotient: use_follow_row_quotient,
    }
}


/// Return a topological state order when every in-range transition is acyclic.
///
/// This small local helper intentionally mirrors the minimizer's Kahn pass
/// instead of exposing a general graph API just for post-DWA normalization.
fn forward_domain_topological_order(dwa: &DWA) -> Option<Vec<usize>> {
    let state_count = dwa.states().len();
    let mut indegree = vec![0u32; state_count];
    for state in dwa.states() {
        for (_, (target, _)) in &state.transitions {
            let target = *target as usize;
            if target < state_count {
                indegree[target] = indegree[target].saturating_add(1);
            }
        }
    }

    let mut queue = indegree
        .iter()
        .enumerate()
        .filter_map(|(state, &degree)| (degree == 0).then_some(state))
        .collect::<Vec<_>>();
    let mut head = 0usize;
    let mut order = Vec::with_capacity(state_count);
    while head < queue.len() {
        let source = queue[head];
        head += 1;
        order.push(source);
        for (_, (target, _)) in &dwa.states()[source].transitions {
            let target = *target as usize;
            if target >= state_count {
                continue;
            }
            indegree[target] -= 1;
            if indegree[target] == 0 {
                queue.push(target);
            }
        }
    }
    (order.len() == state_count).then_some(order)
}

/// Exact intersection cache scoped to one forward-domain normalization pass.
///
/// The global Weight operation memo protects correctness and reuse across the
/// compiler, but this tight pass repeatedly intersects the same weight bodies
/// with the same per-state domains. Keeping strong local results avoids weak
/// interner/memo traffic and also shares the final restriction rewrite.
fn forward_domain_intersection(
    cache: &mut FxHashMap<(usize, usize), Weight>,
    left: &Weight,
    right: &Weight,
) -> Weight {
    if left.is_empty() || right.is_empty() {
        return Weight::empty();
    }
    if left.is_full() {
        return right.clone();
    }
    if right.is_full() || left.ptr_key() == right.ptr_key() {
        return left.clone();
    }
    let (left_key, right_key) = if left.ptr_key() <= right.ptr_key() {
        (left.ptr_key(), right.ptr_key())
    } else {
        (right.ptr_key(), left.ptr_key())
    };
    cache
        .entry((left_key, right_key))
        .or_insert_with(|| left.intersection(right))
        .clone()
}

/// Collapse a target's pending domain contributions once, at the point where a
/// topological traversal proves all predecessors have contributed. Pointer
/// deduplication is exact because each Weight body is immutable.
fn union_forward_domain_parts(parts: &mut Vec<Weight>) -> Weight {
    if parts.is_empty() {
        return Weight::empty();
    }
    if parts.iter().any(Weight::is_full) {
        parts.clear();
        return Weight::all();
    }
    parts.retain(|weight| !weight.is_empty());
    if parts.is_empty() {
        return Weight::empty();
    }
    if parts.len() == 1 {
        return parts.pop().expect("one retained forward-domain part");
    }
    parts.sort_unstable_by_key(Weight::ptr_key);
    parts.dedup_by_key(|weight| weight.ptr_key());
    if parts.len() == 1 {
        return parts.pop().expect("one deduplicated forward-domain part");
    }
    Weight::union_all(parts.iter())
}

/// Propagate forward domains and rewrite the product DWA in one exact
/// topological pass. Every incoming contribution is the same intersection
/// that becomes the normalized transition weight, so retaining it avoids
/// computing `source_domain ∩ transition_weight` twice.
fn normalize_weights_to_forward_domains_acyclic_in_place(dwa: &mut DWA) -> bool {
    let state_count = dwa.states().len();
    let Some(order) = forward_domain_topological_order(dwa) else {
        return false;
    };
    let start = dwa.start_state() as usize;
    if start >= state_count {
        return true;
    }

    let mut pending = vec![Vec::<Weight>::new(); state_count];
    pending[start].push(Weight::all());
    let mut intersection_cache = FxHashMap::<(usize, usize), Weight>::default();

    for source in order {
        let source_domain = union_forward_domain_parts(&mut pending[source]);
        let state = &mut dwa.states_mut()[source];
        if source_domain.is_empty() {
            state.final_weight = None;
            state.transitions.clear();
            continue;
        }

        state.final_weight = state
            .final_weight
            .as_ref()
            .map(|weight| forward_domain_intersection(&mut intersection_cache, weight, &source_domain))
            .filter(|weight| !weight.is_empty());
        state.transitions.retain(|_, (target, weight)| {
            let normalized = forward_domain_intersection(
                &mut intersection_cache,
                weight,
                &source_domain,
            );
            let target = *target as usize;
            if target < state_count && !normalized.is_empty() {
                pending[target].push(normalized.clone());
            }
            *weight = normalized;
            !weight.is_empty()
        });
    }

    true
}

fn forward_domains_fixed_point(dwa: &DWA) -> Vec<Weight> {
    let state_count = dwa.states().len();
    let mut domains = vec![Weight::empty(); state_count];
    let mut worklist = VecDeque::new();
    let start = dwa.start_state() as usize;
    domains[start] = Weight::all();
    worklist.push_back(start);
    let mut intersection_cache = FxHashMap::<(usize, usize), Weight>::default();
    while let Some(source) = worklist.pop_front() {
        let source_domain = domains[source].clone();
        for (target, weight) in dwa.states()[source].transitions.values() {
            let target = *target as usize;
            if target >= state_count {
                continue;
            }
            let incoming = forward_domain_intersection(
                &mut intersection_cache,
                &source_domain,
                weight,
            );
            if incoming.is_empty() {
                continue;
            }
            let merged = domains[target].union(&incoming);
            if merged != domains[target] {
                domains[target] = merged;
                worklist.push_back(target);
            }
        }
    }
    domains
}

/// Restrict each final/transition weight to coordinates that can reach its
/// source state from the DWA start, mutating an already-owned product DWA.
/// This preserves every completed path while dropping unreachable
/// transport-factor fragments before minimization without cloning the full
/// raw-follow state vector a second time.
pub fn restrict_weights_to_forward_domains_in_place(dwa: &mut DWA) {
    let state_count = dwa.states().len();
    if state_count == 0 || (dwa.start_state() as usize) >= state_count {
        return;
    }

    if normalize_weights_to_forward_domains_acyclic_in_place(dwa) {
        return;
    }

    let domains = forward_domains_fixed_point(dwa);
    let mut restriction_cache = FxHashMap::<(usize, usize), Weight>::default();
    for (state, domain) in dwa.states_mut().iter_mut().zip(domains.iter()) {
        state.final_weight = state
            .final_weight
            .as_ref()
            .map(|weight| forward_domain_intersection(&mut restriction_cache, weight, domain))
            .filter(|weight| !weight.is_empty());
        state.transitions.retain(|_, (_, weight)| {
            *weight = forward_domain_intersection(&mut restriction_cache, weight, domain);
            !weight.is_empty()
        });
    }
}

/// Owned convenience wrapper for tests and callers that need to retain the
/// unnormalized input DWA.
pub fn restrict_weights_to_forward_domains(dwa: &DWA) -> DWA {
    let mut normalized = dwa.clone();
    restrict_weights_to_forward_domains_in_place(&mut normalized);
    normalized
}

#[cfg(test)]
mod tests ;

#[cfg(test)]
#[test]
fn scoped_follow_restoration_matches_global_transparency_and_keeps_predecessor() {
    let mut source = DWA::new(1, 2);
    for _ in 0..3 { source.add_state(); }
    source.states_mut()[0].transitions.insert(0, (1, Weight::all()));
    source.states_mut()[1].transitions.insert(2, (2, Weight::all()));
    source.states_mut()[2].transitions.insert(1, (3, Weight::all()));
    source.states_mut()[3].final_weight = Some(Weight::all());
    let mut transparent = BitSet::new(3); transparent.set(2);
    for block_real_successor in [false, true] {
        let mut blocked = BitSet::new(3); blocked.set(2);
        if block_real_successor { blocked.set(1); }
        let follows = BTreeMap::from([(0, blocked), (2, BitSet::all(3))]);
        let expected = restore_raw_follow_constraints_after_expansion(&source, &follows, 3, Some(2));
        let actual = restore_raw_follow_constraints_after_expansion_with_transparent(
            &source, &follows, 3, None, Some(&transparent),
        );
        assert_eq!(expected.dwa.start_state(), actual.dwa.start_state());
        assert_eq!(expected.dwa.states(), actual.dwa.states());
        let mut q = Some(actual.dwa.start_state());
        for label in [0,2,1] { q = q.and_then(|q| actual.dwa.states()[q as usize].transitions.get(&label).map(|&(target,_)| target)); }
        let accepts = q.is_some_and(|q| actual.dwa.states()[q as usize].final_weight.as_ref().is_some_and(|w| !w.is_empty()));
        assert_eq!(accepts, !block_real_successor);
    }
}

#[cfg(test)]
#[test]
fn boundary_follow_row_quotient_matches_large_unquotiented_products() {
    let mut seed=19u64;
    let mut random=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32)as usize};
    let mut selected=0;
    for case in 0..128 {
        let mut source=DWA::new(0,4);for _ in 1..40{source.add_state();}
        for q in 0..40 {
            let mask=random()as u32%16;
            source.set_final_weight(q,Weight::from_token_set_for_tsid(0,(0..4).filter(|b|mask&(1<<b)!=0).collect()));
            if q==39{continue;}
            for label in 0..12 {
                if random()%3==0{continue;}
                let target=q+1+random()as u32%(39-q);let mask=random()as u32%16;
                source.add_transition(q,label,target,Weight::from_token_set_for_tsid(0,(0..4).filter(|b|mask&(1<<b)!=0).collect()));
            }
        }
        let rows=(0..12).map(|t|{let mut row=BitSet::new(12);for next in 0..12{if(t%3+next)%4==0{row.set(next);}}(t as u32,row)}).collect();
        let ignore=(case%2==0).then_some(11);
        let reference=restore_raw_follow_constraints_after_expansion(&source,&rows,12,ignore);
        let candidate=restore_boundary_follow_constraints(&source,&rows,12,ignore);
        assert!(!reference.used_follow_row_quotient);
        selected+=usize::from(candidate.used_follow_row_quotient);
        assert!(crate::automata::weighted::equivalence::find_difference(&reference.dwa,&candidate.dwa).unwrap().is_none());
        assert!(crate::automata::weighted::equivalence::find_difference(&candidate.dwa,&reference.dwa).unwrap().is_none());
    }
    assert_eq!(selected,128);
}
