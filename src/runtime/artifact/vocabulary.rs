//! Dynamic vocabulary ownership, construction, and token-coordinate maps.
//!
//! Definition order, cache ownership, and lazy preparation are intentionally
//! retained. Other private modules implement proof, cache, and transfer operations.

use super::{
    DYNAMIC_MASK_LLG_MASTER_CACHE_ID, DirectRegularDynamicFrontierCacheEntry,
    DirectRegularTerminalSupport, DynamicBoundedObservationSets, DynamicDenseSubset16,
    DynamicLazyUnionCache, DynamicMaskCache, DynamicMaskSliceTrie, DynamicMaskTrie,
    FastTokenizerTransitions, dynamic_mask_llg_master_is_whitespace,
    dynamic_mask_llg_master_safe_chars,
};
use crate::automata::lexer::runtime_repeat_product::VirtualBinaryRepeatIntersectionMaskProjection;
use crate::automata::lexer::runtime_unit_repeat::VirtualZeroMinUnitRepeatMaskProjection;
use crate::automata::lexer::tokenizer::TerminalProjectedQuotient;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::tokenizer::VirtualResidualMaskProjection;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::grammar::flat::TerminalID;
use rustc_hash::FxHashMap;
use rustc_hash::FxHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum PackedDynamicMaskTokenAliases {
    Single(u32),
    Many(Box<[u32]>),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum DynamicMaskAliasStore {
    Ordered(Arc<Vec<Vec<u32>>>),
    /// Canonical alias groups already flattened into the runtime hot layout.
    /// Offsets has `canonical_count + 1` entries into `originals`.
    Flat {
        offsets: Arc<Vec<u32>>,
        originals: Arc<Vec<u32>>,
    },
    Packed(Arc<Vec<Option<PackedDynamicMaskTokenAliases>>>),
}

#[inline]
pub(crate) fn dynamic_mask_state_key_hash(state: &DynamicMaskStateKey) -> u64 {
    let mut hasher = FxHasher::default();
    state.hash(&mut hasher);
    hasher.finish()
}

/// Canonical semantic snapshot of a dynamic-mask residual. Flattening the GSS
/// deliberately removes representation-only Arc identities and accumulator
/// node organization, so equivalent residuals reached after different token
/// commits share one exact cached mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum DynamicMaskLexerStateKey {
    Exact(u32),
    /// Exact scoped tokenizer coordinate used by recursive/provider-native
    /// composition. Keep it distinct from the ordinary tokenizer coordinate:
    /// both are u32 IDs but belong to different state spaces.
    RecursiveExact(u32),
    /// Exact lexer coordinate consumed by dynamic mask execution. Distinct
    /// source/runtime lexer states may map here when their complete one-model-
    /// token continuation languages are identical. Preserve whether the source
    /// state is the true lexer initial state because reset semantics can make
    /// that distinction observable outside the projected coordinate itself.
    MaskProjection {
        state: u32,
        initial: bool,
    },
    TerminalObservation {
        terminal: TerminalID,
        class: u32,
        initial: bool,
    },
    /// Sound pre-minimization coordinate of a virtual bounded-code residual's
    /// finite one-model-token projection. This gives the dynamic cache the
    /// useful count collapse of the finite projection without constructing its
    /// potentially large DFA.
    VirtualDenseProjection {
        runtime: u32,
        state: u32,
        initial: bool,
    },
}

pub(crate) type DynamicMaskStateKey = Vec<(
    DynamicMaskLexerStateKey,
    Vec<(Vec<u32>, Vec<(u32, Vec<TerminalID>)>)>,
)>;

#[derive(Debug, Clone)]
pub(crate) struct DynamicMaskVocabSource {
    pub(crate) trie: Arc<VocabPrefixTree>,
    pub(crate) token_aliases: Arc<Vec<Vec<u32>>>,
}

/// Runtime-only vocabulary data for direct dynamic mask generation.
#[derive(Debug, Clone)]
pub(crate) struct DynamicMaskVocab {
    pub(crate) trie: Arc<DynamicMaskTrie>,
    pub(super) token_aliases: DynamicMaskAliasStore,
    pub(super) canonical_original_token_offsets: Arc<Vec<u32>>,
    pub(super) canonical_original_tokens: Arc<Vec<u32>>,
    pub(super) canonical_original_word_offsets: Arc<Vec<u32>>,
    pub(super) canonical_original_word_masks: Arc<Vec<(u32, u32)>>,
    pub(super) node_token_markers: Arc<Vec<u64>>,
    /// Token markers in the exact order token endpoints are encountered by
    /// `full_walk_ops`. This removes an extra node-id indirection from the
    /// strict walk's very hot endpoint path without changing which endpoints
    /// are visited.
    pub(super) full_walk_token_markers: Arc<Vec<u64>>,
    pub(super) subtree_original_token_offsets: Arc<Vec<u32>>,
    pub(super) subtree_original_tokens: Arc<Vec<u32>>,
    pub(super) all_original_token_words: Arc<Vec<u32>>,
    pub(super) llg_slice_leftovers: Arc<Vec<Arc<DynamicMaskSliceTrie>>>,
    /// Cumulative admitted-token bitsets for the dynamic-radius master trie.
    /// Index = `safe_radius * 2 + whitespace_proved`; each row contains exactly
    /// the whole tokens that can be accepted without walking the trie. Shared by
    /// every constraint using the same model vocabulary.
    pub(super) llg_master_admitted_words: Arc<Vec<Vec<u32>>>,
    /// Exact strict-walk operation count remaining after admitting each master
    /// `(safe_radius, whitespace)` class combination. Indexed identically to
    /// `llg_master_admitted_words`. This is runtime-only derived metadata and
    /// is rebuilt from the master trie after construction/load.
    pub(super) llg_master_residual_ops: Arc<Vec<u32>>,
    pub(super) llg_master_max_safe_chars: u16,
    /// Positive-only parser-independent proof rows for the two overlapping
    /// master slice languages. Row = `source_tsid * 2 + slice_slot`, where
    /// slot 0 is safe+ and slot 1 is whitespace. Offsets index the flattened
    /// terminal list. A terminal in a row certifies that the entire slice
    /// language remains inside that terminal's exact projected residual from
    /// the source TSID. Missing rows/terminals simply fall back to runtime
    /// proof; they never imply rejection or admission.
    pub(super) prepared_master_prover_row_ids: Arc<[u32]>,
    pub(super) prepared_master_prover_offsets: Arc<[u32]>,
    pub(super) prepared_master_prover_terminals: Arc<[TerminalID]>,
    /// Exact coverage rows corresponding to `prepared_master_prover_*`.
    /// A terminal present here means the build-time proof solved this
    /// `(source, slice, terminal)` residual completely. Membership in the
    /// positive row is therefore `true`; coverage without positive membership
    /// is an exact `false`. This lets compact direct-residual proofs retain
    /// exact negative answers without carrying a projected-terminal quotient.
    pub(super) prepared_master_coverage_row_ids: Arc<[u32]>,
    pub(super) prepared_master_coverage_offsets: Arc<[u32]>,
    pub(super) prepared_master_coverage_terminals: Arc<[TerminalID]>,
    /// Global safe+ terminal completeness marker for direct residual proofs.
    /// Sorted and unique. See `PreparedMasterProofArtifact`.
    pub(super) prepared_safe_plus_complete_terminals: Arc<[TerminalID]>,
    /// Exact positive bounded safe-slice rows. Row = exact source TSID; entries
    /// are `(terminal, max safe Unicode-scalar radius)` for every prepared
    /// projected terminal residual. Radius is capped at the largest safe-token
    /// scalar count in this model vocabulary, because larger values are
    /// observationally identical for one-token masking.
    pub(super) prepared_safe_radius_row_ids: Arc<[u32]>,
    pub(super) prepared_safe_radius_offsets: Arc<[u32]>,
    pub(super) prepared_safe_radius_entries: Arc<[(TerminalID, u16)]>,
    pub(super) pending_source: Option<DynamicMaskVocabSource>,
    pub(super) initialized: bool,
    /// True only when trie endpoints represent grammar-proven vocabulary
    /// equivalence classes rather than byte-identical token aliases.
    pub(super) grammar_quotiented: bool,
    /// Strong identity of the complete original model vocabulary from which a
    /// grammar quotient was built. Ordinary dynamic vocabularies leave this
    /// unset; compiler-created O2 quotients carry it into transfer metadata.
    pub(super) source_vocab_digest: Option<[u8; 32]>,
    pub(super) mask_cache: Arc<Mutex<DynamicMaskCache>>,
    pub(super) dense_subset16_cache: Arc<Mutex<FxHashMap<Vec<u32>, Arc<DynamicDenseSubset16>>>>,
    pub(super) lazy_union_cache: Arc<Mutex<DynamicLazyUnionCache>>,
    pub(super) direct_regular_frontier_cache:
        Arc<Mutex<FxHashMap<usize, DirectRegularDynamicFrontierCacheEntry>>>,
    pub(super) direct_regular_wide_frontier_index_cache: Arc<Mutex<FxHashMap<usize, usize>>>,
    pub(super) direct_regular_terminal_support: Arc<DirectRegularTerminalSupport>,
    pub(super) bounded_observation_sets: Arc<DynamicBoundedObservationSets>,
    pub(super) terminal_observation_classes: Arc<[(TerminalID, Arc<[u32]>)]>,
    pub(super) projected_terminal_quotients: Arc<[(TerminalID, Arc<TerminalProjectedQuotient>)]>,
    /// Runtime-only lazy quotient analysis. Dynamic compilation keeps quotient
    /// construction off the build/first-mask path until a master-slice proof
    /// actually needs it. The persisted/prepared sidecar above still takes
    /// precedence when present.
    pub(super) runtime_projected_terminal_quotients:
        Arc<OnceLock<Arc<[(TerminalID, Arc<TerminalProjectedQuotient>)]>>>,
    /// Incremental runtime-only quotient cache used by hot-path containment
    /// proofs. Unlike `runtime_projected_terminal_quotients`, this can prepare
    /// only the terminal actually requested by the current proof instead of
    /// materializing every safe-alphabet candidate at once. `None` is a cached
    /// exact "no quotient available" result for that terminal.
    pub(super) runtime_projected_terminal_quotient_cache:
        Arc<Mutex<FxHashMap<TerminalID, Option<Arc<TerminalProjectedQuotient>>>>>,
    /// True once the exact projected-terminal analysis has run, including
    /// when it proved that no quotient is worth retaining.  This distinguishes
    /// a legitimate empty result from an unprepared legacy/runtime value.
    pub(super) projected_terminal_quotients_prepared: bool,
    /// Parser-independent exact projected-text proof results.  The key is the
    /// terminal residual coordinate plus the vocabulary alphabet being proved.
    /// Sharing this across sequences is safe because parser admission only
    /// decides whether a proof is queried; the proof result itself depends
    /// solely on immutable lexer/vocabulary data.
    pub(super) projected_terminal_text_cache:
        Arc<Mutex<FxHashMap<(TerminalID, u32, U8Set, bool), bool>>>,
    /// Exact regular-language containment results for named proof slices.
    /// High key bits distinguish projected, symbolic-residual, and finite-direct
    /// proof namespaces over the same immutable terminal/source coordinates.
    pub(super) projected_terminal_partition_cache:
        Arc<Mutex<FxHashMap<(TerminalID, u32, u32), bool>>>,
    /// Exact bounded repetition radii for regular-language proof slices. The
    /// extra key component is the caller's maximum relevant repetition count;
    /// for model-token masking this is the largest safe-token scalar length.
    pub(super) projected_terminal_radius_cache:
        Arc<Mutex<FxHashMap<(TerminalID, u32, u32, u32), u32>>>,
    /// Conservative bounded certificates over exact physical source coordinates.
    /// Never shared with a different grammar's fresh runtime vocabulary instance.
    pub(super) raw_terminal_radius_cache:
        Arc<Mutex<FxHashMap<(TerminalID, u32, u32, u32, usize), u32>>>,
    /// Exact original-token masks rejected by a token-start maximal-munch
    /// guard. Keys are the canonical sorted `(mask lexer state, terminal)`
    /// memories carried by `InitialPruneGuard`. The result depends only on the
    /// immutable lexer/vocabulary coordinate, so sequences may safely share it.
    pub(super) pending_guard_blocked_mask_cache:
        Arc<Mutex<FxHashMap<Vec<(u32, TerminalID)>, Arc<Vec<u32>>>>>,
    /// Optional mask-only finite-token quotient. Commit continues to use the
    /// exact tokenizer stored on `Constraint`; dynamic mask projections may be
    /// built in this smaller coordinate and indexed from exact runtime states
    /// through `full_to_mask_state`.
    pub(super) mask_tokenizer: Option<Arc<Tokenizer>>,
    /// Optional deterministic derivative of `mask_tokenizer` used only by
    /// mask generation. This is particularly useful for finite one-token
    /// projections of lazy/virtual lexers whose compact serialized projection
    /// still contains epsilon fan-in. Commit never uses this coordinate.
    pub(super) mask_determinized_tokenizer: Option<Arc<Tokenizer>>,
    /// Dense map from the serialized/base mask-tokenizer state coordinate to
    /// `mask_determinized_tokenizer`. Empty when no second-stage
    /// determinization is active.
    pub(super) mask_projection_to_determinized: Arc<[u32]>,
    pub(super) mask_tokenizer_fast_transitions: Option<FastTokenizerTransitions>,
    pub(super) full_to_mask_state: Arc<[u32]>,
    /// Derived exact subset provenance for the dense mask tokenizer. Keys are
    /// epsilon-closed source-tokenizer state sets and values are the already
    /// materialized deterministic mask states representing those sets.
    /// Runtime mask roots may use this only when every source state in the set
    /// carries the same parser object by identity.
    pub(super) mask_state_source_subsets: Arc<[Arc<[u32]>]>,
    pub(super) mask_source_subset_to_state: Arc<FxHashMap<Arc<[u32]>, u32>>,
    pub(super) virtual_unit_repeat_projection: Option<VirtualZeroMinUnitRepeatMaskProjection>,
    pub(super) virtual_repeat_intersection_projections:
        Vec<VirtualBinaryRepeatIntersectionMaskProjection>,
    pub(super) virtual_residual_projections: Vec<VirtualResidualMaskProjection>,
}

impl Default for DynamicMaskVocab {
    fn default() -> Self {
        Self {
            trie: Arc::new(DynamicMaskTrie::new()),
            token_aliases: DynamicMaskAliasStore::Packed(Arc::new(Vec::new())),
            canonical_original_token_offsets: Arc::new(vec![0]),
            canonical_original_tokens: Arc::new(Vec::new()),
            canonical_original_word_offsets: Arc::new(vec![0]),
            canonical_original_word_masks: Arc::new(Vec::new()),
            node_token_markers: Arc::new(vec![0]),
            full_walk_token_markers: Arc::new(Vec::new()),
            subtree_original_token_offsets: Arc::new(vec![0]),
            subtree_original_tokens: Arc::new(Vec::new()),
            all_original_token_words: Arc::new(Vec::new()),
            llg_slice_leftovers: Arc::new(Vec::new()),
            llg_master_admitted_words: Arc::new(Vec::new()),
            llg_master_residual_ops: Arc::new(Vec::new()),
            llg_master_max_safe_chars: 0,
            prepared_master_prover_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_master_coverage_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_plus_complete_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_radius_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_offsets: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_entries: Arc::from(Vec::<(TerminalID, u16)>::new()),
            pending_source: None,
            initialized: false,
            grammar_quotiented: false,
            source_vocab_digest: None,
            mask_cache: Arc::new(Mutex::new(DynamicMaskCache::default())),
            dense_subset16_cache: Arc::new(Mutex::new(FxHashMap::default())),
            lazy_union_cache: Arc::new(Mutex::new(DynamicLazyUnionCache::default())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_terminal_support: Arc::new(DirectRegularTerminalSupport::default()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            terminal_observation_classes: Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new()),
            projected_terminal_quotients: Arc::from(Vec::<(
                TerminalID,
                Arc<TerminalProjectedQuotient>,
            )>::new()),
            runtime_projected_terminal_quotients: Arc::new(OnceLock::new()),
            runtime_projected_terminal_quotient_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_quotients_prepared: false,
            projected_terminal_text_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_partition_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            raw_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            pending_guard_blocked_mask_cache: Arc::new(Mutex::new(FxHashMap::default())),
            mask_tokenizer: None,
            mask_determinized_tokenizer: None,
            mask_projection_to_determinized: Arc::from(Vec::<u32>::new()),
            mask_tokenizer_fast_transitions: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
            mask_state_source_subsets: Arc::from(Vec::<Arc<[u32]>>::new()),
            mask_source_subset_to_state: Arc::new(FxHashMap::default()),
            virtual_unit_repeat_projection: None,
            virtual_repeat_intersection_projections: Vec::new(),
            virtual_residual_projections: Vec::new(),
        }
    }
}

impl DynamicMaskVocab {
    pub(super) const FULL_WALK_DENSE_TRANSITION_BYTES: usize = 64 * 1024 * 1024;
    pub(crate) const PREPARED_PROOF_SLOT_COUNT: usize = 2;
    pub(super) const PREPARED_SAFE_PLUS_SLOT: usize = 0;
    pub(super) const PREPARED_WHITESPACE_SLOT: usize = 1;

    #[inline]
    pub(crate) fn max_token_byte_len(&self) -> usize {
        self.trie
            .nodes
            .first()
            .map_or(0, |root| root.subtree_max_byte_len as usize)
    }

    pub(crate) fn from_compiler_artifacts(
        trie: Arc<VocabPrefixTree>,
        token_aliases: Arc<Vec<Vec<u32>>>,
    ) -> Self {
        Self::from_source(DynamicMaskVocabSource {
            trie,
            token_aliases,
        })
    }

    pub(crate) fn from_compiler_artifacts_materialized(
        trie: Arc<VocabPrefixTree>,
        token_aliases: Arc<Vec<Vec<u32>>>,
    ) -> Self {
        let mut vocab = Self::from_compiler_artifacts(trie, token_aliases);
        let materialized = vocab.materialize_pending_source();
        debug_assert!(materialized);
        vocab
    }

    pub(crate) fn from_materialized_ordered(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: Arc<Vec<Vec<u32>>>,
    ) -> Self {
        Self::from_materialized_ordered_with_all_original_token_words(trie, token_aliases, None)
    }

    pub(crate) fn from_materialized_ordered_with_all_original_token_words(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: Arc<Vec<Vec<u32>>>,
        prepared_all_original_token_words: Option<Arc<Vec<u32>>>,
    ) -> Self {
        // Grammar quotients can contain one overwhelmingly large alias group
        // (for example the permanently-dead class of a singleton language).
        // When this Arc is uniquely owned and its first group already reserved
        // enough capacity for the complete flattened list, reuse that Vec as
        // the canonical runtime storage instead of copying ~the whole model
        // vocabulary into a second allocation. Small/shared callers retain the
        // established grouped representation.
        let (token_aliases, preflattened) = match Arc::try_unwrap(token_aliases) {
            Ok(groups) => {
                let total = groups.iter().map(Vec::len).sum::<usize>();
                let can_reuse_first = groups
                    .first()
                    .is_some_and(|first| first.len() >= 4_096 && first.capacity() >= total);
                if can_reuse_first {
                    let mut offsets = Vec::with_capacity(groups.len() + 1);
                    offsets.push(0);
                    for group in &groups {
                        offsets.push(offsets.last().copied().unwrap_or(0) + group.len() as u32);
                    }
                    let mut iter = groups.into_iter();
                    let mut originals = iter.next().unwrap_or_default();
                    for group in iter {
                        originals.extend_from_slice(&group);
                    }
                    let offsets = Arc::new(offsets);
                    let originals = Arc::new(originals);
                    (
                        DynamicMaskAliasStore::Flat {
                            offsets: Arc::clone(&offsets),
                            originals: Arc::clone(&originals),
                        },
                        Some((offsets, originals)),
                    )
                } else {
                    (DynamicMaskAliasStore::Ordered(Arc::new(groups)), None)
                }
            }
            Err(token_aliases) => (DynamicMaskAliasStore::Ordered(token_aliases), None),
        };
        let (canonical_original_token_offsets, canonical_original_tokens) =
            preflattened.unwrap_or_else(|| Self::flatten_canonical_original_tokens(&token_aliases));
        let build_words = || {
            Self::build_canonical_original_word_masks_sorted(
                &canonical_original_token_offsets,
                &canonical_original_tokens,
            )
        };
        let build_markers = || {
            let node_token_markers = Self::build_node_token_markers(
                trie.as_ref(),
                &canonical_original_token_offsets,
                &canonical_original_tokens,
            );
            let full_walk_token_markers =
                Self::build_full_walk_token_markers(trie.as_ref(), &node_token_markers);
            (node_token_markers, full_walk_token_markers)
        };
        let build_subtree = || {
            // `from_materialized_ordered` receives canonical tokens in lexical
            // trie order. `DynamicMaskTrie::all_subtree_tokens()` therefore
            // enumerates canonical IDs as 0..N, so the canonical flattened
            // aliases are already exactly the subtree-order flattened aliases.
            // Reuse the same immutable buffers instead of copying every model
            // token ID a second time. Fall back defensively if a future caller
            // violates the ordered-coordinate contract.
            let subtree_identity = trie
                .all_subtree_tokens()
                .iter()
                .copied()
                .enumerate()
                .all(|(index, canonical)| index as u32 == canonical);
            if subtree_identity {
                (
                    Arc::clone(&canonical_original_token_offsets),
                    Arc::clone(&canonical_original_tokens),
                )
            } else {
                Self::flatten_subtree_original_tokens(
                    trie.as_ref(),
                    &canonical_original_token_offsets,
                    &canonical_original_tokens,
                )
            }
        };
        let build_all_words = || {
            prepared_all_original_token_words
                .clone()
                .unwrap_or_else(|| Self::build_all_original_token_words(&canonical_original_tokens))
        };

        let (
            (canonical_original_word_offsets, canonical_original_word_masks),
            (node_token_markers, full_walk_token_markers),
            (subtree_original_token_offsets, subtree_original_tokens),
            all_original_token_words,
        ) = if canonical_original_tokens.len() >= 4_096 && rayon::current_num_threads() > 1 {
            let ((word_masks, markers), (subtree, all_words)) = rayon::join(
                || rayon::join(build_words, build_markers),
                || rayon::join(build_subtree, build_all_words),
            );
            (word_masks, markers, subtree, all_words)
        } else {
            (
                build_words(),
                build_markers(),
                build_subtree(),
                build_all_words(),
            )
        };
        Self {
            trie,
            token_aliases,
            canonical_original_token_offsets,
            canonical_original_tokens,
            canonical_original_word_offsets,
            canonical_original_word_masks,
            node_token_markers,
            full_walk_token_markers,
            subtree_original_token_offsets,
            subtree_original_tokens,
            all_original_token_words,
            llg_slice_leftovers: Arc::new(Vec::new()),
            llg_master_admitted_words: Arc::new(Vec::new()),
            llg_master_residual_ops: Arc::new(Vec::new()),
            llg_master_max_safe_chars: 0,
            prepared_master_prover_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_master_coverage_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_plus_complete_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_radius_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_offsets: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_entries: Arc::from(Vec::<(TerminalID, u16)>::new()),
            pending_source: None,
            initialized: true,
            grammar_quotiented: false,
            source_vocab_digest: None,
            mask_cache: Arc::new(Mutex::new(DynamicMaskCache::default())),
            dense_subset16_cache: Arc::new(Mutex::new(FxHashMap::default())),
            lazy_union_cache: Arc::new(Mutex::new(DynamicLazyUnionCache::default())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_terminal_support: Arc::new(DirectRegularTerminalSupport::default()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            terminal_observation_classes: Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new()),
            projected_terminal_quotients: Arc::from(Vec::<(
                TerminalID,
                Arc<TerminalProjectedQuotient>,
            )>::new()),
            runtime_projected_terminal_quotients: Arc::new(OnceLock::new()),
            runtime_projected_terminal_quotient_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_quotients_prepared: false,
            projected_terminal_text_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_partition_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            raw_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            pending_guard_blocked_mask_cache: Arc::new(Mutex::new(FxHashMap::default())),
            mask_tokenizer: None,
            mask_determinized_tokenizer: None,
            mask_projection_to_determinized: Arc::from(Vec::<u32>::new()),
            mask_tokenizer_fast_transitions: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
            mask_state_source_subsets: Arc::from(Vec::<Arc<[u32]>>::new()),
            mask_source_subset_to_state: Arc::new(FxHashMap::default()),
            virtual_unit_repeat_projection: None,
            virtual_repeat_intersection_projections: Vec::new(),
            virtual_residual_projections: Vec::new(),
        }
    }

    /// Create a constraint-local runtime value from a fully initialized,
    /// vocabulary-only template.
    ///
    /// The immutable trie and token indexes are shared. Every cache or
    /// accelerator whose contents can depend on parser, lexer, or constraint
    /// state is recreated empty, so repeated schema builds cannot inherit
    /// schema-derived runtime state.
    /// Minimal producer-side representation for an external-vocabulary
    /// transfer artifact. The worker never executes masks with this value: the
    /// execution parent reconstructs all runtime-only indexes from trie+aliases
    /// after load. Keeping those indexes lazy here avoids doing the same O(vocab)
    /// work twice per schema.
    pub(crate) fn from_materialized_ordered_for_transfer(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: Arc<Vec<Vec<u32>>>,
    ) -> Self {
        let mut vocab = Self::default();
        vocab.trie = trie;
        vocab.token_aliases = DynamicMaskAliasStore::Ordered(token_aliases);
        vocab.pending_source = None;
        vocab.initialized = true;
        vocab
    }

    pub(crate) fn fresh_runtime_instance(&self) -> Self {
        debug_assert!(self.initialized);
        debug_assert!(self.pending_source.is_none());
        Self {
            trie: Arc::clone(&self.trie),
            token_aliases: self.token_aliases.clone(),
            canonical_original_token_offsets: Arc::clone(&self.canonical_original_token_offsets),
            canonical_original_tokens: Arc::clone(&self.canonical_original_tokens),
            canonical_original_word_offsets: Arc::clone(&self.canonical_original_word_offsets),
            canonical_original_word_masks: Arc::clone(&self.canonical_original_word_masks),
            node_token_markers: Arc::clone(&self.node_token_markers),
            full_walk_token_markers: Arc::clone(&self.full_walk_token_markers),
            subtree_original_token_offsets: Arc::clone(&self.subtree_original_token_offsets),
            subtree_original_tokens: Arc::clone(&self.subtree_original_tokens),
            all_original_token_words: Arc::clone(&self.all_original_token_words),
            llg_slice_leftovers: Arc::clone(&self.llg_slice_leftovers),
            llg_master_admitted_words: Arc::clone(&self.llg_master_admitted_words),
            llg_master_residual_ops: Arc::clone(&self.llg_master_residual_ops),
            llg_master_max_safe_chars: self.llg_master_max_safe_chars,
            prepared_master_prover_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_master_coverage_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_plus_complete_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_radius_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_offsets: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_entries: Arc::from(Vec::<(TerminalID, u16)>::new()),
            pending_source: None,
            initialized: true,
            grammar_quotiented: self.grammar_quotiented,
            source_vocab_digest: self.source_vocab_digest,
            mask_cache: Arc::new(Mutex::new(DynamicMaskCache::default())),
            dense_subset16_cache: Arc::new(Mutex::new(FxHashMap::default())),
            lazy_union_cache: Arc::new(Mutex::new(DynamicLazyUnionCache::default())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_terminal_support: Arc::new(DirectRegularTerminalSupport::default()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            terminal_observation_classes: Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new()),
            projected_terminal_quotients: Arc::from(Vec::<(
                TerminalID,
                Arc<TerminalProjectedQuotient>,
            )>::new()),
            runtime_projected_terminal_quotients: Arc::new(OnceLock::new()),
            runtime_projected_terminal_quotient_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_quotients_prepared: false,
            projected_terminal_text_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_partition_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            raw_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            pending_guard_blocked_mask_cache: Arc::new(Mutex::new(FxHashMap::default())),
            mask_tokenizer: None,
            mask_determinized_tokenizer: None,
            mask_projection_to_determinized: Arc::from(Vec::<u32>::new()),
            mask_tokenizer_fast_transitions: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
            mask_state_source_subsets: Arc::from(Vec::<Arc<[u32]>>::new()),
            mask_source_subset_to_state: Arc::new(FxHashMap::default()),
            virtual_unit_repeat_projection: None,
            virtual_repeat_intersection_projections: Vec::new(),
            virtual_residual_projections: Vec::new(),
        }
    }

    pub(super) fn from_source(source: DynamicMaskVocabSource) -> Self {
        Self {
            trie: Arc::new(DynamicMaskTrie::new()),
            token_aliases: DynamicMaskAliasStore::Packed(Arc::new(Vec::new())),
            canonical_original_token_offsets: Arc::new(vec![0]),
            canonical_original_tokens: Arc::new(Vec::new()),
            canonical_original_word_offsets: Arc::new(vec![0]),
            canonical_original_word_masks: Arc::new(Vec::new()),
            node_token_markers: Arc::new(vec![0]),
            full_walk_token_markers: Arc::new(Vec::new()),
            subtree_original_token_offsets: Arc::new(vec![0]),
            subtree_original_tokens: Arc::new(Vec::new()),
            all_original_token_words: Arc::new(Vec::new()),
            llg_slice_leftovers: Arc::new(Vec::new()),
            llg_master_admitted_words: Arc::new(Vec::new()),
            llg_master_residual_ops: Arc::new(Vec::new()),
            llg_master_max_safe_chars: 0,
            prepared_master_prover_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_master_coverage_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_plus_complete_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_radius_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_offsets: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_entries: Arc::from(Vec::<(TerminalID, u16)>::new()),
            pending_source: Some(source),
            initialized: false,
            grammar_quotiented: false,
            source_vocab_digest: None,
            mask_cache: Arc::new(Mutex::new(DynamicMaskCache::default())),
            dense_subset16_cache: Arc::new(Mutex::new(FxHashMap::default())),
            lazy_union_cache: Arc::new(Mutex::new(DynamicLazyUnionCache::default())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_terminal_support: Arc::new(DirectRegularTerminalSupport::default()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            terminal_observation_classes: Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new()),
            projected_terminal_quotients: Arc::from(Vec::<(
                TerminalID,
                Arc<TerminalProjectedQuotient>,
            )>::new()),
            runtime_projected_terminal_quotients: Arc::new(OnceLock::new()),
            runtime_projected_terminal_quotient_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_quotients_prepared: false,
            projected_terminal_text_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_partition_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            raw_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            pending_guard_blocked_mask_cache: Arc::new(Mutex::new(FxHashMap::default())),
            mask_tokenizer: None,
            mask_determinized_tokenizer: None,
            mask_projection_to_determinized: Arc::from(Vec::<u32>::new()),
            mask_tokenizer_fast_transitions: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
            mask_state_source_subsets: Arc::from(Vec::<Arc<[u32]>>::new()),
            mask_source_subset_to_state: Arc::new(FxHashMap::default()),
            virtual_unit_repeat_projection: None,
            virtual_repeat_intersection_projections: Vec::new(),
            virtual_residual_projections: Vec::new(),
        }
    }

    pub(crate) fn mark_grammar_quotiented(&mut self) {
        self.grammar_quotiented = true;
    }

    pub(crate) fn set_source_vocab_digest(&mut self, digest: [u8; 32]) {
        self.source_vocab_digest = Some(digest);
    }

    pub(crate) fn source_vocab_digest(&self) -> Option<[u8; 32]> {
        self.source_vocab_digest
    }

    pub(crate) fn is_grammar_quotiented(&self) -> bool {
        self.grammar_quotiented
    }

    pub(crate) fn from_packed(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: Arc<Vec<Option<PackedDynamicMaskTokenAliases>>>,
    ) -> Self {
        Self::from_alias_store(trie, DynamicMaskAliasStore::Packed(token_aliases))
    }

    pub(super) fn from_alias_store(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: DynamicMaskAliasStore,
    ) -> Self {
        Self::from_alias_store_with_slices(trie, token_aliases, None)
    }

    pub(super) fn from_alias_store_with_slices(
        trie: Arc<DynamicMaskTrie>,
        token_aliases: DynamicMaskAliasStore,
        slices: Option<&[Arc<DynamicMaskSliceTrie>]>,
    ) -> Self {
        let (canonical_original_token_offsets, canonical_original_tokens) =
            Self::flatten_canonical_original_tokens(&token_aliases);
        let master_layout_classes = slices.and_then(|slices| {
            Self::llg_master_layout_classes_from_slices(
                slices,
                canonical_original_token_offsets.len().saturating_sub(1),
            )
        });
        let (
            canonical_original_word_offsets,
            canonical_original_word_masks,
            fused_master_admission,
        ) = if let Some(classes) = master_layout_classes.as_deref() {
            Self::build_canonical_original_word_masks_and_master_admission(
                &canonical_original_token_offsets,
                &canonical_original_tokens,
                classes,
            )
        } else {
            let (offsets, masks) = Self::build_canonical_original_word_masks(
                &canonical_original_token_offsets,
                &canonical_original_tokens,
            );
            (offsets, masks, None)
        };
        let node_token_markers = Self::build_node_token_markers(
            trie.as_ref(),
            &canonical_original_token_offsets,
            &canonical_original_tokens,
        );
        let full_walk_token_markers =
            Self::build_full_walk_token_markers(trie.as_ref(), &node_token_markers);
        let (subtree_original_token_offsets, subtree_original_tokens) =
            Self::flatten_subtree_original_tokens(
                trie.as_ref(),
                &canonical_original_token_offsets,
                &canonical_original_tokens,
            );
        let all_original_token_words =
            Self::build_all_original_token_words(&subtree_original_tokens);
        Self {
            trie,
            token_aliases,
            canonical_original_token_offsets,
            canonical_original_tokens,
            canonical_original_word_offsets,
            canonical_original_word_masks,
            node_token_markers,
            full_walk_token_markers,
            subtree_original_token_offsets,
            subtree_original_tokens,
            all_original_token_words,
            llg_slice_leftovers: Arc::new(Vec::new()),
            llg_master_admitted_words: fused_master_admission
                .as_ref()
                .map_or_else(|| Arc::new(Vec::new()), |(_, words)| Arc::clone(words)),
            llg_master_residual_ops: Arc::new(Vec::new()),
            llg_master_max_safe_chars: fused_master_admission
                .map_or(0, |(max_safe_chars, _)| max_safe_chars),
            prepared_master_prover_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_prover_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_master_coverage_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_offsets: Arc::from(Vec::<u32>::new()),
            prepared_master_coverage_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_plus_complete_terminals: Arc::from(Vec::<TerminalID>::new()),
            prepared_safe_radius_row_ids: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_offsets: Arc::from(Vec::<u32>::new()),
            prepared_safe_radius_entries: Arc::from(Vec::<(TerminalID, u16)>::new()),
            pending_source: None,
            initialized: true,
            grammar_quotiented: false,
            source_vocab_digest: None,
            mask_cache: Arc::new(Mutex::new(DynamicMaskCache::default())),
            dense_subset16_cache: Arc::new(Mutex::new(FxHashMap::default())),
            lazy_union_cache: Arc::new(Mutex::new(DynamicLazyUnionCache::default())),
            direct_regular_frontier_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_wide_frontier_index_cache: Arc::new(Mutex::new(FxHashMap::default())),
            direct_regular_terminal_support: Arc::new(DirectRegularTerminalSupport::default()),
            bounded_observation_sets: Arc::new(DynamicBoundedObservationSets::default()),
            terminal_observation_classes: Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new()),
            projected_terminal_quotients: Arc::from(Vec::<(
                TerminalID,
                Arc<TerminalProjectedQuotient>,
            )>::new()),
            runtime_projected_terminal_quotients: Arc::new(OnceLock::new()),
            runtime_projected_terminal_quotient_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_quotients_prepared: false,
            projected_terminal_text_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_partition_cache: Arc::new(Mutex::new(FxHashMap::default())),
            projected_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            raw_terminal_radius_cache: Arc::new(Mutex::new(FxHashMap::default())),
            pending_guard_blocked_mask_cache: Arc::new(Mutex::new(FxHashMap::default())),
            mask_tokenizer: None,
            mask_determinized_tokenizer: None,
            mask_projection_to_determinized: Arc::from(Vec::<u32>::new()),
            mask_tokenizer_fast_transitions: None,
            full_to_mask_state: Arc::from(Vec::<u32>::new()),
            mask_state_source_subsets: Arc::from(Vec::<Arc<[u32]>>::new()),
            mask_source_subset_to_state: Arc::new(FxHashMap::default()),
            virtual_unit_repeat_projection: None,
            virtual_repeat_intersection_projections: Vec::new(),
            virtual_residual_projections: Vec::new(),
        }
    }

    pub(crate) fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub(crate) fn materialize_pending_source(&mut self) -> bool {
        let Some(source) = self.pending_source.take() else {
            return false;
        };
        self.trie = Arc::new(DynamicMaskTrie::from_vocab_prefix_tree(
            source.trie.as_ref(),
        ));
        self.token_aliases = DynamicMaskAliasStore::Ordered(source.token_aliases);
        (
            self.canonical_original_token_offsets,
            self.canonical_original_tokens,
        ) = Self::flatten_canonical_original_tokens(&self.token_aliases);
        (
            self.canonical_original_word_offsets,
            self.canonical_original_word_masks,
        ) = Self::build_canonical_original_word_masks(
            &self.canonical_original_token_offsets,
            &self.canonical_original_tokens,
        );
        self.node_token_markers = Self::build_node_token_markers(
            self.trie.as_ref(),
            &self.canonical_original_token_offsets,
            &self.canonical_original_tokens,
        );
        self.full_walk_token_markers =
            Self::build_full_walk_token_markers(self.trie.as_ref(), &self.node_token_markers);
        (
            self.subtree_original_token_offsets,
            self.subtree_original_tokens,
        ) = Self::flatten_subtree_original_tokens(
            self.trie.as_ref(),
            &self.canonical_original_token_offsets,
            &self.canonical_original_tokens,
        );
        self.all_original_token_words =
            Self::build_all_original_token_words(&self.subtree_original_tokens);
        self.initialized = true;
        true
    }

    pub(super) fn flatten_canonical_original_tokens(
        token_aliases: &DynamicMaskAliasStore,
    ) -> (Arc<Vec<u32>>, Arc<Vec<u32>>) {
        if let DynamicMaskAliasStore::Flat { offsets, originals } = token_aliases {
            return (Arc::clone(offsets), Arc::clone(originals));
        }
        let alias_slots = match token_aliases {
            DynamicMaskAliasStore::Ordered(aliases) => aliases.len(),
            DynamicMaskAliasStore::Flat { .. } => unreachable!("flat alias store returned above"),
            DynamicMaskAliasStore::Packed(aliases) => aliases.len(),
        };
        let mut offsets = Vec::with_capacity(alias_slots + 1);
        let mut originals = Vec::new();
        offsets.push(0);
        for canonical_token in 0..alias_slots {
            match token_aliases {
                DynamicMaskAliasStore::Ordered(aliases) => {
                    originals.extend_from_slice(&aliases[canonical_token]);
                }
                DynamicMaskAliasStore::Flat { .. } => {
                    unreachable!("flat alias store returned above")
                }
                DynamicMaskAliasStore::Packed(aliases) => {
                    if let Some(alias) = aliases[canonical_token].as_ref() {
                        match alias {
                            PackedDynamicMaskTokenAliases::Single(token_id) => {
                                originals.push(*token_id);
                            }
                            PackedDynamicMaskTokenAliases::Many(token_ids) => {
                                originals.extend_from_slice(token_ids);
                            }
                        }
                    }
                }
            }
            offsets.push(originals.len() as u32);
        }
        (Arc::new(offsets), Arc::new(originals))
    }

    pub(super) fn llg_master_layout_classes_from_slices(
        slices: &[Arc<DynamicMaskSliceTrie>],
        canonical_count: usize,
    ) -> Option<Vec<u16>> {
        let master = slices
            .iter()
            .find(|slice| slice.cache_id() == DYNAMIC_MASK_LLG_MASTER_CACHE_ID)?;
        let trie = master.trie();
        let mut classes = vec![0u16; canonical_count];
        let mut seen = vec![false; canonical_count];
        if let Some(root_token) = trie.node(0).token_id {
            let slot = root_token as usize;
            if slot >= canonical_count {
                return None;
            }
            seen[slot] = true;
        }
        let all_tokens = trie.all_subtree_tokens();
        for (root_slot, edge) in trie.children(0).iter().enumerate() {
            let class = trie.root_layout_class(root_slot)?;
            let canonical_tokens = all_tokens.get(trie.subtree_token_index_range(edge.child))?;
            for &canonical in canonical_tokens {
                let slot = canonical as usize;
                if slot >= canonical_count {
                    return None;
                }
                if seen[slot] && classes[slot] != class {
                    return None;
                }
                classes[slot] = class;
                seen[slot] = true;
            }
        }
        seen.iter().all(|&value| value).then_some(classes)
    }

    pub(super) fn build_canonical_original_word_masks_and_master_admission(
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
        master_layout_classes: &[u16],
    ) -> (
        Arc<Vec<u32>>,
        Arc<Vec<(u32, u32)>>,
        Option<(u16, Arc<Vec<Vec<u32>>>)>,
    ) {
        let canonical_count = canonical_offsets.len().saturating_sub(1);
        if master_layout_classes.len() != canonical_count {
            let (offsets, masks) = Self::build_canonical_original_word_masks(
                canonical_offsets,
                canonical_original_tokens,
            );
            return (offsets, masks, None);
        }
        let word_len = canonical_original_tokens
            .iter()
            .copied()
            .max()
            .map_or(0, |token| token as usize / 32 + 1);
        let max_safe_chars = master_layout_classes
            .iter()
            .copied()
            .map(dynamic_mask_llg_master_safe_chars)
            .max()
            .unwrap_or(0);
        let mut exact_safe_words = vec![vec![0u32; word_len]; usize::from(max_safe_chars) + 1];
        let mut whitespace_words = vec![0u32; word_len];
        let mut scratch = vec![0u32; word_len];
        let mut touched = Vec::<u32>::new();
        let mut offsets = Vec::<u32>::with_capacity(canonical_count + 1);
        let mut masks = Vec::<(u32, u32)>::new();
        offsets.push(0);
        for canonical in 0..canonical_count {
            let class = master_layout_classes[canonical];
            let safe_chars = dynamic_mask_llg_master_safe_chars(class);
            let is_whitespace = dynamic_mask_llg_master_is_whitespace(class);
            let start = canonical_offsets[canonical] as usize;
            let end = canonical_offsets[canonical + 1] as usize;
            for &token_id in &canonical_original_tokens[start..end] {
                let word = token_id / 32;
                let word_slot = word as usize;
                let bit = 1u32 << (token_id % 32);
                let slot = unsafe { scratch.get_unchecked_mut(word_slot) };
                if *slot == 0 {
                    touched.push(word);
                }
                *slot |= bit;
                if safe_chars != 0 {
                    unsafe {
                        *exact_safe_words
                            .get_unchecked_mut(usize::from(safe_chars))
                            .get_unchecked_mut(word_slot) |= bit;
                    }
                }
                if is_whitespace {
                    unsafe {
                        *whitespace_words.get_unchecked_mut(word_slot) |= bit;
                    }
                }
            }
            for word in touched.drain(..) {
                let bits = unsafe { *scratch.get_unchecked(word as usize) };
                debug_assert_ne!(bits, 0);
                masks.push((word, bits));
                unsafe {
                    *scratch.get_unchecked_mut(word as usize) = 0;
                }
            }
            offsets.push(masks.len() as u32);
        }

        let mut admitted_words =
            Vec::<Vec<u32>>::with_capacity((usize::from(max_safe_chars) + 1) * 2);
        let mut safe_prefix = vec![0u32; word_len];
        for radius in 0..=usize::from(max_safe_chars) {
            if radius != 0 {
                for (target, &source) in safe_prefix.iter_mut().zip(&exact_safe_words[radius]) {
                    *target |= source;
                }
            }
            admitted_words.push(safe_prefix.clone());
            let mut with_whitespace = safe_prefix.clone();
            for (target, &source) in with_whitespace.iter_mut().zip(&whitespace_words) {
                *target |= source;
            }
            admitted_words.push(with_whitespace);
        }
        (
            Arc::new(offsets),
            Arc::new(masks),
            Some((max_safe_chars, Arc::new(admitted_words))),
        )
    }

    pub(super) fn build_canonical_original_word_masks(
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
    ) -> (Arc<Vec<u32>>, Arc<Vec<(u32, u32)>>) {
        let canonical_count = canonical_offsets.len().saturating_sub(1);
        let word_len = canonical_original_tokens
            .iter()
            .copied()
            .max()
            .map_or(0, |token| token as usize / 32 + 1);
        let mut scratch = vec![0u32; word_len];
        let mut touched = Vec::<u32>::new();
        let mut offsets = Vec::<u32>::with_capacity(canonical_count + 1);
        let mut masks = Vec::<(u32, u32)>::new();
        offsets.push(0);
        for canonical in 0..canonical_count {
            let start = canonical_offsets[canonical] as usize;
            let end = canonical_offsets[canonical + 1] as usize;
            for &token_id in &canonical_original_tokens[start..end] {
                let word = token_id / 32;
                let slot = unsafe { scratch.get_unchecked_mut(word as usize) };
                if *slot == 0 {
                    touched.push(word);
                }
                *slot |= 1u32 << (token_id % 32);
            }
            for word in touched.drain(..) {
                let bits = unsafe { *scratch.get_unchecked(word as usize) };
                debug_assert_ne!(bits, 0);
                masks.push((word, bits));
                unsafe {
                    *scratch.get_unchecked_mut(word as usize) = 0;
                }
            }
            offsets.push(masks.len() as u32);
        }
        (Arc::new(offsets), Arc::new(masks))
    }

    /// Faster exact mask construction for the ordered vocabulary coordinate.
    /// Each canonical alias list is sorted by original token ID before this
    /// constructor is called, so equal 32-token words are contiguous.  Build
    /// sparse word masks in one linear pass rather than clearing/probing a
    /// model-vocabulary-sized scratch bitset for every canonical class.
    pub(super) fn build_canonical_original_word_masks_sorted(
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
    ) -> (Arc<Vec<u32>>, Arc<Vec<(u32, u32)>>) {
        let canonical_count = canonical_offsets.len().saturating_sub(1);
        let mut offsets = Vec::<u32>::with_capacity(canonical_count + 1);
        let mut masks = Vec::<(u32, u32)>::new();
        offsets.push(0);
        for canonical in 0..canonical_count {
            let start = canonical_offsets[canonical] as usize;
            let end = canonical_offsets[canonical + 1] as usize;
            let aliases = &canonical_original_tokens[start..end];
            debug_assert!(aliases.windows(2).all(|pair| pair[0] <= pair[1]));
            let mut current_word = u32::MAX;
            let mut bits = 0u32;
            for &token_id in aliases {
                let word = token_id / 32;
                if word != current_word {
                    if current_word != u32::MAX {
                        masks.push((current_word, bits));
                    }
                    current_word = word;
                    bits = 0;
                }
                bits |= 1u32 << (token_id % 32);
            }
            if current_word != u32::MAX {
                masks.push((current_word, bits));
            }
            offsets.push(masks.len() as u32);
        }
        (Arc::new(offsets), Arc::new(masks))
    }

    pub(super) fn build_all_original_token_words(originals: &[u32]) -> Arc<Vec<u32>> {
        let word_len = originals
            .iter()
            .copied()
            .max()
            .map_or(0, |token| token as usize / 32 + 1);
        let mut words = vec![0u32; word_len];
        for &token in originals {
            words[token as usize / 32] |= 1u32 << (token % 32);
        }
        Arc::new(words)
    }

    #[inline]
    pub(crate) fn all_original_token_words(&self) -> &[u32] {
        self.all_original_token_words.as_ref()
    }

    pub(crate) fn all_original_token_words_arc(&self) -> Arc<Vec<u32>> {
        Arc::clone(&self.all_original_token_words)
    }

    #[inline]
    pub(crate) fn canonical_token_count(&self) -> usize {
        let indexed = self
            .canonical_original_token_offsets
            .len()
            .saturating_sub(1);
        if indexed != 0 {
            return indexed;
        }
        match &self.token_aliases {
            DynamicMaskAliasStore::Ordered(aliases) => aliases.len(),
            DynamicMaskAliasStore::Flat { offsets, .. } => offsets.len().saturating_sub(1),
            DynamicMaskAliasStore::Packed(aliases) => aliases.len(),
        }
    }

    pub(crate) fn token_ids(&self, canonical_token_id: u32) -> Option<&[u32]> {
        let index = canonical_token_id as usize;
        let end_index = index.checked_add(1)?;
        let (&start, &end) = self
            .canonical_original_token_offsets
            .get(index)
            .zip(self.canonical_original_token_offsets.get(end_index))?;
        (start != end).then(|| &self.canonical_original_tokens[start as usize..end as usize])
    }

    #[inline(always)]
    pub(crate) fn token_word_masks(&self, canonical_token_id: u32) -> &[(u32, u32)] {
        let index = canonical_token_id as usize;
        let start = unsafe { *self.canonical_original_word_offsets.get_unchecked(index) } as usize;
        let end = unsafe {
            *self
                .canonical_original_word_offsets
                .get_unchecked(index + 1)
        } as usize;
        unsafe { self.canonical_original_word_masks.get_unchecked(start..end) }
    }

    #[inline(always)]
    pub(crate) fn node_token_marker(&self, node: u32) -> u64 {
        debug_assert!((node as usize) < self.node_token_markers.len());
        unsafe { *self.node_token_markers.get_unchecked(node as usize) }
    }

    #[inline(always)]
    pub(crate) fn full_walk_token_markers(&self) -> &[u64] {
        self.full_walk_token_markers.as_ref()
    }
}
