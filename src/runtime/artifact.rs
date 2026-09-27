//! Compiled constraint layout and private artifact facade.
//!
//! Implementation modules preserve this internal import surface. Persistence
//! remains in runtime::persistence; public save/load are the supported contract.

mod boundary;
mod deferred;
mod mask_cache;
mod mask_rows;
mod master_proofs;
mod projections;
mod recursive;
mod regular;
mod slices;
mod transitions;
mod trie;
mod vocab_transfer;
mod vocabulary;

pub use boundary::BoundaryTriggerDetail;
#[allow(unused_imports)]
pub(crate) use boundary::{
    BoundaryCandidateFingerprint, BoundaryCandidateSummary, BoundaryTrigger,
    CompositionGrammarSummary, OriginalTokenSet, SummaryPrecision, SummaryUnavailable,
};
#[allow(unused_imports)]
pub(crate) use deferred::{DeferredCompositionMetadataBytes, DeferredTerminalExprBytes};
#[allow(unused_imports)]
use mask_cache::DynamicMaskCache;
#[allow(unused_imports)]
pub(crate) use mask_cache::{
    DynamicDenseSubset16, DynamicLazyUnionCache, DynamicLazyUnionMetadata, DynamicLazyUnionRow,
};
#[allow(unused_imports)]
pub(crate) use mask_rows::{
    BackedInternalTokenBufMasks, DenseAcceptanceRows, DenseBufMaskRows, DenseBufMaskRowsIter,
    DenseWeightBufMaskCache, DenseWeightMaskCache, DenseWords, InternalTokenBufMasks,
    PackedDwaDenseWeightMaskCache, PackedInternalTokenBufMask, PackedNonDwaWeights,
    RangeFinalTokenSetCache, SeedTerminalDenseMasks, SparseWeightBufMaskCache, empty_dense_words,
};
#[allow(unused_imports)]
pub(crate) use master_proofs::PreparedMasterProofArtifact;
#[allow(unused_imports)]
pub(crate) use projections::DynamicBoundedObservationSets;
#[allow(unused_imports)]
pub(crate) use recursive::{
    BoundaryTerminalNwa, BoundaryTerminalNwaNode, BoundaryTerminalNwaTransition,
    BoundaryTerminalTrieNode, RecursiveParserLayout, RecursiveParserLeafLayout,
    RecursiveVirtualTokenizerStates, SegmentedBoundaryParser, SegmentedBoundaryShard,
    SegmentedBoundaryShardBackend, SegmentedBoundaryTerminalTrie, SegmentedParserComponent,
    SegmentedParserComponentTables, SegmentedParserLink, StaticDynamicOverlayMetadata,
};
#[allow(unused_imports)]
pub(crate) use regular::{
    DirectRegularDynamicFrontierCacheEntry, DirectRegularDynamicHotFrontier,
    DirectRegularParserStateAcceptance, DirectRegularTerminalSupport,
    DirectRegularWideFrontierAcceptance,
};
#[allow(unused_imports)]
pub(crate) use transitions::{
    FastCommitTemplateDfas, FastDwaTransitionRow, FastDwaTransitions, FastTemplateDfa,
    FastTemplateDfaState, FastTemplateDfasByTerminal, FastTemplateTransitionRow,
    FastTokenizerTransitions, IndexedDagDenseMask, IndexedDagDenseTransition,
    IndexedDagDenseTransitionMasks, IndexedDagDenseTransitionRow, IndexedDagDenseTransitions,
    TemplateDfasByTerminal,
};
#[allow(unused_imports)]
pub(crate) use trie::{
    DYNAMIC_MASK_LLG_MASTER_CACHE_ID, DynamicMaskSliceTrie, DynamicMaskTrie, DynamicMaskTrieEdge,
    DynamicMaskTrieFullWalkOp, DynamicMaskTrieNode, DynamicMaskTrieWalkEdge,
    dynamic_mask_llg_master_is_whitespace, dynamic_mask_llg_master_layout_class,
    dynamic_mask_llg_master_safe_chars, dynamic_mask_vocab_layout_class,
};
#[allow(unused_imports)]
pub(crate) use vocab_transfer::DynamicMaskVocabArtifact;
#[allow(unused_imports)]
pub(crate) use vocabulary::{
    DynamicMaskAliasStore, DynamicMaskLexerStateKey, DynamicMaskStateKey, DynamicMaskVocab,
    DynamicMaskVocabSource, PackedDynamicMaskTokenAliases, dynamic_mask_state_key_hash,
};

#[cfg(test)]
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::regex::Expr;
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::table::GLRTable;
#[cfg(test)]
use crate::compiler::stages::id_map_and_terminal_dwa::classify::{
    VocabPartitionDfa, classify_vocab_char_type,
};
use crate::compiler::stages::templates::characterize::TerminalCharacterization;
use crate::ds::weight::Weight;
use crate::grammar::flat::DirectRegularAutomaton;
use crate::grammar::flat::TerminalID;
use crate::runtime::mask_mapping::FinalMaskMapping;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
pub(crate) type PossibleMatchesByTerminal = BTreeMap<TerminalID, Weight>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct SpecialTokenTerminal {
    pub(crate) terminal_id: TerminalID,
    pub(crate) token_id: u32,
}

/// Version-scoped serde for the inverse token-id map. Current sectioned
/// artifacts carry only `original_token_to_internal`; the inverse is exactly
/// derivable from it and is rebuilt after the core section is decoded. Older
/// artifact versions leave this mode disabled and retain their historical wire
/// shape.
pub(crate) mod internal_token_inverse_artifact_serde {
    use std::cell::Cell;

    use serde::{Deserialize, Serialize};

    thread_local! {
        static OMIT_INVERSE: Cell<bool> = const { Cell::new(false) };
    }

    pub(crate) fn set_omit(enabled: bool) -> bool {
        OMIT_INVERSE.with(|mode| mode.replace(enabled))
    }

    pub fn serialize<S>(value: &[Vec<u32>], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if OMIT_INVERSE.with(Cell::get) {
            return ().serialize(serializer);
        }
        value.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Vec<u32>>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if OMIT_INVERSE.with(Cell::get) {
            <()>::deserialize(deserializer)?;
            return Ok(Vec::new());
        }
        Vec::<Vec<u32>>::deserialize(deserializer)
    }
}

/// Current-core serialization can omit the tokenizer-state inverse when it is
/// exactly derivable from the scalar state -> internal-TSID map. The default
/// mode preserves the historical `Vec<Vec<u32>>` bincode wire, so legacy
/// artifact decoding is unchanged.
pub(crate) mod internal_tsid_inverse_artifact_serde {
    use std::cell::Cell;

    use serde::{Deserialize, Serialize};

    thread_local! {
        static OMIT_INVERSE: Cell<bool> = const { Cell::new(false) };
    }

    pub(crate) fn set_omit(enabled: bool) -> bool {
        OMIT_INVERSE.with(|mode| mode.replace(enabled))
    }

    pub fn serialize<S>(value: &[Vec<u32>], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if OMIT_INVERSE.with(Cell::get) {
            return ().serialize(serializer);
        }
        value.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Vec<u32>>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if OMIT_INVERSE.with(Cell::get) {
            <()>::deserialize(deserializer)?;
            return Ok(Vec::new());
        }
        Vec::<Vec<u32>>::deserialize(deserializer)
    }
}

/// Compact v14+ wire form for the dense original-token -> internal-token map.
/// Internal IDs are normally only a few thousand wide even for 128k-token
/// vocabularies, so fixed-width u32 storage wastes roughly half this field.
/// Zero encodes the historical `u32::MAX` sentinel; ordinary IDs are stored as
/// `id + 1` varints. Older artifact versions leave this mode disabled.
pub(crate) mod original_token_map_artifact_serde;

/// Compact v14+ wire encoding for the immutable model-token byte vocabulary.
/// Ordinary LLM vocabs are dense in token id, so the historical
/// `BTreeMap<u32, Vec<u8>>` representation spends more bytes on map keys and
/// per-Vec lengths than on useful token data. The packed form stores a dense
/// sequence of varint lengths followed by token bytes, with a sparse fallback
/// for unusual vocabularies. Deserialization reconstructs the exact historical
/// in-memory BTreeMap, so compiler/composition/runtime APIs do not change.
pub(crate) mod token_bytes_artifact_serde;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum ConstraintRuntimeBackend {
    #[default]
    Static,
    Dynamic,
}

/// Compact persisted metadata for an unresolved external-grammar slot.
///
/// The terminal ID is an internal linker coordinate, never a public/model
/// token ID. Keeping this record beside the compiled parser artifact lets a
/// loaded parent be linked without parsing or recompiling its source grammar.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct LateGrammarSlot {
    pub(crate) name: String,
    pub(crate) terminal_id: TerminalID,
}

/// Fully compiled, immutable grammar constraint.
///
/// A `Constraint` is intended to be reused across generated sequences. Call
/// [`Constraint::start`] to create a mutable per-sequence state.
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Final generation policy, not part of the embeddable compiled body.
    /// The outer root artifact stores these separately from compiler data.
    pub(crate) end_tokens: Arc<[u32]>,
    pub(crate) runtime_backend: ConstraintRuntimeBackend,
    pub(crate) static_dynamic_overlay: Option<StaticDynamicOverlayMetadata>,
    /// Reusable component-local trigger metadata for dynamic composition.
    /// Ordinary static/dynamic compilation leaves this at `None` so there is
    /// no trigger construction cost unless explicitly requested in the future.
    pub(crate) boundary_trigger: BoundaryTrigger,
    /// Grammar-aware proper-prefix model-token summary used only by static
    /// composition boundary preparation.  It is intentionally independent of
    /// `boundary_trigger`: the latter is a dynamic-runtime accelerator with a
    /// different observation contract.  The lock permits lazy preparation by
    /// composition without mutating the immutable constraint API.
    pub(crate) boundary_candidate_summary: OnceLock<BoundaryCandidateSummary>,
    /// Named, compiler-generated linker terminals for unresolved
    /// `extern grammar` declarations. The token IDs backing these terminals
    /// are deliberately private and outside the model vocabulary; callers
    /// address slots only by `name` through the late-binding API.
    pub(crate) late_grammar_slots: Vec<LateGrammarSlot>,
    /// Runtime-only public vocabulary reconstructed from this immutable
    /// constraint for late subgrammar binding. Keeping it here lets repeated
    /// binds share `Vocab`'s pure derived-artifact cache (adjacent-byte index,
    /// tries, etc.) instead of rebuilding those structures for every child.
    pub(crate) late_bind_vocab: OnceLock<crate::Vocab>,
    /// Runtime-derived exact original-token sets for `Skip` terminals in a
    /// composed grammar. Each token is wholly in `L(skip)+`: it can be
    /// consumed as one or more complete instances of that scoped-ignore
    /// terminal with a lexer reset between instances. This is deliberately
    /// not serialized; it is cheap to rebuild from the retained terminal
    /// expression and vocabulary and therefore does not change artifact wire
    /// compatibility.
    pub(crate) scoped_ignore_only_tokens: Vec<(TerminalID, Box<[u32]>)>,
    /// Exact byte-token fusions `(fused, suffix)` grouped by scoped Skip. The
    /// fused token begins with one or more complete instances of the Skip
    /// language and the remaining bytes equal `suffix` exactly. If `suffix`
    /// is admitted by the ordinary static mask, `fused` is therefore admitted
    /// as well. Runtime-only for the same wire-compatibility reason as above.
    pub(crate) scoped_ignore_prefix_fusions: Vec<(TerminalID, Box<[(u32, u32)]>)>,
    pub(crate) parser_dwa: DWA,
    /// Current-format loaded constraints retain the immutable parser DWA in
    /// its compact canonical pools instead of reconstructing RangeSet/Weight
    /// objects. Compiler-created and legacy-loaded constraints leave this
    /// empty and use `parser_dwa` directly.
    pub(crate) packed_parser_dwa: Option<Arc<crate::automata::weighted::dwa::PackedRuntimeDwa>>,
    /// Runtime-only override for the parser-DWA start final. Composition uses
    /// this to suppress a component's standalone globally-erased ignore at the
    /// union root without materializing or mutating the full parser DWA.
    pub(crate) parser_start_final_override: Option<Weight>,
    /// Exact depth-one parser acceptance kept separate from the deeper parser
    /// DWA. Keys are encoded parser-state labels; values are already the
    /// transition/final-weight intersection for accepting after that one
    /// stack symbol.
    pub(crate) parser_top_accept: BTreeMap<i32, Weight>,
    /// Uncombined exact depth-one acceptance parts. Direct-regular grammars
    /// retain terminal completion weights separately to avoid constructing one
    /// large union weight per parser state at compile time.
    pub(crate) parser_top_accept_parts: BTreeMap<i32, Vec<Weight>>,
    /// Immediate-completion L1 terminal weights for direct-regular parsers.
    /// Kept once per grammar terminal rather than duplicated across every
    /// epsilon-closed parser row.
    pub(crate) direct_regular_l1_complete_by_terminal: BTreeMap<TerminalID, Weight>,
    pub(crate) packed_non_dwa_weights: Option<Arc<PackedNonDwaWeights>>,
    /// Runtime-derived exact acceptance summaries for wide direct-regular
    /// replace-top frontiers. Rebuilt after compile/load from the table and
    /// parser-top acceptance artifacts.
    pub(crate) direct_regular_wide_frontier_acceptance: Vec<DirectRegularWideFrontierAcceptance>,
    /// Runtime-only exact transition maps for the direct automaton's initial
    /// frontier and its single widest successor frontier. Dynamic masking
    /// repeatedly queries these two frontiers at token boundaries.
    pub(crate) direct_regular_dynamic_hot_frontiers: Vec<DirectRegularDynamicHotFrontier>,
    /// Runtime-derived exact dense acceptance for the broadest direct-regular
    /// parser row(s). This avoids replaying thousands of L1 terminal weights on
    /// every mask while keeping the cached result source-state exact.
    pub(crate) direct_regular_parser_state_acceptance: Vec<DirectRegularParserStateAcceptance>,
    /// Sparse terminal-level automaton retained for exact direct-regular
    /// runtime indexes. Static artifact format versioning covers this field.
    pub(crate) direct_regular_automaton: Option<DirectRegularAutomaton>,
    pub(crate) table: GLRTable,
    pub(crate) terminal_display_names: Vec<String>,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) boundary_completion_index: Option<
        Arc<crate::compiler::composition::boundary::precomputed_completion::PreparedCompletion>,
    >,
    /// Cached tokenizer topology flag. `Tokenizer::has_epsilon_transitions()`
    /// scans every tokenizer state, so runtime dispatch must not recompute it.
    pub(crate) tokenizer_has_epsilon_transitions: bool,
    pub(crate) ignore_terminal: Option<TerminalID>,
    pub(crate) special_token_terminals: Vec<SpecialTokenTerminal>,

    /// Runtime-only vocabulary data for direct dynamic masking.
    pub(crate) dynamic_mask_vocab: DynamicMaskVocab,
    /// Lazily materialized static-mode fallback vocabulary. Ordinary static
    /// masking never touches this; it is initialized only if an empty
    /// possible-matches table encounters a token-start exclusion.
    pub(crate) lazy_dynamic_mask_vocab: OnceLock<DynamicMaskVocab>,

    /// possible_matches keyed by grammar terminal id.
    ///
    /// An empty table may represent deferred possible-match construction in
    /// legacy code only.
    ///
    /// IMPORTANT: the dynamic possible-matches fallback is intentionally
    /// terrible and is planned for removal. New compiler paths MUST construct
    /// complete exact possible matches and MUST NOT set
    /// `possible_matches_complete` to false as an implementation shortcut.
    /// DO NOT REMOVE OR WEAKEN THIS COMMENT.
    ///
    /// Each Weight maps final shared internal tokenizer-state ids to token sets
    /// in the final shared constraint-internal vocab space. Parser-DWA weights
    /// and possible_matches weights are reconciled into this same space during
    /// compilation.
    pub(crate) possible_matches: PossibleMatchesByTerminal,
    /// Whether `possible_matches` is a complete table. New static constraints
    /// must set this to true. False exists only for legacy dynamic/deferred
    /// construction and is not permitted as a fallback strategy for new
    /// compiler features.
    pub(crate) possible_matches_complete: bool,
    pub(crate) state_to_internal_tsid: Vec<u32>,
    pub(crate) internal_tsid_to_states: Vec<Vec<u32>>,
    /// Ordinary tokenizers have one internal TSID per physical state, making
    /// `internal_tsid_to_states` the exact bucket inverse of
    /// `state_to_internal_tsid`. Current artifacts can omit that redundant
    /// allocation and reconstruct it only for composition/debug paths.
    pub(crate) deferred_internal_tsid_to_states: OnceLock<Vec<Vec<u32>>>,
    /// Composition-preparation cache: row `t` lists original model-token IDs
    /// which, from this component's lexer reset, complete terminal `t` exactly
    /// at the end of the model token.  This is not part of the historical inner
    /// `Constraint` bincode layout; artifact V13 stores it in the outer
    /// envelope so V12 constraints remain loadable unchanged.
    pub(crate) composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
    /// Named unresolved `extern grammar` slots retained by a compiled parent.
    /// Values are parent-local hidden placeholder terminal IDs. Stored in the
    /// outer composition metadata so cached parents can be rebound after load.
    pub(crate) unbound_grammar_placeholders: BTreeMap<String, TerminalID>,
    /// Composition-time parser stack-effect templates retained from the
    /// original compile. These are the unspecialized per-terminal DFAs used to
    /// build parser DWAs, so a later linker can transport unchanged component
    /// behavior instead of re-characterizing the component LR table.
    /// Stored in the outer versioned artifact envelope for compatibility with
    /// older inner `Constraint` bincode layouts.
    pub(crate) composition_parser_templates_by_terminal: Vec<Option<UnweightedDfa>>,
    /// Composition-time symbolic parser characterizations retained from the
    /// original compile. A later linker can append only the boundary-induced
    /// reductions/rereductions and recompile affected terminal templates,
    /// rather than re-solving the component's reduction closure from scratch.
    pub(crate) composition_parser_characterizations_by_terminal:
        Vec<Option<TerminalCharacterization>>,
    /// Composition-time grammar adjacency summary. Stored in the outer
    /// versioned artifact envelope so older inner `Constraint` layouts remain
    /// loadable unchanged.
    pub(crate) composition_grammar_summary: Option<CompositionGrammarSummary>,
    /// Runtime-only inverse lexer-metadata index used by compiled-constraint
    /// composition. Row `t` lists exactly the raw tokenizer states whose
    /// epsilon closure has terminal `t` matched or still reachable.
    pub(crate) terminal_live_states: Vec<Vec<u32>>,
    /// Runtime-only CSR view of the exact state -> internal-TSID relation.
    /// Ordinary tokenizers have one entry per state. A fully determinized
    /// runtime lexer may represent several old lexer states and therefore
    /// several independent TSID lanes in one physical state.
    pub(crate) state_internal_tsid_offsets: Vec<u32>,
    pub(crate) state_internal_tsids: Vec<u32>,
    /// Final-runtime subset states followed by an exact copy of the source
    /// tokenizer. `runtime_source_state_offset` is the boundary between the
    /// two coordinates. Empty metadata means no runtime-only determinization.
    pub(crate) runtime_source_state_offset: Option<u32>,
    /// CSR offsets for product-state -> exact source-state subset. There is one
    /// row per product state and therefore `product_state_count + 1` offsets.
    pub(crate) runtime_product_source_offsets: Vec<u32>,
    pub(crate) runtime_product_source_states: Vec<u32>,
    /// Scalar source representative for product states that are exactly one
    /// source state's epsilon closure; `u32::MAX` otherwise.
    pub(crate) runtime_product_exact_source_states: Vec<u32>,
    /// Runtime-only inverse used to re-coalesce a uniform source frontier.
    pub(crate) runtime_product_state_by_source_subset: FxHashMap<Box<[u32]>, u32>,
    pub(crate) template_dfas_by_terminal: TemplateDfasByTerminal,
    /// Runtime-only compact transition view for commit template products.
    pub(crate) fast_template_dfas_by_terminal: FastTemplateDfasByTerminal,
    /// Original token -> final shared constraint-internal token id.
    ///
    /// This is not necessarily equal to the parser-DWA compaction vocab map
    /// produced before possible-match reconciliation. It may contain additional
    /// splits required by possible_matches.
    pub(crate) original_token_to_internal: Vec<u32>,
    /// Current-format loads retain the fixed-width original-token map inside
    /// the owned artifact instead of expanding all model-token entries to
    /// `u32`. Ordinary static mask/commit performs direct packed lookups; only
    /// composition/debug-style bulk access materializes the vector lazily.
    pub(crate) packed_original_token_to_internal:
        Option<Arc<original_token_map_artifact_serde::PackedOriginalTokenMap>>,
    pub(crate) deferred_original_token_to_internal: OnceLock<Vec<u32>>,
    /// Final shared constraint-internal token id -> original token ids.
    ///
    /// Parser-DWA weights and Constraint.possible_matches bitmaps both use these
    /// final internal token ids.
    pub(crate) internal_token_to_tokens: Vec<Vec<u32>>,
    /// Current-format loads can defer reconstructing the explicit inverse of
    /// `original_token_to_internal`. Static mask/commit only needs the internal
    /// token count and the already-serialized mask fragments; composition and
    /// token-space expansion materialize this inverse on first use.
    pub(crate) deferred_internal_token_to_tokens: OnceLock<Vec<Vec<u32>>>,
    pub(crate) token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    /// Indexed immutable vocabulary used directly by runtime token lookup and
    /// iteration. Loaded constraints can point into artifact backing; compiled
    /// constraints own the same indexed representation alongside the source
    /// map used by compiler/composition code.
    pub(crate) packed_token_bytes: Option<Arc<token_bytes_artifact_serde::PackedTokenBytes>>,
    // Compiler-side scratch/result metadata only. No runtime or composition
    // path reads this field; composition rebuilds the map for its result when
    // needed. Persisting it duplicated token bytes inside every constraint.
    pub(crate) internal_token_bytes: BTreeMap<u32, Vec<u8>>,
    pub(crate) token_bytes_dense: Vec<Option<Box<[u8]>>>,

    /// Precomputed bitmask fragments for each internal token.
    /// `internal_token_buf_masks[i]` contains (word_index, or_mask) pairs
    /// for all original tokens that map to internal token `i`.
    pub(crate) internal_token_buf_masks: Vec<InternalTokenBufMasks>,
    /// Precomputed combined buf output for each group of 64 internal tokens.
    /// `word_group_buf_masks[w]` is the combined mask for internal tokens [w*64 .. (w+1)*64).
    /// Used as a fast path in `or_to_buf` when a dense word is all-ones (!0u64).
    pub(crate) word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 128 internal tokens.
    pub(crate) pair_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 256 internal tokens.
    pub(crate) quad_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 512 internal tokens.
    pub(crate) super_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 1024 internal tokens.
    pub(crate) mega_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 2048 internal tokens.
    pub(crate) giga_word_group_buf_masks: DenseBufMaskRows,
    /// Sparse OR-union for each 64-token internal word group.
    pub(crate) word_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense prefix-unions of 64-token internal word groups.
    ///
    /// `word_group_prefix_buf_masks[i]` is the OR-union of word groups
    /// `[0, i)`. Internal-token groups are disjoint in original-token space,
    /// so `prefix[end] & !prefix[start]` is the exact dense mask for a full
    /// internal-word run `[start, end)`.
    pub(crate) word_group_prefix_buf_masks: DenseBufMaskRows,
    /// Prefix sums of `word_group_sparse_masks[i].len()`.
    pub(crate) word_group_sparse_prefix_entries: Vec<usize>,
    pub(crate) quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense output masks for quad groups whose sparse replay is more
    /// expensive than a sequential output-buffer scan.
    pub(crate) quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    pub(crate) byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense output masks for byte groups whose sparse replay is more
    /// expensive than a sequential output-buffer scan.
    pub(crate) byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
    pub(crate) word_group_sparse_total_entries: usize,
    pub(crate) word_group_sparse_max_entries: usize,
    /// Precomputed buf output for the full internal token universe (OR of all word_group_buf_masks).
    pub(crate) all_tokens_buf_mask: Box<[u32]>,
    pub(crate) internal_token_dense_words: usize,
    pub(crate) weight_token_dense_masks: DenseWeightMaskCache,
    /// Dense masks for wide token sets retained in a current-format packed
    /// parser DWA. Unlike `weight_token_dense_masks`, these are keyed by the
    /// packed token-set id and can therefore be rebuilt directly from the
    /// artifact without materializing RangeSet/Weight objects.
    pub(crate) packed_dwa_token_dense_masks: PackedDwaDenseWeightMaskCache,
    pub(crate) weight_token_buf_masks: DenseWeightBufMaskCache,
    pub(crate) weight_token_sparse_buf_masks: SparseWeightBufMaskCache,
    /// Final-weight token sets eligible for the direct sparse-intersection
    /// path. Their full output masks are intentionally not materialized: the
    /// runtime intersects them with the current dense state on every use.
    pub(crate) range_final_token_sets: RangeFinalTokenSetCache,
    /// Precomputed dense bitmask for the seed phase: for each (tokenizer_state, terminal_id),
    /// the dense bitmap of internal tokens that terminal covers in that state.
    pub(crate) seed_terminal_dense: SeedTerminalDenseMasks,
    /// Exact masks lazily materialized for delayed-exclusion pairs that are not
    /// represented by `possible_matches`. Shared across sequence states cloned
    /// from this immutable constraint.
    pub(crate) seed_terminal_dense_fallback: Arc<Mutex<SeedTerminalDenseMasks>>,
    /// Dense bitmap of the full internal token universe.
    pub(crate) seed_universe_dense: DenseWords,
    /// Fast DWA transition lookup (FxHashMap instead of BTreeMap).
    /// Built from parser_dwa.states at load/build time.
    pub(crate) dwa_fast_transitions: FastDwaTransitions,
    /// Runtime-only readiness marker for caches derived from the final parser
    /// DWA and final internal-token coordinate. Composition may build these at
    /// the final parser-union boundary so generic post-link finalization does
    /// not rescan the same parser artifact.
    pub(crate) parser_runtime_caches_prebuilt: bool,
    /// Runtime-only parser-DWA transitions with exact dense masks materialized
    /// for the final internal tokenizer states present in each transition
    /// weight; absent states are implicitly empty. Indexed-DAG masking uses
    /// this table directly instead of hashing a transition tuple and lazily
    /// rebuilding the same dense transition record at runtime.
    pub(crate) indexed_dag_dense_transitions: IndexedDagDenseTransitions,
    /// Runtime-only exact dense final weights, indexed by parser-DWA state.
    /// This is the final-weight analogue of `indexed_dag_dense_transitions`:
    /// absent tokenizer states are empty, and full final weights stay implicit.
    pub(crate) indexed_dag_dense_finals: Vec<IndexedDagDenseTransitionMasks>,
    /// Dense tokenizer transition lookup for commit-time byte scans.
    pub(crate) tokenizer_fast_transitions: FastTokenizerTransitions,
    /// Dense buf masks for "heavy" internal tokens (those with many buf entries).
    /// Indexed by internal token ID; None for light tokens.
    pub(crate) heavy_token_dense_masks: Vec<Option<Box<[u32]>>>,
    /// Flattened contiguous array of all internal token buf mask entries.
    /// All tokens' (word_index, or_mask) pairs concatenated in token order.
    /// Improves cache locality vs separate Vec allocations per token.
    pub(crate) internal_token_buf_flat: Box<[PackedInternalTokenBufMask]>,
    /// Current IBM3 loads can retain the runtime-native flat sparse-mask slab
    /// directly inside the owned artifact instead of copying ~0.5-1 MiB.
    pub(crate) backed_internal_token_buf_flat: Option<BackedInternalTokenBufMasks>,
    /// Offsets into `internal_token_buf_flat` for each internal token.
    /// `internal_token_buf_flat[offsets[i]..offsets[i+1]]` gives token i's entries.
    /// Length = n_internal + 1 (sentinel at end).
    pub(crate) internal_token_buf_offsets: Box<[u32]>,
    /// Pre-computed total cost (sum of entry counts) for all internal tokens.
    /// Used to avoid O(n_internal) cost analysis in the convert phase.
    pub(crate) total_internal_buf_cost: usize,
    /// Indices of heavy tokens for fast iteration. Length == n_heavy_tokens.
    pub(crate) heavy_token_indices: Vec<usize>,
    /// Total cost of all heavy tokens combined (n_heavy Ã— buf_len).
    pub(crate) heavy_total_cost: usize,
    /// Average cost per light token: (total_cost - heavy_total) / n_light.
    /// Pre-multiplied by 256 for fixed-point arithmetic to avoid float.
    pub(crate) light_avg_cost_x256: usize,
    /// Exact materialization cost per internal token, after heavy-token dense masks
    /// have been chosen.
    pub(crate) internal_token_buf_op_costs: Vec<usize>,
    /// Exact materialization cost per 64-token internal word group.
    pub(crate) word_group_buf_op_costs: Vec<usize>,
    /// Self-contained final internal-token -> original-token bitset materializer.
    pub(crate) final_mask_mapping: FinalMaskMapping,
    /// Optional exact quotient of positive parser-state labels used by composed
    /// parser DWAs. Entry `s` is a synthetic fallback label for parser state
    /// `s`; `i32::MAX` means no component-local fallback. Concrete parser-state
    /// transitions always take precedence, followed by this label, then the
    /// ordinary global DEFAULT. Empty for ordinary non-composed constraints.
    pub(crate) parser_state_domain_labels: Vec<i32>,
    /// Exact source expression for the globally erasable ignore terminal.
    ///
    /// Tokenizer source expressions are compile-time data and are normally
    /// omitted from artifacts. Retaining this one expression lets a loaded
    /// compiled constraint participate in later subgrammar composition without
    /// conservatively degrading an identical global ignore into scoped skips.
    pub(crate) ignore_expr: Option<Expr>,
    /// Exact current-format artifact backing for an unchanged loaded
    /// constraint. Runtime cache rebuilds do not alter serialized semantics,
    /// so resave can return a single bulk copy instead of rediscovering and
    /// re-encoding the same canonical pools.
    pub(crate) serialized_artifact_cache: Option<Arc<Vec<u8>>>,
    /// Current-format terminal source expressions can be retained as their
    /// canonical bincode payload instead of recursively rebuilding every Expr
    /// node during an ordinary static load. Composition materializes the list
    /// lazily through `retained_terminal_exprs` when it actually needs source
    /// language proofs.
    pub(crate) deferred_terminal_exprs_blob: Option<DeferredTerminalExprBytes>,
    pub(crate) deferred_terminal_exprs: OnceLock<Arc<[Expr]>>,
    /// Serialized composition-only metadata (reset-token rows, parser template
    /// cache, symbolic characterizations, and grammar summary). Current-format
    /// loads keep this cold section backed by the artifact and materialize it
    /// only if the constraint is later used as a composition component.
    pub(crate) deferred_composition_metadata_blob: Option<DeferredCompositionMetadataBytes>,
    /// Runtime-only marker distinguishing "the lightweight linking metadata
    /// has already been decoded" from "the heavy static compiler-cache blob is
    /// still deferred". Dynamic A+B intentionally keeps the latter deferred.
    pub(crate) composition_link_metadata_materialized: bool,
    /// Large current-format GLR rule vectors are composition metadata rather
    /// than runtime parser data. Keep their canonical payload undecoded during
    /// ordinary load; composition materializes it lazily through
    /// `retained_table_rules`.
    pub(crate) deferred_table_rules_blob:
        Option<crate::compiler::glr::table::artifact_serde::DeferredRuleBytes>,
    pub(crate) deferred_table_rules: OnceLock<Arc<[crate::grammar::flat::Rule]>>,
}

// Private Serde definition used only by the versioned artifact encoder/decoder.
// Keeping this remote definition separate prevents `Constraint` itself from
// implementing Serde, so `Constraint::save`/`Constraint::load` remain the only
// public persistence contract.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(remote = "Constraint")]
pub(crate) struct ConstraintSerde {
    #[serde(skip, default)]
    pub(crate) end_tokens: Arc<[u32]>,
    #[serde(default)]
    pub(crate) runtime_backend: ConstraintRuntimeBackend,
    #[serde(skip, default)]
    pub(crate) static_dynamic_overlay: Option<StaticDynamicOverlayMetadata>,
    #[serde(skip, default)]
    pub(crate) boundary_trigger: BoundaryTrigger,
    #[serde(skip, default)]
    pub(crate) boundary_candidate_summary: OnceLock<BoundaryCandidateSummary>,
    #[serde(skip, default)]
    pub(crate) late_grammar_slots: Vec<LateGrammarSlot>,
    #[serde(skip, default)]
    pub(crate) late_bind_vocab: OnceLock<crate::Vocab>,
    /// Runtime-derived exact original-token sets for `Skip` terminals in a
    /// composed grammar. Each token is wholly in `L(skip)+`: it can be
    /// consumed as one or more complete instances of that scoped-ignore
    /// terminal with a lexer reset between instances. This is deliberately
    /// not serialized; it is cheap to rebuild from the retained terminal
    /// expression and vocabulary and therefore does not change artifact wire
    /// compatibility.
    #[serde(skip, default)]
    pub(crate) scoped_ignore_only_tokens: Vec<(TerminalID, Box<[u32]>)>,
    /// Exact byte-token fusions `(fused, suffix)` grouped by scoped Skip. The
    /// fused token begins with one or more complete instances of the Skip
    /// language and the remaining bytes equal `suffix` exactly. If `suffix`
    /// is admitted by the ordinary static mask, `fused` is therefore admitted
    /// as well. Runtime-only for the same wire-compatibility reason as above.
    #[serde(skip, default)]
    pub(crate) scoped_ignore_prefix_fusions: Vec<(TerminalID, Box<[(u32, u32)]>)>,
    pub(crate) parser_dwa: DWA,
    /// Current-format loaded constraints retain the immutable parser DWA in
    /// its compact canonical pools instead of reconstructing RangeSet/Weight
    /// objects. Compiler-created and legacy-loaded constraints leave this
    /// empty and use `parser_dwa` directly.
    #[serde(skip, default)]
    pub(crate) packed_parser_dwa: Option<Arc<crate::automata::weighted::dwa::PackedRuntimeDwa>>,
    #[serde(skip, default)]
    pub(crate) parser_start_final_override: Option<Weight>,
    /// Exact depth-one parser acceptance kept separate from the deeper parser
    /// DWA. Keys are encoded parser-state labels; values are already the
    /// transition/final-weight intersection for accepting after that one
    /// stack symbol.
    #[serde(default)]
    pub(crate) parser_top_accept: BTreeMap<i32, Weight>,
    /// Uncombined exact depth-one acceptance parts. Direct-regular grammars
    /// retain terminal completion weights separately to avoid constructing one
    /// large union weight per parser state at compile time.
    #[serde(default)]
    pub(crate) parser_top_accept_parts: BTreeMap<i32, Vec<Weight>>,
    /// Immediate-completion L1 terminal weights for direct-regular parsers.
    /// Kept once per grammar terminal rather than duplicated across every
    /// epsilon-closed parser row.
    #[serde(default)]
    pub(crate) direct_regular_l1_complete_by_terminal: BTreeMap<TerminalID, Weight>,
    #[serde(skip, default)]
    pub(crate) packed_non_dwa_weights: Option<Arc<PackedNonDwaWeights>>,
    /// Runtime-derived exact acceptance summaries for wide direct-regular
    /// replace-top frontiers. Rebuilt after compile/load from the table and
    /// parser-top acceptance artifacts.
    #[serde(skip, default)]
    pub(crate) direct_regular_wide_frontier_acceptance: Vec<DirectRegularWideFrontierAcceptance>,
    /// Runtime-only exact transition maps for the direct automaton's initial
    /// frontier and its single widest successor frontier. Dynamic masking
    /// repeatedly queries these two frontiers at token boundaries.
    #[serde(skip, default)]
    pub(crate) direct_regular_dynamic_hot_frontiers: Vec<DirectRegularDynamicHotFrontier>,
    /// Runtime-derived exact dense acceptance for the broadest direct-regular
    /// parser row(s). This avoids replaying thousands of L1 terminal weights on
    /// every mask while keeping the cached result source-state exact.
    #[serde(skip, default)]
    pub(crate) direct_regular_parser_state_acceptance: Vec<DirectRegularParserStateAcceptance>,
    /// Sparse terminal-level automaton retained for exact direct-regular
    /// runtime indexes. Static artifact format versioning covers this field.
    #[serde(default)]
    pub(crate) direct_regular_automaton: Option<DirectRegularAutomaton>,
    #[serde(with = "crate::compiler::glr::table::artifact_serde")]
    pub(crate) table: GLRTable,
    #[serde(default)]
    pub(crate) terminal_display_names: Vec<String>,
    #[serde(with = "crate::runtime::artifact::immutable_tokenizer_serde")]
    pub(crate) tokenizer: Arc<Tokenizer>,
    #[serde(skip)]
    pub(crate) boundary_completion_index: Option<
        Arc<crate::compiler::composition::boundary::precomputed_completion::PreparedCompletion>,
    >,
    /// Cached tokenizer topology flag. `Tokenizer::has_epsilon_transitions()`
    /// scans every tokenizer state, so runtime dispatch must not recompute it.
    #[serde(skip, default)]
    pub(crate) tokenizer_has_epsilon_transitions: bool,
    #[serde(default)]
    pub(crate) ignore_terminal: Option<TerminalID>,
    #[serde(default)]
    pub(crate) special_token_terminals: Vec<SpecialTokenTerminal>,

    /// Runtime-only vocabulary data for direct dynamic masking.
    #[serde(skip, default)]
    pub(crate) dynamic_mask_vocab: DynamicMaskVocab,
    /// Lazily materialized static-mode fallback vocabulary. Ordinary static
    /// masking never touches this; it is initialized only if an empty
    /// possible-matches table encounters a token-start exclusion.
    #[serde(skip, default)]
    pub(crate) lazy_dynamic_mask_vocab: OnceLock<DynamicMaskVocab>,

    /// possible_matches keyed by grammar terminal id.
    ///
    /// An empty table may represent deferred possible-match construction in
    /// legacy code only.
    ///
    /// IMPORTANT: the dynamic possible-matches fallback is intentionally
    /// terrible and is planned for removal. New compiler paths MUST construct
    /// complete exact possible matches and MUST NOT set
    /// `possible_matches_complete` to false as an implementation shortcut.
    /// DO NOT REMOVE OR WEAKEN THIS COMMENT.
    ///
    /// Each Weight maps final shared internal tokenizer-state ids to token sets
    /// in the final shared constraint-internal vocab space. Parser-DWA weights
    /// and possible_matches weights are reconciled into this same space during
    /// compilation.
    pub(crate) possible_matches: PossibleMatchesByTerminal,
    /// Whether `possible_matches` is a complete table. New static constraints
    /// must set this to true. False exists only for legacy dynamic/deferred
    /// construction and is not permitted as a fallback strategy for new
    /// compiler features.
    #[serde(default)]
    pub(crate) possible_matches_complete: bool,
    pub(crate) state_to_internal_tsid: Vec<u32>,
    #[serde(default, with = "internal_tsid_inverse_artifact_serde")]
    pub(crate) internal_tsid_to_states: Vec<Vec<u32>>,
    /// Ordinary tokenizers have one internal TSID per physical state, making
    /// `internal_tsid_to_states` the exact bucket inverse of
    /// `state_to_internal_tsid`. Current artifacts can omit that redundant
    /// allocation and reconstruct it only for composition/debug paths.
    #[serde(skip, default)]
    pub(crate) deferred_internal_tsid_to_states: OnceLock<Vec<Vec<u32>>>,
    /// Composition-preparation cache: row `t` lists original model-token IDs
    /// which, from this component's lexer reset, complete terminal `t` exactly
    /// at the end of the model token.  This is not part of the historical inner
    /// `Constraint` bincode layout; artifact V13 stores it in the outer
    /// envelope so V12 constraints remain loadable unchanged.
    #[serde(skip, default)]
    pub(crate) composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
    /// Named unresolved `extern grammar` slots retained by a compiled parent.
    /// Values are parent-local hidden placeholder terminal IDs. Stored in the
    /// outer composition metadata so cached parents can be rebound after load.
    #[serde(skip, default)]
    pub(crate) unbound_grammar_placeholders: BTreeMap<String, TerminalID>,
    /// Composition-time parser stack-effect templates retained from the
    /// original compile. These are the unspecialized per-terminal DFAs used to
    /// build parser DWAs, so a later linker can transport unchanged component
    /// behavior instead of re-characterizing the component LR table.
    /// Stored in the outer versioned artifact envelope for compatibility with
    /// older inner `Constraint` bincode layouts.
    #[serde(skip, default)]
    pub(crate) composition_parser_templates_by_terminal: Vec<Option<UnweightedDfa>>,
    /// Composition-time symbolic parser characterizations retained from the
    /// original compile. A later linker can append only the boundary-induced
    /// reductions/rereductions and recompile affected terminal templates,
    /// rather than re-solving the component's reduction closure from scratch.
    #[serde(skip, default)]
    pub(crate) composition_parser_characterizations_by_terminal:
        Vec<Option<TerminalCharacterization>>,
    /// Composition-time grammar adjacency summary. Stored in the outer
    /// versioned artifact envelope so older inner `Constraint` layouts remain
    /// loadable unchanged.
    #[serde(skip, default)]
    pub(crate) composition_grammar_summary: Option<CompositionGrammarSummary>,
    /// Runtime-only inverse lexer-metadata index used by compiled-constraint
    /// composition. Row `t` lists exactly the raw tokenizer states whose
    /// epsilon closure has terminal `t` matched or still reachable.
    #[serde(skip, default)]
    pub(crate) terminal_live_states: Vec<Vec<u32>>,
    /// Runtime-only CSR view of the exact state -> internal-TSID relation.
    /// Ordinary tokenizers have one entry per state. A fully determinized
    /// runtime lexer may represent several old lexer states and therefore
    /// several independent TSID lanes in one physical state.
    #[serde(skip, default)]
    pub(crate) state_internal_tsid_offsets: Vec<u32>,
    #[serde(skip, default)]
    pub(crate) state_internal_tsids: Vec<u32>,
    /// Final-runtime subset states followed by an exact copy of the source
    /// tokenizer. `runtime_source_state_offset` is the boundary between the
    /// two coordinates. Empty metadata means no runtime-only determinization.
    #[serde(default)]
    pub(crate) runtime_source_state_offset: Option<u32>,
    /// CSR offsets for product-state -> exact source-state subset. There is one
    /// row per product state and therefore `product_state_count + 1` offsets.
    #[serde(default)]
    pub(crate) runtime_product_source_offsets: Vec<u32>,
    #[serde(default)]
    pub(crate) runtime_product_source_states: Vec<u32>,
    /// Scalar source representative for product states that are exactly one
    /// source state's epsilon closure; `u32::MAX` otherwise.
    #[serde(default)]
    pub(crate) runtime_product_exact_source_states: Vec<u32>,
    /// Runtime-only inverse used to re-coalesce a uniform source frontier.
    #[serde(skip, default)]
    pub(crate) runtime_product_state_by_source_subset: FxHashMap<Box<[u32]>, u32>,
    pub(crate) template_dfas_by_terminal: TemplateDfasByTerminal,
    /// Runtime-only compact transition view for commit template products.
    #[serde(skip, default)]
    pub(crate) fast_template_dfas_by_terminal: FastTemplateDfasByTerminal,
    /// Original token -> final shared constraint-internal token id.
    ///
    /// This is not necessarily equal to the parser-DWA compaction vocab map
    /// produced before possible-match reconciliation. It may contain additional
    /// splits required by possible_matches.
    #[serde(default, with = "original_token_map_artifact_serde")]
    pub(crate) original_token_to_internal: Vec<u32>,
    /// Current-format loads retain the fixed-width original-token map inside
    /// the owned artifact instead of expanding all model-token entries to
    /// `u32`. Ordinary static mask/commit performs direct packed lookups; only
    /// composition/debug-style bulk access materializes the vector lazily.
    #[serde(skip, default)]
    pub(crate) packed_original_token_to_internal:
        Option<Arc<original_token_map_artifact_serde::PackedOriginalTokenMap>>,
    #[serde(skip, default)]
    pub(crate) deferred_original_token_to_internal: OnceLock<Vec<u32>>,
    /// Final shared constraint-internal token id -> original token ids.
    ///
    /// Parser-DWA weights and Constraint.possible_matches bitmaps both use these
    /// final internal token ids.
    #[serde(default, with = "internal_token_inverse_artifact_serde")]
    pub(crate) internal_token_to_tokens: Vec<Vec<u32>>,
    /// Current-format loads can defer reconstructing the explicit inverse of
    /// `original_token_to_internal`. Static mask/commit only needs the internal
    /// token count and the already-serialized mask fragments; composition and
    /// token-space expansion materialize this inverse on first use.
    #[serde(skip, default)]
    pub(crate) deferred_internal_token_to_tokens: OnceLock<Vec<Vec<u32>>>,
    #[serde(with = "token_bytes_artifact_serde")]
    pub(crate) token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    /// Indexed immutable vocabulary used directly by runtime token lookup and
    /// iteration. Loaded constraints can point into artifact backing; compiled
    /// constraints own the same indexed representation alongside the source
    /// map used by compiler/composition code.
    #[serde(skip, default)]
    pub(crate) packed_token_bytes: Option<Arc<token_bytes_artifact_serde::PackedTokenBytes>>,
    // Compiler-side scratch/result metadata only. No runtime or composition
    // path reads this field; composition rebuilds the map for its result when
    // needed. Persisting it duplicated token bytes inside every constraint.
    #[serde(skip, default)]
    pub(crate) internal_token_bytes: BTreeMap<u32, Vec<u8>>,
    #[serde(skip)]
    pub(crate) token_bytes_dense: Vec<Option<Box<[u8]>>>,

    /// Precomputed bitmask fragments for each internal token.
    /// `internal_token_buf_masks[i]` contains (word_index, or_mask) pairs
    /// for all original tokens that map to internal token `i`.
    #[serde(skip)]
    pub(crate) internal_token_buf_masks: Vec<InternalTokenBufMasks>,
    /// Precomputed combined buf output for each group of 64 internal tokens.
    /// `word_group_buf_masks[w]` is the combined mask for internal tokens [w*64 .. (w+1)*64).
    /// Used as a fast path in `or_to_buf` when a dense word is all-ones (!0u64).
    #[serde(skip)]
    pub(crate) word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 128 internal tokens.
    #[serde(skip)]
    pub(crate) pair_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 256 internal tokens.
    #[serde(skip)]
    pub(crate) quad_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 512 internal tokens.
    #[serde(skip)]
    pub(crate) super_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 1024 internal tokens.
    #[serde(skip)]
    pub(crate) mega_word_group_buf_masks: DenseBufMaskRows,
    /// Precomputed dense output masks for groups of 2048 internal tokens.
    #[serde(skip)]
    pub(crate) giga_word_group_buf_masks: DenseBufMaskRows,
    /// Sparse OR-union for each 64-token internal word group.
    #[serde(skip)]
    pub(crate) word_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense prefix-unions of 64-token internal word groups.
    ///
    /// `word_group_prefix_buf_masks[i]` is the OR-union of word groups
    /// `[0, i)`. Internal-token groups are disjoint in original-token space,
    /// so `prefix[end] & !prefix[start]` is the exact dense mask for a full
    /// internal-word run `[start, end)`.
    #[serde(skip)]
    pub(crate) word_group_prefix_buf_masks: DenseBufMaskRows,
    /// Prefix sums of `word_group_sparse_masks[i].len()`.
    #[serde(skip)]
    pub(crate) word_group_sparse_prefix_entries: Vec<usize>,
    #[serde(skip)]
    pub(crate) quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense output masks for quad groups whose sparse replay is more
    /// expensive than a sequential output-buffer scan.
    #[serde(skip)]
    pub(crate) quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    #[serde(skip)]
    pub(crate) byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense output masks for byte groups whose sparse replay is more
    /// expensive than a sequential output-buffer scan.
    #[serde(skip)]
    pub(crate) byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
    pub(crate) word_group_sparse_total_entries: usize,
    #[serde(skip)]
    pub(crate) word_group_sparse_max_entries: usize,
    /// Precomputed buf output for the full internal token universe (OR of all word_group_buf_masks).
    #[serde(skip)]
    pub(crate) all_tokens_buf_mask: Box<[u32]>,
    #[serde(skip)]
    pub(crate) internal_token_dense_words: usize,
    #[serde(skip)]
    pub(crate) weight_token_dense_masks: DenseWeightMaskCache,
    /// Dense masks for wide token sets retained in a current-format packed
    /// parser DWA. Unlike `weight_token_dense_masks`, these are keyed by the
    /// packed token-set id and can therefore be rebuilt directly from the
    /// artifact without materializing RangeSet/Weight objects.
    #[serde(skip, default)]
    pub(crate) packed_dwa_token_dense_masks: PackedDwaDenseWeightMaskCache,
    #[serde(skip)]
    pub(crate) weight_token_buf_masks: DenseWeightBufMaskCache,
    #[serde(skip)]
    pub(crate) weight_token_sparse_buf_masks: SparseWeightBufMaskCache,
    /// Final-weight token sets eligible for the direct sparse-intersection
    /// path. Their full output masks are intentionally not materialized: the
    /// runtime intersects them with the current dense state on every use.
    #[serde(skip)]
    pub(crate) range_final_token_sets: RangeFinalTokenSetCache,
    /// Precomputed dense bitmask for the seed phase: for each (tokenizer_state, terminal_id),
    /// the dense bitmap of internal tokens that terminal covers in that state.
    #[serde(skip)]
    pub(crate) seed_terminal_dense: SeedTerminalDenseMasks,
    /// Exact masks lazily materialized for delayed-exclusion pairs that are not
    /// represented by `possible_matches`. Shared across sequence states cloned
    /// from this immutable constraint.
    #[serde(skip, default)]
    pub(crate) seed_terminal_dense_fallback: Arc<Mutex<SeedTerminalDenseMasks>>,
    /// Dense bitmap of the full internal token universe.
    #[serde(skip, default = "empty_dense_words")]
    pub(crate) seed_universe_dense: DenseWords,
    /// Fast DWA transition lookup (FxHashMap instead of BTreeMap).
    /// Built from parser_dwa.states at load/build time.
    #[serde(skip)]
    pub(crate) dwa_fast_transitions: FastDwaTransitions,
    /// Runtime-only readiness marker for caches derived from the final parser
    /// DWA and final internal-token coordinate. Composition may build these at
    /// the final parser-union boundary so generic post-link finalization does
    /// not rescan the same parser artifact.
    #[serde(skip, default)]
    pub(crate) parser_runtime_caches_prebuilt: bool,
    /// Runtime-only parser-DWA transitions with exact dense masks materialized
    /// for the final internal tokenizer states present in each transition
    /// weight; absent states are implicitly empty. Indexed-DAG masking uses
    /// this table directly instead of hashing a transition tuple and lazily
    /// rebuilding the same dense transition record at runtime.
    #[serde(skip, default)]
    pub(crate) indexed_dag_dense_transitions: IndexedDagDenseTransitions,
    /// Runtime-only exact dense final weights, indexed by parser-DWA state.
    /// This is the final-weight analogue of `indexed_dag_dense_transitions`:
    /// absent tokenizer states are empty, and full final weights stay implicit.
    #[serde(skip, default)]
    pub(crate) indexed_dag_dense_finals: Vec<IndexedDagDenseTransitionMasks>,
    /// Dense tokenizer transition lookup for commit-time byte scans.
    #[serde(skip)]
    pub(crate) tokenizer_fast_transitions: FastTokenizerTransitions,
    /// Dense buf masks for "heavy" internal tokens (those with many buf entries).
    /// Indexed by internal token ID; None for light tokens.
    #[serde(skip)]
    pub(crate) heavy_token_dense_masks: Vec<Option<Box<[u32]>>>,
    /// Flattened contiguous array of all internal token buf mask entries.
    /// All tokens' (word_index, or_mask) pairs concatenated in token order.
    /// Improves cache locality vs separate Vec allocations per token.
    #[serde(skip)]
    pub(crate) internal_token_buf_flat: Box<[PackedInternalTokenBufMask]>,
    /// Current IBM3 loads can retain the runtime-native flat sparse-mask slab
    /// directly inside the owned artifact instead of copying ~0.5-1 MiB.
    #[serde(skip, default)]
    pub(crate) backed_internal_token_buf_flat: Option<BackedInternalTokenBufMasks>,
    /// Offsets into `internal_token_buf_flat` for each internal token.
    /// `internal_token_buf_flat[offsets[i]..offsets[i+1]]` gives token i's entries.
    /// Length = n_internal + 1 (sentinel at end).
    #[serde(skip)]
    pub(crate) internal_token_buf_offsets: Box<[u32]>,
    /// Pre-computed total cost (sum of entry counts) for all internal tokens.
    /// Used to avoid O(n_internal) cost analysis in the convert phase.
    #[serde(skip)]
    pub(crate) total_internal_buf_cost: usize,
    /// Indices of heavy tokens for fast iteration. Length == n_heavy_tokens.
    #[serde(skip)]
    pub(crate) heavy_token_indices: Vec<usize>,
    /// Total cost of all heavy tokens combined (n_heavy Ã— buf_len).
    #[serde(skip)]
    pub(crate) heavy_total_cost: usize,
    /// Average cost per light token: (total_cost - heavy_total) / n_light.
    /// Pre-multiplied by 256 for fixed-point arithmetic to avoid float.
    #[serde(skip)]
    pub(crate) light_avg_cost_x256: usize,
    /// Exact materialization cost per internal token, after heavy-token dense masks
    /// have been chosen.
    #[serde(skip)]
    pub(crate) internal_token_buf_op_costs: Vec<usize>,
    /// Exact materialization cost per 64-token internal word group.
    #[serde(skip)]
    pub(crate) word_group_buf_op_costs: Vec<usize>,
    /// Self-contained final internal-token -> original-token bitset materializer.
    #[serde(skip)]
    pub(crate) final_mask_mapping: FinalMaskMapping,
    /// Optional exact quotient of positive parser-state labels used by composed
    /// parser DWAs. Entry `s` is a synthetic fallback label for parser state
    /// `s`; `i32::MAX` means no component-local fallback. Concrete parser-state
    /// transitions always take precedence, followed by this label, then the
    /// ordinary global DEFAULT. Empty for ordinary non-composed constraints.
    #[serde(skip, default)]
    pub(crate) parser_state_domain_labels: Vec<i32>,
    /// Exact source expression for the globally erasable ignore terminal.
    ///
    /// Tokenizer source expressions are compile-time data and are normally
    /// omitted from artifacts. Retaining this one expression lets a loaded
    /// compiled constraint participate in later subgrammar composition without
    /// conservatively degrading an identical global ignore into scoped skips.
    #[serde(skip, default)]
    pub(crate) ignore_expr: Option<Expr>,
    /// Exact current-format artifact backing for an unchanged loaded
    /// constraint. Runtime cache rebuilds do not alter serialized semantics,
    /// so resave can return a single bulk copy instead of rediscovering and
    /// re-encoding the same canonical pools.
    #[serde(skip, default)]
    pub(crate) serialized_artifact_cache: Option<Arc<Vec<u8>>>,
    /// Current-format terminal source expressions can be retained as their
    /// canonical bincode payload instead of recursively rebuilding every Expr
    /// node during an ordinary static load. Composition materializes the list
    /// lazily through `retained_terminal_exprs` when it actually needs source
    /// language proofs.
    #[serde(skip, default)]
    pub(crate) deferred_terminal_exprs_blob: Option<DeferredTerminalExprBytes>,
    #[serde(skip, default)]
    pub(crate) deferred_terminal_exprs: OnceLock<Arc<[Expr]>>,
    /// Serialized composition-only metadata (reset-token rows, parser template
    /// cache, symbolic characterizations, and grammar summary). Current-format
    /// loads keep this cold section backed by the artifact and materialize it
    /// only if the constraint is later used as a composition component.
    #[serde(skip, default)]
    pub(crate) deferred_composition_metadata_blob: Option<DeferredCompositionMetadataBytes>,
    #[serde(skip, default)]
    pub(crate) composition_link_metadata_materialized: bool,
    /// Large current-format GLR rule vectors are composition metadata rather
    /// than runtime parser data. Keep their canonical payload undecoded during
    /// ordinary load; composition materializes it lazily through
    /// `retained_table_rules`.
    #[serde(skip, default)]
    pub(crate) deferred_table_rules_blob:
        Option<crate::compiler::glr::table::artifact_serde::DeferredRuleBytes>,
    #[serde(skip, default)]
    pub(crate) deferred_table_rules: OnceLock<Arc<[crate::grammar::flat::Rule]>>,
}

#[cfg(test)]
mod dynamic_mask_vocab_cache_boundary_tests;

/// Arc ownership is a runtime/lifecycle detail; keep the existing tokenizer
/// artifact bytes unchanged. A retained observer can pin this same allocation,
/// and all subsequent mutation must use copy-on-write on the Constraint side.
pub(crate) mod immutable_tokenizer_serde {
    use super::{Arc, Tokenizer};
    use serde::{Deserializer, Serializer};
    pub(crate) fn serialize<S: Serializer>(
        tokenizer: &Arc<Tokenizer>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        crate::automata::lexer::tokenizer::artifact_serde::serialize(tokenizer.as_ref(), serializer)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Arc<Tokenizer>, D::Error> {
        crate::automata::lexer::tokenizer::artifact_serde::deserialize(deserializer).map(Arc::new)
    }
}

#[cfg(test)]
mod immutable_tokenizer_tests {
    use super::*;
    use crate::automata::lexer::ast::{bytes, choice, plus};
    use crate::automata::lexer::compile::{build_regex_monolithic, build_regex_partitioned};
    use serde::{Deserialize, Serialize};
    #[derive(Serialize)]
    struct OwnedRef<'a>(
        #[serde(with = "crate::automata::lexer::tokenizer::artifact_serde")] &'a Tokenizer,
    );
    #[derive(Serialize, Deserialize)]
    struct Shared(#[serde(with = "super::immutable_tokenizer_serde")] Arc<Tokenizer>);
    #[test]
    fn shared_tokenizer_retains_exact_owned_wire_and_byte_behavior() {
        let expressions = vec![
            choice(vec![bytes(b"abc"), bytes(b"abd")]),
            plus(bytes(b" ")),
        ];
        for tokenizer in [
            build_regex_monolithic(&expressions).into_tokenizer(2, None),
            build_regex_partitioned(&expressions, &[0, 1]).into_tokenizer(2, None),
        ] {
            let owned_wire = bincode::serialize(&OwnedRef(&tokenizer)).unwrap();
            let shared = Shared(Arc::new(tokenizer));
            let wire = bincode::serialize(&shared).unwrap();
            assert_eq!(owned_wire, wire);
            let loaded: Shared = bincode::deserialize(&wire).unwrap();
            assert_eq!(bincode::serialize(&loaded).unwrap(), owned_wire);
            for q in 0..shared.0.num_states() {
                for b in 0..=255u8 {
                    assert_eq!(shared.0.step_all(&[q], b), loaded.0.step_all(&[q], b));
                }
            }
        }
    }
    #[test]
    fn pinned_tokenizer_is_unchanged_after_copy_on_write_mutation() {
        let expressions = vec![Expr::Epsilon, bytes(b"a")];
        let mut owner = Arc::new(build_regex_monolithic(&expressions).into_tokenizer(2, None));
        let pinned = Arc::clone(&owner);
        let original = bincode::serialize(&OwnedRef(pinned.as_ref())).unwrap();
        assert!(Arc::ptr_eq(&owner, &pinned));
        let drained = Arc::make_mut(&mut owner).isolate_start_state_and_drain_nullable_terminals();
        assert!(
            drained.contains(&0),
            "fixture must exercise a real nullable-state mutation"
        );
        assert!(!Arc::ptr_eq(&owner, &pinned));
        assert_eq!(
            bincode::serialize(&OwnedRef(pinned.as_ref())).unwrap(),
            original
        );
        assert_ne!(
            bincode::serialize(&OwnedRef(owner.as_ref())).unwrap(),
            original
        );
        assert!(
            !Arc::make_mut(&mut owner)
                .isolate_start_state_and_drain_nullable_terminals()
                .contains(&0)
        );
    }
    #[test]
    fn constraint_clone_shares_the_same_immutable_lexer() {
        let vocab = crate::Vocab::new(vec![(1, b"a".to_vec()), (2, b"b".to_vec())]);
        let source =
            Constraint::from_glrm_grammar(r#"start x; nt x ::= "a" | "b";"#, &vocab).unwrap();
        let copy = source.clone();
        assert!(Arc::ptr_eq(&source.tokenizer, &copy.tokenizer));
        assert_eq!(source.save(), copy.save());
    }
}

#[cfg(test)]
mod raw_terminal_radius_tests {
    use super::*;
    use crate::{Constraint, Grammar, Vocab};

    /// Exhaustively enumerate the finite test alphabet, independent of the
    /// product/shortest-distance algorithm being checked. No quotient proofs.
    fn reference_radius(
        tokenizer: &Tokenizer,
        terminal: TerminalID,
        source: u32,
        atoms: &[&[u8]],
        horizon: u32,
    ) -> u32 {
        let mut frontier = vec![source];
        for count in 1..=horizon {
            let mut next = Vec::new();
            for state in frontier {
                for atom in atoms {
                    let mut end = state;
                    for &byte in *atom {
                        let Some(target) = tokenizer.step(end, byte) else {
                            return count - 1;
                        };
                        if target >= tokenizer.num_states()
                            || (!tokenizer
                                .possible_future_terminals(target)
                                .contains(terminal as usize)
                                && !tokenizer
                                    .matched_terminals_slice(target)
                                    .contains(&terminal))
                        {
                            return count - 1;
                        }
                        end = target;
                    }
                    next.push(end);
                }
            }
            frontier = next;
        }
        horizon
    }

    #[test]
    fn raw_terminal_radius_matches_exhaustive_ascii_and_multibyte_words() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, "é".as_bytes().to_vec()),
        ]);
        let fixtures: [(&str, &str, Vec<&[u8]>); 3] = [
            (r"[ab]{1,4}", r"[ab]+", vec![b"a", b"b"]),
            (r"a{1,4}", r"[ab]+", vec![b"a", b"b"]),
            (
                r"(?:a|é){1,4}",
                r"(?:a|\xC3\xA9)+",
                vec![b"a", "é".as_bytes()],
            ),
        ];
        let mut positive = 0usize;
        let mut checked = 0usize;
        for (pattern, slice_pattern, atoms) in fixtures {
            let grammar = format!("glrm 1; start s; t WORD = /{pattern}/; nt s = WORD;");
            let constraint = Constraint::compile(Grammar::glrm(&grammar), &vocab).unwrap();
            let terminal = constraint
                .terminal_display_names
                .iter()
                .position(|name| name == "WORD")
                .unwrap() as TerminalID;
            let slice = VocabPartitionDfa::compile_byte_regex("test-atoms", slice_pattern).unwrap();
            for source in 0..constraint.tokenizer.num_states() {
                if constraint.tokenizer.state_has_epsilon_transitions(source)
                    || constraint.tokenizer.state_is_virtual_runtime(source)
                {
                    continue;
                }
                let expected = reference_radius(&constraint.tokenizer, terminal, source, &atoms, 6);
                let actual = constraint
                    .dynamic_mask_vocab
                    .raw_terminal_slice_repeat_radius(
                        &constraint.tokenizer,
                        terminal,
                        source,
                        901,
                        &slice,
                        6,
                        65_536,
                    );
                assert_eq!(actual, Some(expected), "pattern={pattern} source={source}");
                checked += 1;
                positive += usize::from(expected > 0);
                // Budget exhaustion never creates a certificate, and its key
                // cannot poison a subsequent larger-budget query.
                assert_eq!(
                    constraint
                        .dynamic_mask_vocab
                        .raw_terminal_slice_repeat_radius(
                            &constraint.tokenizer,
                            terminal,
                            source,
                            901,
                            &slice,
                            6,
                            0,
                        ),
                    Some(0)
                );
                assert_eq!(
                    constraint
                        .dynamic_mask_vocab
                        .raw_terminal_slice_repeat_radius(
                            &constraint.tokenizer,
                            terminal,
                            source,
                            901,
                            &slice,
                            6,
                            65_536,
                        ),
                    actual
                );
            }
            let empty = VocabPartitionDfa::compile_byte_regex("nullable", r"a*").unwrap();
            assert_eq!(
                constraint
                    .dynamic_mask_vocab
                    .raw_terminal_slice_repeat_radius(
                        &constraint.tokenizer,
                        terminal,
                        0,
                        902,
                        &empty,
                        6,
                        65_536,
                    ),
                None
            );
            assert_eq!(
                constraint
                    .dynamic_mask_vocab
                    .raw_terminal_slice_repeat_radius(
                        &constraint.tokenizer,
                        terminal,
                        u32::MAX,
                        901,
                        &slice,
                        6,
                        65_536,
                    ),
                None
            );
        }
        assert!(
            checked > 4 && positive > 0,
            "must exercise real positive and negative certificates"
        );
    }
}
