#[cfg(any(test, feature = "internal-api"))]
pub(crate) mod boundary_cpu_profile;
mod artifact;
mod commit;
mod constraint;
pub(crate) mod parser_backend;
mod dynamic_mask;
pub(crate) use dynamic_mask::dynamic_mask_profile_enabled;
mod finalize;
mod mask;
mod mask_cache_payload;
mod mask_cache_values;
pub(crate) mod mask_mapping;
pub(crate) mod serde;
pub(crate) use serde::compact_large_non_dwa_weight_runtime;
mod state;
mod token_space;
pub(crate) mod static_observation;
pub(crate) use glrmask_artifact::CommitTemplateDfas;
#[allow(unused_imports)]
pub(crate) use artifact::{
    dynamic_mask_vocab_layout_class, BoundaryTerminalNwa, BoundaryTerminalNwaNode,
    BoundaryTerminalNwaTransition, BoundaryTerminalTrieNode, BoundaryTrigger,
    BoundaryCandidateFingerprint, BoundaryCandidateSummary, OriginalTokenSet,
    SummaryPrecision, SummaryUnavailable,
    CompositionGrammarSummary, ConstraintRuntimeBackend, DynamicMaskTrie, DynamicMaskVocab,
    DynamicMaskVocabArtifact, DeferredTerminalExprBytes, FastCommitTemplateDfas,
    FastTokenizerTransitions, LateGrammarSlot,
    PreparedMasterProofArtifact,
    SegmentedBoundaryParser, SegmentedBoundaryShard, SegmentedBoundaryShardBackend,
    SegmentedBoundaryTerminalTrie, SegmentedParserComponent, SegmentedParserComponentTables,
    SegmentedParserLink, SpecialTokenTerminal, StaticDynamicOverlayMetadata,
    dynamic_mask_llg_master_is_whitespace,
    dynamic_mask_llg_master_layout_class,
    dynamic_mask_llg_master_safe_chars,
    DYNAMIC_MASK_LLG_MASTER_CACHE_ID,};
pub(crate) use artifact::token_bytes_artifact_serde::PackedTokenBytes;
#[allow(unused_imports)]
pub use crate::compiler::glr::parser::{AdvanceTrace, AdvanceTraceStep};
#[allow(unused_imports)]
pub use commit::profile::{CommitProfile, GssProfileSummary, PerAdvanceEntry};
pub use artifact::BoundaryTriggerDetail;
pub use constraint::Constraint;
pub(crate) use constraint::{InternalTokenMaskPrebuild, RuntimeWeightRef, TokenMaskCachePrebuild};
#[allow(unused_imports)]
pub use mask::profile::MaskProfile;
pub use state::ConstraintState;

pub(crate) fn initialize_hot_path_config() {
    mask::initialize_runtime_config();
    commit::initialize_runtime_config();
}
