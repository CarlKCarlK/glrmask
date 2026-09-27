//! Boundary-candidate summaries and component grammar metadata.

use crate::automata::weighted::dwa::DWA;
use crate::ds::bitset::BitSet;
use std::sync::Arc;
/// Compile-time detail requested for the reusable dynamic-boundary trigger.
///
/// This setting is orthogonal to whether ordinary component masking is static
/// or dynamic. `None` is the zero-cost default; `Tokens` adds only a
/// parser-state-independent candidate set; `Exact` builds the full
/// GSS-sensitive trigger Parser DWA when the component supports it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BoundaryTriggerDetail {
    #[default]
    None,
    Tokens,
    Exact,
}

/// Optional reusable hint for dynamic composition boundaries. This is an
/// accelerator only: `None` means the composition runtime must conservatively
/// assume that any model token may cross a component boundary.
#[derive(Debug, Clone, Default)]
pub(crate) enum BoundaryTrigger {
    #[default]
    None,
    /// Conservative original model-token IDs that may contain the first
    /// internal boundary crossing. Parser-state independent.
    Tokens(Arc<[u32]>),
    /// Exact component-local parser DWA. Its coordinate is deliberately
    /// independent of the ordinary parser-DWA quotient: parser labels are
    /// local LR-state IDs, weight TSIDs are raw local tokenizer-state IDs, and
    /// weight token IDs are original/model token IDs. Proper-prefix boundary
    /// behavior is a stronger observation than the ordinary whole-token mask
    /// language, so reusing the normal TSID/internal-token quotient would need
    /// a separate equivalence proof.
    Exact(Arc<DWA>),
}

impl BoundaryTrigger {
    pub(crate) fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub(crate) fn token_summary(&self) -> Option<&[u32]> {
        match self {
            Self::Tokens(tokens) => Some(tokens),
            Self::None | Self::Exact(_) => None,
        }
    }
}

/// Stable identity of the inputs that make a boundary-candidate summary
/// reusable.  The four pieces are intentionally separate so diagnostics can
/// distinguish stale component semantics, interface changes, and vocabulary
/// mismatches instead of treating every miss as an opaque hash failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BoundaryCandidateFingerprint {
    pub(crate) algorithm_version: u16,
    pub(crate) component_semantics: [u8; 32],
    pub(crate) public_interface: [u8; 32],
    pub(crate) vocabulary: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SummaryPrecision {
    RegularUpperBound,
    ContextRefinedUpperBound,
    BudgetWidenedUpperBound,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SummaryUnavailable {
    Disabled,
    Deferred,
    LegacyArtifact,
    MissingGrammarMetadata,
    UnsupportedFiniteLexer,
    InvalidatedBinding,
    FingerprintMismatch,
    MalformedMetadata,
}

/// Original/model-token ID set used by boundary summaries.  Sparse is the
/// canonical wire form; Dense is useful when a component legitimately retains
/// a large irregular set.  `AllByteTokensAtLeastTwo` is the safe top element
/// for the proper-prefix question: one-byte tokens can never contain a
/// non-empty proper byte prefix.
#[derive(Debug, Clone)]
pub(crate) enum OriginalTokenSet {
    Empty,
    Sparse(Arc<[u32]>),
    Dense(Arc<BitSet>),
    AllByteTokensAtLeastTwo,
}

impl OriginalTokenSet {
    pub(crate) fn from_sorted_unique(ids: Vec<u32>, max_token_id: u32) -> Self {
        if ids.is_empty() {
            return Self::Empty;
        }
        // Dense becomes cheaper only when the set is genuinely dense. Keep a
        // deliberately conservative crossover because Sparse is also the
        // canonical persisted representation.
        let dense_words = (max_token_id as usize + 64) / 64;
        if ids.len() > dense_words.saturating_mul(3) {
            let mut bits = BitSet::new(max_token_id as usize + 1);
            for id in ids {
                bits.set(id as usize);
            }
            Self::Dense(Arc::new(bits))
        } else {
            Self::Sparse(Arc::from(ids.into_boxed_slice()))
        }
    }

    pub(crate) fn contains(&self, token_id: u32, bytes: &[u8]) -> bool {
        match self {
            Self::Empty => false,
            Self::Sparse(ids) => ids.binary_search(&token_id).is_ok(),
            Self::Dense(bits) => (token_id as usize) < bits.len() && bits.get(token_id as usize),
            Self::AllByteTokensAtLeastTwo => bytes.len() >= 2,
        }
    }

    pub(crate) fn canonical_ids<'a>(
        &'a self,
        tokens: impl Iterator<Item = (u32, &'a [u8])>,
    ) -> Vec<u32> {
        match self {
            Self::Empty => Vec::new(),
            Self::Sparse(ids) => ids.to_vec(),
            Self::Dense(bits) => bits.iter_ones().map(|id| id as u32).collect(),
            Self::AllByteTokensAtLeastTwo => tokens
                .filter_map(|(id, bytes)| (bytes.len() >= 2).then_some(id))
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum BoundaryCandidateSummary {
    Unknown {
        reason: SummaryUnavailable,
    },
    Known {
        fingerprint: BoundaryCandidateFingerprint,
        tokens: OriginalTokenSet,
        precision: SummaryPrecision,
    },
}

impl Default for BoundaryCandidateSummary {
    fn default() -> Self {
        Self::Unknown {
            reason: SummaryUnavailable::Deferred,
        }
    }
}

impl BoundaryCandidateSummary {
    pub(crate) fn known_tokens_for(
        &self,
        fingerprint: &BoundaryCandidateFingerprint,
    ) -> Option<&OriginalTokenSet> {
        match self {
            Self::Known {
                fingerprint: actual,
                tokens,
                ..
            } if actual == fingerprint => Some(tokens),
            _ => None,
        }
    }

    pub(crate) fn is_known(&self) -> bool {
        matches!(self, Self::Known { .. })
    }
}

/// Small composition-time grammar summary retained with a compiled component.
///
/// For a nonnullable child, substituting the child's language for a parent
/// placeholder needs only:
/// * terminal adjacency (`allowed_follows`),
/// * FIRST/LAST of the component root, and
/// * root nullability.
///
/// Keeping this summary in the outer artifact envelope lets the linker compose
/// grammar legality algebraically instead of rebuilding FIRST/FOLLOW over the
/// fully merged rule graph.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(crate) struct CompositionGrammarSummary {
    pub(crate) allowed_follows: Vec<BitSet>,
    pub(crate) root_first: BitSet,
    pub(crate) root_last: BitSet,
    pub(crate) root_nullable: bool,
}
