// PRE-RELEASE COMPATIBILITY POLICY
// ================================
// GLRMask is still pre-release. Historical DynamicConstraint/transfer wire
// layouts are NOT a compatibility contract and must not constrain the current
// architecture. Prefer the cleanest exact current representation; legacy
// readers below are best-effort development conveniences only and may be
// deleted or broken whenever preserving them would complicate correctness,
// performance, or the wire format. Do not add migration machinery unless
// compatibility becomes an explicit product requirement.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::{
    TerminalProjectedQuotient, Tokenizer, VirtualTokenizerRuntimeMetadata,
};
use crate::automata::regex::Expr;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::constraint_possible_matches::ConstraintPossibleMatchesComputation;
use crate::grammar::flat::{DirectRegularAutomaton, GrammarDef, Symbol, Terminal, TerminalID};
use crate::Vocab;

use crate::runtime::{
    dynamic_mask_profile_enabled, CommitProfile, Constraint, ConstraintState, DynamicMaskVocab,
    SpecialTokenTerminal,
};

const DYNAMIC_CONSTRAINT_MAGIC: [u8; 8] = *b"GLRDYN\0\0";
const LEGACY_DYNAMIC_CONSTRAINT_VERSION_V12: u16 = 12;
const LEGACY_DYNAMIC_CONSTRAINT_VERSION_V13: u16 = 13;
// v14 is the glrmask-main composition artifact that first persisted late
// grammar slots. The residual-runtime branch independently used v14 during
// development, but that layout was never published; do not conflate the two.
const LEGACY_DYNAMIC_CONSTRAINT_VERSION_V14_MAIN: u16 = 14;
// v15 is the residual-runtime artifact produced by the completed feature
// branch before integration with the composition work.
const LEGACY_DYNAMIC_CONSTRAINT_VERSION_V15_RESIDUAL: u16 = 15;
// v16 combines the v14 composition metadata and v15 residual-runtime metadata.
const LEGACY_DYNAMIC_CONSTRAINT_VERSION_V16: u16 = 16;
// v17 additionally persists exact terminal-observation quotient certificates.
const LEGACY_DYNAMIC_CONSTRAINT_VERSION_V17: u16 = 17;
// v18 additionally persists the initialized dynamic-mask vocabulary trie so
// self-contained loads do not rebuild the full vocabulary index from token bytes.
const LEGACY_DYNAMIC_CONSTRAINT_VERSION_V18: u16 = 18;
// v20 additionally persists exact terminal-projected lexer quotients used by
// dynamic-mask subtree certificates. As documented above, the pre-release
// wire is intentionally allowed to replace the previous current layout.
const DYNAMIC_CONSTRAINT_VERSION: u16 = 20;
// Template backends use the same table-free Constraint body as static mode.
// This avoids a second parser codec and cannot restore an LR fallback.
const TEMPLATE_DYNAMIC_CONSTRAINT_VERSION: u16 = 21;
const DYNAMIC_CONSTRAINT_HEADER_LEN: usize = DYNAMIC_CONSTRAINT_MAGIC.len() + 2 + 8;
const DYNAMIC_TRANSFER_MAGIC: [u8; 8] = *b"GLRDXF\0\0";
const DYNAMIC_TRANSFER_VERSION_V1: u16 = 1;
const DYNAMIC_TRANSFER_VERSION_V2: u16 = 2;
const DYNAMIC_TRANSFER_VERSION_V3: u16 = 3;
const DYNAMIC_TRANSFER_VERSION_V4: u16 = 4;
const LEGACY_DYNAMIC_TRANSFER_VERSION_V5: u16 = 5;
// v6 is the integrated residual-runtime transfer layout.
const LEGACY_DYNAMIC_TRANSFER_VERSION_V6: u16 = 6;
// v7 additionally carries exact terminal-observation quotient certificates.
const LEGACY_DYNAMIC_TRANSFER_VERSION_V7: u16 = 7;
// v9 also carries exact terminal-projected lexer quotients so a compile worker
// and runtime process never rebuild them independently.
const LEGACY_DYNAMIC_TRANSFER_VERSION_V9: u16 = 9;
// v10 stores the GLR table through the shared compact sectioned table codec.
// This avoids generic serde traversal of thousands of parser rows/rules and
// allows source rules to remain deferred after load until composition needs them.
const LEGACY_DYNAMIC_TRANSFER_VERSION_V10: u16 = 10;
// v11 moves the large GLR-table, tokenizer, terminal-expression, and recursive
// artifact byte blobs out of the bincode metadata object into raw top-level
// sections. The loader can therefore decode table/tokenizer views directly
// against one owned backing allocation instead of first copying each blob into
// an intermediate Vec.
const LEGACY_DYNAMIC_TRANSFER_VERSION_V11: u16 = 11;
const DYNAMIC_TRANSFER_V11_PAYLOAD_HEADER_LEN: usize = 8;
const DYNAMIC_TRANSFER_V11_ALT_DESCRIPTOR_LEN: usize = 5 * 8;
// v12 carries the compiled SRM3 virtual-residual mask projection wire format as
// a dedicated 6th payload section. On load, the projection is restored directly
// onto the dynamic mask vocabulary, eliminating projection reconstruction on first mask.
const LEGACY_DYNAMIC_TRANSFER_VERSION_V12: u16 = 12;
// v13 keeps the same six-section framing and wraps the v12 metadata with exact
// compact master-slice proof/coverage/radius rows. This replaces giant
// projected-terminal quotient payloads for the dynamic master proof fast path.
const DYNAMIC_TRANSFER_VERSION: u16 = 13;
const TEMPLATE_DYNAMIC_TRANSFER_VERSION: u16 = 14;
const DYNAMIC_TRANSFER_V12_PAYLOAD_HEADER_LEN: usize = 8;
const DYNAMIC_TRANSFER_V12_ALT_DESCRIPTOR_LEN: usize = 6 * 8;

mod compressed_terminal_exprs_serde {
    use super::Expr;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn encode(exprs: &[Expr]) -> Result<Vec<u8>, String> {
        let raw = bincode::serialize(exprs).map_err(|err| err.to_string())?;
        zstd::bulk::compress(&raw, 1).map_err(|err| err.to_string())
    }

    pub fn decode(compressed: &[u8]) -> Result<Vec<Expr>, String> {
        let raw = zstd::stream::decode_all(compressed).map_err(|err| err.to_string())?;
        bincode::deserialize(&raw).map_err(|err| err.to_string())
    }

    pub fn serialize<S>(exprs: &Option<Vec<Expr>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error as _;
        let compressed = match exprs {
            None => None,
            Some(exprs) => Some(encode(exprs).map_err(S::Error::custom)?),
        };
        compressed.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<Expr>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error as _;
        let compressed = Option::<Vec<u8>>::deserialize(deserializer)?;
        let Some(compressed) = compressed else {
            return Ok(None);
        };
        decode(&compressed).map(Some).map_err(D::Error::custom)
    }
}

#[derive(Debug)]
struct CompactTransferTokenizer(Tokenizer);

impl serde::Serialize for CompactTransferTokenizer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        crate::automata::lexer::tokenizer::compact_artifact_serde::serialize(&self.0, serializer)
    }
}

/// Runtime-native tokenizer wire used by the current worker transfer. The
/// older dynamic compact serde reconstructs every DFA row through generic
/// serde on load. TKF3 instead restores the packed runtime transition and
/// metadata representation already used by current Constraint artifacts.
#[derive(Debug)]
struct FastTransferTokenizer(Tokenizer);

impl serde::Serialize for FastTransferTokenizer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        crate::automata::lexer::tokenizer::artifact_serde::to_fast_bytes_with_packed_metadata(
            &self.0,
        )
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for FastTransferTokenizer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        crate::automata::lexer::tokenizer::artifact_serde::from_fast_bytes(&bytes)
            .map(Self)
            .map_err(D::Error::custom)
    }
}

impl<'de> serde::Deserialize<'de> for CompactTransferTokenizer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        crate::automata::lexer::tokenizer::compact_artifact_serde::deserialize(deserializer)
            .map(Self)
    }
}

#[derive(Debug)]
struct CompactTransferTable {
    table: GLRTable,
    deferred_rules: Option<crate::compiler::glr::table::artifact_serde::DeferredRuleBytes>,
}

impl serde::Serialize for CompactTransferTable {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error as _;
        let materialized;
        let table = if let Some(rules) = self.deferred_rules.as_ref() {
            let decoded = bincode::deserialize::<Vec<crate::grammar::flat::Rule>>(rules.as_slice())
                .map_err(S::Error::custom)?;
            if decoded.len() != self.table.num_rules as usize
                || decoded.first() != self.table.rules.first()
            {
                return Err(S::Error::custom("invalid deferred GLR rules in dynamic transfer"));
            }
            materialized = {
                let mut table = self.table.clone();
                table.rules = decoded;
                table
            };
            &materialized
        } else {
            &self.table
        };
        crate::compiler::glr::table::artifact_serde::to_compact_bytes(table).serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for CompactTransferTable {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        let decoded = crate::compiler::glr::table::artifact_serde::from_compact_bytes_deferred(&bytes)
            .map_err(D::Error::custom)?;
        Ok(Self {
            table: decoded.table,
            deferred_rules: decoded.deferred_rules,
        })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV1<Table = GLRTable> {
    table: Table,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    #[serde(default)]
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    #[serde(default)]
    ignore_expr: Option<Expr>,
    #[serde(default, with = "compressed_terminal_exprs_serde")]
    terminal_exprs: Option<Vec<Expr>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    eos_token_id: Option<u32>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV2<Table = GLRTable> {
    v1: DynamicConstraintPayloadV1<Table>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV2 {
    v1: LegacyDynamicConstraintPayloadV1,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV10V1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    #[serde(default)]
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV10V2 {
    v1: LegacyDynamicConstraintPayloadV10V1,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV10V3 {
    alternatives: Vec<LegacyDynamicConstraintPayloadV10V2>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV9V1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV9 {
    v1: LegacyDynamicConstraintPayloadV9V1,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV3 {
    alternatives: Vec<DynamicConstraintPayloadV2>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV14MainAlternative {
    constraint: DynamicConstraintPayloadV2,
    late_grammar_slots: Vec<crate::runtime::LateGrammarSlot>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV14Main {
    alternatives: Vec<LegacyDynamicConstraintPayloadV14MainAlternative>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV15ResidualAlternative {
    v2: DynamicConstraintPayloadV2,
    virtual_runtimes: Vec<VirtualTokenizerRuntimeMetadata>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV15Residual {
    alternatives: Vec<LegacyDynamicConstraintPayloadV15ResidualAlternative>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV4Alternative {
    v2: DynamicConstraintPayloadV2,
    late_grammar_slots: Vec<crate::runtime::LateGrammarSlot>,
    virtual_runtimes: Vec<VirtualTokenizerRuntimeMetadata>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV4 {
    alternatives: Vec<DynamicConstraintPayloadV4Alternative>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV5Alternative {
    base: DynamicConstraintPayloadV4Alternative,
    terminal_observation_classes: Vec<(TerminalID, Vec<u32>)>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV5 {
    alternatives: Vec<DynamicConstraintPayloadV5Alternative>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV6 {
    alternatives: Vec<DynamicConstraintPayloadV5Alternative>,
    /// Vocabulary-only runtime index shared by every union alternative.
    dynamic_mask_vocab: Option<crate::runtime::DynamicMaskVocabArtifact>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum DynamicBoundaryTriggerWire {
    None,
    Tokens(Vec<u32>),
    Exact(DWA),
}

impl DynamicBoundaryTriggerWire {
    fn from_trigger(trigger: &crate::runtime::BoundaryTrigger) -> Self {
        match trigger {
            crate::runtime::BoundaryTrigger::None => Self::None,
            crate::runtime::BoundaryTrigger::Tokens(tokens) => Self::Tokens(tokens.to_vec()),
            crate::runtime::BoundaryTrigger::Exact(dwa) => Self::Exact((**dwa).clone()),
        }
    }

    fn into_trigger(self) -> crate::runtime::BoundaryTrigger {
        match self {
            Self::None => crate::runtime::BoundaryTrigger::None,
            Self::Tokens(mut tokens) => {
                tokens.sort_unstable();
                tokens.dedup();
                crate::runtime::BoundaryTrigger::Tokens(Arc::from(tokens.into_boxed_slice()))
            }
            Self::Exact(dwa) => crate::runtime::BoundaryTrigger::Exact(Arc::new(dwa)),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV7Alternative {
    base: DynamicConstraintPayloadV5Alternative,
    projected_terminal_quotients: Vec<(TerminalID, TerminalProjectedQuotient)>,
    boundary_trigger: DynamicBoundaryTriggerWire,
    /// Present only for recursively composed alternatives. The ordinary
    /// dynamic fields above remain the compact standalone representation; the
    /// universal Constraint artifact is authoritative for the recursive
    /// provider/component tree and its lazy compiler views.
    recursive_constraint_artifact: Option<Vec<u8>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV7 {
    alternatives: Vec<DynamicConstraintPayloadV7Alternative>,
    /// Vocabulary-only runtime index shared by every union alternative.
    dynamic_mask_vocab: Option<crate::runtime::DynamicMaskVocabArtifact>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV11V1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    #[serde(default)]
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV11V2 {
    v1: LegacyDynamicConstraintPayloadV11V1,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV11V3 {
    alternatives: Vec<LegacyDynamicConstraintPayloadV11V2>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV12V1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    #[serde(default)]
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    #[serde(default)]
    ignore_expr: Option<Expr>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV12V2 {
    v1: LegacyDynamicConstraintPayloadV12V1,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV12V3 {
    alternatives: Vec<LegacyDynamicConstraintPayloadV12V2>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferAlternativeV1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    #[serde(default)]
    ignore_expr: Option<Expr>,
    #[serde(default, with = "compressed_terminal_exprs_serde")]
    terminal_exprs: Option<Vec<Expr>>,
    mask_tokenizer: Option<CompactTransferTokenizer>,
    full_to_mask_state: Vec<u32>,

}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferPayloadV1 {
    alternatives: Vec<DynamicConstraintTransferAlternativeV1>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferAlternativeV2 {
    v1: DynamicConstraintTransferAlternativeV1,
    virtual_runtimes: Vec<VirtualTokenizerRuntimeMetadata>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferPayloadV2 {
    alternatives: Vec<DynamicConstraintTransferAlternativeV2>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferAlternativeV3 {
    base: DynamicConstraintTransferAlternativeV2,
    terminal_observation_classes: Vec<(TerminalID, Vec<u32>)>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferPayloadV3 {
    alternatives: Vec<DynamicConstraintTransferAlternativeV3>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferAlternativeV4 {
    base: DynamicConstraintTransferAlternativeV3,
    projected_terminal_quotients: Vec<(TerminalID, TerminalProjectedQuotient)>,
    boundary_trigger: DynamicBoundaryTriggerWire,
    recursive_constraint_artifact: Option<Vec<u8>>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferPayloadV4 {
    alternatives: Vec<DynamicConstraintTransferAlternativeV4>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferAlternativeV5 {
    table: CompactTransferTable,
    terminal_display_names: Vec<String>,
    tokenizer: FastTransferTokenizer,
    ignore_terminal: Option<TerminalID>,
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    #[serde(default)]
    ignore_expr: Option<Expr>,
    /// Opaque zstd-compressed bincode terminal-expression section. Keeping the
    /// bytes opaque is important: ordinary execution never needs source
    /// expression trees, while later composition can materialize them lazily.
    terminal_exprs_compressed: Option<Vec<u8>>,
    mask_tokenizer: Option<CompactTransferTokenizer>,
    full_to_mask_state: Vec<u32>,
    virtual_runtimes: Vec<VirtualTokenizerRuntimeMetadata>,
    terminal_observation_classes: Vec<(TerminalID, Vec<u32>)>,
    projected_terminal_quotients: Vec<(TerminalID, TerminalProjectedQuotient)>,
    boundary_trigger: DynamicBoundaryTriggerWire,
    recursive_constraint_artifact: Option<Vec<u8>>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferPayloadV5 {
    alternatives: Vec<DynamicConstraintTransferAlternativeV5>,
}

/// Small structured metadata for one v11 transfer alternative. Large runtime
/// byte sections deliberately live outside this bincode object.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferMetadataV11 {
    terminal_display_names: Vec<String>,
    ignore_terminal: Option<TerminalID>,
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    ignore_expr: Option<Expr>,
    mask_tokenizer: Option<CompactTransferTokenizer>,
    full_to_mask_state: Vec<u32>,
    virtual_runtimes: Vec<VirtualTokenizerRuntimeMetadata>,
    #[serde(default)]
    residual_runtime_oracles: Vec<(TerminalID, Vec<u8>)>,
    terminal_observation_classes: Vec<(TerminalID, Vec<u32>)>,
    projected_terminal_quotients: Vec<(TerminalID, TerminalProjectedQuotient)>,
    /// Distinguish "not prepared yet" from "prepared and proved that no
    /// quotient is useful". An empty quotient vector alone cannot encode that
    /// distinction now that dynamic runtime proof artifacts are materialized
    /// lazily on first mask request.
    projected_terminal_quotients_prepared: bool,
    boundary_trigger: DynamicBoundaryTriggerWire,
    /// Present only for the opt-in grammar-equivalence dynamic vocabulary.
    /// Ordinary transfer artifacts continue to reconstruct the pure model-vocab
    /// trie from the caller-supplied Vocab.
    #[serde(default)]
    grammar_quotient_vocab: Option<crate::runtime::DynamicMaskVocabArtifact>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferMetadataV13 {
    base: DynamicConstraintTransferMetadataV11,
    prepared_master_proofs: crate::runtime::PreparedMasterProofArtifact,
    #[serde(default)]
    virtual_residual_master_slice_artifacts:
        Vec<crate::automata::lexer::tokenizer::VirtualResidualMasterSliceArtifact>,
}

struct DynamicConstraintTransferSectionsV11 {
    table: Vec<u8>,
    tokenizer: Vec<u8>,
    terminal_exprs_compressed: Vec<u8>,
    recursive_constraint_artifact: Vec<u8>,
    metadata: Vec<u8>,
}

struct DynamicConstraintTransferSectionsV12 {
    table: Vec<u8>,
    tokenizer: Vec<u8>,
    terminal_exprs_compressed: Vec<u8>,
    recursive_constraint_artifact: Vec<u8>,
    metadata: Vec<u8>,
    virtual_residual_wire: Vec<u8>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintTransferAlternativeV3 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    #[serde(default)]
    ignore_expr: Option<Expr>,
    mask_tokenizer: Option<CompactTransferTokenizer>,
    full_to_mask_state: Vec<u32>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintTransferPayloadV3 {
    alternatives: Vec<LegacyDynamicConstraintTransferAlternativeV3>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintTransferAlternativeV2 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    #[serde(default)]
    ignore_expr: Option<Expr>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintTransferPayloadV2 {
    alternatives: Vec<LegacyDynamicConstraintTransferAlternativeV2>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintTransferAlternativeV1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintTransferPayloadV1 {
    alternatives: Vec<LegacyDynamicConstraintTransferAlternativeV1>,
}

/// A constraint optimized for low compilation latency.
///
/// Unlike [`Constraint`], this omits terminal-DWA, possible-match, parser-DWA,
/// token-remapping, and dense-mask compilation. It produces the same masks as
/// [`Constraint`] but performs more work during mask generation.
#[derive(Debug, Clone)]
pub struct DynamicConstraint {
    pub(crate) inner: Constraint,
    alternatives: Vec<Constraint>,
    // Retained only in memory so a freshly compiled dynamic artifact can be
    // materialized as an exact static composer input without bloating its
    // serialized representation. Entries align with inner + alternatives.
    composition_grammars: Vec<Option<GrammarDef>>,
    /// Canonical current external-vocabulary artifact prepared as part of
    /// constraint finalization (or retained verbatim by load).  `save()` is a
    /// persistence operation, not a second compilation/finalization pass.
    external_vocab_artifact_cache: Option<Arc<Vec<u8>>>,
}

impl DynamicConstraint {
    pub(crate) fn from_parts(
        table: GLRTable,
        terminal_display_names: Vec<String>,
        tokenizer: Tokenizer,
        direct_regular_automaton: Option<DirectRegularAutomaton>,
        ignore_terminal: Option<TerminalID>,
        special_token_terminals: Vec<SpecialTokenTerminal>,
        vocab: &Vocab,
    ) -> Self {
        let dynamic_mask_vocab =
            crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(vocab);
        Self::from_parts_with_dynamic_vocab(
            table,
            terminal_display_names,
            tokenizer,
            direct_regular_automaton,
            ignore_terminal,
            special_token_terminals,
            vocab,
            dynamic_mask_vocab,
        )
    }

    pub(crate) fn from_parts_with_dynamic_vocab(
        table: GLRTable,
        terminal_display_names: Vec<String>,
        tokenizer: Tokenizer,
        direct_regular_automaton: Option<DirectRegularAutomaton>,
        ignore_terminal: Option<TerminalID>,
        special_token_terminals: Vec<SpecialTokenTerminal>,
        vocab: &Vocab,
        dynamic_mask_vocab: DynamicMaskVocab,
    ) -> Self {
        let ignore_expr = ignore_terminal
            .and_then(|terminal| tokenizer.terminal_expr(terminal).cloned());
        let terminal_exprs = tokenizer.terminal_exprs().map(ToOwned::to_owned);
        Self::from_payload_v2_with_dynamic_vocab(
            DynamicConstraintPayloadV2 {
                v1: DynamicConstraintPayloadV1 {
                    table,
                    terminal_display_names,
                    tokenizer,
                    ignore_terminal,
                    direct_regular_automaton,
                    token_bytes: vocab.entries_arc(),
                    ignore_expr,
                    terminal_exprs,
                },
                special_token_terminals,
            },
            dynamic_mask_vocab,
        )
    }

    pub(crate) fn from_parts_with_dynamic_vocab_unfinalized(
        table: GLRTable,
        terminal_display_names: Vec<String>,
        tokenizer: Tokenizer,
        direct_regular_automaton: Option<DirectRegularAutomaton>,
        ignore_terminal: Option<TerminalID>,
        special_token_terminals: Vec<SpecialTokenTerminal>,
        vocab: &Vocab,
        dynamic_mask_vocab: DynamicMaskVocab,
    ) -> Self {
        let ignore_expr = ignore_terminal
            .and_then(|terminal| tokenizer.terminal_expr(terminal).cloned());
        let terminal_exprs = tokenizer.terminal_exprs().map(ToOwned::to_owned);
        let payload = DynamicConstraintPayloadV2 {
            v1: DynamicConstraintPayloadV1 {
                table,
                terminal_display_names,
                tokenizer,
                ignore_terminal,
                direct_regular_automaton,
                token_bytes: vocab.entries_arc(),
                ignore_expr,
                terminal_exprs,
            },
            special_token_terminals,
        };
        Self {
            inner: Self::constraint_from_payload_v2_with_dynamic_vocab(
                payload,
                dynamic_mask_vocab,
            ),
            alternatives: Vec::new(),
            composition_grammars: vec![None],
            external_vocab_artifact_cache: None,
        }
    }

    pub(crate) fn from_parts_with_possible_matches(
        table: GLRTable,
        terminal_display_names: Vec<String>,
        tokenizer: Tokenizer,
        ignore_terminal: Option<TerminalID>,
        special_token_terminals: Vec<SpecialTokenTerminal>,
        vocab: &Vocab,
        computation: ConstraintPossibleMatchesComputation,
    ) -> Self {
        let ConstraintPossibleMatchesComputation {
            mapped_possible_matches,
            runtime_dynamic_vocab,
            complete,
            profile: _,
        } = computation;
        let (possible_matches, mut id_map) = mapped_possible_matches.into_parts();
        id_map.materialize_deferred_vocab_singletons();
        let ignore_expr = ignore_terminal
            .and_then(|terminal| tokenizer.terminal_expr(terminal).cloned());
        let terminal_exprs = tokenizer.terminal_exprs().map(ToOwned::to_owned);

        let mut result = Self::from_payload_v2_with_dynamic_vocab(
            DynamicConstraintPayloadV2 {
                v1: DynamicConstraintPayloadV1 {
                    table,
                    terminal_display_names,
                    tokenizer,
                    ignore_terminal,
                    direct_regular_automaton: None,
                    token_bytes: vocab.entries_arc(),
                    ignore_expr,
                    terminal_exprs,
                },
                special_token_terminals,
            },
            runtime_dynamic_vocab.vocab,
        );
        result.inner.possible_matches = possible_matches;
        result.inner.possible_matches_complete = complete;
        result.inner.state_to_internal_tsid = id_map.tokenizer_states.original_to_internal;
        result.inner.internal_tsid_to_states = id_map.tokenizer_states.internal_to_originals;
        result.inner.original_token_to_internal = id_map.vocab_tokens.original_to_internal;
        result.inner.internal_token_to_tokens = id_map.vocab_tokens.internal_to_originals;
        result
    }

    fn migrate_legacy_v1(
        payload: LegacyDynamicConstraintPayloadV1,
    ) -> crate::Result<DynamicConstraintPayloadV1> {
        if payload.eos_token_id.is_some() {
            return Err(crate::GlrMaskError::Serialization(
                "legacy dynamic constraint artifact embeds Vocab-level EOS semantics; rebuild it with grammar-level end tokens"
                    .to_owned(),
            ));
        }
        Ok(DynamicConstraintPayloadV1 {
            table: payload.table.into(),
            terminal_display_names: payload.terminal_display_names,
            tokenizer: payload.tokenizer.into(),
            ignore_terminal: payload.ignore_terminal,
            direct_regular_automaton: None,
            token_bytes: payload.token_bytes,
            ignore_expr: None,
            terminal_exprs: None,
        })
    }

    fn from_legacy_payload_v1(payload: LegacyDynamicConstraintPayloadV1) -> crate::Result<Self> {
        Ok(Self::from_payload_v2(DynamicConstraintPayloadV2 {
            v1: Self::migrate_legacy_v1(payload)?,
            special_token_terminals: Vec::new(),
        }))
    }

    fn from_legacy_payload_v2(payload: LegacyDynamicConstraintPayloadV2) -> crate::Result<Self> {
        Ok(Self::from_payload_v2(DynamicConstraintPayloadV2 {
            v1: Self::migrate_legacy_v1(payload.v1)?,
            special_token_terminals: payload.special_token_terminals,
        }))
    }

    fn from_payload_v2(payload: DynamicConstraintPayloadV2) -> Self {
        Self::from_payload_v2_with_dynamic_vocab(payload, DynamicMaskVocab::default())
    }

    fn from_payload_v2_with_dynamic_vocab(
        payload: DynamicConstraintPayloadV2,
        dynamic_mask_vocab: DynamicMaskVocab,
    ) -> Self {
        let mut inner = Self::constraint_from_payload_v2_with_dynamic_vocab(
            payload,
            dynamic_mask_vocab,
        );
        inner.rebuild_dynamic_runtime_caches();
        Self {
            inner,
            alternatives: Vec::new(),
            composition_grammars: vec![None],
            external_vocab_artifact_cache: None,
        }
    }

    fn constraint_from_payload_v2_with_dynamic_vocab(
        payload: DynamicConstraintPayloadV2,
        dynamic_mask_vocab: DynamicMaskVocab,
    ) -> Constraint {
        let DynamicConstraintPayloadV2 { v1, special_token_terminals } = payload;
        let DynamicConstraintPayloadV1 { table, terminal_display_names, tokenizer,
            ignore_terminal, direct_regular_automaton, token_bytes, ignore_expr, terminal_exprs } = v1;
        Self::constraint_from_runtime_parts(DynamicConstraintPayloadV2 {
            v1: DynamicConstraintPayloadV1 { table: table.into(), terminal_display_names, tokenizer,
                ignore_terminal, direct_regular_automaton, token_bytes, ignore_expr, terminal_exprs },
            special_token_terminals,
        }, dynamic_mask_vocab)
    }

    /// Shared assembly accepts an absent parser store. A data-only constructor
    /// does not construct even a placeholder GLR table and installs its program
    /// before any runtime-derived caches are initialized.
    fn constraint_from_runtime_parts(
        payload: DynamicConstraintPayloadV2<crate::runtime::parser_backend::ParserTableStorage>,
        dynamic_mask_vocab: DynamicMaskVocab,
    ) -> Constraint {
        let DynamicConstraintPayloadV2 {
            v1: mut payload,
            special_token_terminals,
        } = payload;
        payload
            .tokenizer
            .restore_terminal_exprs(payload.terminal_exprs.take())
            .expect("dynamic payload terminal expressions must match tokenizer terminal count");
        let max_token_id = payload
            .token_bytes
            .keys()
            .next_back()
            .copied()
            .into_iter()
            .chain(special_token_terminals.iter().map(|special| special.token_id))
            .max()
            .unwrap_or(0);
        let inner = Constraint {
            end_tokens: std::sync::Arc::from([]),
            runtime_backend: crate::runtime::ConstraintRuntimeBackend::Dynamic,
            static_dynamic_overlay: None,
            boundary_trigger: crate::runtime::BoundaryTrigger::None,
            boundary_candidate_summary: std::sync::OnceLock::new(),
            late_grammar_slots: Vec::new(),
            late_bind_vocab: std::sync::OnceLock::new(),
            scoped_ignore_only_tokens: Vec::new(),
            scoped_ignore_prefix_fusions: Vec::new(),
            parser_dwa: DWA::new(payload.tokenizer.num_states(), max_token_id),
            packed_parser_dwa: None,
            parser_start_final_override: None,
            parser_top_accept: BTreeMap::new(),
            parser_top_accept_parts: BTreeMap::new(),
            direct_regular_l1_complete_by_terminal: BTreeMap::new(),
            packed_non_dwa_weights: None,
            direct_regular_wide_frontier_acceptance: Vec::new(),
            direct_regular_dynamic_hot_frontiers: Vec::new(),
            direct_regular_parser_state_acceptance: Vec::new(),
            direct_regular_automaton: payload.direct_regular_automaton,
            table: payload.table,
            terminal_display_names: payload.terminal_display_names,
            tokenizer: payload.tokenizer.into(),
            boundary_completion_index: None,
            tokenizer_has_epsilon_transitions: false,
            ignore_terminal: payload.ignore_terminal,
            special_token_terminals,
            dynamic_mask_vocab,
            lazy_dynamic_mask_vocab: std::sync::OnceLock::new(),
            empty_byte_token_ids: payload.token_bytes.iter()
                .filter_map(|(&id, bytes)| bytes.is_empty().then_some(id))
                .collect::<Vec<u32>>().into(),
            possible_matches: BTreeMap::new(),
            possible_matches_complete: false,
            state_to_internal_tsid: Vec::new(),
            internal_tsid_to_states: Vec::new(),
            deferred_internal_tsid_to_states: Default::default(),
            composition_reset_tokens_by_terminal: Vec::new(),
            unbound_grammar_placeholders: BTreeMap::new(),
            composition_parser_templates_by_terminal: Vec::new(),
            composition_parser_characterizations_by_terminal: Vec::new(),
            composition_grammar_summary: None,
            terminal_live_states: Vec::new(),
            state_internal_tsid_offsets: Vec::new(),
            state_internal_tsids: Vec::new(),
            runtime_source_state_offset: None,
            runtime_product_source_offsets: Vec::new(),
            runtime_product_source_states: Vec::new(),
            runtime_product_exact_source_states: Vec::new(),
            runtime_product_state_by_source_subset: Default::default(),
            template_dfas_by_terminal: Vec::new(),
            fast_template_dfas_by_terminal: Vec::new(),
            template_parser: None,
            original_token_to_internal: Vec::new(),
            packed_original_token_to_internal: None,
            deferred_original_token_to_internal: std::sync::OnceLock::new(),
            internal_token_to_tokens: Vec::new(),
            deferred_internal_token_to_tokens: std::sync::OnceLock::new(),
            token_bytes: payload.token_bytes,
            packed_token_bytes: None,
            internal_token_bytes: BTreeMap::new(),
            token_bytes_dense: Vec::new(),
            internal_token_buf_masks: Vec::new(),
            word_group_buf_masks: Vec::new(),
            pair_word_group_buf_masks: Default::default(),
            quad_word_group_buf_masks: Default::default(),
            super_word_group_buf_masks: Default::default(),
            mega_word_group_buf_masks: Default::default(),
            giga_word_group_buf_masks: Default::default(),
            word_group_sparse_masks: Vec::new(),
            word_group_prefix_buf_masks: Default::default(),
            word_group_sparse_prefix_entries: Vec::new(),
            quad_group_sparse_masks: Vec::new(),
            quad_group_dense_masks: Vec::new(),
            byte_group_sparse_masks: Vec::new(),
            byte_group_dense_masks: Vec::new(),
            word_group_sparse_total_entries: 0,
            word_group_sparse_max_entries: 0,
            all_tokens_buf_mask: Box::new([]),
            internal_token_dense_words: 0,
            weight_token_dense_masks: Default::default(),
            packed_dwa_token_dense_masks: Default::default(),
            weight_token_buf_masks: Default::default(),
            weight_token_sparse_buf_masks: Default::default(),
            direct_sparse_weight_token_sets: Default::default(),
            seed_terminal_dense: Default::default(),
            seed_terminal_dense_fallback: Default::default(),
            seed_universe_dense: Arc::from(Vec::<u64>::new().into_boxed_slice()),
            dwa_fast_transitions: Default::default(),
            parser_runtime_caches_prebuilt: false,
            indexed_dag_dense_transitions: Vec::new(),
            indexed_dag_dense_finals: Vec::new(),
            tokenizer_fast_transitions: Default::default(),
            heavy_token_dense_masks: Vec::new(),
            internal_token_buf_flat: Box::new([]),
            backed_internal_token_buf_flat: None,
            internal_token_buf_offsets: Box::new([]),
            total_internal_buf_cost: 0,
            heavy_token_indices: Vec::new(),
            heavy_total_cost: 0,
            light_avg_cost_x256: 0,
            internal_token_buf_op_costs: Vec::new(),
            word_group_buf_op_costs: Vec::new(),
            final_mask_mapping: Default::default(),
            parser_state_domain_labels: Vec::new(),
            ignore_expr: payload.ignore_expr,
            serialized_artifact_cache: None,
            deferred_terminal_exprs_blob: None,
            deferred_terminal_exprs: Default::default(),
            deferred_composition_metadata_blob: None,
            composition_link_metadata_materialized: true,
            deferred_table_rules_blob: None,
            deferred_table_rules: Default::default(),
        };
        inner
    }

    pub(crate) fn from_template_runtime_parts(
        tokenizer: Tokenizer,
        terminal_display_names: Vec<String>,
        ignore_terminal: Option<TerminalID>,
        templates: Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
        parser: Arc<crate::runtime::parser_backend::TemplateParser>,
        vocab: &Vocab,
    ) -> Constraint {
        let ignore_expr = ignore_terminal.and_then(|t|tokenizer.terminal_expr(t).cloned());
        let terminal_exprs = tokenizer.terminal_exprs().map(ToOwned::to_owned);
        let payload = DynamicConstraintPayloadV2 {
            v1: DynamicConstraintPayloadV1 {
                table: crate::runtime::parser_backend::ParserTableStorage::absent(),
                terminal_display_names, tokenizer, ignore_terminal,
                direct_regular_automaton: None, token_bytes: vocab.entries_arc(),
                ignore_expr, terminal_exprs,
            },
            special_token_terminals: Vec::new(),
        };
        let dynamic_vocab = crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(vocab);
        let mut inner = Self::constraint_from_runtime_parts(payload, dynamic_vocab);
        inner.template_dfas_by_terminal = templates;
        inner.template_parser = Some(parser);
        inner.fast_template_dfas_by_terminal = inner.compute_fast_template_dfas();
        let _ = inner.late_bind_vocab.set(vocab.clone());
        inner.rebuild_dynamic_runtime_caches();
        assert!(!inner.table.is_present(), "data-only constructor created an LR table");
        inner
    }

    pub(crate) fn from_alternatives(mut alternatives: Vec<Self>) -> Self {
        assert!(!alternatives.is_empty(), "dynamic union requires at least one alternative");
        let first = alternatives.remove(0);
        let mut external_vocab_artifact_cache = first.external_vocab_artifact_cache;
        let mut result = Self {
            inner: first.inner,
            alternatives: first.alternatives,
            composition_grammars: first.composition_grammars,
            external_vocab_artifact_cache: None,
        };
        for alternative in alternatives {
            // Combining alternatives changes the serialized payload; none of
            // the per-alternative cached artifacts represents the new union.
            external_vocab_artifact_cache = None;
            result.alternatives.push(alternative.inner);
            result.alternatives.extend(alternative.alternatives);
            result.composition_grammars.extend(alternative.composition_grammars);
        }
        result.external_vocab_artifact_cache = external_vocab_artifact_cache;
        result
    }

    pub(crate) fn from_constraints(mut constraints: Vec<Constraint>) -> Self {
        assert!(!constraints.is_empty(), "dynamic union requires at least one alternative");
        let inner = constraints.remove(0);
        let composition_grammars = vec![None; constraints.len() + 1];
        Self { inner, alternatives: constraints, composition_grammars, external_vocab_artifact_cache: None }
    }

    pub(crate) fn clone_constraints(&self) -> Vec<Constraint> {
        std::iter::once(&self.inner).chain(&self.alternatives).cloned().collect()
    }

    pub(crate) fn into_constraints(self) -> Vec<Constraint> {
        std::iter::once(self.inner)
            .chain(self.alternatives)
            .collect()
    }

    pub(crate) fn constraints_mut(&mut self) -> impl Iterator<Item = &mut Constraint> {
        self.external_vocab_artifact_cache = None;
        std::iter::once(&mut self.inner).chain(&mut self.alternatives)
    }

    pub(crate) fn targets_vocab(&self, vocab: &Vocab) -> bool {
        std::iter::once(&self.inner)
            .chain(&self.alternatives)
            .all(|constraint| constraint.token_bytes_match_vocab(vocab))
    }

    pub(crate) fn attach_late_grammar_placeholders(
        &mut self,
        placeholders: &[(u32, String)],
    ) -> crate::Result<()> {
        self.external_vocab_artifact_cache = None;
        for constraint in std::iter::once(&mut self.inner).chain(&mut self.alternatives) {
            constraint.late_grammar_slots.clear();
            for (placeholder_token_id, binding_name) in placeholders {
                let mut matching = constraint
                    .special_token_terminals
                    .iter()
                    .filter(|special| special.token_id == *placeholder_token_id)
                    .map(|special| special.terminal_id);
                let Some(terminal_id) = matching.next() else {
                    // Dynamic alternatives can omit an unreachable choice.
                    continue;
                };
                if matching.next().is_some() {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "compiled GLRM external subgrammar {binding_name:?} has multiple hidden linker terminals",
                    )));
                }
                constraint.late_grammar_slots.push(crate::runtime::LateGrammarSlot {
                    name: binding_name.clone(),
                    terminal_id,
                });
            }
        }
        Ok(())
    }

    pub(crate) fn set_composition_grammar(&mut self, grammar: GrammarDef) {
        self.external_vocab_artifact_cache = None;
        assert_eq!(self.composition_grammars.len(), 1);
        self.composition_grammars[0] = Some(grammar);
    }

    pub(crate) fn reconstruct_composition_grammar(constraint: &Constraint) -> crate::Result<GrammarDef> {
        let exprs = constraint.tokenizer.terminal_exprs().ok_or_else(|| {
            crate::GlrMaskError::Compilation(
                "this legacy dynamic artifact does not retain terminal expressions required for compiled-child composition; rebuild it".to_owned(),
            )
        })?;
        let num_terminals = constraint.tokenizer.num_terminals() as usize;
        if exprs.len() != num_terminals {
            return Err(crate::GlrMaskError::Compilation(format!(
                "dynamic artifact terminal-expression count {} does not match tokenizer terminal count {num_terminals}",
                exprs.len(),
            )));
        }

        let special_by_terminal = constraint
            .special_token_terminals
            .iter()
            .map(|special| (special.terminal_id, special.token_id))
            .collect::<BTreeMap<_, _>>();
        let terminals = exprs
            .iter()
            .cloned()
            .enumerate()
            .map(|(id, expr)| {
                let id = id as u32;
                special_by_terminal.get(&id).copied().map_or(
                    Terminal::Expr { id, expr },
                    |token_id| Terminal::SpecialToken { id, token_id },
                )
            })
            .collect::<Vec<_>>();
        let terminal_names = constraint
            .terminal_display_names
            .iter()
            .cloned()
            .enumerate()
            .map(|(id, name)| (id as u32, name))
            .collect::<BTreeMap<_, _>>();

        let (start, rules, nonterminal_names) = if constraint.direct_regular_automaton.is_some() {
            (0, Vec::new(), BTreeMap::new())
        } else {
            let augmented = constraint.table.rules.first().ok_or_else(|| {
                crate::GlrMaskError::Compilation(
                    "dynamic artifact has no augmented-start rule for compiled-child composition"
                        .to_owned(),
                )
            })?;
            let start = match augmented.rhs.as_slice() {
                [Symbol::Nonterminal(start)] => *start,
                rhs => {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "dynamic artifact augmented-start rule must contain one nonterminal, found {rhs:?}",
                    )));
                }
            };
            let nonterminal_names = constraint
                .table
                .nonterminal_display_names
                .iter()
                .cloned()
                .enumerate()
                .filter(|(id, _)| *id as u32 != augmented.lhs)
                .map(|(id, name)| (id as u32, name))
                .collect::<BTreeMap<_, _>>();
            (start, constraint.table.rules[1..].to_vec(), nonterminal_names)
        };

        Ok(GrammarDef {
            rules,
            start,
            terminals,
            nonterminal_names,
            terminal_names,
            ignore_terminal: constraint.ignore_terminal,
            lexer_partitions: BTreeMap::new(),
            residual_isolation_classes: BTreeMap::new(),
            requires_global_terminal_observation: true,
            direct_regular_automaton: constraint.direct_regular_automaton.clone(),
        })
    }

    pub(crate) fn bind_vocab_exact(&mut self, vocab: &Vocab) -> Result<(), String> {
        self.external_vocab_artifact_cache = None;
        self.inner.bind_vocab_exact(vocab)?;
        for alternative in &mut self.alternatives {
            alternative.bind_vocab_exact(vocab)?;
        }
        Ok(())
    }

    pub(crate) fn into_constraint(self) -> Constraint {
        assert!(
            self.alternatives.is_empty(),
            "a union dynamic constraint cannot be converted to one Constraint",
        );
        self.inner
    }

    fn payload_for_constraint(constraint: &Constraint) -> DynamicConstraintPayloadV2 {
        let retain_terminal_exprs = std::env::var("GLRMASK_DYNAMIC_TRANSFER_TERMINAL_EXPRS")
            .ok()
            .is_none_or(|value| !matches!(value.trim(), "0" | "false" | "no" | "off"));
        let terminal_exprs = retain_terminal_exprs
            .then(|| constraint.tokenizer.terminal_exprs().map(ToOwned::to_owned))
            .flatten();
        DynamicConstraintPayloadV2 {
            v1: DynamicConstraintPayloadV1 {
                table: constraint.table.clone_lr(),
                terminal_display_names: constraint.terminal_display_names.clone(),
                tokenizer: constraint.tokenizer.as_ref().clone(),
                ignore_terminal: constraint.ignore_terminal,
                direct_regular_automaton: constraint.direct_regular_automaton.clone(),
                token_bytes: Arc::clone(&constraint.token_bytes),
                ignore_expr: constraint.ignore_expr.clone(),
                terminal_exprs,
            },
            special_token_terminals: constraint.special_token_terminals.clone(),
        }
    }

    fn payload_v4_for_constraint(constraint: &Constraint) -> DynamicConstraintPayloadV4Alternative {
        DynamicConstraintPayloadV4Alternative {
            v2: Self::payload_for_constraint(constraint),
            late_grammar_slots: constraint.late_grammar_slots.clone(),
            virtual_runtimes: constraint.tokenizer.virtual_runtime_metadata(),
        }
    }

    fn payload_v5_for_constraint(constraint: &Constraint) -> DynamicConstraintPayloadV5Alternative {
        DynamicConstraintPayloadV5Alternative {
            base: Self::payload_v4_for_constraint(constraint),
            terminal_observation_classes: constraint
                .dynamic_mask_vocab
                .terminal_observation_classes_for_artifact(),
        }
    }

    fn payload_v7_for_constraint(constraint: &Constraint) -> DynamicConstraintPayloadV7Alternative {
        DynamicConstraintPayloadV7Alternative {
            base: Self::payload_v5_for_constraint(constraint),
            projected_terminal_quotients: constraint
                .dynamic_mask_vocab
                .projected_terminal_quotients_for_artifact(),
            boundary_trigger: DynamicBoundaryTriggerWire::from_trigger(
                &constraint.boundary_trigger,
            ),
            recursive_constraint_artifact: constraint
                .uses_compact_segmented_parser_runtime()
                .then(|| constraint.save()),
        }
    }

    fn restore_terminal_observation_classes(
        constraint: &mut Constraint,
        rows: Vec<(TerminalID, Vec<u32>)>,
    ) -> crate::Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut seen = BTreeSet::<TerminalID>::new();
        let expected_states = constraint.tokenizer.num_states() as usize;
        let mut restored = Vec::with_capacity(rows.len());
        for (terminal, classes) in rows {
            if terminal >= constraint.tokenizer.num_terminals() {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "terminal-observation certificate references terminal {terminal}, but tokenizer has {} terminals",
                    constraint.tokenizer.num_terminals(),
                )));
            }
            if !seen.insert(terminal) {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "terminal-observation certificate repeats terminal {terminal}",
                )));
            }
            if classes.len() != expected_states {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "terminal-observation certificate for terminal {terminal} has {} states, expected {expected_states}",
                    classes.len(),
                )));
            }
            restored.push((terminal, Arc::from(classes)));
        }
        constraint
            .dynamic_mask_vocab
            .set_terminal_observation_classes(restored);
        Ok(())
    }

    fn restore_projected_terminal_quotients(
        constraint: &mut Constraint,
        rows: Vec<(TerminalID, TerminalProjectedQuotient)>,
    ) -> crate::Result<()> {
        let mut seen = BTreeSet::<TerminalID>::new();
        for (terminal, quotient) in &rows {
            if !seen.insert(*terminal) {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "projected-terminal quotient repeats terminal {terminal}"
                )));
            }
            quotient
                .validate_exact_projection(&constraint.tokenizer, *terminal)
                .map_err(crate::GlrMaskError::Serialization)?;
        }
        // Calling the setter even for an empty row set is intentional: v20/v9
        // use empty+prepared to mean compile-time analysis completed and found
        // no useful quotient, so runtime load must not repeat that work.
        constraint
            .dynamic_mask_vocab
            .set_projected_terminal_quotients(rows);
        Ok(())
    }

    fn transfer_payload_from_constraint_owned(
        constraint: Constraint,
    ) -> DynamicConstraintTransferAlternativeV5 {
        let terminal_exprs_compressed = constraint
            .tokenizer
            .terminal_exprs()
            .map(|exprs| {
                compressed_terminal_exprs_serde::encode(exprs)
                    .expect("dynamic transfer terminal expression compression should succeed")
            });
        let virtual_runtimes = constraint.tokenizer.virtual_runtime_metadata();
        let mask_quotient = constraint
            .dynamic_mask_vocab
            .mask_tokenizer_quotient_for_transfer();
        let terminal_observation_classes = constraint
            .dynamic_mask_vocab
            .terminal_observation_classes_for_artifact();
        let projected_terminal_quotients = constraint
            .dynamic_mask_vocab
            .projected_terminal_quotients_for_artifact();
        let projected_terminal_quotients_prepared = constraint
            .dynamic_mask_vocab
            .projected_terminal_quotients_prepared();
        let boundary_trigger = DynamicBoundaryTriggerWire::from_trigger(&constraint.boundary_trigger);
        let recursive_constraint_artifact = constraint
            .uses_compact_segmented_parser_runtime()
            .then(|| constraint.save());
        DynamicConstraintTransferAlternativeV5 {
            table: CompactTransferTable {
                table: constraint.table.into_lr(),
                deferred_rules: constraint.deferred_table_rules_blob,
            },
            terminal_display_names: constraint.terminal_display_names,
            tokenizer: FastTransferTokenizer(Arc::unwrap_or_clone(constraint.tokenizer)),
            ignore_terminal: constraint.ignore_terminal,
            direct_regular_automaton: constraint.direct_regular_automaton,
            special_token_terminals: constraint.special_token_terminals,
            ignore_expr: constraint.ignore_expr,
            terminal_exprs_compressed,
            mask_tokenizer: mask_quotient
                .as_ref()
                .map(|(tokenizer, _)| CompactTransferTokenizer(tokenizer.clone())),
            full_to_mask_state: mask_quotient.map_or_else(Vec::new, |(_, mapping)| mapping),
            virtual_runtimes,
            terminal_observation_classes,
            projected_terminal_quotients,
            boundary_trigger,
            recursive_constraint_artifact,
        }
    }

    fn compact_table_bytes_for_transfer(constraint: &Constraint) -> Vec<u8> {
        let materialized;
        let table = if let Some(rules) = constraint.deferred_table_rules_blob.as_ref() {
            let decoded = bincode::deserialize::<Vec<crate::grammar::flat::Rule>>(rules.as_slice())
                .expect("loaded deferred GLR rules must remain valid");
            assert_eq!(
                decoded.len(),
                constraint.table.num_rules as usize,
                "loaded deferred GLR rule count changed before transfer save",
            );
            assert_eq!(
                decoded.first(),
                constraint.table.rules.first(),
                "loaded deferred GLR augmented rule changed before transfer save",
            );
            materialized = {
                let mut table = constraint.table.clone();
                table.rules = decoded;
                table
            };
            &materialized
        } else {
            &constraint.table
        };
        crate::compiler::glr::table::artifact_serde::to_compact_bytes(table)
    }

    fn transfer_sections_v11_from_constraint(
        constraint: &Constraint,
    ) -> DynamicConstraintTransferSectionsV11 {
        let table = Self::compact_table_bytes_for_transfer(constraint);
        let tokenizer =
            crate::automata::lexer::tokenizer::artifact_serde::to_fast_bytes_with_packed_metadata(
                &constraint.tokenizer,
            );
        let terminal_exprs_compressed = if let Some(exprs) = constraint.tokenizer.terminal_exprs() {
            compressed_terminal_exprs_serde::encode(exprs)
                .expect("dynamic transfer terminal expression compression should succeed")
        } else if let Some(blob) = constraint.deferred_terminal_exprs_blob.as_ref() {
            match blob {
                crate::runtime::DeferredTerminalExprBytes::CompressedOwned(_)
                | crate::runtime::DeferredTerminalExprBytes::CompressedBacked { .. } => {
                    blob.as_slice().to_vec()
                }
                crate::runtime::DeferredTerminalExprBytes::Owned(_)
                | crate::runtime::DeferredTerminalExprBytes::Backed { .. } => {
                    zstd::bulk::compress(blob.as_slice(), 1)
                        .expect("deferred terminal expression compression should succeed")
                }
            }
        } else {
            Vec::new()
        };
        let virtual_runtimes = constraint.tokenizer.virtual_runtime_metadata();
        let residual_runtime_oracles = std::env::var("GLRMASK_DYNAMIC_TRANSFER_RESIDUAL_ORACLE_SIDECAR")
            .ok()
            .is_some_and(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"))
            .then(|| constraint.tokenizer.virtual_residual_runtime_oracles())
            .unwrap_or_default();
        let mask_quotient = constraint
            .dynamic_mask_vocab
            .mask_tokenizer_quotient_for_transfer();
        let terminal_observation_classes = constraint
            .dynamic_mask_vocab
            .terminal_observation_classes_for_artifact();
        let projected_terminal_quotients = constraint
            .dynamic_mask_vocab
            .projected_terminal_quotients_for_artifact();
        let projected_terminal_quotients_prepared = constraint
            .dynamic_mask_vocab
            .projected_terminal_quotients_prepared();
        let boundary_trigger = DynamicBoundaryTriggerWire::from_trigger(&constraint.boundary_trigger);
        let recursive_constraint_artifact = constraint
            .uses_compact_segmented_parser_runtime()
            .then(|| constraint.save())
            .unwrap_or_default();
        let metadata = DynamicConstraintTransferMetadataV11 {
            terminal_display_names: constraint.terminal_display_names.clone(),
            ignore_terminal: constraint.ignore_terminal,
            direct_regular_automaton: constraint.direct_regular_automaton.clone(),
            special_token_terminals: constraint.special_token_terminals.clone(),
            ignore_expr: constraint.ignore_expr.clone(),
            mask_tokenizer: mask_quotient
                .as_ref()
                .map(|(tokenizer, _)| CompactTransferTokenizer(tokenizer.clone())),
            full_to_mask_state: mask_quotient.map_or_else(Vec::new, |(_, mapping)| mapping),
            virtual_runtimes,
            residual_runtime_oracles,
            terminal_observation_classes,
            projected_terminal_quotients,
            projected_terminal_quotients_prepared,
            boundary_trigger,
            grammar_quotient_vocab: constraint
                .dynamic_mask_vocab
                .is_grammar_quotiented()
                .then(|| constraint.dynamic_mask_vocab.to_external_vocab_artifact())
                .flatten(),
        };
        DynamicConstraintTransferSectionsV11 {
            table,
            tokenizer,
            terminal_exprs_compressed,
            recursive_constraint_artifact,
            metadata: bincode::serialize(&metadata)
                .expect("dynamic v11 transfer metadata serialization should succeed"),
        }
    }

    fn transfer_sections_v12_from_constraint(
        constraint: &Constraint,
    ) -> DynamicConstraintTransferSectionsV12 {
        let profile_transfer = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile_transfer.then(std::time::Instant::now);
        let base_started = profile_transfer.then(std::time::Instant::now);
        let table = Self::compact_table_bytes_for_transfer(constraint);
        let tokenizer = if constraint.tokenizer.has_deterministic_dispatch() {
            // TKF3 serialization is longer than the live scalar-dispatch proof on
            // the dynamic tail. Run them together, then add the proof certificate
            // to the already-built header. Tokenizers without reset dispatch keep
            // the old serial path and pay no Rayon scheduling overhead.
            let ((mut tokenizer, tokenizer_ms), (scalar_dispatch, scalar_proof_ms)) = rayon::join(
                || {
                    let started = profile_transfer.then(std::time::Instant::now);
                    let tokenizer = crate::automata::lexer::tokenizer::artifact_serde::to_fast_bytes_with_packed_metadata(
                        &constraint.tokenizer,
                    );
                    let elapsed = started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1e3);
                    (tokenizer, elapsed)
                },
                || {
                    let started = profile_transfer.then(std::time::Instant::now);
                    let scalar_dispatch = constraint.tokenizer.has_scalar_deterministic_dispatch();
                    let elapsed = started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1e3);
                    (scalar_dispatch, elapsed)
                },
            );
            if scalar_dispatch {
                crate::automata::lexer::tokenizer::artifact_serde::mark_fast_wire_scalar_deterministic_dispatch(
                    &mut tokenizer,
                );
            }
            if profile_transfer {
                eprintln!(
                    "[glrmask/profile][dynamic_transfer_scalar_proof_overlap] tokenizer_ms={:.3} scalar_proof_ms={:.3} scalar_dispatch={}",
                    tokenizer_ms, scalar_proof_ms, scalar_dispatch,
                );
            }
            tokenizer
        } else {
            crate::automata::lexer::tokenizer::artifact_serde::to_fast_bytes_with_packed_metadata(
                &constraint.tokenizer,
            )
        };
        let decoded_fallback_exprs;
        let (terminal_exprs_compressed, fallback_exprs) = if let Some(exprs) = constraint.tokenizer.terminal_exprs() {
            (
                compressed_terminal_exprs_serde::encode(exprs)
                    .expect("dynamic transfer terminal expression compression should succeed"),
                Some(exprs),
            )
        } else if let Some(blob) = constraint.deferred_terminal_exprs_blob.as_ref() {
            let bytes = match blob {
                crate::runtime::DeferredTerminalExprBytes::CompressedOwned(_)
                | crate::runtime::DeferredTerminalExprBytes::CompressedBacked { .. } => {
                    blob.as_slice().to_vec()
                }
                crate::runtime::DeferredTerminalExprBytes::Owned(_)
                | crate::runtime::DeferredTerminalExprBytes::Backed { .. } => {
                    zstd::bulk::compress(blob.as_slice(), 1)
                        .expect("deferred terminal expression compression should succeed")
                }
            };
            decoded_fallback_exprs = blob.decode_exprs().ok();
            (bytes, decoded_fallback_exprs.as_deref())
        } else {
            (Vec::new(), None)
        };
        let virtual_runtimes = constraint.tokenizer.virtual_runtime_metadata();
        let residual_runtime_oracles = std::env::var("GLRMASK_DYNAMIC_TRANSFER_RESIDUAL_ORACLE_SIDECAR")
            .ok()
            .is_some_and(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"))
            .then(|| constraint.tokenizer.virtual_residual_runtime_oracles())
            .unwrap_or_default();
        let mask_quotient = constraint
            .dynamic_mask_vocab
            .mask_tokenizer_quotient_for_transfer();
        let terminal_observation_classes = constraint
            .dynamic_mask_vocab
            .terminal_observation_classes_for_artifact();
        let projected_terminal_quotients = constraint
            .dynamic_mask_vocab
            .projected_terminal_quotients_for_artifact();
        let projected_terminal_quotients_prepared = constraint
            .dynamic_mask_vocab
            .projected_terminal_quotients_prepared();
        let boundary_trigger = DynamicBoundaryTriggerWire::from_trigger(&constraint.boundary_trigger);
        let recursive_constraint_artifact = constraint
            .uses_compact_segmented_parser_runtime()
            .then(|| constraint.save())
            .unwrap_or_default();
        let metadata_base = DynamicConstraintTransferMetadataV11 {
            terminal_display_names: constraint.terminal_display_names.clone(),
            ignore_terminal: constraint.ignore_terminal,
            direct_regular_automaton: constraint.direct_regular_automaton.clone(),
            special_token_terminals: constraint.special_token_terminals.clone(),
            ignore_expr: constraint.ignore_expr.clone(),
            mask_tokenizer: mask_quotient
                .as_ref()
                .map(|(tokenizer, _)| CompactTransferTokenizer(tokenizer.clone())),
            full_to_mask_state: mask_quotient.map_or_else(Vec::new, |(_, mapping)| mapping),
            virtual_runtimes,
            residual_runtime_oracles,
            terminal_observation_classes,
            projected_terminal_quotients,
            projected_terminal_quotients_prepared,
            boundary_trigger,
            grammar_quotient_vocab: constraint
                .dynamic_mask_vocab
                .is_grammar_quotiented()
                .then(|| constraint.dynamic_mask_vocab.to_external_vocab_artifact())
                .flatten(),
        };
        let metadata = DynamicConstraintTransferMetadataV13 {
            base: metadata_base,
            prepared_master_proofs: constraint
                .dynamic_mask_vocab
                .prepared_master_proof_artifact(),
            virtual_residual_master_slice_artifacts: constraint
                .tokenizer
                .virtual_residual_master_slice_artifacts(),
        };

        let base_ms = base_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let residual_enabled = std::env::var("GLRMASK_DYNAMIC_TRANSFER_VIRTUAL_RESIDUAL_PROJECTIONS")
            .ok()
            .is_none_or(|value| !matches!(value.trim(), "0" | "false" | "no" | "off"));
        let mut residual_prepare_ms = 0.0;
        let mut residual_encode_ms = 0.0;
        let mut residual_existing = false;
        let virtual_residual_wire = if residual_enabled {
            if let Some((mask_tokenizer, projections)) = constraint
                .dynamic_mask_vocab
                .virtual_residual_mask_projection_parts()
            {
                residual_existing = true;
                let started = profile_transfer.then(std::time::Instant::now);
                let wire = crate::runtime::serde::encode_static_virtual_residual_mask_wire_with_fallback(
                    &constraint.tokenizer,
                    fallback_exprs,
                    mask_tokenizer,
                    projections,
                );
                residual_encode_ms = started
                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                wire
            } else if constraint.tokenizer.has_any_virtual_runtime() {
                let mut local_vocab = constraint.dynamic_mask_vocab.clone();
                let started = profile_transfer.then(std::time::Instant::now);
                constraint.prepare_dynamic_virtual_residual_mask_projection(&mut local_vocab);
                residual_prepare_ms = started
                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                if let Some((mask_tokenizer, projections)) =
                    local_vocab.virtual_residual_mask_projection_parts()
                {
                    let started = profile_transfer.then(std::time::Instant::now);
                    let wire = crate::runtime::serde::encode_static_virtual_residual_mask_wire_with_fallback(
                        &constraint.tokenizer,
                        fallback_exprs,
                        mask_tokenizer,
                        projections,
                    );
                    residual_encode_ms = started
                        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                    wire
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        if profile_transfer {
            eprintln!(
                "[glrmask/profile][dynamic_transfer_v12_save_sections] base_ms={:.3} residual_existing={} residual_prepare_ms={:.3} residual_encode_ms={:.3} residual_bytes={} total_ms={:.3}",
                base_ms,
                residual_existing,
                residual_prepare_ms,
                residual_encode_ms,
                virtual_residual_wire.len(),
                total_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }

        DynamicConstraintTransferSectionsV12 {
            table,
            tokenizer,
            terminal_exprs_compressed,
            recursive_constraint_artifact,
            metadata: bincode::serialize(&metadata)
                .expect("dynamic v12 transfer metadata serialization should succeed"),
            virtual_residual_wire,
        }
    }

    fn build_external_vocab_artifact_bytes(&self) -> Vec<u8> {
        if self.inner.has_template_parser() || self.alternatives.iter().any(Constraint::has_template_parser) {
            return self.save_template_alternatives(true);
        }
        let profile_transfer = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile_transfer.then(std::time::Instant::now);
        let sections_started = profile_transfer.then(std::time::Instant::now);
        let alternatives = std::iter::once(&self.inner)
            .chain(self.alternatives.iter())
            .map(Self::transfer_sections_v12_from_constraint)
            .collect::<Vec<_>>();
        let sections_ms = sections_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let alternative_count = u32::try_from(alternatives.len())
            .expect("dynamic transfer alternative count exceeds u32");
        let descriptor_bytes = alternatives
            .len()
            .checked_mul(DYNAMIC_TRANSFER_V12_ALT_DESCRIPTOR_LEN)
            .expect("dynamic transfer descriptor size overflow");
        let section_bytes = alternatives.iter().fold(0usize, |total, alternative| {
            total
                .checked_add(alternative.table.len())
                .and_then(|total| total.checked_add(alternative.tokenizer.len()))
                .and_then(|total| total.checked_add(alternative.terminal_exprs_compressed.len()))
                .and_then(|total| total.checked_add(alternative.recursive_constraint_artifact.len()))
                .and_then(|total| total.checked_add(alternative.metadata.len()))
                .and_then(|total| total.checked_add(alternative.virtual_residual_wire.len()))
                .expect("dynamic transfer section size overflow")
        });
        let payload_capacity = DYNAMIC_TRANSFER_V12_PAYLOAD_HEADER_LEN
            .checked_add(descriptor_bytes)
            .and_then(|total| total.checked_add(section_bytes))
            .expect("dynamic transfer payload size overflow");
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload_capacity);
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_MAGIC);
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&alternative_count.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let descriptor_start = bytes.len();
        bytes.resize(descriptor_start + descriptor_bytes, 0);
        for (index, alternative) in alternatives.into_iter().enumerate() {
            let lengths = [
                alternative.table.len(),
                alternative.tokenizer.len(),
                alternative.terminal_exprs_compressed.len(),
                alternative.recursive_constraint_artifact.len(),
                alternative.metadata.len(),
                alternative.virtual_residual_wire.len(),
            ];
            let mut descriptor_pos =
                descriptor_start + index * DYNAMIC_TRANSFER_V12_ALT_DESCRIPTOR_LEN;
            for length in lengths {
                let length = u64::try_from(length)
                    .expect("dynamic transfer section length exceeds u64");
                bytes[descriptor_pos..descriptor_pos + 8].copy_from_slice(&length.to_le_bytes());
                descriptor_pos += 8;
            }
            bytes.extend_from_slice(&alternative.table);
            bytes.extend_from_slice(&alternative.tokenizer);
            bytes.extend_from_slice(&alternative.terminal_exprs_compressed);
            bytes.extend_from_slice(&alternative.recursive_constraint_artifact);
            bytes.extend_from_slice(&alternative.metadata);
            bytes.extend_from_slice(&alternative.virtual_residual_wire);
        }
        let payload_len = bytes.len() - DYNAMIC_CONSTRAINT_HEADER_LEN;
        bytes[10..18].copy_from_slice(&(payload_len as u64).to_le_bytes());
        if profile_transfer {
            eprintln!(
                "[glrmask/profile][dynamic_transfer_v12_save] sections_ms={:.3} copy_ms={:.3} bytes={} total_ms={:.3}",
                sections_ms,
                total_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0) - sections_ms,
                bytes.len(),
                total_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        bytes
    }

    /// Materialize the current external-vocabulary artifact while the
    /// constraint is still inside compilation/finalization. This keeps
    /// persistence honest: `save()` later copies bytes rather than running
    /// another serializer/finalizer whose cost would be hidden outside build.
    pub(crate) fn cache_external_vocab_artifact_for_save(&mut self) {
        if self.external_vocab_artifact_cache.is_some() {
            return;
        }
        let bytes = self.build_external_vocab_artifact_bytes();
        self.external_vocab_artifact_cache = Some(Arc::new(bytes));
    }

    /// Compact transfer artifact that deliberately omits vocabulary bytes.
    /// Pair with `load_with_vocab`. This is the natural persisted format for
    /// APIs (such as Python) whose load operation already requires a Vocab.
    pub fn save_with_external_vocab(&self) -> Vec<u8> {
        if let Some(bytes) = &self.external_vocab_artifact_cache {
            return bytes.as_ref().clone();
        }
        self.build_external_vocab_artifact_bytes()
    }

    pub(crate) fn into_saved(self) -> Vec<u8> {
        self.save_with_external_vocab()
    }

    fn load_transfer_v11(bytes: &[u8], vocab: &Vocab) -> crate::Result<Self> {
        let profile = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_LOAD").is_some();
        let total_started = profile.then(std::time::Instant::now);

        // `load_with_vocab` is a borrowed API, while the backed table/tokenizer
        // decoders retain views beyond this call. Copy the complete artifact
        // once into one shared backing allocation; importantly, do not allocate
        // separate table/tokenizer/expression Vecs as v10 bincode did.
        let backing_started = profile.then(std::time::Instant::now);
        let backing = Arc::new(bytes.to_vec());
        let backing_ms = backing_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let framing_started = profile.then(std::time::Instant::now);
        let payload_start = DYNAMIC_CONSTRAINT_HEADER_LEN;
        let payload = backing.get(payload_start..).ok_or_else(|| {
            crate::GlrMaskError::Serialization("missing dynamic v11 transfer payload".to_owned())
        })?;
        if payload.len() < DYNAMIC_TRANSFER_V11_PAYLOAD_HEADER_LEN {
            return Err(crate::GlrMaskError::Serialization(
                "truncated dynamic v11 transfer payload header".to_owned(),
            ));
        }
        let alternative_count = usize::try_from(u32::from_le_bytes(
            payload[0..4]
                .try_into()
                .expect("dynamic v11 alternative count has fixed width"),
        ))
        .expect("u32 alternative count fits usize on supported platforms");
        if alternative_count == 0 {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic v11 transfer artifact has no alternatives".to_owned(),
            ));
        }
        let flags = u32::from_le_bytes(
            payload[4..8]
                .try_into()
                .expect("dynamic v11 transfer flags have fixed width"),
        );
        if flags != 0 {
            return Err(crate::GlrMaskError::Serialization(
                "unsupported dynamic v11 transfer flags".to_owned(),
            ));
        }
        let descriptor_bytes = alternative_count
            .checked_mul(DYNAMIC_TRANSFER_V11_ALT_DESCRIPTOR_LEN)
            .ok_or_else(|| {
                crate::GlrMaskError::Serialization(
                    "dynamic v11 transfer descriptor size overflow".to_owned(),
                )
            })?;
        let descriptor_end = payload_start
            .checked_add(DYNAMIC_TRANSFER_V11_PAYLOAD_HEADER_LEN)
            .and_then(|value| value.checked_add(descriptor_bytes))
            .ok_or_else(|| {
                crate::GlrMaskError::Serialization(
                    "dynamic v11 transfer descriptor range overflow".to_owned(),
                )
            })?;
        if descriptor_end > backing.len() {
            return Err(crate::GlrMaskError::Serialization(
                "truncated dynamic v11 transfer descriptors".to_owned(),
            ));
        }

        let mut descriptors = Vec::<[usize; 5]>::with_capacity(alternative_count);
        let descriptor_start = payload_start + DYNAMIC_TRANSFER_V11_PAYLOAD_HEADER_LEN;
        for index in 0..alternative_count {
            let mut pos = descriptor_start + index * DYNAMIC_TRANSFER_V11_ALT_DESCRIPTOR_LEN;
            let mut lengths = [0usize; 5];
            for length in &mut lengths {
                let raw = u64::from_le_bytes(
                    backing[pos..pos + 8]
                        .try_into()
                        .expect("validated v11 descriptor has fixed-width field"),
                );
                *length = usize::try_from(raw).map_err(|_| {
                    crate::GlrMaskError::Serialization(
                        "dynamic v11 transfer section length does not fit platform".to_owned(),
                    )
                })?;
                pos += 8;
            }
            if lengths[0] == 0 || lengths[1] == 0 || lengths[4] == 0 {
                return Err(crate::GlrMaskError::Serialization(
                    "dynamic v11 transfer has an empty required section".to_owned(),
                ));
            }
            descriptors.push(lengths);
        }
        let expected_end = descriptors.iter().try_fold(descriptor_end, |cursor, lengths| {
            lengths
                .iter()
                .try_fold(cursor, |cursor, &length| cursor.checked_add(length))
        });
        if expected_end != Some(backing.len()) {
            return Err(crate::GlrMaskError::Serialization(
                "invalid dynamic v11 transfer section lengths".to_owned(),
            ));
        }
        let framing_ms = framing_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let decode_started = profile.then(std::time::Instant::now);
        let token_bytes = vocab.entries_arc();
        let mut cursor = descriptor_end;
        let mut alternatives = Vec::with_capacity(alternative_count);
        for lengths in descriptors {
            let section = |cursor: &mut usize, length: usize| -> std::ops::Range<usize> {
                let start = *cursor;
                *cursor += length;
                start..*cursor
            };
            let table_range = section(&mut cursor, lengths[0]);
            let tokenizer_range = section(&mut cursor, lengths[1]);
            let terminal_exprs_range = section(&mut cursor, lengths[2]);
            let recursive_range = section(&mut cursor, lengths[3]);
            let metadata_range = section(&mut cursor, lengths[4]);

            let metadata_started = profile.then(std::time::Instant::now);
            let metadata: DynamicConstraintTransferMetadataV11 =
                bincode::deserialize(&backing[metadata_range]).map_err(|err| {
                    crate::GlrMaskError::Serialization(format!(
                        "invalid dynamic v11 transfer metadata: {err}"
                    ))
                })?;
            if profile {
                eprintln!(
                    "[glrmask/profile][dynamic_transfer_v11_metadata] projected_terminal_quotients_prepared={} projected_terminal_quotients={}",
                    metadata.projected_terminal_quotients_prepared,
                    metadata.projected_terminal_quotients.len(),
                );
            }
            let metadata_ms = metadata_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

            if !recursive_range.is_empty() {
                let mut inner = Constraint::load_with_vocab(&backing[recursive_range], vocab)?;
                if metadata.projected_terminal_quotients_prepared {
                    Self::restore_projected_terminal_quotients(
                        &mut inner,
                        metadata.projected_terminal_quotients,
                    )?;
                }
                inner.boundary_trigger = metadata.boundary_trigger.into_trigger();
                alternatives.push(Self {
                    inner,
                    alternatives: Vec::new(),
                    composition_grammars: vec![None],
                    external_vocab_artifact_cache: None,
                });
                continue;
            }

            let table_start = table_range.start;
            let tokenizer_start = tokenizer_range.start;
            let table_tokenizer_started = profile.then(std::time::Instant::now);
            let decode_table = || {
                crate::compiler::glr::table::artifact_serde::from_compact_bytes_deferred_backed(
                    &backing[table_range],
                    Arc::clone(&backing),
                    table_start,
                )
            };
            let decode_tokenizer = || {
                crate::automata::lexer::tokenizer::artifact_serde::from_fast_bytes_backed(
                    &backing[tokenizer_range],
                    Arc::clone(&backing),
                    tokenizer_start,
                )
            };
            const PARALLEL_TRANSFER_DECODE_MIN_BYTES: usize = 64 * 1024;
            let parallel_decode = rayon::current_num_threads() > 1
                && lengths[0].saturating_add(lengths[1]) >= PARALLEL_TRANSFER_DECODE_MIN_BYTES;
            let (table_result, tokenizer_result) = if parallel_decode {
                rayon::join(decode_table, decode_tokenizer)
            } else {
                (decode_table(), decode_tokenizer())
            };
            let decoded_table = table_result.map_err(crate::GlrMaskError::Serialization)?;
            let mut tokenizer = tokenizer_result.map_err(crate::GlrMaskError::Serialization)?;
            let table_tokenizer_ms = table_tokenizer_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

            let deferred_terminal_exprs = (!terminal_exprs_range.is_empty()).then(|| {
                crate::runtime::DeferredTerminalExprBytes::CompressedBacked {
                    backing: Arc::clone(&backing),
                    start: terminal_exprs_range.start,
                    len: terminal_exprs_range.len(),
                }
            });

            if !metadata.virtual_runtimes.is_empty() {
                let expressions = deferred_terminal_exprs
                    .as_ref()
                    .ok_or_else(|| {
                        crate::GlrMaskError::Serialization(
                            "dynamic v11 virtual runtime metadata has no terminal expressions"
                                .to_owned(),
                        )
                    })?
                    .decode_exprs()
                    .map_err(crate::GlrMaskError::Serialization)?;
                tokenizer
                    .restore_terminal_exprs_with_virtual_runtime_metadata_and_oracles(
                        Some(expressions),
                        &metadata.virtual_runtimes,
                        &metadata.residual_runtime_oracles,
                        false,
                    )
                    .map_err(crate::GlrMaskError::Serialization)?;
            }

            let assemble_started = profile.then(std::time::Instant::now);
            let dynamic_mask_vocab = Self::dynamic_vocab_from_transfer_artifact(
                metadata.grammar_quotient_vocab.clone(),
                vocab,
            )?;
            let mut inner = Self::constraint_from_payload_v2_with_dynamic_vocab(
                DynamicConstraintPayloadV2 {
                    v1: DynamicConstraintPayloadV1 {
                        table: decoded_table.table,
                        terminal_display_names: metadata.terminal_display_names,
                        tokenizer,
                        ignore_terminal: metadata.ignore_terminal,
                        direct_regular_automaton: metadata.direct_regular_automaton,
                        token_bytes: Arc::clone(&token_bytes),
                        ignore_expr: metadata.ignore_expr,
                        terminal_exprs: None,
                    },
                    special_token_terminals: metadata.special_token_terminals,
                },
                dynamic_mask_vocab,
            );
            if let Some(mask_tokenizer) = metadata.mask_tokenizer {
                inner.dynamic_mask_vocab.set_mask_tokenizer_quotient(
                    mask_tokenizer.0,
                    metadata.full_to_mask_state,
                );
            }
            Self::restore_terminal_observation_classes(
                &mut inner,
                metadata.terminal_observation_classes,
            )?;
            if metadata.projected_terminal_quotients_prepared {
                Self::restore_projected_terminal_quotients(
                    &mut inner,
                    metadata.projected_terminal_quotients,
                )?;
            }
            inner.boundary_trigger = metadata.boundary_trigger.into_trigger();
            inner.deferred_table_rules_blob = decoded_table.deferred_rules;
            inner.deferred_table_rules = std::sync::OnceLock::new();
            if inner.tokenizer.terminal_exprs().is_none() {
                inner.deferred_terminal_exprs_blob = deferred_terminal_exprs;
                inner.deferred_terminal_exprs = std::sync::OnceLock::new();
            }
            let assemble_ms = assemble_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            let rebuild_started = profile.then(std::time::Instant::now);
            inner.rebuild_dynamic_runtime_caches();
            let rebuild_ms = rebuild_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            if profile {
                eprintln!(
                    "[glrmask/profile][dynamic_transfer_v11_alt] metadata_ms={:.3} table_tokenizer_ms={:.3} assemble_ms={:.3} rebuild_ms={:.3}",
                    metadata_ms,
                    table_tokenizer_ms,
                    assemble_ms,
                    rebuild_ms,
                );
            }
            alternatives.push(Self {
                inner,
                alternatives: Vec::new(),
                composition_grammars: vec![None],
                external_vocab_artifact_cache: None,
            });
        }
        let payload_decode_ms = decode_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let loaded = Self::from_alternatives(alternatives);
        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][dynamic_transfer_load] version={} bytes={} backing_ms={:.3} framing_ms={:.3} payload_decode_ms={:.3} finalize_ms={:.3} total_ms={:.3}",
                LEGACY_DYNAMIC_TRANSFER_VERSION_V11,
                bytes.len(),
                backing_ms,
                framing_ms,
                payload_decode_ms,
                0.0,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(loaded)
    }

    fn load_transfer_v12(bytes: &[u8], vocab: &Vocab) -> crate::Result<Self> {
        let profile = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_LOAD").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);

        let backing_started = profile.then(std::time::Instant::now);
        let backing = Arc::new(bytes.to_vec());
        let backing_ms = backing_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let framing_started = profile.then(std::time::Instant::now);
        let payload_start = DYNAMIC_CONSTRAINT_HEADER_LEN;
        let payload = backing.get(payload_start..).ok_or_else(|| {
            crate::GlrMaskError::Serialization("missing dynamic v12 transfer payload".to_owned())
        })?;
        if payload.len() < DYNAMIC_TRANSFER_V12_PAYLOAD_HEADER_LEN {
            return Err(crate::GlrMaskError::Serialization(
                "truncated dynamic v12 transfer payload header".to_owned(),
            ));
        }
        let alternative_count = usize::try_from(u32::from_le_bytes(
            payload[0..4]
                .try_into()
                .expect("dynamic v12 alternative count has fixed width"),
        ))
        .expect("u32 alternative count fits usize on supported platforms");
        if alternative_count == 0 {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic v12 transfer artifact has no alternatives".to_owned(),
            ));
        }
        let flags = u32::from_le_bytes(
            payload[4..8]
                .try_into()
                .expect("dynamic v12 transfer flags have fixed width"),
        );
        if flags != 0 {
            return Err(crate::GlrMaskError::Serialization(
                "unsupported dynamic v12 transfer flags".to_owned(),
            ));
        }
        let descriptor_bytes = alternative_count
            .checked_mul(DYNAMIC_TRANSFER_V12_ALT_DESCRIPTOR_LEN)
            .ok_or_else(|| {
                crate::GlrMaskError::Serialization(
                    "dynamic v12 transfer descriptor size overflow".to_owned(),
                )
            })?;
        let descriptor_end = payload_start
            .checked_add(DYNAMIC_TRANSFER_V12_PAYLOAD_HEADER_LEN)
            .and_then(|value| value.checked_add(descriptor_bytes))
            .ok_or_else(|| {
                crate::GlrMaskError::Serialization(
                    "dynamic v12 transfer descriptor range overflow".to_owned(),
                )
            })?;
        if descriptor_end > backing.len() {
            return Err(crate::GlrMaskError::Serialization(
                "truncated dynamic v12 transfer descriptors".to_owned(),
            ));
        }

        let mut descriptors = Vec::<[usize; 6]>::with_capacity(alternative_count);
        let descriptor_start = payload_start + DYNAMIC_TRANSFER_V12_PAYLOAD_HEADER_LEN;
        for index in 0..alternative_count {
            let mut pos = descriptor_start + index * DYNAMIC_TRANSFER_V12_ALT_DESCRIPTOR_LEN;
            let mut lengths = [0usize; 6];
            for length in &mut lengths {
                let raw = u64::from_le_bytes(
                    backing[pos..pos + 8]
                        .try_into()
                        .expect("validated v12 descriptor has fixed-width field"),
                );
                *length = usize::try_from(raw).map_err(|_| {
                    crate::GlrMaskError::Serialization(
                        "dynamic v12 transfer section length does not fit platform".to_owned(),
                    )
                })?;
                pos += 8;
            }
            if lengths[0] == 0 || lengths[1] == 0 || lengths[4] == 0 {
                return Err(crate::GlrMaskError::Serialization(
                    "dynamic v12 transfer has an empty required section".to_owned(),
                ));
            }
            descriptors.push(lengths);
        }
        let expected_end = descriptors.iter().try_fold(descriptor_end, |cursor, lengths| {
            lengths
                .iter()
                .try_fold(cursor, |cursor, &length| cursor.checked_add(length))
        });
        if expected_end != Some(backing.len()) {
            return Err(crate::GlrMaskError::Serialization(
                "invalid dynamic v12 transfer section lengths".to_owned(),
            ));
        }
        let framing_ms = framing_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let decode_started = profile.then(std::time::Instant::now);
        let token_bytes = vocab.entries_arc();
        let mut cursor = descriptor_end;
        let mut alternatives = Vec::with_capacity(alternative_count);
        for lengths in descriptors {
            let section = |cursor: &mut usize, length: usize| -> std::ops::Range<usize> {
                let start = *cursor;
                *cursor += length;
                start..*cursor
            };
            let table_range = section(&mut cursor, lengths[0]);
            let tokenizer_range = section(&mut cursor, lengths[1]);
            let terminal_exprs_range = section(&mut cursor, lengths[2]);
            let recursive_range = section(&mut cursor, lengths[3]);
            let metadata_range = section(&mut cursor, lengths[4]);
            let virtual_residual_range = section(&mut cursor, lengths[5]);

            let metadata_started = profile.then(std::time::Instant::now);
            let (metadata, prepared_master_proofs, virtual_residual_master_slice_artifacts) = if version == DYNAMIC_TRANSFER_VERSION {
                let wire: DynamicConstraintTransferMetadataV13 =
                    bincode::deserialize(&backing[metadata_range]).map_err(|err| {
                        crate::GlrMaskError::Serialization(format!(
                            "invalid dynamic v13 transfer metadata: {err}"
                        ))
                    })?;
                (
                    wire.base,
                    wire.prepared_master_proofs,
                    wire.virtual_residual_master_slice_artifacts,
                )
            } else {
                let metadata: DynamicConstraintTransferMetadataV11 =
                    bincode::deserialize(&backing[metadata_range]).map_err(|err| {
                        crate::GlrMaskError::Serialization(format!(
                            "invalid dynamic v12 transfer metadata: {err}"
                        ))
                    })?;
                (
                    metadata,
                    crate::runtime::PreparedMasterProofArtifact::default(),
                    Vec::new(),
                )
            };
            if profile {
                eprintln!(
                    "[glrmask/profile][dynamic_transfer_v12_metadata] version={} projected_terminal_quotients_prepared={} projected_terminal_quotients={} prepared_master_proofs={}",
                    version,
                    metadata.projected_terminal_quotients_prepared,
                    metadata.projected_terminal_quotients.len(),
                    !prepared_master_proofs.is_empty(),
                );
            }
            let metadata_ms = metadata_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

            if !recursive_range.is_empty() {
                let mut inner = Constraint::load_with_vocab(&backing[recursive_range], vocab)?;
                inner
                    .tokenizer
                    .restore_virtual_residual_master_slice_artifacts(
                        virtual_residual_master_slice_artifacts,
                    )
                    .map_err(crate::GlrMaskError::Serialization)?;
                if metadata.projected_terminal_quotients_prepared {
                    Self::restore_projected_terminal_quotients(
                        &mut inner,
                        metadata.projected_terminal_quotients,
                    )?;
                }
                if !prepared_master_proofs.is_empty() {
                    let source_state_count = inner.tokenizer.num_states() as usize;
                    inner
                        .dynamic_mask_vocab
                        .restore_prepared_master_proof_artifact(
                            prepared_master_proofs,
                            source_state_count,
                        )
                        .map_err(crate::GlrMaskError::Serialization)?;
                }
                inner.boundary_trigger = metadata.boundary_trigger.into_trigger();
                alternatives.push(Self {
                    inner,
                    alternatives: Vec::new(),
                    composition_grammars: vec![None],
                    external_vocab_artifact_cache: None,
                });
                continue;
            }

            let table_start = table_range.start;
            let tokenizer_start = tokenizer_range.start;
            let table_tokenizer_started = profile.then(std::time::Instant::now);
            let ((table_result, tokenizer_result), ()) = rayon::join(
                || {
                    rayon::join(
                        || {
                            crate::compiler::glr::table::artifact_serde::from_compact_bytes_deferred_backed(
                                &backing[table_range],
                                Arc::clone(&backing),
                                table_start,
                            )
                        },
                        || {
                            crate::automata::lexer::tokenizer::artifact_serde::from_fast_bytes_backed(
                                &backing[tokenizer_range],
                                Arc::clone(&backing),
                                tokenizer_start,
                            )
                        },
                    )
                },
                || (),
            );
            let decoded_table = table_result.map_err(crate::GlrMaskError::Serialization)?;
            let mut tokenizer = tokenizer_result.map_err(crate::GlrMaskError::Serialization)?;
            let table_tokenizer_ms = table_tokenizer_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

            let deferred_terminal_exprs = (!terminal_exprs_range.is_empty()).then(|| {
                crate::runtime::DeferredTerminalExprBytes::CompressedBacked {
                    backing: Arc::clone(&backing),
                    start: terminal_exprs_range.start,
                    len: terminal_exprs_range.len(),
                }
            });

            // Decode the finite residual projection before rebuilding the source
            // virtual runtimes. SRM3 carries the exact bounded-code oracle used
            // by the worker when it built the finite mask projection. Reusing
            // those oracle bytes keeps the source runtime and sparse projection
            // in the same exact coordinate system.
            let residual_decode_started = profile.then(std::time::Instant::now);
            let mut decoded_residual = if virtual_residual_range.is_empty() {
                None
            } else {
                Some(
                    crate::runtime::serde::decode_static_virtual_residual_mask_wire(
                        &backing[virtual_residual_range.clone()],
                        Arc::clone(&backing),
                    )
                    .map_err(crate::GlrMaskError::Serialization)?,
                )
            };
            let residual_decode_ms = residual_decode_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            let mut terminal_expr_decode_ms = 0.0;
            let mut virtual_runtime_restore_ms = 0.0;
            if !metadata.virtual_runtimes.is_empty() {
                let restore_started = profile.then(std::time::Instant::now);
                let restore_result = if let Some(residual) = decoded_residual.as_ref() {
                    tokenizer.restore_compiled_static_residual_runtimes(
                        &metadata.virtual_runtimes,
                        residual.projections(),
                    )
                } else {
                    let expr_started = profile.then(std::time::Instant::now);
                    let expressions = deferred_terminal_exprs
                        .as_ref()
                        .ok_or_else(|| {
                            crate::GlrMaskError::Serialization(
                                "dynamic v12 virtual runtime metadata has no terminal expressions"
                                    .to_owned(),
                            )
                        })?
                        .decode_exprs()
                        .map_err(crate::GlrMaskError::Serialization)?;
                    terminal_expr_decode_ms = expr_started
                        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                    let can_use_compiled_dynamic_residual = metadata
                        .virtual_runtimes
                        .iter()
                        .all(|entry| {
                            entry.kind
                                == crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeKind::ResidualExpr
                        });
                    if can_use_compiled_dynamic_residual {
                        tokenizer.restore_compiled_dynamic_residual_runtimes(
                            &expressions,
                            &metadata.virtual_runtimes,
                            &metadata.residual_runtime_oracles,
                            &virtual_residual_master_slice_artifacts,
                        )
                    } else {
                        tokenizer
                            .restore_terminal_exprs_with_virtual_runtime_metadata_and_oracles_preserving_coordinates(
                                Some(expressions),
                                &metadata.virtual_runtimes,
                                &metadata.residual_runtime_oracles,
                                false,
                                false,
                            )
                    }
                };
                restore_result.map_err(crate::GlrMaskError::Serialization)?;
                virtual_runtime_restore_ms = restore_started
                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            }
            tokenizer
                .restore_virtual_residual_master_slice_artifacts(
                    virtual_residual_master_slice_artifacts,
                )
                .map_err(crate::GlrMaskError::Serialization)?;

            let assemble_started = profile.then(std::time::Instant::now);
            let dynamic_mask_vocab = Self::dynamic_vocab_from_transfer_artifact(
                metadata.grammar_quotient_vocab.clone(),
                vocab,
            )?;
            let mut inner = Self::constraint_from_payload_v2_with_dynamic_vocab(
                DynamicConstraintPayloadV2 {
                    v1: DynamicConstraintPayloadV1 {
                        table: decoded_table.table,
                        terminal_display_names: metadata.terminal_display_names,
                        tokenizer,
                        ignore_terminal: metadata.ignore_terminal,
                        direct_regular_automaton: metadata.direct_regular_automaton,
                        token_bytes: Arc::clone(&token_bytes),
                        ignore_expr: metadata.ignore_expr,
                        terminal_exprs: None,
                    },
                    special_token_terminals: metadata.special_token_terminals,
                },
                dynamic_mask_vocab,
            );
            if let Some(mask_tokenizer) = metadata.mask_tokenizer {
                inner.dynamic_mask_vocab.set_mask_tokenizer_quotient(
                    mask_tokenizer.0,
                    metadata.full_to_mask_state,
                );
            }
            Self::restore_terminal_observation_classes(
                &mut inner,
                metadata.terminal_observation_classes,
            )?;
            if metadata.projected_terminal_quotients_prepared {
                Self::restore_projected_terminal_quotients(
                    &mut inner,
                    metadata.projected_terminal_quotients,
                )?;
            }
            if !prepared_master_proofs.is_empty() {
                let source_state_count = inner.tokenizer.num_states() as usize;
                inner
                    .dynamic_mask_vocab
                    .restore_prepared_master_proof_artifact(
                        prepared_master_proofs,
                        source_state_count,
                    )
                    .map_err(crate::GlrMaskError::Serialization)?;
            }
            inner.boundary_trigger = metadata.boundary_trigger.into_trigger();
            inner.deferred_table_rules_blob = decoded_table.deferred_rules;
            inner.deferred_table_rules = std::sync::OnceLock::new();
            if inner.tokenizer.terminal_exprs().is_none() {
                inner.deferred_terminal_exprs_blob = deferred_terminal_exprs;
                inner.deferred_terminal_exprs = std::sync::OnceLock::new();
            }

            if let Some(decoded_residual) = decoded_residual.take() {
                let srm_started = profile.then(std::time::Instant::now);
                decoded_residual.restore_projections(&mut inner)?;
                if profile {
                    eprintln!(
                        "[glrmask/profile][dynamic_transfer_v12_srm] restore_ms={:.3} bytes={}",
                        srm_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0),
                        virtual_residual_range.len(),
                    );
                }
            }
            let assemble_ms = assemble_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            let rebuild_started = profile.then(std::time::Instant::now);
            inner.rebuild_dynamic_runtime_caches();
            let rebuild_ms = rebuild_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            if profile {
                eprintln!(
                    "[glrmask/profile][dynamic_transfer_v12_alt] metadata_ms={:.3} table_tokenizer_ms={:.3} residual_decode_ms={:.3} terminal_expr_decode_ms={:.3} virtual_runtime_restore_ms={:.3} assemble_ms={:.3} rebuild_ms={:.3}",
                    metadata_ms,
                    table_tokenizer_ms,
                    residual_decode_ms,
                    terminal_expr_decode_ms,
                    virtual_runtime_restore_ms,
                    assemble_ms,
                    rebuild_ms,
                );
            }
            alternatives.push(Self {
                inner,
                alternatives: Vec::new(),
                composition_grammars: vec![None],
                external_vocab_artifact_cache: None,
            });
        }
        let payload_decode_ms = decode_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let mut loaded = Self::from_alternatives(alternatives);
        // Current transfer bytes already are the canonical save artifact.
        // Retain the same backing allocation so a load->save round trip does
        // not re-run any serializer/finalizer work.
        if version == DYNAMIC_TRANSFER_VERSION {
            loaded.external_vocab_artifact_cache = Some(Arc::clone(&backing));
        }
        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][dynamic_transfer_load] version={} bytes={} backing_ms={:.3} framing_ms={:.3} payload_decode_ms={:.3} finalize_ms={:.3} total_ms={:.3}",
                version,
                bytes.len(),
                backing_ms,
                framing_ms,
                payload_decode_ms,
                0.0,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(loaded)
    }

    fn migrate_legacy_v10_payload(
        payload: LegacyDynamicConstraintPayloadV10V3,
    ) -> DynamicConstraintPayloadV3 {
        DynamicConstraintPayloadV3 {
            alternatives: payload
                .alternatives
                .into_iter()
                .map(|alternative| DynamicConstraintPayloadV2 {
                    v1: DynamicConstraintPayloadV1 {
                        table: alternative.v1.table,
                        terminal_display_names: alternative.v1.terminal_display_names,
                        tokenizer: alternative.v1.tokenizer,
                        ignore_terminal: alternative.v1.ignore_terminal,
                        direct_regular_automaton: alternative.v1.direct_regular_automaton,
                        token_bytes: alternative.v1.token_bytes,
                        ignore_expr: None,
                        terminal_exprs: None,
                    },
                    special_token_terminals: alternative.special_token_terminals,
                })
                .collect(),
        }
    }

    fn migrate_legacy_v11_payload(
        payload: LegacyDynamicConstraintPayloadV11V3,
    ) -> DynamicConstraintPayloadV3 {
        DynamicConstraintPayloadV3 {
            alternatives: payload
                .alternatives
                .into_iter()
                .map(|alternative| DynamicConstraintPayloadV2 {
                    v1: DynamicConstraintPayloadV1 {
                        table: alternative.v1.table,
                        terminal_display_names: alternative.v1.terminal_display_names,
                        tokenizer: alternative.v1.tokenizer,
                        ignore_terminal: alternative.v1.ignore_terminal,
                        direct_regular_automaton: alternative.v1.direct_regular_automaton,
                        token_bytes: alternative.v1.token_bytes,
                        ignore_expr: None,
                        terminal_exprs: None,
                    },
                    special_token_terminals: alternative.special_token_terminals,
                })
                .collect(),
        }
    }

    fn migrate_legacy_v12_payload(
        payload: LegacyDynamicConstraintPayloadV12V3,
    ) -> DynamicConstraintPayloadV3 {
        DynamicConstraintPayloadV3 {
            alternatives: payload
                .alternatives
                .into_iter()
                .map(|alternative| DynamicConstraintPayloadV2 {
                    v1: DynamicConstraintPayloadV1 {
                        table: alternative.v1.table,
                        terminal_display_names: alternative.v1.terminal_display_names,
                        tokenizer: alternative.v1.tokenizer,
                        ignore_terminal: alternative.v1.ignore_terminal,
                        direct_regular_automaton: alternative.v1.direct_regular_automaton,
                        token_bytes: alternative.v1.token_bytes,
                        ignore_expr: alternative.v1.ignore_expr,
                        terminal_exprs: None,
                    },
                    special_token_terminals: alternative.special_token_terminals,
                })
                .collect(),
        }
    }

    fn from_payload_v3(payload: DynamicConstraintPayloadV3) -> crate::Result<Self> {
        let mut alternatives = payload
            .alternatives
            .into_iter()
            .map(Self::from_payload_v2)
            .collect::<Vec<_>>();
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(std::mem::take(&mut alternatives)))
    }

    fn from_payload_v4(
        payload: DynamicConstraintPayloadV4,
        allow_legacy_exact_dead_residual_roots: bool,
    ) -> crate::Result<Self> {
        let mut alternatives = Vec::with_capacity(payload.alternatives.len());
        for mut alternative in payload.alternatives {
            let exprs = alternative.v2.v1.terminal_exprs.clone();
            alternative
                .v2
                .v1
                .tokenizer
                .restore_terminal_exprs_with_virtual_runtime_metadata(
                    exprs,
                    &alternative.virtual_runtimes,
                    allow_legacy_exact_dead_residual_roots,
                )
                .map_err(crate::GlrMaskError::Serialization)?;
            let mut constraint = Self::from_payload_v2(alternative.v2);
            constraint.inner.late_grammar_slots = alternative.late_grammar_slots;
            alternatives.push(constraint);
        }
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(alternatives))
    }

    fn from_payload_v5(
        payload: DynamicConstraintPayloadV5,
        allow_legacy_exact_dead_residual_roots: bool,
    ) -> crate::Result<Self> {
        let mut alternatives = Vec::with_capacity(payload.alternatives.len());
        for alternative in payload.alternatives {
            let DynamicConstraintPayloadV5Alternative {
                mut base,
                terminal_observation_classes,
            } = alternative;
            let exprs = base.v2.v1.terminal_exprs.clone();
            base.v2
                .v1
                .tokenizer
                .restore_terminal_exprs_with_virtual_runtime_metadata(
                    exprs,
                    &base.virtual_runtimes,
                    allow_legacy_exact_dead_residual_roots,
                )
                .map_err(crate::GlrMaskError::Serialization)?;
            let mut inner = Self::constraint_from_payload_v2_with_dynamic_vocab(
                base.v2,
                DynamicMaskVocab::default(),
            );
            inner.late_grammar_slots = base.late_grammar_slots;
            Self::restore_terminal_observation_classes(
                &mut inner,
                terminal_observation_classes,
            )?;
            inner.rebuild_dynamic_runtime_caches();
            alternatives.push(Self {
                inner,
                alternatives: Vec::new(),
                composition_grammars: vec![None],
                external_vocab_artifact_cache: None,
            });
        }
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(alternatives))
    }

    fn from_payload_v6(payload: DynamicConstraintPayloadV6) -> crate::Result<Self> {
        Self::from_payload_v6_with_projected_quotients(payload, None)
    }

    fn from_payload_v6_with_projected_quotients(
        payload: DynamicConstraintPayloadV6,
        projected_terminal_quotients: Option<Vec<Vec<(TerminalID, TerminalProjectedQuotient)>>>,
    ) -> crate::Result<Self> {
        if payload.dynamic_mask_vocab.is_some() {
            let mut alternatives = payload.alternatives.iter();
            if let Some(first) = alternatives.next() {
                let token_bytes = &first.base.v2.v1.token_bytes;
                if alternatives.any(|alternative| alternative.base.v2.v1.token_bytes != *token_bytes) {
                    return Err(crate::GlrMaskError::Serialization(
                        "dynamic artifact shares a vocabulary index across alternatives with different token bytes"
                            .to_owned(),
                    ));
                }
            }
        }
        let mut shared_dynamic_vocab = payload
            .dynamic_mask_vocab
            .map(DynamicMaskVocab::from_artifact)
            .transpose()
            .map_err(crate::GlrMaskError::Serialization)?;
        if let (Some(vocab), Some(first)) = (&shared_dynamic_vocab, payload.alternatives.first()) {
            if !vocab.matches_token_bytes_exact(&first.base.v2.v1.token_bytes) {
                return Err(crate::GlrMaskError::Serialization(
                    "dynamic artifact vocabulary index does not match serialized token bytes"
                        .to_owned(),
                ));
            }
        }
        if let (Some(vocab), Some(first)) = (&mut shared_dynamic_vocab, payload.alternatives.first()) {
            vocab.restore_root_layout_metadata_from_token_bytes(&first.base.v2.v1.token_bytes);
        }
        let projected_terminal_quotients_prepared = projected_terminal_quotients.is_some();
        let projected_terminal_quotients = projected_terminal_quotients
            .unwrap_or_else(|| vec![Vec::new(); payload.alternatives.len()]);
        if projected_terminal_quotients.len() != payload.alternatives.len() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic artifact projected-terminal quotient rows do not align with alternatives"
                    .to_owned(),
            ));
        }
        let mut alternatives = Vec::with_capacity(payload.alternatives.len());
        for (alternative, projected_terminal_quotients) in payload
            .alternatives
            .into_iter()
            .zip(projected_terminal_quotients)
        {
            let DynamicConstraintPayloadV5Alternative {
                mut base,
                terminal_observation_classes,
            } = alternative;
            let exprs = base.v2.v1.terminal_exprs.clone();
            base.v2
                .v1
                .tokenizer
                .restore_terminal_exprs_with_virtual_runtime_metadata(
                    exprs,
                    &base.virtual_runtimes,
                    false,
                )
                .map_err(crate::GlrMaskError::Serialization)?;
            let dynamic_mask_vocab = shared_dynamic_vocab
                .as_ref()
                .map(|shared| {
                    let mut fresh = shared.fresh_runtime_instance();
                    // The shared artifact may carry immutable constraint-local
                    // lexer metadata (notably the exact mask-only deterministic
                    // tokenizer for a single-alternative dynamic constraint).
                    // Preserve it while still recreating all mutable runtime
                    // caches on the fresh instance.
                    fresh.inherit_dynamic_lexer_metadata_from(shared);
                    fresh
                })
                .unwrap_or_default();
            let mut inner = Self::constraint_from_payload_v2_with_dynamic_vocab(
                base.v2,
                dynamic_mask_vocab,
            );
            inner.late_grammar_slots = base.late_grammar_slots;
            Self::restore_terminal_observation_classes(
                &mut inner,
                terminal_observation_classes,
            )?;
            if projected_terminal_quotients_prepared {
                Self::restore_projected_terminal_quotients(
                    &mut inner,
                    projected_terminal_quotients,
                )?;
            }
            inner.rebuild_dynamic_runtime_caches();
            alternatives.push(Self {
                inner,
                alternatives: Vec::new(),
                composition_grammars: vec![None],
                external_vocab_artifact_cache: None,
            });
        }
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(alternatives))
    }

    fn from_payload_v7(payload: DynamicConstraintPayloadV7) -> crate::Result<Self> {
        let mut metadata = Vec::with_capacity(payload.alternatives.len());
        let mut projected_terminal_quotients = Vec::with_capacity(payload.alternatives.len());
        let mut alternatives = Vec::with_capacity(payload.alternatives.len());
        for alternative in payload.alternatives {
            metadata.push((
                alternative.boundary_trigger,
                alternative.recursive_constraint_artifact,
            ));
            projected_terminal_quotients.push(alternative.projected_terminal_quotients);
            alternatives.push(alternative.base);
        }
        let base = DynamicConstraintPayloadV6 {
            alternatives,
            dynamic_mask_vocab: payload.dynamic_mask_vocab,
        };
        let mut constraint = Self::from_payload_v6_with_projected_quotients(
            base,
            Some(projected_terminal_quotients),
        )?;
        if constraint.constraints_mut().count() != metadata.len() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic artifact alternative metadata count mismatch".to_owned(),
            ));
        }
        for (inner, (trigger, recursive_artifact)) in
            constraint.constraints_mut().zip(metadata)
        {
            if let Some(recursive_artifact) = recursive_artifact {
                let projected_terminal_quotients_prepared = inner
                    .dynamic_mask_vocab
                    .projected_terminal_quotients_prepared();
                let projected_terminal_quotients = inner
                    .dynamic_mask_vocab
                    .projected_terminal_quotients_for_artifact();
                let loaded = Constraint::load(recursive_artifact)?;
                let same_vocab = loaded.token_bytes_count() == inner.token_bytes_count()
                    && inner.token_bytes.iter().all(|(&token_id, bytes)| {
                        loaded.token_bytes_for_id(token_id) == Some(bytes.as_slice())
                    });
                if !same_vocab {
                    return Err(crate::GlrMaskError::Serialization(
                        "recursive dynamic alternative artifact targets a different vocabulary"
                            .to_owned(),
                    ));
                }
                *inner = loaded;
                if projected_terminal_quotients_prepared {
                    Self::restore_projected_terminal_quotients(
                        inner,
                        projected_terminal_quotients,
                    )?;
                }
            }
            inner.boundary_trigger = trigger.into_trigger();
        }
        Ok(constraint)
    }

    fn from_legacy_payload_v14_main(
        payload: LegacyDynamicConstraintPayloadV14Main,
    ) -> crate::Result<Self> {
        let mut alternatives = payload
            .alternatives
            .into_iter()
            .map(|alternative| {
                let mut constraint = Self::from_payload_v2(alternative.constraint);
                constraint.inner.late_grammar_slots = alternative.late_grammar_slots;
                constraint
            })
            .collect::<Vec<_>>();
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(std::mem::take(&mut alternatives)))
    }

    fn from_legacy_payload_v15_residual(
        payload: LegacyDynamicConstraintPayloadV15Residual,
    ) -> crate::Result<Self> {
        let mut alternatives = Vec::with_capacity(payload.alternatives.len());
        for mut alternative in payload.alternatives {
            let exprs = alternative.v2.v1.terminal_exprs.clone();
            alternative
                .v2
                .v1
                .tokenizer
                .restore_terminal_exprs_with_virtual_runtime_metadata(
                    exprs,
                    &alternative.virtual_runtimes,
                    false,
                )
                .map_err(crate::GlrMaskError::Serialization)?;
            alternatives.push(Self::from_payload_v2(alternative.v2));
        }
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(alternatives))
    }


    fn from_payload_v3_with_vocab(
        payload: DynamicConstraintPayloadV3,
        vocab: &Vocab,
    ) -> crate::Result<Self> {
        let mut alternatives = payload
            .alternatives
            .into_iter()
            .map(|alternative| {
                Self::from_payload_v2_with_dynamic_vocab(
                    alternative,
                    crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(
                        vocab,
                    ),
                )
            })
            .collect::<Vec<_>>();
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(std::mem::take(&mut alternatives)))
    }

    fn save_template_alternatives(&self, external_vocab: bool) -> Vec<u8> {
        let constraints = std::iter::once(&self.inner).chain(&self.alternatives);
        let count = self.alternatives.len().checked_add(1).expect("dynamic alternative count");
        let mut payload = Vec::new();
        payload.extend_from_slice(&u32::try_from(count).expect("dynamic alternative count fits u32").to_le_bytes());
        for constraint in constraints {
            assert!(constraint.has_template_parser(), "mixed LR/template dynamic artifacts are unsupported");
            let bytes = if external_vocab { constraint.save_template_with_external_vocab() } else { constraint.save() };
            payload.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            payload.extend_from_slice(&bytes);
        }
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(if external_vocab { &DYNAMIC_TRANSFER_MAGIC } else { &DYNAMIC_CONSTRAINT_MAGIC });
        bytes.extend_from_slice(&(if external_vocab { TEMPLATE_DYNAMIC_TRANSFER_VERSION } else { TEMPLATE_DYNAMIC_CONSTRAINT_VERSION }).to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes
    }

    fn load_template_alternatives(payload: &[u8], vocab: Option<&Vocab>) -> crate::Result<Self> {
        let error = |message: &str| crate::GlrMaskError::Serialization(message.into());
        let count = payload.get(..4).ok_or_else(|| error("truncated template dynamic count"))?;
        let count = u32::from_le_bytes(count.try_into().unwrap()) as usize;
        if count == 0 || count > payload.len().saturating_sub(4) / 26 {
            return Err(error("invalid template dynamic alternative count"));
        }
        let mut offset = 4usize;
        let mut constraints = Vec::new();
        for _ in 0..count {
            let header_end = offset.checked_add(8).ok_or_else(|| error("template dynamic offset overflow"))?;
            let length = payload.get(offset..header_end).ok_or_else(|| error("truncated template dynamic descriptor"))?;
            let length = usize::try_from(u64::from_le_bytes(length.try_into().unwrap()))
                .map_err(|_| error("template dynamic length does not fit this platform"))?;
            let end = header_end.checked_add(length).ok_or_else(|| error("template dynamic length overflow"))?;
            let body = payload.get(header_end..end).ok_or_else(|| error("truncated template dynamic alternative"))?;
            let constraint = match vocab { Some(vocab) => Constraint::load_with_vocab(body, vocab)?, None => Constraint::load(body)? };
            if !constraint.has_template_parser() || !constraint.uses_dynamic_runtime() {
                return Err(error("template dynamic alternative is not a table-free dynamic constraint"));
            }
            constraints.push(constraint);
            offset = end;
        }
        if offset != payload.len() { return Err(error("trailing template dynamic bytes")); }
        Ok(Self::from_constraints(constraints))
    }

    /// Serialize this dynamic constraint to a versioned binary artifact.
    pub fn save(&self) -> Vec<u8> {
        if self.inner.has_template_parser() || self.alternatives.iter().any(Constraint::has_template_parser) {
            return self.save_template_alternatives(false);
        }
        if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_ARTIFACT").is_some() {
            for (index, constraint) in std::iter::once(&self.inner)
                .chain(self.alternatives.iter())
                .enumerate()
            {
                let (finalizer_bits, future_bits, max_finalizers, max_futures) =
                    constraint.tokenizer.artifact_metadata_stats();
                eprintln!(
                    "[glrmask/profile][dynamic_artifact_metadata] alternative={} states={} terminals={} finalizer_bits={} future_bits={} max_finalizers={} max_futures={}",
                    index,
                    constraint.tokenizer.num_states(),
                    constraint.tokenizer.num_terminals(),
                    finalizer_bits,
                    future_bits,
                    max_finalizers,
                    max_futures,
                );
            }
        }
        let constraints = std::iter::once(&self.inner)
            .chain(self.alternatives.iter())
            .collect::<Vec<_>>();
        let share_vocab = constraints.first().is_none_or(|first| {
            constraints
                .iter()
                .skip(1)
                .all(|constraint| constraint.token_bytes == first.token_bytes)
        });
        // The self-contained wire has one shared dynamic-vocabulary slot.
        // Ordinary alternatives can share the model-vocabulary trie, but
        // grammar-quotiented alternatives may have different partitions. In
        // that case omit the shared accelerator and let load lazily rebuild the
        // full vocabulary per alternative; semantics stay exact, while the
        // external-vocab transfer format below can retain each quotient
        // independently.
        let has_per_alternative_quotient = constraints.len() > 1
            && constraints
                .iter()
                .any(|constraint| constraint.dynamic_mask_vocab.is_grammar_quotiented());
        let dynamic_mask_vocab = (share_vocab && !has_per_alternative_quotient)
            .then(|| {
                if constraints.len() == 1 {
                    constraints[0].dynamic_mask_vocab.to_artifact()
                } else {
                    constraints
                        .iter()
                        .find_map(|constraint| constraint.dynamic_mask_vocab.to_vocab_artifact())
                }
            })
            .flatten();
        let payload = DynamicConstraintPayloadV7 {
            alternatives: constraints
                .into_iter()
                .map(Self::payload_v7_for_constraint)
                .collect(),
            dynamic_mask_vocab,
        };
        let payload = bincode::serialize(&payload)
            .expect("DynamicConstraint serialization should succeed");
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes
    }

    pub(crate) fn dynamic_vocab_from_transfer_artifact(
        artifact: Option<crate::runtime::DynamicMaskVocabArtifact>,
        vocab: &Vocab,
    ) -> crate::Result<DynamicMaskVocab> {
        let Some(artifact) = artifact else {
            return Ok(
                crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(vocab),
            );
        };
        let profile = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_LOAD").is_some();
        let from_started = profile.then(std::time::Instant::now);
        let full_vocab_template =
            crate::compiler::constraint_possible_matches::prepared_runtime_dynamic_vocab_for_vocab(
                vocab,
            );
        let quotient = DynamicMaskVocab::from_external_vocab_artifact(
            artifact,
            full_vocab_template.as_ref(),
        )
        .map_err(crate::GlrMaskError::Serialization)?;
        let from_ms = from_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let validate_started = profile.then(std::time::Instant::now);
        if quotient.source_vocab_digest()
            != Some(crate::compiler::compile::vocab_content_digest(vocab))
        {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic transfer grammar-quotient vocabulary does not match supplied vocabulary"
                    .to_owned(),
            ));
        }
        let validate_ms = validate_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile {
            eprintln!(
                "[glrmask/profile][dynamic_transfer_quotient_assemble] from_artifact_ms={from_ms:.3} validate_ms={validate_ms:.3}",
            );
        }
        Ok(quotient)
    }

    /// Load either a self-contained artifact or a transfer artifact whose
    /// vocabulary bytes are supplied out of band.
    pub(crate) fn load_with_vocab(bytes: &[u8], vocab: &Vocab) -> crate::Result<Self> {
        if !bytes.starts_with(&DYNAMIC_TRANSFER_MAGIC) {
            let mut loaded = Self::load(bytes)?;
            loaded
                .bind_vocab_exact(vocab)
                .map_err(crate::GlrMaskError::Serialization)?;
            return Ok(loaded);
        }
        if bytes.len() < DYNAMIC_CONSTRAINT_HEADER_LEN {
            return Err(crate::GlrMaskError::Serialization(
                "invalid dynamic transfer artifact header".to_owned(),
            ));
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if !matches!(
            version,
            DYNAMIC_TRANSFER_VERSION_V1
                | DYNAMIC_TRANSFER_VERSION_V2
                | DYNAMIC_TRANSFER_VERSION_V3
                | DYNAMIC_TRANSFER_VERSION_V4
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V5
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V6
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V7
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V9
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V10
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V11
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V12
                | DYNAMIC_TRANSFER_VERSION
                | TEMPLATE_DYNAMIC_TRANSFER_VERSION
        ) {
            return Err(crate::GlrMaskError::Serialization(format!(
                "unsupported dynamic transfer artifact version {version}",
            )));
        }
        let payload_len = usize::try_from(u64::from_le_bytes(
            bytes[10..18]
                .try_into()
                .expect("dynamic transfer header has fixed width"),
        ))
        .map_err(|_| {
            crate::GlrMaskError::Serialization(
                "dynamic transfer payload length does not fit this platform".to_owned(),
            )
        })?;
        if bytes.len() != DYNAMIC_CONSTRAINT_HEADER_LEN.saturating_add(payload_len) {
            return Err(crate::GlrMaskError::Serialization(
                "invalid dynamic transfer artifact payload length".to_owned(),
            ));
        }
        if version == TEMPLATE_DYNAMIC_TRANSFER_VERSION {
            let mut loaded = Self::load_template_alternatives(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..], Some(vocab))?;
            loaded.external_vocab_artifact_cache = Some(Arc::new(bytes.to_vec()));
            return Ok(loaded);
        }
        if matches!(
            version,
            LEGACY_DYNAMIC_TRANSFER_VERSION_V12 | DYNAMIC_TRANSFER_VERSION
        ) {
            return Self::load_transfer_v12(bytes, vocab);
        }
        if version == LEGACY_DYNAMIC_TRANSFER_VERSION_V11 {
            return Self::load_transfer_v11(bytes, vocab);
        }
        let load_profile = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_LOAD").is_some();
        let load_total_started = load_profile.then(std::time::Instant::now);
        let payload_decode_started = load_profile.then(std::time::Instant::now);
        let exact_virtual_metadata = matches!(
            version,
            LEGACY_DYNAMIC_TRANSFER_VERSION_V5
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V6
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V7
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V9
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V10
        );
        let allow_legacy_exact_dead_residual_roots =
            version == LEGACY_DYNAMIC_TRANSFER_VERSION_V5;
        let payload_alternatives: Vec<(
            DynamicConstraintTransferAlternativeV2,
            Vec<(TerminalID, Vec<u32>)>,
            Vec<(TerminalID, TerminalProjectedQuotient)>,
            bool,
            DynamicBoundaryTriggerWire,
            Option<Vec<u8>>,
            Option<crate::compiler::glr::table::artifact_serde::DeferredRuleBytes>,
            Option<Vec<u8>>,
        )> = match version {
            LEGACY_DYNAMIC_TRANSFER_VERSION_V10 => {
                bincode::deserialize::<DynamicConstraintTransferPayloadV5>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?
                .alternatives
                .into_iter()
                .map(|alternative| {
                    let CompactTransferTable { table, deferred_rules } = alternative.table;
                    (
                        DynamicConstraintTransferAlternativeV2 {
                            v1: DynamicConstraintTransferAlternativeV1 {
                                table,
                                terminal_display_names: alternative.terminal_display_names,
                                tokenizer: alternative.tokenizer.0,
                                ignore_terminal: alternative.ignore_terminal,
                                direct_regular_automaton: alternative.direct_regular_automaton,
                                special_token_terminals: alternative.special_token_terminals,
                                ignore_expr: alternative.ignore_expr,
                                terminal_exprs: None,
                                mask_tokenizer: alternative.mask_tokenizer,
                                full_to_mask_state: alternative.full_to_mask_state,
                            },
                            virtual_runtimes: alternative.virtual_runtimes,
                        },
                        alternative.terminal_observation_classes,
                        alternative.projected_terminal_quotients,
                        true,
                        alternative.boundary_trigger,
                        alternative.recursive_constraint_artifact,
                        deferred_rules,
                        alternative.terminal_exprs_compressed,
                    )
                })
                .collect()
            }
            LEGACY_DYNAMIC_TRANSFER_VERSION_V9 => {
                bincode::deserialize::<DynamicConstraintTransferPayloadV4>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?
                .alternatives
                .into_iter()
                .map(|alternative| {
                    (
                        alternative.base.base,
                        alternative.base.terminal_observation_classes,
                        alternative.projected_terminal_quotients,
                        true,
                        alternative.boundary_trigger,
                        alternative.recursive_constraint_artifact,
                        None,
                        None,
                    )
                })
                .collect()
            }
            LEGACY_DYNAMIC_TRANSFER_VERSION_V7 => {
                bincode::deserialize::<DynamicConstraintTransferPayloadV3>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?
                .alternatives
                .into_iter()
                .map(|alternative| {
                    (
                        alternative.base,
                        alternative.terminal_observation_classes,
                        Vec::new(),
                        false,
                        DynamicBoundaryTriggerWire::None,
                        None,
                        None,
                        None,
                    )
                })
                .collect()
            }
            LEGACY_DYNAMIC_TRANSFER_VERSION_V5 | LEGACY_DYNAMIC_TRANSFER_VERSION_V6 => {
                bincode::deserialize::<DynamicConstraintTransferPayloadV2>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?
                .alternatives
                .into_iter()
                .map(|alternative| {
                    (
                        alternative,
                        Vec::new(),
                        Vec::new(),
                        false,
                        DynamicBoundaryTriggerWire::None,
                        None,
                        None,
                        None,
                    )
                })
                .collect()
            }
            DYNAMIC_TRANSFER_VERSION_V4 => {
                let legacy = bincode::deserialize::<DynamicConstraintTransferPayloadV1>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                legacy
                    .alternatives
                    .into_iter()
                    .map(|v1| {
                        (
                            DynamicConstraintTransferAlternativeV2 {
                                v1,
                                virtual_runtimes: Vec::new(),
                            },
                            Vec::new(),
                            Vec::new(),
                            false,
                            DynamicBoundaryTriggerWire::None,
                            None,
                            None,
                            None,
                        )
                    })
                    .collect()
            }
            DYNAMIC_TRANSFER_VERSION_V3 => {
                let legacy = bincode::deserialize::<LegacyDynamicConstraintTransferPayloadV3>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                legacy.alternatives.into_iter().map(|alternative| {
                    (
                        DynamicConstraintTransferAlternativeV2 {
                            v1: DynamicConstraintTransferAlternativeV1 {
                                table: alternative.table,
                                terminal_display_names: alternative.terminal_display_names,
                                tokenizer: alternative.tokenizer,
                                ignore_terminal: alternative.ignore_terminal,
                                direct_regular_automaton: alternative.direct_regular_automaton,
                                special_token_terminals: alternative.special_token_terminals,
                                ignore_expr: alternative.ignore_expr,
                                terminal_exprs: None,
                                mask_tokenizer: alternative.mask_tokenizer,
                                full_to_mask_state: alternative.full_to_mask_state,
                            },
                            virtual_runtimes: Vec::new(),
                        },
                        Vec::new(),
                        Vec::new(),
                        false,
                        DynamicBoundaryTriggerWire::None,
                        None,
                        None,
                        None,
                    )
                }).collect()
            }
            DYNAMIC_TRANSFER_VERSION_V2 => {
                let legacy = bincode::deserialize::<LegacyDynamicConstraintTransferPayloadV2>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                legacy.alternatives.into_iter().map(|alternative| {
                    (
                        DynamicConstraintTransferAlternativeV2 {
                            v1: DynamicConstraintTransferAlternativeV1 {
                                table: alternative.table,
                                terminal_display_names: alternative.terminal_display_names,
                                tokenizer: alternative.tokenizer,
                                ignore_terminal: alternative.ignore_terminal,
                                direct_regular_automaton: alternative.direct_regular_automaton,
                                special_token_terminals: alternative.special_token_terminals,
                                ignore_expr: alternative.ignore_expr,
                                terminal_exprs: None,
                                mask_tokenizer: None,
                                full_to_mask_state: Vec::new(),
                            },
                            virtual_runtimes: Vec::new(),
                        },
                        Vec::new(),
                        Vec::new(),
                        false,
                        DynamicBoundaryTriggerWire::None,
                        None,
                        None,
                        None,
                    )
                }).collect()
            }
            DYNAMIC_TRANSFER_VERSION_V1 => {
                let legacy = bincode::deserialize::<LegacyDynamicConstraintTransferPayloadV1>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                legacy.alternatives.into_iter().map(|alternative| {
                    (
                        DynamicConstraintTransferAlternativeV2 {
                            v1: DynamicConstraintTransferAlternativeV1 {
                                table: alternative.table,
                                terminal_display_names: alternative.terminal_display_names,
                                tokenizer: alternative.tokenizer,
                                ignore_terminal: alternative.ignore_terminal,
                                direct_regular_automaton: alternative.direct_regular_automaton,
                                special_token_terminals: alternative.special_token_terminals,
                                ignore_expr: None,
                                terminal_exprs: None,
                                mask_tokenizer: None,
                                full_to_mask_state: Vec::new(),
                            },
                            virtual_runtimes: Vec::new(),
                        },
                        Vec::new(),
                        Vec::new(),
                        false,
                        DynamicBoundaryTriggerWire::None,
                        None,
                        None,
                        None,
                    )
                }).collect()
            }
            _ => unreachable!("transfer version was validated above"),
        };
        let payload_decode_ms = payload_decode_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let finalize_started = load_profile.then(std::time::Instant::now);
        let token_bytes = vocab.entries_arc();
        let mut alternatives = payload_alternatives
            .into_iter()
            .map(|(
                mut alternative,
                terminal_observation_classes,
                projected_terminal_quotients,
                projected_terminal_quotients_prepared,
                boundary_trigger,
                recursive_artifact,
                deferred_table_rules,
                deferred_terminal_exprs_compressed,
            )| -> crate::Result<Self> {
                if let Some(recursive_artifact) = recursive_artifact {
                    let mut inner = Constraint::load_with_vocab(recursive_artifact, vocab)?;
                    if projected_terminal_quotients_prepared {
                        Self::restore_projected_terminal_quotients(
                            &mut inner,
                            projected_terminal_quotients,
                        )?;
                    }
                    inner.boundary_trigger = boundary_trigger.into_trigger();
                    return Ok(Self {
                        inner,
                        alternatives: Vec::new(),
                        composition_grammars: vec![None],
                        external_vocab_artifact_cache: None,
                    });
                }
                if exact_virtual_metadata {
                    if alternative.v1.terminal_exprs.is_none()
                        && !alternative.virtual_runtimes.is_empty()
                        && let Some(compressed) = deferred_terminal_exprs_compressed.as_deref()
                    {
                        alternative.v1.terminal_exprs = Some(
                            compressed_terminal_exprs_serde::decode(compressed)
                                .map_err(crate::GlrMaskError::Serialization)?,
                        );
                    }
                    let exprs = alternative.v1.terminal_exprs.clone();
                    alternative
                        .v1
                        .tokenizer
                        .restore_terminal_exprs_with_virtual_runtime_metadata(
                            exprs,
                            &alternative.virtual_runtimes,
                            allow_legacy_exact_dead_residual_roots,
                        )
                        .map_err(crate::GlrMaskError::Serialization)?;
                }
                let alternative = alternative.v1;
                let mut inner = Self::constraint_from_payload_v2_with_dynamic_vocab(
                    DynamicConstraintPayloadV2 {
                        v1: DynamicConstraintPayloadV1 {
                            table: alternative.table,
                            terminal_display_names: alternative.terminal_display_names,
                            tokenizer: alternative.tokenizer,
                            ignore_terminal: alternative.ignore_terminal,
                            direct_regular_automaton: alternative.direct_regular_automaton,
                            token_bytes: Arc::clone(&token_bytes),
                            ignore_expr: alternative.ignore_expr,
                            terminal_exprs: alternative.terminal_exprs,
                        },
                        special_token_terminals: alternative.special_token_terminals,
                    },
                    crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(
                        vocab,
                    ),
                );
                if let Some(mask_tokenizer) = alternative.mask_tokenizer {
                    inner.dynamic_mask_vocab.set_mask_tokenizer_quotient(
                        mask_tokenizer.0,
                        alternative.full_to_mask_state,
                    );
                }
                Self::restore_terminal_observation_classes(
                    &mut inner,
                    terminal_observation_classes,
                )?;
                if projected_terminal_quotients_prepared {
                    Self::restore_projected_terminal_quotients(
                        &mut inner,
                        projected_terminal_quotients,
                    )?;
                }
                inner.boundary_trigger = boundary_trigger.into_trigger();
                inner.deferred_table_rules_blob = deferred_table_rules;
                inner.deferred_table_rules = std::sync::OnceLock::new();
                if inner.tokenizer.terminal_exprs().is_none()
                    && let Some(compressed) = deferred_terminal_exprs_compressed
                {
                    inner.deferred_terminal_exprs_blob = Some(
                        crate::runtime::DeferredTerminalExprBytes::CompressedOwned(Arc::from(
                            compressed.into_boxed_slice(),
                        )),
                    );
                    inner.deferred_terminal_exprs = std::sync::OnceLock::new();
                }
                inner.rebuild_dynamic_runtime_caches();
                Ok(Self {
                    inner,
                    alternatives: Vec::new(),
                    composition_grammars: vec![None],
                    external_vocab_artifact_cache: None,
                })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union transfer artifact has no alternatives".to_owned(),
            ));
        }
        let loaded = Self::from_alternatives(std::mem::take(&mut alternatives));
        if let Some(total_started) = load_total_started {
            eprintln!(
                "[glrmask/profile][dynamic_transfer_load] version={} bytes={} payload_decode_ms={:.3} finalize_ms={:.3} total_ms={:.3}",
                version,
                bytes.len(),
                payload_decode_ms,
                finalize_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(loaded)
    }

    /// Load an artifact produced by [`DynamicConstraint::save`].
    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        if bytes.len() < DYNAMIC_CONSTRAINT_HEADER_LEN
            || !bytes.starts_with(&DYNAMIC_CONSTRAINT_MAGIC)
        {
            return Err(crate::GlrMaskError::Serialization(
                "invalid dynamic constraint artifact header".to_owned(),
            ));
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if !matches!(
            version,
            1 | 2
                | 7
                | 8
                | 9
                | 10
                | 11
                | LEGACY_DYNAMIC_CONSTRAINT_VERSION_V12
                | LEGACY_DYNAMIC_CONSTRAINT_VERSION_V13
                | LEGACY_DYNAMIC_CONSTRAINT_VERSION_V14_MAIN
                | LEGACY_DYNAMIC_CONSTRAINT_VERSION_V15_RESIDUAL
                | LEGACY_DYNAMIC_CONSTRAINT_VERSION_V16
                | LEGACY_DYNAMIC_CONSTRAINT_VERSION_V17
                | LEGACY_DYNAMIC_CONSTRAINT_VERSION_V18
                | DYNAMIC_CONSTRAINT_VERSION
                | TEMPLATE_DYNAMIC_CONSTRAINT_VERSION
        ) {
            return Err(crate::GlrMaskError::Serialization(format!(
                "unsupported dynamic constraint artifact version {version}",
            )));
        }
        let payload_len = usize::try_from(u64::from_le_bytes(
            bytes[10..18]
                .try_into()
                .expect("dynamic constraint header has fixed width"),
        ))
        .map_err(|_| {
            crate::GlrMaskError::Serialization(
                "dynamic constraint payload length does not fit this platform".to_owned(),
            )
        })?;
        if bytes.len() != DYNAMIC_CONSTRAINT_HEADER_LEN.saturating_add(payload_len) {
            return Err(crate::GlrMaskError::Serialization(
                "invalid dynamic constraint artifact payload length".to_owned(),
            ));
        }
        match version {
            TEMPLATE_DYNAMIC_CONSTRAINT_VERSION => Self::load_template_alternatives(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..], None),
            1 => {
                let payload: LegacyDynamicConstraintPayloadV1 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_legacy_payload_v1(payload)
            }
            2 => {
                let payload: LegacyDynamicConstraintPayloadV2 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_legacy_payload_v2(payload)
            }
            7 | 8 => Err(crate::GlrMaskError::Serialization(
                "dynamic constraint artifact contains removed precompiled token programs; rebuild it"
                    .to_owned(),
            )),
            9 => {
                let payload: LegacyDynamicConstraintPayloadV9 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Ok(Self::from_payload_v2(DynamicConstraintPayloadV2 {
                    v1: DynamicConstraintPayloadV1 {
                        table: payload.v1.table,
                        terminal_display_names: payload.v1.terminal_display_names,
                        tokenizer: payload.v1.tokenizer,
                        ignore_terminal: payload.v1.ignore_terminal,
                        direct_regular_automaton: None,
                        token_bytes: payload.v1.token_bytes,
                        ignore_expr: None,
                        terminal_exprs: None,
                    },
                    special_token_terminals: payload.special_token_terminals,
                }))
            }
            10 => {
                let payload: LegacyDynamicConstraintPayloadV10V3 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v3(Self::migrate_legacy_v10_payload(payload))
            }
            11 => {
                let payload: LegacyDynamicConstraintPayloadV11V3 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v3(Self::migrate_legacy_v11_payload(payload))
            }
            LEGACY_DYNAMIC_CONSTRAINT_VERSION_V12 => {
                let payload: LegacyDynamicConstraintPayloadV12V3 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v3(Self::migrate_legacy_v12_payload(payload))
            }
            LEGACY_DYNAMIC_CONSTRAINT_VERSION_V13 => {
                let payload: DynamicConstraintPayloadV3 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v3(payload)
            }
            LEGACY_DYNAMIC_CONSTRAINT_VERSION_V14_MAIN => {
                let payload: LegacyDynamicConstraintPayloadV14Main =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_legacy_payload_v14_main(payload)
            }
            LEGACY_DYNAMIC_CONSTRAINT_VERSION_V15_RESIDUAL => {
                let payload: LegacyDynamicConstraintPayloadV15Residual =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_legacy_payload_v15_residual(payload)
            }
            LEGACY_DYNAMIC_CONSTRAINT_VERSION_V16 => {
                let payload: DynamicConstraintPayloadV4 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v4(payload, false)
            }
            LEGACY_DYNAMIC_CONSTRAINT_VERSION_V17 => {
                let payload: DynamicConstraintPayloadV5 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v5(payload, false)
            }
            LEGACY_DYNAMIC_CONSTRAINT_VERSION_V18 => {
                let payload: DynamicConstraintPayloadV6 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v6(payload)
            }
            DYNAMIC_CONSTRAINT_VERSION => {
                let payload: DynamicConstraintPayloadV7 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v7(payload)
            }
            _ => unreachable!("version was validated above"),
        }
    }

    /// Return the number of `u32` words required for a packed token mask.
    pub fn mask_len(&self) -> usize {
        std::iter::once(&self.inner)
            .chain(self.alternatives.iter())
            .map(Constraint::mask_len)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn max_original_token_id(&self) -> Option<u32> {
        std::iter::once(&self.inner)
            .chain(self.alternatives.iter())
            .filter_map(Constraint::max_original_token_id)
            .max()
    }

    /// Create a fresh state for one generated sequence.
    pub fn start(&self) -> DynamicConstraintState<'_> {
        DynamicConstraintState {
            alternatives: std::iter::once(&self.inner)
                .chain(self.alternatives.iter())
                .map(Constraint::start_dynamic)
                .collect(),
            mask_len: self.mask_len(),
        }
    }
}

/// Mutable per-sequence state for a [`DynamicConstraint`].
#[derive(Clone)]
pub struct DynamicConstraintState<'a> {
    alternatives: Vec<ConstraintState<'a>>,
    mask_len: usize,
}

impl<'a> DynamicConstraintState<'a> {
    fn retain_committing(
        &mut self,
        mut commit: impl FnMut(&mut ConstraintState<'a>) -> Result<(), String>,
    ) -> Result<(), String> {
        self.alternatives.retain_mut(|state| commit(state).is_ok());
        if self.alternatives.is_empty() {
            Err("commit rejected: no valid parser states remain".to_owned())
        } else {
            Ok(())
        }
    }

    /// Advance the state by raw bytes.
    pub fn commit_bytes(&mut self, bytes: &[u8]) -> crate::Result<()> {
        self.commit_bytes_raw(bytes).map_err(crate::Error::State)
    }

    fn commit_bytes_raw(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.retain_committing(|state| state.commit_bytes_raw(bytes))
    }

    /// Advance the state by one model token ID.
    pub fn commit_token(&mut self, token_id: u32) -> crate::Result<()> {
        if self
            .alternatives
            .iter()
            .all(|state| !state.knows_token_id(token_id))
        {
            return Err(crate::Error::State(format!(
                "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
            )));
        }
        self.commit_token_raw(token_id).map_err(crate::Error::State)
    }

    fn commit_token_raw(&mut self, token_id: u32) -> Result<(), String> {
        self.retain_committing(|state| state.commit_token_raw(token_id))
    }

    /// Diagnostic commit profile for the overwhelmingly common single-
    /// alternative dynamic constraint. Multi-alternative dynamic constraints
    /// deliberately keep using the ordinary commit path rather than inventing
    /// an ambiguous aggregate profile.
    #[doc(hidden)]
    pub fn commit_token_profiled(&mut self, token_id: u32) -> Result<CommitProfile, String> {
        let [state] = self.alternatives.as_mut_slice() else {
            return Err(format!(
                "profiled dynamic commit requires exactly one alternative (have {})",
                self.alternatives.len()
            ));
        };
        state.commit_token_profiled(token_id)
    }

    /// Bounded diagnostic mirror of the common state profiler. No timing
    /// benchmark should use this stack-enumerating inspection API.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn commit_token_per_advance(&mut self, token_id: u32) -> Result<
        (Vec<crate::runtime::PerAdvanceEntry>, Vec<(u32, Vec<Vec<u32>>)>, CommitProfile),
        String,
    > {
        let [state] = self.alternatives.as_mut_slice() else {
            return Err("per-advance profiling requires one dynamic alternative".into());
        };
        state.commit_token_per_advance(token_id)
    }

    /// Fill `buf` with the allowed-token mask as a packed bitset.
    pub fn fill_mask(&self, buf: &mut [u32]) {
        assert!(buf.len() >= self.mask_len, "mask buffer is smaller than constraint mask");
        let Some((first, rest)) = self.alternatives.split_first() else {
            buf.fill(0);
            return;
        };
        let profile = dynamic_mask_profile_enabled(first.generation);
        let total_started = profile.then(std::time::Instant::now);
        let first_started = profile.then(std::time::Instant::now);
        first.fill_mask(buf);
        if profile {
            eprintln!(
                "[glrmask/profile][dynamic_mask_union_alt] alt=0 alternatives={} ms={:.3}",
                self.alternatives.len(),
                first_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        if rest.is_empty() {
            if let Some(started) = total_started {
                eprintln!(
                    "[glrmask/profile][dynamic_mask_union] alternatives=1 total_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return;
        }
        let mut scratch = vec![0u32; buf.len()];
        for (index, state) in rest.iter().enumerate() {
            let started = profile.then(std::time::Instant::now);
            state.fill_mask(&mut scratch);
            for (target, source) in buf.iter_mut().zip(&scratch) {
                *target |= *source;
            }
            scratch.fill(0);
            if profile {
                eprintln!(
                    "[glrmask/profile][dynamic_mask_union_alt] alt={} alternatives={} ms={:.3}",
                    index + 1,
                    self.alternatives.len(),
                    started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                );
            }
        }
        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][dynamic_mask_union] alternatives={} total_ms={:.3}",
                self.alternatives.len(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }


    /// Return a forced token sequence when one can be determined.
    pub fn forced(&self) -> Vec<u32> {
        let Some((first, rest)) = self.alternatives.split_first() else {
            return Vec::new();
        };
        let forced = first.forced();
        (!forced.is_empty()
            && rest.iter().all(|state| state.forced() == forced))
            .then_some(forced)
            .unwrap_or_default()
    }

    /// Return whether the committed prefix is currently accepted by the grammar.
    ///
    /// An accepting prefix may still admit additional tokens.
    pub fn is_accepting(&self) -> bool {
        self.alternatives.iter().any(ConstraintState::is_accepting)
    }

    /// Return whether the committed prefix has been irrecoverably rejected.
    pub fn is_rejected(&self) -> bool {
        self.alternatives.is_empty()
    }

    /// Return the allowed-token mask as a packed `u32` bitset.
    pub fn mask(&self) -> Vec<u32> {
        let mut mask = vec![0u32; self.mask_len];
        self.fill_mask(&mut mask);
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins `TEST_COMPAT_MODE` to JsonSchema for full-JSON tests, restoring
    /// the prior mode on drop. Mirrors `EnvVarGuard`'s compat handling in the
    /// json_schema tests; the mode cell is the live control (env only seeds
    /// fresh threads), so no env change is needed.
    struct CompatModeGuard {
        original: crate::import::json_schema::string::JsonStringCompatMode,
    }

    impl CompatModeGuard {
        fn json_schema() -> Self {
            Self::set_mode(crate::import::json_schema::string::JsonStringCompatMode::JsonSchema)
        }

        fn native() -> Self {
            Self::set_mode(
                crate::import::json_schema::string::JsonStringCompatMode::LlGuidanceNative,
            )
        }

        fn set_mode(mode: crate::import::json_schema::string::JsonStringCompatMode) -> Self {
            use crate::import::json_schema::string::TEST_COMPAT_MODE;
            let original = TEST_COMPAT_MODE.with(|cell| cell.get());
            TEST_COMPAT_MODE.with(|cell| cell.set(mode));
            Self { original }
        }
    }

    impl Drop for CompatModeGuard {
        fn drop(&mut self) {
            use crate::import::json_schema::string::TEST_COMPAT_MODE;
            TEST_COMPAT_MODE.with(|cell| cell.set(self.original));
        }
    }

    mod projected_quotient_fixture {
        use crate as glrmask;

        include!("../../tests/fixtures/snowplow_hostname.rsinc");

        pub(super) fn schema_and_vocab() -> (&'static str, Vocab) {
            (SNOWPLOW_SCHEMA, snowplow_vocab())
        }

        pub(super) fn replay_ids() -> &'static [u32] {
            SNOWPLOW_REPLAY_IDS
        }
    }

    fn token_allowed(mask: &[u32], token_id: u32) -> bool {
        let word = token_id as usize / 32;
        let bit = token_id % 32;
        mask.get(word)
            .is_some_and(|bits| bits & (1u32 << bit) != 0)
    }

    fn compressed_lark_grammar(source: &str) -> crate::grammar::flat::GrammarDef {
        let mut named = crate::import::lark::parse_lark_to_named_uncompressed(source).unwrap();
        assert!(crate::grammar::right_linear::compress_large_right_linear_grammar(
            &mut named
        ));
        let factored = crate::grammar::factoring::factor_named_grammar(named);
        crate::grammar::ast::lower(&factored).unwrap()
    }

    fn compile_compressed_static(source: &str, vocab: &Vocab) -> crate::Constraint {
        crate::compiler::pipeline::compile_owned_with_table_construction(
            compressed_lark_grammar(source),
            vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
    }

    fn compile_compressed_dynamic(source: &str, vocab: &Vocab) -> DynamicConstraint {
        crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            compressed_lark_grammar(source),
            vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap()
    }

    fn vocab() -> Vocab {
        Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"aa".to_vec()),
            (4, b" ".to_vec()),
        ])
    }

    #[test]
    fn dynamic_constraint_matches_constraint_masks_and_commits() {
        let vocab = vocab();
        let grammar = r#"
            start start;
            t A ::= 'a'+;
            t B ::= 'b';
            nt start ::= A B;
        "#;
        let normal = crate::Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let dynamic = DynamicConstraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut normal_state = normal.start();
        let mut dynamic_state = dynamic.start();

        assert_eq!(normal_state.mask(), dynamic_state.mask());
        normal_state.commit_token(3).unwrap();
        dynamic_state.commit_token(3).unwrap();
        assert_eq!(normal_state.mask(), dynamic_state.mask());
        normal_state.commit_token(1).unwrap();
        dynamic_state.commit_token(1).unwrap();
        assert_eq!(normal_state.is_accepting(), dynamic_state.is_accepting());
        assert_eq!(normal_state.mask(), dynamic_state.mask());
    }

    #[test]
    fn dynamic_json_schema_ordinary_bounded_array_item_stays_a_lazy_terminal() {
        let vocab = Vocab::new(vec![
            (0, b"[".to_vec()),
            (1, b"]".to_vec()),
            (2, b",".to_vec()),
            (3, b"\"".to_vec()),
            (4, b"a".to_vec()),
            (5, b"b".to_vec()),
            (6, b"aa".to_vec()),
            (7, b"\"a\"".to_vec()),
            (8, b"\"b\"".to_vec()),
            (9, b" ".to_vec()),
        ]);
        let schema = r#"{
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1024
            }
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count(),
            1,
        );
        assert!(
            dynamic.inner.tokenizer.num_states() < 1_000,
            "the 1024-character item bound must not be materialized into the physical tokenizer",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(accepts(br#"["a", "bb"]"#));
        assert!(!accepts(br#"[""]"#));

        let mut at_limit = Vec::with_capacity(1028);
        at_limit.extend_from_slice(b"[\"");
        at_limit.extend(std::iter::repeat_n(b'a', 1024));
        at_limit.extend_from_slice(b"\"]");
        assert!(accepts(&at_limit));

        let mut too_long = Vec::with_capacity(1029);
        too_long.extend_from_slice(b"[\"");
        too_long.extend(std::iter::repeat_n(b'a', 1025));
        too_long.extend_from_slice(b"\"]");
        assert!(!accepts(&too_long));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut loaded_state = loaded.start();
        for (step, token) in [0u32, 7, 2, 9, 8, 1]
        .into_iter()
        .enumerate()
        {
            assert_eq!(dynamic_state.is_accepting(), loaded_state.is_accepting());
            let dynamic_mask = dynamic_state.mask();
            let loaded_mask = loaded_state.mask();
            assert_eq!(dynamic_mask, loaded_mask, "save/load mask mismatch at step {step}");
            assert!(
                token_allowed(&dynamic_mask, token),
                "fixture token {token} must be admitted at step {step}",
            );
            dynamic_state
                .commit_token(token)
                .unwrap_or_else(|error| panic!("dynamic commit failed at step {step}: {error}"));
            loaded_state
                .commit_token(token)
                .unwrap_or_else(|error| panic!("loaded commit failed at step {step}: {error}"));
        }
        assert_eq!(dynamic_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(dynamic_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_json_schema_ordinary_bounded_string_bound_is_symbolic() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
        ]);
        let mut physical_state_counts = Vec::new();
        for max_length in [255usize, 1024, 1_000_000_000_000] {
            let schema = format!(r#"{{"type":"string","maxLength":{max_length}}}"#);
            let dynamic = DynamicConstraint::from_json_schema(&schema, &vocab).unwrap();
            assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
            assert_eq!(
                dynamic
                    .inner
                    .tokenizer
                    .virtual_residual_bounded_code_liveness_oracle_count(),
                1,
            );
            physical_state_counts.push(dynamic.inner.tokenizer.num_states());
        }
        assert!(
            physical_state_counts.windows(2).all(|pair| pair[0] == pair[1]),
            "physical tokenizer size must not depend on the numeric bounded-string maximum: {physical_state_counts:?}",
        );
    }

    #[test]
    fn dynamic_json_schema_bounded_pattern_rejects_escaped_spellings_in_native_mode() {
        // Native-default counterpart to the JsonSchema-mode escaped-spelling
        // tests: the LlGuidanceNative lexical language deliberately does not
        // materialize `\uXXXX` spellings (production default for automata
        // size); raw spellings keep working. Same schema as the bounded
        // liveness-oracle test; representative single pin, not a clone set.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _compat = CompatModeGuard::native();
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b"\\".to_vec()),
            (5, b"u".to_vec()),
            (6, b"0".to_vec()),
            (7, b"6".to_vec()),
            (8, b"1".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^(?:a|bb)+$",
            "minLength": 2,
            "maxLength": 5000
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(accepts(br#""aa""#));
        assert!(!accepts(br#""\u0061\u0061""#));
        assert!(!accepts(br#""a\u0061""#));
    }

    #[test]
    fn dynamic_json_schema_bounded_pattern_uses_exact_code_liveness_oracle() {
        // Pin JsonSchema mode: full-JSON expectations predate the native default (assertions unchanged).
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _compat = CompatModeGuard::json_schema();
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b"\\".to_vec()),
            (5, b"u".to_vec()),
            (6, b"0".to_vec()),
            (7, b"6".to_vec()),
            (8, b"1".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^(?:a|bb)+$",
            "minLength": 2,
            "maxLength": 5000
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert!(
            !dynamic
                .inner
                .dynamic_mask_vocab_for_runtime()
                .has_terminal_observation_classes(),
            "physical terminal-observation quotients must stay disabled for virtual residual runtimes",
        );
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "the importer-generated pattern/length intersection must carry the certified prefix-code liveness oracle",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts(br#""a""#));
        assert!(accepts(br#""aa""#));
        assert!(
            accepts(br#""\u0061\u0061""#),
            "two escaped spellings of decoded 'a' must count as two JSON characters",
        );
        assert!(!accepts(br#""ab""#));

        let mut at_limit = Vec::with_capacity(5002);
        at_limit.push(b'"');
        at_limit.extend(std::iter::repeat_n(b'a', 5000));
        at_limit.push(b'"');
        assert!(accepts(&at_limit));

        let mut too_long = Vec::with_capacity(5003);
        too_long.push(b'"');
        too_long.extend(std::iter::repeat_n(b'a', 5001));
        too_long.push(b'"');
        assert!(!accepts(&too_long));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(
            !loaded
                .inner
                .dynamic_mask_vocab_for_runtime()
                .has_terminal_observation_classes(),
            "save/load must not attach a physical observation quotient to a virtual residual runtime",
        );
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "save/load must reconstruct the certified liveness oracle from the retained terminal expression",
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_json_schema_bounded_format_uses_exact_code_liveness_oracle() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b".".to_vec()),
            (4, b"-".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "format": "hostname",
            "minLength": 3,
            "maxLength": 5000
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "format/length intersections must carry the same certified JSON decoded-length liveness oracle as pattern/length intersections",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts(br#""aa""#));
        assert!(accepts(br#""aaa""#));
        assert!(accepts(br#""a.b""#));
        assert!(!accepts(br#""a..b""#));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "save/load must reconstruct the bounded format liveness oracle",
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_json_schema_pattern_format_and_length_share_exact_code_liveness_oracle() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b".".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^(?:a|bb)+$",
            "format": "hostname",
            "minLength": 2,
            "maxLength": 128
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "nested pattern/format/length intersections below the generic giant-repeat threshold must still use the certified residual representation",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts(br#""a""#));
        assert!(accepts(br#""aa""#));
        assert!(accepts(br#""bb""#));
        assert!(!accepts(br#""a.b""#));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());

        let encode_current = |payload: DynamicConstraintPayloadV5| {
            let payload = bincode::serialize(&payload).unwrap();
            let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
            bytes.extend_from_slice(&LEGACY_DYNAMIC_CONSTRAINT_VERSION_V17.to_le_bytes());
            bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&payload);
            bytes
        };

        let mut missing_owner = DynamicConstraintPayloadV5 {
            alternatives: vec![DynamicConstraint::payload_v5_for_constraint(&dynamic.inner)],
        };
        assert_eq!(missing_owner.alternatives[0].base.virtual_runtimes.len(), 1);
        missing_owner.alternatives[0].base.virtual_runtimes.clear();
        let error = DynamicConstraint::load(&encode_current(missing_owner)).unwrap_err();
        assert!(
            error.to_string().contains("terminal ownership mismatch"),
            "dropping a below-threshold residual owner from its physical proxy artifact must fail closed: {error}",
        );

        let mut forged_owner = DynamicConstraintPayloadV5 {
            alternatives: vec![DynamicConstraint::payload_v5_for_constraint(&dynamic.inner)],
        };
        let terminal = forged_owner.alternatives[0].base.virtual_runtimes[0].terminal as usize;
        forged_owner.alternatives[0]
            .base
            .v2
            .v1
            .terminal_exprs
            .as_mut()
            .expect("current dynamic artifact retains terminal expressions")[terminal] =
            Expr::U8Seq(b"a".to_vec());
        let error = DynamicConstraint::load(&encode_current(forged_owner)).unwrap_err();
        assert!(
            error.to_string().contains("certified bounded-code residual"),
            "a below-threshold residual owner cannot be forged for an uncertified expression: {error}",
        );
    }

    #[test]
    fn dynamic_json_schema_cross_branch_bounded_string_constraints_share_exact_code_liveness_oracle() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b"bb".to_vec()),
            (5, b".".to_vec()),
            (6, b"-".to_vec()),
        ]);
        let schemas = [
            r#"{
                "allOf": [
                    {"type":"string","format":"hostname","minLength":2,"maxLength":5000},
                    {"type":"string","pattern":"^(?:a|bb)+$","minLength":3,"maxLength":5000},
                    {"type":"string","pattern":"^(?:a|bbb)+$","maxLength":5000}
                ]
            }"#,
            r#"{
                "allOf": [
                    {"type":"string","pattern":"^(?:a|bb)+$","minLength":2,"maxLength":6000},
                    {"type":"string","pattern":"^(?:a|bbb)+$","minLength":3,"maxLength":5000},
                    {"type":"string","format":"hostname","maxLength":5500}
                ]
            }"#,
        ];

        for schema in schemas {
            let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
            assert!(
                dynamic.inner.tokenizer.has_virtual_residual_runtime(),
                "expected virtual residual runtime for cross-branch schema: {schema}",
            );
            assert!(
                dynamic
                    .inner
                    .tokenizer
                    .virtual_residual_bounded_code_liveness_oracle_count()
                    > 0,
                "cross-branch bounded string constraints must flatten to one common JSON decoded-length envelope plus finite constraint operands: {schema}",
            );
            assert!(
                !dynamic
                    .inner
                    .dynamic_mask_vocab_for_runtime()
                    .has_terminal_observation_classes(),
                "virtual residual constraints must not attach a physical observation quotient",
            );

            let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
            assert!(
                loaded
                    .inner
                    .tokenizer
                    .virtual_residual_bounded_code_liveness_oracle_count()
                    > 0,
                "save/load must reconstruct the cross-branch bounded-code oracle: {schema}",
            );
            assert_eq!(loaded.start().mask(), dynamic.start().mask());
        }
    }

    #[test]
    fn dynamic_json_schema_allof_bounded_patterns_share_exact_code_liveness_oracle() {
        // Pin JsonSchema mode: full-JSON expectations predate the native default (assertions unchanged).
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _compat = CompatModeGuard::json_schema();
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b"c".to_vec()),
            (5, b"\\".to_vec()),
            (6, b"u".to_vec()),
            (7, b"0".to_vec()),
            (8, b"6".to_vec()),
            (9, b"1".to_vec()),
        ]);
        let schema = r#"{
            "allOf": [
                {
                    "type": "string",
                    "pattern": "^(?:a|bb)+$",
                    "minLength": 2,
                    "maxLength": 5000
                },
                {
                    "type": "string",
                    "pattern": "^(?:a|cc)+$",
                    "minLength": 3,
                    "maxLength": 4000
                }
            ]
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "allOf branches with the same JSON length envelope language should share one exact bounded-code oracle",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts(br#""aa""#));
        assert!(accepts(br#""aaa""#));
        assert!(accepts(br#""\u0061\u0061\u0061""#));
        assert!(!accepts(br#""bbb""#));
        assert!(!accepts(br#""ccc""#));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "save/load must reconstruct the coalesced allOf oracle",
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_json_schema_bounded_unicode_pattern_keeps_raw_and_escaped_spellings_exact() {
        // Pin JsonSchema mode: full-JSON expectations predate the native default (assertions unchanged).
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _compat = CompatModeGuard::json_schema();
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, "é".as_bytes().to_vec()),
            (2, br"\u00e9".to_vec()),
            (3, br"\u00E9".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^(?:é|xx)+$",
            "minLength": 2,
            "maxLength": 5000
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts("\"é\"".as_bytes()));
        assert!(accepts("\"éé\"".as_bytes()));
        assert!(accepts(br#""\u00e9\u00E9""#));
        assert!(accepts("\"é\\u00e9\"".as_bytes()));
        assert!(!accepts("\"éx\"".as_bytes()));
    }

    #[test]
    fn static_constraint_artifact_preserves_dynamic_runtime_sidecars() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"x".to_vec()),
        ]);
        let dynamic = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a{0,10000}/;
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();
        let original = dynamic.clone().into_constraint();
        assert!(!original.tokenizer.virtual_runtime_metadata().is_empty());
        let loaded = Constraint::load(original.save()).unwrap();
        assert_eq!(
            loaded.tokenizer.virtual_runtime_metadata(),
            original.tokenizer.virtual_runtime_metadata(),
        );
        assert_eq!(loaded.start().mask(), original.start().mask());
    }

    #[test]
    fn dynamic_constraint_save_load_round_trip() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                ignore WS;
                t WS ::= " "+;
                nt start ::= "a"+ "b";
            "#,
            &vocab,
        )
        .unwrap();
        let loaded = DynamicConstraint::load(&constraint.save()).unwrap();
        assert!(constraint.inner.ignore_expr.is_some());
        assert_eq!(loaded.inner.ignore_expr, constraint.inner.ignore_expr);
        assert_eq!(
            loaded.inner.tokenizer.terminal_exprs(),
            constraint.inner.tokenizer.terminal_exprs(),
        );
        assert_eq!(constraint.mask_len(), loaded.mask_len());
        assert_eq!(constraint.start().mask(), loaded.start().mask());
    }

    #[test]
    fn dynamic_v20_persists_vocab_and_validates_shared_vocab() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"b".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a+/;
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();

        let current = constraint.save();
        assert_eq!(
            u16::from_le_bytes([current[8], current[9]]),
            DYNAMIC_CONSTRAINT_VERSION,
        );
        let payload: DynamicConstraintPayloadV7 =
            bincode::deserialize(&current[DYNAMIC_CONSTRAINT_HEADER_LEN..]).unwrap();
        assert!(payload.dynamic_mask_vocab.is_some());
        let current_loaded = DynamicConstraint::load(&current).unwrap();
        assert_eq!(current_loaded.start().mask(), constraint.start().mask());

        let mut mismatched_shared_vocab = payload.clone();
        let mut second = mismatched_shared_vocab.alternatives[0].clone();
        second.base.base.v2.v1.token_bytes =
            Arc::new(BTreeMap::from([(0, b"z".to_vec())]));
        mismatched_shared_vocab.alternatives.push(second);
        let mismatched_shared_vocab = bincode::serialize(&mismatched_shared_vocab).unwrap();
        let mut malformed =
            Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + mismatched_shared_vocab.len());
        malformed.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
        malformed.extend_from_slice(&DYNAMIC_CONSTRAINT_VERSION.to_le_bytes());
        malformed.extend_from_slice(&(mismatched_shared_vocab.len() as u64).to_le_bytes());
        malformed.extend_from_slice(&mismatched_shared_vocab);
        let error = DynamicConstraint::load(&malformed).unwrap_err();
        assert!(error
            .to_string()
            .contains("shares a vocabulary index across alternatives with different token bytes"));

    }

    #[test]
    fn dynamic_virtual_unit_repeat_save_load_round_trip() {
        let vocab = Vocab::new(vec![
            (0, b"b".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"aaa".to_vec()),
            (4, b"baaa".to_vec()),
            (5, b"baaaaa".to_vec()),
            (6, b"x".to_vec()),
        ]);
        let grammar = r#"
            start start;
            t A ::= /a{0,1000000000}/;
            t B ::= 'b';
            nt start ::= B A;
        "#;
        let constraint = DynamicConstraint::from_glrm_grammar(grammar, &vocab).unwrap();
        assert!(
            constraint
                .inner
                .tokenizer
                .virtual_zero_min_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
        );

        let bytes = constraint.save();
        let loaded = DynamicConstraint::load(&bytes).unwrap();
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_zero_min_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
            "load must reconstruct the exact arithmetic lexer sidecar",
        );
        assert_eq!(constraint.start().mask(), loaded.start().mask());

        let mut original_state = constraint.start();
        let mut loaded_state = loaded.start();
        original_state.commit_token(4).unwrap();
        loaded_state.commit_token(4).unwrap();
        assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(original_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_virtual_runtime_metadata_is_exact_and_fail_closed() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a{0,10000}/;
                t B ::= /b{0,9000}/;
                nt start ::= A | B;
            "#,
            &vocab,
        )
        .unwrap();
        let metadata = constraint.inner.tokenizer.virtual_runtime_metadata();
        assert_eq!(metadata.len(), 2);

        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            DYNAMIC_CONSTRAINT_VERSION,
        );
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.inner.tokenizer.virtual_runtime_metadata().len(), 2);
        assert_eq!(constraint.start().mask(), loaded.start().mask());

        let transfer = constraint.clone().into_saved();
        assert_eq!(
            u16::from_le_bytes([transfer[8], transfer[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert_eq!(transferred.inner.tokenizer.virtual_runtime_metadata().len(), 2);
        assert_eq!(constraint.start().mask(), transferred.start().mask());

        fn encode_payload(base: DynamicConstraintPayloadV4Alternative) -> Vec<u8> {
            let payload = DynamicConstraintPayloadV7 {
                alternatives: vec![DynamicConstraintPayloadV7Alternative {
                    base: DynamicConstraintPayloadV5Alternative {
                        base,
                        terminal_observation_classes: Vec::new(),
                    },
                    projected_terminal_quotients: Vec::new(),
                    boundary_trigger: DynamicBoundaryTriggerWire::None,
                    recursive_constraint_artifact: None,
                }],
                dynamic_mask_vocab: None,
            };
            let payload = bincode::serialize(&payload).unwrap();
            let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_VERSION.to_le_bytes());
            bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&payload);
            bytes
        }

        let mut missing = DynamicConstraint::payload_v4_for_constraint(&constraint.inner);
        missing.virtual_runtimes.pop();
        let error = DynamicConstraint::load(&encode_payload(missing)).unwrap_err();
        assert!(
            error.to_string().contains("terminal ownership mismatch"),
            "unexpected missing-runtime error: {error}",
        );

        let mut duplicate_root = DynamicConstraint::payload_v4_for_constraint(&constraint.inner);
        let root = duplicate_root.virtual_runtimes[0].root_state;
        duplicate_root.virtual_runtimes[1].root_state = root;
        let error = DynamicConstraint::load(&encode_payload(duplicate_root)).unwrap_err();
        assert!(
            error.to_string().contains("invalid terminal/root ownership"),
            "unexpected duplicate-root error: {error}",
        );

        let mut mismatched_support = DynamicConstraint::payload_v4_for_constraint(&constraint.inner);
        let terminal = mismatched_support.virtual_runtimes[0].terminal as usize;
        if let Some(exprs) = mismatched_support.v2.v1.terminal_exprs.as_mut() {
            exprs[terminal] = Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"z".to_vec())),
                min: 0,
                max: Some(10_000),
            };
        }
        let error = DynamicConstraint::load(&encode_payload(mismatched_support)).unwrap_err();
        assert!(
            error.to_string().contains("byte support"),
            "unexpected byte-support mismatch error: {error}",
        );
    }

    #[test]
    fn dynamic_transfer_v13_virtual_residuals_round_trip_and_v12_v11_backward_compat() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b"bb".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^(?:a|bb)+$",
            "minLength": 2,
            "maxLength": 5000
        }"#;
        let constraint = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(constraint.inner.tokenizer.has_any_virtual_runtime());

        // Test current V13 save and load.
        let v13_bytes = constraint.save_with_external_vocab();
        assert_eq!(u16::from_le_bytes([v13_bytes[8], v13_bytes[9]]), DYNAMIC_TRANSFER_VERSION);
        assert_eq!(DYNAMIC_TRANSFER_VERSION, 13);

        let v13_loaded = DynamicConstraint::load_with_vocab(&v13_bytes, &vocab).unwrap();
        // Loaded constraint should carry the mask tokenizer projection directly from the wire
        assert!(v13_loaded.inner.dynamic_mask_vocab.mask_projection_tokenizer().is_some());
        assert_eq!(v13_loaded.start().mask(), constraint.start().mask());

        // Test V12 backward compatibility. V13 deliberately keeps the V12
        // six-section framing; only the metadata section is wrapped with the
        // compact prepared-master-proof artifact. Re-encode the exact same
        // sections with the legacy base metadata and V12 header.
        let mut v12_sections =
            DynamicConstraint::transfer_sections_v12_from_constraint(&constraint.inner);
        let v13_metadata: DynamicConstraintTransferMetadataV13 =
            bincode::deserialize(&v12_sections.metadata).unwrap();
        v12_sections.metadata = bincode::serialize(&v13_metadata.base).unwrap();
        let lengths = [
            v12_sections.table.len(),
            v12_sections.tokenizer.len(),
            v12_sections.terminal_exprs_compressed.len(),
            v12_sections.recursive_constraint_artifact.len(),
            v12_sections.metadata.len(),
            v12_sections.virtual_residual_wire.len(),
        ];
        let payload_capacity = DYNAMIC_TRANSFER_V12_PAYLOAD_HEADER_LEN
            + DYNAMIC_TRANSFER_V12_ALT_DESCRIPTOR_LEN
            + lengths.iter().sum::<usize>();
        let mut v12_bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload_capacity);
        v12_bytes.extend_from_slice(&DYNAMIC_TRANSFER_MAGIC);
        v12_bytes.extend_from_slice(&LEGACY_DYNAMIC_TRANSFER_VERSION_V12.to_le_bytes());
        v12_bytes.extend_from_slice(&0u64.to_le_bytes());
        v12_bytes.extend_from_slice(&1u32.to_le_bytes());
        v12_bytes.extend_from_slice(&0u32.to_le_bytes());
        for &length in &lengths {
            v12_bytes.extend_from_slice(&(length as u64).to_le_bytes());
        }
        v12_bytes.extend_from_slice(&v12_sections.table);
        v12_bytes.extend_from_slice(&v12_sections.tokenizer);
        v12_bytes.extend_from_slice(&v12_sections.terminal_exprs_compressed);
        v12_bytes.extend_from_slice(&v12_sections.recursive_constraint_artifact);
        v12_bytes.extend_from_slice(&v12_sections.metadata);
        v12_bytes.extend_from_slice(&v12_sections.virtual_residual_wire);
        let payload_len = v12_bytes.len() - DYNAMIC_CONSTRAINT_HEADER_LEN;
        v12_bytes[10..18].copy_from_slice(&(payload_len as u64).to_le_bytes());

        let v12_loaded = DynamicConstraint::load_with_vocab(&v12_bytes, &vocab).unwrap();
        assert!(v12_loaded.inner.dynamic_mask_vocab.mask_projection_tokenizer().is_some());
        assert_eq!(v12_loaded.start().mask(), constraint.start().mask());

        // Test V11 backward compatibility: construct a V11 transfer payload
        let v11_sections = DynamicConstraint::transfer_sections_v11_from_constraint(&constraint.inner);
        let v11_alt_count = 1u32;
        let v11_descriptor_bytes = DYNAMIC_TRANSFER_V11_ALT_DESCRIPTOR_LEN;
        let v11_section_bytes = v11_sections.table.len()
            + v11_sections.tokenizer.len()
            + v11_sections.terminal_exprs_compressed.len()
            + v11_sections.recursive_constraint_artifact.len()
            + v11_sections.metadata.len();
        let mut v11_bytes = Vec::with_capacity(
            DYNAMIC_CONSTRAINT_HEADER_LEN
                + DYNAMIC_TRANSFER_V11_PAYLOAD_HEADER_LEN
                + v11_descriptor_bytes
                + v11_section_bytes,
        );
        v11_bytes.extend_from_slice(&DYNAMIC_TRANSFER_MAGIC);
        v11_bytes.extend_from_slice(&LEGACY_DYNAMIC_TRANSFER_VERSION_V11.to_le_bytes());
        v11_bytes.extend_from_slice(&0u64.to_le_bytes());
        v11_bytes.extend_from_slice(&v11_alt_count.to_le_bytes());
        v11_bytes.extend_from_slice(&0u32.to_le_bytes());
        let descriptor_pos = v11_bytes.len();
        v11_bytes.resize(descriptor_pos + v11_descriptor_bytes, 0);
        let lengths = [
            v11_sections.table.len() as u64,
            v11_sections.tokenizer.len() as u64,
            v11_sections.terminal_exprs_compressed.len() as u64,
            v11_sections.recursive_constraint_artifact.len() as u64,
            v11_sections.metadata.len() as u64,
        ];
        let mut cur = descriptor_pos;
        for len in lengths {
            v11_bytes[cur..cur + 8].copy_from_slice(&len.to_le_bytes());
            cur += 8;
        }
        v11_bytes.extend_from_slice(&v11_sections.table);
        v11_bytes.extend_from_slice(&v11_sections.tokenizer);
        v11_bytes.extend_from_slice(&v11_sections.terminal_exprs_compressed);
        v11_bytes.extend_from_slice(&v11_sections.recursive_constraint_artifact);
        v11_bytes.extend_from_slice(&v11_sections.metadata);
        let payload_len = (v11_bytes.len() - DYNAMIC_CONSTRAINT_HEADER_LEN) as u64;
        v11_bytes[10..18].copy_from_slice(&payload_len.to_le_bytes());

        let v11_loaded = DynamicConstraint::load_with_vocab(&v11_bytes, &vocab).unwrap();
        assert_eq!(v11_loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn current_dynamic_transfer_defers_terminal_expression_trees() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a+/;
                t B ::= /b+/;
                nt start ::= A B;
            "#,
            &vocab,
        )
        .unwrap();
        let expected = constraint
            .inner
            .tokenizer
            .terminal_exprs()
            .expect("fresh constraint retains terminal expressions")
            .to_vec();
        assert!(constraint.inner.tokenizer.virtual_runtime_metadata().is_empty());

        let transfer = constraint.into_saved();
        let loaded = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            loaded.inner.tokenizer.terminal_exprs().is_none(),
            "ordinary transfer load must not eagerly materialize source expression trees",
        );
        assert!(matches!(
            loaded.inner.deferred_terminal_exprs_blob,
            Some(crate::runtime::DeferredTerminalExprBytes::CompressedBacked { .. })
        ));
        assert_eq!(
            loaded
                .inner
                .retained_terminal_exprs()
                .expect("composition must recover deferred source expressions"),
            expected,
        );
        assert_eq!(
            loaded
                .inner
                .retained_terminal_exprs()
                .expect("decoded expressions should stay cached"),
            expected,
        );
    }

    #[test]
    fn dynamic_v20_terminal_observation_certificate_is_shape_validated() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a+/;
                t B ::= /b+/;
                nt start ::= A | B;
            "#,
            &vocab,
        )
        .unwrap();
        let states = constraint.inner.tokenizer.num_states() as usize;
        let terminals = constraint.inner.tokenizer.num_terminals();

        let encode = |terminal_observation_classes: Vec<(TerminalID, Vec<u32>)>| {
            let mut alternative = DynamicConstraint::payload_v5_for_constraint(&constraint.inner);
            alternative.terminal_observation_classes = terminal_observation_classes;
            let payload = DynamicConstraintPayloadV7 {
                alternatives: vec![DynamicConstraintPayloadV7Alternative {
                    base: alternative,
                    projected_terminal_quotients: Vec::new(),
                    boundary_trigger: DynamicBoundaryTriggerWire::None,
                    recursive_constraint_artifact: None,
                }],
                dynamic_mask_vocab: None,
            };
            let payload = bincode::serialize(&payload).unwrap();
            let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_VERSION.to_le_bytes());
            bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&payload);
            bytes
        };

        let bad_terminal = DynamicConstraint::load(&encode(vec![(
            terminals,
            vec![1; states],
        )]))
        .unwrap_err();
        assert!(bad_terminal
            .to_string()
            .contains("terminal-observation certificate references terminal"));

        let duplicate = DynamicConstraint::load(&encode(vec![
            (0, vec![1; states]),
            (0, vec![1; states]),
        ]))
        .unwrap_err();
        assert!(duplicate
            .to_string()
            .contains("terminal-observation certificate repeats terminal"));

        let bad_len = DynamicConstraint::load(&encode(vec![(
            0,
            vec![1; states.saturating_sub(1)],
        )]))
        .unwrap_err();
        assert!(bad_len
            .to_string()
            .contains("terminal-observation certificate for terminal"));
    }

    #[test]
    fn dynamic_standalone_virtual_unit_v20_and_transfer_v9_round_trip() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a{0,10000}/;
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();
        let metadata = constraint.inner.tokenizer.virtual_runtime_metadata();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].kind, crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeKind::UnitRepeat);
        assert_eq!(metadata[0].root_state, 0);

        let saved = constraint.save();
        assert_eq!(u16::from_le_bytes([saved[8], saved[9]]), DYNAMIC_CONSTRAINT_VERSION);
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.inner.tokenizer.virtual_runtime_metadata(), metadata);
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let transfer = constraint.clone().into_saved();
        assert_eq!(u16::from_le_bytes([transfer[8], transfer[9]]), DYNAMIC_TRANSFER_VERSION);
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert_eq!(transferred.inner.tokenizer.virtual_runtime_metadata(), metadata);
        assert_eq!(transferred.start().mask(), constraint.start().mask());

        for token in [0u32, 1, 2] {
            let mut original = constraint.start();
            let mut loaded_state = loaded.start();
            let mut transferred_state = transferred.start();
            original.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            transferred_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.is_accepting(), original.is_accepting());
            assert_eq!(transferred_state.is_accepting(), original.is_accepting());
            assert_eq!(loaded_state.mask(), original.mask());
            assert_eq!(transferred_state.mask(), original.mask());
        }

    }

    #[test]
    fn dynamic_v20_combines_composition_and_residual_runtime_metadata() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let mut constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a{0,10000}/;
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();
        constraint.inner.late_grammar_slots = vec![crate::runtime::LateGrammarSlot {
            name: "child".to_owned(),
            terminal_id: 0,
        }];
        constraint.inner.boundary_trigger = crate::runtime::BoundaryTrigger::Tokens(
            Arc::from(vec![1u32].into_boxed_slice()),
        );
        let state_count = constraint.inner.tokenizer.num_states() as u32;
        constraint.inner.dynamic_mask_vocab.set_terminal_observation_classes(vec![(
            0,
            Arc::from((1..=state_count).collect::<Vec<_>>().into_boxed_slice()),
        )]);
        assert!(constraint.inner.dynamic_mask_vocab.to_vocab_artifact().is_some());
        let metadata = constraint.inner.tokenizer.virtual_runtime_metadata();
        assert!(!metadata.is_empty());

        // This test mutates private persistence metadata after public
        // compilation has already frozen the canonical transfer artifact.
        // Refresh that cache explicitly so the transfer half of the test
        // continues to cover the wire representation produced from the
        // mutated fixture. Production constraints are immutable after build.
        constraint.external_vocab_artifact_cache = None;
        constraint.cache_external_vocab_artifact_for_save();

        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            DYNAMIC_CONSTRAINT_VERSION,
        );
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.inner.late_grammar_slots, constraint.inner.late_grammar_slots);
        assert_eq!(loaded.inner.tokenizer.virtual_runtime_metadata(), metadata);
        assert!(loaded.inner.dynamic_mask_vocab.has_terminal_observation_classes());
        assert!(loaded.inner.dynamic_mask_vocab.to_vocab_artifact().is_some());
        assert_eq!(
            loaded.inner.boundary_trigger.token_summary().map(|tokens| tokens.to_vec()),
            Some(vec![1]),
        );
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let transfer = constraint.clone().into_saved();
        assert_eq!(
            u16::from_le_bytes([transfer[8], transfer[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert_eq!(transferred.inner.tokenizer.virtual_runtime_metadata(), metadata);
        assert!(transferred.inner.dynamic_mask_vocab.has_terminal_observation_classes());
        assert_eq!(
            transferred
                .inner
                .boundary_trigger
                .token_summary()
                .map(|tokens| tokens.to_vec()),
            Some(vec![1]),
        );
        assert_eq!(transferred.start().mask(), constraint.start().mask());

    }

    #[test]
    #[ignore = "pre-release legacy dynamic artifact compatibility is not an acceptance requirement"]
    fn dynamic_main_v14_late_slot_artifact_still_loads() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let mut constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();
        constraint.inner.late_grammar_slots = vec![crate::runtime::LateGrammarSlot {
            name: "child".to_owned(),
            terminal_id: 0,
        }];
        let legacy = LegacyDynamicConstraintPayloadV14Main {
            alternatives: vec![LegacyDynamicConstraintPayloadV14MainAlternative {
                constraint: DynamicConstraint::payload_for_constraint(&constraint.inner),
                late_grammar_slots: constraint.inner.late_grammar_slots.clone(),
            }],
        };
        let payload = bincode::serialize(&legacy).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&LEGACY_DYNAMIC_CONSTRAINT_VERSION_V14_MAIN.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let loaded = DynamicConstraint::load(&bytes).unwrap();
        assert_eq!(loaded.inner.late_grammar_slots, constraint.inner.late_grammar_slots);
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    #[ignore = "pre-release legacy dynamic artifact compatibility is not an acceptance requirement"]
    fn dynamic_virtual_runtime_legacy_v13_and_transfer_v4_still_load() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"x".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a{0,10000}/;
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();
        let original_mask = constraint.start().mask();

        let legacy_persistent = DynamicConstraintPayloadV3 {
            alternatives: vec![DynamicConstraint::payload_for_constraint(&constraint.inner)],
        };
        let payload = bincode::serialize(&legacy_persistent).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&LEGACY_DYNAMIC_CONSTRAINT_VERSION_V13.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        let loaded = DynamicConstraint::load(&bytes).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection()
            || loaded.inner.tokenizer.virtual_zero_min_unit_repeat_mask_tokenizer(2).is_some());
        assert_eq!(loaded.start().mask(), original_mask);

        let legacy_transfer = DynamicConstraintTransferPayloadV1 {
            alternatives: vec![DynamicConstraintTransferAlternativeV1 {
                table: constraint.inner.table.clone_lr(),
                terminal_display_names: constraint.inner.terminal_display_names.clone(),
                tokenizer: constraint.inner.tokenizer.as_ref().clone(),
                ignore_terminal: constraint.inner.ignore_terminal,
                direct_regular_automaton: constraint.inner.direct_regular_automaton.clone(),
                special_token_terminals: constraint.inner.special_token_terminals.clone(),
                ignore_expr: constraint.inner.ignore_expr.clone(),
                terminal_exprs: constraint.inner.tokenizer.terminal_exprs().map(ToOwned::to_owned),
                mask_tokenizer: None,
                full_to_mask_state: Vec::new(),
            }],
        };
        let payload = bincode::serialize(&legacy_transfer).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_MAGIC);
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_VERSION_V4.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        let loaded = DynamicConstraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert_eq!(loaded.start().mask(), original_mask);
    }

    fn variable_width_repeat_intersection_grammar(
        left_max: usize,
        right_max: usize,
    ) -> crate::grammar::flat::GrammarDef {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let left_body = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"bb".to_vec()),
        ]);
        let right_body = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
        ]);
        GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(left_body),
                        min: 0,
                        max: Some(left_max),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(right_body),
                        min: 0,
                        max: Some(right_max),
                    }),
                },
            }],
            ..GrammarDef::default()
        }
    }

    fn variable_width_repeat_intersection_vocab() -> Vocab {
        Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"bb".to_vec()),
            (2, b"abb".to_vec()),
            (3, b"aabb".to_vec()),
            (4, b"bbbb".to_vec()),
            (5, b"ab".to_vec()),
            (6, b"b".to_vec()),
            (7, b"x".to_vec()),
        ])
    }

    #[test]
    fn dynamic_billion_by_billion_repeat_intersection_is_lazy_end_to_end() {
        let vocab = variable_width_repeat_intersection_vocab();
        let grammar = variable_width_repeat_intersection_grammar(
            1_000_000_000,
            900_000_000,
        );
        let constraint = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            variable_width_repeat_intersection_grammar(32, 32),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(constraint.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            constraint.inner.tokenizer.num_states() < 32,
            "physical tokenizer must not scale with either billion-sized bound",
        );
        assert!(
            constraint
                .inner
                .tokenizer
                .virtual_binary_repeat_intersection_interned_state_count()
                <= 4,
            "build-time exact residual discovery must stay constant and independent of N*M",
        );
        assert!(
            constraint
                .inner
                .dynamic_mask_vocab
                .mask_projection_tokenizer()
                .is_some(),
            "finite mask projection is an immutable runtime accelerator and must be prepared during build finalization",
        );

        let mask_tokenizer = constraint
            .inner
            .dynamic_mask_vocab
            .mask_projection_tokenizer()
            .expect("build finalization must prepare the finite exact product");
        let start_mask = constraint.start().mask();
        assert!(
            constraint.inner.lazy_dynamic_mask_vocab.get().is_none(),
            "first mask must not hide projection construction in a lazy runtime cache",
        );
        assert!(
            mask_tokenizer.num_states() < 2_000,
            "mask tokenizer must scale with vocab horizon/body DFAs, not N*M",
        );

        assert_eq!(
            start_mask,
            oracle.start().mask(),
            "far from either upper bound, the billion-scale lazy lexer must have the same finite-vocabulary observations as a materialized 32x32 oracle",
        );
        assert!(!token_allowed(&start_mask, 7), "x cannot begin either repeat language");

        let mut state = constraint.start();
        let mut oracle_state = oracle.start();
        state.commit_token(2).unwrap(); // "abb" = "a" + "bb"
        oracle_state.commit_token(2).unwrap();
        assert!(state.is_accepting());
        let after = state.mask();
        assert_eq!(after, oracle_state.mask());
        assert!(
            constraint
                .inner
                .tokenizer
                .virtual_binary_repeat_intersection_interned_state_count()
                < 32,
            "a short commit must discover only a short exact residual path",
        );
    }

    #[test]
    fn dynamic_repeat_intersection_matches_materialized_boundary_oracle() {
        let vocab = variable_width_repeat_intersection_vocab();
        let grammar = variable_width_repeat_intersection_grammar(4, 5);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar.clone(),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let ordinary = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());

        let mut dynamic_state = dynamic.start();
        let mut ordinary_state = ordinary.start();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        dynamic_state.commit_token(2).unwrap();
        ordinary_state.commit_token(2).unwrap();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        dynamic_state.commit_token(1).unwrap();
        ordinary_state.commit_token(1).unwrap();
        assert_eq!(dynamic_state.is_accepting(), ordinary_state.is_accepting());
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
    }

    #[test]
    fn dynamic_virtual_repeat_intersection_save_load_round_trip() {
        let vocab = variable_width_repeat_intersection_vocab();
        let constraint = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            variable_width_repeat_intersection_grammar(1_000_000_000, 900_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        assert!(constraint.inner.tokenizer.has_virtual_binary_repeat_intersection());

        let bytes = constraint.save();
        let loaded = DynamicConstraint::load(&bytes).unwrap();
        assert!(
            loaded.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "load must reconstruct the lazy exact product sidecar",
        );
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_binary_repeat_intersection_interned_state_count()
                <= 4,
            "load-time residual reconstruction must stay constant and independent of N*M",
        );
        assert_eq!(constraint.start().mask(), loaded.start().mask());

        let mut original_state = constraint.start();
        let mut loaded_state = loaded.start();
        original_state.commit_token(2).unwrap();
        loaded_state.commit_token(2).unwrap();
        assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(original_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_virtual_repeat_intersection_rejects_dead_common_prefix() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar = GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"ab".to_vec())),
                        min: 0,
                        max: Some(1_000_000_000),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"ac".to_vec())),
                        min: 0,
                        max: Some(900_000_000),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"ac".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let constraint = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle_grammar = GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"ab".to_vec())),
                        min: 0,
                        max: Some(8),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"ac".to_vec())),
                        min: 0,
                        max: Some(8),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            oracle_grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(constraint.inner.tokenizer.has_virtual_binary_repeat_intersection());
        let mask = constraint.start().mask();
        assert_eq!(mask, oracle.start().mask());
        assert!(
            !token_allowed(&mask, 0),
            "the common first byte 'a' is dead: left requires b next while right requires c",
        );
        assert!(mask.iter().all(|&word| word == 0));
    }

    #[test]
    fn dynamic_billion_bound_variable_width_repeat_is_lazy_end_to_end() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    expr: Box::new(Expr::Choice(vec![
                        Expr::U8Seq(b"ab".to_vec()),
                        Expr::U8Seq(b"ac".to_vec()),
                    ])),
                    min: 0,
                    max: Some(max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"ac".to_vec()),
            (3, b"abab".to_vec()),
            (4, b"acab".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(16),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(dynamic.inner.tokenizer.num_states() < 32);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [3, 4, 1, 2] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());

        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        original_state.commit_token(0).unwrap();
        loaded_state.commit_token(0).unwrap();
        assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
        assert_eq!(loaded_state.mask(), original_state.mask());
    }

    #[test]
    fn dynamic_billion_nonzero_min_variable_width_repeat_is_lazy_end_to_end() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |min, max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    expr: Box::new(Expr::Choice(vec![
                        Expr::U8Seq(b"ab".to_vec()),
                        Expr::U8Seq(b"c".to_vec()),
                    ])),
                    min,
                    max: Some(max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"c".to_vec()),
            (3, b"abc".to_vec()),
            (4, b"cab".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(999_999_997, 1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        // A small ordinary oracle with the same local lower/upper-bound
        // geometry is enough for the first few model-token walks. Both starts
        // are farther than one vocabulary horizon from either boundary.
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(10, 13),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(dynamic.inner.tokenizer.num_states() < 32);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [1u32, 2, 3] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert!(!dynamic_state.is_accepting());
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_binary_repeat_intersections_mask_tokenizer(vocab.max_token_byte_len())
                .is_some()
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());

        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        for token in [1u32, 2, 3] {
            original_state.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
            assert_eq!(loaded_state.mask(), original_state.mask());
        }
    }

    #[test]
    fn dynamic_same_body_nonzero_repeat_intersection_factors_end_to_end() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |left_min, left_max, right_min, right_max| {
            let body = Expr::Choice(vec![
                Expr::U8Seq(b"ab".to_vec()),
                Expr::U8Seq(b"c".to_vec()),
            ]);
            GrammarDef {
                start: 0,
                rules: vec![Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                }],
                terminals: vec![Terminal::Expr {
                    id: 0,
                    expr: Expr::Intersect {
                        expr: Box::new(Expr::Repeat {
                            expr: Box::new(body.clone()),
                            min: left_min,
                            max: Some(left_max),
                        }),
                        intersect: Box::new(Expr::Repeat {
                            expr: Box::new(body),
                            min: right_min,
                            max: Some(right_max),
                        }),
                    },
                }],
                ..GrammarDef::default()
            }
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"c".to_vec()),
            (3, b"abc".to_vec()),
            (4, b"cab".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(900_000_000, 1_000_000_000, 999_999_997, 999_999_999),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(5, 13, 10, 12),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(dynamic.inner.tokenizer.num_states() < 32);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [1u32, 2, 3] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        for token in [1u32, 2, 3] {
            original_state.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
            assert_eq!(loaded_state.mask(), original_state.mask());
        }
    }

    #[test]
    fn dynamic_billion_nonzero_min_unit_repeat_uses_arithmetic_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |min, max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                    min,
                    max: Some(max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(500_000_000, 1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
            "one-byte nonzero-min repeats should use the O(1) arithmetic sidecar",
        );

        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(10, 20),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [2u32, 1, 0] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(!loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
        );
        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        for token in [2u32, 1, 0] {
            original_state.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
            assert_eq!(loaded_state.mask(), original_state.mask());
        }
    }

    #[test]
    fn dynamic_aligned_nonzero_unit_intersection_uses_arithmetic_runtime() {
        use crate::automata::regex::Expr;
        use crate::ds::u8set::U8Set;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |left_min, left_max, right_min, right_max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                        min: left_min,
                        max: Some(left_max),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
                        min: right_min,
                        max: Some(right_max),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"b".to_vec()),
            (1, b"bb".to_vec()),
            (2, b"bbb".to_vec()),
            (3, b"a".to_vec()),
            (4, b"c".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(500_000_000, 1_000_000_000, 600_000_000, 900_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(5, 20, 10, 15),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
            "aligned one-byte intersection should factor to the arithmetic repeat runtime",
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [2u32, 1, 0] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(!loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_giant_aligned_unit_empty_intersection_normalizes_before_validation() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |min| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min,
                        max: Some(1_000_000_000),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                        min,
                        max: Some(1_000_000_000),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);

        for min in [0usize, 1] {
            let dynamic =
                crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
                    grammar_for(min),
                    &vocab,
                    crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
                )
                .unwrap();
            assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
            assert!(
                dynamic
                    .inner
                    .tokenizer
                    .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                    .is_none(),
                "empty/epsilon normalization should remove the giant repeat entirely",
            );
            let saved = dynamic.save();
            let loaded = DynamicConstraint::load(&saved).unwrap();
            assert_eq!(loaded.start().mask(), dynamic.start().mask());
        }
    }

    #[test]
    fn dynamic_distinct_nonzero_min_giant_intersection_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 3,
                        max: Some(max),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"aa".to_vec())),
                        min: 2,
                        max: Some(max),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [
            vec![0u32],
            vec![1u32],
            vec![0u32, 0, 0],
            vec![1u32, 1],
            vec![0u32, 1, 0],
        ] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_multiple_large_plain_repeats_use_exact_virtual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(
            dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "both large terminals must stay on exact lazy virtual runtimes",
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![0u32], vec![1u32], vec![0u32, 1], vec![0u32, 0, 1]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_non_prefix_free_large_repeat_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    // `a | aa` is not prefix-free, so one scalar repeat
                    // coordinate is not a complete exact residual.
                    expr: Box::new(Expr::Choice(vec![
                        Expr::U8Seq(b"a".to_vec()),
                        Expr::U8Seq(b"aa".to_vec()),
                    ])),
                    min: 0,
                    max: Some(max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![0u32], vec![1u32], vec![0u32, 1], vec![1u32, 1, 0]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_general_residual_prunes_semantically_dead_token_prefix() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| {
            let giant_branch = |suffix: &[u8]| {
                Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"x".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                    Expr::U8Seq(suffix.to_vec()),
                ])
            };
            GrammarDef {
                start: 0,
                rules: vec![Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                }],
                terminals: vec![Terminal::Expr {
                    id: 0,
                    expr: Expr::Intersect {
                        expr: Box::new(Expr::Choice(vec![giant_branch(b"ab"), Expr::U8Seq(b"c".to_vec())])),
                        intersect: Box::new(Expr::Choice(vec![
                            giant_branch(b"ac"),
                            Expr::U8Seq(b"c".to_vec()),
                        ])),
                    },
                }],
                ..GrammarDef::default()
            }
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"c".to_vec()),
            (2, b"x".to_vec()),
            (3, b"b".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let start_mask = dynamic.start().mask();
        let allowed = |token: u32| {
            let word = token as usize / 32;
            let bit = token % 32;
            start_mask[word] & (1u32 << bit) != 0
        };
        assert!(!allowed(0), "a leads to the dead residual b ∩ c and must be pruned");
        assert!(allowed(1), "c is the one common accepted word");
        assert!(!allowed(2), "entering the giant x* branches can never reach a common suffix");

        let mut rejected = dynamic.start();
        assert!(rejected.commit_token(0).is_err());
        let mut accepted = dynamic.start();
        accepted.commit_token(1).unwrap();
        assert!(accepted.is_accepting());

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), start_mask);
    }

    #[test]
    fn dynamic_nested_large_repeat_suffix_compiles_lazily_and_exactly() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                    Expr::U8Seq(b"b".to_vec()),
                ]),
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![1u32], vec![0u32, 1], vec![0u32, 0, 1]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_nested_large_repeat_in_lazy_intersection_remains_supported() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::U8Seq(b"[".to_vec()),
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(max),
                        },
                        Expr::U8Seq(b"]".to_vec()),
                    ])),
                    intersect: Box::new(Expr::Seq(vec![
                        Expr::U8Seq(b"[".to_vec()),
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(7),
                        },
                        Expr::U8Seq(b"]".to_vec()),
                    ])),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"[".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"]".to_vec()),
            (4, b"[a]".to_vec()),
            (5, b"[aa]".to_vec()),
            (6, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(
            dynamic.inner.tokenizer.num_states() < 128,
            "nested giant repeat must stay on the existing lazy intersection path",
        );
        assert!(
            !dynamic
                .inner
                .tokenizer
                .has_virtual_binary_repeat_intersection(),
            "this regression must exercise the nested repeat-with-suffix lane, not the top-level virtual repeat product",
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [0, 2, 3] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }
    }

    #[test]
    fn dynamic_general_residual_repeat_supports_u32_max_bound() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"a".to_vec()),
                            Expr::U8Seq(b"aa".to_vec()),
                        ])),
                        min: 0,
                        max: Some(max),
                    },
                    Expr::U8Seq(b"z".to_vec()),
                ]),
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"z".to_vec()),
            (3, b"aaz".to_vec()),
        ]);

        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(u32::MAX as usize),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![2u32], vec![3u32], vec![0u32, 2], vec![1u32, 2]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_hybrid_unit_repeat_near_state_id_limit_uses_repeat_product() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar = GrammarDef {
            start: 0,
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(((1u64 << 31) - 1) as usize),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::U8Seq(b"b".to_vec()),
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"b".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        assert!(
            dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "hybrid physical states leave too little high-bit state-ID space for the arithmetic unit lane",
        );
        assert!(dynamic.inner.tokenizer.num_states() < 64);

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_top_level_large_repeat_inside_finite_intersection_stays_exact() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"ab".to_vec()),
                            Expr::U8Seq(b"c".to_vec()),
                        ])),
                        min: 3,
                        max: Some(max),
                    }),
                    // `abcabc` has the unique body factorization
                    // `ab · c · ab · c`, so it is accepted at count four.
                    // The finite right coordinate bounds generic product
                    // discovery independently of the giant repeat maximum.
                    intersect: Box::new(Expr::U8Seq(b"abcabc".to_vec())),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"abcabc".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"c".to_vec()),
            (3, b"abc".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(
            !dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "this case should use the generic product's virtual repeat coordinate, not a runtime sidecar",
        );
        assert!(dynamic.inner.tokenizer.num_states() < 64);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        for sequence in [vec![0u32], vec![1u32, 2, 1, 2], vec![3u32, 3]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
        let mut loaded_state = loaded.start();
        loaded_state.commit_token(0).unwrap();
        assert!(loaded_state.is_accepting());
    }

    #[test]
    fn dynamic_top_level_large_repeat_inside_cyclic_intersection_uses_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(max),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: None,
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![0u32], vec![1u32], vec![1u32, 1], vec![0u32, 1, 0]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_nested_giant_with_budgeted_other_repeat_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::ds::u8set::U8Set;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |left_max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(left_max),
                        },
                        Expr::U8Seq(b"b".to_vec()),
                    ])),
                    // This root repeat is above the generic product's direct
                    // threshold and would normally stay virtual. Its exact DFA
                    // is nevertheless tiny enough for the lazy intersection's
                    // explicitly budgeted ordinary side.
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                        min: 0,
                        max: Some(100),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"b".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(128),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(
            dynamic.inner.tokenizer.num_states() < 512,
            "the physical tokenizer must stay small independently of the giant max",
        );
        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            dynamic.inner.tokenizer.has_virtual_residual_runtime(),
            "nested giant intersections outside the arithmetic fast paths must use the general residual runtime",
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        for sequence in [vec![2u32], vec![3u32], vec![1u32, 2], vec![0u32, 0, 2]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let mut dynamic_boundary = dynamic.start();
        let mut oracle_boundary = oracle.start();
        for _ in 0..49 {
            dynamic_boundary.commit_token(1).unwrap();
            oracle_boundary.commit_token(1).unwrap();
        }
        assert_eq!(dynamic_boundary.is_accepting(), oracle_boundary.is_accepting());
        assert_eq!(dynamic_boundary.mask(), oracle_boundary.mask());
        assert!(!dynamic_boundary.is_accepting());

        // At 98 leading 'a' bytes, one more 'a' is still live but another
        // two-byte "aa" token would overshoot the right-hand 100-byte cap once
        // the required trailing 'b' is included.
        let mut dynamic_overrun = dynamic_boundary.clone();
        let mut oracle_overrun = oracle_boundary.clone();
        assert!(dynamic_overrun.commit_token(1).is_err());
        assert!(oracle_overrun.commit_token(1).is_err());

        dynamic_boundary.commit_token(0).unwrap();
        oracle_boundary.commit_token(0).unwrap();
        assert_eq!(dynamic_boundary.is_accepting(), oracle_boundary.is_accepting());
        assert_eq!(dynamic_boundary.mask(), oracle_boundary.mask());
        assert!(!dynamic_boundary.is_accepting());
        dynamic_boundary.commit_token(2).unwrap();
        oracle_boundary.commit_token(2).unwrap();
        assert!(dynamic_boundary.is_accepting());
        assert_eq!(dynamic_boundary.is_accepting(), oracle_boundary.is_accepting());
        assert_eq!(dynamic_boundary.mask(), oracle_boundary.mask());

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
        assert!(
            loaded.inner.tokenizer.has_virtual_residual_runtime(),
            "save/load must reconstruct the general residual runtime",
        );
        let mut loaded_state = loaded.start();
        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        loaded_state.commit_token(3).unwrap();
        dynamic_state.commit_token(3).unwrap();
        oracle_state.commit_token(3).unwrap();
        assert!(loaded_state.is_accepting());
        assert_eq!(loaded_state.is_accepting(), dynamic_state.is_accepting());
        assert_eq!(loaded_state.is_accepting(), oracle_state.is_accepting());
        assert_eq!(loaded_state.mask(), dynamic_state.mask());
        assert_eq!(loaded_state.mask(), oracle_state.mask());
    }

    #[test]
    fn dynamic_nested_giant_with_large_finite_other_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let mut long_body = vec![b'a'; 31];
        long_body.push(b'b');
        let grammar_for = |left_max, right_max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(left_max),
                        },
                        Expr::U8Seq(b"b".to_vec()),
                    ])),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(long_body.clone())),
                        min: 0,
                        max: Some(right_max),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, long_body.clone()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000, 4_095),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(64, 4),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        dynamic_state.commit_token(2).unwrap();
        oracle_state.commit_token(2).unwrap();
        assert!(dynamic_state.is_accepting());
        assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
        assert_eq!(dynamic_state.mask(), oracle_state.mask());

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_empty_giant_cyclic_intersection_becomes_dead_residual_proxy() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(max),
                        },
                        Expr::U8Seq(b"b".to_vec()),
                    ])),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: None,
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        assert!(dynamic.start().mask().iter().all(|&word| word == 0));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_two_nested_large_repeat_components_with_disjoint_delimiters_factor_to_empty() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let repeated_suffix = |suffix| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                    min: 0,
                    max: Some(10_000),
                },
                Expr::U8Seq(vec![suffix]),
            ])
        };
        let grammar = GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(repeated_suffix(b'b')),
                    intersect: Box::new(repeated_suffix(b'c')),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let start_mask = dynamic.start().mask();
        assert!(
            start_mask.iter().all(|&word| word == 0),
            "different uniquely-delimited literal suffixes make the intersection empty",
        );

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert_eq!(loaded.start().mask(), start_mask);
    }

    #[test]
    fn dynamic_positive_min_giant_suffix_with_disjoint_counts_factors_to_empty() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let component = |min, max| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                    min,
                    max: Some(max),
                },
                Expr::U8Seq(b"b".to_vec()),
            ])
        };
        let grammar = GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(component(5, 10_000)),
                    intersect: Box::new(component(1, 3)),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        assert!(dynamic.start().mask().iter().all(|&word| word == 0));
    }

    #[test]
    fn dynamic_multiple_giant_terminals_share_exact_virtual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |a_max, b_max| GrammarDef {
            start: 0,
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(a_max),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"a".to_vec()),
                            Expr::U8Seq(b"b".to_vec()),
                        ])),
                        min: 0,
                        max: Some(b_max),
                    },
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"ab".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000, 900_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8, 7),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(
            dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "multiple giant terminals must stay on shared lazy exact runtimes",
        );
        let exact_after_a = dynamic.inner.tokenizer.run(b"a");
        assert_eq!(
            exact_after_a.len(),
            2,
            "overlapping virtual terminals must retain distinct exact states",
        );
        let matched_after_a = exact_after_a
            .iter()
            .flat_map(|&state| dynamic.inner.tokenizer.matched_terminals(state))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(matched_after_a, [0u32, 1u32].into_iter().collect());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        for token in [0u32, 1, 2, 3, 4] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(dynamic.start().mask(), loaded.start().mask());
        for token in [0u32, 1] {
            let mut original_state = dynamic.start();
            let mut loaded_state = loaded.start();
            original_state.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
            assert_eq!(original_state.mask(), loaded_state.mask());
        }

        let transfer = dynamic.clone().into_saved();
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            transferred
                .inner
                .tokenizer
                .has_virtual_binary_repeat_intersection(),
            "v6 transfer load must reconstruct all shared lazy repeat runtimes",
        );
        assert_eq!(dynamic.start().mask(), transferred.start().mask());
        for token in [0u32, 1] {
            let mut original_state = dynamic.start();
            let mut transferred_state = transferred.start();
            original_state.commit_token(token).unwrap();
            transferred_state.commit_token(token).unwrap();
            assert_eq!(
                original_state.is_accepting(),
                transferred_state.is_accepting()
            );
            assert_eq!(original_state.mask(), transferred_state.mask());
        }
    }

    #[test]
    fn dynamic_mixed_specialized_and_general_giants_share_residual_runtime_family() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Seq(vec![
                        Expr::Repeat {
                            // Prefix ambiguity keeps this out of the simple
                            // bounded-repeat descriptor fast path.
                            expr: Box::new(Expr::Choice(vec![
                                Expr::U8Seq(b"x".to_vec()),
                                Expr::U8Seq(b"xx".to_vec()),
                            ])),
                            min: 0,
                            max: Some(max),
                        },
                        Expr::U8Seq(b"z".to_vec()),
                    ]),
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"x".to_vec()),
            (2, b"xx".to_vec()),
            (3, b"z".to_vec()),
            (4, b"xz".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![0u32], vec![3u32], vec![4u32], vec![2u32, 4]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_multiple_variable_width_giant_terminals_match_small_oracle() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |left_max, right_max| GrammarDef {
            start: 0,
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"a".to_vec()),
                            Expr::U8Seq(b"bb".to_vec()),
                        ])),
                        min: 0,
                        max: Some(left_max),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"a".to_vec()),
                            Expr::U8Seq(b"cc".to_vec()),
                        ])),
                        min: 0,
                        max: Some(right_max),
                    },
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"bb".to_vec()),
            (2, b"cc".to_vec()),
            (3, b"abb".to_vec()),
            (4, b"acc".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000, 900_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8, 7),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        let exact_after_a = dynamic.inner.tokenizer.run(b"a");
        assert_eq!(exact_after_a.len(), 2);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for token in 0u32..=4 {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert_eq!(dynamic.start().mask(), loaded.start().mask());
        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        original_state.commit_token(3).unwrap();
        loaded_state.commit_token(3).unwrap();
        assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(original_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_nested_giant_repeat_body_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |inner_max, outer_max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(inner_max),
                        },
                        Expr::U8Seq(b"b".to_vec()),
                    ])),
                    min: 0,
                    max: Some(outer_max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"aab".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000, 10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(4, 4),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![1u32], vec![2u32], vec![3u32], vec![4u32], vec![2u32, 1]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    #[ignore = "pre-release legacy dynamic artifact compatibility is not an acceptance requirement"]
    fn dynamic_constraint_loads_v11_payload_without_ignore_descriptor() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                ignore WS;
                t WS ::= " "+;
                nt start ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let original_mask = constraint.start().mask();
        let legacy = LegacyDynamicConstraintPayloadV11V3 {
            alternatives: vec![LegacyDynamicConstraintPayloadV11V2 {
                v1: LegacyDynamicConstraintPayloadV11V1 {
                    table: constraint.inner.table.clone_lr(),
                    terminal_display_names: constraint.inner.terminal_display_names.clone(),
                    tokenizer: constraint.inner.tokenizer.as_ref().clone(),
                    ignore_terminal: constraint.inner.ignore_terminal,
                    direct_regular_automaton: constraint.inner.direct_regular_automaton.clone(),
                    token_bytes: Arc::clone(&constraint.inner.token_bytes),
                },
                special_token_terminals: constraint.inner.special_token_terminals.clone(),
            }],
        };
        let payload = bincode::serialize(&legacy).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&11u16.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let loaded = DynamicConstraint::load(&bytes).unwrap();
        assert!(loaded.inner.ignore_expr.is_none());
        assert_eq!(loaded.start().mask(), original_mask);
    }


    #[test]
    #[ignore = "pre-release legacy dynamic artifact compatibility is not an acceptance requirement"]
    fn dynamic_constraint_loads_v12_payload_without_terminal_exprs() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_glrm_grammar(
            "start start; nt start ::= \"a\" \"b\";",
            &vocab,
        )
        .unwrap();
        let original_mask = constraint.start().mask();
        let legacy = LegacyDynamicConstraintPayloadV12V3 {
            alternatives: vec![LegacyDynamicConstraintPayloadV12V2 {
                v1: LegacyDynamicConstraintPayloadV12V1 {
                    table: constraint.inner.table.clone_lr(),
                    terminal_display_names: constraint.inner.terminal_display_names.clone(),
                    tokenizer: constraint.inner.tokenizer.as_ref().clone(),
                    ignore_terminal: constraint.inner.ignore_terminal,
                    direct_regular_automaton: constraint.inner.direct_regular_automaton.clone(),
                    token_bytes: Arc::clone(&constraint.inner.token_bytes),
                    ignore_expr: constraint.inner.ignore_expr.clone(),
                },
                special_token_terminals: constraint.inner.special_token_terminals.clone(),
            }],
        };
        let payload = bincode::serialize(&legacy).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&12u16.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let loaded = DynamicConstraint::load(&bytes).unwrap();
        assert!(loaded.inner.tokenizer.terminal_exprs().is_none());
        assert_eq!(loaded.start().mask(), original_mask);
    }


    #[test]
    fn dynamic_transfer_load_uses_prepared_vocab_and_matches_original() {
        let vocab = vocab();
        crate::compiler::constraint_possible_matches::prepare_vocab_for_dynamic_mask(&vocab);
        let original = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let original_mask = original.start().mask();
        let transfer = original.into_saved();
        assert!(transfer.starts_with(&DYNAMIC_TRANSFER_MAGIC));

        let loaded = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert_eq!(original_mask, loaded.start().mask());
    }

    #[test]
    fn dynamic_v20_round_trips_minimal_nonempty_projected_quotients() {
        // Serialization-unit intent: a minimal deterministic nonempty
        // projected-quotient sidecar round-trips through save/load/transfer
        // and preserves mask semantics exactly. The quotients come from the
        // existing per-terminal containment builder (same exactness proofs as
        // the shared-component selector) on a two-literal fixture — no
        // reset-dispatcher sharing, no optimizer-selection assumptions, no
        // JSON lowering, no compat-mode sensitivity.
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"d".to_vec()),
            (4, b"e".to_vec()),
            (5, b"f".to_vec()),
            (6, b"g".to_vec()),
        ]);
        let mut constraint = DynamicConstraint::from_glrm_grammar(
            "start start; t A ::= \"abcdef\"; t B ::= \"abcdeg\"; nt start ::= A | B;",
            &vocab,
        )
        .unwrap();
        let twin = constraint.clone();
        assert!(
            !twin
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared(),
            "reference twin must be quotient-free (unprepared)"
        );
        assert!(
            !twin
                .inner
                .dynamic_mask_vocab
                .has_projected_terminal_quotients(),
            "reference twin must be quotient-free (empty)"
        );
        let quotients = constraint
            .inner
            .tokenizer
            .build_terminal_projected_quotients_for_containment();
        assert_eq!(
            quotients.len(),
            2,
            "both terminals of the minimal fixture must yield exact quotients"
        );
        constraint
            .inner
            .dynamic_mask_vocab
            .set_projected_terminal_quotients(quotients);
        assert!(
            constraint
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert!(
            constraint
                .inner
                .dynamic_mask_vocab
                .has_projected_terminal_quotients()
        );
        let expected = constraint
            .inner
            .dynamic_mask_vocab
            .projected_terminal_quotients_for_artifact();
        assert!(!expected.is_empty());

        // See the Snowplow sidecar test below: this is a test-only mutation of
        // otherwise immutable post-build persistence metadata.
        constraint.external_vocab_artifact_cache = None;
        constraint.cache_external_vocab_artifact_for_save();

        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            DYNAMIC_CONSTRAINT_VERSION,
        );
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(
            loaded
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert!(
            loaded
                .inner
                .dynamic_mask_vocab
                .has_projected_terminal_quotients()
        );
        assert_eq!(
            bincode::serialize(
                &loaded
                    .inner
                    .dynamic_mask_vocab
                    .projected_terminal_quotients_for_artifact()
            )
            .unwrap(),
            bincode::serialize(&expected).unwrap(),
        );

        let transfer = constraint.clone().into_saved();
        assert_eq!(
            u16::from_le_bytes([transfer[8], transfer[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            transferred
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert!(
            transferred
                .inner
                .dynamic_mask_vocab
                .has_projected_terminal_quotients()
        );
        assert_eq!(
            bincode::serialize(
                &transferred
                    .inner
                    .dynamic_mask_vocab
                    .projected_terminal_quotients_for_artifact()
            )
            .unwrap(),
            bincode::serialize(&expected).unwrap(),
        );

        // Semantic exactness: prepared quotients must not change the accepted
        // language or any mask. Differential of the quotient-free twin against
        // the quotient-bearing original plus both reloaded artifacts over
        // accept/reject paths.
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![0],
            vec![0, 1],
            vec![0, 1, 2, 3, 4, 5],
            vec![0, 1, 2, 3, 4, 6],
            vec![1],
            vec![0, 0],
        ];
        for path in &paths {
            let mut states = [
                twin.start(),
                constraint.start(),
                loaded.start(),
                transferred.start(),
            ];
            let mut failed = false;
            for &token in path {
                let masks: Vec<_> = states.iter().map(|state| state.mask()).collect();
                for (index, mask) in masks.iter().enumerate().skip(1) {
                    assert_eq!(
                        *mask, masks[0],
                        "mask agreement at {path:?} before token {token} (state {index})",
                    );
                }
                let results: Vec<bool> = states
                    .iter_mut()
                    .map(|state| state.commit_token(token).is_ok())
                    .collect();
                for (index, result) in results.iter().enumerate().skip(1) {
                    assert_eq!(
                        *result, results[0],
                        "commit agreement at {path:?} token {token} (state {index})",
                    );
                }
                if !results[0] {
                    // All four agree on rejection; the API's post-failure
                    // guarantee is `is_rejected`, so assert that (not masks or
                    // acceptance) and stop this path.
                    for (index, state) in states.iter().enumerate() {
                        assert!(
                            state.is_rejected(),
                            "rejected state {index} at {path:?} token {token} must report is_rejected",
                        );
                    }
                    failed = true;
                    break;
                }
            }
            if failed {
                continue;
            }
            // Every commit succeeded: compare the post-path masks (initial and
            // after-every-successful-commit coverage) and acceptance.
            let masks: Vec<_> = states.iter().map(|state| state.mask()).collect();
            for (index, mask) in masks.iter().enumerate().skip(1) {
                assert_eq!(
                    *mask, masks[0],
                    "final mask agreement at {path:?} (state {index})",
                );
            }
            let accepting: Vec<bool> =
                states.iter().map(|state| state.is_accepting()).collect();
            for (index, value) in accepting.iter().enumerate().skip(1) {
                assert_eq!(
                    *value, accepting[0],
                    "acceptance agreement at {path:?} (state {index})",
                );
            }
        }
        // Explicit acceptance/rejection pins on every artifact (not only
        // cross-artifact equality), so the differential above is not vacuous.
        for (label, artifact) in [
            ("twin", &twin),
            ("original", &constraint),
            ("loaded", &loaded),
            ("transferred", &transferred),
        ] {
            for path in [vec![0u32, 1, 2, 3, 4, 5], vec![0, 1, 2, 3, 4, 6]] {
                let mut state = artifact.start();
                for &token in &path {
                    state.commit_token(token).unwrap();
                }
                assert!(state.is_accepting(), "{label} must accept {path:?}");
            }
            let mut state = artifact.start();
            assert!(
                !state.is_accepting(),
                "{label} must not accept the empty prefix"
            );
            assert!(
                state.commit_token(1).is_err(),
                "{label} must reject a leading b"
            );
            let mut state = artifact.start();
            state.commit_token(0).unwrap();
            assert!(
                !state.is_accepting(),
                "{label} must not accept the proper prefix [a]"
            );
            assert!(
                state.commit_token(0).is_err(),
                "{label} must reject the aa divergence"
            );
        }
    }

    #[test]
    fn dynamic_v20_round_trips_snowplow_optimizer_selected_quotients() {
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let (schema, vocab) = projected_quotient_fixture::schema_and_vocab();
        // Integration tolerance: whatever the shared-component optimizer
        // selects for Snowplow (currently nothing — the lexer isolates this
        // fixture's large terminals), the sidecar round-trips and masks agree
        // end to end. Nonempty wire-format coverage lives in the minimal
        // unit test above; this test must NOT pin the optimizer's selection.
        // Do not restore production sharing just to make Snowplow nonempty.
        let schema_value: serde_json::Value = serde_json::from_str(schema).unwrap();
        let named = crate::import::json_schema::schema_to_named_grammar(&schema_value).unwrap();
        let mut factored = crate::grammar::factoring::factor_named_grammar(named);
        crate::import::json_schema::prepare_named_grammar(&mut factored).unwrap();
        let grammar = crate::grammar::ast::lower(&factored).unwrap();
        let mut constraint = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::LegacyRowBisim,
        )
        .unwrap();
        let quotients = constraint
            .inner
            .tokenizer
            .build_shared_component_terminal_projected_quotients(256);
        constraint
            .inner
            .dynamic_mask_vocab
            .set_projected_terminal_quotients(quotients);
        let expected = constraint
            .inner
            .dynamic_mask_vocab
            .projected_terminal_quotients_for_artifact();

        // The private mutation above occurs after public finalization froze the
        // canonical transfer artifact. Refresh it explicitly for wire-format
        // coverage; ordinary callers cannot mutate a compiled constraint.
        constraint.external_vocab_artifact_cache = None;
        constraint.cache_external_vocab_artifact_for_save();

        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            DYNAMIC_CONSTRAINT_VERSION,
        );
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(loaded.inner.dynamic_mask_vocab.projected_terminal_quotients_prepared());
        assert_eq!(
            bincode::serialize(
                &loaded
                    .inner
                    .dynamic_mask_vocab
                    .projected_terminal_quotients_for_artifact()
            )
            .unwrap(),
            bincode::serialize(&expected).unwrap(),
        );
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let transfer = constraint.clone().into_saved();
        assert_eq!(
            u16::from_le_bytes([transfer[8], transfer[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            transferred
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert_eq!(
            bincode::serialize(
                &transferred
                    .inner
                    .dynamic_mask_vocab
                    .projected_terminal_quotients_for_artifact()
            )
            .unwrap(),
            bincode::serialize(&expected).unwrap(),
        );
        assert_eq!(transferred.start().mask(), constraint.start().mask());

        let mut original_state = constraint.start();
        let mut loaded_state = loaded.start();
        let mut transferred_state = transferred.start();
        for &token in projected_quotient_fixture::replay_ids() {
            let expected = original_state.mask();
            assert_eq!(loaded_state.mask(), expected, "self-contained load before token {token}");
            assert_eq!(
                transferred_state.mask(),
                expected,
                "transfer load before token {token}"
            );
            original_state.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            transferred_state.commit_token(token).unwrap();
        }
        assert_eq!(loaded_state.mask(), original_state.mask());
        assert_eq!(transferred_state.mask(), original_state.mask());
    }

    #[test]
    fn dynamic_v20_round_trips_prepared_empty_projected_terminal_quotients() {
        let vocab = vocab();
        let mut constraint = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        constraint
            .inner
            .dynamic_mask_vocab
            .set_projected_terminal_quotients(Vec::new());
        assert!(constraint.inner.dynamic_mask_vocab.projected_terminal_quotients_prepared());
        assert!(!constraint.inner.dynamic_mask_vocab.has_projected_terminal_quotients());

        // See the nonempty sidecar test above: this is a test-only mutation of
        // otherwise immutable post-build persistence metadata.
        constraint.external_vocab_artifact_cache = None;
        constraint.cache_external_vocab_artifact_for_save();

        let loaded = DynamicConstraint::load(&constraint.save()).unwrap();
        assert!(loaded.inner.dynamic_mask_vocab.projected_terminal_quotients_prepared());
        assert!(!loaded.inner.dynamic_mask_vocab.has_projected_terminal_quotients());
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let transfer = constraint.clone().into_saved();
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            transferred
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert!(!transferred.inner.dynamic_mask_vocab.has_projected_terminal_quotients());
        assert_eq!(transferred.start().mask(), constraint.start().mask());
    }

    #[test]
    fn current_dynamic_transfer_preserves_unprepared_projected_terminal_quotients() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
start start;
t A ::= /a{0,1000000000}/;
nt start ::= A;
"#,
            &vocab,
        )
        .unwrap();
        assert!(constraint.inner.tokenizer.has_any_virtual_runtime());
        assert!(
            !constraint
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared(),
            "fresh dynamic compilation should defer projected-terminal proof artifacts",
        );

        let transfer = constraint.into_saved();
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            !transferred
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared(),
            "transfer load must preserve unprepared-empty rather than converting it to prepared-empty",
        );
    }

    #[test]
    fn self_contained_dynamic_load_with_vocab_shares_exact_vocab() {
        let vocab = vocab();
        let original = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let loaded = DynamicConstraint::load_with_vocab(&original.save(), &vocab).unwrap();

        assert!(std::sync::Arc::ptr_eq(
            &loaded.inner.token_bytes,
            &vocab.entries_arc(),
        ));
        assert!(std::sync::Arc::ptr_eq(
            &loaded
                .inner
                .late_bind_vocab
                .get()
                .expect("dynamic load_with_vocab should seed late-bind vocab")
                .entries_arc(),
            &vocab.entries_arc(),
        ));
    }

    #[test]
    fn dynamic_transfer_loads_v1_payload_without_ignore_descriptor() {
        let vocab = vocab();
        crate::compiler::constraint_possible_matches::prepare_vocab_for_dynamic_mask(&vocab);
        let original = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                ignore WS;
                t WS ::= " "+;
                nt start ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let original_mask = original.start().mask();
        let legacy = LegacyDynamicConstraintTransferPayloadV1 {
            alternatives: vec![LegacyDynamicConstraintTransferAlternativeV1 {
                table: original.inner.table.clone_lr(),
                terminal_display_names: original.inner.terminal_display_names.clone(),
                tokenizer: original.inner.tokenizer.as_ref().clone(),
                ignore_terminal: original.inner.ignore_terminal,
                direct_regular_automaton: original.inner.direct_regular_automaton.clone(),
                special_token_terminals: original.inner.special_token_terminals.clone(),
            }],
        };
        let payload = bincode::serialize(&legacy).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_MAGIC);
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let loaded = DynamicConstraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert!(loaded.inner.ignore_expr.is_none());
        assert_eq!(loaded.start().mask(), original_mask);
    }


    #[test]
    fn dynamic_transfer_loads_v2_payload_without_terminal_exprs() {
        let vocab = vocab();
        crate::compiler::constraint_possible_matches::prepare_vocab_for_dynamic_mask(&vocab);
        let original = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let original_mask = original.start().mask();
        let legacy = LegacyDynamicConstraintTransferPayloadV2 {
            alternatives: vec![LegacyDynamicConstraintTransferAlternativeV2 {
                table: original.inner.table.clone_lr(),
                terminal_display_names: original.inner.terminal_display_names.clone(),
                tokenizer: original.inner.tokenizer.as_ref().clone(),
                ignore_terminal: original.inner.ignore_terminal,
                direct_regular_automaton: original.inner.direct_regular_automaton.clone(),
                special_token_terminals: original.inner.special_token_terminals.clone(),
                ignore_expr: original.inner.ignore_expr.clone(),
            }],
        };
        let payload = bincode::serialize(&legacy).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_MAGIC);
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let loaded = DynamicConstraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert!(loaded.inner.tokenizer.terminal_exprs().is_none());
        assert_eq!(loaded.start().mask(), original_mask);
    }


    #[test]
    fn dynamic_transfer_loads_v3_mask_quotient_payload_without_terminal_exprs() {
        let vocab = vocab();
        crate::compiler::constraint_possible_matches::prepare_vocab_for_dynamic_mask(&vocab);
        let original = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let original_mask = original.start().mask();
        let mask_quotient = original
            .inner
            .dynamic_mask_vocab
            .mask_tokenizer_quotient_for_transfer();
        let legacy = LegacyDynamicConstraintTransferPayloadV3 {
            alternatives: vec![LegacyDynamicConstraintTransferAlternativeV3 {
                table: original.inner.table.clone_lr(),
                terminal_display_names: original.inner.terminal_display_names.clone(),
                tokenizer: original.inner.tokenizer.as_ref().clone(),
                ignore_terminal: original.inner.ignore_terminal,
                direct_regular_automaton: original.inner.direct_regular_automaton.clone(),
                special_token_terminals: original.inner.special_token_terminals.clone(),
                ignore_expr: original.inner.ignore_expr.clone(),
                mask_tokenizer: mask_quotient
                    .as_ref()
                    .map(|(tokenizer, _)| CompactTransferTokenizer(tokenizer.clone())),
                full_to_mask_state: mask_quotient.map_or_else(Vec::new, |(_, mapping)| mapping),
            }],
        };
        let payload = bincode::serialize(&legacy).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_MAGIC);
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_VERSION_V3.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let loaded = DynamicConstraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert!(loaded.inner.tokenizer.terminal_exprs().is_none());
        assert_eq!(loaded.start().mask(), original_mask);
    }

    #[test]
    fn precompiled_dynamic_artifacts_require_rebuild() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let mut bytes = constraint.save();
        bytes[8..10].copy_from_slice(&8u16.to_le_bytes());
        let error = DynamicConstraint::load(&bytes).unwrap_err().to_string();
        assert!(error.contains("removed precompiled token programs"));
    }

    #[test]
    fn direct_regular_dynamic_constraint_uses_dynamic_runtime() {
        let vocab = vocab();
        let mut grammar = String::from("start: r0\n");
        for index in 0..63 {
            grammar.push_str(&format!("r{index}: \"a\" r{}\n", index + 1));
        }
        grammar.push_str("r63: \"b\"\n");

        let normal = compile_compressed_static(&grammar, &vocab);
        let dynamic = compile_compressed_dynamic(&grammar, &vocab);
        assert_eq!(dynamic.inner.table.num_rules, 0);
        assert!(dynamic.inner.uses_dynamic_runtime());
        assert_eq!(normal.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn compressed_static_unions_mixed_l1_l2p_token_lengths() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let mut grammar = String::from("start: r0\n");
        for index in 0..63 {
            grammar.push_str(&format!("r{index}: \"a\" r{}\n", index + 1));
        }
        grammar.push_str("r63: \"a\"\n");

        let static_constraint = compile_compressed_static(&grammar, &vocab);
        let dynamic_constraint = compile_compressed_dynamic(&grammar, &vocab);
        assert!(!static_constraint.uses_dynamic_runtime());
        assert_eq!(static_constraint.table.num_rules, 0);
        assert!(
            !static_constraint.parser_top_accept_parts.is_empty(),
            "regression must exercise the direct parser-acceptance summaries",
        );

        let static_mask = static_constraint.start().mask();
        assert_ne!(static_mask[0] & (1 << 0), 0, "single-byte token must be allowed");
        assert_ne!(static_mask[0] & (1 << 1), 0, "two-byte token must be allowed");
        assert_eq!(static_mask, dynamic_constraint.start().mask());
    }

    #[test]
    fn compressed_right_linear_plus_loop_commits_terminal_at_token_boundary() {
        let mut grammar = String::from("start: H s0\nplus_line: PLUS_LINE\n");
        let n = 15;
        for i in 0..n {
            grammar.push_str(&format!(
                "s{i}: line{i}{}\n",
                if i + 1 < n {
                    format!(" | s{}", i + 1)
                } else {
                    String::new()
                }
            ));
        }
        for i in 0..n {
            grammar.push_str(&format!(
                "line{i}: plus_line* SRC_{i} plus_line* {}\n",
                if i + 1 < n {
                    format!("line{}?", i + 1)
                } else {
                    String::new()
                }
            ));
        }
        grammar.push_str("H: \"h\\n\"\nPLUS_LINE: /\\+[^\\n\\r]*\\n/\n");
        for i in 0..n {
            grammar.push_str(&format!("SRC_{i}: \" a{i}\\n\"\n"));
        }
        let vocab = Vocab::new(vec![
            (0, b"h\n".to_vec()),
            (1, b"+x".to_vec()),
            (2, b"\n".to_vec()),
            (3, b" a0\n".to_vec()),
        ]);
        let constraint = compile_compressed_static(&grammar, &vocab);
        assert!(!constraint.uses_dynamic_runtime());
        let mut state = constraint.start();

        state.commit_token(0).unwrap();
        state.commit_token(1).unwrap();
        assert_ne!(state.mask()[0] & (1 << 2), 0);
        state.commit_token(2).unwrap();
        assert_ne!(state.mask()[0] & (1 << 3), 0);
        state.commit_token(3).unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn direct_regular_static_constraint_roundtrips_and_matches_dynamic() {
        let vocab = vocab();
        let mut grammar = String::from("start: r0\n");
        for index in 0..63 {
            grammar.push_str(&format!("r{index}: \"a\" r{}\n", index + 1));
        }
        grammar.push_str("r63: \"b\"\n");

        let constraint = crate::Constraint::from_lark(&grammar, &vocab).unwrap();
        assert!(!constraint.uses_dynamic_runtime());
        assert!(constraint.possible_matches_complete);
        let dynamic = compile_compressed_dynamic(&grammar, &vocab);
        assert!(!dynamic.inner.possible_matches_complete);

        let mut static_state = constraint.start();
        let mut dynamic_state = dynamic.start();
        assert_eq!(static_state.mask(), dynamic_state.mask());
        assert_eq!(static_state.forced(), dynamic_state.forced());

        for token in [3, 3] {
            static_state.commit_token(token).unwrap();
            dynamic_state.commit_token(token).unwrap();
            assert_eq!(static_state.mask(), dynamic_state.mask());
            assert_eq!(static_state.is_accepting(), dynamic_state.is_accepting());
        }

        let before_third = static_state.mask();
        let checkpoint = static_state.clone();
        static_state.commit_token(0).unwrap();
        dynamic_state.commit_token(0).unwrap();
        let after_third = static_state.mask();
        assert_eq!(after_third, dynamic_state.mask());
        static_state = checkpoint;
        assert_eq!(static_state.mask(), before_third);
        static_state.commit_token(0).unwrap();
        assert_eq!(static_state.mask(), after_third);

        let loaded = crate::Constraint::load(&constraint.save()).unwrap();
        assert!(!loaded.uses_dynamic_runtime());
        let mut loaded_state = loaded.start();
        let mut original_state = constraint.start();
        assert_eq!(loaded_state.mask(), original_state.mask());
        for token in [3, 3, 0] {
            loaded_state.commit_token(token).unwrap();
            original_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.mask(), original_state.mask());
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
        }
    }

    #[test]
    fn non_regular_constraint_keeps_static_backend() {
        let vocab = vocab();
        let constraint = crate::Constraint::from_ebnf(
            "start ::= 'a' start 'b' | ''",
            &vocab,
        )
        .unwrap();
        assert!(!constraint.uses_dynamic_runtime());
    }

    #[test]
    fn dynamic_forced_uses_dynamic_masks() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let constraint = DynamicConstraint::from_ebnf("start ::= 'a'", &vocab).unwrap();
        assert_eq!(constraint.start().forced(), vec![0]);
    }
}
