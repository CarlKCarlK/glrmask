//! Dynamic constraints, exact union state, and current artifact transport.
//!
//! Only the current standalone and transfer formats are supported. Serialized
//! certificates and vocabulary identity are validated before runtime reuse.

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
const DYNAMIC_CONSTRAINT_VERSION: u16 = 20;
const DYNAMIC_CONSTRAINT_HEADER_LEN: usize = DYNAMIC_CONSTRAINT_MAGIC.len() + 2 + 8;
const DYNAMIC_TRANSFER_MAGIC: [u8; 8] = *b"GLRDXF\0\0";
const DYNAMIC_TRANSFER_VERSION: u16 = 13;
// The transfer envelope has six byte sections per alternative. Large immutable
// views borrow this backing instead of round-tripping through temporary Vecs.
const TRANSFER_PAYLOAD_HEADER_LEN: usize = 8;
const TRANSFER_ALT_DESCRIPTOR_LEN: usize = 6 * 8;

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

impl<'de> serde::Deserialize<'de> for CompactTransferTokenizer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        crate::automata::lexer::tokenizer::compact_artifact_serde::deserialize(deserializer)
            .map(Self)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicCore {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    ignore_expr: Option<Expr>,
    #[serde(with = "compressed_terminal_exprs_serde")]
    terminal_exprs: Option<Vec<Expr>>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
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
struct DynamicAlternative {
    core: DynamicCore,
    late_grammar_slots: Vec<crate::runtime::LateGrammarSlot>,
    virtual_runtimes: Vec<VirtualTokenizerRuntimeMetadata>,
    terminal_observation_classes: Vec<(TerminalID, Vec<u32>)>,
    projected_terminal_quotients: Vec<(TerminalID, TerminalProjectedQuotient)>,
    boundary_trigger: DynamicBoundaryTriggerWire,
    /// The recursive provider tree is authoritative when present; the core
    /// still validates vocabulary identity and attached lexer certificates.
    recursive_constraint_artifact: Option<Vec<u8>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicArtifact {
    alternatives: Vec<DynamicAlternative>,
    /// Vocabulary-only runtime index shared by every union alternative.
    dynamic_mask_vocab: Option<crate::runtime::DynamicMaskVocabArtifact>,
}

/// Small structured metadata for one transfer alternative. Large runtime
/// byte sections deliberately live outside this bincode object.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct TransferMetadata {
    terminal_display_names: Vec<String>,
    ignore_terminal: Option<TerminalID>,
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    ignore_expr: Option<Expr>,
    mask_tokenizer: Option<CompactTransferTokenizer>,
    full_to_mask_state: Vec<u32>,
    virtual_runtimes: Vec<VirtualTokenizerRuntimeMetadata>,
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
    grammar_quotient_vocab: Option<crate::runtime::DynamicMaskVocabArtifact>,
    prepared_master_proofs: crate::runtime::PreparedMasterProofArtifact,
    virtual_residual_master_slice_artifacts:
        Vec<crate::automata::lexer::tokenizer::VirtualResidualMasterSliceArtifact>,
}

struct TransferSections {
    table: Vec<u8>,
    tokenizer: Vec<u8>,
    terminal_exprs_compressed: Vec<u8>,
    recursive_constraint_artifact: Vec<u8>,
    metadata: Vec<u8>,
    virtual_residual_wire: Vec<u8>,
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
        Self::from_core(
            DynamicCore {
                table,
                terminal_display_names,
                tokenizer,
                ignore_terminal,
                direct_regular_automaton,
                token_bytes: vocab.entries_arc(),
                ignore_expr,
                terminal_exprs,
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
        let payload = DynamicCore {
                table,
                terminal_display_names,
                tokenizer,
                ignore_terminal,
                direct_regular_automaton,
                token_bytes: vocab.entries_arc(),
                ignore_expr,
                terminal_exprs,
                special_token_terminals,
            };
        Self {
            inner: Self::constraint_from_core(
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

        let mut result = Self::from_core(
            DynamicCore {
                table,
                terminal_display_names,
                tokenizer,
                ignore_terminal,
                direct_regular_automaton: None,
                token_bytes: vocab.entries_arc(),
                ignore_expr,
                terminal_exprs,
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

    fn from_core(
        payload: DynamicCore,
        dynamic_mask_vocab: DynamicMaskVocab,
    ) -> Self {
        let mut inner = Self::constraint_from_core(
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

    fn constraint_from_core(
        mut payload: DynamicCore,
        dynamic_mask_vocab: DynamicMaskVocab,
    ) -> Constraint {
        let special_token_terminals = payload.special_token_terminals;
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
            range_final_token_sets: Default::default(),
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

    fn core_for_constraint(constraint: &Constraint) -> DynamicCore {
        let retain_terminal_exprs = std::env::var("GLRMASK_DYNAMIC_TRANSFER_TERMINAL_EXPRS")
            .ok()
            .is_none_or(|value| !matches!(value.trim(), "0" | "false" | "no" | "off"));
        let terminal_exprs = retain_terminal_exprs
            .then(|| constraint.tokenizer.terminal_exprs().map(ToOwned::to_owned))
            .flatten();
        DynamicCore {
                table: constraint.table.clone(),
                terminal_display_names: constraint.terminal_display_names.clone(),
                tokenizer: constraint.tokenizer.as_ref().clone(),
                ignore_terminal: constraint.ignore_terminal,
                direct_regular_automaton: constraint.direct_regular_automaton.clone(),
                token_bytes: Arc::clone(&constraint.token_bytes),
                ignore_expr: constraint.ignore_expr.clone(),
                terminal_exprs,
                special_token_terminals: constraint.special_token_terminals.clone(),
            }
    }

    fn alternative_for_constraint(constraint: &Constraint) -> DynamicAlternative {
        DynamicAlternative {
            core: Self::core_for_constraint(constraint),
            late_grammar_slots: constraint.late_grammar_slots.clone(),
            virtual_runtimes: constraint.tokenizer.virtual_runtime_metadata(),
            terminal_observation_classes: constraint.dynamic_mask_vocab
                .terminal_observation_classes_for_artifact(),
            projected_terminal_quotients: constraint.dynamic_mask_vocab
                .projected_terminal_quotients_for_artifact(),
            boundary_trigger: DynamicBoundaryTriggerWire::from_trigger(&constraint.boundary_trigger),
            recursive_constraint_artifact: constraint.uses_compact_segmented_parser_runtime()
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
        // Calling the setter even for an empty row set is intentional: current formats
        // use empty+prepared to mean compile-time analysis completed and found
        // no useful quotient, so runtime load must not repeat that work.
        constraint
            .dynamic_mask_vocab
            .set_projected_terminal_quotients(rows);
        Ok(())
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

    fn transfer_sections_for_constraint(
        constraint: &Constraint,
    ) -> TransferSections {
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
        let metadata = TransferMetadata {
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
                let wire = crate::runtime::persistence::encode_static_virtual_residual_mask_wire_with_fallback(
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
                    let wire = crate::runtime::persistence::encode_static_virtual_residual_mask_wire_with_fallback(
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

        TransferSections {
            table,
            tokenizer,
            terminal_exprs_compressed,
            recursive_constraint_artifact,
            metadata: bincode::serialize(&metadata)
                .expect("dynamic transfer metadata serialization should succeed"),
            virtual_residual_wire,
        }
    }

    fn build_external_vocab_artifact_bytes(&self) -> Vec<u8> {
        let profile_transfer = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile_transfer.then(std::time::Instant::now);
        let sections_started = profile_transfer.then(std::time::Instant::now);
        let alternatives = std::iter::once(&self.inner)
            .chain(self.alternatives.iter())
            .map(Self::transfer_sections_for_constraint)
            .collect::<Vec<_>>();
        let sections_ms = sections_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let alternative_count = u32::try_from(alternatives.len())
            .expect("dynamic transfer alternative count exceeds u32");
        let descriptor_bytes = alternatives
            .len()
            .checked_mul(TRANSFER_ALT_DESCRIPTOR_LEN)
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
        let payload_capacity = TRANSFER_PAYLOAD_HEADER_LEN
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
                descriptor_start + index * TRANSFER_ALT_DESCRIPTOR_LEN;
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

    fn decode_transfer_sections(bytes: &[u8], vocab: &Vocab) -> crate::Result<Self> {
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
            crate::GlrMaskError::Serialization("missing dynamic transfer payload".to_owned())
        })?;
        if payload.len() < TRANSFER_PAYLOAD_HEADER_LEN {
            return Err(crate::GlrMaskError::Serialization(
                "truncated dynamic transfer payload header".to_owned(),
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
                "dynamic transfer artifact has no alternatives".to_owned(),
            ));
        }
        let flags = u32::from_le_bytes(
            payload[4..8]
                .try_into()
                .expect("dynamic transfer flags have fixed width"),
        );
        if flags != 0 {
            return Err(crate::GlrMaskError::Serialization(
                "unsupported dynamic transfer flags".to_owned(),
            ));
        }
        let descriptor_bytes = alternative_count
            .checked_mul(TRANSFER_ALT_DESCRIPTOR_LEN)
            .ok_or_else(|| {
                crate::GlrMaskError::Serialization(
                    "dynamic transfer descriptor size overflow".to_owned(),
                )
            })?;
        let descriptor_end = payload_start
            .checked_add(TRANSFER_PAYLOAD_HEADER_LEN)
            .and_then(|value| value.checked_add(descriptor_bytes))
            .ok_or_else(|| {
                crate::GlrMaskError::Serialization(
                    "dynamic transfer descriptor range overflow".to_owned(),
                )
            })?;
        if descriptor_end > backing.len() {
            return Err(crate::GlrMaskError::Serialization(
                "truncated dynamic transfer descriptors".to_owned(),
            ));
        }

        let mut descriptors = Vec::<[usize; 6]>::with_capacity(alternative_count);
        let descriptor_start = payload_start + TRANSFER_PAYLOAD_HEADER_LEN;
        for index in 0..alternative_count {
            let mut pos = descriptor_start + index * TRANSFER_ALT_DESCRIPTOR_LEN;
            let mut lengths = [0usize; 6];
            for length in &mut lengths {
                let raw = u64::from_le_bytes(
                    backing[pos..pos + 8]
                        .try_into()
                        .expect("validated v12 descriptor has fixed-width field"),
                );
                *length = usize::try_from(raw).map_err(|_| {
                    crate::GlrMaskError::Serialization(
                        "dynamic transfer section length does not fit platform".to_owned(),
                    )
                })?;
                pos += 8;
            }
            if lengths[0] == 0 || lengths[1] == 0 || lengths[4] == 0 {
                return Err(crate::GlrMaskError::Serialization(
                    "dynamic transfer has an empty required section".to_owned(),
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
                "invalid dynamic transfer section lengths".to_owned(),
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
            let metadata: TransferMetadata = bincode::deserialize(&backing[metadata_range])
                .map_err(|err| crate::GlrMaskError::Serialization(format!(
                    "invalid dynamic transfer metadata: {err}"
                )))?;
            let prepared_master_proofs = metadata.prepared_master_proofs;
            let virtual_residual_master_slice_artifacts = metadata.virtual_residual_master_slice_artifacts;
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
            let (table_result, tokenizer_result) = rayon::join(
                || crate::compiler::glr::table::artifact_serde::from_compact_bytes_deferred_backed(
                    &backing[table_range], Arc::clone(&backing), table_start,
                ),
                || crate::automata::lexer::tokenizer::artifact_serde::from_fast_bytes_backed(
                    &backing[tokenizer_range], Arc::clone(&backing), tokenizer_start,
                ),
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
                    crate::runtime::persistence::decode_static_virtual_residual_mask_wire(
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
            let mut inner = Self::constraint_from_core(
                DynamicCore {
                table: decoded_table.table,
                terminal_display_names: metadata.terminal_display_names,
                tokenizer,
                ignore_terminal: metadata.ignore_terminal,
                direct_regular_automaton: metadata.direct_regular_automaton,
                token_bytes: Arc::clone(&token_bytes),
                ignore_expr: metadata.ignore_expr,
                terminal_exprs: None,
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

    fn from_artifact(artifact: DynamicArtifact) -> crate::Result<Self> {
        let Some(first) = artifact.alternatives.first() else {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        };
        if artifact.dynamic_mask_vocab.is_some()
            && artifact.alternatives.iter().any(|alternative|
                alternative.core.token_bytes != first.core.token_bytes)
        {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic artifact shares a vocabulary index across alternatives with different token bytes"
                    .to_owned(),
            ));
        }
        let mut shared_vocab = artifact.dynamic_mask_vocab
            .map(DynamicMaskVocab::from_artifact).transpose()
            .map_err(crate::GlrMaskError::Serialization)?;
        if let Some(vocab) = &mut shared_vocab {
            if !vocab.matches_token_bytes_exact(&first.core.token_bytes) {
                return Err(crate::GlrMaskError::Serialization(
                    "dynamic artifact vocabulary index does not match serialized token bytes".to_owned(),
                ));
            }
            vocab.restore_root_layout_metadata_from_token_bytes(&first.core.token_bytes);
        }

        let mut alternatives = Vec::with_capacity(artifact.alternatives.len());
        for DynamicAlternative {
            mut core, late_grammar_slots, virtual_runtimes,
            terminal_observation_classes, projected_terminal_quotients,
            boundary_trigger, recursive_constraint_artifact,
        } in artifact.alternatives {
            let exprs = core.terminal_exprs.clone();
            core.tokenizer.restore_terminal_exprs_with_virtual_runtime_metadata(
                exprs, &virtual_runtimes, false,
            ).map_err(crate::GlrMaskError::Serialization)?;
            let vocab = shared_vocab.as_ref().map(|shared| {
                let mut fresh = shared.fresh_runtime_instance();
                // Share immutable lexer certificates, never mutable execution caches.
                fresh.inherit_dynamic_lexer_metadata_from(shared);
                fresh
            }).unwrap_or_default();
            let mut inner = Self::constraint_from_core(core, vocab);
            inner.late_grammar_slots = late_grammar_slots;
            Self::restore_terminal_observation_classes(&mut inner, terminal_observation_classes)?;
            // An empty row set still means prepared: do not rediscover it at runtime.
            Self::restore_projected_terminal_quotients(&mut inner, projected_terminal_quotients)?;
            inner.rebuild_dynamic_runtime_caches();
            if let Some(recursive) = recursive_constraint_artifact {
                let quotients_prepared = inner.dynamic_mask_vocab.projected_terminal_quotients_prepared();
                let quotients = inner.dynamic_mask_vocab.projected_terminal_quotients_for_artifact();
                let loaded = Constraint::load(recursive)?;
                let same_vocab = loaded.token_bytes_count() == inner.token_bytes_count()
                    && inner.token_bytes.iter().all(|(&id, bytes)|
                        loaded.token_bytes_for_id(id) == Some(bytes.as_slice()));
                if !same_vocab {
                    return Err(crate::GlrMaskError::Serialization(
                        "recursive dynamic alternative artifact targets a different vocabulary".to_owned(),
                    ));
                }
                inner = loaded;
                if quotients_prepared {
                    Self::restore_projected_terminal_quotients(&mut inner, quotients)?;
                }
            }
            inner.boundary_trigger = boundary_trigger.into_trigger();
            alternatives.push(Self {
                inner, alternatives: Vec::new(), composition_grammars: vec![None],
                external_vocab_artifact_cache: None,
            });
        }
        Ok(Self::from_alternatives(alternatives))
    }

    /// Serialize this dynamic constraint to a versioned binary artifact.
    pub fn save(&self) -> Vec<u8> {
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
        let payload = DynamicArtifact {
            alternatives: constraints
                .into_iter()
                .map(Self::alternative_for_constraint)
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

    fn dynamic_vocab_from_transfer_artifact(
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
        if version != DYNAMIC_TRANSFER_VERSION {
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
        Self::decode_transfer_sections(bytes, vocab)
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
        if version != DYNAMIC_CONSTRAINT_VERSION {
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
        {
                let payload: DynamicArtifact =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_artifact(payload)
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
mod tests;
