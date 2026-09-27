//! Independent lazy link metadata and retained compiler caches.

use super::{Arc, BTreeMap, Constraint, Cow, Deserialize, Serialize, TerminalID};

pub(super) const COMPOSITION_METADATA_SPLIT_MAGIC: [u8; 4] = *b"CMS5";

pub(super) const COMPOSITION_METADATA_SPLIT_HEADER_LEN: usize = 40;

pub(super) const COMPOSITION_METADATA_COMPRESS_MIN_BYTES: usize = 64 * 1024;

pub(super) const COMPOSITION_METADATA_SPLIT_LINK_COMPRESSED: u32 = 1 << 0;

pub(super) const COMPOSITION_METADATA_SPLIT_CACHE_COMPRESSED: u32 = 1 << 1;

#[derive(Serialize)]
pub(super) enum BoundaryTriggerWireRef<'a> {
    None,
    Tokens(&'a [u32]),
    Exact(&'a crate::automata::weighted_u32::dwa::DWA),
}

#[derive(Serialize, Deserialize)]
pub(super) enum BoundaryTriggerWire {
    None,
    Tokens(Vec<u32>),
    Exact(crate::automata::weighted_u32::dwa::DWA),
}

pub(super) fn boundary_trigger_wire_ref(
    trigger: &crate::runtime::BoundaryTrigger,
) -> BoundaryTriggerWireRef<'_> {
    match trigger {
        crate::runtime::BoundaryTrigger::None => BoundaryTriggerWireRef::None,
        crate::runtime::BoundaryTrigger::Tokens(tokens) => {
            BoundaryTriggerWireRef::Tokens(tokens.as_ref())
        }
        crate::runtime::BoundaryTrigger::Exact(dwa) => BoundaryTriggerWireRef::Exact(dwa.as_ref()),
    }
}

pub(super) fn restore_boundary_trigger(trigger: BoundaryTriggerWire) -> crate::runtime::BoundaryTrigger {
    match trigger {
        BoundaryTriggerWire::None => crate::runtime::BoundaryTrigger::None,
        BoundaryTriggerWire::Tokens(mut tokens) => {
            tokens.sort_unstable();
            tokens.dedup();
            crate::runtime::BoundaryTrigger::Tokens(Arc::from(tokens.into_boxed_slice()))
        }
        BoundaryTriggerWire::Exact(dwa) => crate::runtime::BoundaryTrigger::Exact(Arc::new(dwa)),
    }
}

#[derive(Serialize, Deserialize)]
pub(super) enum BoundaryOriginalTokenSetWire {
    Empty,
    Sparse(Vec<u32>),
    AllByteTokensAtLeastTwo,
}

#[derive(Serialize, Deserialize)]
pub(super) enum BoundaryCandidateSummaryWire {
    Unknown {
        reason: u8,
    },
    Known {
        algorithm_version: u16,
        component_semantics: [u8; 32],
        public_interface: [u8; 32],
        vocabulary: [u8; 32],
        tokens: BoundaryOriginalTokenSetWire,
        precision: u8,
    },
}

pub(super) fn summary_unavailable_code(reason: &crate::runtime::SummaryUnavailable) -> u8 {
    match reason {
        crate::runtime::SummaryUnavailable::Disabled => 0,
        crate::runtime::SummaryUnavailable::Deferred => 1,
        crate::runtime::SummaryUnavailable::LegacyArtifact => 2,
        crate::runtime::SummaryUnavailable::MissingGrammarMetadata => 3,
        crate::runtime::SummaryUnavailable::UnsupportedFiniteLexer => 4,
        crate::runtime::SummaryUnavailable::InvalidatedBinding => 5,
        crate::runtime::SummaryUnavailable::FingerprintMismatch => 6,
        crate::runtime::SummaryUnavailable::MalformedMetadata => 7,
    }
}

pub(super) fn summary_unavailable_from_code(code: u8) -> Result<crate::runtime::SummaryUnavailable, String> {
    Ok(match code {
        0 => crate::runtime::SummaryUnavailable::Disabled,
        1 => crate::runtime::SummaryUnavailable::Deferred,
        2 => crate::runtime::SummaryUnavailable::LegacyArtifact,
        3 => crate::runtime::SummaryUnavailable::MissingGrammarMetadata,
        4 => crate::runtime::SummaryUnavailable::UnsupportedFiniteLexer,
        5 => crate::runtime::SummaryUnavailable::InvalidatedBinding,
        6 => crate::runtime::SummaryUnavailable::FingerprintMismatch,
        7 => crate::runtime::SummaryUnavailable::MalformedMetadata,
        other => return Err(format!("invalid boundary summary unavailable code {other}")),
    })
}

pub(super) fn summary_precision_code(precision: crate::runtime::SummaryPrecision) -> u8 {
    match precision {
        crate::runtime::SummaryPrecision::RegularUpperBound => 0,
        crate::runtime::SummaryPrecision::ContextRefinedUpperBound => 1,
        crate::runtime::SummaryPrecision::BudgetWidenedUpperBound => 2,
    }
}

pub(super) fn summary_precision_from_code(code: u8) -> Result<crate::runtime::SummaryPrecision, String> {
    Ok(match code {
        0 => crate::runtime::SummaryPrecision::RegularUpperBound,
        1 => crate::runtime::SummaryPrecision::ContextRefinedUpperBound,
        2 => crate::runtime::SummaryPrecision::BudgetWidenedUpperBound,
        other => return Err(format!("invalid boundary summary precision code {other}")),
    })
}

pub(super) fn boundary_candidate_summary_wire(constraint: &Constraint) -> BoundaryCandidateSummaryWire {
    let summary = constraint.boundary_candidate_summary.get();
    match summary {
        None => BoundaryCandidateSummaryWire::Unknown {
            reason: summary_unavailable_code(&crate::runtime::SummaryUnavailable::Deferred),
        },
        Some(crate::runtime::BoundaryCandidateSummary::Unknown { reason }) => {
            BoundaryCandidateSummaryWire::Unknown {
                reason: summary_unavailable_code(reason),
            }
        }
        Some(crate::runtime::BoundaryCandidateSummary::Known {
            fingerprint,
            tokens,
            precision,
        }) => {
            let tokens = match tokens {
                crate::runtime::OriginalTokenSet::Empty => BoundaryOriginalTokenSetWire::Empty,
                crate::runtime::OriginalTokenSet::AllByteTokensAtLeastTwo => {
                    BoundaryOriginalTokenSetWire::AllByteTokensAtLeastTwo
                }
                crate::runtime::OriginalTokenSet::Sparse(ids) => {
                    BoundaryOriginalTokenSetWire::Sparse(ids.to_vec())
                }
                crate::runtime::OriginalTokenSet::Dense(_) => {
                    BoundaryOriginalTokenSetWire::Sparse(
                        tokens.canonical_ids(constraint.token_bytes_iter()),
                    )
                }
            };
            BoundaryCandidateSummaryWire::Known {
                algorithm_version: fingerprint.algorithm_version,
                component_semantics: fingerprint.component_semantics,
                public_interface: fingerprint.public_interface,
                vocabulary: fingerprint.vocabulary,
                tokens,
                precision: summary_precision_code(*precision),
            }
        }
    }
}

pub(super) fn unavailable_boundary_candidate_summary_wire() -> BoundaryCandidateSummaryWire {
    BoundaryCandidateSummaryWire::Unknown {
        reason: summary_unavailable_code(&crate::runtime::SummaryUnavailable::LegacyArtifact),
    }
}

pub(super) fn restore_boundary_candidate_summary(
    wire: BoundaryCandidateSummaryWire,
    constraint: &Constraint,
) -> Result<crate::runtime::BoundaryCandidateSummary, String> {
    match wire {
        BoundaryCandidateSummaryWire::Unknown { reason } => {
            Ok(crate::runtime::BoundaryCandidateSummary::Unknown {
                reason: summary_unavailable_from_code(reason)?,
            })
        }
        BoundaryCandidateSummaryWire::Known {
            algorithm_version,
            component_semantics,
            public_interface,
            vocabulary,
            tokens,
            precision,
        } => {
            let tokens = match tokens {
                BoundaryOriginalTokenSetWire::Empty => crate::runtime::OriginalTokenSet::Empty,
                BoundaryOriginalTokenSetWire::AllByteTokensAtLeastTwo => {
                    crate::runtime::OriginalTokenSet::AllByteTokensAtLeastTwo
                }
                BoundaryOriginalTokenSetWire::Sparse(ids) => {
                    if ids.windows(2).any(|pair| pair[0] >= pair[1]) {
                        return Err(
                            "boundary summary sparse token IDs are not strictly sorted/unique"
                                .to_owned(),
                        );
                    }
                    for &id in &ids {
                        if constraint.token_bytes_for_id(id).is_none() {
                            return Err(format!(
                                "boundary summary token ID {id} is outside the artifact vocabulary"
                            ));
                        }
                    }
                    crate::runtime::OriginalTokenSet::Sparse(Arc::from(ids.into_boxed_slice()))
                }
            };
            Ok(crate::runtime::BoundaryCandidateSummary::Known {
                fingerprint: crate::runtime::BoundaryCandidateFingerprint {
                    algorithm_version,
                    component_semantics,
                    public_interface,
                    vocabulary,
                },
                tokens,
                precision: summary_precision_from_code(precision)?,
            })
        }
    }
}

#[derive(Serialize)]
pub(super) struct ConstraintCompositionLinkMetadataRef<'a> {
    pub(super) composition_reset_tokens_by_terminal: &'a [Vec<u32>],
    pub(super) unbound_grammar_placeholders: &'a BTreeMap<String, TerminalID>,
    pub(super) composition_grammar_summary:
        &'a Option<crate::runtime::artifact::CompositionGrammarSummary>,
    pub(super) boundary_trigger: BoundaryTriggerWireRef<'a>,
    pub(super) boundary_candidate_summary: BoundaryCandidateSummaryWire,
}

#[derive(Serialize, Deserialize)]
pub(super) struct ConstraintCompositionLinkMetadata {
    pub(super) composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
    pub(super) unbound_grammar_placeholders: BTreeMap<String, TerminalID>,
    pub(super) composition_grammar_summary: Option<crate::runtime::artifact::CompositionGrammarSummary>,
    pub(super) boundary_trigger: BoundaryTriggerWire,
    pub(super) boundary_candidate_summary: BoundaryCandidateSummaryWire,
}

#[derive(Serialize)]
pub(super) struct ConstraintCompositionCacheMetadataRef<'a> {
    pub(super) composition_parser_templates_by_terminal:
        &'a [Option<crate::automata::unweighted_u32::dfa::DFA>],
    pub(super) composition_parser_characterizations_by_terminal:
        &'a [Option<crate::compiler::stages::templates::characterize::TerminalCharacterization>],
}

#[derive(Serialize, Deserialize)]
pub(super) struct ConstraintCompositionCacheMetadata {
    pub(super) composition_parser_templates_by_terminal:
        Vec<Option<crate::automata::unweighted_u32::dfa::DFA>>,
    pub(super) composition_parser_characterizations_by_terminal:
        Vec<Option<crate::compiler::stages::templates::characterize::TerminalCharacterization>>,
}

#[derive(Deserialize)]
pub(super) struct ConstraintCompositionMetadata {
    pub(super) composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
    pub(super) unbound_grammar_placeholders: BTreeMap<String, TerminalID>,
    pub(super) composition_parser_templates_by_terminal:
        Vec<Option<crate::automata::unweighted_u32::dfa::DFA>>,
    pub(super) composition_parser_characterizations_by_terminal:
        Vec<Option<crate::compiler::stages::templates::characterize::TerminalCharacterization>>,
    pub(super) composition_grammar_summary: Option<crate::runtime::artifact::CompositionGrammarSummary>,
    pub(super) boundary_trigger: BoundaryTriggerWire,
    pub(super) boundary_candidate_summary: BoundaryCandidateSummaryWire,
}

pub(super) struct CompositionMetadataSplitParts<'a> {
    pub(super) link_raw_len: usize,
    pub(super) link_wire: &'a [u8],
    pub(super) link_compressed: bool,
    pub(super) cache_raw_len: usize,
    pub(super) cache_wire: &'a [u8],
    pub(super) cache_compressed: bool,
}

pub(super) fn encode_composition_metadata_part(raw: Vec<u8>) -> (usize, Vec<u8>, bool) {
    let raw_len = raw.len();
    if raw_len >= COMPOSITION_METADATA_COMPRESS_MIN_BYTES {
        let compressed = zstd::bulk::compress(&raw, 1)
            .expect("composition metadata compression should succeed");
        if compressed.len() < raw_len {
            return (raw_len, compressed, true);
        }
    }
    (raw_len, raw, false)
}

pub(super) fn assemble_composition_metadata_split(
    link_raw_len: usize,
    link_wire: &[u8],
    link_compressed: bool,
    cache_raw_len: usize,
    cache_wire: &[u8],
    cache_compressed: bool,
) -> Vec<u8> {
    let mut flags = 0u32;
    if link_compressed {
        flags |= COMPOSITION_METADATA_SPLIT_LINK_COMPRESSED;
    }
    if cache_compressed {
        flags |= COMPOSITION_METADATA_SPLIT_CACHE_COMPRESSED;
    }
    let mut out = Vec::with_capacity(
        COMPOSITION_METADATA_SPLIT_HEADER_LEN + link_wire.len() + cache_wire.len(),
    );
    out.extend_from_slice(&COMPOSITION_METADATA_SPLIT_MAGIC);
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&(link_raw_len as u64).to_le_bytes());
    out.extend_from_slice(&(link_wire.len() as u64).to_le_bytes());
    out.extend_from_slice(&(cache_raw_len as u64).to_le_bytes());
    out.extend_from_slice(&(cache_wire.len() as u64).to_le_bytes());
    out.extend_from_slice(link_wire);
    out.extend_from_slice(cache_wire);
    out
}

pub(super) fn decode_composition_metadata_part<'a>(
    wire: &'a [u8],
    raw_len: usize,
    compressed: bool,
) -> Result<Cow<'a, [u8]>, String> {
    if !compressed {
        if wire.len() != raw_len {
            return Err("invalid raw split composition metadata length".to_owned());
        }
        return Ok(Cow::Borrowed(wire));
    }
    if raw_len == 0 || wire.is_empty() {
        return Err("invalid compressed split composition metadata section".to_owned());
    }
    let raw = zstd::bulk::decompress(wire, raw_len).map_err(|err| err.to_string())?;
    if raw.len() != raw_len {
        return Err("invalid decompressed split composition metadata length".to_owned());
    }
    Ok(Cow::Owned(raw))
}

pub(super) fn split_composition_metadata_parts(
    input: &[u8],
) -> Result<CompositionMetadataSplitParts<'_>, String> {
    if input.len() < COMPOSITION_METADATA_SPLIT_HEADER_LEN
        || !input.starts_with(&COMPOSITION_METADATA_SPLIT_MAGIC)
    {
        return Err("invalid split composition metadata section".to_owned());
    }
    let flags = u32::from_le_bytes(input[4..8].try_into().unwrap());
    if flags
        & !(COMPOSITION_METADATA_SPLIT_LINK_COMPRESSED
            | COMPOSITION_METADATA_SPLIT_CACHE_COMPRESSED)
        != 0
    {
        return Err("invalid split composition metadata flags".to_owned());
    }
    let read_len = |range: std::ops::Range<usize>| -> Result<usize, String> {
        usize::try_from(u64::from_le_bytes(input[range].try_into().unwrap()))
            .map_err(|_| "split composition metadata length does not fit platform".to_owned())
    };
    let link_raw_len = read_len(8..16)?;
    let link_wire_len = read_len(16..24)?;
    let cache_raw_len = read_len(24..32)?;
    let cache_wire_len = read_len(32..40)?;
    let link_start = COMPOSITION_METADATA_SPLIT_HEADER_LEN;
    let link_end = link_start
        .checked_add(link_wire_len)
        .ok_or_else(|| "split composition metadata link range overflow".to_owned())?;
    let cache_end = link_end
        .checked_add(cache_wire_len)
        .ok_or_else(|| "split composition metadata cache range overflow".to_owned())?;
    if cache_end != input.len() {
        return Err("invalid split composition metadata section length".to_owned());
    }
    let link_compressed = flags & COMPOSITION_METADATA_SPLIT_LINK_COMPRESSED != 0;
    let cache_compressed = flags & COMPOSITION_METADATA_SPLIT_CACHE_COMPRESSED != 0;
    if !link_compressed && link_wire_len != link_raw_len {
        return Err("invalid raw split composition link length".to_owned());
    }
    if !cache_compressed && cache_wire_len != cache_raw_len {
        return Err("invalid raw split composition cache length".to_owned());
    }
    if link_compressed && (link_raw_len == 0 || link_wire_len == 0) {
        return Err("invalid compressed split composition link section".to_owned());
    }
    if cache_compressed && (cache_raw_len == 0 || cache_wire_len == 0) {
        return Err("invalid compressed split composition cache section".to_owned());
    }
    Ok(CompositionMetadataSplitParts {
        link_raw_len,
        link_wire: &input[link_start..link_end],
        link_compressed,
        cache_raw_len,
        cache_wire: &input[link_end..cache_end],
        cache_compressed,
    })
}

pub(super) fn encode_composition_metadata(constraint: &Constraint) -> Vec<u8> {
    if constraint.composition_reset_tokens_by_terminal.is_empty()
        && constraint.unbound_grammar_placeholders.is_empty()
        && constraint.composition_parser_templates_by_terminal.is_empty()
        && constraint.composition_parser_characterizations_by_terminal.is_empty()
        && constraint.composition_grammar_summary.is_none()
        && constraint.boundary_trigger.is_none()
        && constraint.boundary_candidate_summary.get().is_none()
    {
        return Vec::new();
    }
    // Link-time grammar/reset metadata is kept independently from the much
    // larger static parser-template caches. Explicit dynamic A+B needs only
    // the former; keeping it as a separately decodable section avoids paying
    // megabytes of parser-cache decompression/allocation merely to discover B.
    let link_raw = bincode::serialize(&ConstraintCompositionLinkMetadataRef {
        composition_reset_tokens_by_terminal: &constraint.composition_reset_tokens_by_terminal,
        unbound_grammar_placeholders: &constraint.unbound_grammar_placeholders,
        composition_grammar_summary: &constraint.composition_grammar_summary,
        boundary_trigger: boundary_trigger_wire_ref(&constraint.boundary_trigger),
        boundary_candidate_summary: boundary_candidate_summary_wire(constraint),
    })
    .expect("composition link metadata serialization should succeed");
    let cache_raw = bincode::serialize(&ConstraintCompositionCacheMetadataRef {
        composition_parser_templates_by_terminal:
            &constraint.composition_parser_templates_by_terminal,
        composition_parser_characterizations_by_terminal:
            &constraint.composition_parser_characterizations_by_terminal,
    })
    .expect("composition cache metadata serialization should succeed");
    let (link_raw_len, link_wire, link_compressed) =
        encode_composition_metadata_part(link_raw);
    let (cache_raw_len, cache_wire, cache_compressed) =
        encode_composition_metadata_part(cache_raw);
    assemble_composition_metadata_split(
        link_raw_len,
        &link_wire,
        link_compressed,
        cache_raw_len,
        &cache_wire,
        cache_compressed,
    )
}

pub(super) fn encode_composition_metadata_for_save(constraint: &Constraint) -> Vec<u8> {
    let bytes=encode_composition_metadata_base_for_save(constraint);
    if let Some(wire)=crate::compiler::composition::boundary::precomputed_completion::saved_wire(constraint) {
        crate::compiler::composition::boundary::precomputed_completion::wrap_envelope(bytes,wire)
            .expect("certified component index fits its bounded envelope")
    } else {
        crate::compiler::composition::boundary::precomputed_completion::split_envelope(&bytes)
            .expect("retained composition envelope validated at load").0.to_vec()
    }
}

pub(super) fn encode_composition_metadata_base_for_save(constraint: &Constraint) -> Vec<u8> {
    let Some(blob) = constraint.deferred_composition_metadata_blob.as_ref() else {
        return encode_composition_metadata(constraint);
    };
    if !constraint.composition_link_metadata_materialized {
        return blob.as_slice().to_vec();
    }

    let link_raw = bincode::serialize(&ConstraintCompositionLinkMetadataRef {
        composition_reset_tokens_by_terminal: &constraint.composition_reset_tokens_by_terminal,
        unbound_grammar_placeholders: &constraint.unbound_grammar_placeholders,
        composition_grammar_summary: &constraint.composition_grammar_summary,
        boundary_trigger: boundary_trigger_wire_ref(&constraint.boundary_trigger),
        boundary_candidate_summary: boundary_candidate_summary_wire(constraint),
    })
    .expect("composition link metadata serialization should succeed");
    let (link_raw_len, link_wire, link_compressed) =
        encode_composition_metadata_part(link_raw);

    let (input, _) = crate::compiler::composition::boundary::precomputed_completion::split_envelope(blob.as_slice())
        .expect("retained composition envelope validated at load");
    let parts = split_composition_metadata_parts(input)
        .expect("loaded deferred composition metadata must remain structurally valid");
    assemble_composition_metadata_split(
        link_raw_len, &link_wire, link_compressed,
        parts.cache_raw_len, parts.cache_wire, parts.cache_compressed,
    )
}

pub(super) fn validate_composition_metadata_wire(input: &[u8]) -> Result<(), String> {
    let (input, _) = crate::compiler::composition::boundary::precomputed_completion::split_envelope(input)?;
    if !input.is_empty() {
        split_composition_metadata_parts(input)?;
    }
    Ok(())
}

pub(super) fn decode_composition_link_metadata(input: &[u8]) -> Result<ConstraintCompositionLinkMetadata, String> {
    let (input, _) = crate::compiler::composition::boundary::precomputed_completion::split_envelope(input)?;
    if input.is_empty() {
        return Ok(ConstraintCompositionLinkMetadata {
            composition_reset_tokens_by_terminal: Vec::new(),
            unbound_grammar_placeholders: BTreeMap::new(),
            composition_grammar_summary: None,
            boundary_trigger: BoundaryTriggerWire::None,
            boundary_candidate_summary: unavailable_boundary_candidate_summary_wire(),
        });
    }
    let parts = split_composition_metadata_parts(input)?;
    let raw = decode_composition_metadata_part(parts.link_wire, parts.link_raw_len, parts.link_compressed)?;
    bincode::deserialize(raw.as_ref()).map_err(|err| err.to_string())
}

pub(super) fn decode_composition_metadata(input: &[u8]) -> Result<ConstraintCompositionMetadata, String> {
    let (input, _) = crate::compiler::composition::boundary::precomputed_completion::split_envelope(input)?;
    let link = decode_composition_link_metadata(input)?;
    let cache = if input.is_empty() {
        ConstraintCompositionCacheMetadata {
            composition_parser_templates_by_terminal: Vec::new(),
            composition_parser_characterizations_by_terminal: Vec::new(),
        }
    } else {
        let parts = split_composition_metadata_parts(input)?;
        let raw = decode_composition_metadata_part(parts.cache_wire, parts.cache_raw_len, parts.cache_compressed)?;
        bincode::deserialize::<ConstraintCompositionCacheMetadata>(raw.as_ref()).map_err(|err| err.to_string())?
    };
    Ok(ConstraintCompositionMetadata {
        composition_reset_tokens_by_terminal: link.composition_reset_tokens_by_terminal,
        unbound_grammar_placeholders: link.unbound_grammar_placeholders,
        composition_parser_templates_by_terminal: cache.composition_parser_templates_by_terminal,
        composition_parser_characterizations_by_terminal: cache.composition_parser_characterizations_by_terminal,
        composition_grammar_summary: link.composition_grammar_summary,
        boundary_trigger: link.boundary_trigger,
        boundary_candidate_summary: link.boundary_candidate_summary,
    })
}

impl Constraint {
/// Borrow already-materialized local templates, or decode only the template
    /// vector from a deferred composition cache. Unlike full materialization,
    /// this does not clone a Constraint or reconstruct unused characterizations,
    /// token-reset rows, parser DWAs, or mutable runtime state.
    pub(crate) fn retained_parser_templates_for_compilation(
        &self,
    ) -> Result<Cow<'_, [Option<crate::automata::unweighted_u32::dfa::DFA>]>, String> {
        if !self.composition_parser_templates_by_terminal.is_empty() {
            return Ok(Cow::Borrowed(&self.composition_parser_templates_by_terminal));
        }
        let Some(blob) = self.deferred_composition_metadata_blob.as_ref() else {
            return Ok(Cow::Borrowed(&[]));
        };
        let (input, _) = crate::compiler::composition::boundary::precomputed_completion::split_envelope(blob.as_slice())?;
        if input.is_empty() {
            return Ok(Cow::Borrowed(&[]));
        }
        let parts = split_composition_metadata_parts(input)?;
        let cache_raw = decode_composition_metadata_part(
            parts.cache_wire, parts.cache_raw_len, parts.cache_compressed,
        )?;
        // Templates are the first cache field; do not decode characterizations.
        let templates = bincode::deserialize_from(cache_raw.as_ref())
            .map_err(|error| error.to_string())?;
        Ok(Cow::Owned(templates))
    }

pub(crate) fn materialize_composition_metadata_for_compilation(
        &mut self,
    ) -> Result<(), String> {
        let Some(blob) = self.deferred_composition_metadata_blob.clone() else {
            return Ok(());
        };
        let metadata = decode_composition_metadata(blob.as_slice())?;
        self.composition_reset_tokens_by_terminal = metadata.composition_reset_tokens_by_terminal;
        self.unbound_grammar_placeholders = metadata.unbound_grammar_placeholders;
        self.composition_parser_templates_by_terminal =
            metadata.composition_parser_templates_by_terminal;
        self.composition_parser_characterizations_by_terminal =
            metadata.composition_parser_characterizations_by_terminal;
        self.composition_grammar_summary = metadata.composition_grammar_summary;
        self.boundary_trigger = restore_boundary_trigger(metadata.boundary_trigger);
        if self.boundary_candidate_summary.get().is_none() {
            let summary = restore_boundary_candidate_summary(
                metadata.boundary_candidate_summary,
                self,
            )?;
            let _ = self.boundary_candidate_summary.set(summary);
        }
        self.composition_link_metadata_materialized = true;
        self.deferred_composition_metadata_blob = None;
        Ok(())
    }

/// Materialize only the metadata needed to link an explicit dynamic A+B
    /// composition. New split-format artifacts keep this data independent of
    /// the large static parser-template caches, so dynamic late binding can
    /// remain cheap. The deferred blob is intentionally retained: a later
    /// static/generic composition of the same constraint can still request the
    /// complete compiler cache through `materialize_composition_metadata_for_compilation`.
    pub(crate) fn materialize_composition_link_metadata_for_compilation(
        &mut self,
    ) -> Result<(), String> {
        let Some(blob) = self.deferred_composition_metadata_blob.clone() else {
            return Ok(());
        };
        let metadata = decode_composition_link_metadata(blob.as_slice())?;
        self.composition_reset_tokens_by_terminal = metadata.composition_reset_tokens_by_terminal;
        self.unbound_grammar_placeholders = metadata.unbound_grammar_placeholders;
        self.composition_grammar_summary = metadata.composition_grammar_summary;
        self.boundary_trigger = restore_boundary_trigger(metadata.boundary_trigger);
        if self.boundary_candidate_summary.get().is_none() {
            let summary = restore_boundary_candidate_summary(
                metadata.boundary_candidate_summary,
                self,
            )?;
            let _ = self.boundary_candidate_summary.set(summary);
        }
        self.composition_link_metadata_materialized = true;
        Ok(())
    }
}
