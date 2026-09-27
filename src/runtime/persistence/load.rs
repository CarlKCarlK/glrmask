//! Decode current sections, validate runtime coordinates, and restore execution caches.

use super::{
    Arc, CONSTRAINT_HEADER_LEN, CONSTRAINT_MAGIC, CONSTRAINT_VERSION, Constraint, Cow,
    DecodedConstraintCore, DecodedConstraintRuntime, DecodedInternalTokenBufMasks,
    DecodedOriginalTokenMap, ROOT_POLICY_MAGIC, TokenMaskCacheArtifact,
    attach_packed_non_dwa_weights, constraint_sections, decode_current_core,
    decode_current_runtime_wire, decode_internal_token_buf_masks, decode_token_mask_cache,
    decode_token_mask_cache_backed, install_token_mask_cache, restore_segmented_runtime,
    validate_composition_metadata_wire,
};

impl Constraint {
/// Load a compiled constraint from an artifact produced by [`Constraint::save`].
    ///
    /// Passing an owned `Vec<u8>` transfers its allocation into the loaded
    /// constraint without copying the artifact. Borrowed byte slices are also
    /// accepted; current-format artifacts copy borrowed input once because
    /// runtime structures retain zero-copy views into persistent backing bytes.
    pub fn load<'a>(bytes: impl Into<Cow<'a, [u8]>>) -> crate::Result<Self> {
        let bytes = bytes.into();
        if !bytes.starts_with(ROOT_POLICY_MAGIC) {
            let body = Self::load_body(bytes)?;
            if !body.late_grammar_slots.is_empty() {
                return Err(crate::Error::Serialization(
                    "constraint artifact has unresolved slots; load it as an UnlinkedConstraint instead"
                        .to_owned(),
                ));
            }
            return Ok(body);
        }
        if bytes.len() < 16 {
            return Err(crate::Error::Serialization("truncated root policy header".to_owned()));
        }
        let end_count =
            u32::from_le_bytes(bytes[8..12].try_into().expect("header checked")) as usize;
        let exact_count =
            u32::from_le_bytes(bytes[12..16].try_into().expect("header checked")) as usize;
        let metadata_count = end_count.checked_add(exact_count)
            .ok_or_else(|| crate::Error::Serialization("invalid root policy counts".to_owned()))?;
        let start = metadata_count.checked_mul(4).and_then(|len| 16usize.checked_add(len))
            .filter(|&start| start < bytes.len())
            .ok_or_else(|| crate::Error::Serialization("invalid root policy length".to_owned()))?;
        let end_ids = bytes[16..16 + end_count * 4].chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("chunk width checked")))
            .collect::<Vec<_>>();
        let exact_ids = bytes[16 + end_count * 4..start].chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("chunk width checked")))
            .collect::<Vec<_>>();
        if end_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || exact_ids.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(crate::Error::Serialization("noncanonical root vocabulary policy".to_owned()));
        }
        let mut body = match bytes {
            Cow::Owned(mut bytes) => Self::load_body(Cow::Owned(bytes.split_off(start)))?,
            Cow::Borrowed(bytes) => Self::load_body(Cow::Borrowed(&bytes[start..]))?,
        };
        if !body.late_grammar_slots.is_empty() {
            return Err(crate::Error::Serialization("root artifact has unresolved slots".to_owned()));
        }
        if exact_ids.iter().any(|&id| body.token_bytes_for_id(id).is_some()) {
            return Err(crate::Error::Serialization(
                "root exact-only token domain overlaps byte vocabulary".to_owned(),
            ));
        }
        if !exact_ids.is_empty() {
            let vocab = crate::Vocab::new_with_exact_token_ids(
                body.token_bytes_iter()
                    .map(|(token_id, bytes)| (token_id, bytes.to_vec()))
                    .collect(),
                exact_ids,
            );
            body.bind_vocab_exact(&vocab)
                .map_err(crate::Error::Serialization)?;
        }
        body.with_end_tokens(&end_ids)
            .map_err(|error| crate::Error::Serialization(error.to_string()))
    }

pub(super) fn load_body(bytes: Cow<'_, [u8]>) -> crate::Result<Self> {
        match bytes {
            Cow::Owned(bytes) => {
                let backing = std::sync::Arc::new(bytes);
                Self::load_impl(backing.as_slice(), Some(std::sync::Arc::clone(&backing)))
            }
            Cow::Borrowed(bytes) => Self::load_impl(bytes, None),
        }
    }

/// Load an embeddable compiled body without enforcing the public
    /// closed-root invariant.
    ///
    /// Only UnlinkedConstraint and segmented-runtime persistence use this. Public callers
    /// must use `Constraint::load`, which rejects unresolved slots.
    pub(crate) fn load_body_artifact<'a>(
        bytes: impl Into<Cow<'a, [u8]>>,
    ) -> crate::Result<Self> {
        Self::load_body(bytes.into())
    }

/// Load a compiled constraint and bind it to an already-existing exact
    /// model vocabulary.
    ///
    /// This preserves the zero-copy packed artifact representation while
    /// sharing the caller's `Vocab` identity and derived-artifact cache. Later
    /// subgrammar composition can therefore prove vocabulary compatibility by
    /// `Arc` identity instead of reconstructing and repeatedly comparing the
    /// same token byte map.
    pub fn load_with_vocab<'a>(
        bytes: impl Into<Cow<'a, [u8]>>,
        vocab: &crate::Vocab,
    ) -> crate::Result<Self> {
        let mut constraint = Self::load(bytes)?;
        constraint
            .bind_vocab_exact(vocab)
            .map_err(crate::GlrMaskError::Serialization)?;
        Ok(constraint)
    }

pub(super) fn load_impl(
        bytes: &[u8],
        owned_artifact: Option<std::sync::Arc<Vec<u8>>>,
    ) -> crate::Result<Self> {
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let decompress_ms = 0.0;
        if bytes.len() < CONSTRAINT_HEADER_LEN || !bytes.starts_with(&CONSTRAINT_MAGIC) {
            return Err(crate::GlrMaskError::Serialization(
                "invalid constraint artifact header".to_owned(),
            ));
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if version != CONSTRAINT_VERSION {
            return Err(crate::GlrMaskError::Serialization(format!(
                "unsupported constraint artifact version {version}"
            )));
        }
        let payload_len = usize::try_from(u64::from_le_bytes(
            bytes[10..18]
                .try_into()
                .expect("constraint artifact header has fixed width"),
        ))
        .map_err(|_| {
            crate::GlrMaskError::Serialization(
                "constraint artifact payload length does not fit this platform".to_owned(),
            )
        })?;
        if bytes.len() != CONSTRAINT_HEADER_LEN.saturating_add(payload_len) {
            return Err(crate::GlrMaskError::Serialization(
                "invalid constraint artifact payload length".to_owned(),
            ));
        }
        // v17 runtime sections may retain zero-copy views into the artifact.
        // If the caller did not transfer ownership, make the one compatibility
        // copy up front so every retained view and the unchanged-resave cache
        // share the same backing allocation.
        let current_backing = {
            Some(
                owned_artifact
                    .clone()
                    .unwrap_or_else(|| std::sync::Arc::new(bytes.to_vec())),
            )
        };
        let section_bytes = current_backing
            .as_ref()
            .map_or(bytes, |backing| backing.as_slice());
        let payload = &section_bytes[CONSTRAINT_HEADER_LEN..];
        let serialized = {
            payload
        };
        let deserialize_started = profile.then(std::time::Instant::now);
        let mut prepared_completion_wire: Option<Arc<[u8]>> = None;
        let mut loaded_packed_dwa_dense_masks = false;
        let mut constraint = {
            let sections = constraint_sections(serialized)
                .map_err(crate::GlrMaskError::Serialization)?;
            let weight_section = sections.weight;
            let dwa_section = sections.dwa;
            let table_section = sections.table;
            let core_section = sections.core;
            let runtime_section = Some(sections.runtime);
            let token_bytes_section = Some(sections.token_bytes);
            let original_token_map_section = Some(sections.original_map);
            let tokenizer_section = Some(sections.tokenizer);
            let internal_token_buf_masks_section = Some(sections.internal_masks);
            let token_mask_cache_section = Some(sections.token_mask_cache);
            let composition_metadata_section = Some(sections.composition_metadata);
            let (((dwa_result, (table_result, runtime_result)), ((tokenizer_result, original_token_map_result), (internal_token_buf_masks_result, token_mask_cache_result))), core_result) = rayon::join(
                || rayon::join(
                    || rayon::join(
                        || {
                            let started = profile.then(std::time::Instant::now);
                            let result = {
                                let decoded = if dwa_section.starts_with(b"DWF3")
                                    || dwa_section.starts_with(b"DWF4")
                    || dwa_section.starts_with(b"DWF5")
                    || dwa_section.starts_with(b"DWF6")
                    || dwa_section.starts_with(b"DWF7")
                    || dwa_section.starts_with(b"DWF8")
                                {
                                    let backing = current_backing.as_ref().ok_or_else(|| {
                                        "current backed DWA has no artifact backing".to_owned()
                                    });
                                    match backing {
                                        Ok(backing) => {
                                            let base = backing.as_ptr() as usize;
                                            let section_start = (dwa_section.as_ptr() as usize)
                                                .checked_sub(base)
                                                .ok_or_else(|| {
                                                    "DWA section does not belong to artifact backing"
                                                        .to_owned()
                                                });
                                            match section_start {
                                                Ok(section_start) => crate::automata::weighted::dwa::PackedRuntimeDwa::from_fast_wire_bytes_backed(
                                                    dwa_section,
                                                    std::sync::Arc::clone(backing),
                                                    section_start,
                                                ),
                                                Err(err) => Err(err),
                                            }
                                        }
                                        Err(err) => Err(err),
                                    }
                                } else if dwa_section.starts_with(b"DWF1")
                                    || dwa_section.starts_with(b"DWF2")
                                {
                                    crate::automata::weighted::dwa::PackedRuntimeDwa::from_fast_wire_bytes(
                                        dwa_section,
                                    )
                                } else {
                                    crate::automata::weighted::dwa::PackedRuntimeDwa::from_packed_bytes(
                                        dwa_section,
                                    )
                                };
                                decoded.map(|dwa| {
                                    std::sync::Arc::new(dwa)
                                })
                            };
                            if let Some(started) = started {
                                eprintln!("[glrmask/profile][constraint_section] name=dwa ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                            }
                            result
                        },
                        || {
                            rayon::join(
                                || {
                                    let started = profile.then(std::time::Instant::now);
                                    let result = {
                                        let backing = current_backing.as_ref().ok_or_else(|| {
                                            "current GLR table has no artifact backing".to_owned()
                                        });
                                        match backing {
                                            Ok(backing) => {
                                                let base = backing.as_ptr() as usize;
                                                let section_start = (table_section.as_ptr() as usize)
                                                    .checked_sub(base)
                                                    .ok_or_else(|| {
                                                        "GLR table section does not belong to artifact backing"
                                                            .to_owned()
                                                    });
                                                match section_start {
                                                    Ok(section_start) => crate::compiler::glr::table::artifact_serde::from_compact_bytes_deferred_backed(
                                                        table_section,
                                                        std::sync::Arc::clone(backing),
                                                        section_start,
                                                    ),
                                                    Err(err) => Err(err),
                                                }
                                            }
                                            Err(err) => Err(err),
                                        }
                                    };
                                    if let Some(started) = started {
                                        eprintln!("[glrmask/profile][constraint_section] name=table ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                                    }
                                    result
                                },
                                || -> Result<Option<DecodedConstraintRuntime>, String> {
                                    let Some(runtime_section) = runtime_section else {
                                        return Ok(None);
                                    };
                                    let started = profile.then(std::time::Instant::now);
                                    let result = {
                                        current_backing
                                            .as_ref()
                                            .ok_or_else(|| bincode::Error::new(bincode::ErrorKind::Custom("current runtime has no artifact backing".to_owned())))
                                            .and_then(|backing| {
                                                decode_current_runtime_wire(runtime_section, std::sync::Arc::clone(backing))
                                                    .map_err(|err| bincode::Error::new(bincode::ErrorKind::Custom(err)))
                                            })
                                        .and_then(|(runtime, static_virtual_residual_mask)| {
                                            let packed_dwa_dense_masks = if runtime
                                                .packed_dwa_dense_mask_ids
                                                .is_empty()
                                            {
                                                if !runtime.packed_dwa_dense_mask_rows.is_empty() {
                                                    return Err(bincode::Error::new(
                                                        bincode::ErrorKind::Custom(
                                                            "packed DWA dense-mask slab has rows but no ids"
                                                                .to_owned(),
                                                        ),
                                                    ));
                                                }
                                                None
                                            } else {
                                                Some((
                                                    runtime.packed_dwa_dense_mask_ids,
                                                    runtime.packed_dwa_dense_mask_rows,
                                                ))
                                            };
                                            Ok(DecodedConstraintRuntime {
                                                terminal_live_states: runtime.terminal_live_states,

                                                segmented_runtime: runtime.segmented_runtime,
                                                dynamic_mask_vocab: runtime.dynamic_mask_vocab,
                                                virtual_runtimes: runtime.virtual_runtimes,
                                                static_virtual_residual_mask,
                                                packed_dwa_dense_masks,
                                            })
                                        })
                                    }
                                    .map(Some)
                                    .map_err(|err| err.to_string());
                                    if let Some(started) = started {
                                        eprintln!("[glrmask/profile][constraint_section] name=runtime ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                                    }
                                    result
                                },
                            )
                        },
                    ),
                    || rayon::join(
                        || rayon::join(
                            || -> Result<Option<crate::automata::lexer::tokenizer::Tokenizer>, String> {
                            let Some(section) = tokenizer_section else {
                                return Ok(None);
                            };
                            let started = profile.then(std::time::Instant::now);
                            let result = {
                                let backing = current_backing.as_ref().ok_or_else(|| {
                                    "current tokenizer has no artifact backing".to_owned()
                                })?;
                                let base = backing.as_ptr() as usize;
                                let section_start = (section.as_ptr() as usize)
                                    .checked_sub(base)
                                    .ok_or_else(|| {
                                        "tokenizer section does not belong to artifact backing"
                                            .to_owned()
                                    })?;
                                crate::automata::lexer::tokenizer::artifact_serde::from_fast_bytes_backed(
                                    section,
                                    std::sync::Arc::clone(backing),
                                    section_start,
                                )
                                .map(Some)
                            };
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_section] name=tokenizer ms={:.3}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                );
                            }
                            result
                            },
                            || -> Result<Option<DecodedOriginalTokenMap>, String> {
                            let Some(section) = original_token_map_section else {
                                return Ok(None);
                            };
                            let started = profile.then(std::time::Instant::now);
                            let result = {
                                let backing = current_backing.as_ref().ok_or_else(|| {
                                    "current original-token map has no artifact backing".to_owned()
                                })?;
                                let base = backing.as_ptr() as usize;
                                let section_start = (section.as_ptr() as usize)
                                    .checked_sub(base)
                                    .ok_or_else(|| {
                                        "original-token map section does not belong to artifact backing"
                                            .to_owned()
                                    })?;
                                crate::runtime::artifact::original_token_map_artifact_serde::PackedOriginalTokenMap::parse_backed(
                                    std::sync::Arc::clone(backing),
                                    section_start,
                                    section.len(),
                                )
                                .map(|packed| {
                                    Some(DecodedOriginalTokenMap::Packed(std::sync::Arc::new(
                                        packed,
                                    )))
                                })
                            };
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_section] name=original_token_map ms={:.3}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                );
                            }
                            result
                            },
                        ),
                        || rayon::join(
                            || -> Result<Option<DecodedInternalTokenBufMasks>, String> {
                                let Some(section) = internal_token_buf_masks_section else {
                                    return Ok(None);
                                };
                                let started = profile.then(std::time::Instant::now);
                                let backing = current_backing
                                    .as_ref()
                                    .map(|backing| {
                                        let base = backing.as_ptr() as usize;
                                        let section_start = (section.as_ptr() as usize)
                                            .checked_sub(base)
                                            .ok_or_else(|| {
                                                "internal-token buffer-mask section does not belong to artifact backing"
                                                    .to_owned()
                                            })?;
                                        Ok::<_, String>((
                                            std::sync::Arc::clone(backing),
                                            section_start,
                                        ))
                                    })
                                    .transpose()?;
                                let result = decode_internal_token_buf_masks(section, backing).map(Some);
                                if let Some(started) = started {
                                    eprintln!(
                                        "[glrmask/profile][constraint_section] name=internal_token_buf_masks ms={:.3}",
                                        started.elapsed().as_secs_f64() * 1000.0,
                                    );
                                }
                                result
                            },
                            || -> Result<Option<TokenMaskCacheArtifact>, String> {
                                let Some(section) = token_mask_cache_section else {
                                    return Ok(None);
                                };
                                if section.is_empty() {
                                    return Ok(None);
                                }
                                let started = profile.then(std::time::Instant::now);
                                let result = if let Some(backing) = current_backing.as_ref() {
                                    let base = backing.as_ptr() as usize;
                                    let section_start = (section.as_ptr() as usize)
                                        .checked_sub(base)
                                        .ok_or_else(|| {
                                            "token-mask cache section does not belong to artifact backing"
                                                .to_owned()
                                        })?;
                                    decode_token_mask_cache_backed(
                                        section,
                                        std::sync::Arc::clone(backing),
                                        section_start,
                                    )
                                    .map(Some)
                                } else {
                                    decode_token_mask_cache(section).map(Some)
                                };
                                if let Some(started) = started {
                                    eprintln!(
                                        "[glrmask/profile][constraint_section] name=token_mask_cache ms={:.3}",
                                        started.elapsed().as_secs_f64() * 1000.0,
                                    );
                                }
                                result
                            },
                        ),
                    ),
                ),
                || -> Result<
                    (
                        DecodedConstraintCore,
                        Option<std::sync::Arc<crate::runtime::artifact::token_bytes_artifact_serde::PackedTokenBytes>>,
                        Option<(
                            std::sync::Arc<crate::ds::weight::PackedRuntimeWeightPool>,
                            Vec<u32>,
                        )>,
                    ),
                    String,
                > {
                    let section_started = profile.then(std::time::Instant::now);
                    let weight_count = {
                        Some(crate::ds::weight::PackedRuntimeWeightPool::peek_weight_count(
                            weight_section,
                        )?)
                    };

                    let decode_core = || -> Result<_, String> {
                        if let Some(weight_count) = weight_count {
                            crate::ds::weight::begin_pooled_weight_serde_deferred_decode(
                                weight_count,
                            );
                        } else {
                            let weights_started = profile.then(std::time::Instant::now);
                            let weights = crate::ds::weight::unpack_pooled_weights(weight_section)?;
                            if let Some(started) = weights_started {
                                eprintln!("[glrmask/profile][constraint_section] name=weights ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                            }
                            crate::ds::weight::begin_pooled_weight_serde_decode(weights);
                        }
                        let previous_external =
                            crate::automata::weighted::dwa::set_external_serde(true);
                        let previous_external_table =
                            crate::compiler::glr::table::artifact_serde::set_external_serde(true);
                        let previous_compact_tokenizer =
                            crate::automata::lexer::tokenizer::set_compact_artifact_serde(true);
                        let previous_external_tokenizer =
                            crate::automata::lexer::tokenizer::set_external_artifact_serde(
                                true,
                            );
                        let previous_omit_inverse =
                            crate::runtime::artifact::internal_token_inverse_artifact_serde::set_omit(
                                true,
                            );
                        let previous_packed_original_token_map =
                            crate::runtime::artifact::original_token_map_artifact_serde::set_packed(
                                true,
                            );
                        let previous_external_original_token_map =
                            crate::runtime::artifact::original_token_map_artifact_serde::set_external(
                                true,
                            );
                        let previous_packed_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_packed(true);
                        let previous_external_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_external(
                                true,
                            );
                        let previous_defer_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_defer_unpack(
                                false,
                            );
                        let core_started = profile.then(std::time::Instant::now);
                        let decoded = {
                            {
                                let core_backing = current_backing.as_ref().and_then(|backing| {
                                    let base = backing.as_ptr() as usize;
                                    let start = (core_section.as_ptr() as usize).checked_sub(base)?;
                                    Some((std::sync::Arc::clone(backing), start))
                                });
                                decode_current_core(core_section, core_backing)
                                .map(|(artifact, terminal_exprs_blob)| {
                                    let mut constraint = artifact.constraint;
                                    constraint.static_dynamic_overlay = artifact.static_dynamic_overlay;
                                    constraint.late_grammar_slots = artifact.late_grammar_slots;
                                    DecodedConstraintCore {
                                        constraint,
                                        ignore_expr: artifact.ignore_expr,
                                        terminal_exprs: None,
                                        terminal_exprs_blob,
                                        parser_state_domain_labels:
                                            artifact.parser_state_domain_labels,
                                        internal_token_buf_masks: Vec::new(),
                                    }
                                })
                            }
                        };
                        let deferred_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::take_deferred();
                        let deferred_weight_ids = if weight_count.is_some() {
                            crate::ds::weight::take_pooled_weight_serde_deferred_ids()
                        } else {
                            Vec::new()
                        };
                        if let Some(started) = core_started {
                            eprintln!("[glrmask/profile][constraint_section] name=core_bincode ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                        }
                        crate::automata::lexer::tokenizer::set_compact_artifact_serde(
                            previous_compact_tokenizer,
                        );
                        crate::automata::lexer::tokenizer::set_external_artifact_serde(
                            previous_external_tokenizer,
                        );
                        crate::runtime::artifact::token_bytes_artifact_serde::set_packed(
                            previous_packed_token_bytes,
                        );
                        crate::runtime::artifact::token_bytes_artifact_serde::set_external(
                            previous_external_token_bytes,
                        );
                        crate::runtime::artifact::token_bytes_artifact_serde::set_defer_unpack(
                            previous_defer_token_bytes,
                        );
                        crate::runtime::artifact::internal_token_inverse_artifact_serde::set_omit(
                            previous_omit_inverse,
                        );
                        crate::runtime::artifact::original_token_map_artifact_serde::set_packed(
                            previous_packed_original_token_map,
                        );
                        crate::runtime::artifact::original_token_map_artifact_serde::set_external(
                            previous_external_original_token_map,
                        );
                        crate::compiler::glr::table::artifact_serde::set_external_serde(
                            previous_external_table,
                        );
                        crate::automata::weighted::dwa::set_external_serde(previous_external);
                        crate::ds::weight::end_pooled_weight_serde_decode();
                        decoded.map(|artifact| {
                            (artifact, deferred_token_bytes, deferred_weight_ids)
                        })
                    };

                    let (artifact, deferred_token_bytes, packed_weights) =
                        {
                            let weights_started = profile.then(std::time::Instant::now);
                            // WPL3 current-format runtime indexing is now only
                            // a small linear framing scan + one section copy.
                            // Running that tiny job as another nested Rayon
                            // branch competes with the much heavier core/DWA
                            // decoders and increases wall time on Windows.
                            let packed_weights =
                                crate::ds::weight::PackedRuntimeWeightPool::from_packed_bytes(
                                    weight_section,
                                )?;
                            if let Some(started) = weights_started {
                                eprintln!("[glrmask/profile][constraint_section] name=weights_packed ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                            }
                            let (artifact, deferred_token_bytes, deferred_weight_ids) = decode_core()?;
                            (
                                artifact,
                                deferred_token_bytes,
                                Some((std::sync::Arc::new(packed_weights), deferred_weight_ids)),
                            )
                        };
                    if let Some(started) = section_started {
                        eprintln!("[glrmask/profile][constraint_section] name=core_total ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                    }
                    Ok((artifact, deferred_token_bytes, packed_weights))
                },
            );
            let parser_dwa = dwa_result.map_err(crate::GlrMaskError::Serialization)?;
            let decoded_table = table_result.map_err(crate::GlrMaskError::Serialization)?;
            let table = decoded_table.table;
            let deferred_table_rules_blob = decoded_table.deferred_rules;
            let runtime = runtime_result.map_err(crate::GlrMaskError::Serialization)?;
            let tokenizer = tokenizer_result.map_err(crate::GlrMaskError::Serialization)?;
            let original_token_map =
                original_token_map_result.map_err(crate::GlrMaskError::Serialization)?;
            let external_internal_token_buf_masks =
                internal_token_buf_masks_result.map_err(crate::GlrMaskError::Serialization)?;
            let token_mask_cache =
                token_mask_cache_result.map_err(crate::GlrMaskError::Serialization)?;
            let (artifact, deferred_token_bytes, packed_weights) =
                core_result.map_err(crate::GlrMaskError::Serialization)?;
            let token_bytes_started = profile.then(std::time::Instant::now);
            let external_token_bytes = if let Some(token_bytes_section) = token_bytes_section {
                let backing = current_backing
                    .as_ref()
                    .expect("current-format token section has artifact backing");
                let start = (token_bytes_section.as_ptr() as usize)
                    .checked_sub(section_bytes.as_ptr() as usize)
                    .ok_or_else(|| {
                        crate::GlrMaskError::Serialization(
                            "token-byte section does not belong to artifact backing".to_owned(),
                        )
                    })?;
                Some(std::sync::Arc::new(
                    crate::runtime::artifact::token_bytes_artifact_serde::PackedTokenBytes::parse_backed(
                        std::sync::Arc::clone(backing),
                        start,
                        token_bytes_section.len(),
                    )
                    .map_err(crate::GlrMaskError::Serialization)?,
                ))
            } else {
                None
            };
            let token_bytes_ms = token_bytes_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            let mut constraint = artifact.constraint;
            if let Some(tokenizer) = tokenizer {
                constraint.tokenizer = tokenizer.into();
            }
            if let Some(original_token_map) = original_token_map {
                match original_token_map {
                    DecodedOriginalTokenMap::Materialized(map) => {
                        constraint.original_token_to_internal = map;
                        constraint.packed_original_token_to_internal = None;
                    }
                    DecodedOriginalTokenMap::Packed(map) => {
                        constraint.original_token_to_internal = Vec::new();
                        constraint.packed_original_token_to_internal = Some(map);
                    }
                }
            }
            let attach_weights_started = profile.then(std::time::Instant::now);
            if let Some((pool, ids)) = packed_weights {
                attach_packed_non_dwa_weights(&mut constraint, pool, ids)
                    .map_err(crate::GlrMaskError::Serialization)?;
            }
            let attach_weights_ms = attach_weights_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            let attach_dwa_started = profile.then(std::time::Instant::now);
            constraint.parser_dwa = crate::automata::weighted::dwa::DWA::new(0, 0);
            constraint.packed_parser_dwa = Some(parser_dwa);
            let attach_dwa_ms = attach_dwa_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            constraint.table = table;
            constraint.deferred_table_rules_blob = deferred_table_rules_blob;
            constraint.deferred_table_rules = Default::default();
            let invert_ms = 0.0;
            constraint.ignore_expr = artifact.ignore_expr;
            constraint.parser_state_domain_labels = artifact.parser_state_domain_labels;
            constraint.deferred_terminal_exprs_blob = artifact.terminal_exprs_blob;
            constraint.deferred_terminal_exprs = Default::default();
            constraint.deferred_composition_metadata_blob = if let Some(section) = composition_metadata_section {
                let (_, prepared)=crate::compiler::composition::boundary::precomputed_completion::split_envelope(section)
                    .map_err(crate::GlrMaskError::Serialization)?;
                prepared_completion_wire=prepared.map(Arc::from);
                validate_composition_metadata_wire(section)
                    .map_err(crate::GlrMaskError::Serialization)?;
                if section.is_empty() {
                    None
                } else if let Some(backing) = current_backing.as_ref() {
                    let start = (section.as_ptr() as usize)
                        .checked_sub(backing.as_ptr() as usize)
                        .ok_or_else(|| {
                            crate::GlrMaskError::Serialization(
                                "composition metadata section does not belong to artifact backing"
                                    .to_owned(),
                            )
                        })?;
                    let end = start.checked_add(section.len()).ok_or_else(|| {
                        crate::GlrMaskError::Serialization(
                            "composition metadata backing range overflow".to_owned(),
                        )
                    })?;
                    if backing.get(start..end) != Some(section) {
                        return Err(crate::GlrMaskError::Serialization(
                            "composition metadata bytes do not match artifact backing".to_owned(),
                        ));
                    }
                    Some(crate::runtime::artifact::DeferredCompositionMetadataBytes::Backed {
                        backing: std::sync::Arc::clone(backing),
                        start,
                        len: section.len(),
                    })
                } else {
                    Some(crate::runtime::artifact::DeferredCompositionMetadataBytes::Owned(
                        std::sync::Arc::from(section.to_vec().into_boxed_slice()),
                    ))
                }
            } else {
                None
            };
            constraint.composition_link_metadata_materialized =
                constraint.deferred_composition_metadata_blob.is_none();
            if let Some(decoded) = external_internal_token_buf_masks {
                constraint.internal_token_buf_masks = Vec::new();
                constraint.internal_token_buf_flat = decoded.flat;
                constraint.backed_internal_token_buf_flat = decoded.backed;
                constraint.internal_token_buf_offsets = decoded.offsets;
            } else {
                constraint.internal_token_buf_masks = artifact.internal_token_buf_masks;
                constraint.backed_internal_token_buf_flat = None;
            }
            constraint.packed_token_bytes = external_token_bytes.or(deferred_token_bytes);
            if let Some(cache) = token_mask_cache {
                install_token_mask_cache(&mut constraint, cache)
                    .map_err(crate::GlrMaskError::Serialization)?;
            }
            let mut virtual_runtimes = Vec::new();
            let mut static_virtual_residual_mask = None;
            if let Some(runtime) = runtime {
                virtual_runtimes = runtime.virtual_runtimes;
                static_virtual_residual_mask = runtime.static_virtual_residual_mask;
                constraint.terminal_live_states = runtime.terminal_live_states;
                if let Some((ids, rows)) = runtime.packed_dwa_dense_masks {
                    let expected_words = constraint.internal_token_count().div_ceil(64);
                    let token_set_count = constraint
                        .packed_parser_dwa
                        .as_ref()
                        .map_or(0, |dwa| dwa.token_set_count());
                    let cache = crate::runtime::artifact::PackedDwaDenseWeightMaskCache::from_flat(
                        token_set_count,
                        expected_words,
                        ids,
                        rows,
                    )
                    .map_err(crate::GlrMaskError::Serialization)?;
                    constraint.packed_dwa_token_dense_masks = cache;
                    loaded_packed_dwa_dense_masks = true;
                }
                if let Some(dynamic_mask_vocab) = runtime.dynamic_mask_vocab {
                    constraint.dynamic_mask_vocab =
                        crate::runtime::artifact::DynamicMaskVocab::from_artifact(dynamic_mask_vocab)
                            .map_err(crate::GlrMaskError::Serialization)?;
                }

                if let Some(segmented_runtime) = runtime.segmented_runtime {
                    restore_segmented_runtime(&mut constraint, segmented_runtime)?;
                }
            }
            let restore_exprs_started = profile.then(std::time::Instant::now);
            if virtual_runtimes.is_empty() {
                Arc::make_mut(&mut constraint.tokenizer).restore_terminal_exprs(artifact.terminal_exprs)
                    .map_err(crate::GlrMaskError::Serialization)?;
            } else {
                let compiled_static_residual = !constraint.uses_dynamic_runtime()
                    && static_virtual_residual_mask.as_ref().is_some_and(|static_mask| {
                        static_mask.projections().len() == virtual_runtimes.len()
                            && static_mask.projections().iter().all(|projection| !projection.runtime_expr_bytes().is_empty())
                            && virtual_runtimes.iter().all(|entry| entry.kind == crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeKind::ResidualExpr)
                    });
                let restore_result = if compiled_static_residual {
                    Arc::make_mut(&mut constraint.tokenizer).restore_compiled_static_residual_runtimes(
                        &virtual_runtimes, static_virtual_residual_mask.as_ref().unwrap().projections(),
                    )
                } else {
                    let terminal_exprs = artifact.terminal_exprs.or_else(|| {
                        constraint.retained_terminal_exprs().map(|exprs| exprs.to_vec())
                    });
                    if constraint.uses_dynamic_runtime() {
                        Arc::make_mut(&mut constraint.tokenizer).restore_terminal_exprs_with_virtual_runtime_metadata(
                            terminal_exprs, &virtual_runtimes, false,
                        )
                    } else if let Some(static_mask) = static_virtual_residual_mask
                        .as_ref()
                        .filter(|static_mask| {
                            static_mask
                                .projections()
                                .iter()
                                .all(|projection| !projection.oracle_bytes().is_empty())
                        })
                    {
                        Arc::make_mut(&mut constraint.tokenizer).restore_terminal_exprs_with_precompiled_static_residual_oracles(
                            terminal_exprs, &virtual_runtimes, static_mask.projections(), false,
                        )
                    } else {
                        Arc::make_mut(&mut constraint.tokenizer).restore_terminal_exprs_with_virtual_runtime_metadata_preserving_residual_coordinates(
                            terminal_exprs, &virtual_runtimes, false,
                        )
                    }
                };
                restore_result.map_err(crate::GlrMaskError::Serialization)?;
            }
            if let Some(static_mask) = static_virtual_residual_mask {
                static_mask.restore_projections(&mut constraint)?;
            }
            let restore_exprs_ms = restore_exprs_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            if profile {
                eprintln!(
                    "[glrmask/profile][constraint_post_decode] token_bytes_ms={token_bytes_ms:.3} attach_weights_ms={attach_weights_ms:.3} attach_dwa_ms={attach_dwa_ms:.3} invert_original_map_ms={invert_ms:.3} restore_exprs_ms={restore_exprs_ms:.3}"
                );
            }
            constraint
        };
        let deserialize_ms = deserialize_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if !constraint.parser_state_domain_labels.is_empty() {
            if constraint.parser_state_domain_labels.len() != constraint.table.num_states as usize {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "parser-state domain map has {} entries for {} parser states",
                    constraint.parser_state_domain_labels.len(),
                    constraint.table.num_states,
                )));
            }
            let first_synthetic = constraint.table.num_states as i64;
            let default_label = crate::compiler::glr::labels::DEFAULT_LABEL as i64;
            for &label in &constraint.parser_state_domain_labels {
                if label == i32::MAX {
                    continue;
                }
                let label64 = label as i64;
                if label64 < first_synthetic || label64 >= default_label {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "invalid parser-state domain label {label} for {} parser states",
                        constraint.table.num_states,
                    )));
                }
            }
        }
        let rebuild_started = profile.then(std::time::Instant::now);
        let skip_runtime_rebuild_for_profile =
            std::env::var_os("GLRMASK_SKIP_RUNTIME_REBUILD_FOR_PROFILE").is_some();
        if !skip_runtime_rebuild_for_profile {
            // Current late-bind artifacts are written with linker-only
            // placeholder token IDs removed from the public token coordinate.
            // Repair older/current-process artifacts that were saved before
            // that invariant was enforced, before any derived mask cache is
            // rebuilt against the smaller public `mask_len()`.
            constraint.sanitize_late_grammar_placeholder_token_domain();
            if constraint.uses_dynamic_runtime() {
                constraint.rebuild_dynamic_runtime_caches();
            } else {
                if loaded_packed_dwa_dense_masks {
                    constraint.rebuild_runtime_caches_preserving_packed_dwa_dense_masks();
                } else {
                    constraint.rebuild_runtime_caches();
                }
            }
        }
        if let Some(total_started) = total_started {
            let rebuild_ms = rebuild_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            eprintln!(
                "[glrmask/profile][constraint_load] bytes={} decompress_ms={decompress_ms:.3} deserialize_ms={deserialize_ms:.3} rebuild_ms={rebuild_ms:.3} total_ms={:.3}",
                bytes.len(),
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        {
            constraint.serialized_artifact_cache = current_backing.or_else(|| {
                Some(owned_artifact.unwrap_or_else(|| std::sync::Arc::new(bytes.to_vec())))
            });
        }
        if let Some(wire)=prepared_completion_wire {
            let started=std::time::Instant::now();
            let prepared=crate::compiler::composition::boundary::precomputed_completion::PreparedCompletion::load(
                Arc::clone(&constraint.tokenizer),wire,
            ).map_err(crate::GlrMaskError::Serialization)?;
            constraint.boundary_completion_index=Some(Arc::new(prepared));
            if std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some(){
                eprintln!("[glrmask/profile][component_completion_load] certified=true ms={:.3}",started.elapsed().as_secs_f64()*1000.0);
            }
        }
        Ok(constraint)
    }
}
