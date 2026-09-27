//! Validated residual-lexer mask projections backed by artifact bytes.

use super::{Arc, Deserialize, Serialize};

pub(super) const STATIC_RESIDUAL_MASK_MAGIC: [u8; 4] = *b"SRM3";

#[derive(Serialize)]
pub(super) struct StaticVirtualResidualMaskArtifactRef<'a> {
    pub(super) mask_tokenizer_bytes: Vec<u8>,
    pub(super) projections: Vec<
        crate::automata::lexer::tokenizer::VirtualResidualMaskProjectionArtifactRef<'a>,
    >,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StaticVirtualResidualMaskArtifact {
    pub(super) mask_tokenizer_bytes: Vec<u8>,
    pub(super) projections: Vec<
        crate::automata::lexer::tokenizer::VirtualResidualMaskProjectionArtifact,
    >,
}

#[derive(Debug)]
pub(crate) enum DecodedStaticVirtualResidualMask {
    Owned(StaticVirtualResidualMaskArtifact),
    Backed {
        backing: Arc<Vec<u8>>,
        tokenizer_start: usize,
        tokenizer_len: usize,
        projections: Vec<crate::automata::lexer::tokenizer::VirtualResidualMaskProjectionArtifact>,
    },
}

impl DecodedStaticVirtualResidualMask {
    pub(crate) fn projections(&self) -> &[crate::automata::lexer::tokenizer::VirtualResidualMaskProjectionArtifact] {
        match self {
            Self::Owned(artifact) => &artifact.projections,
            Self::Backed { projections, .. } => projections,
        }
    }

    pub(crate) fn restore_projections(
        self,
        constraint: &mut super::Constraint,
    ) -> crate::Result<()> {
        let (mask_tokenizer, projections, compiled_stencil) = match self {
            Self::Owned(StaticVirtualResidualMaskArtifact {
                mask_tokenizer_bytes,
                projections,
            }) => {
                let tokenizer = crate::automata::lexer::tokenizer::artifact_serde::from_fast_bytes(
                    &mask_tokenizer_bytes,
                )
                .map_err(crate::GlrMaskError::Serialization)?;
                (tokenizer, projections, false)
            }
            Self::Backed {
                backing,
                tokenizer_start,
                tokenizer_len,
                projections,
            } => {
                let end = tokenizer_start.checked_add(tokenizer_len).ok_or_else(|| {
                    crate::GlrMaskError::Serialization(
                        "static residual tokenizer range overflow".to_owned(),
                    )
                })?;
                let bytes = backing.get(tokenizer_start..end).ok_or_else(|| {
                    crate::GlrMaskError::Serialization(
                        "static residual tokenizer is outside artifact backing".to_owned(),
                    )
                })?;
                let tokenizer =
                    crate::automata::lexer::tokenizer::artifact_serde::from_fast_bytes_backed(
                        bytes,
                        std::sync::Arc::clone(&backing),
                        tokenizer_start,
                    )
                    .map_err(crate::GlrMaskError::Serialization)?;
                (tokenizer, projections, true)
            }
        };
        let restored = if compiled_stencil {
            constraint
                .tokenizer
                .restore_compiled_virtual_residual_mask_projections(
                    &mask_tokenizer,
                    projections,
                )
        } else {
            constraint
                .tokenizer
                .restore_virtual_residual_mask_projections(
                    constraint.max_token_byte_len(),
                    &mask_tokenizer,
                    projections,
                )
        }
        .map_err(crate::GlrMaskError::Serialization)?;
        constraint
            .dynamic_mask_vocab
            .set_virtual_residuals_mask_projection(mask_tokenizer, restored);
        Ok(())
    }
}

pub(crate) fn encode_static_virtual_residual_mask_wire(
    source_tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
    mask_tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
    projections: &[crate::automata::lexer::tokenizer::VirtualResidualMaskProjection],
) -> Vec<u8> {
    encode_static_virtual_residual_mask_wire_with_fallback(
        source_tokenizer,
        None,
        mask_tokenizer,
        projections,
    )
}

pub(crate) fn encode_static_virtual_residual_mask_wire_with_fallback(
    source_tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
    fallback_exprs: Option<&[crate::automata::lexer::ast::Expr]>,
    mask_tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
    projections: &[crate::automata::lexer::tokenizer::VirtualResidualMaskProjection],
) -> Vec<u8> {
    let tokenizer_bytes = if mask_tokenizer.has_compressed_transition_segments() {
        crate::automata::lexer::tokenizer::artifact_serde::build_huge_bytes(mask_tokenizer)
            .unwrap_or_else(|| {
                crate::automata::lexer::tokenizer::artifact_serde::to_segment_bytes(mask_tokenizer)
            })
    } else {
        crate::automata::lexer::tokenizer::artifact_serde::to_fast_bytes_with_packed_metadata(mask_tokenizer)
    };
    let source_exprs = source_tokenizer
        .terminal_exprs()
        .or(fallback_exprs)
        .expect("residual constraint retains terminal expressions for wire encoding");
    let parts = projections
        .iter()
        .map(|projection| {
            let (terminal, state_offset, map, oracle, mask_max, crossed_boundaries) = projection.artifact_wire_parts();
            let live_count = map.iter().filter(|&&state| state != u32::MAX).count();
            let expression = source_exprs
                .get(terminal as usize)
                .expect("residual projection terminal expression exists");
            let expr = bincode::serialize(expression)
                .expect("residual runtime expression serialization should succeed");
            (terminal, state_offset, map, live_count, oracle, expr, mask_max, crossed_boundaries)
        })
        .collect::<Vec<_>>();
    // SRM3 stores the overwhelmingly-sentinel projection coordinate sparsely
    // as fixed-width (dense_index, mask_state) pairs. Load expands it back to
    // the exact dense runtime representation, so runtime projection is unchanged.
    let descriptor_len = parts.len().saturating_mul(56);
    let maps_len = parts
        .iter()
        .map(|(_, _, _, live_count, _, _, _, _)| live_count.saturating_mul(8))
        .sum::<usize>();
    let oracle_len = parts.iter().map(|(_, _, _, _, oracle, _, _, _)| oracle.len()).sum::<usize>();
    let expr_len = parts.iter().map(|(_, _, _, _, _, expr, _, _)| expr.len()).sum::<usize>();
    let mut out = Vec::with_capacity(16 + descriptor_len + tokenizer_bytes.len() + maps_len + oracle_len + expr_len);
    out.extend_from_slice(&STATIC_RESIDUAL_MASK_MAGIC);
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    out.extend_from_slice(&(tokenizer_bytes.len() as u64).to_le_bytes());
    for (terminal, state_offset, map, live_count, oracle, expr, mask_max, crossed_boundaries) in &parts {
        out.extend_from_slice(&terminal.to_le_bytes());
        out.extend_from_slice(&state_offset.to_le_bytes());
        out.extend_from_slice(&(map.len() as u64).to_le_bytes());
        out.extend_from_slice(&(*live_count as u64).to_le_bytes());
        out.extend_from_slice(&(oracle.len() as u64).to_le_bytes());
        out.extend_from_slice(&(expr.len() as u64).to_le_bytes());
        out.extend_from_slice(&(*mask_max as u64).to_le_bytes());
        out.extend_from_slice(&(*crossed_boundaries as u64).to_le_bytes());
    }
    out.extend_from_slice(&tokenizer_bytes);
    for (_, _, map, _, oracle, expr, _, _) in parts {
        for (index, &state) in map.iter().enumerate() {
            if state == u32::MAX { continue; }
            out.extend_from_slice(&(index as u32).to_le_bytes());
            out.extend_from_slice(&state.to_le_bytes());
        }
        out.extend_from_slice(&oracle);
        out.extend_from_slice(&expr);
    }
    out
}

pub(crate) fn decode_static_virtual_residual_mask_wire(
    section: &[u8],
    backing: Arc<Vec<u8>>,
) -> Result<DecodedStaticVirtualResidualMask, String> {
    if section.len() < 16 {
        return Err("invalid static residual mask wire header".to_owned());
    }
    if !section.starts_with(&STATIC_RESIDUAL_MASK_MAGIC) {
        return Err("invalid static residual mask wire header".to_owned());
    }
    let count = u32::from_le_bytes(section[4..8].try_into().unwrap()) as usize;
    let tokenizer_len = usize::try_from(u64::from_le_bytes(section[8..16].try_into().unwrap()))
        .map_err(|_| "static residual tokenizer length does not fit platform".to_owned())?;
    let descriptor_width = { 56usize };
    let descriptor_bytes = count.checked_mul(descriptor_width).ok_or_else(|| "static residual descriptor length overflow".to_owned())?;
    let mut pos = 16usize;
    let descriptor_end = pos.checked_add(descriptor_bytes).ok_or_else(|| "static residual descriptor overflow".to_owned())?;
    if descriptor_end > section.len() { return Err("truncated static residual descriptors".to_owned()); }
    let mut descriptors = Vec::with_capacity(count);
    while pos < descriptor_end {
        let terminal = u32::from_le_bytes(section[pos..pos+4].try_into().unwrap()); pos += 4;
        let state_offset = u32::from_le_bytes(section[pos..pos+4].try_into().unwrap()); pos += 4;
        let map_len = usize::try_from(u64::from_le_bytes(section[pos..pos+8].try_into().unwrap())).map_err(|_| "static residual map length does not fit platform".to_owned())?; pos += 8;
        let live_count = {
            let value = usize::try_from(u64::from_le_bytes(section[pos..pos+8].try_into().unwrap())).map_err(|_| "static residual live map length does not fit platform".to_owned())?;
            pos += 8;
            if value > map_len { return Err("static residual live map length exceeds dense length".to_owned()); }
            value
        };
        let oracle_len = usize::try_from(u64::from_le_bytes(section[pos..pos+8].try_into().unwrap())).map_err(|_| "static residual oracle length does not fit platform".to_owned())?; pos += 8;
        let expr_len = usize::try_from(u64::from_le_bytes(section[pos..pos+8].try_into().unwrap())).map_err(|_| "static residual expression length does not fit platform".to_owned())?; pos += 8;
        let mask_max = usize::try_from(u64::from_le_bytes(section[pos..pos+8].try_into().unwrap())).map_err(|_| "static residual mask max does not fit platform".to_owned())?; pos += 8;
        let crossed_boundaries = usize::try_from(u64::from_le_bytes(section[pos..pos+8].try_into().unwrap())).map_err(|_| "static residual boundary horizon does not fit platform".to_owned())?; pos += 8;
        descriptors.push((terminal, state_offset, map_len, live_count, oracle_len, expr_len, mask_max, crossed_boundaries));
    }
    let tokenizer_local_start = descriptor_end;
    let tokenizer_local_end = tokenizer_local_start.checked_add(tokenizer_len).ok_or_else(|| "static residual tokenizer range overflow".to_owned())?;
    if tokenizer_local_end > section.len() { return Err("truncated static residual tokenizer".to_owned()); }
    let mask_state_count = {
        let tokenizer_wire = &section[tokenizer_local_start..tokenizer_local_end];
        if tokenizer_wire.len() < 12
            || !(tokenizer_wire.starts_with(b"TKF2")
                || tokenizer_wire.starts_with(b"TKF3")
                || tokenizer_wire.starts_with(b"TKS2")
                || tokenizer_wire.starts_with(b"TKS3"))
        {
            return Err("SRM3 requires a supported mask tokenizer wire".to_owned());
        }
        u32::from_le_bytes(tokenizer_wire[8..12].try_into().unwrap())
    };
    {
        for index in 0..descriptors.len() {
            let start = descriptors[index].1;
            let end = descriptors.get(index + 1).map_or(mask_state_count, |next| next.1);
            if start >= end || end > mask_state_count {
                return Err("invalid static residual projection component range".to_owned());
            }
        }
    }
    let base = backing.as_ptr() as usize;
    let section_start = (section.as_ptr() as usize).checked_sub(base).ok_or_else(|| "static residual section does not belong to artifact backing".to_owned())?;
    let tokenizer_start = section_start.checked_add(tokenizer_local_start).ok_or_else(|| "static residual tokenizer backing offset overflow".to_owned())?;
    pos = tokenizer_local_end;
    let mut projections = Vec::with_capacity(count);
    for (descriptor_index, (terminal, state_offset, map_len, live_count, oracle_len, expr_len, mask_max, crossed_boundaries)) in descriptors.iter().copied().enumerate() {
        let component_state_count = {
            descriptors
                .get(descriptor_index + 1)
                .map_or(mask_state_count, |next| next.1)
                .checked_sub(state_offset)
                .ok_or_else(|| "invalid static residual projection component width".to_owned())?
        };
        let mut map = { vec![u32::MAX; map_len] };
        {
            let map_bytes = live_count.checked_mul(8).ok_or_else(|| "static residual sparse map byte length overflow".to_owned())?;
            let map_end = pos.checked_add(map_bytes).ok_or_else(|| "static residual sparse map range overflow".to_owned())?;
            if map_end > section.len() { return Err("truncated static residual sparse projection map".to_owned()); }
            let mut previous = None;
            for pair in section[pos..map_end].chunks_exact(8) {
                let index = u32::from_le_bytes(pair[0..4].try_into().unwrap()) as usize;
                let state = u32::from_le_bytes(pair[4..8].try_into().unwrap());
                if index >= map_len
                    || state == u32::MAX
                    || state >= component_state_count
                    || previous.is_some_and(|prev| index <= prev)
                {
                    return Err("invalid static residual sparse projection entry".to_owned());
                }
                map[index] = state;
                previous = Some(index);
            }
            pos = map_end;
        }
        let oracle_end = pos.checked_add(oracle_len).ok_or_else(|| "static residual oracle range overflow".to_owned())?;
        if oracle_end > section.len() { return Err("truncated static residual oracle".to_owned()); }
        let oracle_bytes = section[pos..oracle_end].to_vec();
        pos = oracle_end;
        let expr_end = pos.checked_add(expr_len).ok_or_else(|| "static residual expression range overflow".to_owned())?;
        if expr_end > section.len() { return Err("truncated static residual expression".to_owned()); }
        let runtime_expr_bytes = section[pos..expr_end].to_vec();
        pos = expr_end;
        let projection = {
            crate::automata::lexer::tokenizer::VirtualResidualMaskProjectionArtifact::from_validated_wire(
                terminal, state_offset, map, oracle_bytes, runtime_expr_bytes, mask_max,
                crossed_boundaries, component_state_count,
            )
        };
        projections.push(projection);
    }
    if pos != section.len() { return Err("trailing bytes in static residual mask wire".to_owned()); }
    Ok(DecodedStaticVirtualResidualMask::Backed { backing, tokenizer_start, tokenizer_len, projections })
}
