//! Versioned section framing and current runtime wire structures.

use super::{
    Arc, DecodedStaticVirtualResidualMask, Deserialize, SegmentedRuntimeArtifact,
    SegmentedRuntimeArtifactRef, Serialize, StaticVirtualResidualMaskArtifact,
    StaticVirtualResidualMaskArtifactRef, decode_static_virtual_residual_mask_wire,
};

pub(super) const ROOT_POLICY_MAGIC: &[u8; 8] = b"GLRROOT2";

pub(super) const CONSTRAINT_MAGIC: [u8; 8] = *b"GLRCONS\0";

pub(super) const CONSTRAINT_VERSION: u16 = 33;

pub(super) const CONSTRAINT_HEADER_LEN: usize = CONSTRAINT_MAGIC.len() + 2 + 8;

pub(super) const SECTION_MAGIC: [u8; 4] = *b"S33\0";

pub(super) const SECTION_HEADER_LEN: usize = SECTION_MAGIC.len() + 11 * 8;

pub(super) const CURRENT_RUNTIME_MAGIC: [u8; 4] = *b"R29\0";

pub(super) const CURRENT_RUNTIME_HEADER_LEN: usize = CURRENT_RUNTIME_MAGIC.len() + 2 * 8;

pub(super) fn encode_current_runtime_wire(
    metadata: &ConstraintArtifactCurrentRuntimeRef<'_>,
    static_residual: &[u8],
) -> Vec<u8> {
    let mut meta = Vec::with_capacity(256 * 1024);
    bincode::serialize_into(&mut meta, metadata).expect("constraint runtime metadata serialization should succeed");
    let mut out = Vec::with_capacity(CURRENT_RUNTIME_HEADER_LEN + meta.len() + static_residual.len());
    out.extend_from_slice(&CURRENT_RUNTIME_MAGIC);
    out.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    out.extend_from_slice(&(static_residual.len() as u64).to_le_bytes());
    out.extend_from_slice(&meta);
    out.extend_from_slice(static_residual);
    out
}

pub(super) fn decode_current_runtime_wire(
    section: &[u8],
    backing: Arc<Vec<u8>>,
) -> Result<(ConstraintArtifactCurrentRuntime, Option<DecodedStaticVirtualResidualMask>), String> {
    if section.len() < CURRENT_RUNTIME_HEADER_LEN || !section.starts_with(&CURRENT_RUNTIME_MAGIC) {
        return Err("invalid current runtime wire header".to_owned());
    }
    let meta_len = usize::try_from(u64::from_le_bytes(section[4..12].try_into().unwrap())).map_err(|_| "runtime metadata length does not fit platform".to_owned())?;
    let static_len = usize::try_from(u64::from_le_bytes(section[12..20].try_into().unwrap())).map_err(|_| "runtime static length does not fit platform".to_owned())?;
    let meta_end = CURRENT_RUNTIME_HEADER_LEN.checked_add(meta_len).ok_or_else(|| "runtime metadata range overflow".to_owned())?;
    let static_end = meta_end.checked_add(static_len).ok_or_else(|| "runtime static range overflow".to_owned())?;
    if static_end != section.len() { return Err("invalid current runtime wire lengths".to_owned()); }
    let mut runtime: ConstraintArtifactCurrentRuntime = bincode::deserialize(&section[CURRENT_RUNTIME_HEADER_LEN..meta_end]).map_err(|err| err.to_string())?;
    if runtime.static_virtual_residual_mask.is_some() { return Err("current runtime metadata unexpectedly embeds static residual payload".to_owned()); }
    let static_mask = if static_len == 0 { None } else { Some(decode_static_virtual_residual_mask_wire(&section[meta_end..static_end], backing)?) };
    runtime.static_virtual_residual_mask = None;
    Ok((runtime, static_mask))
}

#[derive(Serialize)]
pub(super) struct ConstraintArtifactCurrentRuntimeRef<'a> {
    pub(super) terminal_live_states: &'a [Vec<u32>],
    pub(super) segmented_runtime: Option<SegmentedRuntimeArtifactRef<'a>>,
    pub(super) dynamic_mask_vocab: Option<crate::runtime::artifact::DynamicMaskVocabArtifact>,
    pub(super) virtual_runtimes: Vec<crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata>,
    pub(super) static_virtual_residual_mask: Option<StaticVirtualResidualMaskArtifactRef<'a>>,
    pub(super) packed_dwa_dense_mask_ids: &'a [u32],
    pub(super) packed_dwa_dense_mask_rows: &'a [u64],
}

#[derive(Deserialize)]
pub(super) struct ConstraintArtifactCurrentRuntime {
    pub(super) terminal_live_states: Vec<Vec<u32>>,
    pub(super) segmented_runtime: Option<SegmentedRuntimeArtifact>,
    pub(super) dynamic_mask_vocab: Option<crate::runtime::artifact::DynamicMaskVocabArtifact>,
    pub(super) virtual_runtimes: Vec<crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata>,
    pub(super) static_virtual_residual_mask: Option<StaticVirtualResidualMaskArtifact>,
    pub(super) packed_dwa_dense_mask_ids: Vec<u32>,
    pub(super) packed_dwa_dense_mask_rows: Vec<u64>,
}

pub(super) struct DecodedConstraintRuntime {
    pub(super) terminal_live_states: Vec<Vec<u32>>,

    pub(super) segmented_runtime: Option<SegmentedRuntimeArtifact>,
    pub(super) dynamic_mask_vocab: Option<crate::runtime::artifact::DynamicMaskVocabArtifact>,
    pub(super) virtual_runtimes: Vec<crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata>,
    pub(super) static_virtual_residual_mask: Option<DecodedStaticVirtualResidualMask>,
    pub(super) packed_dwa_dense_masks: Option<(Vec<u32>, Vec<u64>)>,
}

/// Borrowed sections of the one supported compiled-constraint format.
///
/// Field names are shared with the writer's section order; no decoding copies
/// the payload or changes the backing allocation used by deferred readers.
#[derive(Debug)]
pub(super) struct ConstraintSections<'a> {
    pub(super) weight: &'a [u8],
    pub(super) dwa: &'a [u8],
    pub(super) table: &'a [u8],
    pub(super) core: &'a [u8],
    pub(super) runtime: &'a [u8],
    pub(super) token_bytes: &'a [u8],
    pub(super) original_map: &'a [u8],
    pub(super) tokenizer: &'a [u8],
    pub(super) internal_masks: &'a [u8],
    pub(super) token_mask_cache: &'a [u8],
    pub(super) composition_metadata: &'a [u8],
}

pub(super) fn constraint_sections(payload: &[u8]) -> Result<ConstraintSections<'_>, String> {
    if payload.len() < SECTION_HEADER_LEN || !payload.starts_with(&SECTION_MAGIC) {
        return Err("invalid current constraint section header".to_owned());
    }
    let mut position = SECTION_HEADER_LEN;
    let mut sections = [&[][..]; 11];
    let lengths = payload[SECTION_MAGIC.len()..SECTION_HEADER_LEN].chunks_exact(8);
    for (section, encoded_length) in sections.iter_mut().zip(lengths) {
        let length = usize::try_from(u64::from_le_bytes(
            encoded_length.try_into().expect("section lengths have fixed width"),
        ))
        .map_err(|_| "constraint section length does not fit this platform".to_owned())?;
        let end = position
            .checked_add(length)
            .ok_or_else(|| "constraint section range overflow".to_owned())?;
        *section = payload
            .get(position..end)
            .ok_or_else(|| "invalid current constraint section lengths".to_owned())?;
        position = end;
    }
    if position != payload.len() {
        return Err("invalid current constraint section lengths".to_owned());
    }
    let [weight, dwa, table, core, runtime, token_bytes, original_map,
         tokenizer, internal_masks, token_mask_cache, composition_metadata] = sections;
    Ok(ConstraintSections {
        weight, dwa, table, core, runtime, token_bytes, original_map,
        tokenizer, internal_masks, token_mask_cache, composition_metadata,
    })
}

#[cfg(test)]
mod tests;
