//! Small core metadata plus independently retained lexer expressions.

use super::{Constraint, Deserialize, Expr, InternalTokenBufMasks, Serialize, deserialize_constraint, serialize_constraint};

pub(super) const CURRENT_CORE_MAGIC: [u8; 4] = *b"C22\0";

pub(super) const CURRENT_CORE_HEADER_LEN: usize = CURRENT_CORE_MAGIC.len() + 4 + 2 * 8;

pub(super) const CURRENT_CORE_FLAG_OMIT_TSID_INVERSE: u32 = 1;

#[derive(Serialize)]
pub(super) struct ConstraintArtifactCurrentCoreBaseRef<'a> {
    #[serde(serialize_with = "serialize_constraint")]
    pub(super) constraint: &'a Constraint,
    pub(super) ignore_expr: &'a Option<Expr>,
    pub(super) parser_state_domain_labels: &'a [i32],
    pub(super) static_dynamic_overlay: &'a Option<crate::runtime::artifact::StaticDynamicOverlayMetadata>,
    pub(super) late_grammar_slots: &'a [crate::runtime::artifact::LateGrammarSlot],
}

#[derive(Deserialize)]
pub(super) struct ConstraintArtifactCurrentCoreBase {
    #[serde(deserialize_with = "deserialize_constraint")]
    pub(super) constraint: Constraint,
    pub(super) ignore_expr: Option<Expr>,
    pub(super) parser_state_domain_labels: Vec<i32>,
    pub(super) static_dynamic_overlay: Option<crate::runtime::artifact::StaticDynamicOverlayMetadata>,
    pub(super) late_grammar_slots: Vec<crate::runtime::artifact::LateGrammarSlot>,
}

pub(super) fn decode_current_core(
    input: &[u8],
    backing: Option<(std::sync::Arc<Vec<u8>>, usize)>,
) -> Result<(
    ConstraintArtifactCurrentCoreBase,
    Option<crate::runtime::artifact::DeferredTerminalExprBytes>,
), String> {
    if input.len() < CURRENT_CORE_HEADER_LEN || !input.starts_with(&CURRENT_CORE_MAGIC) {
        return Err("invalid current constraint core header".to_owned());
    }
    let header_len = CURRENT_CORE_HEADER_LEN;
    let flags = u32::from_le_bytes(input[4..8].try_into().expect("core header checked"));
    if flags & !CURRENT_CORE_FLAG_OMIT_TSID_INVERSE != 0 {
        return Err("unsupported current constraint core flags".to_owned());
    }
    let base_len = u64::from_le_bytes(input[8..16].try_into().expect("core header checked"));
    let expr_len = u64::from_le_bytes(input[16..24].try_into().expect("core header checked"));
    let base_len = usize::try_from(base_len)
        .map_err(|_| "current core base length does not fit platform".to_owned())?;
    let expr_len = usize::try_from(expr_len)
        .map_err(|_| "current core expression length does not fit platform".to_owned())?;
    let base_end = header_len
        .checked_add(base_len)
        .ok_or_else(|| "current core base length overflow".to_owned())?;
    let expr_end = base_end
        .checked_add(expr_len)
        .ok_or_else(|| "current core expression length overflow".to_owned())?;
    if expr_end != input.len() {
        return Err("invalid current constraint core section lengths".to_owned());
    }
    let previous_omit_tsid_inverse =
        crate::runtime::artifact::internal_tsid_inverse_artifact_serde::set_omit(
            flags & CURRENT_CORE_FLAG_OMIT_TSID_INVERSE != 0,
        );
    let decoded = bincode::deserialize::<ConstraintArtifactCurrentCoreBase>(
        &input[header_len..base_end],
    ).map_err(|err| err.to_string());
    crate::runtime::artifact::internal_tsid_inverse_artifact_serde::set_omit(
        previous_omit_tsid_inverse,
    );
    let base = decoded?;
    let exprs = if expr_len == 0 {
        None
    } else if let Some((backing, section_start)) = backing {
        let start = section_start
            .checked_add(base_end)
            .ok_or_else(|| "current core expression backing offset overflow".to_owned())?;
        let end = start
            .checked_add(expr_len)
            .ok_or_else(|| "current core expression backing range overflow".to_owned())?;
        if backing.get(start..end) != Some(&input[base_end..expr_end]) {
            return Err("current core expression bytes do not match artifact backing".to_owned());
        }
        Some(crate::runtime::artifact::DeferredTerminalExprBytes::Backed {
            backing,
            start,
            len: expr_len,
        })
    } else {
        Some(crate::runtime::artifact::DeferredTerminalExprBytes::Owned(
            std::sync::Arc::from(input[base_end..expr_end].to_vec().into_boxed_slice()),
        ))
    };
    Ok((base, exprs))
}

pub(super) struct DecodedConstraintCore {
    pub(super) constraint: Constraint,
    pub(super) ignore_expr: Option<Expr>,
    pub(super) terminal_exprs: Option<Vec<Expr>>,
    pub(super) terminal_exprs_blob: Option<crate::runtime::artifact::DeferredTerminalExprBytes>,
    pub(super) parser_state_domain_labels: Vec<i32>,
    pub(super) internal_token_buf_masks: Vec<InternalTokenBufMasks>,
}
