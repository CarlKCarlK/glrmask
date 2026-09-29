// PRE-RELEASE COMPATIBILITY POLICY
// ================================
// GLRMask has not reached a compatibility-stable release. Old serialized
// constraint versions and historical runtime layouts are NOT a design
// requirement. Prefer the cleanest exact current representation even when that
// means old development artifacts stop loading. Existing legacy readers below
// are transitional code, not an API promise: delete or simplify them whenever
// they complicate the architecture, correctness proof, performance, or wire
// format. Do not add compatibility machinery unless compatibility is explicitly
// made a requirement again.

use crate::automata::lexer::Lexer;
use crate::runtime::Constraint;
use crate::runtime::artifact::{
    BackedInternalTokenBufMasks, ConstraintSerde, DenseBufMaskRows, InternalTokenBufMasks,
    PackedInternalTokenBufMask,
};
use crate::automata::regex::Expr;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;
use std::io::Read;
use std::sync::Arc;

const PREVIOUS_ROOT_POLICY_MAGIC: &[u8; 8] = b"GLRROOT1";
const ROOT_POLICY_MAGIC: &[u8; 8] = b"GLRROOT2";

const CONSTRAINT_MAGIC: [u8; 8] = *b"GLRCONS\0";
const LEGACY_CONSTRAINT_VERSION: u16 = 7;
const PREVIOUS_COMPRESSED_CONSTRAINT_VERSION: u16 = 9;
const PREVIOUS_EXPRLESS_CONSTRAINT_VERSION: u16 = 10;
const PREVIOUS_TERMINAL_EXPRS_CONSTRAINT_VERSION: u16 = 11;
const PREVIOUS_DOMAIN_LABELS_CONSTRAINT_VERSION: u16 = 12;
const PREVIOUS_UNCOMPRESSED_CONSTRAINT_VERSION: u16 = 13;
const PREVIOUS_SECTIONED_CONSTRAINT_VERSION: u16 = 14;
const PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION: u16 = 15;
const PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION: u16 = 16;
const PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION: u16 = 17;
const PREVIOUS_EXTERNAL_RUNTIME_CONSTRAINT_VERSION: u16 = 18;
/// Last artifact published by the serialization-only branch. The combined
/// composition/serialization format keeps its section framing but extends the
/// current core/runtime payloads, so v19 remains independently loadable.
const PREVIOUS_SERIALIZATION_CURRENT_CONSTRAINT_VERSION: u16 = 19;
const PREVIOUS_COMBINED_CONSTRAINT_VERSION: u16 = 20;
const PREVIOUS_SEGMENTED_MATERIALIZATION_CONSTRAINT_VERSION: u16 = 21;
const PREVIOUS_BOUNDARY_SHARDLESS_CONSTRAINT_VERSION: u16 = 22;
const PREVIOUS_BOUNDARY_SHARDED_CONSTRAINT_VERSION: u16 = 23;
const PREVIOUS_RECURSIVE_PARSER_CONSTRAINT_VERSION: u16 = 24;
const PREVIOUS_STATIC_RESIDUAL_CONSTRAINT_VERSION: u16 = 27;
const PREVIOUS_STATIC_PROJECTION_CONSTRAINT_VERSION: u16 = 28;
const PREVIOUS_BOUNDARY_SUMMARYLESS_CONSTRAINT_VERSION: u16 = 29;
const CONSTRAINT_VERSION: u16 = 30;
// V31 retains S30 framing but replaces the LR section with a TPR1 acyclic
// parser program. Ordinary LR artifacts deliberately continue writing V30.
const TEMPLATE_CONSTRAINT_VERSION: u16 = 31;
const CONSTRAINT_HEADER_LEN: usize = CONSTRAINT_MAGIC.len() + 2 + 8;
const COMPRESSED_PAYLOAD_HEADER_LEN: usize = 8;
const CONSTRAINT_COMPRESSION_LEVEL: i32 = 1;
const V14_SECTION_MAGIC: [u8; 4] = *b"S14\0";
const V14_SECTION_HEADER_LEN: usize = V14_SECTION_MAGIC.len() + 4 * 8;
const V15_SECTION_MAGIC: [u8; 4] = *b"S15\0";
const V15_SECTION_HEADER_LEN: usize = V15_SECTION_MAGIC.len() + 5 * 8;
const V16_SECTION_MAGIC: [u8; 4] = *b"S16\0";
const V16_SECTION_HEADER_LEN: usize = V16_SECTION_MAGIC.len() + 5 * 8;
const V17_SECTION_MAGIC: [u8; 4] = *b"S17\0";
const V17_SECTION_HEADER_LEN: usize = V17_SECTION_MAGIC.len() + 6 * 8;
const V18_SECTION_MAGIC: [u8; 4] = *b"S18\0";
const V18_SECTION_HEADER_LEN: usize = V18_SECTION_MAGIC.len() + 9 * 8;
const V19_SECTION_MAGIC: [u8; 4] = *b"S19\0";
const V19_SECTION_HEADER_LEN: usize = V19_SECTION_MAGIC.len() + 10 * 8;
const V20_SECTION_MAGIC: [u8; 4] = *b"S20\0";
const V20_SECTION_HEADER_LEN: usize = V20_SECTION_MAGIC.len() + 11 * 8;
const V21_SECTION_MAGIC: [u8; 4] = *b"S21\0";
const V21_SECTION_HEADER_LEN: usize = V21_SECTION_MAGIC.len() + 11 * 8;
const V22_SECTION_MAGIC: [u8; 4] = *b"S22\0";
const V22_SECTION_HEADER_LEN: usize = V22_SECTION_MAGIC.len() + 11 * 8;
const V23_SECTION_MAGIC: [u8; 4] = *b"S23\0";
const V23_SECTION_HEADER_LEN: usize = V23_SECTION_MAGIC.len() + 11 * 8;
const V24_SECTION_MAGIC: [u8; 4] = *b"S24\0";
const V24_SECTION_HEADER_LEN: usize = V24_SECTION_MAGIC.len() + 11 * 8;
const V27_SECTION_MAGIC: [u8; 4] = *b"S27\0";
const V27_SECTION_HEADER_LEN: usize = V27_SECTION_MAGIC.len() + 11 * 8;
const V28_SECTION_MAGIC: [u8; 4] = *b"S28\0";
const V28_SECTION_HEADER_LEN: usize = V28_SECTION_MAGIC.len() + 11 * 8;
const V29_SECTION_MAGIC: [u8; 4] = *b"S29\0";
const V29_SECTION_HEADER_LEN: usize = V29_SECTION_MAGIC.len() + 11 * 8;
const V30_SECTION_MAGIC: [u8; 4] = *b"S30\0";
const V30_SECTION_HEADER_LEN: usize = V30_SECTION_MAGIC.len() + 11 * 8;
const CURRENT_RUNTIME_MAGIC: [u8; 4] = *b"R29\0";
const CURRENT_RUNTIME_HEADER_LEN: usize = CURRENT_RUNTIME_MAGIC.len() + 2 * 8;
const PREVIOUS_STATIC_RESIDUAL_MASK_MAGIC: [u8; 4] = *b"SRM2";
const STATIC_RESIDUAL_MASK_MAGIC: [u8; 4] = *b"SRM3";
const PREVIOUS_PREVIOUS_PREVIOUS_CURRENT_CORE_MAGIC: [u8; 4] = *b"C19\0";
const PREVIOUS_PREVIOUS_PREVIOUS_CURRENT_CORE_HEADER_LEN: usize =
    PREVIOUS_PREVIOUS_PREVIOUS_CURRENT_CORE_MAGIC.len() + 2 * 8;
const PREVIOUS_PREVIOUS_CURRENT_CORE_MAGIC: [u8; 4] = *b"C20\0";
const PREVIOUS_PREVIOUS_CURRENT_CORE_HEADER_LEN: usize =
    PREVIOUS_PREVIOUS_CURRENT_CORE_MAGIC.len() + 4 + 2 * 8;
const PREVIOUS_CURRENT_CORE_MAGIC: [u8; 4] = *b"C21\0";
const PREVIOUS_CURRENT_CORE_HEADER_LEN: usize = PREVIOUS_CURRENT_CORE_MAGIC.len() + 4 + 2 * 8;
const CURRENT_CORE_MAGIC: [u8; 4] = *b"C22\0";
const CURRENT_CORE_HEADER_LEN: usize = CURRENT_CORE_MAGIC.len() + 4 + 2 * 8;
const CURRENT_CORE_FLAG_OMIT_TSID_INVERSE: u32 = 1;

#[inline]
fn uses_external_runtime_sections(version: u16) -> bool {
    matches!(
        version,
        CONSTRAINT_VERSION
            | TEMPLATE_CONSTRAINT_VERSION
            | PREVIOUS_BOUNDARY_SUMMARYLESS_CONSTRAINT_VERSION
            | PREVIOUS_STATIC_PROJECTION_CONSTRAINT_VERSION
            | PREVIOUS_STATIC_RESIDUAL_CONSTRAINT_VERSION
            | PREVIOUS_RECURSIVE_PARSER_CONSTRAINT_VERSION
            | PREVIOUS_BOUNDARY_SHARDED_CONSTRAINT_VERSION
            | PREVIOUS_BOUNDARY_SHARDLESS_CONSTRAINT_VERSION
            | PREVIOUS_SEGMENTED_MATERIALIZATION_CONSTRAINT_VERSION
            | PREVIOUS_COMBINED_CONSTRAINT_VERSION
            | PREVIOUS_SERIALIZATION_CURRENT_CONSTRAINT_VERSION
            | PREVIOUS_EXTERNAL_RUNTIME_CONSTRAINT_VERSION
    )
}

/// Clone a large already-serialized artifact without forcing one core to move
/// the entire byte slab.  Current JS artifacts are ~20 MiB, so a serial
/// `Vec::clone` is itself multiple milliseconds on some hosts even though the
/// bytes need no transformation at all.
fn clone_serialized_artifact(bytes: &[u8]) -> Vec<u8> {
    const PARALLEL_COPY_MIN_BYTES: usize = 4 * 1024 * 1024;
    let len = bytes.len();
    if len < PARALLEL_COPY_MIN_BYTES || rayon::current_num_threads() <= 1 {
        return bytes.to_vec();
    }

    // On the 16-thread Windows benchmark host, 12 ~1.7 MiB chunks saturate
    // memory bandwidth without the scheduler overhead seen at 16+ chunks.
    // Cap rather than multiplying the worker count: this operation is pure
    // bandwidth, not compute.
    let target_chunks = rayon::current_num_threads().clamp(2, 12);
    let chunk_size = len.div_ceil(target_chunks).max(1024 * 1024);
    let chunk_count = len.div_ceil(chunk_size);
    let mut out = Vec::<u8>::with_capacity(len);
    let dst_base = out.as_mut_ptr() as usize;
    let src_base = bytes.as_ptr() as usize;
    (0..chunk_count).into_par_iter().for_each(|chunk| {
        let start = chunk * chunk_size;
        let count = (len - start).min(chunk_size);
        // SAFETY: each Rayon job owns one disjoint destination range, both
        // source and destination allocations remain alive for the full join,
        // and `count` is bounded by `len - start`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (src_base + start) as *const u8,
                (dst_base + start) as *mut u8,
                count,
            );
        }
    });
    // SAFETY: every byte in 0..len was initialized by exactly one job above.
    unsafe {
        out.set_len(len);
    }
    out
}

fn serialize_constraint<S>(constraint: &&Constraint, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    ConstraintSerde::serialize(*constraint, serializer)
}

fn deserialize_constraint<'de, D>(deserializer: D) -> Result<Constraint, D::Error>
where
    D: serde::Deserializer<'de>,
{
    ConstraintSerde::deserialize(deserializer)
}

struct DeserializedConstraint(Constraint);

impl<'de> serde::Deserialize<'de> for DeserializedConstraint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        ConstraintSerde::deserialize(deserializer).map(Self)
    }
}

struct SerializedConstraint<'a>(&'a Constraint);

impl serde::Serialize for SerializedConstraint<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        ConstraintSerde::serialize(self.0, serializer)
    }
}

#[derive(Serialize)]
struct ConstraintArtifactV10Ref<'a> {
    #[serde(serialize_with = "serialize_constraint")]
    constraint: &'a Constraint,
    ignore_expr: &'a Option<Expr>,
}

#[derive(Deserialize)]
struct ConstraintArtifactV10 {
    #[serde(deserialize_with = "deserialize_constraint")]
    constraint: Constraint,
    ignore_expr: Option<Expr>,
}

#[derive(Serialize)]
struct ConstraintArtifactV11Ref<'a> {
    #[serde(serialize_with = "serialize_constraint")]
    constraint: &'a Constraint,
    ignore_expr: &'a Option<Expr>,
    terminal_exprs: Option<&'a [Expr]>,
}

#[derive(Deserialize)]
struct ConstraintArtifactV11 {
    #[serde(deserialize_with = "deserialize_constraint")]
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    terminal_exprs: Option<Vec<Expr>>,
}

#[derive(Serialize)]
struct ConstraintArtifactV12Ref<'a> {
    #[serde(serialize_with = "serialize_constraint")]
    constraint: &'a Constraint,
    ignore_expr: &'a Option<Expr>,
    terminal_exprs: Option<&'a [Expr]>,
    parser_state_domain_labels: &'a [i32],
}

#[derive(Deserialize)]
struct ConstraintArtifactV12 {
    #[serde(deserialize_with = "deserialize_constraint")]
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    terminal_exprs: Option<Vec<Expr>>,
    parser_state_domain_labels: Vec<i32>,
}

#[derive(Serialize)]
struct ConstraintArtifactV13Ref<'a> {
    /// Packed pool for all Weight-bearing Constraint fields outside parser_dwa.
    /// parser_dwa has its own packed pool because its transition topology is
    /// encoded together with the weight references.
    weight_pool: &'a [u8],
    #[serde(serialize_with = "serialize_constraint")]
    constraint: &'a Constraint,
    ignore_expr: &'a Option<Expr>,
    terminal_exprs: Option<&'a [Expr]>,
    parser_state_domain_labels: &'a [i32],
    /// Portable runtime-ready mapping from each internal token to the output
    /// token-mask fragments it represents. Recomputing this from
    /// internal_token_to_tokens is a substantial part of large-constraint load.
    internal_token_buf_masks: &'a [InternalTokenBufMasks],
}

struct PooledWeightDecodeActivator;

impl<'de> Deserialize<'de> for PooledWeightDecodeActivator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        let weights = crate::ds::weight::unpack_pooled_weights(&bytes)
            .map_err(serde::de::Error::custom)?;
        crate::ds::weight::begin_pooled_weight_serde_decode(weights);
        Ok(Self)
    }
}

#[derive(Deserialize)]
struct ConstraintArtifactV13 {
    _weight_pool: PooledWeightDecodeActivator,
    #[serde(deserialize_with = "deserialize_constraint")]
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    terminal_exprs: Option<Vec<Expr>>,
    parser_state_domain_labels: Vec<i32>,
    internal_token_buf_masks: Vec<InternalTokenBufMasks>,
}

#[derive(Serialize)]
struct ConstraintArtifactV14CoreRef<'a> {
    #[serde(serialize_with = "serialize_constraint")]
    constraint: &'a Constraint,
    ignore_expr: &'a Option<Expr>,
    terminal_exprs: Option<&'a [Expr]>,
    parser_state_domain_labels: &'a [i32],
    internal_token_buf_masks: &'a [InternalTokenBufMasks],
}

#[derive(Deserialize)]
struct ConstraintArtifactV14Core {
    #[serde(deserialize_with = "deserialize_constraint")]
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    terminal_exprs: Option<Vec<Expr>>,
    parser_state_domain_labels: Vec<i32>,
    internal_token_buf_masks: Vec<InternalTokenBufMasks>,
}

#[derive(Serialize)]
struct ConstraintArtifactV18CoreRef<'a> {
    #[serde(serialize_with = "serialize_constraint")]
    constraint: &'a Constraint,
    ignore_expr: &'a Option<Expr>,
    terminal_exprs: Option<&'a [Expr]>,
    parser_state_domain_labels: &'a [i32],
}

#[derive(Deserialize)]
struct ConstraintArtifactV18Core {
    #[serde(deserialize_with = "deserialize_constraint")]
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    terminal_exprs: Option<Vec<Expr>>,
    parser_state_domain_labels: Vec<i32>,
}

#[derive(Serialize)]
struct ConstraintArtifactCurrentCoreBaseRef<'a> {
    #[serde(serialize_with = "serialize_constraint")]
    constraint: &'a Constraint,
    ignore_expr: &'a Option<Expr>,
    parser_state_domain_labels: &'a [i32],
    static_dynamic_overlay: &'a Option<crate::runtime::artifact::StaticDynamicOverlayMetadata>,
    late_grammar_slots: &'a [crate::runtime::artifact::LateGrammarSlot],
}

#[derive(Deserialize)]
struct ConstraintArtifactCurrentCoreBase {
    #[serde(deserialize_with = "deserialize_constraint")]
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    parser_state_domain_labels: Vec<i32>,
    static_dynamic_overlay: Option<crate::runtime::artifact::StaticDynamicOverlayMetadata>,
    late_grammar_slots: Vec<crate::runtime::artifact::LateGrammarSlot>,
}

/// Core payload used by C21, before unresolved named slots were persisted.
#[derive(Deserialize)]
struct ConstraintArtifactPreviousOverlayCoreBase {
    #[serde(deserialize_with = "deserialize_constraint")]
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    parser_state_domain_labels: Vec<i32>,
    static_dynamic_overlay: Option<crate::runtime::artifact::StaticDynamicOverlayMetadata>,
}

/// Core payload used by serialization artifact v19 (C19/C20). Composition
/// metadata lived only on the diverged composition branch at that point, so it
/// is absent from the canonical v19 payload and must default empty on load.
#[derive(Deserialize)]
struct ConstraintArtifactPreviousCurrentCoreBase {
    #[serde(deserialize_with = "deserialize_constraint")]
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    parser_state_domain_labels: Vec<i32>,
}

fn decode_current_core(
    input: &[u8],
    backing: Option<(std::sync::Arc<Vec<u8>>, usize)>,
) -> Result<(
    ConstraintArtifactCurrentCoreBase,
    Option<crate::runtime::artifact::DeferredTerminalExprBytes>,
), String> {
    let (header_len, flags, base_len, expr_len, has_current_overlay, has_late_slots) =
        if input.starts_with(&CURRENT_CORE_MAGIC) {
        if input.len() < CURRENT_CORE_HEADER_LEN {
            return Err("truncated current constraint core header".to_owned());
        }
        let flags = u32::from_le_bytes(
            input[4..8]
                .try_into()
                .expect("current core flags have fixed width"),
        );
        if flags & !CURRENT_CORE_FLAG_OMIT_TSID_INVERSE != 0 {
            return Err("unsupported current constraint core flags".to_owned());
        }
        let base_len = u64::from_le_bytes(
            input[8..16]
                .try_into()
                .expect("current core base length has fixed width"),
        );
        let expr_len = u64::from_le_bytes(
            input[16..24]
                .try_into()
                .expect("current core expression length has fixed width"),
        );
        (CURRENT_CORE_HEADER_LEN, flags, base_len, expr_len, true, true)
    } else if input.starts_with(&PREVIOUS_CURRENT_CORE_MAGIC) {
        if input.len() < PREVIOUS_CURRENT_CORE_HEADER_LEN {
            return Err("truncated previous current-core header".to_owned());
        }
        let flags = u32::from_le_bytes(
            input[4..8]
                .try_into()
                .expect("previous current-core flags have fixed width"),
        );
        if flags & !CURRENT_CORE_FLAG_OMIT_TSID_INVERSE != 0 {
            return Err("unsupported previous current constraint core flags".to_owned());
        }
        let base_len = u64::from_le_bytes(
            input[8..16]
                .try_into()
                .expect("previous current-core base length has fixed width"),
        );
        let expr_len = u64::from_le_bytes(
            input[16..24]
                .try_into()
                .expect("previous current-core expression length has fixed width"),
        );
        (
            PREVIOUS_CURRENT_CORE_HEADER_LEN,
            flags,
            base_len,
            expr_len,
            true,
            false,
        )
    } else if input.starts_with(&PREVIOUS_PREVIOUS_CURRENT_CORE_MAGIC) {
        if input.len() < PREVIOUS_PREVIOUS_CURRENT_CORE_HEADER_LEN {
            return Err("truncated previous-previous current-core header".to_owned());
        }
        let base_len = u64::from_le_bytes(
            input[4..12]
                .try_into()
                .expect("previous-previous current-core base length has fixed width"),
        );
        let expr_len = u64::from_le_bytes(
            input[12..20]
                .try_into()
                .expect("previous-previous current-core expression length has fixed width"),
        );
        (
            PREVIOUS_PREVIOUS_CURRENT_CORE_HEADER_LEN,
            0,
            base_len,
            expr_len,
            false,
            false,
        )
    } else if input.starts_with(&PREVIOUS_PREVIOUS_PREVIOUS_CURRENT_CORE_MAGIC) {
        if input.len() < PREVIOUS_PREVIOUS_PREVIOUS_CURRENT_CORE_HEADER_LEN {
            return Err("truncated previous-previous-previous current-core header".to_owned());
        }
        let base_len = u64::from_le_bytes(
            input[4..12]
                .try_into()
                .expect("previous-previous-previous current-core base length has fixed width"),
        );
        let expr_len = u64::from_le_bytes(
            input[12..20]
                .try_into()
                .expect("previous-previous-previous current-core expression length has fixed width"),
        );
        (
            PREVIOUS_PREVIOUS_PREVIOUS_CURRENT_CORE_HEADER_LEN,
            0,
            base_len,
            expr_len,
            false,
            false,
        )
    } else {
        return Err("invalid current constraint core header".to_owned());
    };
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
    let decoded = if has_late_slots {
        bincode::deserialize::<ConstraintArtifactCurrentCoreBase>(&input[header_len..base_end])
            .map_err(|err| err.to_string())
    } else if has_current_overlay {
        bincode::deserialize::<ConstraintArtifactPreviousOverlayCoreBase>(
            &input[header_len..base_end],
        )
        .map(|base| ConstraintArtifactCurrentCoreBase {
            constraint: base.constraint,
            ignore_expr: base.ignore_expr,
            parser_state_domain_labels: base.parser_state_domain_labels,
            static_dynamic_overlay: base.static_dynamic_overlay,
            late_grammar_slots: Vec::new(),
        })
        .map_err(|err| err.to_string())
    } else {
        bincode::deserialize::<ConstraintArtifactPreviousCurrentCoreBase>(
            &input[header_len..base_end],
        )
        .map(|base| ConstraintArtifactCurrentCoreBase {
            constraint: base.constraint,
            ignore_expr: base.ignore_expr,
            parser_state_domain_labels: base.parser_state_domain_labels,
            static_dynamic_overlay: None,
            late_grammar_slots: Vec::new(),
        })
        .map_err(|err| err.to_string())
    };
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

const PREVIOUS_COMPOSITION_METADATA_RAW_MAGIC: [u8; 4] = *b"CMP1";
const PREVIOUS_COMPOSITION_METADATA_ZSTD_MAGIC: [u8; 4] = *b"CMZ1";
const COMPOSITION_METADATA_RAW_MAGIC: [u8; 4] = *b"CMP2";
const COMPOSITION_METADATA_ZSTD_MAGIC: [u8; 4] = *b"CMZ2";
const PREVIOUS_PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC: [u8; 4] = *b"CMS3";
const PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC: [u8; 4] = *b"CMS4";
const COMPOSITION_METADATA_SPLIT_MAGIC: [u8; 4] = *b"CMS5";
const COMPOSITION_METADATA_HEADER_LEN: usize = 12;
const COMPOSITION_METADATA_SPLIT_HEADER_LEN: usize = 40;
const COMPOSITION_METADATA_COMPRESS_MIN_BYTES: usize = 64 * 1024;
const COMPOSITION_METADATA_SPLIT_LINK_COMPRESSED: u32 = 1 << 0;
const COMPOSITION_METADATA_SPLIT_CACHE_COMPRESSED: u32 = 1 << 1;

#[derive(Serialize)]
enum BoundaryTriggerWireRef<'a> {
    None,
    Tokens(&'a [u32]),
    Exact(&'a crate::automata::weighted_u32::dwa::DWA),
}

#[derive(Serialize, Deserialize)]
enum BoundaryTriggerWire {
    None,
    Tokens(Vec<u32>),
    Exact(crate::automata::weighted_u32::dwa::DWA),
}

fn boundary_trigger_wire_ref(
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

fn restore_boundary_trigger(trigger: BoundaryTriggerWire) -> crate::runtime::BoundaryTrigger {
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
enum BoundaryOriginalTokenSetWire {
    Empty,
    Sparse(Vec<u32>),
    AllByteTokensAtLeastTwo,
}

#[derive(Serialize, Deserialize)]
enum BoundaryCandidateSummaryWire {
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

fn summary_unavailable_code(reason: &crate::runtime::SummaryUnavailable) -> u8 {
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

fn summary_unavailable_from_code(code: u8) -> Result<crate::runtime::SummaryUnavailable, String> {
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

fn summary_precision_code(precision: crate::runtime::SummaryPrecision) -> u8 {
    match precision {
        crate::runtime::SummaryPrecision::RegularUpperBound => 0,
        crate::runtime::SummaryPrecision::ContextRefinedUpperBound => 1,
        crate::runtime::SummaryPrecision::BudgetWidenedUpperBound => 2,
    }
}

fn summary_precision_from_code(code: u8) -> Result<crate::runtime::SummaryPrecision, String> {
    Ok(match code {
        0 => crate::runtime::SummaryPrecision::RegularUpperBound,
        1 => crate::runtime::SummaryPrecision::ContextRefinedUpperBound,
        2 => crate::runtime::SummaryPrecision::BudgetWidenedUpperBound,
        other => return Err(format!("invalid boundary summary precision code {other}")),
    })
}

fn boundary_candidate_summary_wire(constraint: &Constraint) -> BoundaryCandidateSummaryWire {
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

fn legacy_boundary_candidate_summary_wire() -> BoundaryCandidateSummaryWire {
    BoundaryCandidateSummaryWire::Unknown {
        reason: summary_unavailable_code(&crate::runtime::SummaryUnavailable::LegacyArtifact),
    }
}

fn restore_boundary_candidate_summary(
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
struct ConstraintCompositionLinkMetadataRef<'a> {
    composition_reset_tokens_by_terminal: &'a [Vec<u32>],
    unbound_grammar_placeholders: &'a BTreeMap<String, TerminalID>,
    composition_grammar_summary:
        &'a Option<crate::runtime::artifact::CompositionGrammarSummary>,
    boundary_trigger: BoundaryTriggerWireRef<'a>,
    boundary_candidate_summary: BoundaryCandidateSummaryWire,
}

#[derive(Serialize, Deserialize)]
struct ConstraintCompositionLinkMetadata {
    composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
    unbound_grammar_placeholders: BTreeMap<String, TerminalID>,
    composition_grammar_summary: Option<crate::runtime::artifact::CompositionGrammarSummary>,
    boundary_trigger: BoundaryTriggerWire,
    boundary_candidate_summary: BoundaryCandidateSummaryWire,
}

#[derive(Serialize, Deserialize)]
struct PreviousConstraintCompositionLinkMetadata {
    composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
    unbound_grammar_placeholders: BTreeMap<String, TerminalID>,
    composition_grammar_summary: Option<crate::runtime::artifact::CompositionGrammarSummary>,
    boundary_trigger: BoundaryTriggerWire,
}

#[derive(Serialize, Deserialize)]
struct PreviousPreviousConstraintCompositionLinkMetadata {
    composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
    unbound_grammar_placeholders: BTreeMap<String, TerminalID>,
    composition_grammar_summary: Option<crate::runtime::artifact::CompositionGrammarSummary>,
}

#[derive(Serialize)]
struct ConstraintCompositionCacheMetadataRef<'a> {
    composition_parser_templates_by_terminal:
        &'a [Option<crate::automata::unweighted_u32::dfa::DFA>],
    composition_parser_characterizations_by_terminal:
        &'a [Option<crate::compiler::stages::templates::characterize::TerminalCharacterization>],
}

#[derive(Serialize, Deserialize)]
struct ConstraintCompositionCacheMetadata {
    composition_parser_templates_by_terminal:
        Vec<Option<crate::automata::unweighted_u32::dfa::DFA>>,
    composition_parser_characterizations_by_terminal:
        Vec<Option<crate::compiler::stages::templates::characterize::TerminalCharacterization>>,
}

#[derive(Serialize, Deserialize)]
struct PreviousConstraintCompositionMetadata {
    composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
    composition_parser_templates_by_terminal:
        Vec<Option<crate::automata::unweighted_u32::dfa::DFA>>,
    composition_parser_characterizations_by_terminal:
        Vec<Option<crate::compiler::stages::templates::characterize::TerminalCharacterization>>,
    composition_grammar_summary: Option<crate::runtime::artifact::CompositionGrammarSummary>,
}

/// CMP2/CMZ2 and CMS3 predate reusable boundary-trigger metadata but already
/// carried named unresolved grammar slots.
#[derive(Serialize, Deserialize)]
struct PreTriggerConstraintCompositionMetadata {
    composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
    unbound_grammar_placeholders: BTreeMap<String, TerminalID>,
    composition_parser_templates_by_terminal:
        Vec<Option<crate::automata::unweighted_u32::dfa::DFA>>,
    composition_parser_characterizations_by_terminal:
        Vec<Option<crate::compiler::stages::templates::characterize::TerminalCharacterization>>,
    composition_grammar_summary: Option<crate::runtime::artifact::CompositionGrammarSummary>,
}

#[derive(Deserialize)]
struct ConstraintCompositionMetadata {
    composition_reset_tokens_by_terminal: Vec<Vec<u32>>,
    unbound_grammar_placeholders: BTreeMap<String, TerminalID>,
    composition_parser_templates_by_terminal:
        Vec<Option<crate::automata::unweighted_u32::dfa::DFA>>,
    composition_parser_characterizations_by_terminal:
        Vec<Option<crate::compiler::stages::templates::characterize::TerminalCharacterization>>,
    composition_grammar_summary: Option<crate::runtime::artifact::CompositionGrammarSummary>,
    boundary_trigger: BoundaryTriggerWire,
    boundary_candidate_summary: BoundaryCandidateSummaryWire,
}

struct CompositionMetadataSplitParts<'a> {
    link_raw_len: usize,
    link_wire: &'a [u8],
    link_compressed: bool,
    cache_raw_len: usize,
    cache_wire: &'a [u8],
    cache_compressed: bool,
}

fn encode_composition_metadata_part(raw: Vec<u8>) -> (usize, Vec<u8>, bool) {
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

fn assemble_composition_metadata_split(
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

fn decode_composition_metadata_part<'a>(
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

fn split_composition_metadata_parts(
    input: &[u8],
) -> Result<CompositionMetadataSplitParts<'_>, String> {
    if input.len() < COMPOSITION_METADATA_SPLIT_HEADER_LEN
        || !(input.starts_with(&COMPOSITION_METADATA_SPLIT_MAGIC)
            || input.starts_with(&PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC)
            || input.starts_with(&PREVIOUS_PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC))
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

fn encode_composition_metadata(constraint: &Constraint) -> Vec<u8> {
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

fn encode_composition_metadata_for_save(constraint: &Constraint) -> Vec<u8> {
    // Composition is explicitly unsupported by this standalone backend. Do
    // not retain dead compiler/LR caches in an otherwise table-free artifact.
    if constraint.has_template_parser() { return Vec::new(); }
    let bytes=encode_composition_metadata_base_for_save(constraint);
    if let Some(wire)=crate::compiler::boundary_precomputed_completion::saved_wire(constraint) {
        crate::compiler::boundary_precomputed_completion::wrap_envelope(bytes,wire)
            .expect("certified component index fits its bounded envelope")
    } else {
        crate::compiler::boundary_precomputed_completion::split_envelope(&bytes)
            .expect("retained composition envelope validated at load").0.to_vec()
    }
}
fn encode_composition_metadata_base_for_save(constraint: &Constraint) -> Vec<u8> {
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

    let (input, _) = crate::compiler::boundary_precomputed_completion::split_envelope(blob.as_slice())
        .expect("retained composition envelope validated at load");
    if input.starts_with(&COMPOSITION_METADATA_SPLIT_MAGIC)
        || input.starts_with(&PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC)
        || input.starts_with(&PREVIOUS_PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC)
    {
        let parts = split_composition_metadata_parts(input)
            .expect("loaded deferred composition metadata must remain structurally valid");
        return assemble_composition_metadata_split(
            link_raw_len,
            &link_wire,
            link_compressed,
            parts.cache_raw_len,
            parts.cache_wire,
            parts.cache_compressed,
        );
    }

    // Pre-split artifacts cannot splice the cache section directly. Decode the
    // old combined metadata only when a loaded component's link metadata was
    // actually modified, then re-emit it in the current split format.
    let old = decode_composition_metadata(input)
        .expect("loaded deferred composition metadata must remain decodable");
    let cache_raw = bincode::serialize(&ConstraintCompositionCacheMetadataRef {
        composition_parser_templates_by_terminal: &old.composition_parser_templates_by_terminal,
        composition_parser_characterizations_by_terminal:
            &old.composition_parser_characterizations_by_terminal,
    })
    .expect("composition cache metadata serialization should succeed");
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

fn validate_composition_metadata_wire(input: &[u8]) -> Result<(), String> {
    let (input, _) = crate::compiler::boundary_precomputed_completion::split_envelope(input)?;
    if input.is_empty() {
        return Ok(());
    }
    if input.starts_with(&COMPOSITION_METADATA_SPLIT_MAGIC)
        || input.starts_with(&PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC)
        || input.starts_with(&PREVIOUS_PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC)
    {
        split_composition_metadata_parts(input)?;
        return Ok(());
    }
    if input.len() < COMPOSITION_METADATA_HEADER_LEN {
        return Err("truncated composition metadata section".to_owned());
    }
    let raw_len = usize::try_from(u64::from_le_bytes(
        input[4..12]
            .try_into()
            .expect("composition metadata raw length has fixed width"),
    ))
    .map_err(|_| "composition metadata raw length does not fit platform".to_owned())?;
    if input.starts_with(&COMPOSITION_METADATA_RAW_MAGIC)
        || input.starts_with(&PREVIOUS_COMPOSITION_METADATA_RAW_MAGIC)
    {
        if input.len() != COMPOSITION_METADATA_HEADER_LEN + raw_len {
            return Err("invalid raw composition metadata section length".to_owned());
        }
        Ok(())
    } else if input.starts_with(&COMPOSITION_METADATA_ZSTD_MAGIC)
        || input.starts_with(&PREVIOUS_COMPOSITION_METADATA_ZSTD_MAGIC)
    {
        if raw_len == 0 || input.len() == COMPOSITION_METADATA_HEADER_LEN {
            return Err("invalid compressed composition metadata section".to_owned());
        }
        Ok(())
    } else {
        Err("invalid composition metadata section magic".to_owned())
    }
}

fn decode_composition_link_metadata(
    input: &[u8],
) -> Result<ConstraintCompositionLinkMetadata, String> {
    let (input, _) = crate::compiler::boundary_precomputed_completion::split_envelope(input)?;
    validate_composition_metadata_wire(input)?;
    if input.is_empty() {
        return Ok(ConstraintCompositionLinkMetadata {
            composition_reset_tokens_by_terminal: Vec::new(),
            unbound_grammar_placeholders: BTreeMap::new(),
            composition_grammar_summary: None,
            boundary_trigger: BoundaryTriggerWire::None,
            boundary_candidate_summary: legacy_boundary_candidate_summary_wire(),
        });
    }
    if input.starts_with(&COMPOSITION_METADATA_SPLIT_MAGIC)
        || input.starts_with(&PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC)
        || input.starts_with(&PREVIOUS_PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC)
    {
        let parts = split_composition_metadata_parts(input)?;
        let raw = decode_composition_metadata_part(
            parts.link_wire,
            parts.link_raw_len,
            parts.link_compressed,
        )?;
        if input.starts_with(&COMPOSITION_METADATA_SPLIT_MAGIC) {
            return bincode::deserialize(raw.as_ref()).map_err(|err| err.to_string());
        }
        if input.starts_with(&PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC) {
            let old: PreviousConstraintCompositionLinkMetadata =
                bincode::deserialize(raw.as_ref()).map_err(|err| err.to_string())?;
            return Ok(ConstraintCompositionLinkMetadata {
                composition_reset_tokens_by_terminal: old.composition_reset_tokens_by_terminal,
                unbound_grammar_placeholders: old.unbound_grammar_placeholders,
                composition_grammar_summary: old.composition_grammar_summary,
                boundary_trigger: old.boundary_trigger,
                boundary_candidate_summary: legacy_boundary_candidate_summary_wire(),
            });
        }
        let old: PreviousPreviousConstraintCompositionLinkMetadata =
            bincode::deserialize(raw.as_ref()).map_err(|err| err.to_string())?;
        return Ok(ConstraintCompositionLinkMetadata {
            composition_reset_tokens_by_terminal: old.composition_reset_tokens_by_terminal,
            unbound_grammar_placeholders: old.unbound_grammar_placeholders,
            composition_grammar_summary: old.composition_grammar_summary,
            boundary_trigger: BoundaryTriggerWire::None,
            boundary_candidate_summary: legacy_boundary_candidate_summary_wire(),
        });
    }
    // CMP1/CMP2 predate the split and can only be decoded as one object.
    let metadata = decode_composition_metadata(input)?;
    Ok(ConstraintCompositionLinkMetadata {
        composition_reset_tokens_by_terminal: metadata.composition_reset_tokens_by_terminal,
        unbound_grammar_placeholders: metadata.unbound_grammar_placeholders,
        composition_grammar_summary: metadata.composition_grammar_summary,
        boundary_trigger: metadata.boundary_trigger,
        boundary_candidate_summary: metadata.boundary_candidate_summary,
    })
}

fn decode_composition_metadata(input: &[u8]) -> Result<ConstraintCompositionMetadata, String> {
    let (input, _) = crate::compiler::boundary_precomputed_completion::split_envelope(input)?;
    validate_composition_metadata_wire(input)?;
    if input.is_empty() {
        return Ok(ConstraintCompositionMetadata {
            composition_reset_tokens_by_terminal: Vec::new(),
            unbound_grammar_placeholders: BTreeMap::new(),
            composition_parser_templates_by_terminal: Vec::new(),
            composition_parser_characterizations_by_terminal: Vec::new(),
            composition_grammar_summary: None,
            boundary_trigger: BoundaryTriggerWire::None,
            boundary_candidate_summary: legacy_boundary_candidate_summary_wire(),
        });
    }
    if input.starts_with(&COMPOSITION_METADATA_SPLIT_MAGIC)
        || input.starts_with(&PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC)
        || input.starts_with(&PREVIOUS_PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC)
    {
        let parts = split_composition_metadata_parts(input)?;
        let link_raw = decode_composition_metadata_part(
            parts.link_wire,
            parts.link_raw_len,
            parts.link_compressed,
        )?;
        let cache_raw = decode_composition_metadata_part(
            parts.cache_wire,
            parts.cache_raw_len,
            parts.cache_compressed,
        )?;
        let link = if input.starts_with(&COMPOSITION_METADATA_SPLIT_MAGIC) {
            bincode::deserialize::<ConstraintCompositionLinkMetadata>(link_raw.as_ref())
                .map_err(|err| err.to_string())?
        } else if input.starts_with(&PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC) {
            let old: PreviousConstraintCompositionLinkMetadata =
                bincode::deserialize(link_raw.as_ref()).map_err(|err| err.to_string())?;
            ConstraintCompositionLinkMetadata {
                composition_reset_tokens_by_terminal: old.composition_reset_tokens_by_terminal,
                unbound_grammar_placeholders: old.unbound_grammar_placeholders,
                composition_grammar_summary: old.composition_grammar_summary,
                boundary_trigger: old.boundary_trigger,
                boundary_candidate_summary: legacy_boundary_candidate_summary_wire(),
            }
        } else {
            let old: PreviousPreviousConstraintCompositionLinkMetadata =
                bincode::deserialize(link_raw.as_ref()).map_err(|err| err.to_string())?;
            ConstraintCompositionLinkMetadata {
                composition_reset_tokens_by_terminal: old.composition_reset_tokens_by_terminal,
                unbound_grammar_placeholders: old.unbound_grammar_placeholders,
                composition_grammar_summary: old.composition_grammar_summary,
                boundary_trigger: BoundaryTriggerWire::None,
                boundary_candidate_summary: legacy_boundary_candidate_summary_wire(),
            }
        };
        let cache: ConstraintCompositionCacheMetadata =
            bincode::deserialize(cache_raw.as_ref()).map_err(|err| err.to_string())?;
        return Ok(ConstraintCompositionMetadata {
            composition_reset_tokens_by_terminal: link.composition_reset_tokens_by_terminal,
            unbound_grammar_placeholders: link.unbound_grammar_placeholders,
            composition_parser_templates_by_terminal:
                cache.composition_parser_templates_by_terminal,
            composition_parser_characterizations_by_terminal:
                cache.composition_parser_characterizations_by_terminal,
            composition_grammar_summary: link.composition_grammar_summary,
            boundary_trigger: link.boundary_trigger,
            boundary_candidate_summary: link.boundary_candidate_summary,
        });
    }
    let raw_len = usize::try_from(u64::from_le_bytes(input[4..12].try_into().unwrap()))
        .map_err(|_| "composition metadata raw length does not fit platform".to_owned())?;
    let body = &input[COMPOSITION_METADATA_HEADER_LEN..];
    let previous = input.starts_with(&PREVIOUS_COMPOSITION_METADATA_RAW_MAGIC)
        || input.starts_with(&PREVIOUS_COMPOSITION_METADATA_ZSTD_MAGIC);
    let raw = if input.starts_with(&COMPOSITION_METADATA_RAW_MAGIC)
        || input.starts_with(&PREVIOUS_COMPOSITION_METADATA_RAW_MAGIC)
    {
        Cow::Borrowed(body)
    } else {
        let raw = zstd::bulk::decompress(body, raw_len).map_err(|err| err.to_string())?;
        if raw.len() != raw_len {
            return Err("invalid decompressed composition metadata length".to_owned());
        }
        Cow::Owned(raw)
    };
    if previous {
        let old: PreviousConstraintCompositionMetadata =
            bincode::deserialize(raw.as_ref()).map_err(|err| err.to_string())?;
        Ok(ConstraintCompositionMetadata {
            composition_reset_tokens_by_terminal: old.composition_reset_tokens_by_terminal,
            unbound_grammar_placeholders: BTreeMap::new(),
            composition_parser_templates_by_terminal: old.composition_parser_templates_by_terminal,
            composition_parser_characterizations_by_terminal:
                old.composition_parser_characterizations_by_terminal,
            composition_grammar_summary: old.composition_grammar_summary,
            boundary_trigger: BoundaryTriggerWire::None,
            boundary_candidate_summary: legacy_boundary_candidate_summary_wire(),
        })
    } else {
        let old: PreTriggerConstraintCompositionMetadata =
            bincode::deserialize(raw.as_ref()).map_err(|err| err.to_string())?;
        Ok(ConstraintCompositionMetadata {
            composition_reset_tokens_by_terminal: old.composition_reset_tokens_by_terminal,
            unbound_grammar_placeholders: old.unbound_grammar_placeholders,
            composition_parser_templates_by_terminal: old.composition_parser_templates_by_terminal,
            composition_parser_characterizations_by_terminal:
                old.composition_parser_characterizations_by_terminal,
            composition_grammar_summary: old.composition_grammar_summary,
            boundary_trigger: BoundaryTriggerWire::None,
            boundary_candidate_summary: legacy_boundary_candidate_summary_wire(),
        })
    }
}

struct DecodedConstraintCore {
    constraint: Constraint,
    ignore_expr: Option<Expr>,
    terminal_exprs: Option<Vec<Expr>>,
    terminal_exprs_blob: Option<crate::runtime::artifact::DeferredTerminalExprBytes>,
    parser_state_domain_labels: Vec<i32>,
    internal_token_buf_masks: Vec<InternalTokenBufMasks>,
}

#[derive(Serialize)]
struct TokenMaskCacheTailRef<'a> {
    guarded_shift_index: &'a [rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >],
    seed_terminal_dense: &'a crate::runtime::artifact::SeedTerminalDenseMasks,
    seed_universe_dense: &'a [u64],
    word_group_sparse_masks: &'a [InternalTokenBufMasks],
    word_group_sparse_prefix_entries: &'a [usize],
    quad_group_sparse_masks: &'a [InternalTokenBufMasks],
    quad_group_dense_masks: &'a [Option<Box<[u32]>>],
    byte_group_sparse_masks: &'a [InternalTokenBufMasks],
    byte_group_dense_masks: &'a [Option<Box<[u32]>>],
    word_group_sparse_total_entries: usize,
    word_group_sparse_max_entries: usize,
    all_tokens_buf_mask: &'a [u32],
    total_internal_buf_cost: usize,
    heavy_token_indices: &'a [usize],
    heavy_total_cost: usize,
    light_avg_cost_x256: usize,
    internal_token_buf_op_costs: &'a [usize],
    word_group_buf_op_costs: &'a [usize],
}

#[derive(Deserialize)]
struct TokenMaskCacheTail {
    guarded_shift_index: Vec<rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >>,
    seed_terminal_dense: crate::runtime::artifact::SeedTerminalDenseMasks,
    seed_universe_dense: Vec<u64>,
    word_group_sparse_masks: Vec<InternalTokenBufMasks>,
    word_group_sparse_prefix_entries: Vec<usize>,
    quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
    word_group_sparse_total_entries: usize,
    word_group_sparse_max_entries: usize,
    all_tokens_buf_mask: Box<[u32]>,
    total_internal_buf_cost: usize,
    heavy_token_indices: Vec<usize>,
    heavy_total_cost: usize,
    light_avg_cost_x256: usize,
    internal_token_buf_op_costs: Vec<usize>,
    word_group_buf_op_costs: Vec<usize>,
}

#[derive(Serialize)]
struct TokenMaskCacheIrregularRef<'a> {
    guarded_shift_index: &'a [rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >],
    seed_terminal_dense: &'a crate::runtime::artifact::SeedTerminalDenseMasks,
    seed_universe_dense: &'a [u64],
    quad_group_sparse_masks: &'a [InternalTokenBufMasks],
    quad_group_dense_masks: &'a [Option<Box<[u32]>>],
    byte_group_sparse_masks: &'a [InternalTokenBufMasks],
    byte_group_dense_masks: &'a [Option<Box<[u32]>>],
}

#[derive(Deserialize)]
struct TokenMaskCacheIrregular {
    guarded_shift_index: Vec<rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >>,
    seed_terminal_dense: crate::runtime::artifact::SeedTerminalDenseMasks,
    seed_universe_dense: Vec<u64>,
    quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
}

#[derive(Serialize, Deserialize)]
struct SeedTerminalDenseCompact {
    masks: Vec<crate::runtime::artifact::DenseWords>,
    entries: Vec<(u32, crate::grammar::flat::TerminalID, u32)>,
}

impl SeedTerminalDenseCompact {
    fn from_map(map: &crate::runtime::artifact::SeedTerminalDenseMasks) -> Self {
        let mut ordered = map.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|&(&key, _)| key);
        let mut mask_ids = rustc_hash::FxHashMap::<
            crate::runtime::artifact::DenseWords,
            u32,
        >::default();
        let mut masks = Vec::new();
        let mut entries = Vec::with_capacity(ordered.len());
        for (&(state, terminal), mask) in ordered {
            let mask_id = if let Some(&mask_id) = mask_ids.get(mask) {
                mask_id
            } else {
                let mask_id = masks.len() as u32;
                let shared = std::sync::Arc::clone(mask);
                mask_ids.insert(std::sync::Arc::clone(&shared), mask_id);
                masks.push(shared);
                mask_id
            };
            entries.push((state, terminal, mask_id));
        }
        Self { masks, entries }
    }

    fn into_map(self) -> Result<crate::runtime::artifact::SeedTerminalDenseMasks, String> {
        let mut result = crate::runtime::artifact::SeedTerminalDenseMasks::default();
        result.reserve(self.entries.len());
        for (state, terminal, mask_id) in self.entries {
            let mask = self
                .masks
                .get(mask_id as usize)
                .ok_or_else(|| "token-mask seed mask id is out of range".to_owned())?;
            if result
                .insert((state, terminal), std::sync::Arc::clone(mask))
                .is_some()
            {
                return Err("duplicate token-mask seed key".to_owned());
            }
        }
        Ok(result)
    }
}

#[derive(Serialize)]
struct TokenMaskCacheIrregularV5Ref<'a> {
    guarded_shift_index: &'a [rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >],
    seed_terminal_dense: &'a SeedTerminalDenseCompact,
    seed_universe_dense: &'a [u64],
    quad_group_sparse_masks: &'a [InternalTokenBufMasks],
    quad_group_dense_masks: &'a [Option<Box<[u32]>>],
    byte_group_sparse_masks: &'a [InternalTokenBufMasks],
    byte_group_dense_masks: &'a [Option<Box<[u32]>>],
}

#[derive(Deserialize)]
struct TokenMaskCacheIrregularV5 {
    guarded_shift_index: Vec<rustc_hash::FxHashMap<
        crate::grammar::flat::TerminalID,
        crate::compiler::glr::table::GuardedShiftCellIndex,
    >>,
    seed_terminal_dense: SeedTerminalDenseCompact,
    seed_universe_dense: Vec<u64>,
    quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    quad_group_dense_masks: Vec<Option<Box<[u32]>>>,
    byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    byte_group_dense_masks: Vec<Option<Box<[u32]>>>,
}

impl TokenMaskCacheIrregularV5 {
    fn into_irregular(self) -> Result<TokenMaskCacheIrregular, String> {
        Ok(TokenMaskCacheIrregular {
            guarded_shift_index: self.guarded_shift_index,
            seed_terminal_dense: self.seed_terminal_dense.into_map()?,
            seed_universe_dense: self.seed_universe_dense,
            quad_group_sparse_masks: self.quad_group_sparse_masks,
            quad_group_dense_masks: self.quad_group_dense_masks,
            byte_group_sparse_masks: self.byte_group_sparse_masks,
            byte_group_dense_masks: self.byte_group_dense_masks,
        })
    }
}

enum TokenMaskCacheArtifact {
    Full {
        tail: TokenMaskCacheTail,
        word_group_prefix_buf_masks: DenseBufMaskRows,
    },
    Fast {
        irregular: TokenMaskCacheIrregular,
        word_group_sparse_masks: Vec<InternalTokenBufMasks>,
        word_group_prefix_buf_masks: DenseBufMaskRows,
    },
    WordSparse(Vec<InternalTokenBufMasks>),
}

fn encode_word_sparse_token_mask_cache(constraint: &Constraint) -> Vec<u8> {
    const MAGIC: &[u8; 4] = b"TWS1";
    const MAX_BYTES: usize = 512 * 1024;
    let expected_groups = constraint.internal_token_count().div_ceil(64);
    if constraint.word_group_sparse_masks.len() != expected_groups {
        return Vec::new();
    }
    let entry_count = constraint
        .word_group_sparse_masks
        .iter()
        .map(Vec::len)
        .sum::<usize>();
    let encoded_len = 12usize
        .saturating_add((expected_groups + 1).saturating_mul(4))
        .saturating_add(entry_count.saturating_mul(6));
    if encoded_len > MAX_BYTES {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(encoded_len);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(expected_groups as u32).to_le_bytes());
    out.extend_from_slice(&(entry_count as u32).to_le_bytes());
    let mut end = 0u32;
    out.extend_from_slice(&end.to_le_bytes());
    for group in &constraint.word_group_sparse_masks {
        end = end.saturating_add(group.len() as u32);
        out.extend_from_slice(&end.to_le_bytes());
    }
    for group in &constraint.word_group_sparse_masks {
        for &(word, bits) in group {
            out.extend_from_slice(&word.to_le_bytes());
            out.extend_from_slice(&bits.to_le_bytes());
        }
    }
    debug_assert_eq!(out.len(), encoded_len);
    out
}

fn decode_word_sparse_token_mask_cache(input: &[u8]) -> Result<Vec<InternalTokenBufMasks>, String> {
    const HEADER_LEN: usize = 12;
    if input.len() < HEADER_LEN || !input.starts_with(b"TWS1") {
        return Err("invalid sparse word-group cache header".to_owned());
    }
    let group_count = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
    let entry_count = u32::from_le_bytes(input[8..12].try_into().unwrap()) as usize;
    let offsets_bytes = (group_count + 1)
        .checked_mul(4)
        .ok_or_else(|| "sparse word-group cache offsets overflow".to_owned())?;
    let entries_bytes = entry_count
        .checked_mul(6)
        .ok_or_else(|| "sparse word-group cache entries overflow".to_owned())?;
    let expected = HEADER_LEN
        .checked_add(offsets_bytes)
        .and_then(|n| n.checked_add(entries_bytes))
        .ok_or_else(|| "sparse word-group cache length overflow".to_owned())?;
    if input.len() != expected {
        return Err("invalid sparse word-group cache length".to_owned());
    }
    let offsets_body = &input[HEADER_LEN..HEADER_LEN + offsets_bytes];
    let mut offsets = Vec::<u32>::with_capacity(group_count + 1);
    for bytes in offsets_body.chunks_exact(4) {
        offsets.push(u32::from_le_bytes(bytes.try_into().unwrap()));
    }
    if offsets.first().copied() != Some(0)
        || offsets.last().copied() != Some(entry_count as u32)
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err("invalid sparse word-group cache offsets".to_owned());
    }
    let entries = &input[HEADER_LEN + offsets_bytes..];
    let mut groups = Vec::with_capacity(group_count);
    for group in 0..group_count {
        let start = offsets[group] as usize;
        let end = offsets[group + 1] as usize;
        let mut decoded = Vec::with_capacity(end - start);
        for entry in start..end {
            let pos = entry * 6;
            decoded.push((
                u16::from_le_bytes(entries[pos..pos + 2].try_into().unwrap()),
                u32::from_le_bytes(entries[pos + 2..pos + 6].try_into().unwrap()),
            ));
        }
        groups.push(decoded);
    }
    Ok(groups)
}

#[inline]
fn append_cache_u32s(out: &mut Vec<u8>, values: &[u32]) {
    if cfg!(target_endian = "little") {
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
        };
        out.extend_from_slice(bytes);
    } else {
        for &value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

fn decode_cache_u32_rows(
    input: &[u8],
    pos: &mut usize,
    rows: usize,
    row_len: usize,
) -> Result<DenseBufMaskRows, String> {
    if !DenseBufMaskRows::prefer_flat(rows, row_len) {
        let row_bytes = row_len
            .checked_mul(4)
            .ok_or_else(|| "token-mask prefix row byte length overflow".to_owned())?;
        let mut decoded_rows = Vec::with_capacity(rows);
        for _ in 0..rows {
            let end = pos
                .checked_add(row_bytes)
                .ok_or_else(|| "token-mask prefix row offset overflow".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated token-mask prefix row".to_owned())?;
            let mut row = Vec::<u32>::with_capacity(row_len);
            if cfg!(target_endian = "little") {
                unsafe {
                    row.set_len(row_len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        row.as_mut_ptr().cast::<u8>(),
                        row_bytes,
                    );
                }
            } else {
                row.extend(
                    bytes
                        .chunks_exact(4)
                        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())),
                );
            }
            *pos = end;
            decoded_rows.push(row.into_boxed_slice());
        }
        return DenseBufMaskRows::from_rows(decoded_rows);
    }
    let values = rows
        .checked_mul(row_len)
        .ok_or_else(|| "token-mask prefix dimensions overflow".to_owned())?;
    let byte_len = values
        .checked_mul(4)
        .ok_or_else(|| "token-mask prefix byte length overflow".to_owned())?;
    let end = pos
        .checked_add(byte_len)
        .ok_or_else(|| "token-mask prefix offset overflow".to_owned())?;
    let bytes = input
        .get(*pos..end)
        .ok_or_else(|| "truncated token-mask prefix".to_owned())?;
    let mut flat = Vec::<u32>::with_capacity(values);
    if cfg!(target_endian = "little") {
        unsafe {
            flat.set_len(values);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), flat.as_mut_ptr().cast::<u8>(), byte_len);
        }
    } else {
        flat.extend(
            bytes
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap())),
        );
    }
    *pos = end;
    DenseBufMaskRows::from_flat(flat.into_boxed_slice(), rows, row_len)
}

fn encode_token_mask_cache(constraint: &Constraint) -> Vec<u8> {
    const HEADER_LEN: usize = 24;
    // JS's exact prefix matrix is ~1.3 MiB. Keeping the old 1 MiB cutoff made
    // every load throw that useful runtime-native cache away and rebuild it.
    const MAX_PREFIX_BYTES: usize = 2 * 1024 * 1024;
    const MAX_CACHE_BYTES: usize = 4 * 1024 * 1024;
    if !constraint.token_mask_caches_ready() {
        return Vec::new();
    }
    let mask_words = constraint.body_mask_len();
    let prefix_rows = constraint.word_group_prefix_buf_masks.len();
    let word_groups = constraint.word_group_sparse_masks.len();
    let word_entries = constraint
        .word_group_sparse_masks
        .iter()
        .map(Vec::len)
        .sum::<usize>();
    let prefix_bytes = prefix_rows
        .saturating_mul(mask_words)
        .saturating_mul(std::mem::size_of::<u32>());
    if prefix_bytes > MAX_PREFIX_BYTES {
        return encode_word_sparse_token_mask_cache(constraint);
    }
    if prefix_rows != word_groups.saturating_add(1)
        || constraint
            .word_group_prefix_buf_masks
            .iter()
            .any(|row| row.len() != mask_words)
    {
        return Vec::new();
    }
    let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
    let tail_started = profile.then(std::time::Instant::now);
    let mut tail = Vec::with_capacity(32 * 1024);
    // Deduplicating tiny seed maps costs more bookkeeping than it saves. Large
    // seed tables, however, commonly repeat the same dense token mask across
    // hundreds of tokenizer-state/terminal keys; TMC5 pools those immutable
    // masks once and shares the Arc again after load.
    let compact_seed = constraint.seed_terminal_dense.len() >= 64;
    if compact_seed {
        let seed_terminal_dense = SeedTerminalDenseCompact::from_map(&constraint.seed_terminal_dense);
        bincode::serialize_into(
            &mut tail,
            &TokenMaskCacheIrregularV5Ref {
                guarded_shift_index: constraint.table.as_lr().map_or(&[], |table| table.guarded_shift_index.as_slice()),
                seed_terminal_dense: &seed_terminal_dense,
                seed_universe_dense: &constraint.seed_universe_dense,
                quad_group_sparse_masks: &constraint.quad_group_sparse_masks,
                quad_group_dense_masks: &constraint.quad_group_dense_masks,
                byte_group_sparse_masks: &constraint.byte_group_sparse_masks,
                byte_group_dense_masks: &constraint.byte_group_dense_masks,
            },
        )
        .expect("token-mask cache serialization should succeed");
    } else {
        bincode::serialize_into(
            &mut tail,
            &TokenMaskCacheIrregularRef {
                guarded_shift_index: constraint.table.as_lr().map_or(&[], |table| table.guarded_shift_index.as_slice()),
                seed_terminal_dense: &constraint.seed_terminal_dense,
                seed_universe_dense: &constraint.seed_universe_dense,
                quad_group_sparse_masks: &constraint.quad_group_sparse_masks,
                quad_group_dense_masks: &constraint.quad_group_dense_masks,
                byte_group_sparse_masks: &constraint.byte_group_sparse_masks,
                byte_group_dense_masks: &constraint.byte_group_dense_masks,
            },
        )
        .expect("token-mask cache serialization should succeed");
    }
    let tail_ms = tail_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let word_offsets_bytes = (word_groups + 1).saturating_mul(4);
    let word_entries_bytes = word_entries.saturating_mul(6);
    // TMC6/7 align the dense u32 matrix so an owned load can retain it directly
    // from the artifact rather than copying ~1.3 MiB merely for alignment.
    let prefix_unaligned_start = HEADER_LEN
        .saturating_add(tail.len())
        .saturating_add(word_offsets_bytes)
        .saturating_add(word_entries_bytes);
    let prefix_padding = (4 - (prefix_unaligned_start & 3)) & 3;
    let total_len = HEADER_LEN
        .saturating_add(tail.len())
        .saturating_add(word_offsets_bytes)
        .saturating_add(word_entries_bytes)
        .saturating_add(prefix_padding)
        .saturating_add(prefix_bytes);
    if total_len > MAX_CACHE_BYTES {
        return encode_word_sparse_token_mask_cache(constraint);
    }
    let prefix_started = profile.then(std::time::Instant::now);
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(if compact_seed { b"TMC7" } else { b"TMC6" });
    for value in [tail.len(), mask_words, word_groups, word_entries, prefix_rows] {
        out.extend_from_slice(
            &u32::try_from(value)
                .expect("token-mask cache dimension fits u32")
                .to_le_bytes(),
        );
    }
    out.extend_from_slice(&tail);
    let mut end = 0u32;
    out.extend_from_slice(&end.to_le_bytes());
    for group in &constraint.word_group_sparse_masks {
        end = end.saturating_add(group.len() as u32);
        out.extend_from_slice(&end.to_le_bytes());
    }
    for group in &constraint.word_group_sparse_masks {
        for &(word, bits) in group {
            out.extend_from_slice(&word.to_le_bytes());
            out.extend_from_slice(&bits.to_le_bytes());
        }
    }
    out.resize(out.len() + prefix_padding, 0);
    if let Some(flat) = constraint.word_group_prefix_buf_masks.as_contiguous() {
        append_cache_u32s(&mut out, flat);
    } else {
        for row in &constraint.word_group_prefix_buf_masks {
            append_cache_u32s(&mut out, row);
        }
    }
    debug_assert_eq!(out.len(), total_len);
    if let Some(started) = prefix_started {
        eprintln!(
            "[glrmask/profile][token_mask_cache_encode] tail_ms={tail_ms:.3} body_ms={:.3} tail_bytes={} word_sparse_bytes={} prefix_bytes={} total_bytes={}",
            started.elapsed().as_secs_f64() * 1000.0,
            tail.len(),
            word_offsets_bytes + word_entries_bytes,
            prefix_bytes,
            out.len(),
        );
    }
    out
}

fn decode_token_mask_cache(input: &[u8]) -> Result<TokenMaskCacheArtifact, String> {
    decode_token_mask_cache_impl(input, None)
}

fn decode_token_mask_cache_backed(
    input: &[u8],
    backing: std::sync::Arc<Vec<u8>>,
    section_start: usize,
) -> Result<TokenMaskCacheArtifact, String> {
    decode_token_mask_cache_impl(input, Some((backing, section_start)))
}

fn decode_token_mask_cache_impl(
    input: &[u8],
    backing: Option<(std::sync::Arc<Vec<u8>>, usize)>,
) -> Result<TokenMaskCacheArtifact, String> {
    // Current S20 artifacts may prefix TMC6/7 with up to three zero bytes so
    // the cache's aligned dense matrix also lands at a u32-aligned offset in
    // the whole artifact. Older S20 artifacts had no section-local prefix, so
    // continue to accept both layouts.
    let has_known_magic = |bytes: &[u8]| {
        bytes.starts_with(b"TWS1")
            || bytes.starts_with(b"TMC3")
            || bytes.starts_with(b"TMC4")
            || bytes.starts_with(b"TMC5")
            || bytes.starts_with(b"TMC6")
            || bytes.starts_with(b"TMC7")
    };
    let leading_padding = if has_known_magic(input) {
        0
    } else {
        (1..=3)
            .find(|&padding| {
                input
                    .get(..padding)
                    .is_some_and(|prefix| prefix.iter().all(|&byte| byte == 0))
                    && input.get(padding..).is_some_and(has_known_magic)
            })
            .unwrap_or(0)
    };
    let input = &input[leading_padding..];
    let backing = backing.map(|(backing, section_start)| {
        (
            backing,
            section_start
                .checked_add(leading_padding)
                .expect("token-mask cache section offset cannot overflow"),
        )
    });
    const MAGIC: &[u8; 4] = b"TMC3";
    const HEADER_LEN: usize = 16;
    if input.starts_with(b"TWS1") {
        return decode_word_sparse_token_mask_cache(input).map(TokenMaskCacheArtifact::WordSparse);
    }
    if input.starts_with(b"TMC7")
        || input.starts_with(b"TMC6")
        || input.starts_with(b"TMC5")
        || input.starts_with(b"TMC4")
    {
        let compact_seed = input.starts_with(b"TMC7") || input.starts_with(b"TMC5");
        let aligned_prefix = input.starts_with(b"TMC7") || input.starts_with(b"TMC6");
        const FAST_HEADER_LEN: usize = 24;
        if input.len() < FAST_HEADER_LEN {
            return Err("invalid fast token-mask cache header".to_owned());
        }
        let read = |offset: usize| {
            u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap()) as usize
        };
        let tail_len = read(4);
        let mask_words = read(8);
        let word_groups = read(12);
        let word_entries = read(16);
        let prefix_rows = read(20);
        if prefix_rows != word_groups.saturating_add(1) {
            return Err("fast token-mask prefix row count mismatch".to_owned());
        }
        let tail_end = FAST_HEADER_LEN
            .checked_add(tail_len)
            .ok_or_else(|| "fast token-mask cache tail overflow".to_owned())?;
        let offsets_bytes = (word_groups + 1)
            .checked_mul(4)
            .ok_or_else(|| "fast token-mask sparse offsets overflow".to_owned())?;
        let entries_bytes = word_entries
            .checked_mul(6)
            .ok_or_else(|| "fast token-mask sparse entries overflow".to_owned())?;
        let prefix_bytes = prefix_rows
            .checked_mul(mask_words)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| "fast token-mask prefix bytes overflow".to_owned())?;
        let prefix_unaligned_start = tail_end
            .checked_add(offsets_bytes)
            .and_then(|n| n.checked_add(entries_bytes))
            .ok_or_else(|| "fast token-mask cache prefix offset overflow".to_owned())?;
        let prefix_padding = if aligned_prefix {
            (4 - (prefix_unaligned_start & 3)) & 3
        } else {
            0
        };
        let prefix_start = prefix_unaligned_start
            .checked_add(prefix_padding)
            .ok_or_else(|| "fast token-mask cache prefix padding overflow".to_owned())?;
        let expected = prefix_start
            .checked_add(prefix_bytes)
            .ok_or_else(|| "fast token-mask cache length overflow".to_owned())?;
        if expected != input.len() {
            return Err("invalid fast token-mask cache length".to_owned());
        }
        let offsets_start = tail_end;
        let entries_start = offsets_start + offsets_bytes;
        if aligned_prefix
            && input
                .get(prefix_unaligned_start..prefix_start)
                .is_none_or(|padding| padding.iter().any(|&byte| byte != 0))
        {
            return Err("invalid fast token-mask prefix padding".to_owned());
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let tail_started = profile.then(std::time::Instant::now);
        let tail_bytes = input
            .get(FAST_HEADER_LEN..tail_end)
            .ok_or_else(|| "truncated fast token-mask cache tail".to_owned())?;
        let irregular = if compact_seed {
            bincode::deserialize::<TokenMaskCacheIrregularV5>(tail_bytes)
                .map_err(|err| err.to_string())?
                .into_irregular()?
        } else {
            bincode::deserialize::<TokenMaskCacheIrregular>(tail_bytes)
                .map_err(|err| err.to_string())?
        };
        let tail_ms = tail_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let body_started = profile.then(std::time::Instant::now);
        let mut offsets = Vec::<u32>::with_capacity(word_groups + 1);
        for bytes in input[offsets_start..entries_start].chunks_exact(4) {
            offsets.push(u32::from_le_bytes(bytes.try_into().unwrap()));
        }
        if offsets.first().copied() != Some(0)
            || offsets.last().copied() != Some(word_entries as u32)
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err("invalid fast token-mask sparse offsets".to_owned());
        }
        let entries = &input[entries_start..prefix_start];
        let decode_group = |group: usize| -> Result<InternalTokenBufMasks, String> {
            let start = offsets[group] as usize;
            let end = offsets[group + 1] as usize;
            let mut decoded = Vec::with_capacity(end - start);
            if cfg!(target_endian = "little") {
                // `expected == input.len()` above proves every 6-byte record is
                // present. Read the packed fields directly instead of doing
                // two independently bounds-checked slices + `try_into()` per
                // sparse entry. The wire is intentionally unaligned.
                let base = entries.as_ptr();
                for entry in start..end {
                    let ptr = unsafe { base.add(entry * 6) };
                    let word = unsafe { std::ptr::read_unaligned(ptr.cast::<u16>()) };
                    if word as usize >= mask_words {
                        return Err("fast token-mask sparse word out of range".to_owned());
                    }
                    let bits = unsafe { std::ptr::read_unaligned(ptr.add(2).cast::<u32>()) };
                    decoded.push((word, bits));
                }
            } else {
                for entry in start..end {
                    let pos = entry * 6;
                    let word = u16::from_le_bytes(entries[pos..pos + 2].try_into().unwrap());
                    if word as usize >= mask_words {
                        return Err("fast token-mask sparse word out of range".to_owned());
                    }
                    decoded.push((
                        word,
                        u32::from_le_bytes(entries[pos + 2..pos + 6].try_into().unwrap()),
                    ));
                }
            }
            Ok(decoded)
        };
        let word_group_sparse_masks = if word_groups >= 128 && rayon::current_num_threads() > 1 {
            (0..word_groups)
                .into_par_iter()
                .map(decode_group)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            (0..word_groups)
                .map(decode_group)
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut pos = prefix_start;
        let materialize_prefix = std::env::var_os("GLRMASK_MATERIALIZE_TMC_PREFIX").is_some();
        let word_group_prefix_buf_masks = if aligned_prefix && !materialize_prefix {
            if let Some((backing, section_start)) = backing.as_ref() {
                let absolute_start = section_start
                    .checked_add(prefix_start)
                    .ok_or_else(|| "fast token-mask backed prefix offset overflow".to_owned())?;
                match DenseBufMaskRows::from_backed(
                    std::sync::Arc::clone(backing),
                    absolute_start,
                    prefix_rows,
                    mask_words,
                ) {
                    Ok(rows) => {
                        pos = pos
                            .checked_add(prefix_bytes)
                            .ok_or_else(|| "fast token-mask backed prefix overflow".to_owned())?;
                        rows
                    }
                    Err(_) => decode_cache_u32_rows(input, &mut pos, prefix_rows, mask_words)?,
                }
            } else {
                decode_cache_u32_rows(input, &mut pos, prefix_rows, mask_words)?
            }
        } else {
            decode_cache_u32_rows(input, &mut pos, prefix_rows, mask_words)?
        };
        debug_assert_eq!(pos, input.len());
        if let Some(started) = body_started {
            eprintln!(
                "[glrmask/profile][token_mask_cache_decode] tail_ms={tail_ms:.3} body_ms={:.3} tail_bytes={} word_sparse_bytes={} prefix_bytes={}",
                started.elapsed().as_secs_f64() * 1000.0,
                tail_len,
                offsets_bytes + entries_bytes,
                prefix_bytes,
            );
        }
        return Ok(TokenMaskCacheArtifact::Fast {
            irregular,
            word_group_sparse_masks,
            word_group_prefix_buf_masks,
        });
    }
    if input.len() < HEADER_LEN || !input.starts_with(MAGIC) {
        return Err("invalid token-mask cache header".to_owned());
    }
    let read = |offset: usize| {
        u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap()) as usize
    };
    let tail_len = read(4);
    let mask_words = read(8);
    let prefix_rows = read(12);
    let tail_end = HEADER_LEN
        .checked_add(tail_len)
        .ok_or_else(|| "token-mask cache tail overflow".to_owned())?;
    let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
    let tail_started = profile.then(std::time::Instant::now);
    let tail = bincode::deserialize::<TokenMaskCacheTail>(
        input
            .get(HEADER_LEN..tail_end)
            .ok_or_else(|| "truncated token-mask cache tail".to_owned())?,
    )
    .map_err(|err| err.to_string())?;
    let tail_ms = tail_started
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let prefix_started = profile.then(std::time::Instant::now);
    let mut pos = tail_end;
    let word_group_prefix_buf_masks =
        decode_cache_u32_rows(input, &mut pos, prefix_rows, mask_words)?;
    if pos != input.len() {
        return Err("trailing bytes in token-mask cache".to_owned());
    }
    if let Some(started) = prefix_started {
        eprintln!(
            "[glrmask/profile][token_mask_cache_decode] tail_ms={tail_ms:.3} prefix_ms={:.3} tail_bytes={} prefix_bytes={}",
            started.elapsed().as_secs_f64() * 1000.0,
            tail_len,
            input.len().saturating_sub(tail_end),
        );
    }
    Ok(TokenMaskCacheArtifact::Full {
        tail,
        word_group_prefix_buf_masks,
    })
}

fn install_token_mask_cache(
    constraint: &mut Constraint,
    cache: TokenMaskCacheArtifact,
) -> Result<(), String> {
    match cache {
        TokenMaskCacheArtifact::WordSparse(groups) => {
            constraint.word_group_sparse_masks = groups;
            Ok(())
        }
        TokenMaskCacheArtifact::Fast {
            irregular,
            word_group_sparse_masks,
            word_group_prefix_buf_masks,
        } => {
            if constraint.has_template_parser() {
                if !irregular.guarded_shift_index.is_empty() {
                    return Err("table-free artifact contains an LR guarded-shift cache".into());
                }
            } else {
                constraint.table.guarded_shift_index = irregular.guarded_shift_index;
            }
            constraint.seed_terminal_dense = irregular.seed_terminal_dense;
            constraint.seed_universe_dense = irregular.seed_universe_dense.into();
            constraint.word_group_sparse_masks = word_group_sparse_masks;
            constraint.word_group_prefix_buf_masks = word_group_prefix_buf_masks;
            constraint.quad_group_sparse_masks = irregular.quad_group_sparse_masks;
            constraint.quad_group_dense_masks = irregular.quad_group_dense_masks;
            constraint.byte_group_sparse_masks = irregular.byte_group_sparse_masks;
            constraint.byte_group_dense_masks = irregular.byte_group_dense_masks;
            constraint.rebuild_heavy_and_sliding_token_mask_caches();
            constraint.rebuild_token_mask_cache_stats();
            if constraint.token_mask_caches_ready() {
                Ok(())
            } else {
                Err("fast token-mask cache section does not match constraint dimensions".to_owned())
            }
        }
        TokenMaskCacheArtifact::Full {
            tail: cache,
            word_group_prefix_buf_masks,
        } => {
            if constraint.has_template_parser() {
                if !cache.guarded_shift_index.is_empty() {
                    return Err("table-free artifact contains an LR guarded-shift cache".into());
                }
            } else {
                constraint.table.guarded_shift_index = cache.guarded_shift_index;
            }
            constraint.seed_terminal_dense = cache.seed_terminal_dense;
            constraint.seed_universe_dense = cache.seed_universe_dense.into();
            constraint.word_group_sparse_masks = cache.word_group_sparse_masks;
            constraint.word_group_prefix_buf_masks = word_group_prefix_buf_masks;
            constraint.word_group_sparse_prefix_entries = cache.word_group_sparse_prefix_entries;
            constraint.quad_group_sparse_masks = cache.quad_group_sparse_masks;
            constraint.quad_group_dense_masks = cache.quad_group_dense_masks;
            constraint.byte_group_sparse_masks = cache.byte_group_sparse_masks;
            constraint.byte_group_dense_masks = cache.byte_group_dense_masks;
            constraint.word_group_sparse_total_entries = cache.word_group_sparse_total_entries;
            constraint.word_group_sparse_max_entries = cache.word_group_sparse_max_entries;
            constraint.all_tokens_buf_mask = cache.all_tokens_buf_mask;
            constraint.total_internal_buf_cost = cache.total_internal_buf_cost;
            constraint.heavy_token_indices = cache.heavy_token_indices;
            constraint.heavy_total_cost = cache.heavy_total_cost;
            constraint.light_avg_cost_x256 = cache.light_avg_cost_x256;
            constraint.internal_token_buf_op_costs = cache.internal_token_buf_op_costs;
            constraint.word_group_buf_op_costs = cache.word_group_buf_op_costs;
            constraint.rebuild_heavy_and_sliding_token_mask_caches();
            let rebuilt_heavy_indices = constraint
                .heavy_token_dense_masks
                .iter()
                .enumerate()
                .filter_map(|(index, mask)| mask.is_some().then_some(index))
                .collect::<Vec<_>>();
            if rebuilt_heavy_indices != constraint.heavy_token_indices {
                return Err("token-mask cache heavy-token index mismatch".to_owned());
            }
            if constraint.token_mask_caches_ready() {
                Ok(())
            } else {
                Err("token-mask cache section does not match constraint dimensions".to_owned())
            }
        }
    }
}

fn encode_internal_token_buf_masks(constraint: &Constraint) -> Vec<u8> {
    const MAGIC: &[u8; 4] = b"IBM2";
    const ENTRY_BYTES: usize = std::mem::size_of::<PackedInternalTokenBufMask>();
    let flat_len = constraint.internal_token_buf_flat_len();
    let packed_ready = constraint.internal_token_buf_offsets.len()
        == constraint.internal_token_count().saturating_add(1)
        && constraint
            .internal_token_buf_offsets
            .last()
            .is_some_and(|&end| end as usize == flat_len);
    let group_count = if packed_ready {
        constraint.internal_token_buf_offsets.len().saturating_sub(1)
    } else {
        constraint.internal_token_buf_masks.len()
    };
    let entry_count = if packed_ready {
        flat_len
    } else {
        constraint
            .internal_token_buf_masks
            .iter()
            .map(Vec::len)
            .sum::<usize>()
    };
    let mut out = Vec::with_capacity(
        12usize
            .saturating_add((group_count + 1).saturating_mul(4))
            .saturating_add(entry_count.saturating_mul(ENTRY_BYTES)),
    );
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(group_count as u32).to_le_bytes());
    out.extend_from_slice(&(entry_count as u32).to_le_bytes());
    if packed_ready {
        for &offset in constraint.internal_token_buf_offsets.iter() {
            out.extend_from_slice(&offset.to_le_bytes());
        }
        if let Some(backed) = constraint.backed_internal_token_buf_flat.as_ref() {
            backed.append_wire_bytes(&mut out);
        } else if cfg!(target_endian = "little") {
            let byte_len = entry_count * ENTRY_BYTES;
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    constraint.internal_token_buf_flat.as_ptr().cast::<u8>(),
                    byte_len,
                )
            };
            out.extend_from_slice(bytes);
        } else {
            for entry in constraint.internal_token_buf_flat.iter() {
                out.extend_from_slice(&entry.word_idx.to_le_bytes());
                out.extend_from_slice(&0u16.to_le_bytes());
                out.extend_from_slice(&entry.mask.to_le_bytes());
            }
        }
    } else {
        let mut end = 0u32;
        out.extend_from_slice(&end.to_le_bytes());
        for mask in &constraint.internal_token_buf_masks {
            end = end.saturating_add(mask.len() as u32);
            out.extend_from_slice(&end.to_le_bytes());
        }
        for mask in &constraint.internal_token_buf_masks {
            for &(word, bits) in mask {
                out.extend_from_slice(&word.to_le_bytes());
                out.extend_from_slice(&0u16.to_le_bytes());
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
    }
    out
}

struct DecodedInternalTokenBufMasks {
    flat: Box<[PackedInternalTokenBufMask]>,
    backed: Option<BackedInternalTokenBufMasks>,
    offsets: Box<[u32]>,
}

enum DecodedOriginalTokenMap {
    Materialized(Vec<u32>),
    Packed(std::sync::Arc<
        crate::runtime::artifact::original_token_map_artifact_serde::PackedOriginalTokenMap,
    >),
}

fn decode_internal_token_buf_masks(
    input: &[u8],
    mut backing: Option<(std::sync::Arc<Vec<u8>>, usize)>,
) -> Result<DecodedInternalTokenBufMasks, String> {
    const LEGACY_MAGIC: &[u8; 4] = b"IBM1";
    const FIXED_MAGIC: &[u8; 4] = b"IBM2";
    let mut input = input;
    if !input.starts_with(FIXED_MAGIC) && !input.starts_with(LEGACY_MAGIC) {
        let leading_padding = (1..std::mem::align_of::<PackedInternalTokenBufMask>())
            .find(|&padding| {
                input.len() >= padding + FIXED_MAGIC.len()
                    && input[..padding].iter().all(|&byte| byte == 0)
                    && input[padding..].starts_with(FIXED_MAGIC)
            });
        if let Some(padding) = leading_padding {
            input = &input[padding..];
            if let Some((_, section_start)) = backing.as_mut() {
                *section_start = section_start
                    .checked_add(padding)
                    .ok_or_else(|| "internal-token buffer-mask backing offset overflow".to_owned())?;
            }
        }
    }
    let fixed = input.starts_with(FIXED_MAGIC);
    if input.len() < 12 || (!fixed && !input.starts_with(LEGACY_MAGIC)) {
        return Err("invalid internal-token buffer-mask section".to_owned());
    }
    let group_count = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
    let entry_count = u32::from_le_bytes(input[8..12].try_into().unwrap()) as usize;
    let offsets_bytes = (group_count + 1)
        .checked_mul(4)
        .ok_or_else(|| "internal-token buffer-mask offsets overflow".to_owned())?;
    let entry_width = if fixed {
        std::mem::size_of::<PackedInternalTokenBufMask>()
    } else {
        6
    };
    let entries_bytes = entry_count
        .checked_mul(entry_width)
        .ok_or_else(|| "internal-token buffer-mask entries overflow".to_owned())?;
    let expected = 12usize
        .checked_add(offsets_bytes)
        .and_then(|n| n.checked_add(entries_bytes))
        .ok_or_else(|| "internal-token buffer-mask section length overflow".to_owned())?;
    if expected != input.len() {
        return Err("invalid internal-token buffer-mask section length".to_owned());
    }
    let offsets_body = &input[12..12 + offsets_bytes];
    let mut offsets = Vec::<u32>::with_capacity(group_count + 1);
    if cfg!(target_endian = "little") {
        unsafe {
            offsets.set_len(group_count + 1);
            std::ptr::copy_nonoverlapping(
                offsets_body.as_ptr(),
                offsets.as_mut_ptr().cast::<u8>(),
                offsets_bytes,
            );
        }
    } else {
        offsets.extend(
            offsets_body
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap())),
        );
    }
    if offsets.first().copied() != Some(0)
        || offsets.last().copied().map(|end| end as usize) != Some(entry_count)
    {
        return Err("invalid internal-token buffer-mask offsets".to_owned());
    }
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("non-monotonic internal-token buffer-mask offsets".to_owned());
    }
    let entries_start = 12 + offsets_bytes;
    let entries = &input[entries_start..];
    let backed = if fixed {
        backing
            .map(|(backing, section_start)| {
                BackedInternalTokenBufMasks::new(
                    backing,
                    section_start + entries_start,
                    entry_count,
                )
            })
            .transpose()?
    } else {
        None
    };
    let mut flat = Vec::<PackedInternalTokenBufMask>::with_capacity(if backed.is_some() {
        0
    } else {
        entry_count
    });
    if backed.is_some() {
        // The retained artifact is the runtime storage; offsets remain owned
        // because they are tiny and hot to index.
    } else if fixed && cfg!(target_endian = "little") {
        unsafe {
            flat.set_len(entry_count);
            std::ptr::copy_nonoverlapping(
                entries.as_ptr(),
                flat.as_mut_ptr().cast::<u8>(),
                entries_bytes,
            );
        }
    } else {
        // The section length was validated above, so every record is present.
        // IBM1 has six-byte records; IBM2 is eight bytes with a two-byte pad.
        unsafe {
            flat.set_len(entry_count);
            let src = entries.as_ptr();
            let dst = flat.as_mut_ptr();
            for entry in 0..entry_count {
                let pos = entry * entry_width;
                let word = u16::from_le(std::ptr::read_unaligned(src.add(pos).cast::<u16>()));
                let bits_offset = if fixed { 4 } else { 2 };
                let bits = u32::from_le(std::ptr::read_unaligned(
                    src.add(pos + bits_offset).cast::<u32>(),
                ));
                std::ptr::write(
                    dst.add(entry),
                    PackedInternalTokenBufMask {
                        word_idx: word,
                        _pad: 0,
                        mask: bits,
                    },
                );
            }
        }
    }
    Ok(DecodedInternalTokenBufMasks {
        flat: flat.into_boxed_slice(),
        backed,
        offsets: offsets.into_boxed_slice(),
    })
}

#[derive(Serialize)]
struct ConstraintArtifactV15RuntimeRef<'a> {
    terminal_live_states: &'a [Vec<u32>],
}

#[derive(Deserialize)]
struct ConstraintArtifactV15Runtime {
    terminal_live_states: Vec<Vec<u32>>,
}

#[derive(Serialize)]
struct SegmentedBoundaryParserV20Ref<'a> {
    parser_dwa: Cow<'a, crate::automata::weighted_u32::dwa::DWA>,
    tokenizer_state_to_tsid: &'a [u32],
    internal_token_to_originals: &'a [Vec<u32>],
}

#[derive(Serialize)]
struct SegmentedRuntimeArtifactV20Ref<'a> {
    materialized_component_parser: Option<crate::automata::weighted_u32::dwa::DWA>,
    materialized_parser_state_domain_labels: Vec<i32>,
    boundary_parser: Option<SegmentedBoundaryParserV20Ref<'a>>,
    boundary_terminal_trie: Option<&'a crate::runtime::artifact::SegmentedBoundaryTerminalTrie>,
}

#[derive(Deserialize)]
struct SegmentedRuntimeArtifactV20 {
    materialized_component_parser: Option<crate::automata::weighted_u32::dwa::DWA>,
    materialized_parser_state_domain_labels: Vec<i32>,
    boundary_parser: Option<crate::runtime::artifact::SegmentedBoundaryParser>,
    boundary_terminal_trie: Option<crate::runtime::artifact::SegmentedBoundaryTerminalTrie>,
}

#[derive(Serialize)]
struct SegmentedParserComponentV22Ref<'a> {
    constraint_artifact: Vec<u8>,
    tokenizer_state_offset: u32,
    terminal_offset: u32,
    root_entry_terminals: crate::ds::bitset::BitSet,
    root_disallowed_terminal: Option<u32>,
    global_to_local_parser_state: &'a [u32],
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct SegmentedParserComponentV22 {
    constraint_artifact: Vec<u8>,
    tokenizer_state_offset: u32,
    terminal_offset: u32,
    root_entry_terminals: crate::ds::bitset::BitSet,
    root_disallowed_terminal: Option<u32>,
    global_to_local_parser_state: Vec<u32>,
}

#[derive(Serialize)]
struct BoundaryTerminalTrieNodeV22Ref<'a> {
    children: &'a [(u32, u32)],
    outputs: &'a [u32],
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct BoundaryTerminalTrieNodeV22 {
    children: Vec<(u32, u32)>,
    outputs: Vec<u32>,
}

#[derive(Serialize)]
struct SegmentedBoundaryTerminalTrieV22Ref<'a> {
    nodes: Vec<BoundaryTerminalTrieNodeV22Ref<'a>>,
    root_by_tsid: &'a [u32],
    tokenizer_state_to_tsid: &'a [u32],
    internal_token_to_originals: &'a [Vec<u32>],
    symbolic_nwa: Option<&'a crate::runtime::artifact::BoundaryTerminalNwa>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct SegmentedBoundaryTerminalTrieV22 {
    nodes: Vec<BoundaryTerminalTrieNodeV22>,
    root_by_tsid: Vec<u32>,
    tokenizer_state_to_tsid: Vec<u32>,
    internal_token_to_originals: Vec<Vec<u32>>,
    symbolic_nwa: Option<crate::runtime::artifact::BoundaryTerminalNwa>,
}

#[derive(Serialize)]
struct SegmentedRuntimeArtifactV22Ref<'a> {
    materialized_static_component_parser: Option<crate::automata::weighted_u32::dwa::DWA>,
    materialized_static_parser_state_domain_labels: Vec<i32>,
    components: Vec<SegmentedParserComponentV22Ref<'a>>,
    segmented_mask_authoritative: bool,
    segmented_component_union_root_dispatch: &'a [u32],
    boundary_parser: Option<SegmentedBoundaryParserV20Ref<'a>>,
    boundary_terminal_trie: Option<SegmentedBoundaryTerminalTrieV22Ref<'a>>,
}

#[derive(Deserialize)]
struct SegmentedRuntimeArtifactV22 {
    materialized_static_component_parser: Option<crate::automata::weighted_u32::dwa::DWA>,
    materialized_static_parser_state_domain_labels: Vec<i32>,
    components: Vec<SegmentedParserComponentV22>,
    segmented_mask_authoritative: bool,
    segmented_component_union_root_dispatch: Vec<u32>,
    boundary_parser: Option<crate::runtime::artifact::SegmentedBoundaryParser>,
    boundary_terminal_trie: Option<SegmentedBoundaryTerminalTrieV22>,
}

#[derive(Serialize)]
struct SegmentedBoundaryParserV23Ref<'a> {
    parser_dwa: Cow<'a, crate::automata::weighted_u32::dwa::DWA>,
    uses_composed_tsid_coordinate: bool,
    tokenizer_state_to_tsid: &'a [u32],
    internal_token_to_originals: &'a [Vec<u32>],
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct SegmentedBoundaryParserV23 {
    parser_dwa: crate::automata::weighted_u32::dwa::DWA,
    uses_composed_tsid_coordinate: bool,
    tokenizer_state_to_tsid: Vec<u32>,
    internal_token_to_originals: Vec<Vec<u32>>,
}

#[derive(Serialize)]
enum SegmentedBoundaryShardScopeV23Ref<'a> {
    Global,
    Component {
        start_component: u32,
        start_parser_states: &'a crate::ds::bitset::BitSet,
        accepts_empty_stack: bool,
    },
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
enum SegmentedBoundaryShardScopeV23 {
    Global,
    Component {
        start_component: u32,
        start_parser_states: crate::ds::bitset::BitSet,
        accepts_empty_stack: bool,
    },
}

#[derive(Serialize)]
enum SegmentedBoundaryShardBackendV23Ref<'a> {
    StaticParser(SegmentedBoundaryParserV23Ref<'a>),
    DynamicTerminalTrie(SegmentedBoundaryTerminalTrieV22Ref<'a>),
    DynamicDirect,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
enum SegmentedBoundaryShardBackendV23 {
    StaticParser(SegmentedBoundaryParserV23),
    DynamicTerminalTrie(SegmentedBoundaryTerminalTrieV22),
    DynamicDirect,
}

#[derive(Serialize)]
struct SegmentedBoundaryShardV23Ref<'a> {
    scope: SegmentedBoundaryShardScopeV23Ref<'a>,
    candidate_tokens: Option<&'a [u32]>,
    backend: SegmentedBoundaryShardBackendV23Ref<'a>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct SegmentedBoundaryShardV23 {
    scope: SegmentedBoundaryShardScopeV23,
    candidate_tokens: Option<Vec<u32>>,
    backend: SegmentedBoundaryShardBackendV23,
}

#[derive(Serialize)]
struct SegmentedRuntimeArtifactV23Ref<'a> {
    materialized_static_component_parser: Option<crate::automata::weighted_u32::dwa::DWA>,
    materialized_static_parser_state_domain_labels: Vec<i32>,
    components: Vec<SegmentedParserComponentV22Ref<'a>>,
    segmented_mask_authoritative: bool,
    segmented_component_union_root_dispatch: &'a [u32],
    boundary_shards: Vec<SegmentedBoundaryShardV23Ref<'a>>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct SegmentedRuntimeArtifactV23 {
    materialized_static_component_parser: Option<crate::automata::weighted_u32::dwa::DWA>,
    materialized_static_parser_state_domain_labels: Vec<i32>,
    components: Vec<SegmentedParserComponentV22>,
    segmented_mask_authoritative: bool,
    segmented_component_union_root_dispatch: Vec<u32>,
    boundary_shards: Vec<SegmentedBoundaryShardV23>,
}

#[derive(Serialize)]
struct SegmentedParserComponentV24Ref<'a> {
    constraint_artifact: Vec<u8>,
    tokenizer_state_offset: u32,
    terminal_offset: u32,
    global_terminal_aliases: &'a [(u32, u32)],
    local_tsid_to_global_tsids: &'a [Vec<u32>],
    root_entry_terminals: crate::ds::bitset::BitSet,
    root_disallowed_terminal: Option<u32>,
    global_to_local_parser_state: &'a [u32],
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct SegmentedParserComponentV24 {
    constraint_artifact: Vec<u8>,
    tokenizer_state_offset: u32,
    terminal_offset: u32,
    global_terminal_aliases: Vec<(u32, u32)>,
    local_tsid_to_global_tsids: Vec<Vec<u32>>,
    root_entry_terminals: crate::ds::bitset::BitSet,
    root_disallowed_terminal: Option<u32>,
    global_to_local_parser_state: Vec<u32>,
}

#[derive(Serialize)]
struct SegmentedParserLinkV24Ref {
    parent_component: u32,
    slot_terminal: u32,
    child_component: u32,
    child_start: u32,
    return_pop: u32,
    child_start_nullable: bool,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct SegmentedParserLinkV24 {
    parent_component: u32,
    slot_terminal: u32,
    child_component: u32,
    child_start: u32,
    return_pop: u32,
    child_start_nullable: bool,
}

#[derive(Serialize)]
struct SegmentedRuntimeArtifactV24Ref<'a> {
    materialized_static_component_parser: Option<crate::automata::weighted_u32::dwa::DWA>,
    materialized_static_parser_state_domain_labels: Vec<i32>,
    components: Vec<SegmentedParserComponentV24Ref<'a>>,
    segmented_parser_links: Vec<SegmentedParserLinkV24Ref>,
    segmented_parser_state_offsets: &'a [u32],
    segmented_mask_authoritative: bool,
    segmented_component_union_root_dispatch: &'a [u32],
    boundary_shards: Vec<SegmentedBoundaryShardV23Ref<'a>>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct SegmentedRuntimeArtifactV24 {
    materialized_static_component_parser: Option<crate::automata::weighted_u32::dwa::DWA>,
    materialized_static_parser_state_domain_labels: Vec<i32>,
    components: Vec<SegmentedParserComponentV24>,
    segmented_parser_links: Vec<SegmentedParserLinkV24>,
    segmented_parser_state_offsets: Vec<u32>,
    segmented_mask_authoritative: bool,
    segmented_component_union_root_dispatch: Vec<u32>,
    boundary_shards: Vec<SegmentedBoundaryShardV23>,
}

#[derive(Serialize)]
struct RecursiveSegmentedParserComponentV27Ref<'a> {
    constraint_artifact: Vec<u8>,
    tokenizer_state_offset: u32,
    terminal_offset: u32,
    global_terminal_aliases: &'a [(u32, u32)],
    local_tsid_to_global_tsids: &'a [Vec<u32>],
    root_entry_terminals: crate::ds::bitset::BitSet,
    root_disallowed_terminal: Option<u32>,
}

#[derive(Deserialize)]
struct RecursiveSegmentedParserComponentV27 {
    constraint_artifact: Vec<u8>,
    tokenizer_state_offset: u32,
    terminal_offset: u32,
    global_terminal_aliases: Vec<(u32, u32)>,
    local_tsid_to_global_tsids: Vec<Vec<u32>>,
    root_entry_terminals: crate::ds::bitset::BitSet,
    root_disallowed_terminal: Option<u32>,
}

#[derive(Serialize)]
struct RecursiveSegmentedBoundaryParserV27Ref<'a> {
    parser_dwa: &'a crate::automata::weighted_u32::dwa::DWA,
    uses_composed_tsid_coordinate: bool,
    tokenizer_state_to_tsid: &'a [u32],
    internal_token_to_originals: &'a [Vec<u32>],
}

#[derive(Deserialize)]
struct RecursiveSegmentedBoundaryParserV27 {
    parser_dwa: crate::automata::weighted_u32::dwa::DWA,
    uses_composed_tsid_coordinate: bool,
    tokenizer_state_to_tsid: Vec<u32>,
    internal_token_to_originals: Vec<Vec<u32>>,
}

#[derive(Serialize)]
enum RecursiveSegmentedBoundaryShardBackendV27Ref<'a> {
    StaticParser(RecursiveSegmentedBoundaryParserV27Ref<'a>),
    DynamicDirect,
}

#[derive(Deserialize)]
enum RecursiveSegmentedBoundaryShardBackendV27 {
    StaticParser(RecursiveSegmentedBoundaryParserV27),
    DynamicDirect,
}

#[derive(Serialize)]
struct RecursiveSegmentedBoundaryShardV27Ref<'a> {
    start_component: u32,
    accepts_empty_stack: bool,
    candidate_tokens: Option<&'a [u32]>,
    backend: RecursiveSegmentedBoundaryShardBackendV27Ref<'a>,
}

#[derive(Deserialize)]
struct RecursiveSegmentedBoundaryShardV27 {
    start_component: u32,
    accepts_empty_stack: bool,
    candidate_tokens: Option<Vec<u32>>,
    backend: RecursiveSegmentedBoundaryShardBackendV27,
}

#[derive(Serialize)]
struct RecursiveSegmentedRuntimeArtifactV27Ref<'a> {
    components: Vec<RecursiveSegmentedParserComponentV27Ref<'a>>,
    segmented_parser_links: Vec<SegmentedParserLinkV24Ref>,
    recursive_compiler_table: &'a [u8],
    recursive_tokenizer_internal_tsids: &'a [Vec<u32>],
    segmented_mask_authoritative: bool,
    boundary_shards: Vec<RecursiveSegmentedBoundaryShardV27Ref<'a>>,
}

#[derive(Deserialize)]
struct RecursiveSegmentedRuntimeArtifactV27 {
    components: Vec<RecursiveSegmentedParserComponentV27>,
    segmented_parser_links: Vec<SegmentedParserLinkV24>,
    recursive_compiler_table: Vec<u8>,
    recursive_tokenizer_internal_tsids: Vec<Vec<u32>>,
    segmented_mask_authoritative: bool,
    boundary_shards: Vec<RecursiveSegmentedBoundaryShardV27>,
}

#[derive(Serialize)]
enum SegmentedRuntimeArtifactV27Ref<'a> {
    Recursive(RecursiveSegmentedRuntimeArtifactV27Ref<'a>),
    LegacyV24(SegmentedRuntimeArtifactV24Ref<'a>),
}

#[derive(Deserialize)]
enum SegmentedRuntimeArtifactV27 {
    Recursive(RecursiveSegmentedRuntimeArtifactV27),
    LegacyV24(SegmentedRuntimeArtifactV24),
}

#[derive(Deserialize)]
struct VirtualResidualMaskProjectionArtifactV28 {
    terminal: TerminalID,
    state_offset: u32,
    local_to_mask_state: Vec<u32>,
}

#[derive(Deserialize)]
struct StaticVirtualResidualMaskArtifactV28 {
    mask_tokenizer_bytes: Vec<u8>,
    projections: Vec<VirtualResidualMaskProjectionArtifactV28>,
}

impl StaticVirtualResidualMaskArtifactV28 {
    fn into_current(self) -> StaticVirtualResidualMaskArtifact {
        StaticVirtualResidualMaskArtifact {
            mask_tokenizer_bytes: self.mask_tokenizer_bytes,
            projections: self
                .projections
                .into_iter()
                .map(|projection| {
                    crate::automata::lexer::tokenizer::VirtualResidualMaskProjectionArtifact::from_wire(
                        projection.terminal,
                        projection.state_offset,
                        projection.local_to_mask_state,
                        Vec::new(),
                        Vec::new(),
                        0,
                        0,
                    )
                })
                .collect(),
        }
    }
}

#[derive(Serialize)]
struct StaticVirtualResidualMaskArtifactRef<'a> {
    mask_tokenizer_bytes: Vec<u8>,
    projections: Vec<
        crate::automata::lexer::tokenizer::VirtualResidualMaskProjectionArtifactRef<'a>,
    >,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StaticVirtualResidualMaskArtifact {
    mask_tokenizer_bytes: Vec<u8>,
    projections: Vec<
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
    let sparse = section.starts_with(&STATIC_RESIDUAL_MASK_MAGIC);
    let dense_legacy = section.starts_with(&PREVIOUS_STATIC_RESIDUAL_MASK_MAGIC);
    if !sparse && !dense_legacy {
        return Err("invalid static residual mask wire header".to_owned());
    }
    let count = u32::from_le_bytes(section[4..8].try_into().unwrap()) as usize;
    let tokenizer_len = usize::try_from(u64::from_le_bytes(section[8..16].try_into().unwrap()))
        .map_err(|_| "static residual tokenizer length does not fit platform".to_owned())?;
    let descriptor_width = if sparse { 56usize } else { 48usize };
    let descriptor_bytes = count.checked_mul(descriptor_width).ok_or_else(|| "static residual descriptor length overflow".to_owned())?;
    let mut pos = 16usize;
    let descriptor_end = pos.checked_add(descriptor_bytes).ok_or_else(|| "static residual descriptor overflow".to_owned())?;
    if descriptor_end > section.len() { return Err("truncated static residual descriptors".to_owned()); }
    let mut descriptors = Vec::with_capacity(count);
    while pos < descriptor_end {
        let terminal = u32::from_le_bytes(section[pos..pos+4].try_into().unwrap()); pos += 4;
        let state_offset = u32::from_le_bytes(section[pos..pos+4].try_into().unwrap()); pos += 4;
        let map_len = usize::try_from(u64::from_le_bytes(section[pos..pos+8].try_into().unwrap())).map_err(|_| "static residual map length does not fit platform".to_owned())?; pos += 8;
        let live_count = if sparse {
            let value = usize::try_from(u64::from_le_bytes(section[pos..pos+8].try_into().unwrap())).map_err(|_| "static residual live map length does not fit platform".to_owned())?;
            pos += 8;
            if value > map_len { return Err("static residual live map length exceeds dense length".to_owned()); }
            value
        } else { map_len };
        let oracle_len = usize::try_from(u64::from_le_bytes(section[pos..pos+8].try_into().unwrap())).map_err(|_| "static residual oracle length does not fit platform".to_owned())?; pos += 8;
        let expr_len = usize::try_from(u64::from_le_bytes(section[pos..pos+8].try_into().unwrap())).map_err(|_| "static residual expression length does not fit platform".to_owned())?; pos += 8;
        let mask_max = usize::try_from(u64::from_le_bytes(section[pos..pos+8].try_into().unwrap())).map_err(|_| "static residual mask max does not fit platform".to_owned())?; pos += 8;
        let crossed_boundaries = usize::try_from(u64::from_le_bytes(section[pos..pos+8].try_into().unwrap())).map_err(|_| "static residual boundary horizon does not fit platform".to_owned())?; pos += 8;
        descriptors.push((terminal, state_offset, map_len, live_count, oracle_len, expr_len, mask_max, crossed_boundaries));
    }
    let tokenizer_local_start = descriptor_end;
    let tokenizer_local_end = tokenizer_local_start.checked_add(tokenizer_len).ok_or_else(|| "static residual tokenizer range overflow".to_owned())?;
    if tokenizer_local_end > section.len() { return Err("truncated static residual tokenizer".to_owned()); }
    let mask_state_count = if sparse {
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
    } else {
        0
    };
    if sparse {
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
        let component_state_count = if sparse {
            descriptors
                .get(descriptor_index + 1)
                .map_or(mask_state_count, |next| next.1)
                .checked_sub(state_offset)
                .ok_or_else(|| "invalid static residual projection component width".to_owned())?
        } else {
            0
        };
        let mut map = if sparse { vec![u32::MAX; map_len] } else { Vec::with_capacity(map_len) };
        if sparse {
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
        } else {
            let map_bytes = map_len.checked_mul(4).ok_or_else(|| "static residual map byte length overflow".to_owned())?;
            let map_end = pos.checked_add(map_bytes).ok_or_else(|| "static residual map range overflow".to_owned())?;
            if map_end > section.len() { return Err("truncated static residual projection map".to_owned()); }
            for word in section[pos..map_end].chunks_exact(4) { map.push(u32::from_le_bytes(word.try_into().unwrap())); }
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
        let projection = if sparse {
            crate::automata::lexer::tokenizer::VirtualResidualMaskProjectionArtifact::from_validated_wire(
                terminal, state_offset, map, oracle_bytes, runtime_expr_bytes, mask_max,
                crossed_boundaries, component_state_count,
            )
        } else {
            crate::automata::lexer::tokenizer::VirtualResidualMaskProjectionArtifact::from_wire(
                terminal, state_offset, map, oracle_bytes, runtime_expr_bytes, mask_max,
                crossed_boundaries,
            )
        };
        projections.push(projection);
    }
    if pos != section.len() { return Err("trailing bytes in static residual mask wire".to_owned()); }
    Ok(DecodedStaticVirtualResidualMask::Backed { backing, tokenizer_start, tokenizer_len, projections })
}

fn encode_current_runtime_wire(
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

fn decode_current_runtime_wire(
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
struct ConstraintArtifactCurrentRuntimeRef<'a> {
    terminal_live_states: &'a [Vec<u32>],
    segmented_runtime: Option<SegmentedRuntimeArtifactV27Ref<'a>>,
    dynamic_mask_vocab: Option<crate::runtime::artifact::DynamicMaskVocabArtifact>,
    virtual_runtimes: Vec<crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata>,
    static_virtual_residual_mask: Option<StaticVirtualResidualMaskArtifactRef<'a>>,
    packed_dwa_dense_mask_ids: &'a [u32],
    packed_dwa_dense_mask_rows: &'a [u64],
}

#[derive(Deserialize)]
struct ConstraintArtifactV28Runtime {
    terminal_live_states: Vec<Vec<u32>>,
    segmented_runtime: Option<SegmentedRuntimeArtifactV27>,
    dynamic_mask_vocab: Option<crate::runtime::artifact::DynamicMaskVocabArtifact>,
    virtual_runtimes: Vec<crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata>,
    static_virtual_residual_mask: Option<StaticVirtualResidualMaskArtifactV28>,
    packed_dwa_dense_mask_ids: Vec<u32>,
    packed_dwa_dense_mask_rows: Vec<u64>,
}

#[derive(Deserialize)]
struct ConstraintArtifactCurrentRuntime {
    terminal_live_states: Vec<Vec<u32>>,
    segmented_runtime: Option<SegmentedRuntimeArtifactV27>,
    dynamic_mask_vocab: Option<crate::runtime::artifact::DynamicMaskVocabArtifact>,
    virtual_runtimes: Vec<crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata>,
    static_virtual_residual_mask: Option<StaticVirtualResidualMaskArtifact>,
    packed_dwa_dense_mask_ids: Vec<u32>,
    packed_dwa_dense_mask_rows: Vec<u64>,
}

#[derive(Deserialize)]
struct ConstraintArtifactV27Runtime {
    terminal_live_states: Vec<Vec<u32>>,
    segmented_runtime: Option<SegmentedRuntimeArtifactV27>,
    dynamic_mask_vocab: Option<crate::runtime::artifact::DynamicMaskVocabArtifact>,
    virtual_runtimes: Vec<crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata>,
    packed_dwa_dense_mask_ids: Vec<u32>,
    packed_dwa_dense_mask_rows: Vec<u64>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct ConstraintArtifactV24Runtime {
    terminal_live_states: Vec<Vec<u32>>,
    segmented_runtime: Option<SegmentedRuntimeArtifactV24>,
    dynamic_mask_vocab: Option<crate::runtime::artifact::DynamicMaskVocabArtifact>,
    packed_dwa_dense_mask_ids: Vec<u32>,
    packed_dwa_dense_mask_rows: Vec<u64>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct ConstraintArtifactV23Runtime {
    terminal_live_states: Vec<Vec<u32>>,
    segmented_runtime: Option<SegmentedRuntimeArtifactV23>,
    dynamic_mask_vocab: Option<crate::runtime::artifact::DynamicMaskVocabArtifact>,
    packed_dwa_dense_mask_ids: Vec<u32>,
    packed_dwa_dense_mask_rows: Vec<u64>,
}

#[derive(Deserialize)]
struct ConstraintArtifactV22Runtime {
    terminal_live_states: Vec<Vec<u32>>,
    segmented_runtime: Option<SegmentedRuntimeArtifactV22>,
    dynamic_mask_vocab: Option<crate::runtime::artifact::DynamicMaskVocabArtifact>,
    packed_dwa_dense_mask_ids: Vec<u32>,
    packed_dwa_dense_mask_rows: Vec<u64>,
}

#[derive(Deserialize)]
struct ConstraintArtifactV21Runtime {
    terminal_live_states: Vec<Vec<u32>>,
    segmented_runtime: Option<SegmentedRuntimeArtifactV20>,
    packed_dwa_dense_mask_ids: Vec<u32>,
    packed_dwa_dense_mask_rows: Vec<u64>,
}

#[derive(Deserialize)]
struct ConstraintArtifactV20Runtime {
    terminal_live_states: Vec<Vec<u32>>,
    segmented_runtime: Option<SegmentedRuntimeArtifactV20>,
}

struct DecodedConstraintRuntime {
    terminal_live_states: Vec<Vec<u32>>,
    segmented_runtime_v20: Option<SegmentedRuntimeArtifactV20>,
    segmented_runtime_v22: Option<SegmentedRuntimeArtifactV22>,
    segmented_runtime_v23: Option<SegmentedRuntimeArtifactV23>,
    segmented_runtime_v24: Option<SegmentedRuntimeArtifactV24>,
    segmented_runtime_v27: Option<SegmentedRuntimeArtifactV27>,
    dynamic_mask_vocab: Option<crate::runtime::artifact::DynamicMaskVocabArtifact>,
    virtual_runtimes: Vec<crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata>,
    static_virtual_residual_mask: Option<DecodedStaticVirtualResidualMask>,
    packed_dwa_dense_masks: Option<(Vec<u32>, Vec<u64>)>,
}

fn boundary_parser_artifact_ref(
    boundary: &crate::runtime::artifact::SegmentedBoundaryParser,
) -> SegmentedBoundaryParserV20Ref<'_> {
    let parser_dwa = if let Some(compact) = boundary.compact_parser_dwa.as_ref() {
        Cow::Owned(compact.to_generic_dwa())
    } else {
        Cow::Borrowed(&boundary.parser_dwa)
    };
    SegmentedBoundaryParserV20Ref {
        parser_dwa,
        tokenizer_state_to_tsid: &boundary.tokenizer_state_to_tsid,
        internal_token_to_originals: &boundary.internal_token_to_originals,
    }
}

fn boundary_parser_v23_ref(
    boundary: &crate::runtime::artifact::SegmentedBoundaryParser,
) -> SegmentedBoundaryParserV23Ref<'_> {
    let parser_dwa = if let Some(compact) = boundary.compact_parser_dwa.as_ref() {
        Cow::Owned(compact.to_generic_dwa())
    } else {
        Cow::Borrowed(&boundary.parser_dwa)
    };
    SegmentedBoundaryParserV23Ref {
        parser_dwa,
        uses_composed_tsid_coordinate: boundary.uses_composed_tsid_coordinate,
        tokenizer_state_to_tsid: &boundary.tokenizer_state_to_tsid,
        internal_token_to_originals: &boundary.internal_token_to_originals,
    }
}

fn boundary_terminal_trie_v22_ref(
    boundary: &crate::runtime::artifact::SegmentedBoundaryTerminalTrie,
) -> SegmentedBoundaryTerminalTrieV22Ref<'_> {
    SegmentedBoundaryTerminalTrieV22Ref {
        nodes: boundary
            .nodes
            .iter()
            .map(|node| BoundaryTerminalTrieNodeV22Ref {
                children: &node.children,
                outputs: &node.outputs,
            })
            .collect(),
        root_by_tsid: &boundary.root_by_tsid,
        tokenizer_state_to_tsid: &boundary.tokenizer_state_to_tsid,
        internal_token_to_originals: &boundary.internal_token_to_originals,
        symbolic_nwa: boundary.symbolic_nwa.as_ref(),
    }
}

fn serialized_component_root_entry_terminals(
    component: &crate::runtime::SegmentedParserComponent,
    global_terminal_count: usize,
) -> crate::ds::bitset::BitSet {
    let mut terminals = crate::ds::bitset::BitSet::new(global_terminal_count);
    let end = component
        .terminal_offset
        .saturating_add(component.constraint.table.num_terminals)
        .min(global_terminal_count as u32);
    for terminal in component.terminal_offset..end {
        terminals.set(terminal as usize);
    }
    terminals
}

fn segmented_runtime_artifact_v24_ref(
    constraint: &Constraint,
) -> Option<SegmentedRuntimeArtifactV24Ref<'_>> {
    let overlay = constraint.static_dynamic_overlay.as_ref()?;
    if overlay.segmented_parser_components.is_empty()
        && overlay.segmented_boundary_shards.is_empty()
        && overlay.segmented_boundary_parser.is_none()
        && overlay.segmented_boundary_terminal_trie.is_none()
    {
        return None;
    }
    let components = overlay
        .segmented_parser_components
        .iter()
        .map(|component| SegmentedParserComponentV24Ref {
            // Segmented components are embeddable bodies, not generation
            // roots. They may intentionally retain linker slots, so never
            // wrap them in the closed-root policy/vocabulary envelope.
            constraint_artifact: component.constraint.save_body(),
            tokenizer_state_offset: component.tokenizer_state_offset,
            terminal_offset: component.terminal_offset,
            global_terminal_aliases: &component.global_terminal_aliases,
            local_tsid_to_global_tsids: &component.local_tsid_to_global_tsids,
            root_entry_terminals: serialized_component_root_entry_terminals(
                component,
                constraint.table.num_terminals as usize,
            ),
            root_disallowed_terminal: component.root_disallowed_terminal,
            global_to_local_parser_state: &component.global_to_local_parser_state,
        })
        .collect();

    let boundary_shards = if !overlay.segmented_boundary_shards.is_empty() {
        overlay
            .segmented_boundary_shards
            .iter()
            .map(|shard| SegmentedBoundaryShardV23Ref {
                scope: SegmentedBoundaryShardScopeV23Ref::Component {
                    start_component: shard.start_component,
                    start_parser_states: &shard.start_parser_states,
                    accepts_empty_stack: shard.accepts_empty_stack,
                },
                candidate_tokens: shard.candidate_tokens.as_deref(),
                backend: match &shard.backend {
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(boundary) => {
                        SegmentedBoundaryShardBackendV23Ref::StaticParser(
                            boundary_parser_v23_ref(boundary),
                        )
                    }
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(boundary) => {
                        SegmentedBoundaryShardBackendV23Ref::DynamicTerminalTrie(
                            boundary_terminal_trie_v22_ref(boundary),
                        )
                    }
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect => {
                        SegmentedBoundaryShardBackendV23Ref::DynamicDirect
                    }
                },
            })
            .collect()
    } else {
        let mut shards = Vec::new();
        if let Some(boundary) = overlay.segmented_boundary_parser.as_deref() {
            shards.push(SegmentedBoundaryShardV23Ref {
                scope: SegmentedBoundaryShardScopeV23Ref::Global,
                candidate_tokens: None,
                backend: SegmentedBoundaryShardBackendV23Ref::StaticParser(
                    boundary_parser_v23_ref(boundary),
                ),
            });
        }
        if let Some(boundary) = overlay.segmented_boundary_terminal_trie.as_deref() {
            shards.push(SegmentedBoundaryShardV23Ref {
                scope: SegmentedBoundaryShardScopeV23Ref::Global,
                candidate_tokens: None,
                backend: SegmentedBoundaryShardBackendV23Ref::DynamicTerminalTrie(
                    boundary_terminal_trie_v22_ref(boundary),
                ),
            });
        }
        shards
    };

    let segmented_parser_links = overlay
        .segmented_parser_links
        .iter()
        .map(|link| SegmentedParserLinkV24Ref {
            parent_component: link.parent_component,
            slot_terminal: link.slot_terminal,
            child_component: link.child_component,
            child_start: link.child_start,
            return_pop: link.return_pop,
            child_start_nullable: link.child_start_nullable,
        })
        .collect();

    Some(SegmentedRuntimeArtifactV24Ref {
        materialized_static_component_parser: None,
        materialized_static_parser_state_domain_labels: Vec::new(),
        components,
        segmented_parser_links,
        segmented_parser_state_offsets: &overlay.segmented_parser_state_offsets,
        segmented_mask_authoritative: overlay.segmented_mask_authoritative,
        segmented_component_union_root_dispatch: &overlay.segmented_component_union_root_dispatch,
        boundary_shards,
    })
}

fn segmented_runtime_artifact_ref(
    constraint: &Constraint,
) -> Option<SegmentedRuntimeArtifactV27Ref<'_>> {
    if !constraint.uses_compact_segmented_parser_runtime() {
        return segmented_runtime_artifact_v24_ref(constraint)
            .map(SegmentedRuntimeArtifactV27Ref::LegacyV24);
    }
    let layout = constraint
        .recursive_parser_layout()
        .expect("validated recursive runtime must derive its parser layout before serialization")
        .expect("provider-native segmented runtime must have a recursive parser layout");
    let overlay = constraint.static_dynamic_overlay.as_ref()?;
    let recursive_compiler_table = overlay
        .recursive_compiler_table
        .get()
        .expect("recursive runtime must retain its compiler table blob");
    if overlay.recursive_tokenizer_internal_tsids.get().is_none() {
        let compatibility_omission = recursive_compiler_table.is_empty()
            && !overlay.segmented_parser_components.is_empty()
            && overlay.segmented_parser_components.iter().all(|component| {
                component.boundary.as_ref().is_some_and(|shard| {
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
                    )
                })
            });
        if !compatibility_omission {
            panic!("recursive runtime must retain its scoped tokenizer TSID relation");
        }
        // Wire compatibility only: old v27 readers require one row per scoped
        // tokenizer state. The live DynamicDirect coordinator never consumes
        // this quotient, so materialize the historical all-zero relation only
        // when serialization is explicitly requested.
        overlay
            .recursive_tokenizer_internal_tsids
            .set(Arc::new(vec![vec![0u32]; layout.total_tokenizer_states as usize]))
            .expect("recursive serializer compatibility relation initialized twice");
    }
    let recursive_tokenizer_internal_tsids = overlay
        .recursive_tokenizer_internal_tsids
        .get()
        .expect("recursive runtime must retain its scoped tokenizer TSID relation");
    let components = overlay
        .segmented_parser_components
        .iter()
        .map(|component| RecursiveSegmentedParserComponentV27Ref {
            // Recursive segmented components are likewise internal bodies.
            // The outer root/module artifact carries exact-vocabulary identity
            // and rebinds it recursively after load.
            constraint_artifact: component.constraint.save_body(),
            tokenizer_state_offset: component.tokenizer_state_offset,
            terminal_offset: component.terminal_offset,
            global_terminal_aliases: &component.global_terminal_aliases,
            local_tsid_to_global_tsids: &component.local_tsid_to_global_tsids,
            root_entry_terminals: serialized_component_root_entry_terminals(
                component,
                constraint.table.num_terminals as usize,
            ),
            root_disallowed_terminal: component.root_disallowed_terminal,
        })
        .collect();
    let segmented_parser_links = overlay
        .segmented_parser_links
        .iter()
        .map(|link| SegmentedParserLinkV24Ref {
            parent_component: link.parent_component,
            slot_terminal: link.slot_terminal,
            child_component: link.child_component,
            child_start: link.child_start,
            return_pop: link.return_pop,
            child_start_nullable: link.child_start_nullable,
        })
        .collect();
    let boundary_shards = overlay
        .segmented_boundary_shards
        .iter()
        .map(|shard| {
            let backend = match &shard.backend {
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(boundary) => {
                    let parser_dwa = boundary.recursive_parser_dwa.as_ref().expect(
                        "recursive static boundary must carry its recursive parser-coordinate DWA",
                    );
                    RecursiveSegmentedBoundaryShardBackendV27Ref::StaticParser(
                        RecursiveSegmentedBoundaryParserV27Ref {
                            parser_dwa,
                            uses_composed_tsid_coordinate: boundary.uses_composed_tsid_coordinate,
                            tokenizer_state_to_tsid: &boundary.tokenizer_state_to_tsid,
                            internal_token_to_originals: &boundary.internal_token_to_originals,
                        },
                    )
                }
                crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect => {
                    RecursiveSegmentedBoundaryShardBackendV27Ref::DynamicDirect
                }
                crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(_) => {
                    unreachable!(
                        "recursive provider-native runtime cannot carry a terminal-trie boundary"
                    )
                }
            };
            RecursiveSegmentedBoundaryShardV27Ref {
                start_component: shard.start_component,
                accepts_empty_stack: shard.accepts_empty_stack,
                candidate_tokens: shard.candidate_tokens.as_deref(),
                backend,
            }
        })
        .collect();
    Some(SegmentedRuntimeArtifactV27Ref::Recursive(
        RecursiveSegmentedRuntimeArtifactV27Ref {
            components,
            segmented_parser_links,
            recursive_compiler_table: recursive_compiler_table.as_ref(),
            recursive_tokenizer_internal_tsids:
                recursive_tokenizer_internal_tsids.as_slice(),
            segmented_mask_authoritative: overlay.segmented_mask_authoritative,
            boundary_shards,
        },
    ))
}

fn restore_boundary_terminal_trie_v22(
    boundary: SegmentedBoundaryTerminalTrieV22,
    global_terminal_count: usize,
) -> crate::Result<crate::runtime::artifact::SegmentedBoundaryTerminalTrie> {
    fn weight_within_boundary_domain(weight: &Weight, tsids: usize, tokens: usize) -> bool {
        if weight.is_empty() || weight.is_full() {
            return true;
        }
        weight.range_entries().all(|(_, end_tsid, token_set)| {
            end_tsid < tsids as u32
                && token_set
                    .ranges()
                    .all(|range| *range.end() < tokens as u32)
        })
    }

    let node_count = boundary.nodes.len();
    if boundary
        .root_by_tsid
        .iter()
        .any(|&root| root != u32::MAX && root as usize >= node_count)
    {
        return Err(crate::GlrMaskError::Serialization(
            "serialized boundary terminal trie references an invalid root node".to_owned(),
        ));
    }
    if boundary
        .tokenizer_state_to_tsid
        .iter()
        .any(|&tsid| tsid != u32::MAX && tsid as usize >= boundary.root_by_tsid.len())
    {
        return Err(crate::GlrMaskError::Serialization(
            "serialized boundary terminal trie references an invalid TSID".to_owned(),
        ));
    }
    let internal_token_count = boundary.internal_token_to_originals.len();
    for (node_index, node) in boundary.nodes.iter().enumerate() {
        if node
            .children
            .iter()
            .any(|&(_, child)| child as usize >= node_count)
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "serialized boundary terminal trie node {node_index} references an invalid child"
            )));
        }
        if node
            .outputs
            .iter()
            .any(|&token| token as usize >= internal_token_count)
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "serialized boundary terminal trie node {node_index} references an invalid token class"
            )));
        }
    }

    if let Some(symbolic_nwa) = boundary.symbolic_nwa.as_ref() {
        let symbolic_node_count = symbolic_nwa.nodes.len();
        if symbolic_nwa.topological_order.len() != symbolic_node_count {
            return Err(crate::GlrMaskError::Serialization(
                "serialized boundary terminal NWA has an incomplete topological order".to_owned(),
            ));
        }
        let mut position = vec![usize::MAX; symbolic_node_count];
        for (index, &state) in symbolic_nwa.topological_order.iter().enumerate() {
            let Some(slot) = position.get_mut(state as usize) else {
                return Err(crate::GlrMaskError::Serialization(
                    "serialized boundary terminal NWA topological order references an invalid state"
                        .to_owned(),
                ));
            };
            if *slot != usize::MAX {
                return Err(crate::GlrMaskError::Serialization(
                    "serialized boundary terminal NWA topological order contains a duplicate state"
                        .to_owned(),
                ));
            }
            *slot = index;
        }
        if symbolic_nwa
            .start_states
            .iter()
            .any(|&state| state as usize >= symbolic_node_count)
        {
            return Err(crate::GlrMaskError::Serialization(
                "serialized boundary terminal NWA references an invalid start state".to_owned(),
            ));
        }
        for (source, node) in symbolic_nwa.nodes.iter().enumerate() {
            if node.final_weight.as_ref().is_some_and(|weight| {
                !weight_within_boundary_domain(
                    weight,
                    boundary.root_by_tsid.len(),
                    internal_token_count,
                )
            }) {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "serialized boundary terminal NWA state {source} has an out-of-domain final weight"
                )));
            }
            for transition in &node.transitions {
                if transition.terminal as usize >= global_terminal_count
                    || transition.target as usize >= symbolic_node_count
                    || position[source] >= position[transition.target as usize]
                    || !weight_within_boundary_domain(
                        &transition.weight,
                        boundary.root_by_tsid.len(),
                        internal_token_count,
                    )
                {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "serialized boundary terminal NWA state {source} has an invalid labeled transition"
                    )));
                }
            }
            for (target, weight) in &node.epsilons {
                if *target as usize >= symbolic_node_count
                    || position[source] >= position[*target as usize]
                    || !weight_within_boundary_domain(
                        weight,
                        boundary.root_by_tsid.len(),
                        internal_token_count,
                    )
                {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "serialized boundary terminal NWA state {source} has an invalid epsilon transition"
                    )));
                }
            }
        }
    }

    Ok(crate::runtime::artifact::SegmentedBoundaryTerminalTrie {
        nodes: boundary
            .nodes
            .into_iter()
            .map(|node| crate::runtime::artifact::BoundaryTerminalTrieNode {
                children: node.children,
                outputs: node.outputs,
            })
            .collect(),
        root_by_tsid: boundary.root_by_tsid,
        tokenizer_state_to_tsid: boundary.tokenizer_state_to_tsid,
        internal_token_to_originals: boundary.internal_token_to_originals,
        symbolic_nwa: boundary.symbolic_nwa,
    })
}

fn restore_segmented_runtime_v22(
    constraint: &mut Constraint,
    runtime: SegmentedRuntimeArtifactV22,
) -> crate::Result<()> {
    let global_state_count = constraint.table.num_states as usize;
    let global_terminal_count = constraint.table.num_terminals as usize;
    let global_tokenizer_states = constraint.tokenizer.num_states();
    let has_static_baseline = runtime.materialized_static_component_parser.is_some();
    if let Some(parser_dwa) = runtime.materialized_static_component_parser {
        if !runtime.materialized_static_parser_state_domain_labels.is_empty()
            && runtime.materialized_static_parser_state_domain_labels.len() != global_state_count
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "serialized static component parser has {} domain labels for {global_state_count} outer states",
                runtime.materialized_static_parser_state_domain_labels.len(),
            )));
        }
        constraint.parser_dwa = parser_dwa;
        constraint.packed_parser_dwa = None;
        constraint.parser_state_domain_labels =
            runtime.materialized_static_parser_state_domain_labels;
    } else if !runtime.materialized_static_parser_state_domain_labels.is_empty() {
        return Err(crate::GlrMaskError::Serialization(
            "serialized static component parser labels exist without a parser".to_owned(),
        ));
    }
    let mut components = Vec::with_capacity(runtime.components.len());
    for (index, component) in runtime.components.into_iter().enumerate() {
        if component.global_to_local_parser_state.len() != global_state_count {
            return Err(crate::GlrMaskError::Serialization(format!(
                "segmented component {index} parser-state projection has {} entries for {global_state_count} outer states",
                component.global_to_local_parser_state.len(),
            )));
        }
        if component.root_entry_terminals.len() != global_terminal_count {
            return Err(crate::GlrMaskError::Serialization(format!(
                "segmented component {index} root-entry terminal set has length {} for {global_terminal_count} outer terminals",
                component.root_entry_terminals.len(),
            )));
        }
        let child = Constraint::load_body_artifact(component.constraint_artifact)?;
        if component
            .terminal_offset
            .checked_add(child.table.num_terminals)
            .is_none_or(|end| end as usize > global_terminal_count)
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "segmented component {index} terminal range lies outside outer terminal domain"
            )));
        }
        if component
            .tokenizer_state_offset
            .checked_add(child.tokenizer.num_states().saturating_sub(1))
            .is_none_or(|last| last >= global_tokenizer_states)
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "segmented component {index} tokenizer-state range lies outside outer tokenizer"
            )));
        }
        if component
            .root_disallowed_terminal
            .is_some_and(|terminal| terminal >= child.table.num_terminals)
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "segmented component {index} root-disallowed terminal lies outside the component"
            )));
        }
        if component.global_to_local_parser_state.iter().any(|&local| {
            local != u32::MAX && local >= child.table.num_states
        }) {
            return Err(crate::GlrMaskError::Serialization(format!(
                "segmented component {index} parser-state projection references an invalid local state"
            )));
        }
        components.push(crate::runtime::SegmentedParserComponent {
            constraint: std::sync::Arc::new(child),
            boundary: None,
            tokenizer_state_offset: component.tokenizer_state_offset,
            terminal_offset: component.terminal_offset,
            global_terminal_aliases: Vec::new(),
            local_tsid_to_global_tsids: Vec::new(),
            root_disallowed_terminal: component.root_disallowed_terminal,
            global_to_local_parser_state: component.global_to_local_parser_state,
        });
    }
    if !runtime.segmented_component_union_root_dispatch.is_empty()
        && runtime.segmented_component_union_root_dispatch.len() != global_state_count
    {
        return Err(crate::GlrMaskError::Serialization(
            "segmented component root dispatch has the wrong outer parser-state domain".to_owned(),
        ));
    }
    if runtime
        .segmented_component_union_root_dispatch
        .iter()
        .any(|&component| component != u32::MAX && component as usize >= components.len())
    {
        return Err(crate::GlrMaskError::Serialization(
            "segmented component root dispatch references an unknown component".to_owned(),
        ));
    }
    let overlay = constraint
        .static_dynamic_overlay
        .get_or_insert_with(Default::default);
    overlay.segmented_parser_components = components;
    overlay.segmented_mask_authoritative = runtime.segmented_mask_authoritative;
    overlay.segmented_static_baseline = has_static_baseline;
    overlay.segmented_component_union_root_dispatch =
        runtime.segmented_component_union_root_dispatch;
    overlay.segmented_boundary_parser = runtime.boundary_parser.map(Arc::new);
    overlay.segmented_boundary_terminal_trie = runtime
        .boundary_terminal_trie
        .map(|boundary| restore_boundary_terminal_trie_v22(boundary, global_terminal_count))
        .transpose()?
        .map(Arc::new);
    // v22 stores one global B. Keep the new shard collection empty so runtime
    // deliberately takes the exact legacy/global fallback rather than
    // pretending the old wire format contained per-start-component roots.
    overlay.segmented_boundary_shards.clear();
    Ok(())
}

fn restore_boundary_parser_v23(
    constraint: &Constraint,
    boundary: SegmentedBoundaryParserV23,
) -> crate::Result<crate::runtime::artifact::SegmentedBoundaryParser> {
    let global_tokenizer_states = constraint.tokenizer.num_states();
    let tsid_count = if boundary.uses_composed_tsid_coordinate {
        if !boundary.tokenizer_state_to_tsid.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "composed-coordinate boundary shard redundantly stores a private tokenizer-state map"
                    .to_owned(),
            ));
        }
        constraint.internal_tsid_count()
    } else {
        if boundary.tokenizer_state_to_tsid.len() != global_tokenizer_states as usize {
            return Err(crate::GlrMaskError::Serialization(format!(
                "private-coordinate boundary shard has {} tokenizer-state entries for {global_tokenizer_states} outer states",
                boundary.tokenizer_state_to_tsid.len(),
            )));
        }
        boundary
            .tokenizer_state_to_tsid
            .iter()
            .copied()
            .filter(|&tsid| tsid != u32::MAX)
            .max()
            .map_or(0, |tsid| tsid as usize + 1)
    };
    let token_count = boundary.internal_token_to_originals.len();
    let state_count = boundary.parser_dwa.num_states() as usize;
    if boundary.parser_dwa.start_state() as usize >= state_count {
        return Err(crate::GlrMaskError::Serialization(
            "serialized boundary parser has an invalid start state".to_owned(),
        ));
    }
    let weight_in_domain = |weight: &Weight| {
        weight.is_empty()
            || weight.is_full()
            || weight.raw_range_values().all(|(range, tokens)| {
                *range.end() < tsid_count as u32
                    && tokens
                        .ranges()
                        .all(|token_range| *token_range.end() < token_count as u32)
            })
    };
    for (state_index, state) in boundary.parser_dwa.states().iter().enumerate() {
        if state
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight_in_domain(weight))
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "serialized boundary parser state {state_index} has an out-of-domain final weight"
            )));
        }
        for (target, weight) in state.transitions.values() {
            if *target as usize >= state_count || !weight_in_domain(weight) {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "serialized boundary parser state {state_index} has an invalid transition"
                )));
            }
        }
    }
    if !boundary.uses_composed_tsid_coordinate
        && boundary
            .tokenizer_state_to_tsid
            .iter()
            .any(|&tsid| tsid != u32::MAX && tsid as usize >= tsid_count)
    {
        return Err(crate::GlrMaskError::Serialization(
            "serialized boundary parser tokenizer-state map references an invalid TSID".to_owned(),
        ));
    }
    Ok(crate::runtime::artifact::SegmentedBoundaryParser {
        parser_dwa: boundary.parser_dwa,
        compact_parser_dwa: None,
        recursive_parser_dwa: None,
        uses_composed_tsid_coordinate: boundary.uses_composed_tsid_coordinate,
        tokenizer_state_to_tsid: boundary.tokenizer_state_to_tsid,
        internal_token_to_originals: boundary.internal_token_to_originals,
    })
}

fn restore_recursive_boundary_parser_v27(
    constraint: &Constraint,
    boundary: RecursiveSegmentedBoundaryParserV27,
    recursive_parser_state_count: u32,
    recursive_tokenizer_state_count: u32,
) -> crate::Result<crate::runtime::artifact::SegmentedBoundaryParser> {
    let tsid_count = if boundary.uses_composed_tsid_coordinate {
        if !boundary.tokenizer_state_to_tsid.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "recursive composed-coordinate boundary shard redundantly stores a private tokenizer-state map"
                    .to_owned(),
            ));
        }
        constraint.internal_tsid_count()
    } else {
        // v27 restores compact segmented runtimes only, whose shard queries
        // index live leaf states directly. The private map therefore covers
        // the recursive leaf total — not the outer composed tokenizer, which
        // may canonicalize states the leaves keep distinct.
        if boundary.tokenizer_state_to_tsid.len() != recursive_tokenizer_state_count as usize {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive private-coordinate boundary shard has {} tokenizer-state entries for {recursive_tokenizer_state_count} leaf states",
                boundary.tokenizer_state_to_tsid.len(),
            )));
        }
        boundary
            .tokenizer_state_to_tsid
            .iter()
            .copied()
            .filter(|&tsid| tsid != u32::MAX)
            .max()
            .map_or(0, |tsid| tsid as usize + 1)
    };
    let token_count = boundary.internal_token_to_originals.len();
    let state_count = boundary.parser_dwa.num_states() as usize;
    if boundary.parser_dwa.start_state() as usize >= state_count {
        return Err(crate::GlrMaskError::Serialization(
            "serialized recursive boundary parser has an invalid start state".to_owned(),
        ));
    }
    let weight_in_domain = |weight: &Weight| {
        weight.is_empty()
            || weight.is_full()
            || weight.raw_range_values().all(|(range, tokens)| {
                *range.end() < tsid_count as u32
                    && tokens
                        .ranges()
                        .all(|token_range| *token_range.end() < token_count as u32)
            })
    };
    for (state_index, state) in boundary.parser_dwa.states().iter().enumerate() {
        if state
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight_in_domain(weight))
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "serialized recursive boundary parser state {state_index} has an out-of-domain final weight"
            )));
        }
        for (&label, (target, weight)) in &state.transitions {
            if *target as usize >= state_count || !weight_in_domain(weight) {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "serialized recursive boundary parser state {state_index} has an invalid transition"
                )));
            }
            if label != crate::compiler::glr::labels::DEFAULT_LABEL
                && (label < 0 || label as u32 >= recursive_parser_state_count)
            {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "serialized recursive boundary parser state {state_index} references parser state {label} outside the recursive domain"
                )));
            }
        }
    }
    if !boundary.uses_composed_tsid_coordinate
        && boundary
            .tokenizer_state_to_tsid
            .iter()
            .any(|&tsid| tsid != u32::MAX && tsid as usize >= tsid_count)
    {
        return Err(crate::GlrMaskError::Serialization(
            "serialized recursive boundary parser tokenizer-state map references an invalid TSID"
                .to_owned(),
        ));
    }
    Ok(crate::runtime::artifact::SegmentedBoundaryParser {
        parser_dwa: crate::automata::weighted_u32::dwa::DWA::new(0, 0),
        compact_parser_dwa: None,
        recursive_parser_dwa: Some(boundary.parser_dwa),
        uses_composed_tsid_coordinate: boundary.uses_composed_tsid_coordinate,
        tokenizer_state_to_tsid: boundary.tokenizer_state_to_tsid,
        internal_token_to_originals: boundary.internal_token_to_originals,
    })
}

fn restore_segmented_runtime_v23(
    constraint: &mut Constraint,
    runtime: SegmentedRuntimeArtifactV23,
    allow_empty_component_start_states: bool,
) -> crate::Result<()> {
    let SegmentedRuntimeArtifactV23 {
        materialized_static_component_parser,
        materialized_static_parser_state_domain_labels,
        components,
        segmented_mask_authoritative,
        segmented_component_union_root_dispatch,
        boundary_shards,
    } = runtime;
    restore_segmented_runtime_v22(
        constraint,
        SegmentedRuntimeArtifactV22 {
            materialized_static_component_parser,
            materialized_static_parser_state_domain_labels,
            components,
            segmented_mask_authoritative,
            segmented_component_union_root_dispatch,
            boundary_parser: None,
            boundary_terminal_trie: None,
        },
    )?;

    let global_state_count = constraint.table.num_states as usize;
    let global_terminal_count = constraint.table.num_terminals as usize;
    let component_count = constraint
        .static_dynamic_overlay
        .as_ref()
        .map_or(0, |overlay| overlay.segmented_parser_components.len());
    let mut seen_components = vec![false; component_count];
    let mut global_static = None;
    let mut global_dynamic = None;
    let mut restored_shards = Vec::new();

    for (index, shard) in boundary_shards.into_iter().enumerate() {
        let backend = match shard.backend {
            SegmentedBoundaryShardBackendV23::StaticParser(boundary) => {
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(Arc::new(
                    restore_boundary_parser_v23(constraint, boundary)?,
                ))
            }
            SegmentedBoundaryShardBackendV23::DynamicTerminalTrie(boundary) => {
                crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(Arc::new(
                    restore_boundary_terminal_trie_v22(boundary, global_terminal_count)?,
                ))
            }
            SegmentedBoundaryShardBackendV23::DynamicDirect => {
                crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
            }
        };
        match shard.scope {
            SegmentedBoundaryShardScopeV23::Global => {
                if shard.candidate_tokens.is_some() {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "global boundary shard {index} unexpectedly carries component trigger tokens"
                    )));
                }
                match backend {
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(boundary) => {
                        if global_static.replace(boundary).is_some() {
                            return Err(crate::GlrMaskError::Serialization(
                                "serialized v23 runtime contains multiple global static boundary shards"
                                    .to_owned(),
                            ));
                        }
                    }
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(boundary) => {
                        if global_dynamic.replace(boundary).is_some() {
                            return Err(crate::GlrMaskError::Serialization(
                                "serialized v23 runtime contains multiple global dynamic boundary shards"
                                    .to_owned(),
                            ));
                        }
                    }
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect => {
                        return Err(crate::GlrMaskError::Serialization(
                            "serialized v23 runtime cannot use an unscoped direct-dynamic boundary shard"
                                .to_owned(),
                        ));
                    }
                }
            }
            SegmentedBoundaryShardScopeV23::Component {
                start_component,
                start_parser_states,
                accepts_empty_stack,
            } => {
                let component_index = start_component as usize;
                if component_index >= component_count {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "boundary shard {index} references unknown component {start_component}"
                    )));
                }
                if seen_components[component_index] {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "serialized v23 runtime contains multiple boundary shards for component {start_component}"
                    )));
                }
                seen_components[component_index] = true;
                if start_parser_states.len() != global_state_count
                    && !(allow_empty_component_start_states && start_parser_states.len() == 0)
                {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "boundary shard {index} start-state set has length {} for {global_state_count} outer states",
                        start_parser_states.len(),
                    )));
                }
                if accepts_empty_stack && start_component != 0 {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "boundary shard {index} gives empty-stack ownership to non-root component {start_component}"
                    )));
                }
                let candidate_tokens = shard.candidate_tokens.map(|mut tokens| {
                    tokens.sort_unstable();
                    tokens.dedup();
                    Arc::<[u32]>::from(tokens)
                });
                restored_shards.push(crate::runtime::SegmentedBoundaryShard {
            mask_vocabulary: Default::default(),
                    start_component,
                    start_parser_states,
                    accepts_empty_stack,
                    candidate_tokens,
                    backend,
                });
            }
        }
    }
    if !restored_shards.is_empty() && (global_static.is_some() || global_dynamic.is_some()) {
        return Err(crate::GlrMaskError::Serialization(
            "serialized v23 runtime mixes global and component-scoped boundary shards".to_owned(),
        ));
    }
    let overlay = constraint
        .static_dynamic_overlay
        .get_or_insert_with(Default::default);
    for component in &mut overlay.segmented_parser_components {
        component.boundary = None;
    }
    for shard in &restored_shards {
        if let Some(component) = overlay
            .segmented_parser_components
            .get_mut(shard.start_component as usize)
        {
            component.boundary = Some(shard.clone());
        }
    }
    overlay.segmented_boundary_shards = restored_shards;
    overlay.segmented_boundary_parser = global_static;
    overlay.segmented_boundary_terminal_trie = global_dynamic;
    Ok(())
}

fn restore_segmented_runtime_v24(
    constraint: &mut Constraint,
    runtime: SegmentedRuntimeArtifactV24,
) -> crate::Result<()> {
    let SegmentedRuntimeArtifactV24 {
        materialized_static_component_parser,
        materialized_static_parser_state_domain_labels,
        components,
        segmented_parser_links,
        segmented_parser_state_offsets,
        segmented_mask_authoritative,
        segmented_component_union_root_dispatch,
        boundary_shards,
    } = runtime;

    let mut aliases_by_component = Vec::with_capacity(components.len());
    let components_v23 = components
        .into_iter()
        .map(|component| {
            aliases_by_component.push(component.global_terminal_aliases);
            SegmentedParserComponentV22 {
                constraint_artifact: component.constraint_artifact,
                tokenizer_state_offset: component.tokenizer_state_offset,
                terminal_offset: component.terminal_offset,
                root_entry_terminals: component.root_entry_terminals,
                root_disallowed_terminal: component.root_disallowed_terminal,
                global_to_local_parser_state: component.global_to_local_parser_state,
            }
        })
        .collect();

    restore_segmented_runtime_v23(
        constraint,
        SegmentedRuntimeArtifactV23 {
            materialized_static_component_parser,
            materialized_static_parser_state_domain_labels,
            components: components_v23,
            segmented_mask_authoritative,
            segmented_component_union_root_dispatch,
            boundary_shards,
        },
        !segmented_parser_links.is_empty(),
    )?;

    let global_terminal_count = constraint.table.num_terminals;
    let overlay = constraint
        .static_dynamic_overlay
        .as_mut()
        .expect("v24 segmented runtime restore must create overlay metadata");
    if aliases_by_component.len() != overlay.segmented_parser_components.len() {
        return Err(crate::GlrMaskError::Serialization(
            "serialized v24 component-alias count mismatch".to_owned(),
        ));
    }
    for (index, (component, aliases)) in overlay
        .segmented_parser_components
        .iter_mut()
        .zip(aliases_by_component)
        .enumerate()
    {
        if aliases.iter().any(|&(global, local)| {
            global >= global_terminal_count || local >= component.constraint.table.num_terminals
        }) {
            return Err(crate::GlrMaskError::Serialization(format!(
                "serialized v24 component {index} contains an invalid terminal alias"
            )));
        }
        component.global_terminal_aliases = aliases;
    }

    let links = segmented_parser_links
        .into_iter()
        .map(|link| crate::runtime::SegmentedParserLink {
            parent_component: link.parent_component,
            slot_terminal: link.slot_terminal,
            child_component: link.child_component,
            child_start: link.child_start,
            return_pop: link.return_pop,
            child_start_nullable: link.child_start_nullable,
        })
        .collect::<Vec<_>>();
    overlay.segmented_parser_links = links;
    overlay.segmented_parser_state_offsets = segmented_parser_state_offsets;

    if !overlay.segmented_parser_state_offsets.is_empty() {
        let tables = crate::runtime::SegmentedParserComponentTables::new(
            &overlay.segmented_parser_components,
        );
        crate::compiler::glr::parser::DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &overlay.segmented_parser_links,
            &overlay.segmented_parser_state_offsets,
        )
        .map_err(crate::GlrMaskError::Serialization)?;
    }

    // v24 static B was compiled in the materialized composed-table parser
    // coordinate. There is no exact state-symbol homomorphism which can, in
    // general, transport that stack language to the recursive leaf coordinate:
    // independently mapping each stack symbol loses correlations between stack
    // positions. Keep such artifacts on their exact legacy runtime instead.
    //
    // Transitional v24 artifacts produced while the recursive runtime was being
    // developed may already have cleared their component start-state bitsets.
    // Reconstruct those exactly from the still-serialized materialized -> local
    // component projections. Historical v24 artifacts already carry the same
    // full-length sets, so this is a no-op for them.
    let has_legacy_boundary = overlay.segmented_boundary_parser.is_some()
        || overlay.segmented_boundary_terminal_trie.is_some()
        || overlay.segmented_parser_components.iter().any(|component| {
            component.boundary.as_ref().is_some_and(|shard| {
                !matches!(
                    shard.backend,
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
                )
            })
        });
    if has_legacy_boundary {
        let global_state_count = constraint.table.num_states as usize;
        for component in &mut overlay.segmented_parser_components {
            let Some(shard) = component.boundary.as_mut() else {
                continue;
            };
            if shard.start_parser_states.len() != 0 {
                continue;
            }
            let mut starts = crate::ds::bitset::BitSet::new(global_state_count);
            for (global_state, &local_state) in
                component.global_to_local_parser_state.iter().enumerate()
            {
                if local_state != u32::MAX {
                    starts.set(global_state);
                }
            }
            shard.start_parser_states = starts;
        }
        overlay.segmented_boundary_shards = overlay
            .segmented_parser_components
            .iter()
            .filter_map(|component| component.boundary.clone())
            .collect();
        // Prevent a global/static legacy B from accidentally satisfying the
        // recursive-runtime shape test merely because v24 also serialized
        // linker metadata. As a loaded legacy constraint this node remains an
        // opaque materialized parser if it is later embedded in v27.
        overlay.segmented_parser_links.clear();
        overlay.segmented_parser_state_offsets.clear();
        return Ok(());
    }

    if constraint.uses_compact_segmented_parser_runtime() {
        constraint
            .static_dynamic_overlay
            .as_mut()
            .expect("compact v24 runtime requires overlay")
            .segmented_parser_state_offsets
            .clear();
        constraint.clear_recursive_legacy_boundary_start_states();
        constraint.clear_recursive_legacy_parser_state_projections();
    }
    Ok(())
}

fn restore_recursive_segmented_runtime_v27(
    constraint: &mut Constraint,
    runtime: RecursiveSegmentedRuntimeArtifactV27,
) -> crate::Result<()> {
    if !runtime.segmented_mask_authoritative {
        return Err(crate::GlrMaskError::Serialization(
            "recursive v27 segmented runtime must be mask-authoritative".to_owned(),
        ));
    }
    let recursive_compiler_table = runtime.recursive_compiler_table;
    let recursive_tokenizer_internal_tsids = runtime.recursive_tokenizer_internal_tsids;
    let global_terminal_count = constraint.table.num_terminals as usize;
    let global_tokenizer_states = constraint.tokenizer.num_states();
    // Immediate component artifacts are independent. Decode them in parallel
    // before validating their placement in the outer compiler-oracle
    // coordinates. This is especially important for nested recursive
    // compositions: a sequential loop turns load latency into the sum of every
    // descendant artifact load, while Rayon lets the component tree collapse
    // to roughly its critical path without changing the wire or runtime tree.
    let decoded_components = runtime
        .components
        .into_par_iter()
        .map(|component| {
            let child = Constraint::load_body_artifact(&component.constraint_artifact)?;
            Ok::<_, crate::GlrMaskError>((component, child))
        })
        .collect::<crate::Result<Vec<_>>>()?;
    let mut components = Vec::with_capacity(decoded_components.len());
    let mut expected_tokenizer_state_offset = 0u32;
    for (index, (component, child)) in decoded_components.into_iter().enumerate() {
        if component.root_entry_terminals.len() != global_terminal_count {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 component {index} root-entry terminal set has length {} for {global_terminal_count} outer terminals",
                component.root_entry_terminals.len(),
            )));
        }
        if component
            .terminal_offset
            .checked_add(child.table.num_terminals)
            .is_none_or(|end| end as usize > global_terminal_count)
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 component {index} terminal range lies outside outer terminal domain"
            )));
        }
        if component.tokenizer_state_offset != expected_tokenizer_state_offset {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 component {index} tokenizer-state offset is {}, expected {expected_tokenizer_state_offset}",
                component.tokenizer_state_offset,
            )));
        }
        let child_tokenizer_span = child
            .recursive_parser_layout()
            .map_err(crate::GlrMaskError::Serialization)?
            .map_or(child.tokenizer.num_states(), |layout| layout.total_tokenizer_states);
        expected_tokenizer_state_offset = expected_tokenizer_state_offset
            .checked_add(child_tokenizer_span)
            .ok_or_else(|| {
                crate::GlrMaskError::Serialization(
                    "recursive v27 tokenizer-state coordinate overflow".to_owned(),
                )
            })?;
        if component
            .root_disallowed_terminal
            .is_some_and(|terminal| terminal >= child.table.num_terminals)
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 component {index} root-disallowed terminal lies outside the component"
            )));
        }
        if component.global_terminal_aliases.iter().any(|&(global, local)| {
            global >= constraint.table.num_terminals || local >= child.table.num_terminals
        }) {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 component {index} contains an invalid terminal alias"
            )));
        }
        components.push(crate::runtime::SegmentedParserComponent {
            constraint: Arc::new(child),
            boundary: None,
            tokenizer_state_offset: component.tokenizer_state_offset,
            terminal_offset: component.terminal_offset,
            global_terminal_aliases: component.global_terminal_aliases,
            local_tsid_to_global_tsids: component.local_tsid_to_global_tsids,
            root_disallowed_terminal: component.root_disallowed_terminal,
            global_to_local_parser_state: Vec::new(),
        });
    }
    if expected_tokenizer_state_offset as usize != recursive_tokenizer_internal_tsids.len() {
        return Err(crate::GlrMaskError::Serialization(format!(
            "recursive v27 tokenizer-state relation has {} rows for {expected_tokenizer_state_offset} recursive states",
            recursive_tokenizer_internal_tsids.len(),
        )));
    }
    let links = runtime
        .segmented_parser_links
        .into_iter()
        .map(|link| crate::runtime::SegmentedParserLink {
            parent_component: link.parent_component,
            slot_terminal: link.slot_terminal,
            child_component: link.child_component,
            child_start: link.child_start,
            return_pop: link.return_pop,
            child_start_nullable: link.child_start_nullable,
        })
        .collect::<Vec<_>>();
    if components.is_empty() || links.is_empty() {
        return Err(crate::GlrMaskError::Serialization(
            "recursive v27 segmented runtime requires components and linker controls".to_owned(),
        ));
    }
    {
        let overlay = constraint
            .static_dynamic_overlay
            .get_or_insert_with(Default::default);
        overlay.segmented_parser_components = components;
        overlay.segmented_parser_links = links;
        overlay.segmented_parser_state_offsets.clear();
        overlay.segmented_mask_authoritative = true;
        overlay.segmented_static_baseline = false;
        overlay.segmented_component_union_root_dispatch.clear();
        overlay.segmented_boundary_shards.clear();
        overlay.segmented_boundary_parser = None;
        overlay.segmented_boundary_terminal_trie = None;
        overlay
            .recursive_compiler_table
            .set(Arc::from(recursive_compiler_table.into_boxed_slice()))
            .map_err(|_| {
                crate::GlrMaskError::Serialization(
                    "recursive compiler table was initialized twice".to_owned(),
                )
            })?;
    }
    let layout = constraint
        .recursive_parser_layout_for_pending_root()
        .map_err(crate::GlrMaskError::Serialization)?
        .ok_or_else(|| {
            crate::GlrMaskError::Serialization(
                "recursive v27 runtime failed to derive its parser layout".to_owned(),
            )
        })?;
    constraint
        .install_recursive_tokenizer_internal_tsids(recursive_tokenizer_internal_tsids)
        .map_err(crate::GlrMaskError::Serialization)?;
    let component_count = constraint
        .static_dynamic_overlay
        .as_ref()
        .expect("recursive v27 overlay exists")
        .segmented_parser_components
        .len();
    let mut seen = vec![false; component_count];
    let mut restored_shards = Vec::new();
    for (index, shard) in runtime.boundary_shards.into_iter().enumerate() {
        let component_index = shard.start_component as usize;
        if component_index >= component_count {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 boundary shard {index} references unknown component {}",
                shard.start_component,
            )));
        }
        if seen[component_index] {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 runtime contains multiple boundary shards for component {}",
                shard.start_component,
            )));
        }
        seen[component_index] = true;
        if shard.accepts_empty_stack && shard.start_component != 0 {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 boundary shard {index} gives empty-stack ownership to non-root component {}",
                shard.start_component,
            )));
        }
        let backend = match shard.backend {
            RecursiveSegmentedBoundaryShardBackendV27::StaticParser(boundary) => {
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(Arc::new(
                    restore_recursive_boundary_parser_v27(
                        constraint,
                        boundary,
                        layout.total_states,
                        layout.total_tokenizer_states,
                    )?,
                ))
            }
            RecursiveSegmentedBoundaryShardBackendV27::DynamicDirect => {
                crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
            }
        };
        let candidate_tokens = shard.candidate_tokens.map(|mut tokens| {
            tokens.sort_unstable();
            tokens.dedup();
            Arc::<[u32]>::from(tokens)
        });
        restored_shards.push(crate::runtime::SegmentedBoundaryShard {
            mask_vocabulary: Default::default(),
            start_component: shard.start_component,
            start_parser_states: crate::ds::bitset::BitSet::new(0),
            accepts_empty_stack: shard.accepts_empty_stack,
            candidate_tokens,
            backend,
        });
    }
    let overlay = constraint
        .static_dynamic_overlay
        .as_mut()
        .expect("recursive v27 overlay exists");
    for component in &mut overlay.segmented_parser_components {
        component.boundary = None;
    }
    for shard in &restored_shards {
        overlay.segmented_parser_components[shard.start_component as usize].boundary =
            Some(shard.clone());
    }
    overlay.segmented_boundary_shards = restored_shards;
    if !constraint.uses_compact_segmented_parser_runtime() {
        return Err(crate::GlrMaskError::Serialization(
            "recursive v27 segmented runtime did not restore a provider-native parser"
                .to_owned(),
        ));
    }
    Ok(())
}

fn restore_segmented_runtime_v27(
    constraint: &mut Constraint,
    runtime: SegmentedRuntimeArtifactV27,
) -> crate::Result<()> {
    match runtime {
        SegmentedRuntimeArtifactV27::Recursive(runtime) => {
            restore_recursive_segmented_runtime_v27(constraint, runtime)
        }
        SegmentedRuntimeArtifactV27::LegacyV24(runtime) => {
            restore_segmented_runtime_v24(constraint, runtime)
        }
    }
}

fn restore_segmented_runtime_v20(
    constraint: &mut Constraint,
    runtime: SegmentedRuntimeArtifactV20,
) -> crate::Result<()> {
    if let Some(parser_dwa) = runtime.materialized_component_parser {
        constraint.parser_dwa = parser_dwa;
        constraint.packed_parser_dwa = None;
        constraint.parser_state_domain_labels = runtime.materialized_parser_state_domain_labels;
    }
    let overlay = constraint
        .static_dynamic_overlay
        .get_or_insert_with(Default::default);
    // Persist segmented A as one ordinary deterministic parser DWA. The live
    // zero-copy component collection remains an in-memory optimization and is
    // deliberately not recursively embedded in the artifact.
    overlay.segmented_parser_components.clear();
    overlay.segmented_component_union_root_dispatch.clear();
    overlay.segmented_boundary_parser = runtime.boundary_parser.map(Arc::new);
    overlay.segmented_boundary_terminal_trie = runtime.boundary_terminal_trie.map(Arc::new);
    overlay.segmented_boundary_shards.clear();
    Ok(())
}

enum DecodedParserDwa {
    Materialized(
        crate::automata::weighted::dwa::DWA,
        Option<crate::automata::weighted::dwa::PackedDwaTokenSetInventory>,
    ),
    Packed(std::sync::Arc<crate::automata::weighted::dwa::PackedRuntimeDwa>),
}

fn v14_sections(payload: &[u8]) -> Result<(&[u8], &[u8], &[u8], &[u8]), String> {
    if payload.len() < V14_SECTION_HEADER_LEN || !payload.starts_with(&V14_SECTION_MAGIC) {
        return Err("invalid v14 constraint section header".to_owned());
    }
    let mut pos = V14_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v14 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value).map_err(|_| "v14 section length does not fit this platform".to_owned())
    };
    let weight_len = take_len()?;
    let dwa_len = take_len()?;
    let table_len = take_len()?;
    let core_len = take_len()?;
    let total = V14_SECTION_HEADER_LEN
        .checked_add(weight_len)
        .and_then(|value| value.checked_add(dwa_len))
        .and_then(|value| value.checked_add(table_len))
        .and_then(|value| value.checked_add(core_len))
        .ok_or_else(|| "overflowing v14 section lengths".to_owned())?;
    if total != payload.len() {
        return Err("invalid v14 constraint section lengths".to_owned());
    }
    let weight_start = V14_SECTION_HEADER_LEN;
    let dwa_start = weight_start + weight_len;
    let table_start = dwa_start + dwa_len;
    let core_start = table_start + table_len;
    Ok((
        &payload[weight_start..dwa_start],
        &payload[dwa_start..table_start],
        &payload[table_start..core_start],
        &payload[core_start..],
    ))
}

fn v15_sections(
    payload: &[u8],
) -> Result<(&[u8], &[u8], &[u8], &[u8], &[u8]), String> {
    if payload.len() < V15_SECTION_HEADER_LEN || !payload.starts_with(&V15_SECTION_MAGIC) {
        return Err("invalid v15 constraint section header".to_owned());
    }
    let mut pos = V15_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v15 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v15 section length does not fit this platform".to_owned())
    };
    let weight_len = take_len()?;
    let dwa_len = take_len()?;
    let table_len = take_len()?;
    let core_len = take_len()?;
    let runtime_len = take_len()?;
    let total = V15_SECTION_HEADER_LEN
        .checked_add(weight_len)
        .and_then(|value| value.checked_add(dwa_len))
        .and_then(|value| value.checked_add(table_len))
        .and_then(|value| value.checked_add(core_len))
        .and_then(|value| value.checked_add(runtime_len))
        .ok_or_else(|| "v15 constraint section lengths overflow".to_owned())?;
    if total != payload.len() {
        return Err("invalid v15 constraint section lengths".to_owned());
    }
    let mut pos = V15_SECTION_HEADER_LEN;
    let weight = &payload[pos..pos + weight_len];
    pos += weight_len;
    let dwa = &payload[pos..pos + dwa_len];
    pos += dwa_len;
    let table = &payload[pos..pos + table_len];
    pos += table_len;
    let core = &payload[pos..pos + core_len];
    pos += core_len;
    let runtime = &payload[pos..pos + runtime_len];
    Ok((weight, dwa, table, core, runtime))
}

fn v16_sections(
    payload: &[u8],
) -> Result<(&[u8], &[u8], &[u8], &[u8], &[u8]), String> {
    if payload.len() < V16_SECTION_HEADER_LEN || !payload.starts_with(&V16_SECTION_MAGIC) {
        return Err("invalid v16 constraint section header".to_owned());
    }
    let mut pos = V16_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v16 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v16 section length does not fit this platform".to_owned())
    };
    let weight_len = take_len()?;
    let dwa_len = take_len()?;
    let table_len = take_len()?;
    let core_len = take_len()?;
    let runtime_len = take_len()?;
    let total = V16_SECTION_HEADER_LEN
        .checked_add(weight_len)
        .and_then(|value| value.checked_add(dwa_len))
        .and_then(|value| value.checked_add(table_len))
        .and_then(|value| value.checked_add(core_len))
        .and_then(|value| value.checked_add(runtime_len))
        .ok_or_else(|| "v16 constraint section lengths overflow".to_owned())?;
    if total != payload.len() {
        return Err("invalid v16 constraint section lengths".to_owned());
    }
    let mut pos = V16_SECTION_HEADER_LEN;
    let weight = &payload[pos..pos + weight_len];
    pos += weight_len;
    let dwa = &payload[pos..pos + dwa_len];
    pos += dwa_len;
    let table = &payload[pos..pos + table_len];
    pos += table_len;
    let core = &payload[pos..pos + core_len];
    pos += core_len;
    let runtime = &payload[pos..pos + runtime_len];
    Ok((weight, dwa, table, core, runtime))
}

fn v17_sections(
    payload: &[u8],
) -> Result<(&[u8], &[u8], &[u8], &[u8], &[u8], &[u8]), String> {
    if payload.len() < V17_SECTION_HEADER_LEN || !payload.starts_with(&V17_SECTION_MAGIC) {
        return Err("invalid v17 constraint section header".to_owned());
    }
    let mut pos = V17_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v17 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v17 section length does not fit this platform".to_owned())
    };
    let weight_len = take_len()?;
    let dwa_len = take_len()?;
    let table_len = take_len()?;
    let core_len = take_len()?;
    let runtime_len = take_len()?;
    let token_bytes_len = take_len()?;
    let total = V17_SECTION_HEADER_LEN
        .checked_add(weight_len)
        .and_then(|value| value.checked_add(dwa_len))
        .and_then(|value| value.checked_add(table_len))
        .and_then(|value| value.checked_add(core_len))
        .and_then(|value| value.checked_add(runtime_len))
        .and_then(|value| value.checked_add(token_bytes_len))
        .ok_or_else(|| "v17 constraint section lengths overflow".to_owned())?;
    if total != payload.len() {
        return Err("invalid v17 constraint section lengths".to_owned());
    }
    let mut pos = V17_SECTION_HEADER_LEN;
    let weight = &payload[pos..pos + weight_len];
    pos += weight_len;
    let dwa = &payload[pos..pos + dwa_len];
    pos += dwa_len;
    let table = &payload[pos..pos + table_len];
    pos += table_len;
    let core = &payload[pos..pos + core_len];
    pos += core_len;
    let runtime = &payload[pos..pos + runtime_len];
    pos += runtime_len;
    let token_bytes = &payload[pos..pos + token_bytes_len];
    Ok((weight, dwa, table, core, runtime, token_bytes))
}

fn v18_sections(
    payload: &[u8],
) -> Result<(&[u8], &[u8], &[u8], &[u8], &[u8], &[u8], &[u8], &[u8], &[u8]), String> {
    if payload.len() < V18_SECTION_HEADER_LEN || !payload.starts_with(&V18_SECTION_MAGIC) {
        return Err("invalid v18 constraint section header".to_owned());
    }
    let mut pos = V18_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v18 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v18 section length does not fit this platform".to_owned())
    };
    let lengths = [
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
    ];
    let total = lengths.iter().try_fold(V18_SECTION_HEADER_LEN, |sum, &len| {
        sum.checked_add(len)
            .ok_or_else(|| "v18 constraint section lengths overflow".to_owned())
    })?;
    if total != payload.len() {
        return Err("invalid v18 constraint section lengths".to_owned());
    }
    let mut pos = V18_SECTION_HEADER_LEN;
    let mut next = |len: usize| {
        let section = &payload[pos..pos + len];
        pos += len;
        section
    };
    Ok((
        next(lengths[0]),
        next(lengths[1]),
        next(lengths[2]),
        next(lengths[3]),
        next(lengths[4]),
        next(lengths[5]),
        next(lengths[6]),
        next(lengths[7]),
        next(lengths[8]),
    ))
}

fn v19_sections(
    payload: &[u8],
) -> Result<
    (
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
    ),
    String,
> {
    if payload.len() < V19_SECTION_HEADER_LEN || !payload.starts_with(&V19_SECTION_MAGIC) {
        return Err("invalid v19 constraint section header".to_owned());
    }
    let mut pos = V19_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v19 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v19 section length does not fit this platform".to_owned())
    };
    let lengths = [
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
    ];
    let total = lengths.iter().try_fold(V19_SECTION_HEADER_LEN, |sum, &len| {
        sum.checked_add(len)
            .ok_or_else(|| "v19 constraint section lengths overflow".to_owned())
    })?;
    if total != payload.len() {
        return Err("invalid v19 constraint section lengths".to_owned());
    }
    let mut pos = V19_SECTION_HEADER_LEN;
    let mut next = |len: usize| {
        let section = &payload[pos..pos + len];
        pos += len;
        section
    };
    Ok((
        next(lengths[0]),
        next(lengths[1]),
        next(lengths[2]),
        next(lengths[3]),
        next(lengths[4]),
        next(lengths[5]),
        next(lengths[6]),
        next(lengths[7]),
        next(lengths[8]),
        next(lengths[9]),
    ))
}

fn v20_sections(
    payload: &[u8],
) -> Result<
    (
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
    ),
    String,
> {
    if payload.len() < V20_SECTION_HEADER_LEN || !payload.starts_with(&V20_SECTION_MAGIC) {
        return Err("invalid v20 constraint section header".to_owned());
    }
    let mut pos = V20_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v20 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v20 section length does not fit this platform".to_owned())
    };
    let lengths = [
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
    ];
    let total = lengths.iter().try_fold(V20_SECTION_HEADER_LEN, |sum, &len| {
        sum.checked_add(len)
            .ok_or_else(|| "v20 constraint section lengths overflow".to_owned())
    })?;
    if total != payload.len() {
        return Err("invalid v20 constraint section lengths".to_owned());
    }
    let mut pos = V20_SECTION_HEADER_LEN;
    let mut next = |len: usize| {
        let section = &payload[pos..pos + len];
        pos += len;
        section
    };
    Ok((
        next(lengths[0]),
        next(lengths[1]),
        next(lengths[2]),
        next(lengths[3]),
        next(lengths[4]),
        next(lengths[5]),
        next(lengths[6]),
        next(lengths[7]),
        next(lengths[8]),
        next(lengths[9]),
        next(lengths[10]),
    ))
}

fn v24_sections(
    payload: &[u8],
) -> Result<
    (
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
    ),
    String,
> {
    if payload.len() < V24_SECTION_HEADER_LEN || !payload.starts_with(&V24_SECTION_MAGIC) {
        return Err("invalid v24 constraint section header".to_owned());
    }
    let mut pos = V24_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v23 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v23 section length does not fit this platform".to_owned())
    };
    let lengths = [
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
    ];
    let total = lengths.iter().try_fold(V24_SECTION_HEADER_LEN, |sum, &len| {
        sum.checked_add(len)
            .ok_or_else(|| "v24 constraint section lengths overflow".to_owned())
    })?;
    if total != payload.len() {
        return Err("invalid v24 constraint section lengths".to_owned());
    }
    let mut pos = V24_SECTION_HEADER_LEN;
    let mut next = |len: usize| {
        let section = &payload[pos..pos + len];
        pos += len;
        section
    };
    Ok((
        next(lengths[0]),
        next(lengths[1]),
        next(lengths[2]),
        next(lengths[3]),
        next(lengths[4]),
        next(lengths[5]),
        next(lengths[6]),
        next(lengths[7]),
        next(lengths[8]),
        next(lengths[9]),
        next(lengths[10]),
    ))
}

fn v28_sections(
    payload: &[u8],
) -> Result<
    (
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
    ),
    String,
> {
    if payload.len() < V28_SECTION_HEADER_LEN || !payload.starts_with(&V28_SECTION_MAGIC) {
        return Err("invalid v28 constraint section header".to_owned());
    }
    let mut pos = V28_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v27 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v27 section length does not fit this platform".to_owned())
    };
    let lengths = [
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
    ];
    let total = lengths.iter().try_fold(V28_SECTION_HEADER_LEN, |sum, &len| {
        sum.checked_add(len)
            .ok_or_else(|| "v28 constraint section lengths overflow".to_owned())
    })?;
    if total != payload.len() {
        return Err("invalid v28 constraint section lengths".to_owned());
    }
    let mut pos = V28_SECTION_HEADER_LEN;
    let mut next = |len: usize| {
        let section = &payload[pos..pos + len];
        pos += len;
        section
    };
    Ok((
        next(lengths[0]),
        next(lengths[1]),
        next(lengths[2]),
        next(lengths[3]),
        next(lengths[4]),
        next(lengths[5]),
        next(lengths[6]),
        next(lengths[7]),
        next(lengths[8]),
        next(lengths[9]),
        next(lengths[10]),
    ))
}


fn v27_sections(
    payload: &[u8],
) -> Result<
    (
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
    ),
    String,
> {
    if payload.len() < V27_SECTION_HEADER_LEN || !payload.starts_with(&V27_SECTION_MAGIC) {
        return Err("invalid v27 constraint section header".to_owned());
    }
    let mut pos = V27_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v27 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v27 section length does not fit this platform".to_owned())
    };
    let lengths = [
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
    ];
    let total = lengths.iter().try_fold(V27_SECTION_HEADER_LEN, |sum, &len| {
        sum.checked_add(len)
            .ok_or_else(|| "v27 constraint section lengths overflow".to_owned())
    })?;
    if total != payload.len() {
        return Err("invalid v27 constraint section lengths".to_owned());
    }
    let mut pos = V27_SECTION_HEADER_LEN;
    let mut next = |len: usize| {
        let section = &payload[pos..pos + len];
        pos += len;
        section
    };
    Ok((
        next(lengths[0]),
        next(lengths[1]),
        next(lengths[2]),
        next(lengths[3]),
        next(lengths[4]),
        next(lengths[5]),
        next(lengths[6]),
        next(lengths[7]),
        next(lengths[8]),
        next(lengths[9]),
        next(lengths[10]),
    ))
}

fn eleven_section_payload<'a>(
    payload: &'a [u8],
    magic: &[u8; 4],
    header_len: usize,
    version_label: &str,
) -> Result<
    (
        &'a [u8],
        &'a [u8],
        &'a [u8],
        &'a [u8],
        &'a [u8],
        &'a [u8],
        &'a [u8],
        &'a [u8],
        &'a [u8],
        &'a [u8],
        &'a [u8],
    ),
    String,
> {
    if payload.len() < header_len || !payload.starts_with(magic) {
        return Err(format!("invalid {version_label} constraint section header"));
    }
    let mut pos = magic.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("constraint section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| format!("{version_label} section length does not fit this platform"))
    };
    let lengths = [
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
    ];
    let total = lengths.iter().try_fold(header_len, |sum, &len| {
        sum.checked_add(len)
            .ok_or_else(|| format!("{version_label} constraint section lengths overflow"))
    })?;
    if total != payload.len() {
        return Err(format!("invalid {version_label} constraint section lengths"));
    }
    let mut pos = header_len;
    let mut next = |len: usize| {
        let section = &payload[pos..pos + len];
        pos += len;
        section
    };
    Ok((
        next(lengths[0]),
        next(lengths[1]),
        next(lengths[2]),
        next(lengths[3]),
        next(lengths[4]),
        next(lengths[5]),
        next(lengths[6]),
        next(lengths[7]),
        next(lengths[8]),
        next(lengths[9]),
        next(lengths[10]),
    ))
}

fn v30_sections(
    payload: &[u8],
) -> Result<
    (
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
    ),
    String,
> {
    eleven_section_payload(payload, &V30_SECTION_MAGIC, V30_SECTION_HEADER_LEN, "v30")
}

fn v29_sections(
    payload: &[u8],
) -> Result<
    (
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
    ),
    String,
> {
    eleven_section_payload(payload, &V29_SECTION_MAGIC, V29_SECTION_HEADER_LEN, "v29")
}



fn v23_sections(
    payload: &[u8],
) -> Result<
    (
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
    ),
    String,
> {
    if payload.len() < V23_SECTION_HEADER_LEN || !payload.starts_with(&V23_SECTION_MAGIC) {
        return Err("invalid v23 constraint section header".to_owned());
    }
    let mut pos = V23_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v23 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v23 section length does not fit this platform".to_owned())
    };
    let lengths = [
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
    ];
    let total = lengths.iter().try_fold(V23_SECTION_HEADER_LEN, |sum, &len| {
        sum.checked_add(len)
            .ok_or_else(|| "v23 constraint section lengths overflow".to_owned())
    })?;
    if total != payload.len() {
        return Err("invalid v23 constraint section lengths".to_owned());
    }
    let mut pos = V23_SECTION_HEADER_LEN;
    let mut next = |len: usize| {
        let section = &payload[pos..pos + len];
        pos += len;
        section
    };
    Ok((
        next(lengths[0]),
        next(lengths[1]),
        next(lengths[2]),
        next(lengths[3]),
        next(lengths[4]),
        next(lengths[5]),
        next(lengths[6]),
        next(lengths[7]),
        next(lengths[8]),
        next(lengths[9]),
        next(lengths[10]),
    ))
}


fn v22_sections(
    payload: &[u8],
) -> Result<
    (
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
    ),
    String,
> {
    if payload.len() < V22_SECTION_HEADER_LEN || !payload.starts_with(&V22_SECTION_MAGIC) {
        return Err("invalid v22 constraint section header".to_owned());
    }
    let mut pos = V22_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v22 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v22 section length does not fit this platform".to_owned())
    };
    let lengths = [
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
    ];
    let total = lengths.iter().try_fold(V22_SECTION_HEADER_LEN, |sum, &len| {
        sum.checked_add(len)
            .ok_or_else(|| "v22 constraint section lengths overflow".to_owned())
    })?;
    if total != payload.len() {
        return Err("invalid v22 constraint section lengths".to_owned());
    }
    let mut pos = V22_SECTION_HEADER_LEN;
    let mut next = |len: usize| {
        let section = &payload[pos..pos + len];
        pos += len;
        section
    };
    Ok((
        next(lengths[0]),
        next(lengths[1]),
        next(lengths[2]),
        next(lengths[3]),
        next(lengths[4]),
        next(lengths[5]),
        next(lengths[6]),
        next(lengths[7]),
        next(lengths[8]),
        next(lengths[9]),
        next(lengths[10]),
    ))
}

fn v21_sections(
    payload: &[u8],
) -> Result<
    (
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
        &[u8],
    ),
    String,
> {
    if payload.len() < V22_SECTION_HEADER_LEN || !payload.starts_with(&V22_SECTION_MAGIC) {
        return Err("invalid v21 constraint section header".to_owned());
    }
    let mut pos = V22_SECTION_MAGIC.len();
    let mut take_len = || {
        let end = pos + 8;
        let value = u64::from_le_bytes(
            payload[pos..end]
                .try_into()
                .expect("v21 section length has fixed width"),
        );
        pos = end;
        usize::try_from(value)
            .map_err(|_| "v21 section length does not fit this platform".to_owned())
    };
    let lengths = [
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
        take_len()?,
    ];
    let total = lengths.iter().try_fold(V22_SECTION_HEADER_LEN, |sum, &len| {
        sum.checked_add(len)
            .ok_or_else(|| "v21 constraint section lengths overflow".to_owned())
    })?;
    if total != payload.len() {
        return Err("invalid v21 constraint section lengths".to_owned());
    }
    let mut pos = V22_SECTION_HEADER_LEN;
    let mut next = |len: usize| {
        let section = &payload[pos..pos + len];
        pos += len;
        section
    };
    Ok((
        next(lengths[0]),
        next(lengths[1]),
        next(lengths[2]),
        next(lengths[3]),
        next(lengths[4]),
        next(lengths[5]),
        next(lengths[6]),
        next(lengths[7]),
        next(lengths[8]),
        next(lengths[9]),
        next(lengths[10]),
    ))
}

fn constraint_serialized_weight_pool_with_ids(
    constraint: &Constraint,
) -> (Vec<Weight>, Vec<u32>, usize) {
    let mut by_ptr = FxHashMap::<usize, u32>::default();
    let mut weights = Vec::new();
    let mut ids = Vec::new();
    let mut total_ranges = 0usize;
    let mut push = |weight: &Weight| {
        let key = weight.ptr_key();
        let id = if let Some(&id) = by_ptr.get(&key) {
            id
        } else {
            let id = weights.len() as u32;
            by_ptr.insert(key, id);
            if !weight.is_full() {
                total_ranges = total_ranges.saturating_add(weight.num_ranges());
            }
            weights.push(weight.clone());
            id
        };
        ids.push(id);
    };

    for weight in constraint.parser_top_accept.values() {
        push(weight);
    }
    for parts in constraint.parser_top_accept_parts.values() {
        for weight in parts {
            push(weight);
        }
    }
    for weight in constraint.direct_regular_l1_complete_by_terminal.values() {
        push(weight);
    }
    for weight in constraint.possible_matches.values() {
        push(weight);
    }
    (weights, ids, total_ranges)
}

fn constraint_serialized_weight_pool(constraint: &Constraint) -> Vec<Weight> {
    constraint_serialized_weight_pool_with_ids(constraint).0
}

/// IDs in the same field/map order used by ConstraintSerde. Structural Weight
/// values are placeholders when a packed non-DWA pool is present; interning
/// those empty values would collapse distinct references to the same pool ID.
fn packed_constraint_serialized_weight_ids(constraint: &Constraint) -> Option<Vec<u32>> {
    let packed = constraint.packed_non_dwa_weights.as_ref()?;
    let mut ids = Vec::new();
    for key in constraint.parser_top_accept.keys() {
        ids.push(packed.parser_top_accept[key]);
    }
    for (key, parts) in &constraint.parser_top_accept_parts {
        let packed_parts = &packed.parser_top_accept_parts[key];
        assert_eq!(parts.len(), packed_parts.len(), "packed acceptance-part count mismatch");
        ids.extend_from_slice(packed_parts);
    }
    for key in constraint.direct_regular_l1_complete_by_terminal.keys() {
        ids.push(packed.direct_regular_l1_complete_by_terminal[key]);
    }
    for key in constraint.possible_matches.keys() {
        ids.push(packed.possible_matches[key]);
    }
    Some(ids)
}

fn constraint_serialized_weight_ranges_at_least(
    constraint: &Constraint,
    min_weight_ranges: usize,
) -> bool {
    if min_weight_ranges == 0 {
        return true;
    }
    let mut seen = FxHashSet::<usize>::default();
    let mut total_ranges = 0usize;
    let mut crosses_threshold = |weight: &Weight| {
        if !seen.insert(weight.ptr_key()) || weight.is_full() {
            return false;
        }
        total_ranges = total_ranges.saturating_add(weight.num_ranges());
        total_ranges >= min_weight_ranges
    };

    for weight in constraint.parser_top_accept.values() {
        if crosses_threshold(weight) {
            return true;
        }
    }
    for parts in constraint.parser_top_accept_parts.values() {
        for weight in parts {
            if crosses_threshold(weight) {
                return true;
            }
        }
    }
    for weight in constraint.direct_regular_l1_complete_by_terminal.values() {
        if crosses_threshold(weight) {
            return true;
        }
    }
    for weight in constraint.possible_matches.values() {
        if crosses_threshold(weight) {
            return true;
        }
    }
    false
}

fn compact_non_dwa_weight_runtime_if_at_least(
    constraint: &mut Constraint,
    min_weight_ranges: usize,
) -> bool {
    if constraint.packed_non_dwa_weights.is_some() {
        return false;
    }
    if !constraint_serialized_weight_ranges_at_least(constraint, min_weight_ranges) {
        return false;
    }
    let (weights, ids, _) = constraint_serialized_weight_pool_with_ids(constraint);
    let wire = crate::ds::weight::pack_pooled_weights(&weights);
    let pool = crate::ds::weight::PackedRuntimeWeightPool::from_packed_bytes(&wire)
        .expect("fresh packed non-DWA Weight runtime should decode");
    attach_packed_non_dwa_weights(constraint, std::sync::Arc::new(pool), ids)
        .expect("fresh packed non-DWA Weight ids should match runtime maps");
    true
}

pub(crate) fn compact_large_non_dwa_weight_runtime(constraint: &mut Constraint) -> bool {
    const MIN_WEIGHT_RANGES: usize = 100_000;
    compact_non_dwa_weight_runtime_if_at_least(constraint, MIN_WEIGHT_RANGES)
}

fn attach_packed_non_dwa_weights(
    constraint: &mut Constraint,
    pool: std::sync::Arc<crate::ds::weight::PackedRuntimeWeightPool>,
    ids: Vec<u32>,
) -> Result<(), String> {
    let top_len = constraint.parser_top_accept.len();
    let parts_len = constraint
        .parser_top_accept_parts
        .values()
        .map(Vec::len)
        .sum::<usize>();
    let direct_len = constraint.direct_regular_l1_complete_by_terminal.len();
    let possible_len = constraint.possible_matches.len();
    let expected = top_len
        .checked_add(parts_len)
        .and_then(|value| value.checked_add(direct_len))
        .and_then(|value| value.checked_add(possible_len))
        .ok_or_else(|| "packed Weight id count overflow".to_owned())?;
    if ids.len() != expected {
        return Err(format!(
            "packed Weight id count mismatch: expected {expected}, found {}",
            ids.len(),
        ));
    }

    let (top_ids, rest) = ids.split_at(top_len);
    let (part_ids, rest) = rest.split_at(parts_len);
    let (direct_ids, possible_ids) = rest.split_at(direct_len);

    let build_small = || {
        let parser_top_accept = constraint
            .parser_top_accept
            .keys()
            .copied()
            .zip(top_ids.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let mut part_pos = 0usize;
        let parser_top_accept_parts = constraint
            .parser_top_accept_parts
            .iter()
            .map(|(&label, parts)| {
                let end = part_pos + parts.len();
                let ids = part_ids[part_pos..end].to_vec();
                part_pos = end;
                (label, ids)
            })
            .collect::<BTreeMap<_, _>>();
        debug_assert_eq!(part_pos, part_ids.len());
        (parser_top_accept, parser_top_accept_parts)
    };
    let build_direct = || {
        constraint
            .direct_regular_l1_complete_by_terminal
            .keys()
            .copied()
            .zip(direct_ids.iter().copied())
            .collect::<BTreeMap<_, _>>()
    };
    let build_possible = || {
        constraint
            .possible_matches
            .keys()
            .copied()
            .zip(possible_ids.iter().copied())
            .collect::<BTreeMap<_, _>>()
    };
    let ((parser_top_accept, parser_top_accept_parts), (direct_regular_l1_complete_by_terminal, possible_matches)) =
        if expected >= 1_024 && rayon::current_num_threads() > 1 {
            rayon::join(build_small, || rayon::join(build_direct, build_possible))
        } else {
            (build_small(), (build_direct(), build_possible()))
        };

    constraint.packed_non_dwa_weights = Some(std::sync::Arc::new(
        crate::runtime::artifact::PackedNonDwaWeights {
            pool,
            parser_top_accept,
            parser_top_accept_parts,
            direct_regular_l1_complete_by_terminal,
            possible_matches,
        },
    ));
    Ok(())
}

fn invert_original_token_map(
    original_to_internal: &[u32],
    expected_group_count: usize,
) -> Result<Vec<Vec<u32>>, String> {
    let group_count = if expected_group_count != 0 {
        expected_group_count
    } else {
        let Some(max_internal) = original_to_internal
            .iter()
            .copied()
            .filter(|&internal| internal != u32::MAX)
            .max()
        else {
            return Ok(Vec::new());
        };
        usize::try_from(max_internal)
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| "internal token vocabulary is too large".to_owned())?
    };
    let mut counts = vec![0usize; group_count];
    for &internal in original_to_internal {
        if internal == u32::MAX {
            continue;
        }
        let Some(count) = counts.get_mut(internal as usize) else {
            return Err("original-token map contains an out-of-range internal token".to_owned());
        };
        *count += 1;
    }
    let mut groups = counts
        .into_iter()
        .map(Vec::<u32>::with_capacity)
        .collect::<Vec<_>>();
    for (original, &internal) in original_to_internal.iter().enumerate() {
        if internal == u32::MAX {
            continue;
        }
        let original = u32::try_from(original)
            .map_err(|_| "original token id exceeds u32".to_owned())?;
        groups[internal as usize].push(original);
    }
    Ok(groups)
}

fn envelope(version: u16, payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(CONSTRAINT_HEADER_LEN + payload.len());
    bytes.extend_from_slice(&CONSTRAINT_MAGIC);
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

impl Constraint {
    /// Materialize and retain the canonical current-format artifact once a
    /// compiler-owned constraint has reached its final serialized semantics.
    /// Subsequent `save()` calls then use the same bulk-copy path as an
    /// unchanged loaded constraint instead of re-encoding every section.
    pub(crate) fn cache_serialized_artifact_for_save(&mut self) {
        if self.serialized_artifact_cache.is_some() {
            return;
        }
        let bytes = self.save_body();
        self.serialized_artifact_cache = Some(std::sync::Arc::new(bytes));
    }

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
        let (input, _) = crate::compiler::boundary_precomputed_completion::split_envelope(blob.as_slice())?;
        validate_composition_metadata_wire(input)?;
        if input.starts_with(&COMPOSITION_METADATA_SPLIT_MAGIC)
            || input.starts_with(&PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC)
            || input.starts_with(&PREVIOUS_PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC)
        {
            let parts = split_composition_metadata_parts(input)?;
            let cache_raw = decode_composition_metadata_part(
                parts.cache_wire, parts.cache_raw_len, parts.cache_compressed,
            )?;
            // This vector is the first field of ConstraintCompositionCacheMetadata
            // in every split format. Deserialize that field with a bounded slice
            // reader, leaving its unrelated characterization vector untouched.
            let templates = bincode::deserialize_from(cache_raw.as_ref())
                .map_err(|error| error.to_string())?;
            Ok(Cow::Owned(templates))
        } else {
            Ok(Cow::Owned(decode_composition_metadata(input)?
                .composition_parser_templates_by_terminal))
        }
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

    /// Serialize this compiled constraint to a versioned binary artifact.
    ///
    /// Current artifacts use a compact sectioned representation and retain
    /// runtime-native sections where doing so materially reduces load latency.
    pub fn save(&self) -> Vec<u8> {
        let exact_only_token_ids = self
            .late_bind_vocab
            .get()
            .map(|vocab| vocab.exact_only_token_ids().collect::<Vec<_>>())
            .unwrap_or_default();
        if self.end_tokens.is_empty() && exact_only_token_ids.is_empty() {
            return self.save_body();
        }
        let body = self.save_body();
        let mut bytes = Vec::with_capacity(
            16 + (self.end_tokens.len() + exact_only_token_ids.len()) * 4 + body.len(),
        );
        bytes.extend_from_slice(ROOT_POLICY_MAGIC);
        bytes.extend_from_slice(&(self.end_tokens.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(exact_only_token_ids.len() as u32).to_le_bytes());
        for &id in self.end_tokens.iter() { bytes.extend_from_slice(&id.to_le_bytes()); }
        for id in exact_only_token_ids { bytes.extend_from_slice(&id.to_le_bytes()); }
        bytes.extend_from_slice(&body);
        bytes
    }

    pub(crate) fn save_body(&self) -> Vec<u8> {
        let artifact_version = if self.has_template_parser() { TEMPLATE_CONSTRAINT_VERSION } else { CONSTRAINT_VERSION };
        if let Some(bytes) = &self.serialized_artifact_cache {
            return clone_serialized_artifact(bytes.as_slice());
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        if std::env::var_os("GLRMASK_PROFILE_CACHE_ARTIFACT").is_some() {
            let started = std::time::Instant::now();
            let cache_sizes = [
                ("word_sparse", bincode::serialized_size(&self.word_group_sparse_masks).unwrap_or(0)),
                ("word_prefix", bincode::serialized_size(&self.word_group_prefix_buf_masks).unwrap_or(0)),
                ("word_sparse_prefix", bincode::serialized_size(&self.word_group_sparse_prefix_entries).unwrap_or(0)),
                ("pair", bincode::serialized_size(&self.pair_word_group_buf_masks).unwrap_or(0)),
                ("quad", bincode::serialized_size(&self.quad_word_group_buf_masks).unwrap_or(0)),
                ("super", bincode::serialized_size(&self.super_word_group_buf_masks).unwrap_or(0)),
                ("mega", bincode::serialized_size(&self.mega_word_group_buf_masks).unwrap_or(0)),
                ("giga", bincode::serialized_size(&self.giga_word_group_buf_masks).unwrap_or(0)),
                ("all", bincode::serialized_size(&self.all_tokens_buf_mask).unwrap_or(0)),
                ("heavy", bincode::serialized_size(&self.heavy_token_dense_masks).unwrap_or(0)),
                ("flat", bincode::serialized_size(&self.internal_token_buf_flat).unwrap_or(0)),
                ("offsets", bincode::serialized_size(&self.internal_token_buf_offsets).unwrap_or(0)),
                ("op_costs", bincode::serialized_size(&self.internal_token_buf_op_costs).unwrap_or(0)),
                ("word_costs", bincode::serialized_size(&self.word_group_buf_op_costs).unwrap_or(0)),
            ];
            eprintln!(
                "[glrmask/profile][runtime_mask_cache_sizes] {}",
                cache_sizes
                    .iter()
                    .map(|(name, bytes)| format!("{name}={bytes}"))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            let cache_a = bincode::serialize(&(
                &self.word_group_sparse_masks,
                &self.word_group_prefix_buf_masks,
                &self.word_group_sparse_prefix_entries,
                &self.pair_word_group_buf_masks,
                &self.quad_word_group_buf_masks,
                &self.super_word_group_buf_masks,
                &self.mega_word_group_buf_masks,
                &self.giga_word_group_buf_masks,
                &self.all_tokens_buf_mask,
            ))
            .expect("runtime mask cache profiling serialization should succeed");
            let cache_b = bincode::serialize(&(
                &self.heavy_token_dense_masks,
                &self.internal_token_buf_flat,
                &self.internal_token_buf_offsets,
                self.total_internal_buf_cost,
                &self.heavy_token_indices,
                self.heavy_total_cost,
                self.light_avg_cost_x256,
                &self.internal_token_buf_op_costs,
                &self.word_group_buf_op_costs,
            ))
            .expect("runtime mask cache profiling serialization should succeed");
            eprintln!(
                "[glrmask/profile][runtime_mask_cache_candidate] ms={:.3} bytes={}",
                started.elapsed().as_secs_f64() * 1000.0,
                cache_a.len() + cache_b.len(),
            );
        }
        let total_started = profile.then(std::time::Instant::now);
        const PARALLEL_ASSEMBLY_MAX_PACKED_DWA_BYTES: usize = 1024 * 1024;
        let packed_dwa_wire_len = self
            .packed_parser_dwa
            .as_ref()
            .and_then(|packed| packed.fast_wire_len());
        let direct_dwa_wire_len = self
            .packed_parser_dwa
            .as_ref()
            .and_then(|packed| packed.direct_fast_wire_len());
        let parallel_assembly_candidate = self.packed_parser_dwa.is_none()
            || packed_dwa_wire_len
                .is_some_and(|len| len <= PARALLEL_ASSEMBLY_MAX_PACKED_DWA_BYTES);
        const DIRECT_TOKENIZER_MIN_BYTES: usize = 400 * 1024;
        // TKF2 is deliberately load-oriented: one byte label plus a fixed-width
        // target per transition. That is excellent for normal tokenizers, but
        // a very large DFA with >u16::MAX states can expand dramatically. Keep
        // the old packed/varint tokenizer wire as a size-adaptive escape hatch.
        const FAST_TOKENIZER_MAX_BYTES: usize = 64 * 1024 * 1024;
        let fast_tokenizer_layout =
            crate::automata::lexer::tokenizer::artifact_serde::fast_layout_for_write(
                &self.tokenizer,
            );
        let fast_tokenizer_len = fast_tokenizer_layout.map(|layout| layout.len());
        let preserve_compressed_tokenizer = fast_tokenizer_len.is_none();
        // If no direct TKF2 layout exists, the save path below necessarily
        // preserves the compressed/packed runtime representation. Computing an
        // expanded TKF2 size in that case is dead work and can require several
        // full-state metadata scans for million-state tokenizers.
        let fast_tokenizer_size = fast_tokenizer_len.unwrap_or(0);
        let compact_tokenizer = fast_tokenizer_len
            .is_some_and(|len| len > FAST_TOKENIZER_MAX_BYTES);
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer_save_select] fast_len={:?} fast_size={} compact={} preserve_compressed={} parallel_assembly_candidate={} packed_dwa_len={:?}",
                fast_tokenizer_len,
                fast_tokenizer_size,
                compact_tokenizer,
                preserve_compressed_tokenizer,
                parallel_assembly_candidate,
                packed_dwa_wire_len,
            );
        }
        let direct_tokenizer_len = (parallel_assembly_candidate || direct_dwa_wire_len.is_some())
            .then_some(fast_tokenizer_len)
            .flatten()
            .filter(|_| !compact_tokenizer)
            .filter(|&len| len >= DIRECT_TOKENIZER_MIN_BYTES);
        let ((token_bytes, (original_token_map, (tokenizer, internal_token_buf_masks))), ((weight_pool, core), (dwa, table, runtime, token_mask_cache, composition_metadata))) = rayon::join(
            || rayon::join(
                || {
                    let started = profile.then(std::time::Instant::now);
                    // Compiler-created constraints execute from the indexed
                    // vocabulary too, so persisting that same runtime storage
                    // is not a serialization cache. Loaded artifact-backed
                    // vocabularies may need a section copy if they no longer
                    // own a standalone token wire.
                    let bytes = self.packed_token_bytes.as_ref().map_or_else(
                        || {
                            std::sync::Arc::new(
                                crate::runtime::artifact::token_bytes_artifact_serde::pack_external(
                                    &self.token_bytes,
                                ),
                            )
                        },
                        |packed| {
                            packed.whole_wire_arc().unwrap_or_else(|| {
                                std::sync::Arc::new(packed.wire().to_vec())
                            })
                        },
                    );
                    if let Some(started) = started {
                        eprintln!(
                            "[glrmask/profile][constraint_save_section] name=token_bytes ms={:.3} bytes={}",
                            started.elapsed().as_secs_f64() * 1000.0,
                            bytes.len(),
                        );
                    }
                    bytes
                },
                || rayon::join(
                    || {
                        let started = profile.then(std::time::Instant::now);
                        let bytes =
                            crate::runtime::artifact::original_token_map_artifact_serde::to_fast_bytes(
                                self.original_token_map(),
                            );
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][constraint_save_section] name=original_token_map ms={:.3} bytes={}",
                                started.elapsed().as_secs_f64() * 1000.0,
                                bytes.len(),
                            );
                        }
                        bytes
                    },
                    || rayon::join(
                        || {
                            let started = profile.then(std::time::Instant::now);
                            let bytes = if preserve_compressed_tokenizer {
                                crate::automata::lexer::tokenizer::artifact_serde::build_huge_bytes(
                                    &self.tokenizer,
                                )
                                    .unwrap_or_else(|| {
                                        crate::automata::lexer::tokenizer::artifact_serde::to_segment_bytes(
                                            &self.tokenizer,
                                        )
                                    })
                            } else if compact_tokenizer {
                                crate::automata::lexer::tokenizer::artifact_serde::to_packed_bytes(
                                    &self.tokenizer,
                                )
                            } else if direct_tokenizer_len.is_some() {
                                Vec::new()
                            } else {
                                crate::automata::lexer::tokenizer::artifact_serde::to_fast_bytes(
                                    &self.tokenizer,
                                )
                            };
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_save_section] name=tokenizer ms={:.3} bytes={}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                    direct_tokenizer_len.unwrap_or(bytes.len()),
                                );
                            }
                            bytes
                        },
                        || {
                            let started = profile.then(std::time::Instant::now);
                            let bytes = encode_internal_token_buf_masks(self);
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_save_section] name=internal_token_buf_masks ms={:.3} bytes={}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                    bytes.len(),
                                );
                            }
                            bytes
                        },
                    ),
                ),
            ),
            || rayon::join(
            || {
                let branch_started = profile.then(std::time::Instant::now);
                let weights = if self.packed_non_dwa_weights.is_some() {
                    Vec::new()
                } else {
                    constraint_serialized_weight_pool(self)
                };
                let (weight_pool, encoded) = rayon::join(
                    || {
                        let weights_started = profile.then(std::time::Instant::now);
                        let weight_pool = self.packed_non_dwa_weights.as_ref().map_or_else(
                            || crate::ds::weight::pack_pooled_weights(&weights),
                            |packed| packed.pool.packed_bytes().to_vec(),
                        );
                        if let Some(started) = weights_started {
                            eprintln!(
                                "[glrmask/profile][constraint_save_section] name=weights ms={:.3} bytes={}",
                                started.elapsed().as_secs_f64() * 1000.0,
                                weight_pool.len(),
                            );
                        }
                        weight_pool
                    },
                    || {
                        // The pooled Weight-id map is thread-local, so install
                        // it inside the core branch rather than on the parent
                        // Rayon worker.  Packing WPL3 only reads the same
                        // immutable Weight slice and is independent once ids
                        // have been defined by stable slice order.
                        if let Some(ids) = packed_constraint_serialized_weight_ids(self) {
                            crate::ds::weight::begin_pooled_weight_serde_encode_ids(ids);
                        } else {
                            crate::ds::weight::begin_pooled_weight_serde_encode(&weights);
                        }
                        let previous_external =
                            crate::automata::weighted::dwa::set_external_serde(true);
                        let previous_external_table =
                            crate::compiler::glr::table::artifact_serde::set_external_serde(true);
                        let previous_compact_tokenizer =
                            crate::automata::lexer::tokenizer::set_compact_artifact_serde(true);
                        let previous_external_tokenizer =
                            crate::automata::lexer::tokenizer::set_external_artifact_serde(true);
                        let previous_omit_inverse =
                            crate::runtime::artifact::internal_token_inverse_artifact_serde::set_omit(true);
                        let previous_packed_original_token_map =
                            crate::runtime::artifact::original_token_map_artifact_serde::set_packed(true);
                        let previous_external_original_token_map =
                            crate::runtime::artifact::original_token_map_artifact_serde::set_external(true);
                        let previous_packed_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_packed(true);
                        let previous_external_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_external(true);
                        let started = profile.then(std::time::Instant::now);
                        // `bincode::serialize` first runs `serialized_size` and
                        // then serializes again. Custom compact serializers
                        // (especially the tokenizer) do real packing work in
                        // both passes. Write once into a generously-sized Vec
                        // instead; capacity growth is much cheaper than
                        // rebuilding every packed field twice.
                        let mut encoded = Vec::with_capacity(3 * 1024 * 1024);
                        encoded.extend_from_slice(&CURRENT_CORE_MAGIC);
                        let omit_tsid_inverse = self.can_defer_internal_tsid_inverse();
                        let core_flags = if omit_tsid_inverse {
                            CURRENT_CORE_FLAG_OMIT_TSID_INVERSE
                        } else {
                            0
                        };
                        encoded.extend_from_slice(&core_flags.to_le_bytes());
                        let base_len_offset = encoded.len();
                        encoded.extend_from_slice(&0u64.to_le_bytes());
                        let expr_len_offset = encoded.len();
                        encoded.extend_from_slice(&0u64.to_le_bytes());
                        let base_start = encoded.len();
                        let previous_omit_tsid_inverse =
                            crate::runtime::artifact::internal_tsid_inverse_artifact_serde::set_omit(
                                omit_tsid_inverse,
                            );
                        let encode_result = bincode::serialize_into(
                            &mut encoded,
                            &ConstraintArtifactCurrentCoreBaseRef {
                                constraint: self,
                                ignore_expr: &self.ignore_expr,
                                parser_state_domain_labels: &self.parser_state_domain_labels,
                                static_dynamic_overlay: &self.static_dynamic_overlay,
                                late_grammar_slots: &self.late_grammar_slots,
                            },
                        );
                        crate::runtime::artifact::internal_tsid_inverse_artifact_serde::set_omit(
                            previous_omit_tsid_inverse,
                        );
                        let base_len = encoded.len() - base_start;
                        encoded[base_len_offset..base_len_offset + 8]
                            .copy_from_slice(&(base_len as u64).to_le_bytes());
                        let expr_start = encoded.len();
                        if let Some(blob) = self.deferred_terminal_exprs_blob.as_ref() {
                            blob.append_raw_serialized(&mut encoded)
                                .expect("deferred terminal expression serialization should succeed");
                        } else if let Some(exprs) = self.tokenizer.terminal_exprs() {
                            bincode::serialize_into(&mut encoded, exprs)
                                .expect("terminal expression serialization should succeed");
                        }
                        let expr_len = encoded.len() - expr_start;
                        encoded[expr_len_offset..expr_len_offset + 8]
                            .copy_from_slice(&(expr_len as u64).to_le_bytes());
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
                        crate::ds::weight::end_pooled_weight_serde_encode();
                        encode_result.expect("Constraint core serialization should succeed");
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][constraint_save_section] name=core ms={:.3} bytes={}",
                                started.elapsed().as_secs_f64() * 1000.0,
                                encoded.len(),
                            );
                        }
                        encoded
                    },
                );
                if let Some(started) = branch_started {
                    eprintln!(
                        "[glrmask/profile][constraint_save_section] name=weights_core_branch ms={:.3}",
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                (weight_pool, encoded)
            },
            || {
                let ((dwa, table), ((runtime, token_mask_cache), composition_metadata)) = rayon::join(
                    || {
                        rayon::join(
                    || {
                        let Some(_packed) = self.packed_parser_dwa.as_ref() else {
                            let started = profile.then(std::time::Instant::now);
                            let bytes = self.parser_dwa.artifact_packed_bytes();
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_save_section] name=dwa ms={:.3} bytes={}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                    bytes.len(),
                                );
                            }
                            return bytes;
                        };
                        // A fresh packed DWA is emitted directly into its
                        // final artifact section below. Avoid constructing a
                        // second multi-megabyte wire image here.
                        Vec::new()
                    },
                    || {
                        let started = profile.then(std::time::Instant::now);
                        let bytes = if let Some(parser) = &self.template_parser {
                            crate::runtime::parser_backend::wire::encode(parser)
                        } else {
                            let rules = self.retained_table_rules()
                                .expect("validated retained grammar rules must remain readable");
                            crate::compiler::glr::table::artifact_serde::to_compact_bytes_with_rules(&self.table, rules)
                        };
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][constraint_save_section] name=table ms={:.3} bytes={}",
                                started.elapsed().as_secs_f64() * 1000.0,
                                bytes.len(),
                            );
                        }
                        bytes
                    },
                        )
                    },
                    || rayon::join(
                        || rayon::join(
                        || {
                            let started = profile.then(std::time::Instant::now);
                            let packed_dwa_dense_masks = &self.packed_dwa_token_dense_masks;
                            let static_virtual_residual_wire = if self.uses_dynamic_runtime() {
                                Vec::new()
                            } else {
                                self.dynamic_mask_vocab
                                    .virtual_residual_mask_projection_parts()
                                    .map(|(mask_tokenizer, projections)| encode_static_virtual_residual_mask_wire(&self.tokenizer, mask_tokenizer, projections))
                                    .unwrap_or_default()
                            };
                            let metadata = ConstraintArtifactCurrentRuntimeRef {
                                terminal_live_states: &self.terminal_live_states,
                                segmented_runtime: segmented_runtime_artifact_ref(self),
                                dynamic_mask_vocab: self.uses_dynamic_runtime().then(|| {
                                    // The template-only artifact is also the self-contained
                                    // O2 persistence body; preserve its proven mask quotient.
                                    if self.has_template_parser() { self.dynamic_mask_vocab.to_artifact() }
                                    else { self.dynamic_mask_vocab.to_vocab_artifact() }
                                }).flatten(),
                                virtual_runtimes: self
                                    .tokenizer
                                    .has_any_virtual_runtime()
                                    .then(|| self.tokenizer.virtual_runtime_metadata())
                                    .unwrap_or_default(),
                                static_virtual_residual_mask: None,
                                packed_dwa_dense_mask_ids: packed_dwa_dense_masks.token_set_ids(),
                                packed_dwa_dense_mask_rows: packed_dwa_dense_masks.flat_rows(),
                            };
                            let bytes = encode_current_runtime_wire(&metadata, &static_virtual_residual_wire);
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_save_section] name=runtime ms={:.3} bytes={}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                    bytes.len(),
                                );
                            }
                            bytes
                        },
                        || {
                            let started = profile.then(std::time::Instant::now);
                            let bytes = encode_token_mask_cache(self);
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_save_section] name=token_mask_cache ms={:.3} bytes={}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                    bytes.len(),
                                );
                            }
                            bytes
                        },
                        ),
                        || {
                            let started = profile.then(std::time::Instant::now);
                            let bytes = encode_composition_metadata_for_save(self);
                            if let Some(started) = started {
                                eprintln!(
                                    "[glrmask/profile][constraint_save_section] name=composition_metadata ms={:.3} bytes={}",
                                    started.elapsed().as_secs_f64() * 1000.0,
                                    bytes.len(),
                                );
                            }
                            bytes
                        },
                    ),
                );
                (dwa, table, runtime, token_mask_cache, composition_metadata)
            },
            ),
        );

        // When available, `fast_wire_len()` is the exact current packed-DWA
        // length. Small fallback DWAs may require actual emission before their
        // byte length is known.
        let estimated_dwa_wire_len = self
            .packed_parser_dwa
            .as_ref()
            .and_then(|packed| packed.fast_wire_len())
            .unwrap_or(dwa.len());
        // For ordinary constraints, materializing the small packed-DWA section
        // is much cheaper than serially copying the entire multi-megabyte
        // artifact after every independent serializer has finished.  Once all
        // sections are ordinary byte slices, copy them into disjoint final
        // ranges in parallel. Large JS-like DWAs keep direct emission below to
        // avoid creating a second 10+ MiB DWA buffer.
        let packed_dwa_for_parallel = self
            .packed_parser_dwa
            .as_ref()
            .filter(|packed| packed.backed_fast_wire_bytes().is_none())
            .filter(|packed| {
                packed
                    .fast_wire_len()
                    .is_some_and(|len| len <= PARALLEL_ASSEMBLY_MAX_PACKED_DWA_BYTES)
            })
            .map(|packed| packed.fast_wire_bytes());
        let backed_packed_dwa = self
            .packed_parser_dwa
            .as_ref()
            .and_then(|packed| packed.backed_fast_wire_bytes());
        let parallel_dwa = backed_packed_dwa
            .or_else(|| packed_dwa_for_parallel.as_deref())
            .or_else(|| (!dwa.is_empty()).then_some(dwa.as_slice()))
            .or_else(|| self.packed_parser_dwa.is_none().then_some(dwa.as_slice()));
        let dwa_wire_len = parallel_dwa
            .map(<[u8]>::len)
            .unwrap_or(estimated_dwa_wire_len);
        let tokenizer_wire_len = direct_tokenizer_len.unwrap_or(tokenizer.len());
        let weight_pool_wire = weight_pool.as_slice();
        let table_wire = table.as_slice();
        let runtime_wire = runtime.as_slice();
        let original_token_map_wire = original_token_map.as_slice();
        let internal_token_buf_masks_wire = internal_token_buf_masks.as_slice();
        let token_mask_cache_wire = token_mask_cache.as_slice();
        let composition_metadata_wire = composition_metadata.as_slice();
        let internal_token_buf_masks_absolute_start = CONSTRAINT_HEADER_LEN
            + V30_SECTION_HEADER_LEN
            + weight_pool_wire.len()
            + dwa_wire_len
            + table_wire.len()
            + core.len()
            + runtime_wire.len()
            + token_bytes.len()
            + original_token_map_wire.len()
            + tokenizer_wire_len;
        let internal_token_buf_masks_leading_padding = if internal_token_buf_masks_wire
            .starts_with(b"IBM2")
            && internal_token_buf_masks_wire.len() >= 12
        {
            let group_count = u32::from_le_bytes(
                internal_token_buf_masks_wire[4..8]
                    .try_into()
                    .expect("IBM2 header has fixed width"),
            ) as usize;
            let entries_offset = 12usize.saturating_add((group_count + 1).saturating_mul(4));
            let align = std::mem::align_of::<PackedInternalTokenBufMask>();
            (align - ((internal_token_buf_masks_absolute_start + entries_offset) % align)) % align
        } else {
            0
        };
        let internal_token_buf_masks_section_len = internal_token_buf_masks_leading_padding
            + internal_token_buf_masks_wire.len();
        let token_mask_cache_absolute_start = CONSTRAINT_HEADER_LEN
            + V30_SECTION_HEADER_LEN
            + weight_pool_wire.len()
            + dwa_wire_len
            + table_wire.len()
            + core.len()
            + runtime_wire.len()
            + token_bytes.len()
            + original_token_map_wire.len()
            + tokenizer_wire_len
            + internal_token_buf_masks_section_len;
        let token_mask_cache_leading_padding = if token_mask_cache_wire.starts_with(b"TMC6")
            || token_mask_cache_wire.starts_with(b"TMC7")
        {
            (4 - (token_mask_cache_absolute_start & 3)) & 3
        } else {
            0
        };
        let token_mask_cache_section_len =
            token_mask_cache_leading_padding + token_mask_cache_wire.len();
        let assemble_started = profile.then(std::time::Instant::now);
        let payload_len = V30_SECTION_HEADER_LEN
            + weight_pool_wire.len()
            + dwa_wire_len
            + table_wire.len()
            + core.len()
            + runtime_wire.len()
            + token_bytes.len()
            + original_token_map_wire.len()
            + tokenizer_wire_len
            + internal_token_buf_masks_section_len
            + token_mask_cache_section_len
            + composition_metadata_wire.len();

        let direct_runtime_dwa = parallel_dwa.is_none().then(|| {
            self.packed_parser_dwa.as_deref().filter(|packed| {
                packed.direct_fast_wire_len() == Some(dwa_wire_len)
            })
        }).flatten();
        if parallel_dwa.is_some() || direct_runtime_dwa.is_some() {
            let dwa_bytes = parallel_dwa.unwrap_or(&[]);
            if !dwa_bytes.is_empty() {
                debug_assert_eq!(dwa_bytes.len(), dwa_wire_len);
            }
            let total_len = CONSTRAINT_HEADER_LEN + payload_len;
            let mut bytes = Vec::<u8>::with_capacity(total_len);
            // SAFETY: every byte in the allocation is initialized below before
            // the Vec is observed or returned. The section destinations are
            // disjoint slices split from this one allocation and are each
            // written exactly once.
            unsafe {
                bytes.set_len(total_len);
            }
            let header_len = CONSTRAINT_HEADER_LEN + V30_SECTION_HEADER_LEN;
            let (header, mut body) = bytes.split_at_mut(header_len);
            let mut pos = 0usize;
            header[pos..pos + CONSTRAINT_MAGIC.len()].copy_from_slice(&CONSTRAINT_MAGIC);
            pos += CONSTRAINT_MAGIC.len();
            header[pos..pos + 2].copy_from_slice(&artifact_version.to_le_bytes());
            pos += 2;
            header[pos..pos + 8].copy_from_slice(&(payload_len as u64).to_le_bytes());
            pos += 8;
            header[pos..pos + V30_SECTION_MAGIC.len()].copy_from_slice(&V30_SECTION_MAGIC);
            pos += V30_SECTION_MAGIC.len();
            for len in [
                weight_pool_wire.len(),
                dwa_wire_len,
                table_wire.len(),
                core.len(),
                runtime_wire.len(),
                token_bytes.len(),
                original_token_map_wire.len(),
                tokenizer_wire_len,
                internal_token_buf_masks_section_len,
                token_mask_cache_section_len,
                composition_metadata_wire.len(),
            ] {
                header[pos..pos + 8].copy_from_slice(&(len as u64).to_le_bytes());
                pos += 8;
            }
            debug_assert_eq!(pos, header.len());

            let mut copy_jobs = Vec::<(usize, usize, usize)>::with_capacity(24);
            let mut direct_tokenizer_destination = None;
            let mut direct_dwa_destination = None;
            let sources: [&[u8]; 11] = [
                weight_pool_wire,
                dwa_bytes,
                table_wire,
                core.as_slice(),
                runtime_wire,
                token_bytes.as_slice(),
                original_token_map_wire,
                tokenizer.as_slice(),
                internal_token_buf_masks_wire,
                token_mask_cache_wire,
                composition_metadata_wire,
            ];
            for (index, len) in [
                weight_pool_wire.len(),
                dwa_wire_len,
                table_wire.len(),
                core.len(),
                runtime_wire.len(),
                token_bytes.len(),
                original_token_map_wire.len(),
                tokenizer_wire_len,
                internal_token_buf_masks_section_len,
                token_mask_cache_section_len,
                composition_metadata_wire.len(),
            ]
            .into_iter()
            .enumerate()
            {
                let (section, rest) = body.split_at_mut(len);
                let section = if index == 8 && internal_token_buf_masks_leading_padding != 0 {
                    let (padding, section) = section
                        .split_at_mut(internal_token_buf_masks_leading_padding);
                    padding.fill(0);
                    section
                } else if index == 9 && token_mask_cache_leading_padding != 0 {
                    let (padding, section) =
                        section.split_at_mut(token_mask_cache_leading_padding);
                    padding.fill(0);
                    section
                } else {
                    section
                };
                if index == 1 && direct_runtime_dwa.is_some() {
                    direct_dwa_destination = Some(section);
                } else if index == 7 && direct_tokenizer_len.is_some() {
                    direct_tokenizer_destination = Some(section);
                } else {
                    const PARALLEL_COPY_SPLIT_MIN_BYTES: usize = 4 * 1024 * 1024;
                    let source = sources[index];
                    debug_assert_eq!(section.len(), source.len());
                    if len >= PARALLEL_COPY_SPLIT_MIN_BYTES && rayon::current_num_threads() > 1 {
                        let target_chunks = rayon::current_num_threads().clamp(2, 12);
                        let chunk_size = source.len().div_ceil(target_chunks).max(1024 * 1024);
                        let mut destination = section;
                        let mut source = source;
                        while !source.is_empty() {
                            let count = chunk_size.min(source.len());
                            let (destination_chunk, destination_rest) = destination.split_at_mut(count);
                            let (source_chunk, source_rest) = source.split_at(count);
                            copy_jobs.push((
                                destination_chunk.as_mut_ptr() as usize,
                                source_chunk.as_ptr() as usize,
                                count,
                            ));
                            destination = destination_rest;
                            source = source_rest;
                        }
                    } else {
                        copy_jobs.push((
                            section.as_mut_ptr() as usize,
                            source.as_ptr() as usize,
                            source.len(),
                        ));
                    }
                }
                body = rest;
            }
            debug_assert!(body.is_empty());
            if rayon::current_num_threads() > 1 {
                let copy_all = || {
                    copy_jobs.into_par_iter().for_each(|(destination, source, len)| {
                        // SAFETY: jobs were created from disjoint final-artifact
                        // ranges. Source sections remain alive for the whole
                        // join and never overlap the destination allocation.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                source as *const u8,
                                destination as *mut u8,
                                len,
                            );
                        }
                    });
                };
                let write_tokenizer = || {
                    if let Some(destination) = direct_tokenizer_destination {
                        let started = profile.then(std::time::Instant::now);
                        crate::automata::lexer::tokenizer::artifact_serde::write_fast_bytes_with_layout(
                            &self.tokenizer,
                            fast_tokenizer_layout.expect("direct tokenizer write requires a fast layout"),
                            destination,
                        )
                        .expect("precomputed fast tokenizer layout should match final section");
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][constraint_save_section] name=tokenizer_direct ms={:.3} bytes={}",
                                started.elapsed().as_secs_f64() * 1000.0,
                                destination.len(),
                            );
                        }
                    }
                };
                let write_dwa = || {
                    if let (Some(packed), Some(destination)) =
                        (direct_runtime_dwa, direct_dwa_destination)
                    {
                        let started = profile.then(std::time::Instant::now);
                        packed
                            .write_direct_fast_wire_bytes(destination)
                            .expect("direct DWA length should match final section");
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][constraint_save_section] name=dwa_direct_parallel ms={:.3} bytes={}",
                                started.elapsed().as_secs_f64() * 1000.0,
                                destination.len(),
                            );
                        }
                    }
                };
                rayon::join(copy_all, || rayon::join(write_dwa, write_tokenizer));
            } else {
                for (destination, source, len) in copy_jobs {
                    // SAFETY: same disjointness/liveness argument as above.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            source as *const u8,
                            destination as *mut u8,
                            len,
                        );
                    }
                }
                if let Some(destination) = direct_tokenizer_destination {
                    crate::automata::lexer::tokenizer::artifact_serde::write_fast_bytes_with_layout(
                        &self.tokenizer,
                        fast_tokenizer_layout.expect("direct tokenizer write requires a fast layout"),
                        destination,
                    )
                    .expect("precomputed fast tokenizer layout should match final section");
                }
                if let (Some(packed), Some(destination)) =
                    (direct_runtime_dwa, direct_dwa_destination)
                {
                    packed
                        .write_direct_fast_wire_bytes(destination)
                        .expect("direct DWA length should match final section");
                }
            }
            if let Some(started) = assemble_started {
                eprintln!(
                    "[glrmask/profile][constraint_save_assemble] ms={:.3} mode=parallel",
                    started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            if let Some(started) = total_started {
                eprintln!(
                    "[glrmask/profile][constraint_save] total_ms={:.3} bytes={}",
                    started.elapsed().as_secs_f64() * 1000.0,
                    bytes.len(),
                );
            }
            return bytes;
        }
        let mut bytes = Vec::with_capacity(CONSTRAINT_HEADER_LEN + payload_len);
        bytes.extend_from_slice(&CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&artifact_version.to_le_bytes());
        let payload_len_offset = bytes.len();
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&V30_SECTION_MAGIC);
        bytes.extend_from_slice(&(weight_pool_wire.len() as u64).to_le_bytes());
        let dwa_len_offset = bytes.len();
        bytes.extend_from_slice(&(dwa.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(table_wire.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(core.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(runtime_wire.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(token_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(original_token_map_wire.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(tokenizer_wire_len as u64).to_le_bytes());
        bytes.extend_from_slice(&(internal_token_buf_masks_section_len as u64).to_le_bytes());
        bytes.extend_from_slice(&(token_mask_cache_section_len as u64).to_le_bytes());
        bytes.extend_from_slice(&(composition_metadata_wire.len() as u64).to_le_bytes());
        bytes.extend_from_slice(weight_pool_wire);
        let dwa_start = bytes.len();
        if let Some(packed) = &self.packed_parser_dwa {
            debug_assert!(dwa.is_empty());
            let started = profile.then(std::time::Instant::now);
            packed.append_fast_wire_bytes(&mut bytes);
            if let Some(started) = started {
                eprintln!(
                    "[glrmask/profile][constraint_save_section] name=dwa_direct ms={:.3} bytes={}",
                    started.elapsed().as_secs_f64() * 1000.0,
                    bytes.len() - dwa_start,
                );
            }
        } else {
            bytes.extend_from_slice(&dwa);
        }
        let dwa_len = bytes.len() - dwa_start;
        bytes[dwa_len_offset..dwa_len_offset + 8]
            .copy_from_slice(&(dwa_len as u64).to_le_bytes());
        bytes.extend_from_slice(table_wire);
        bytes.extend_from_slice(&core);
        bytes.extend_from_slice(runtime_wire);
        bytes.extend_from_slice(token_bytes.as_slice());
        bytes.extend_from_slice(original_token_map_wire);
        if direct_tokenizer_len.is_some() {
            unreachable!("direct tokenizer encoding requires the parallel assembly path");
        } else {
            bytes.extend_from_slice(&tokenizer);
        }
        bytes.resize(bytes.len() + internal_token_buf_masks_leading_padding, 0);
        bytes.extend_from_slice(internal_token_buf_masks_wire);
        bytes.resize(bytes.len() + token_mask_cache_leading_padding, 0);
        bytes.extend_from_slice(token_mask_cache_wire);
        bytes.extend_from_slice(composition_metadata_wire);
        let payload_len = bytes.len() - CONSTRAINT_HEADER_LEN;
        bytes[payload_len_offset..payload_len_offset + 8]
            .copy_from_slice(&(payload_len as u64).to_le_bytes());
        if let Some(started) = assemble_started {
            eprintln!(
                "[glrmask/profile][constraint_save_assemble] ms={:.3}",
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][constraint_save] total_ms={:.3} bytes={}",
                started.elapsed().as_secs_f64() * 1000.0,
                bytes.len(),
            );
        }
        bytes
    }

    /// Load a compiled constraint from an artifact produced by [`Constraint::save`].
    ///
    /// Passing an owned `Vec<u8>` transfers its allocation into the loaded
    /// constraint without copying the artifact. Borrowed byte slices are also
    /// accepted; current-format artifacts copy borrowed input once because
    /// runtime structures retain zero-copy views into persistent backing bytes.
    pub fn load<'a>(bytes: impl Into<Cow<'a, [u8]>>) -> crate::Result<Self> {
        let bytes = bytes.into();
        if bytes.starts_with(PREVIOUS_ROOT_POLICY_MAGIC) {
            return Self::load_previous_root_policy(bytes);
        }
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

    fn load_previous_root_policy(bytes: Cow<'_, [u8]>) -> crate::Result<Self> {
        if bytes.len() < 12 {
            return Err(crate::Error::Serialization("truncated root policy header".to_owned()));
        }
        let count = u32::from_le_bytes(bytes[8..12].try_into().expect("header checked")) as usize;
        let start = count.checked_mul(4).and_then(|len| 12usize.checked_add(len))
            .filter(|&start| start < bytes.len())
            .ok_or_else(|| crate::Error::Serialization("invalid root policy length".to_owned()))?;
        let ids: Vec<u32> = bytes[12..start].chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("chunk width checked"))).collect();
        if count == 0 || ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(crate::Error::Serialization("noncanonical end-token policy".to_owned()));
        }
        let body = match bytes {
            Cow::Owned(mut bytes) => Self::load_body(Cow::Owned(bytes.split_off(start)))?,
            Cow::Borrowed(bytes) => Self::load_body(Cow::Borrowed(&bytes[start..]))?,
        };
        if !body.late_grammar_slots.is_empty() {
            return Err(crate::Error::Serialization("root artifact has unresolved slots".to_owned()));
        }
        body.with_end_tokens(&ids).map_err(|error| crate::Error::Serialization(error.to_string()))
    }

    fn load_body(bytes: Cow<'_, [u8]>) -> crate::Result<Self> {
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

    fn load_impl(
        bytes: &[u8],
        owned_artifact: Option<std::sync::Arc<Vec<u8>>>,
    ) -> crate::Result<Self> {
        let profile = std::env::var_os("GLRMASK_PROFILE_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let mut decompress_ms = 0.0;
        if bytes.len() < CONSTRAINT_HEADER_LEN || !bytes.starts_with(&CONSTRAINT_MAGIC) {
            return Err(crate::GlrMaskError::Serialization(
                "invalid constraint artifact header".to_owned(),
            ));
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if !matches!(
            version,
            LEGACY_CONSTRAINT_VERSION
                | PREVIOUS_COMPRESSED_CONSTRAINT_VERSION
                | PREVIOUS_EXPRLESS_CONSTRAINT_VERSION
                | PREVIOUS_TERMINAL_EXPRS_CONSTRAINT_VERSION
                | PREVIOUS_DOMAIN_LABELS_CONSTRAINT_VERSION
                | PREVIOUS_UNCOMPRESSED_CONSTRAINT_VERSION
                | PREVIOUS_SECTIONED_CONSTRAINT_VERSION
                | PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
                | PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION
                | PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
                | PREVIOUS_EXTERNAL_RUNTIME_CONSTRAINT_VERSION
                | PREVIOUS_SERIALIZATION_CURRENT_CONSTRAINT_VERSION
                | PREVIOUS_COMBINED_CONSTRAINT_VERSION
                | PREVIOUS_SEGMENTED_MATERIALIZATION_CONSTRAINT_VERSION
                | PREVIOUS_BOUNDARY_SHARDLESS_CONSTRAINT_VERSION
                | PREVIOUS_BOUNDARY_SHARDED_CONSTRAINT_VERSION
                | PREVIOUS_RECURSIVE_PARSER_CONSTRAINT_VERSION
                | PREVIOUS_STATIC_RESIDUAL_CONSTRAINT_VERSION
                | PREVIOUS_STATIC_PROJECTION_CONSTRAINT_VERSION
                | PREVIOUS_BOUNDARY_SUMMARYLESS_CONSTRAINT_VERSION
                | CONSTRAINT_VERSION
                | TEMPLATE_CONSTRAINT_VERSION
        ) {
            let decompress_started = profile.then(std::time::Instant::now);
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
        let current_backing = if uses_external_runtime_sections(version)
            || version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
        {
            Some(
                owned_artifact
                    .clone()
                    .unwrap_or_else(|| std::sync::Arc::new(bytes.to_vec())),
            )
        } else {
            None
        };
        let section_bytes = current_backing
            .as_ref()
            .map_or(bytes, |backing| backing.as_slice());
        let payload = &section_bytes[CONSTRAINT_HEADER_LEN..];
        let mut raw;
        let serialized = if matches!(
            version,
            PREVIOUS_COMPRESSED_CONSTRAINT_VERSION
                | PREVIOUS_EXPRLESS_CONSTRAINT_VERSION
                | PREVIOUS_TERMINAL_EXPRS_CONSTRAINT_VERSION
                | PREVIOUS_DOMAIN_LABELS_CONSTRAINT_VERSION
        ) {
            let decompress_started = profile.then(std::time::Instant::now);
            if payload.len() < COMPRESSED_PAYLOAD_HEADER_LEN {
                return Err(crate::GlrMaskError::Serialization(
                    "invalid compressed constraint artifact payload".to_owned(),
                ));
            }
            let raw_len = usize::try_from(u64::from_le_bytes(
                payload[..COMPRESSED_PAYLOAD_HEADER_LEN]
                    .try_into()
                    .expect("compressed constraint payload header has fixed width"),
            ))
            .map_err(|_| {
                crate::GlrMaskError::Serialization(
                    "uncompressed constraint artifact length does not fit this platform".to_owned(),
                )
            })?;
            let compressed = &payload[COMPRESSED_PAYLOAD_HEADER_LEN..];
            let frame_len = zstd::zstd_safe::get_frame_content_size(compressed)
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
            if frame_len.is_some_and(|frame_len| frame_len != raw_len as u64) {
                return Err(crate::GlrMaskError::Serialization(
                    "invalid uncompressed constraint artifact length".to_owned(),
                ));
            }

            // Do not reserve the untrusted declared size up front. Stream into
            // a growing buffer and stop after one byte beyond the declared
            // length, so malformed artifacts cannot trigger an immediate huge
            // allocation merely by forging the envelope.
            let output_limit = raw_len.checked_add(1).ok_or_else(|| {
                crate::GlrMaskError::Serialization(
                    "uncompressed constraint artifact length is too large".to_owned(),
                )
            })?;
            let decoder = zstd::stream::read::Decoder::with_buffer(compressed)
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
            raw = Vec::new();
            decoder
                .take(output_limit as u64)
                .read_to_end(&mut raw)
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
            if raw.len() != raw_len {
                return Err(crate::GlrMaskError::Serialization(
                    "invalid uncompressed constraint artifact length".to_owned(),
                ));
            }
            decompress_ms = decompress_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            raw.as_slice()
        } else {
            payload
        };
        let deserialize_started = profile.then(std::time::Instant::now);
        let mut packed_dwa_inventory = None;
        let mut prepared_completion_wire: Option<Arc<[u8]>> = None;
        let mut loaded_packed_dwa_dense_masks = false;
        let mut constraint = if uses_external_runtime_sections(version)
            || version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
            || version == PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION
            || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
            || version == PREVIOUS_SECTIONED_CONSTRAINT_VERSION
        {
            let (
                weight_section,
                dwa_section,
                table_section,
                core_section,
                runtime_section,
                token_bytes_section,
                original_token_map_section,
                tokenizer_section,
                internal_token_buf_masks_section,
                token_mask_cache_section,
                composition_metadata_section,
            ) =
                if matches!(version, CONSTRAINT_VERSION | TEMPLATE_CONSTRAINT_VERSION) {
                    let (weight, dwa, table, core, runtime, token_bytes, original_map, tokenizer, internal_masks, token_mask_cache, composition_metadata) = v30_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (
                        weight,
                        dwa,
                        table,
                        core,
                        Some(runtime),
                        Some(token_bytes),
                        Some(original_map),
                        Some(tokenizer),
                        Some(internal_masks),
                        Some(token_mask_cache),
                        Some(composition_metadata),
                    )
                } else if version == PREVIOUS_BOUNDARY_SUMMARYLESS_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime, token_bytes, original_map, tokenizer, internal_masks, token_mask_cache, composition_metadata) = v29_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (
                        weight,
                        dwa,
                        table,
                        core,
                        Some(runtime),
                        Some(token_bytes),
                        Some(original_map),
                        Some(tokenizer),
                        Some(internal_masks),
                        Some(token_mask_cache),
                        Some(composition_metadata),
                    )
                } else if version == PREVIOUS_STATIC_PROJECTION_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime, token_bytes, original_map, tokenizer, internal_masks, token_mask_cache, composition_metadata) = v28_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (
                        weight,
                        dwa,
                        table,
                        core,
                        Some(runtime),
                        Some(token_bytes),
                        Some(original_map),
                        Some(tokenizer),
                        Some(internal_masks),
                        Some(token_mask_cache),
                        Some(composition_metadata),
                    )
                } else if version == PREVIOUS_STATIC_RESIDUAL_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime, token_bytes, original_map, tokenizer, internal_masks, token_mask_cache, composition_metadata) = v27_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (
                        weight,
                        dwa,
                        table,
                        core,
                        Some(runtime),
                        Some(token_bytes),
                        Some(original_map),
                        Some(tokenizer),
                        Some(internal_masks),
                        Some(token_mask_cache),
                        Some(composition_metadata),
                    )
                } else if version == PREVIOUS_RECURSIVE_PARSER_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime, token_bytes, original_map, tokenizer, internal_masks, token_mask_cache, composition_metadata) = v24_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (
                        weight,
                        dwa,
                        table,
                        core,
                        Some(runtime),
                        Some(token_bytes),
                        Some(original_map),
                        Some(tokenizer),
                        Some(internal_masks),
                        Some(token_mask_cache),
                        Some(composition_metadata),
                    )
                } else if version == PREVIOUS_BOUNDARY_SHARDED_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime, token_bytes, original_map, tokenizer, internal_masks, token_mask_cache, composition_metadata) = v23_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (
                        weight,
                        dwa,
                        table,
                        core,
                        Some(runtime),
                        Some(token_bytes),
                        Some(original_map),
                        Some(tokenizer),
                        Some(internal_masks),
                        Some(token_mask_cache),
                        Some(composition_metadata),
                    )
                } else if version == PREVIOUS_BOUNDARY_SHARDLESS_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime, token_bytes, original_map, tokenizer, internal_masks, token_mask_cache, composition_metadata) = v22_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (
                        weight,
                        dwa,
                        table,
                        core,
                        Some(runtime),
                        Some(token_bytes),
                        Some(original_map),
                        Some(tokenizer),
                        Some(internal_masks),
                        Some(token_mask_cache),
                        Some(composition_metadata),
                    )
                } else if version == PREVIOUS_SEGMENTED_MATERIALIZATION_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime, token_bytes, original_map, tokenizer, internal_masks, token_mask_cache, composition_metadata) = v21_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (
                        weight,
                        dwa,
                        table,
                        core,
                        Some(runtime),
                        Some(token_bytes),
                        Some(original_map),
                        Some(tokenizer),
                        Some(internal_masks),
                        Some(token_mask_cache),
                        Some(composition_metadata),
                    )
                } else if version == PREVIOUS_COMBINED_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime, token_bytes, original_map, tokenizer, internal_masks, token_mask_cache, composition_metadata) = v20_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (
                        weight,
                        dwa,
                        table,
                        core,
                        Some(runtime),
                        Some(token_bytes),
                        Some(original_map),
                        Some(tokenizer),
                        Some(internal_masks),
                        Some(token_mask_cache),
                        Some(composition_metadata),
                    )
                } else if version == PREVIOUS_SERIALIZATION_CURRENT_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime, token_bytes, original_map, tokenizer, internal_masks, token_mask_cache) = v19_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (
                        weight,
                        dwa,
                        table,
                        core,
                        Some(runtime),
                        Some(token_bytes),
                        Some(original_map),
                        Some(tokenizer),
                        Some(internal_masks),
                        Some(token_mask_cache),
                        None,
                    )
                } else if version == PREVIOUS_EXTERNAL_RUNTIME_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime, token_bytes, original_map, tokenizer, internal_masks) = v18_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (
                        weight,
                        dwa,
                        table,
                        core,
                        Some(runtime),
                        Some(token_bytes),
                        Some(original_map),
                        Some(tokenizer),
                        Some(internal_masks),
                        None,
                        None,
                    )
                } else if version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime, token_bytes) = v17_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (weight, dwa, table, core, Some(runtime), Some(token_bytes), None, None, None, None, None)
                } else if version == PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime) = v16_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (weight, dwa, table, core, Some(runtime), None, None, None, None, None, None)
                } else if version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION {
                    let (weight, dwa, table, core, runtime) = v15_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (weight, dwa, table, core, Some(runtime), None, None, None, None, None, None)
                } else {
                    let (weight, dwa, table, core) = v14_sections(serialized)
                        .map_err(crate::GlrMaskError::Serialization)?;
                    (weight, dwa, table, core, None, None, None, None, None, None, None)
                };
            let (((dwa_result, (table_result, runtime_result)), ((tokenizer_result, original_token_map_result), (internal_token_buf_masks_result, token_mask_cache_result))), core_result) = rayon::join(
                || rayon::join(
                    || rayon::join(
                        || {
                            let started = profile.then(std::time::Instant::now);
                            let result = if uses_external_runtime_sections(version)
                                || version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
                                || version == PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION
                                || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
                            {
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
                                    DecodedParserDwa::Packed(std::sync::Arc::new(dwa))
                                })
                            } else {
                                crate::automata::weighted::dwa::DWA::from_artifact_packed_bytes(
                                    dwa_section,
                                )
                                .map(|(dwa, inventory)| {
                                    DecodedParserDwa::Materialized(dwa, inventory)
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
                                    if version == TEMPLATE_CONSTRAINT_VERSION {
                                        return crate::runtime::parser_backend::wire::decode(table_section)
                                            .map(|seed| (crate::runtime::parser_backend::ParserTableStorage::absent(), None, Some(seed)));
                                    }
                                    let result = if uses_external_runtime_sections(version) {
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
                                    } else {
                                        crate::compiler::glr::table::artifact_serde::from_compact_bytes_deferred(table_section)
                                    };
                                    if let Some(started) = started {
                                        eprintln!("[glrmask/profile][constraint_section] name=table ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                                    }
                                    result.map(|decoded| (decoded.table.into(), decoded.deferred_rules, None))
                                },
                                || -> Result<Option<DecodedConstraintRuntime>, String> {
                                    let Some(runtime_section) = runtime_section else {
                                        return Ok(None);
                                    };
                                    let started = profile.then(std::time::Instant::now);
                                    let result = if matches!(version, CONSTRAINT_VERSION | TEMPLATE_CONSTRAINT_VERSION)
                                        || version == PREVIOUS_BOUNDARY_SUMMARYLESS_CONSTRAINT_VERSION
                                    {
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
                                                segmented_runtime_v20: None,
                                                segmented_runtime_v22: None,
                                                segmented_runtime_v23: None,
                                                segmented_runtime_v24: None,
                                                segmented_runtime_v27: runtime.segmented_runtime,
                                                dynamic_mask_vocab: runtime.dynamic_mask_vocab,
                                                virtual_runtimes: runtime.virtual_runtimes,
                                                static_virtual_residual_mask,
                                                packed_dwa_dense_masks,
                                            })
                                        })
                                    } else if version == PREVIOUS_STATIC_PROJECTION_CONSTRAINT_VERSION {
                                        bincode::deserialize::<ConstraintArtifactV28Runtime>(runtime_section)
                                        .and_then(|runtime| {
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
                                                segmented_runtime_v20: None,
                                                segmented_runtime_v22: None,
                                                segmented_runtime_v23: None,
                                                segmented_runtime_v24: None,
                                                segmented_runtime_v27: runtime.segmented_runtime,
                                                dynamic_mask_vocab: runtime.dynamic_mask_vocab,
                                                virtual_runtimes: runtime.virtual_runtimes,
                                                static_virtual_residual_mask: runtime
                                                    .static_virtual_residual_mask
                                                    .map(StaticVirtualResidualMaskArtifactV28::into_current)
                                                    .map(DecodedStaticVirtualResidualMask::Owned),
                                                packed_dwa_dense_masks,
                                            })
                                        })
                                    } else if version == PREVIOUS_STATIC_RESIDUAL_CONSTRAINT_VERSION {
                                        bincode::deserialize::<ConstraintArtifactV27Runtime>(
                                            runtime_section,
                                        )
                                        .and_then(|runtime| {
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
                                                segmented_runtime_v20: None,
                                                segmented_runtime_v22: None,
                                                segmented_runtime_v23: None,
                                                segmented_runtime_v24: None,
                                                segmented_runtime_v27: runtime.segmented_runtime,
                                                dynamic_mask_vocab: runtime.dynamic_mask_vocab,
                                                virtual_runtimes: runtime.virtual_runtimes,
                                                static_virtual_residual_mask: None,
                                                packed_dwa_dense_masks,
                                            })
                                        })
                                    } else if version == PREVIOUS_RECURSIVE_PARSER_CONSTRAINT_VERSION {
                                        bincode::deserialize::<ConstraintArtifactV24Runtime>(
                                            runtime_section,
                                        )
                                        .and_then(|runtime| {
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
                                                segmented_runtime_v20: None,
                                                segmented_runtime_v22: None,
                                                segmented_runtime_v23: None,
                                                segmented_runtime_v24: runtime.segmented_runtime,
                                                segmented_runtime_v27: None,
                                                dynamic_mask_vocab: runtime.dynamic_mask_vocab,
                                                virtual_runtimes: Vec::new(),
                                                static_virtual_residual_mask: None,
                                                packed_dwa_dense_masks,
                                            })
                                        })
                                    } else if version == PREVIOUS_BOUNDARY_SHARDED_CONSTRAINT_VERSION {
                                        bincode::deserialize::<ConstraintArtifactV23Runtime>(
                                            runtime_section,
                                        )
                                        .and_then(|runtime| {
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
                                                segmented_runtime_v20: None,
                                                segmented_runtime_v22: None,
                                                segmented_runtime_v23: runtime.segmented_runtime,
                                                segmented_runtime_v24: None,
                                                segmented_runtime_v27: None,
                                                dynamic_mask_vocab: runtime.dynamic_mask_vocab,
                                                virtual_runtimes: Vec::new(),
                                                static_virtual_residual_mask: None,
                                                packed_dwa_dense_masks,
                                            })
                                        })
                                    } else if version == PREVIOUS_BOUNDARY_SHARDLESS_CONSTRAINT_VERSION {
                                        bincode::deserialize::<ConstraintArtifactV22Runtime>(
                                            runtime_section,
                                        )
                                        .and_then(|runtime| {
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
                                                segmented_runtime_v20: None,
                                                segmented_runtime_v22: runtime.segmented_runtime,
                                                segmented_runtime_v23: None,
                                                segmented_runtime_v24: None,
                                                segmented_runtime_v27: None,
                                                dynamic_mask_vocab: runtime.dynamic_mask_vocab,
                                                virtual_runtimes: Vec::new(),
                                                static_virtual_residual_mask: None,
                                                packed_dwa_dense_masks,
                                            })
                                        })
                                    } else if version == PREVIOUS_SEGMENTED_MATERIALIZATION_CONSTRAINT_VERSION {
                                        bincode::deserialize::<ConstraintArtifactV21Runtime>(
                                            runtime_section,
                                        )
                                        .and_then(|runtime| {
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
                                                segmented_runtime_v20: runtime.segmented_runtime,
                                                segmented_runtime_v22: None,
                                                segmented_runtime_v23: None,
                                                segmented_runtime_v24: None,
                                                segmented_runtime_v27: None,
                                                dynamic_mask_vocab: None,
                                                virtual_runtimes: Vec::new(),
                                                static_virtual_residual_mask: None,
                                                packed_dwa_dense_masks,
                                            })
                                        })
                                    } else if version == PREVIOUS_COMBINED_CONSTRAINT_VERSION {
                                        bincode::deserialize::<ConstraintArtifactV20Runtime>(
                                            runtime_section,
                                        )
                                        .map(|runtime| DecodedConstraintRuntime {
                                            terminal_live_states: runtime.terminal_live_states,
                                            segmented_runtime_v20: runtime.segmented_runtime,
                                            segmented_runtime_v22: None,
                                            segmented_runtime_v23: None,
                                            segmented_runtime_v24: None,
                                            segmented_runtime_v27: None,
                                            dynamic_mask_vocab: None,
                                            virtual_runtimes: Vec::new(),
                                            static_virtual_residual_mask: None,
                                            packed_dwa_dense_masks: None,
                                        })
                                    } else {
                                        bincode::deserialize::<ConstraintArtifactV15Runtime>(
                                            runtime_section,
                                        )
                                        .map(|runtime| DecodedConstraintRuntime {
                                            terminal_live_states: runtime.terminal_live_states,
                                            segmented_runtime_v20: None,
                                            segmented_runtime_v22: None,
                                            segmented_runtime_v23: None,
                                            segmented_runtime_v24: None,
                                            segmented_runtime_v27: None,
                                            dynamic_mask_vocab: None,
                                            virtual_runtimes: Vec::new(),
                                            static_virtual_residual_mask: None,
                                            packed_dwa_dense_masks: None,
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
                            let result = if uses_external_runtime_sections(version) {
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
                            } else {
                                crate::automata::lexer::tokenizer::artifact_serde::from_fast_bytes(
                                    section,
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
                            let result = if uses_external_runtime_sections(version) {
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
                            } else {
                                crate::runtime::artifact::original_token_map_artifact_serde::from_fast_bytes(section)
                                    .map(|map| Some(DecodedOriginalTokenMap::Materialized(map)))
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
                    let weight_count = if uses_external_runtime_sections(version)
                        || version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
                        || version == PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION
                        || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
                    {
                        Some(crate::ds::weight::PackedRuntimeWeightPool::peek_weight_count(
                            weight_section,
                        )?)
                    } else {
                        None
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
                                uses_external_runtime_sections(version),
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
                                uses_external_runtime_sections(version),
                            );
                        let previous_packed_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_packed(true);
                        let previous_external_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_external(
                                uses_external_runtime_sections(version)
                                    || version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION,
                            );
                        let previous_defer_token_bytes =
                            crate::runtime::artifact::token_bytes_artifact_serde::set_defer_unpack(
                                version == PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION
                                    || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION,
                            );
                        let core_started = profile.then(std::time::Instant::now);
                        let decoded = if uses_external_runtime_sections(version) {
                            if core_section.starts_with(&CURRENT_CORE_MAGIC)
                                || core_section.starts_with(&PREVIOUS_CURRENT_CORE_MAGIC)
                                || core_section.starts_with(&PREVIOUS_PREVIOUS_CURRENT_CORE_MAGIC)
                                || core_section.starts_with(&PREVIOUS_PREVIOUS_PREVIOUS_CURRENT_CORE_MAGIC)
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
                            } else {
                                bincode::deserialize::<ConstraintArtifactV18Core>(core_section)
                                    .map(|artifact| DecodedConstraintCore {
                                        constraint: artifact.constraint,
                                        ignore_expr: artifact.ignore_expr,
                                        terminal_exprs: artifact.terminal_exprs,
                                        terminal_exprs_blob: None,
                                        parser_state_domain_labels: artifact.parser_state_domain_labels,
                                        internal_token_buf_masks: Vec::new(),
                                    })
                                    .map_err(|err| err.to_string())
                            }
                        } else {
                            bincode::deserialize::<ConstraintArtifactV14Core>(core_section)
                                .map(|artifact| DecodedConstraintCore {
                                    constraint: artifact.constraint,
                                    ignore_expr: artifact.ignore_expr,
                                    terminal_exprs: artifact.terminal_exprs,
                                    terminal_exprs_blob: None,
                                    parser_state_domain_labels: artifact.parser_state_domain_labels,
                                    internal_token_buf_masks: artifact.internal_token_buf_masks,
                                })
                                .map_err(|err| err.to_string())
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
                        if uses_external_runtime_sections(version)
                            || version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
                            || version == PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION
                            || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
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
                        } else {
                            let (artifact, deferred_token_bytes, _) = decode_core()?;
                            (artifact, deferred_token_bytes, None)
                        };
                    if let Some(started) = section_started {
                        eprintln!("[glrmask/profile][constraint_section] name=core_total ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                    }
                    Ok((artifact, deferred_token_bytes, packed_weights))
                },
            );
            let parser_dwa = dwa_result.map_err(crate::GlrMaskError::Serialization)?;
            let (table, deferred_table_rules_blob, template_parser_seed) =
                table_result.map_err(crate::GlrMaskError::Serialization)?;
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
            match parser_dwa {
                DecodedParserDwa::Materialized(parser_dwa, inventory) => {
                    constraint.parser_dwa = parser_dwa;
                    constraint.packed_parser_dwa = None;
                    packed_dwa_inventory = inventory;
                }
                DecodedParserDwa::Packed(parser_dwa) => {
                    constraint.parser_dwa = crate::automata::weighted::dwa::DWA::new(0, 0);
                    constraint.packed_parser_dwa = Some(parser_dwa);
                }
            }
            let attach_dwa_ms = attach_dwa_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            constraint.table = table;
            if let Some(seed) = template_parser_seed {
                seed.install(&mut constraint).map_err(crate::GlrMaskError::Serialization)?;
            }
            constraint.deferred_table_rules_blob = deferred_table_rules_blob;
            constraint.deferred_table_rules = Default::default();
            let invert_started = profile.then(std::time::Instant::now);
            if !uses_external_runtime_sections(version)
                && version != PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
            {
                constraint.internal_token_to_tokens = invert_original_token_map(
                    &constraint.original_token_to_internal,
                    artifact.internal_token_buf_masks.len(),
                )
                .map_err(crate::GlrMaskError::Serialization)?;
            }
            let invert_ms = invert_started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            constraint.ignore_expr = artifact.ignore_expr;
            constraint.parser_state_domain_labels = artifact.parser_state_domain_labels;
            constraint.deferred_terminal_exprs_blob = artifact.terminal_exprs_blob;
            constraint.deferred_terminal_exprs = Default::default();
            constraint.deferred_composition_metadata_blob = if let Some(section) = composition_metadata_section {
                let (_, prepared)=crate::compiler::boundary_precomputed_completion::split_envelope(section)
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
                if let Some(segmented_runtime) = runtime.segmented_runtime_v20 {
                    restore_segmented_runtime_v20(&mut constraint, segmented_runtime)?;
                }
                if let Some(segmented_runtime) = runtime.segmented_runtime_v22 {
                    restore_segmented_runtime_v22(&mut constraint, segmented_runtime)?;
                }
                if let Some(segmented_runtime) = runtime.segmented_runtime_v23 {
                    restore_segmented_runtime_v23(&mut constraint, segmented_runtime, false)?;
                }
                if let Some(segmented_runtime) = runtime.segmented_runtime_v24 {
                    restore_segmented_runtime_v24(&mut constraint, segmented_runtime)?;
                }
                if let Some(segmented_runtime) = runtime.segmented_runtime_v27 {
                    restore_segmented_runtime_v27(&mut constraint, segmented_runtime)?;
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
        } else if version == PREVIOUS_UNCOMPRESSED_CONSTRAINT_VERSION {
            let previous_dwa_mode = crate::automata::weighted::dwa::set_packed_serde(true);
            let decoded = bincode::deserialize::<ConstraintArtifactV13>(serialized);
            crate::automata::weighted::dwa::set_packed_serde(previous_dwa_mode);
            // The pool activator is the first v13 field and therefore installs
            // the Weight reference table before Constraint is deserialized.
            // Always clear it, including after a malformed later field.
            crate::ds::weight::end_pooled_weight_serde_decode();
            let artifact = decoded
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
            let mut constraint = artifact.constraint;
            constraint.ignore_expr = artifact.ignore_expr;
            constraint.parser_state_domain_labels = artifact.parser_state_domain_labels;
            constraint.internal_token_buf_masks = artifact.internal_token_buf_masks;
            Arc::make_mut(&mut constraint.tokenizer).restore_terminal_exprs(artifact.terminal_exprs)
                .map_err(crate::GlrMaskError::Serialization)?;
            constraint
        } else if version == PREVIOUS_DOMAIN_LABELS_CONSTRAINT_VERSION {
            let artifact: ConstraintArtifactV12 = bincode::deserialize(serialized)
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
            let mut constraint = artifact.constraint;
            constraint.ignore_expr = artifact.ignore_expr;
            constraint.parser_state_domain_labels = artifact.parser_state_domain_labels;
            Arc::make_mut(&mut constraint.tokenizer).restore_terminal_exprs(artifact.terminal_exprs)
                .map_err(crate::GlrMaskError::Serialization)?;
            constraint
        } else if version == PREVIOUS_TERMINAL_EXPRS_CONSTRAINT_VERSION {
            let artifact: ConstraintArtifactV11 = bincode::deserialize(serialized)
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
            let mut constraint = artifact.constraint;
            constraint.ignore_expr = artifact.ignore_expr;
            Arc::make_mut(&mut constraint.tokenizer).restore_terminal_exprs(artifact.terminal_exprs)
                .map_err(crate::GlrMaskError::Serialization)?;
            constraint
        } else if version == PREVIOUS_EXPRLESS_CONSTRAINT_VERSION {
            let artifact: ConstraintArtifactV10 = bincode::deserialize(serialized)
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
            let mut constraint = artifact.constraint;
            constraint.ignore_expr = artifact.ignore_expr;
            constraint
        } else {
            bincode::deserialize::<DeserializedConstraint>(serialized)
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?
                .0
        };
        let deserialize_ms = deserialize_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if !constraint.parser_state_domain_labels.is_empty() {
            if constraint.parser_state_domain_labels.len() != constraint.parser_symbol_count() as usize {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "parser-state domain map has {} entries for {} parser states",
                    constraint.parser_state_domain_labels.len(),
                    constraint.parser_symbol_count(),
                )));
            }
            let first_synthetic = constraint.parser_symbol_count() as i64;
            let default_label = crate::compiler::glr::labels::DEFAULT_LABEL as i64;
            for &label in &constraint.parser_state_domain_labels {
                if label == i32::MAX {
                    continue;
                }
                let label64 = label as i64;
                if label64 < first_synthetic || label64 >= default_label {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "invalid parser-state domain label {label} for {} parser states",
                        constraint.parser_symbol_count(),
                    )));
                }
            }
        }
        // Rebuild this tiny token-domain summary from the authoritative
        // packed/materialized inventory after all external sections load.
        // No vocabulary trie is built and it is not serialized a second time.
        constraint.empty_byte_token_ids = constraint.token_bytes_iter()
            .filter_map(|(id, bytes)| bytes.is_empty().then_some(id))
            .collect::<Vec<u32>>().into();
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
                if let Some(inventory) = packed_dwa_inventory.take() {
                    crate::automata::weighted::dwa::install_packed_decode_token_set_inventory(
                        inventory,
                    );
                }
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
        if uses_external_runtime_sections(version)
            || version == PREVIOUS_VOCAB_SECTION_CONSTRAINT_VERSION
            || version == PREVIOUS_FAST_RUNTIME_CONSTRAINT_VERSION
            || version == PREVIOUS_PACKED_RUNTIME_CONSTRAINT_VERSION
        {
            constraint.serialized_artifact_cache = current_backing.or_else(|| {
                Some(owned_artifact.unwrap_or_else(|| std::sync::Arc::new(bytes.to_vec())))
            });
        }
        if !matches!(version, CONSTRAINT_VERSION | TEMPLATE_CONSTRAINT_VERSION) && constraint.boundary_candidate_summary.get().is_none() {
            let _ = constraint.boundary_candidate_summary.set(
                crate::runtime::BoundaryCandidateSummary::Unknown {
                    reason: crate::runtime::SummaryUnavailable::LegacyArtifact,
                },
            );
        }
        if let Some(wire)=prepared_completion_wire {
            let started=std::time::Instant::now();
            let prepared=crate::compiler::boundary_precomputed_completion::PreparedCompletion::load(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vocab;
    use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
    use crate::runtime::CommitTemplateDfas;
    use std::sync::Arc;

    #[test]
    fn loaded_partitioned_bounded_strings_keep_first_key_prefix_live() {
        use crate::automata::lexer::Lexer;
        let vocab = Vocab::new(
            [b"{\"".as_slice(), b"{", b"\"", b"a", b"b", b"\": \"", b"\", \"", b"\"}", b"x", b" ", b"xx"]
                .into_iter().enumerate().map(|(id, bytes)| (id as u32, bytes.to_vec())).collect(),
        );
        let schema = r#"{
            "type":"object",
            "properties":{
                "a":{"type":"string","minLength":1,"maxLength":200,
                     "pattern":"^(?:\\S+\\s+){0,19}\\S+$"},
                "b":{"type":"string","minLength":1,"maxLength":100}
            },
            "required":["a","b"],"additionalProperties":false
        }"#;
        let original = Constraint::compile(crate::Grammar::json_schema(schema), &vocab).unwrap();
        let loaded = Constraint::load(original.save()).unwrap();
        assert!(original.tokenizer.has_compressed_transition_segments());
        assert!(loaded.tokenizer.has_compressed_transition_segments());
        assert_eq!(original.tokenizer.num_states(), loaded.tokenizer.num_states());
        for state in 0..original.tokenizer.num_states().min(20) {
            assert_eq!(original.tokenizer.step(state, b'"'), loaded.tokenizer.step(state, b'"'),
                "opening-quote transition differs at tokenizer state {state}");
        }
        assert_eq!(original.start().mask(), loaded.start().mask());
        let mut expected = original.start();
        let mut actual = loaded.start();
        expected.commit_token(0).expect("compiled key prefix must commit");
        actual.commit_token(0).expect("loaded key prefix admitted by the mask must commit");
        assert_eq!(expected.mask(), actual.mask());
        // Complete {"a": "x", "b": "x"}, checking every next-token mask.
        for token in [3, 5, 8, 6, 4, 5, 8, 7] {
            expected.commit_token(token).unwrap();
            actual.commit_token(token).unwrap();
            assert_eq!(expected.mask(), actual.mask());
        }
        assert!(expected.is_accepting());
        assert!(actual.is_accepting());
    }

    #[test]
    fn packed_reencode_preserves_non_dwa_ids_after_cache_invalidation() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()), (1, b"a".to_vec()), (2, b"b".to_vec()),
            (3, b"y".to_vec()), (4, b"xay".to_vec()), (5, b"ab".to_vec()),
            (7, b"a".to_vec()), (8, Vec::new()),
        ]);
        let original = Constraint::from_ebnf(
            r#"start ::= @token(9) "b" | "a" "y""#,
            &vocab,
        ).unwrap();
        let mut loaded = Constraint::load(original.save()).unwrap();
        let packed = loaded.packed_non_dwa_weights.as_ref().unwrap();
        let distinct_ids = packed.parser_top_accept.values()
            .copied().collect::<std::collections::BTreeSet<_>>();
        assert!(distinct_ids.len() >= 2, "regression needs distinct packed references");

        // Linking can invalidate a cached artifact while retaining its packed
        // runtime pools. Re-serialization must preserve the sidecar IDs, not
        // intern the empty structural Weight placeholders by pointer.
        loaded.serialized_artifact_cache = None;
        let again = Constraint::load(loaded.save()).unwrap();
        assert_eq!(loaded.packed_non_dwa_weights.as_ref().unwrap().parser_top_accept,
                   again.packed_non_dwa_weights.as_ref().unwrap().parser_top_accept);
        for path in [&[][..], &[1], &[1, 3], &[9], &[9, 2]] {
            let mut expected = original.start();
            let mut actual = again.start();
            for &id in path {
                assert_eq!(expected.mask(), actual.mask());
                expected.commit_token(id).unwrap();
                actual.commit_token(id).unwrap();
            }
            assert_eq!(expected.mask(), actual.mask(), "after {path:?}");
            assert_eq!(expected.is_accepting(), actual.is_accepting());
        }
    }

    fn tiny_constraint() -> Constraint {
        Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                t B ::= "b";
                nt start ::= A B;
            "#,
            &Vocab::new(vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"ab".to_vec()),
            ]),
        )
        .unwrap()
    }

    fn sample_boundary_fingerprint() -> crate::runtime::BoundaryCandidateFingerprint {
        crate::runtime::BoundaryCandidateFingerprint {
            algorithm_version: 77,
            component_semantics: [7; 32],
            public_interface: [8; 32],
            vocabulary: [9; 32],
        }
    }

    fn ignored_constraint() -> Constraint {
        Constraint::from_glrm_grammar(
            r#"
                start start;
                ignore WS;
                t WS ::= " "+;
                nt start ::= "a";
            "#,
            &Vocab::new(vec![(0, b"a".to_vec()), (1, b" ".to_vec())]),
        )
        .unwrap()
    }

    fn rewrite_current_runtime_as_previous(
        saved: &[u8],
        version: u16,
        section_magic: [u8; 4],
        section_header_len: usize,
        runtime: &[u8],
    ) -> Vec<u8> {
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            CONSTRAINT_VERSION,
        );
        let payload = &saved[CONSTRAINT_HEADER_LEN..];
        let (
            weight,
            dwa,
            table,
            core,
            _current_runtime,
            token_bytes,
            original_map,
            tokenizer,
            internal_masks,
            token_mask_cache,
            composition_metadata,
        ) = v27_sections(payload).unwrap();
        let sections: [&[u8]; 11] = [
            weight,
            dwa,
            table,
            core,
            runtime,
            token_bytes,
            original_map,
            tokenizer,
            internal_masks,
            token_mask_cache,
            composition_metadata,
        ];
        let payload_len = section_header_len
            + sections.iter().map(|section| section.len()).sum::<usize>();
        let mut previous = Vec::with_capacity(payload_len);
        previous.extend_from_slice(&section_magic);
        for section in &sections {
            previous.extend_from_slice(&(section.len() as u64).to_le_bytes());
        }
        for section in sections {
            previous.extend_from_slice(section);
        }
        envelope(version, &previous)
    }

    #[test]
    #[ignore = "pre-release: historical artifact compatibility is not a release requirement"]
    fn v23_segmented_runtime_remains_loadable_without_compact_parser_metadata() {
        let current = tiny_constraint();
        let saved = current.save();
        let payload = &saved[CONSTRAINT_HEADER_LEN..];
        let (_, _, _, _, current_runtime, _, _, _, _, _, _) = v27_sections(payload).unwrap();
        let current_runtime =
            bincode::deserialize::<ConstraintArtifactCurrentRuntime>(current_runtime).unwrap();
        let mut root_entry_terminals =
            crate::ds::bitset::BitSet::new(current.table.num_terminals as usize);
        for terminal in 0..current.table.num_terminals as usize {
            root_entry_terminals.set(terminal);
        }
        let previous_runtime = bincode::serialize(&ConstraintArtifactV23Runtime {
            terminal_live_states: current_runtime.terminal_live_states,
            segmented_runtime: Some(SegmentedRuntimeArtifactV23 {
                materialized_static_component_parser: None,
                materialized_static_parser_state_domain_labels: Vec::new(),
                components: vec![SegmentedParserComponentV22 {
                    constraint_artifact: current.save(),
                    tokenizer_state_offset: 0,
                    terminal_offset: 0,
                    root_entry_terminals,
                    root_disallowed_terminal: None,
                    global_to_local_parser_state: (0..current.table.num_states).collect(),
                }],
                segmented_mask_authoritative: false,
                segmented_component_union_root_dispatch: Vec::new(),
                boundary_shards: Vec::new(),
            }),
            dynamic_mask_vocab: current_runtime.dynamic_mask_vocab,
            packed_dwa_dense_mask_ids: current_runtime.packed_dwa_dense_mask_ids,
            packed_dwa_dense_mask_rows: current_runtime.packed_dwa_dense_mask_rows,
        })
        .unwrap();
        let previous = rewrite_current_runtime_as_previous(
            &saved,
            PREVIOUS_BOUNDARY_SHARDED_CONSTRAINT_VERSION,
            V23_SECTION_MAGIC,
            V23_SECTION_HEADER_LEN,
            &previous_runtime,
        );
        let loaded = Constraint::load(previous).unwrap();
        assert!(!loaded.uses_compact_segmented_parser_runtime());
        let overlay = loaded.static_dynamic_overlay.as_ref().unwrap();
        assert_eq!(overlay.segmented_parser_components.len(), 1);
        assert_eq!(
            overlay.segmented_parser_components[0].global_to_local_parser_state,
            (0..loaded.table.num_states).collect::<Vec<_>>(),
        );
        let mut current_state = current.start();
        let mut loaded_state = loaded.start();
        assert_eq!(current_state.mask(), loaded_state.mask());
        current_state.commit_token(2).unwrap();
        loaded_state.commit_token(2).unwrap();
        assert_eq!(current_state.is_accepting(), loaded_state.is_accepting());
        assert!(current_state.is_accepting());
    }

    #[test]
    #[ignore = "pre-release: historical artifact compatibility is not a release requirement"]
    fn v24_legacy_segmented_runtime_remains_loadable() {
        let current = tiny_constraint();
        let saved = current.save();
        let payload = &saved[CONSTRAINT_HEADER_LEN..];
        let (_, _, _, _, current_runtime, _, _, _, _, _, _) = v27_sections(payload).unwrap();
        let current_runtime =
            bincode::deserialize::<ConstraintArtifactCurrentRuntime>(current_runtime).unwrap();
        let mut root_entry_terminals =
            crate::ds::bitset::BitSet::new(current.table.num_terminals as usize);
        for terminal in 0..current.table.num_terminals as usize {
            root_entry_terminals.set(terminal);
        }
        let previous_runtime = bincode::serialize(&ConstraintArtifactV24Runtime {
            terminal_live_states: current_runtime.terminal_live_states,
            segmented_runtime: Some(SegmentedRuntimeArtifactV24 {
                materialized_static_component_parser: None,
                materialized_static_parser_state_domain_labels: Vec::new(),
                components: vec![SegmentedParserComponentV24 {
                    constraint_artifact: current.save(),
                    tokenizer_state_offset: 0,
                    terminal_offset: 0,
                    global_terminal_aliases: Vec::new(),
                    local_tsid_to_global_tsids: Vec::new(),
                    root_entry_terminals,
                    root_disallowed_terminal: None,
                    global_to_local_parser_state: (0..current.table.num_states).collect(),
                }],
                segmented_parser_links: Vec::new(),
                segmented_parser_state_offsets: vec![0],
                segmented_mask_authoritative: false,
                segmented_component_union_root_dispatch: Vec::new(),
                boundary_shards: Vec::new(),
            }),
            dynamic_mask_vocab: current_runtime.dynamic_mask_vocab,
            packed_dwa_dense_mask_ids: current_runtime.packed_dwa_dense_mask_ids,
            packed_dwa_dense_mask_rows: current_runtime.packed_dwa_dense_mask_rows,
        })
        .unwrap();
        let previous = rewrite_current_runtime_as_previous(
            &saved,
            PREVIOUS_RECURSIVE_PARSER_CONSTRAINT_VERSION,
            V24_SECTION_MAGIC,
            V24_SECTION_HEADER_LEN,
            &previous_runtime,
        );
        let loaded = Constraint::load(previous).unwrap();
        assert!(!loaded.uses_compact_segmented_parser_runtime());
        let overlay = loaded.static_dynamic_overlay.as_ref().unwrap();
        assert_eq!(overlay.segmented_parser_state_offsets, vec![0]);
        assert_eq!(loaded.start().mask(), current.start().mask());
    }

    #[test]
    #[ignore = "pre-release: historical artifact compatibility is not a release requirement"]
    fn v24_static_boundary_with_cleared_start_states_stays_exact_legacy() {
        let current = tiny_constraint();
        let saved = current.save();
        let payload = &saved[CONSTRAINT_HEADER_LEN..];
        let (_, _, _, _, current_runtime, _, _, _, _, _, _) = v27_sections(payload).unwrap();
        let current_runtime =
            bincode::deserialize::<ConstraintArtifactCurrentRuntime>(current_runtime).unwrap();

        let mut root_entry_terminals =
            crate::ds::bitset::BitSet::new(current.table.num_terminals as usize);
        for terminal in 0..current.table.num_terminals as usize {
            root_entry_terminals.set(terminal);
        }
        let projection = (0..current.table.num_states).collect::<Vec<_>>();
        let max_internal_token = current
            .internal_token_to_tokens
            .len()
            .checked_sub(1)
            .unwrap_or(0) as u32;
        let empty_static_b = SegmentedBoundaryParserV23 {
            parser_dwa: crate::automata::weighted_u32::dwa::DWA::new(
                current.internal_tsid_count() as u32,
                max_internal_token,
            ),
            uses_composed_tsid_coordinate: true,
            tokenizer_state_to_tsid: Vec::new(),
            internal_token_to_originals: current.internal_token_to_tokens.clone(),
        };
        let previous_runtime = bincode::serialize(&ConstraintArtifactV24Runtime {
            terminal_live_states: current_runtime.terminal_live_states,
            segmented_runtime: Some(SegmentedRuntimeArtifactV24 {
                materialized_static_component_parser: None,
                materialized_static_parser_state_domain_labels: Vec::new(),
                components: vec![
                    SegmentedParserComponentV24 {
                        constraint_artifact: current.save(),
                        tokenizer_state_offset: 0,
                        terminal_offset: 0,
                        global_terminal_aliases: Vec::new(),
                        local_tsid_to_global_tsids: Vec::new(),
                        root_entry_terminals: root_entry_terminals.clone(),
                        root_disallowed_terminal: None,
                        global_to_local_parser_state: projection.clone(),
                    },
                    SegmentedParserComponentV24 {
                        constraint_artifact: current.save(),
                        tokenizer_state_offset: 0,
                        terminal_offset: 0,
                        global_terminal_aliases: Vec::new(),
                        local_tsid_to_global_tsids: Vec::new(),
                        root_entry_terminals,
                        root_disallowed_terminal: None,
                        global_to_local_parser_state: projection.clone(),
                    },
                ],
                // Mimic the late-v24 transitional shape: linker metadata was
                // present, while recursive runtime had already cleared legacy
                // offsets/root dispatch/start-state ownership.
                segmented_parser_links: vec![SegmentedParserLinkV24 {
                    parent_component: 0,
                    slot_terminal: 0,
                    child_component: 1,
                    child_start: 0,
                    return_pop: 1,
                    child_start_nullable: false,
                }],
                segmented_parser_state_offsets: Vec::new(),
                segmented_mask_authoritative: true,
                segmented_component_union_root_dispatch: Vec::new(),
                boundary_shards: vec![SegmentedBoundaryShardV23 {
                    scope: SegmentedBoundaryShardScopeV23::Component {
                        start_component: 0,
                        start_parser_states: crate::ds::bitset::BitSet::new(0),
                        accepts_empty_stack: true,
                    },
                    candidate_tokens: None,
                    backend: SegmentedBoundaryShardBackendV23::StaticParser(empty_static_b),
                }],
            }),
            dynamic_mask_vocab: current_runtime.dynamic_mask_vocab,
            packed_dwa_dense_mask_ids: current_runtime.packed_dwa_dense_mask_ids,
            packed_dwa_dense_mask_rows: current_runtime.packed_dwa_dense_mask_rows,
        })
        .unwrap();
        let previous = rewrite_current_runtime_as_previous(
            &saved,
            PREVIOUS_RECURSIVE_PARSER_CONSTRAINT_VERSION,
            V24_SECTION_MAGIC,
            V24_SECTION_HEADER_LEN,
            &previous_runtime,
        );
        let loaded = Constraint::load(previous).unwrap();
        assert!(!loaded.uses_compact_segmented_parser_runtime());
        let overlay = loaded.static_dynamic_overlay.as_ref().unwrap();
        assert!(overlay.segmented_parser_links.is_empty());
        assert!(overlay.segmented_parser_state_offsets.is_empty());
        let shard = overlay.segmented_parser_components[0]
            .boundary
            .as_ref()
            .expect("legacy static shard must remain attached to its component");
        let mut expected_starts =
            crate::ds::bitset::BitSet::new(current.table.num_states as usize);
        for state in 0..current.table.num_states as usize {
            expected_starts.set(state);
        }
        assert_eq!(shard.start_parser_states, expected_starts);

        let mut expected = current.start();
        let mut actual = loaded.start();
        assert_eq!(actual.mask(), expected.mask());
        expected.commit_token(2).unwrap();
        actual.commit_token(2).unwrap();
        assert_eq!(actual.is_accepting(), expected.is_accepting());
        assert!(actual.is_accepting());
    }

    #[test]
    fn static_bounded_uri_virtual_projection_roundtrips_with_packed_vocab() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"x:".to_vec()),
            (2, b"a".to_vec()),
            (3, b"aa".to_vec()),
        ]);
        let schema = r#"{"type":"string","format":"uri","minLength":1,"maxLength":5000}"#;
        let constraint = Constraint::from_json_schema(schema, &vocab)
            .expect("large bounded URI should compile through the exact Static residual lane");
        assert!(constraint.tokenizer.has_virtual_residual_runtime());
        assert!(constraint.dynamic_mask_vocab.mask_projection_tokenizer().is_some());

        let saved = constraint.save();
        assert!(saved.windows(4).any(|window| window == STATIC_RESIDUAL_MASK_MAGIC));
        assert!(saved.windows(4).any(|window| window == b"TKS3"));
        assert!(saved.windows(4).any(|window| window == b"BCO2"));
        let loaded = Constraint::load(saved.clone())
            .expect("Static residual artifact should round-trip");
        assert!(loaded.tokenizer.has_virtual_residual_runtime());
        assert!(loaded.dynamic_mask_vocab.mask_projection_tokenizer().is_some());
        assert!(loaded.token_bytes.is_empty());
        assert!(loaded.packed_token_bytes.is_some());

        let compare_mask = |prefix: &[u8]| {
            let mut expected = constraint.start();
            let mut actual = loaded.start();
            expected.commit_bytes(prefix).unwrap();
            actual.commit_bytes(prefix).unwrap();
            assert_eq!(actual.mask(), expected.mask(), "prefix length {}", prefix.len());
        };
        compare_mask(b"\"x:");

        let mut at_max = b"\"x:".to_vec();
        at_max.extend(std::iter::repeat_n(b'a', 4_998));
        compare_mask(&at_max);

        let mut expected = constraint.start();
        let mut actual = loaded.start();
        expected.commit_bytes(&at_max).unwrap();
        actual.commit_bytes(&at_max).unwrap();
        expected.commit_bytes(b"\"").unwrap();
        actual.commit_bytes(b"\"").unwrap();
        assert!(expected.is_accepting());
        assert!(actual.is_accepting());

        let mut over_max = b"\"x:".to_vec();
        over_max.extend(std::iter::repeat_n(b'a', 4_999));
        assert!(constraint.start().commit_bytes(&over_max).is_err());
        assert!(loaded.start().commit_bytes(&over_max).is_err());

        let mut corrupted = saved;
        let marker = corrupted
            .windows(4)
            .position(|window| window == STATIC_RESIDUAL_MASK_MAGIC)
            .expect("SRM3 marker must be present");
        corrupted[marker] ^= 0x01;
        assert!(
            Constraint::load(corrupted).is_err(),
            "corrupted SRM3 marker must fail closed",
        );
    }

    #[test]
    fn fresh_packed_non_dwa_weight_runtime_matches_materialized_and_roundtrips() {
        let baseline = tiny_constraint();
        let mut packed = tiny_constraint();
        let (weights, _, _) = constraint_serialized_weight_pool_with_ids(&packed);
        assert!(!weights.is_empty());
        assert!(packed.packed_non_dwa_weights.is_none());
        assert!(compact_non_dwa_weight_runtime_if_at_least(&mut packed, 0));
        assert!(packed.packed_non_dwa_weights.is_some());

        let mut expected = baseline.start();
        let mut actual = packed.start();
        assert_eq!(actual.mask(), expected.mask());
        expected.commit_token(0).unwrap();
        actual.commit_token(0).unwrap();
        assert_eq!(actual.mask(), expected.mask());

        let loaded = Constraint::load(packed.save()).unwrap();
        let mut loaded_state = loaded.start();
        assert_eq!(loaded_state.mask(), baseline.start().mask());
        loaded_state.commit_token(0).unwrap();
        assert_eq!(loaded_state.mask(), expected.mask());
    }

    #[test]
    fn compact_seed_terminal_dense_deduplicates_and_roundtrips() {
        let mut original = crate::runtime::artifact::SeedTerminalDenseMasks::default();
        original.insert((3, 7), Arc::<[u64]>::from([1, 2, 3, 4]));
        original.insert((9, 11), Arc::<[u64]>::from([1, 2, 3, 4]));
        original.insert((12, 13), Arc::<[u64]>::from([8, 9]));

        let compact = SeedTerminalDenseCompact::from_map(&original);
        assert_eq!(compact.masks.len(), 2);
        assert_eq!(compact.entries.len(), 3);
        let decoded = compact.into_map().unwrap();
        assert_eq!(decoded, original);
        assert!(Arc::ptr_eq(
            decoded.get(&(3, 7)).unwrap(),
            decoded.get(&(9, 11)).unwrap(),
        ));
    }

    #[test]
    fn constraint_envelope_roundtrips_and_rejects_previous_formats() {
        let constraint = tiny_constraint();
        let saved = constraint.save();
        assert!(saved.starts_with(&CONSTRAINT_MAGIC));
        assert!(bincode::deserialize::<DeserializedConstraint>(&saved).is_err());
        let loaded = Constraint::load(&saved).unwrap();
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let raw = bincode::serialize(&SerializedConstraint(&constraint)).unwrap();
        assert!(Constraint::load(&raw)
            .unwrap_err()
            .to_string()
            .contains("header"));

        let mut previous_version = saved;
        previous_version[8..10].copy_from_slice(&2u16.to_le_bytes());
        assert!(Constraint::load(&previous_version)
            .unwrap_err()
            .to_string()
            .contains("unsupported"));

        let mut previous_schema = constraint.save();
        previous_schema[8..10].copy_from_slice(&8u16.to_le_bytes());
        assert!(Constraint::load(&previous_schema)
            .unwrap_err()
            .to_string()
            .contains("unsupported"));

        // v25 was the last development wire before recursive compiler-view
        // detachment. Pre-release artifacts are intentionally not migrated:
        // reject them by envelope version rather than trying to deserialize a
        // structurally different payload under the current schema.
        let mut previous_development = constraint.save();
        previous_development[8..10].copy_from_slice(&25u16.to_le_bytes());
        assert!(Constraint::load(&previous_development)
            .unwrap_err()
            .to_string()
            .contains("unsupported"));
    }

    #[test]
    fn load_moves_owned_vec_and_accepts_borrowed_bytes() {
        let constraint = tiny_constraint();

        let owned = constraint.save();
        let owned_ptr = owned.as_ptr();
        let loaded_owned = Constraint::load(owned).unwrap();
        let owned_backing = loaded_owned
            .serialized_artifact_cache
            .as_ref()
            .expect("current owned load should retain artifact backing");
        assert_eq!(owned_backing.as_ptr(), owned_ptr);

        let borrowed = constraint.save();
        let borrowed_ptr = borrowed.as_ptr();
        let loaded_borrowed = Constraint::load(borrowed.as_slice()).unwrap();
        let borrowed_backing = loaded_borrowed
            .serialized_artifact_cache
            .as_ref()
            .expect("current borrowed load should create retained backing");
        assert_ne!(borrowed_backing.as_ptr(), borrowed_ptr);
        assert_eq!(loaded_borrowed.start().mask(), constraint.start().mask());

        // `&Vec<u8>` is a common caller shape and should remain ergonomic.
        let loaded_vec_ref = Constraint::load(&borrowed).unwrap();
        assert_eq!(loaded_vec_ref.start().mask(), constraint.start().mask());
    }

    #[test]
    fn small_public_compile_defers_first_save_artifact() {
        let vocab = crate::Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let constraint = Constraint::compile(
            crate::Grammar::glrm("glrm 1;\nstart start;\nt A = \"a\";\nnt start = A;\n"),
            &vocab,
        )
        .unwrap();
        assert!(
            constraint.serialized_artifact_cache.is_none(),
            "ordinary small compiles should not prepay a complete save artifact",
        );
        let saved = constraint.save();
        let loaded = Constraint::load(saved).unwrap();
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn owned_load_keeps_token_mask_prefix_matrix_in_artifact_backing() {
        let constraint = tiny_constraint();
        let saved = constraint.save();
        let loaded = Constraint::load(saved).unwrap();

        let backing = loaded
            .serialized_artifact_cache
            .as_ref()
            .expect("owned current load should retain artifact backing");
        let prefix = loaded
            .word_group_prefix_buf_masks
            .as_contiguous()
            .expect("current token-mask prefix matrix should be contiguous");
        assert!(!prefix.is_empty());
        let backing_start = backing.as_ptr() as usize;
        let backing_end = backing_start + backing.len();
        let prefix_start = prefix.as_ptr() as usize;
        let prefix_end = prefix_start + std::mem::size_of_val(prefix);
        assert!(
            prefix_start >= backing_start && prefix_end <= backing_end,
            "loaded token-mask prefix matrix should borrow the retained artifact allocation"
        );

        let mut expected = constraint.start();
        let mut actual = loaded.start();
        assert_eq!(actual.mask(), expected.mask());
        expected.commit_token(0).unwrap();
        actual.commit_token(0).unwrap();
        assert_eq!(actual.mask(), expected.mask());
    }

    #[test]
    fn current_save_handles_packed_dwa_without_fast_wire_length() {
        let mut constraint = tiny_constraint();
        let packed = crate::automata::weighted::dwa::PackedRuntimeDwa::from_dwa(
            &constraint.parser_dwa,
        )
        .unwrap();
        assert!(
            packed.fast_wire_len().is_none(),
            "tiny fallback fixture should require actual wire emission for sizing",
        );
        let wire = packed.fast_wire_bytes();
        assert!(wire.starts_with(b"DWF"));

        // This is the fallback branch the old DWF5-specific regression was
        // really protecting: when no exact precomputed wire length exists, the
        // outer serializer must use the actual emitted wire rather than a stale
        // estimate.
        constraint.packed_parser_dwa = Some(Arc::new(packed));
        let saved = constraint.save();
        let loaded = Constraint::load(&saved).unwrap();
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    #[ignore = "pre-release: historical artifact compatibility is not a release requirement"]
    fn constraint_envelope_loads_legacy_payloads() {
        let constraint = tiny_constraint();
        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            CONSTRAINT_VERSION
        );
        let raw = bincode::serialize(&SerializedConstraint(&constraint)).unwrap();

        let loaded = Constraint::load(&envelope(LEGACY_CONSTRAINT_VERSION, &raw))
            .expect("legacy artifact should remain loadable");

        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    #[ignore = "pre-release: historical artifact compatibility is not a release requirement"]
    fn constraint_envelope_loads_previous_compressed_payload_without_ignore_descriptor() {
        let constraint = ignored_constraint();
        assert!(constraint.ignore_expr.is_some());
        let raw = bincode::serialize(&SerializedConstraint(&constraint)).unwrap();
        let compressed = zstd::bulk::compress(&raw, CONSTRAINT_COMPRESSION_LEVEL).unwrap();
        let mut payload = Vec::with_capacity(COMPRESSED_PAYLOAD_HEADER_LEN + compressed.len());
        payload.extend_from_slice(&(raw.len() as u64).to_le_bytes());
        payload.extend_from_slice(&compressed);

        let loaded = Constraint::load(&envelope(
            PREVIOUS_COMPRESSED_CONSTRAINT_VERSION,
            &payload,
        ))
        .expect("the previous compressed wire layout should remain loadable");

        assert!(loaded.ignore_expr.is_none());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    #[ignore = "pre-release: historical artifact compatibility is not a release requirement"]
    fn constraint_envelope_loads_previous_exprless_v10_payload() {
        let constraint = ignored_constraint();
        let raw = bincode::serialize(&ConstraintArtifactV10Ref {
            constraint: &constraint,
            ignore_expr: &constraint.ignore_expr,
        })
        .unwrap();
        let compressed = zstd::bulk::compress(&raw, CONSTRAINT_COMPRESSION_LEVEL).unwrap();
        let mut payload = Vec::with_capacity(COMPRESSED_PAYLOAD_HEADER_LEN + compressed.len());
        payload.extend_from_slice(&(raw.len() as u64).to_le_bytes());
        payload.extend_from_slice(&compressed);

        let loaded = Constraint::load(&envelope(PREVIOUS_EXPRLESS_CONSTRAINT_VERSION, &payload))
            .expect("v10 exprless artifact should remain loadable");

        assert_eq!(loaded.ignore_expr, constraint.ignore_expr);
        assert!(loaded.tokenizer.terminal_exprs().is_none());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    #[ignore = "pre-release: historical artifact compatibility is not a release requirement"]
    fn constraint_envelope_loads_previous_v11_terminal_expr_payload() {
        let constraint = ignored_constraint();
        let raw = bincode::serialize(&ConstraintArtifactV11Ref {
            constraint: &constraint,
            ignore_expr: &constraint.ignore_expr,
            terminal_exprs: constraint.tokenizer.terminal_exprs(),
        })
        .unwrap();
        let compressed = zstd::bulk::compress(&raw, CONSTRAINT_COMPRESSION_LEVEL).unwrap();
        let mut payload = Vec::with_capacity(COMPRESSED_PAYLOAD_HEADER_LEN + compressed.len());
        payload.extend_from_slice(&(raw.len() as u64).to_le_bytes());
        payload.extend_from_slice(&compressed);

        let loaded = Constraint::load(&envelope(
            PREVIOUS_TERMINAL_EXPRS_CONSTRAINT_VERSION,
            &payload,
        ))
        .expect("v11 terminal-expression artifact should remain loadable");

        assert_eq!(loaded.ignore_expr, constraint.ignore_expr);
        assert_eq!(loaded.tokenizer.terminal_exprs(), constraint.tokenizer.terminal_exprs());
        assert!(loaded.parser_state_domain_labels.is_empty());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn current_constraint_artifact_rejects_invalid_parser_state_domain_map() {
        let mut constraint = ignored_constraint();
        constraint.parser_state_domain_labels = vec![0];
        let error = Constraint::load(&constraint.save()).unwrap_err().to_string();
        assert!(error.contains("parser-state domain map") || error.contains("domain label"));
    }

    #[test]
    fn current_constraint_artifact_preserves_parser_state_domain_labels() {
        let mut constraint = ignored_constraint();
        constraint.parser_state_domain_labels =
            vec![i32::MAX; constraint.parser_symbol_count() as usize];
        if let Some(first) = constraint.parser_state_domain_labels.first_mut() {
            *first = constraint.parser_symbol_count() as i32;
        }
        let loaded = Constraint::load(&constraint.save()).unwrap();
        assert_eq!(
            loaded.parser_state_domain_labels,
            constraint.parser_state_domain_labels,
        );
    }

    #[test]
    fn current_constraint_artifact_preserves_static_dynamic_overlay() {
        let mut constraint = tiny_constraint();
        constraint.static_dynamic_overlay = Some(crate::runtime::artifact::StaticDynamicOverlayMetadata {
            terminal_offsets: vec![0, 3, 7],
            tokenizer_state_offsets: vec![1, 11, 29],
            repair_terminals: vec![false, true, false, true],
            non_parent_only_parser_states: vec![true, false, true],
            ..Default::default()
        });

        let loaded = Constraint::load(&constraint.save()).unwrap();
        let overlay = loaded
            .static_dynamic_overlay
            .as_ref()
            .expect("current artifact should preserve composition overlay metadata");
        assert_eq!(overlay.terminal_offsets, vec![0, 3, 7]);
        assert_eq!(overlay.tokenizer_state_offsets, vec![1, 11, 29]);
        assert_eq!(overlay.repair_terminals, vec![false, true, false, true]);
        assert_eq!(
            overlay.non_parent_only_parser_states,
            vec![true, false, true],
        );
        assert!(overlay.segmented_parser_components.is_empty());
        assert!(overlay.segmented_component_union_root_dispatch.is_empty());
        assert!(overlay.segmented_boundary_parser.is_none());
        assert!(overlay.segmented_boundary_terminal_trie.is_none());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    #[ignore = "pre-release: historical artifact compatibility is not a release requirement"]
    fn previous_composition_metadata_wire_remains_loadable() {
        let constraint = tiny_constraint();
        let previous = PreviousConstraintCompositionMetadata {
            composition_reset_tokens_by_terminal: constraint
                .composition_reset_tokens_by_terminal
                .clone(),
            composition_parser_templates_by_terminal: constraint
                .composition_parser_templates_by_terminal
                .clone(),
            composition_parser_characterizations_by_terminal: constraint
                .composition_parser_characterizations_by_terminal
                .clone(),
            composition_grammar_summary: constraint.composition_grammar_summary.clone(),
        };
        let raw = bincode::serialize(&previous).unwrap();

        for compressed in [false, true] {
            let body = if compressed {
                zstd::bulk::compress(&raw, 1).unwrap()
            } else {
                raw.clone()
            };
            let mut wire = Vec::with_capacity(COMPOSITION_METADATA_HEADER_LEN + body.len());
            wire.extend_from_slice(if compressed {
                &PREVIOUS_COMPOSITION_METADATA_ZSTD_MAGIC
            } else {
                &PREVIOUS_COMPOSITION_METADATA_RAW_MAGIC
            });
            wire.extend_from_slice(&(raw.len() as u64).to_le_bytes());
            wire.extend_from_slice(&body);

            let decoded = decode_composition_metadata(&wire).unwrap();
            assert!(decoded.unbound_grammar_placeholders.is_empty());
            assert_eq!(
                decoded.composition_parser_templates_by_terminal,
                previous.composition_parser_templates_by_terminal,
            );
            assert_eq!(
                decoded.composition_parser_characterizations_by_terminal,
                previous.composition_parser_characterizations_by_terminal,
            );
            assert_eq!(
                decoded.composition_grammar_summary,
                previous.composition_grammar_summary,
            );
        }
    }

    #[test]
    #[ignore = "pre-release: historical artifact compatibility is not a release requirement"]
    fn previous_previous_split_composition_metadata_wire_defaults_trigger_to_none() {
        let constraint = tiny_constraint();
        let link = PreviousPreviousConstraintCompositionLinkMetadata {
            composition_reset_tokens_by_terminal: constraint
                .composition_reset_tokens_by_terminal
                .clone(),
            unbound_grammar_placeholders: constraint.unbound_grammar_placeholders.clone(),
            composition_grammar_summary: constraint.composition_grammar_summary.clone(),
        };
        let cache = ConstraintCompositionCacheMetadata {
            composition_parser_templates_by_terminal: constraint
                .composition_parser_templates_by_terminal
                .clone(),
            composition_parser_characterizations_by_terminal: constraint
                .composition_parser_characterizations_by_terminal
                .clone(),
        };
        let link_raw = bincode::serialize(&link).unwrap();
        let cache_raw = bincode::serialize(&cache).unwrap();
        let mut wire = Vec::new();
        wire.extend_from_slice(&PREVIOUS_PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC);
        wire.extend_from_slice(&0u32.to_le_bytes());
        wire.extend_from_slice(&(link_raw.len() as u64).to_le_bytes());
        wire.extend_from_slice(&(link_raw.len() as u64).to_le_bytes());
        wire.extend_from_slice(&(cache_raw.len() as u64).to_le_bytes());
        wire.extend_from_slice(&(cache_raw.len() as u64).to_le_bytes());
        wire.extend_from_slice(&link_raw);
        wire.extend_from_slice(&cache_raw);

        let decoded = decode_composition_metadata(&wire).unwrap();
        assert!(matches!(decoded.boundary_trigger, BoundaryTriggerWire::None));
        assert_eq!(
            decoded.composition_grammar_summary,
            constraint.composition_grammar_summary,
        );
    }

    #[test]
    fn previous_split_composition_metadata_wire_preserves_trigger_and_defaults_summary() {
        let constraint = tiny_constraint();
        let link = PreviousConstraintCompositionLinkMetadata {
            composition_reset_tokens_by_terminal: constraint
                .composition_reset_tokens_by_terminal
                .clone(),
            unbound_grammar_placeholders: constraint.unbound_grammar_placeholders.clone(),
            composition_grammar_summary: constraint.composition_grammar_summary.clone(),
            boundary_trigger: BoundaryTriggerWire::Tokens(vec![1, 3]),
        };
        let cache = ConstraintCompositionCacheMetadata {
            composition_parser_templates_by_terminal: constraint
                .composition_parser_templates_by_terminal
                .clone(),
            composition_parser_characterizations_by_terminal: constraint
                .composition_parser_characterizations_by_terminal
                .clone(),
        };
        let link_raw = bincode::serialize(&link).unwrap();
        let cache_raw = bincode::serialize(&cache).unwrap();
        let mut wire = Vec::new();
        wire.extend_from_slice(&PREVIOUS_COMPOSITION_METADATA_SPLIT_MAGIC);
        wire.extend_from_slice(&0u32.to_le_bytes());
        wire.extend_from_slice(&(link_raw.len() as u64).to_le_bytes());
        wire.extend_from_slice(&(link_raw.len() as u64).to_le_bytes());
        wire.extend_from_slice(&(cache_raw.len() as u64).to_le_bytes());
        wire.extend_from_slice(&(cache_raw.len() as u64).to_le_bytes());
        wire.extend_from_slice(&link_raw);
        wire.extend_from_slice(&cache_raw);

        let decoded = decode_composition_metadata(&wire).unwrap();
        assert!(matches!(
            decoded.boundary_trigger,
            BoundaryTriggerWire::Tokens(ref ids) if ids == &[1, 3]
        ));
        assert!(matches!(
            decoded.boundary_candidate_summary,
            BoundaryCandidateSummaryWire::Unknown { reason }
                if reason == summary_unavailable_code(
                    &crate::runtime::SummaryUnavailable::LegacyArtifact
                )
        ));
    }

    #[test]
    fn current_constraint_artifact_preserves_boundary_token_trigger() {
        let mut constraint = tiny_constraint();
        constraint.boundary_trigger = crate::runtime::BoundaryTrigger::Tokens(Arc::from(
            vec![1u32, 3u32].into_boxed_slice(),
        ));
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(matches!(
            loaded.boundary_trigger,
            crate::runtime::BoundaryTrigger::None
        ));
        loaded
            .materialize_composition_link_metadata_for_compilation()
            .unwrap();
        assert_eq!(loaded.boundary_trigger.token_summary(), Some(&[1u32, 3u32][..]));
    }

    #[test]
    fn current_constraint_artifact_preserves_boundary_candidate_summary() {
        let constraint = tiny_constraint();
        constraint
            .boundary_candidate_summary
            .set(crate::runtime::BoundaryCandidateSummary::Known {
                fingerprint: crate::runtime::BoundaryCandidateFingerprint {
                    algorithm_version: 7,
                    component_semantics: [1; 32],
                    public_interface: [2; 32],
                    vocabulary: [3; 32],
                },
                tokens: crate::runtime::OriginalTokenSet::Sparse(Arc::from(
                    vec![0u32, 2u32].into_boxed_slice(),
                )),
                precision: crate::runtime::SummaryPrecision::RegularUpperBound,
            })
            .unwrap();
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded.boundary_candidate_summary.get().is_none());
        loaded
            .materialize_composition_link_metadata_for_compilation()
            .unwrap();
        let summary = loaded
            .boundary_candidate_summary
            .get()
            .expect("summary must materialize from CMS5");
        match summary {
            crate::runtime::BoundaryCandidateSummary::Known {
                fingerprint,
                tokens,
                precision,
            } => {
                assert_eq!(fingerprint.algorithm_version, 7);
                assert_eq!(fingerprint.component_semantics, [1; 32]);
                assert_eq!(fingerprint.public_interface, [2; 32]);
                assert_eq!(fingerprint.vocabulary, [3; 32]);
                assert_eq!(tokens.canonical_ids(loaded.token_bytes_iter()), vec![0, 2]);
                assert_eq!(*precision, crate::runtime::SummaryPrecision::RegularUpperBound);
            }
            other => panic!("unexpected materialized summary: {other:?}"),
        }
    }

    #[test]
    fn current_constraint_artifact_preserves_empty_all_and_unknown_boundary_summaries() {
        let mut dense = crate::ds::bitset::BitSet::new(3);
        dense.set(0);
        dense.set(2);
        for summary in [
            crate::runtime::BoundaryCandidateSummary::Known {
                fingerprint: sample_boundary_fingerprint(),
                tokens: crate::runtime::OriginalTokenSet::Empty,
                precision: crate::runtime::SummaryPrecision::RegularUpperBound,
            },
            crate::runtime::BoundaryCandidateSummary::Known {
                fingerprint: sample_boundary_fingerprint(),
                tokens: crate::runtime::OriginalTokenSet::Dense(Arc::new(dense)),
                precision: crate::runtime::SummaryPrecision::RegularUpperBound,
            },
            crate::runtime::BoundaryCandidateSummary::Known {
                fingerprint: sample_boundary_fingerprint(),
                tokens: crate::runtime::OriginalTokenSet::AllByteTokensAtLeastTwo,
                precision: crate::runtime::SummaryPrecision::BudgetWidenedUpperBound,
            },
            crate::runtime::BoundaryCandidateSummary::Unknown {
                reason: crate::runtime::SummaryUnavailable::Deferred,
            },
        ] {
            let constraint = tiny_constraint();
            constraint.boundary_candidate_summary.set(summary.clone()).unwrap();
            let mut loaded = Constraint::load(&constraint.save()).unwrap();
            loaded
                .materialize_composition_link_metadata_for_compilation()
                .unwrap();
            let restored = loaded
                .boundary_candidate_summary
                .get()
                .expect("CMS5 summary must materialize");
            match (&summary, restored) {
                (
                    crate::runtime::BoundaryCandidateSummary::Known { tokens: expected, .. },
                    crate::runtime::BoundaryCandidateSummary::Known { tokens: actual, .. },
                ) => assert_eq!(
                    expected.canonical_ids(constraint.token_bytes_iter()),
                    actual.canonical_ids(loaded.token_bytes_iter()),
                ),
                (
                    crate::runtime::BoundaryCandidateSummary::Unknown { reason: expected },
                    crate::runtime::BoundaryCandidateSummary::Unknown { reason: actual },
                ) => assert_eq!(expected, actual),
                pair => panic!("summary variant changed across CMS5 roundtrip: {pair:?}"),
            }
        }
    }

    #[test]
    fn current_boundary_summary_preserves_duplicate_byte_token_ids() {
        let vocab = Vocab::new(vec![
            (10, b"ab".to_vec()),
            (11, b"ab".to_vec()),
            (12, b"a".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start doc;
                nt doc ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        constraint
            .boundary_candidate_summary
            .set(crate::runtime::BoundaryCandidateSummary::Known {
                fingerprint: sample_boundary_fingerprint(),
                tokens: crate::runtime::OriginalTokenSet::Sparse(Arc::from(
                    vec![10u32, 11u32].into_boxed_slice(),
                )),
                precision: crate::runtime::SummaryPrecision::RegularUpperBound,
            })
            .unwrap();
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        loaded
            .materialize_composition_link_metadata_for_compilation()
            .unwrap();
        let crate::runtime::BoundaryCandidateSummary::Known { tokens, .. } = loaded
            .boundary_candidate_summary
            .get()
            .expect("duplicate-ID summary must materialize")
        else {
            panic!("duplicate-ID summary changed variant");
        };
        assert_eq!(tokens.canonical_ids(loaded.token_bytes_iter()), vec![10, 11]);
    }

    #[test]
    fn v29_artifact_without_composition_metadata_loads_legacy_unknown_summary() {
        let mut constraint = tiny_constraint();
        constraint.composition_reset_tokens_by_terminal.clear();
        constraint.unbound_grammar_placeholders.clear();
        constraint.composition_parser_templates_by_terminal.clear();
        constraint
            .composition_parser_characterizations_by_terminal
            .clear();
        constraint.composition_grammar_summary = None;
        constraint.boundary_trigger = crate::runtime::BoundaryTrigger::None;
        let mut saved = constraint.save();
        let (_, _, _, _, _, _, _, _, _, _, composition) =
            v30_sections(&saved[CONSTRAINT_HEADER_LEN..]).unwrap();
        assert!(composition.is_empty(), "legacy-empty fixture must have no CMS section");
        saved[8..10].copy_from_slice(&PREVIOUS_BOUNDARY_SUMMARYLESS_CONSTRAINT_VERSION.to_le_bytes());
        saved[CONSTRAINT_HEADER_LEN..CONSTRAINT_HEADER_LEN + 4]
            .copy_from_slice(&V29_SECTION_MAGIC);

        let loaded = Constraint::load(&saved).unwrap();
        assert!(matches!(
            loaded.boundary_candidate_summary.get(),
            Some(crate::runtime::BoundaryCandidateSummary::Unknown {
                reason: crate::runtime::SummaryUnavailable::LegacyArtifact
            })
        ));
        assert_eq!(loaded.save(), saved, "unchanged v29 artifact must resave byte-for-byte");
    }

    #[test]
    fn current_constraint_artifact_preserves_composition_reset_tokens() {
        let mut constraint = tiny_constraint();
        constraint.ensure_composition_reset_tokens_by_terminal();
        assert_eq!(
            constraint.composition_reset_tokens_by_terminal.len(),
            constraint.table.num_terminals as usize,
        );
        assert!(constraint
            .composition_reset_tokens_by_terminal
            .iter()
            .any(|row| !row.is_empty()));
        let expected = constraint.composition_reset_tokens_by_terminal.clone();
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded.composition_reset_tokens_by_terminal.is_empty());
        assert!(loaded.deferred_composition_metadata_blob.is_some());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
        loaded
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(loaded.composition_reset_tokens_by_terminal, expected);
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn retained_template_read_is_exact_and_does_not_materialize_constraint() {
        let original = tiny_constraint();
        let expected = original.composition_parser_templates_by_terminal.clone();
        assert!(!expected.is_empty());
        assert!(matches!(original.retained_parser_templates_for_compilation().unwrap(), Cow::Borrowed(_)));
        let bytes = original.save();
        let loaded = Constraint::load(&bytes).unwrap();
        assert!(loaded.composition_parser_templates_by_terminal.is_empty());
        let templates = loaded.retained_parser_templates_for_compilation().unwrap();
        assert!(matches!(templates, Cow::Owned(_)));
        assert_eq!(templates.as_ref(), expected.as_slice());
        assert!(loaded.composition_parser_templates_by_terminal.is_empty());
        assert!(loaded.composition_parser_characterizations_by_terminal.is_empty());
        assert!(loaded.deferred_composition_metadata_blob.is_some());
        assert_eq!(loaded.save(), bytes);
        assert_eq!(loaded.start().mask(), original.start().mask());
    }

    #[test]
    fn current_constraint_artifact_preserves_composition_parser_templates() {
        let constraint = tiny_constraint();
        assert_eq!(
            constraint.composition_parser_templates_by_terminal.len(),
            constraint.table.num_terminals as usize,
        );
        assert!(constraint
            .composition_parser_templates_by_terminal
            .iter()
            .any(Option::is_some));
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded.composition_parser_templates_by_terminal.is_empty());
        assert!(loaded.deferred_composition_metadata_blob.is_some());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
        loaded
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(
            loaded.composition_parser_templates_by_terminal,
            constraint.composition_parser_templates_by_terminal,
        );
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn current_constraint_artifact_preserves_composition_parser_characterizations() {
        let constraint = tiny_constraint();
        assert_eq!(
            constraint
                .composition_parser_characterizations_by_terminal
                .len(),
            constraint.table.num_terminals as usize,
        );
        assert!(constraint
            .composition_parser_characterizations_by_terminal
            .iter()
            .any(Option::is_some));
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded
            .composition_parser_characterizations_by_terminal
            .is_empty());
        assert!(loaded.deferred_composition_metadata_blob.is_some());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
        loaded
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(
            loaded.composition_parser_characterizations_by_terminal,
            constraint.composition_parser_characterizations_by_terminal,
        );
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn current_constraint_artifact_preserves_composition_grammar_summary() {
        let constraint = tiny_constraint();
        let expected = constraint
            .composition_grammar_summary
            .clone()
            .expect("fresh static constraint should retain composition grammar summary");
        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded.composition_grammar_summary.is_none());
        assert!(loaded.deferred_composition_metadata_blob.is_some());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
        loaded
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(loaded.composition_grammar_summary, Some(expected));
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn split_composition_metadata_allows_link_only_materialization() {
        let mut constraint = tiny_constraint();
        constraint.ensure_composition_reset_tokens_by_terminal();
        let expected_resets = constraint.composition_reset_tokens_by_terminal.clone();
        let expected_summary = constraint.composition_grammar_summary.clone();
        let expected_templates = constraint.composition_parser_templates_by_terminal.clone();
        assert!(expected_summary.is_some());
        assert!(expected_templates.iter().any(Option::is_some));

        let mut loaded = Constraint::load(&constraint.save()).unwrap();
        let deferred = loaded
            .deferred_composition_metadata_blob
            .as_ref()
            .expect("current artifact should defer composition metadata");
        assert!(deferred.as_slice().starts_with(&COMPOSITION_METADATA_SPLIT_MAGIC));
        assert!(!loaded.composition_link_metadata_materialized);
        assert!(loaded.composition_parser_templates_by_terminal.is_empty());

        loaded
            .materialize_composition_link_metadata_for_compilation()
            .unwrap();
        assert_eq!(loaded.composition_reset_tokens_by_terminal, expected_resets);
        assert_eq!(loaded.composition_grammar_summary, expected_summary);
        assert!(loaded.composition_link_metadata_materialized);
        assert!(
            loaded.composition_parser_templates_by_terminal.is_empty(),
            "link-only materialization must not instantiate static parser caches",
        );
        assert!(
            loaded.deferred_composition_metadata_blob.is_some(),
            "full compiler metadata must remain available for a later static composition",
        );

        loaded
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(
            loaded.composition_parser_templates_by_terminal,
            expected_templates,
        );
        assert!(loaded.deferred_composition_metadata_blob.is_none());
    }

    #[test]
    fn current_constraint_artifact_preserves_global_ignore_descriptor() {
        let constraint = ignored_constraint();
        let loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(constraint.ignore_expr.is_some());
        assert_eq!(loaded.ignore_expr, constraint.ignore_expr);
        assert!(
            loaded.tokenizer.terminal_exprs().is_none(),
            "current load should defer terminal expression reconstruction",
        );
        assert_eq!(
            loaded.retained_terminal_exprs(),
            constraint.tokenizer.terminal_exprs(),
            "current artifacts should lazily retain terminal proof expressions",
        );
        if constraint.can_defer_internal_tsid_inverse() {
            assert!(
                loaded.internal_tsid_to_states.is_empty(),
                "current scalar-state artifacts should defer the redundant TSID inverse",
            );
            assert_eq!(
                loaded.internal_tsid_groups(),
                constraint.internal_tsid_to_states.as_slice(),
                "deferred TSID inverse must reconstruct exactly",
            );
        }
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    fn constraint_envelope_rejects_invalid_compressed_payloads() {
        let constraint = tiny_constraint();
        let raw = bincode::serialize(&SerializedConstraint(&constraint)).unwrap();
        let compressed = zstd::bulk::compress(&raw, CONSTRAINT_COMPRESSION_LEVEL).unwrap();

        let mut wrong_raw_len = Vec::with_capacity(8 + compressed.len());
        wrong_raw_len.extend_from_slice(&((raw.len() + 1) as u64).to_le_bytes());
        wrong_raw_len.extend_from_slice(&compressed);
        assert!(Constraint::load(&envelope(
            PREVIOUS_DOMAIN_LABELS_CONSTRAINT_VERSION,
            &wrong_raw_len,
        ))
            .unwrap_err()
            .to_string()
            .contains("uncompressed"));

        assert!(Constraint::load(&envelope(
            PREVIOUS_DOMAIN_LABELS_CONSTRAINT_VERSION,
            &[0; 8],
        ))
        .is_err());
    }

    #[test]
    fn constraint_envelope_rejects_version_and_length_mismatches() {
        let constraint = tiny_constraint();
        let mut wrong_version = constraint.save();
        wrong_version[8..10].copy_from_slice(&(CONSTRAINT_VERSION + 1).to_le_bytes());
        assert!(Constraint::load(&wrong_version)
            .unwrap_err()
            .to_string()
            .contains("version"));

        let mut wrong_length = constraint.save();
        wrong_length[10..18].copy_from_slice(&0u64.to_le_bytes());
        assert!(Constraint::load(&wrong_length)
            .unwrap_err()
            .to_string()
            .contains("payload length"));
    }

    #[test]
    fn constraint_roundtrip_preserves_commit_template_dfas() {
        let mut constraint = tiny_constraint();
        let mut pop = UnweightedDfa::new();
        let accepted = pop.add_state();
        pop.add_transition(pop.start_state, 7, accepted);
        pop.set_accepting(accepted, true);
        let template = CommitTemplateDfas {
            pop,
            read: UnweightedDfa::default(),
            push: UnweightedDfa::default(),
            pop_to_read: vec![None; 2],
            pop_to_push: vec![None; 2],
            read_to_push: Vec::new(),
        };
        constraint.template_dfas_by_terminal = vec![None, Some(Arc::new(template.clone()))];

        let loaded = Constraint::load(&constraint.save()).expect("template artifact should load");
        let loaded_template = loaded.template_dfas_by_terminal[1]
            .as_deref()
            .expect("serialized template should survive load");
        let loaded_fast_template = loaded.fast_template_dfas_by_terminal[1]
            .as_deref()
            .expect("runtime template transition cache should be rebuilt after load");
        assert_eq!(loaded_template.pop, template.pop);
        assert_eq!(loaded_template.read, template.read);
        assert_eq!(loaded_template.push, template.push);
        assert_eq!(loaded_template.pop_to_read, template.pop_to_read);
        assert_eq!(loaded_template.pop_to_push, template.pop_to_push);
        assert_eq!(loaded_template.read_to_push, template.read_to_push);
        assert_eq!(loaded_fast_template.pop.start_state, template.pop.start_state);
        assert_eq!(
            loaded_fast_template.pop.states[accepted as usize].is_accepting,
            template.pop.states[accepted as usize].is_accepting
        );
        assert_eq!(
            loaded_fast_template.pop.states[template.pop.start_state as usize]
                .transitions
                .get(7),
            Some(accepted)
        );
    }

}
