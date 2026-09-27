//! Current compiled-artifact persistence.
//!
//! Envelope framing, runtime codecs, and compiler-only metadata have separate
//! owners. Loading preserves immutable byte backing and reconstructs mutable
//! caches. Unsupported historical envelopes fail before decoding their bodies.


use crate::runtime::Constraint;
use crate::runtime::artifact::{
    BackedInternalTokenBufMasks, ConstraintSerde, DenseBufMaskRows, InternalTokenBufMasks,
    PackedInternalTokenBufMask,
};
use crate::automata::regex::Expr;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;

use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;
use std::sync::Arc;

pub(super) fn serialize_constraint<S>(constraint: &&Constraint, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    ConstraintSerde::serialize(*constraint, serializer)
}

pub(super) fn deserialize_constraint<'de, D>(deserializer: D) -> Result<Constraint, D::Error>
where
    D: serde::Deserializer<'de>,
{
    ConstraintSerde::deserialize(deserializer)
}

pub(super) struct DeserializedConstraint(Constraint);

impl<'de> serde::Deserialize<'de> for DeserializedConstraint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        ConstraintSerde::deserialize(deserializer).map(Self)
    }
}

pub(super) struct SerializedConstraint<'a>(&'a Constraint);

impl serde::Serialize for SerializedConstraint<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        ConstraintSerde::serialize(self.0, serializer)
    }
}

mod envelope;
use envelope::{
    CONSTRAINT_HEADER_LEN, CONSTRAINT_MAGIC, CONSTRAINT_VERSION,
    ConstraintArtifactCurrentRuntimeRef, DecodedConstraintRuntime, ROOT_POLICY_MAGIC,
    SECTION_HEADER_LEN, SECTION_MAGIC, constraint_sections, decode_current_runtime_wire,
    encode_current_runtime_wire,
};

mod residual;
pub(crate) use residual::{
    DecodedStaticVirtualResidualMask, StaticVirtualResidualMaskArtifact,
    decode_static_virtual_residual_mask_wire, encode_static_virtual_residual_mask_wire,
    encode_static_virtual_residual_mask_wire_with_fallback,
};
use residual::{StaticVirtualResidualMaskArtifactRef};
#[cfg(test)]
use residual::{STATIC_RESIDUAL_MASK_MAGIC};

mod core;
use core::{
    CURRENT_CORE_FLAG_OMIT_TSID_INVERSE, CURRENT_CORE_MAGIC, ConstraintArtifactCurrentCoreBaseRef,
    DecodedConstraintCore, decode_current_core,
};

mod save;

mod composition;
use composition::{encode_composition_metadata_for_save, validate_composition_metadata_wire};
#[cfg(test)]
use composition::{COMPOSITION_METADATA_SPLIT_MAGIC};

mod token_masks;
use token_masks::{
    DecodedInternalTokenBufMasks, DecodedOriginalTokenMap, TokenMaskCacheArtifact,
    decode_internal_token_buf_masks, decode_token_mask_cache, decode_token_mask_cache_backed,
    encode_internal_token_buf_masks, encode_token_mask_cache, install_token_mask_cache,
};
#[cfg(test)]
use token_masks::{SeedTerminalDenseCompact};

mod segmented;
use segmented::{
    SegmentedRuntimeArtifact, SegmentedRuntimeArtifactRef, restore_segmented_runtime,
    segmented_runtime_artifact_ref,
};

mod weights;
pub(crate) use weights::{compact_large_non_dwa_weight_runtime};
use weights::{
    attach_packed_non_dwa_weights, constraint_serialized_weight_pool,
    packed_constraint_serialized_weight_ids,
};
#[cfg(test)]
use weights::{compact_non_dwa_weight_runtime_if_at_least, constraint_serialized_weight_pool_with_ids};

mod load;

#[cfg(test)]
mod tests;
