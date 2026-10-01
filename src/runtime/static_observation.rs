//! Finite boundary observations are independent of a component's local mask
//! quotient. Their horizon covers the full linked vocabulary, including tokens
//! which leave a component before their final byte.
use std::sync::{Arc, OnceLock};
use crate::automata::lexer::{Lexer, tokenizer::{Tokenizer, VirtualResidualMaskProjection},
    runtime_unit_repeat::VirtualZeroMinUnitRepeatMaskProjection,
    runtime_repeat_product::VirtualBinaryRepeatIntersectionMaskProjection};
use super::Constraint;

const MAX_OBSERVATION_STATES: usize = 1_048_576;
const MAX_OBSERVATION_WORK: usize = 32_000_000;
const OBSERVATION_VERSION: u32 = 1;

#[derive(Debug, Clone)]
enum Projection {
    Unit(VirtualZeroMinUnitRepeatMaskProjection),
    Product(Vec<VirtualBinaryRepeatIntersectionMaskProjection>),
    Residual(Vec<VirtualResidualMaskProjection>),
}

#[derive(Debug, Clone)]
struct LeafObservation {
    tokenizer: Tokenizer,
    projection: Projection,
    physical_states: u32,
}

impl LeafObservation {
    fn build(source: &Tokenizer, horizon: usize) -> Result<Self, String> {
        if horizon == 0 || horizon > MAX_OBSERVATION_STATES {
            return Err("static observation horizon exceeds its representation budget".into());
        }
        let residual = source.virtual_residual_mask_projection_dense_state_work(horizon)
            .is_some_and(|work| work <= MAX_OBSERVATION_STATES)
            .then(|| source.virtual_residuals_mask_tokenizer(horizon)).flatten();
        let (tokenizer, projection) = if let Some((view, maps)) = residual {
            (view, Projection::Residual(maps))
        } else if let Some((view, maps)) = source.virtual_binary_repeat_intersections_mask_tokenizer(horizon) {
            (view, Projection::Product(maps))
        } else if let Some((view, map)) = source.virtual_unit_repeat_mask_tokenizer(horizon) {
            (view, Projection::Unit(map))
        } else {
            return Err("virtual lexer has no supported finite boundary observation projection at the full vocabulary horizon".into());
        };
        if tokenizer.has_any_virtual_runtime() || tokenizer.num_states() as usize > MAX_OBSERVATION_STATES {
            return Err("static observation is not finite within its representation budget".into());
        }
        Ok(Self { tokenizer, projection, physical_states: source.num_states() })
    }

    fn project(&self, state: u32) -> Option<u32> {
        let projected = match &self.projection {
            Projection::Unit(map) => map.project(state),
            Projection::Product(maps) => maps.iter().find_map(|map| map.project(state)),
            Projection::Residual(maps) => maps.iter().find_map(|map| map.project(state)),
        };
        projected.or_else(|| (state < self.physical_states).then_some(state))
            .filter(|&state| state < self.tokenizer.num_states())
    }

    /// Commit both the finite graph and the exact-state transport to the wire
    /// descriptor. Rebuilding after load is accepted only if both agree; a
    /// change of projection numbering cannot reinterpret saved B/PM weights.
    fn fingerprint(&self) -> Result<[u8; 32], String> {
        let mut hash = blake3::Hasher::new();
        hash.update(b"glrmask-static-leaf-observation-v1");
        let mut work = MAX_OBSERVATION_WORK;
        fn feed(hash: &mut blake3::Hasher, work: &mut usize, bytes: &[u8]) -> Result<(), String> {
            *work = work.checked_sub(bytes.len()).ok_or("static observation fingerprint exceeds its work budget")?;
            hash.update(&(bytes.len() as u64).to_le_bytes()); hash.update(bytes); Ok(())
        }
        let descriptors: Vec<Vec<u8>> = match &self.projection {
            Projection::Unit(map) => { hash.update(&[0]); vec![map.observation_descriptor()] },
            Projection::Product(maps) => { hash.update(&[1]); maps.iter().map(|m| m.observation_descriptor()).collect() },
            Projection::Residual(maps) => { hash.update(&[2]); maps.iter().map(|m| m.observation_descriptor()).collect() },
        };
        hash.update(&(descriptors.len() as u64).to_le_bytes());
        for descriptor in descriptors { feed(&mut hash, &mut work, &descriptor)?; }
        hash.update(&self.physical_states.to_le_bytes());
        hash.update(&self.tokenizer.num_states().to_le_bytes());
        hash.update(&self.tokenizer.num_terminals().to_le_bytes());
        hash.update(&self.tokenizer.start_state().to_le_bytes());
        for state in 0..self.tokenizer.num_states() {
            let mut epsilon = self.tokenizer.singleton_epsilon_closure(state).to_vec();
            epsilon.sort_unstable(); epsilon.dedup();
            let mut matched = self.tokenizer.matched_terminals_iter(state).collect::<Vec<_>>();
            matched.sort_unstable(); matched.dedup();
            let mut future = self.tokenizer.possible_future_terminals_iter(state).collect::<Vec<_>>();
            future.sort_unstable(); future.dedup();
            let mut edges = self.tokenizer.transitions_from(state).collect::<Vec<_>>();
            edges.sort_unstable(); edges.dedup();
            for values in [&epsilon, &matched, &future] {
                feed(&mut hash, &mut work, &bincode::serialize(values).map_err(|e| e.to_string())?)?;
            }
            feed(&mut hash, &mut work, &bincode::serialize(&edges).map_err(|e| e.to_string())?)?;
        }
        Ok(*hash.finalize().as_bytes())
    }
}

/// Previous unpublished experimental tag had no full-vocabulary horizon or
/// transport identity. It is decoded only to return a precise rebuild error.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct LegacyRecursiveStaticObservation { pub(crate) leaf_offsets: Vec<u32> }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct RecursiveStaticObservation {
    /// Finite leaves follow synthetic root 0; final entry is exclusive end.
    pub(crate) leaf_offsets: Vec<u32>,
    version: u32,
    horizon: u32,
    fingerprints: Vec<Option<[u8; 32]>>,
    #[serde(skip)]
    runtime: OnceLock<Vec<Option<Arc<LeafObservation>>>>,
}

impl RecursiveStaticObservation {
    pub(crate) fn prepare(leaves: &[&Constraint], horizon: usize) -> Result<Self, String> {
        let horizon = horizon.max(1);
        let mut runtime = Vec::with_capacity(leaves.len());
        let mut leaf_offsets = vec![1u32];
        let mut fingerprints = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            let view = leaf.tokenizer.has_any_virtual_runtime()
                .then(|| LeafObservation::build(&leaf.tokenizer, horizon)).transpose()?.map(Arc::new);
            let count = view.as_ref().map_or(leaf.tokenizer.num_states(), |v| v.tokenizer.num_states());
            leaf_offsets.push(leaf_offsets.last().unwrap().checked_add(count).ok_or("finite observation coordinate overflow")?);
            fingerprints.push(view.as_ref().map(|v| v.fingerprint()).transpose()?);
            runtime.push(view);
        }
        Ok(Self { leaf_offsets, version: OBSERVATION_VERSION,
            horizon: u32::try_from(horizon).map_err(|_| "finite observation horizon overflow")?,
            fingerprints, runtime: OnceLock::from(runtime) })
    }

    pub(crate) fn leaf_view<'a>(&'a self, index: usize, leaf: &'a Constraint) -> Option<&'a Tokenizer> {
        Some(self.runtime.get()?.get(index)?.as_ref().map_or(leaf.tokenizer.as_ref(), |v| &v.tokenizer))
    }

    pub(crate) fn leaf_key(&self, leaf_index: usize, leaf: &Constraint, state: u32) -> Option<u32> {
        let local = if let Some(view) = self.runtime.get()?.get(leaf_index)?.as_ref() {
            view.project(state)?
        } else { (state < leaf.tokenizer.num_states()).then_some(state)? };
        let start = *self.leaf_offsets.get(leaf_index)?;
        let end = *self.leaf_offsets.get(leaf_index + 1)?;
        start.checked_add(local).filter(|&key| key < end)
    }

    pub(crate) fn validate(&self, constraint: &Constraint) -> Result<(), String> {
        let layout = constraint.recursive_parser_layout()?
            .ok_or("projected static observations require a recursive parser")?;
        if self.version != OBSERVATION_VERSION || self.horizon == 0
            || self.horizon as usize > MAX_OBSERVATION_STATES
            || (self.horizon as usize) < constraint.max_token_byte_len().max(1) {
            return Err("invalid or insufficient static observation vocabulary horizon".into());
        }
        if self.leaf_offsets.len() != layout.leaves.len() + 1
            || self.fingerprints.len() != layout.leaves.len() || self.leaf_offsets.first() != Some(&1) {
            return Err("invalid recursive static observation leaf inventory".into());
        }
        let leaves = layout.leaves.iter().map(|leaf|
            constraint.constraint_at_recursive_component_path(&leaf.component_path)
                .ok_or("invalid projected observation component path"))
            .collect::<Result<Vec<_>, _>>()?;
        if self.runtime.get().is_none() {
            let prepared = Self::prepare(&leaves, self.horizon as usize)?;
            if prepared.leaf_offsets != self.leaf_offsets || prepared.fingerprints != self.fingerprints {
                return Err("saved static observation transport differs from its exact finite reconstruction".into());
            }
            let _ = self.runtime.set(prepared.runtime.into_inner().unwrap());
        }
        let mut next = 1u32;
        for (index, body) in leaves.iter().enumerate() {
            let view = self.leaf_view(index, body).ok_or("missing finite observation runtime")?;
            if view.has_any_virtual_runtime() || self.leaf_offsets[index] != next {
                return Err("recursive static observation is not a finite contiguous coordinate".into());
            }
            next = next.checked_add(view.num_states()).ok_or("recursive static observation overflow")?;
            if self.leaf_offsets[index + 1] != next {
                return Err("recursive static observation size disagrees with its component".into());
            }
        }
        if constraint.internal_tsid_count() != next as usize {
            return Err("projected static observation differs from coordinator TSID domain".into());
        }
        let inverse = constraint.internal_tsid_groups();
        if inverse.len() != next as usize || inverse.iter().enumerate().any(|(id, values)| values.as_slice() != [id as u32]) {
            return Err("projected static observation requires the exact identity exclusion coordinate".into());
        }
        let rows = constraint.static_dynamic_overlay.as_ref()
            .and_then(|overlay| overlay.recursive_tokenizer_internal_tsids.get())
            .ok_or("projected static observation has no physical runtime image")?;
        if rows.len() != layout.total_tokenizer_states as usize {
            return Err("projected static observation physical image has the wrong length".into());
        }
        for (index, body) in leaves.iter().enumerate() {
            for local in 0..body.tokenizer.num_states() {
                let key = self.leaf_key(index, body, local).ok_or("physical state leaves its finite observation view")?;
                let scoped = layout.leaf_tokenizer_state_offsets[index] + local;
                if rows[scoped as usize].as_slice() != [key] {
                    return Err("projected static observation physical image is inconsistent".into());
                }
            }
        }
        Ok(())
    }
}

impl Constraint {
    /// Key for exact remembered-lexeme exclusions. Runtime GSS annotations
    /// continue to hold exact states; only the static lookup uses this view.
    pub(crate) fn static_exclusion_state(&self, state: u32) -> Option<u32> {
        let Some(observation) = self.static_dynamic_overlay.as_ref()
            .and_then(|overlay| overlay.recursive_static_observation.as_ref()) else { return Some(state); };
        let (index, local) = self.recursive_tokenizer_leaf_state(state)?;
        let layout = self.recursive_parser_layout().ok().flatten()?;
        let body = self.constraint_at_recursive_component_path(&layout.leaves.get(index)?.component_path)?;
        observation.leaf_key(index, body, local)
    }
}
