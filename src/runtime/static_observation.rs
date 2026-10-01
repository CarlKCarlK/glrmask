//! Construction-only finite lexer views have a different coordinate from
//! lazily allocated exact runtime states. Keep that mapping explicit; a local
//! component's A-mask quotient is not evidence for B or exclusion equivalence.
use crate::automata::lexer::Lexer;
use super::Constraint;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct RecursiveStaticObservation {
    /// Finite leaf views in order, preceded by the compiler union's synthetic
    /// root. The final entry is the exclusive end of the observation domain.
    pub(crate) leaf_offsets: Vec<u32>,
}

impl RecursiveStaticObservation {
    pub(crate) fn leaf_key(&self, leaf_index: usize, leaf: &Constraint, state: u32) -> Option<u32> {
        let local = if leaf.tokenizer.has_any_virtual_runtime() {
            leaf.dynamic_mask_vocab.mask_projection_tokenizer()?;
            leaf.dynamic_mask_vocab.mask_projection_state(state)
        } else { state };
        let start = *self.leaf_offsets.get(leaf_index)?;
        let end = *self.leaf_offsets.get(leaf_index + 1)?;
        start.checked_add(local).filter(|&key| key < end)
    }

    pub(crate) fn validate(&self, constraint: &Constraint) -> Result<(), String> {
        let layout = constraint.recursive_parser_layout()?
            .ok_or("projected static observations require a recursive parser")?;
        if self.leaf_offsets.len() != layout.leaves.len() + 1 || self.leaf_offsets.first() != Some(&1) {
            return Err("invalid recursive static observation leaf inventory".into());
        }
        let mut next = 1u32;
        for (index, leaf) in layout.leaves.iter().enumerate() {
            let body = constraint.constraint_at_recursive_component_path(&leaf.component_path)
                .ok_or("invalid projected observation component path")?;
            let view = if body.tokenizer.has_any_virtual_runtime() {
                body.dynamic_mask_vocab.mask_projection_tokenizer()
                    .ok_or("virtual component has no persisted finite observation projection")?
            } else { body.tokenizer.as_ref() };
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
        // The projection is deliberately unquotiented. B and PM each retain
        // all their distinctions; no local A equivalence is silently reused.
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
        for (index, leaf) in layout.leaves.iter().enumerate() {
            let body = constraint.constraint_at_recursive_component_path(&leaf.component_path).unwrap();
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
