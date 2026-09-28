//! Final model-token admission policy, independent of mask execution strategy.

use crate::runtime::state::ConstraintState;

impl ConstraintState<'_> {
    /// Empty byte payloads cannot advance a byte-language path. Their model IDs
    /// may still name exact terminals; only those independently viable paths
    /// can restore an empty ID. Generation-root end policy runs afterwards.
    pub(crate) fn restrict_empty_byte_tokens(&self, mask: &mut [u32]) {
        if let Some(packed) = &self.constraint.packed_token_bytes {
            for &id in packed.empty_token_ids() {
                self.set_empty_token_admission(mask, id);
            }
        } else {
            // Construction-only fallback. Finalization must provide shared
            // indexed vocabulary data before production masking.
            for (&id, bytes) in self.constraint.token_bytes.iter() {
                if bytes.is_empty() { self.set_empty_token_admission(mask, id); }
            }
        }
    }

    fn set_empty_token_admission(&self, mask: &mut [u32], id: u32) {
        let Some(word) = mask.get_mut(id as usize / 32) else { return; };
        let bit = 1u32 << (id % 32);
        *word &= !bit;
        let exact = self.constraint.special_token_terminals.iter().any(|special| {
            special.token_id == id
                && !self.constraint.is_late_grammar_placeholder_terminal(special.terminal_id)
        });
        if exact && super::super::commit::advance_special_token_paths(
            self.constraint, &self.state, id,
        ).is_some_and(|gss| !gss.is_empty()) {
            *word |= bit;
        }
    }
}
