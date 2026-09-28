//! Vocabulary identity, token bytes, and runtime token-coordinate relations.

use crate::automata::lexer::Lexer;
use crate::grammar::flat::TerminalID;
use crate::runtime::artifact::Constraint;
use smallvec::SmallVec;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::OnceLock;

impl Constraint {


    /// Whether the stored internal-TSID inverse is redundant with the scalar
    /// state -> TSID map. Runtime-product tokenizers can have several TSID
    /// lanes per physical state and therefore must retain their explicit
    /// inverse/relation metadata.
    pub(crate) fn can_defer_internal_tsid_inverse(&self) -> bool {
        if self.runtime_source_state_offset.is_some()
            || self.state_to_internal_tsid.len() != self.tokenizer.num_states() as usize
            || self.state_to_internal_tsid.iter().any(|&tsid| tsid == u32::MAX)
        {
            return false;
        }
        let Some(count) = self
            .state_to_internal_tsid
            .iter()
            .copied()
            .max()
            .map(|max| max as usize + 1)
        else {
            return self.internal_tsid_to_states.is_empty();
        };
        // A previously loaded omitted inverse is already certified by the
        // current-format writer. The complete scalar map is sufficient to
        // reconstruct it exactly.
        if self.internal_tsid_to_states.is_empty() {
            return true;
        }
        if self.internal_tsid_to_states.len() != count {
            return false;
        }
        let mut entries = 0usize;
        for (tsid, states) in self.internal_tsid_to_states.iter().enumerate() {
            entries = entries.saturating_add(states.len());
            for &state in states {
                if self
                    .state_to_internal_tsid
                    .get(state as usize)
                    .copied()
                    != Some(tsid as u32)
                {
                    return false;
                }
            }
        }
        entries == self.state_to_internal_tsid.len()
    }

    pub(crate) fn internal_tsid_count(&self) -> usize {
        if !self.internal_tsid_to_states.is_empty() {
            return self.internal_tsid_to_states.len();
        }
        if let Some(groups) = self.deferred_internal_tsid_to_states.get() {
            return groups.len();
        }
        self.state_to_internal_tsid
            .iter()
            .copied()
            .filter(|&tsid| tsid != u32::MAX)
            .max()
            .map_or(0, |max| max as usize + 1)
    }

    pub(crate) fn internal_tsid_groups(&self) -> &[Vec<u32>] {
        if !self.internal_tsid_to_states.is_empty() {
            return &self.internal_tsid_to_states;
        }
        let groups = self.deferred_internal_tsid_to_states.get_or_init(|| {
            let mut groups = vec![Vec::new(); self.internal_tsid_count()];
            for (state, &tsid) in self.state_to_internal_tsid.iter().enumerate() {
                if tsid != u32::MAX
                    && let Some(states) = groups.get_mut(tsid as usize)
                {
                    states.push(state as u32);
                }
            }
            groups
        });
        groups.as_slice()
    }

    pub(crate) fn token_bytes_match_vocab(&self, vocab: &crate::Vocab) -> bool {
        let vocab_entries = vocab.entries_arc();
        if Arc::ptr_eq(&self.token_bytes, &vocab_entries) {
            return true;
        }
        if let Some(packed) = &self.packed_token_bytes {
            // Current artifacts and Vocab's prepared packed-token artifact use
            // the same canonical indexed token-byte encoding. Comparing those byte strings is
            // an exact vocabulary equality proof and lets a 128k-token bind use
            // one contiguous memcmp instead of 128k map/offset lookups. The
            // packed Vocab artifact is owned by the caller's Vocab-derived
            // cache, not a process-global load cache, so this remains honest
            // model-vocabulary preparation and does not make later constraints
            // cheaper merely because an earlier constraint was loaded.
            if let Some(vocab_packed) =
                crate::compiler::compile::prepared_vocab_packed_token_bytes(vocab)
                && packed.wire() == vocab_packed.wire()
            {
                return true;
            }
            // Legacy/noncanonical packed encodings can still describe the same
            // vocabulary. Preserve the exact structural fallback rather than
            // treating unequal wire encodings as unequal vocabularies.
            return packed.len() == vocab.entries_map().len()
                && packed.iter().eq(
                    vocab
                        .entries_map()
                        .iter()
                        .map(|(&token_id, bytes)| (token_id, bytes.as_slice())),
                );
        }
        self.token_bytes.as_ref() == vocab.entries_map()
    }

    #[inline]
    pub(crate) fn token_bytes_for_id(&self, token_id: u32) -> Option<&[u8]> {
        self.packed_token_bytes
            .as_ref()
            .and_then(|packed| packed.get(token_id))
            .or_else(|| self.token_bytes.get(&token_id).map(Vec::as_slice))
    }

    #[inline]
    pub(crate) fn token_bytes_count(&self) -> usize {
        self.packed_token_bytes
            .as_ref()
            .map_or_else(|| self.token_bytes.len(), |packed| packed.len())
    }

    #[inline]
    pub(crate) fn max_token_byte_len(&self) -> usize {
        self.packed_token_bytes.as_ref().map_or_else(
            || self.token_bytes.values().map(Vec::len).max().unwrap_or(0),
            |packed| packed.iter().map(|(_, bytes)| bytes.len()).max().unwrap_or(0),
        )
    }

    pub(crate) fn token_bytes_iter(&self) -> Box<dyn Iterator<Item = (u32, &[u8])> + '_> {
        if let Some(packed) = &self.packed_token_bytes {
            Box::new(packed.iter())
        } else {
            Box::new(
                self.token_bytes
                    .iter()
                    .map(|(&token_id, bytes)| (token_id, bytes.as_slice())),
            )
        }
    }

    /// Bind a loaded constraint to an exact model vocabulary once.
    ///
    /// The deep byte-map equality check is paid at load/bind time; successful
    /// binding then lets repeated composition prove compatibility by `Arc` identity.
    #[doc(hidden)]
    pub(crate) fn bind_vocab_exact(&mut self, vocab: &crate::Vocab) -> Result<(), String> {
        if let Some(existing) = self.late_bind_vocab.get()
            && !existing.same_model_vocab(vocab)
        {
            return Err(
                "constraint was not compiled for the supplied exact vocabulary".to_string(),
            );
        }
        let entries = vocab.entries_arc();
        if Arc::ptr_eq(&self.token_bytes, &entries) {
            self.late_bind_vocab = OnceLock::from(vocab.clone());
        } else {
            if !self.token_bytes_match_vocab(vocab) {
                return Err("constraint was not compiled for the supplied vocabulary".to_string());
            }
            self.token_bytes = entries;
            // A successful exact bind establishes the precise public vocabulary
            // for every later late-subgrammar bind as well. Keep the caller's
            // already-built `Vocab` (and its pure derived-artifact cache) instead
            // of reconstructing the same bytes again in `constraint_vocab()`.
            self.late_bind_vocab = OnceLock::from(vocab.clone());
        }
        if let Some(overlay) = self.static_dynamic_overlay.as_mut() {
            for component in &mut overlay.segmented_parser_components {
                Arc::make_mut(&mut component.constraint).bind_vocab_exact(vocab)?;
            }
        }
        Ok(())
    }

    fn compute_dense_token_bytes(&self) -> Vec<Option<Box<[u8]>>> {
        let Some(max_token_id) = self.max_original_token_id() else {
            return Vec::new();
        };

        let mut dense = vec![None; max_token_id as usize + 1];
        for (token_id, bytes) in self.token_bytes_iter() {
            dense[token_id as usize] = Some(bytes.to_vec().into_boxed_slice());
        }
        dense
    }

    pub(crate) fn build_dense_token_bytes(&mut self) {
        self.token_bytes_dense = self.compute_dense_token_bytes();
    }

    pub(super) fn rebuild_state_internal_tsid_relation(&mut self) {
        let state_count = self.tokenizer.num_states() as usize;
        let mut relation = (0..state_count)
            .map(|_| SmallVec::<[u32; 2]>::new())
            .collect::<Vec<_>>();
        for (internal_tsid, states) in self.internal_tsid_groups().iter().enumerate() {
            for &state in states {
                if let Some(tsids) = relation.get_mut(state as usize) {
                    tsids.push(internal_tsid as u32);
                }
            }
        }
        for (state, tsids) in relation.iter_mut().enumerate() {
            if let Some(&primary) = self.state_to_internal_tsid.get(state)
                && !tsids.contains(&primary)
            {
                tsids.push(primary);
            }
            tsids.sort_unstable();
            tsids.dedup();
        }

        self.state_internal_tsid_offsets.clear();
        self.state_internal_tsid_offsets.reserve(state_count + 1);
        self.state_internal_tsids.clear();
        self.state_internal_tsid_offsets.push(0);
        for tsids in relation {
            self.state_internal_tsids.extend_from_slice(&tsids);
            self.state_internal_tsid_offsets
                .push(self.state_internal_tsids.len() as u32);
        }
    }

    pub(super) fn rebuild_runtime_product_state_lookup(&mut self) {
        self.runtime_product_state_by_source_subset.clear();
        let Some(source_offset) = self.runtime_source_state_offset else {
            return;
        };
        let product_states = source_offset as usize;
        if self.runtime_product_source_offsets.len() != product_states + 1
            || self.runtime_product_exact_source_states.len() != product_states
        {
            self.runtime_source_state_offset = None;
            self.runtime_product_source_offsets.clear();
            self.runtime_product_source_states.clear();
            self.runtime_product_exact_source_states.clear();
            return;
        }

        self.runtime_product_state_by_source_subset
            .reserve(product_states);
        for product_state in 0..product_states {
            let start = self.runtime_product_source_offsets[product_state] as usize;
            let end = self.runtime_product_source_offsets[product_state + 1] as usize;
            let Some(states) = self.runtime_product_source_states.get(start..end) else {
                self.runtime_product_state_by_source_subset.clear();
                self.runtime_source_state_offset = None;
                return;
            };
            self.runtime_product_state_by_source_subset
                .insert(states.into(), product_state as u32);
        }
    }

    pub(crate) fn max_original_token_id(&self) -> Option<u32> {
        self.body_max_original_token_id().into_iter()
            .chain(self.end_tokens.last().copied()).max()
    }

    pub(crate) fn body_max_original_token_id(&self) -> Option<u32> {
        self.token_bytes
            .keys()
            .next_back()
            .copied()
            .or_else(|| {
                self.packed_token_bytes
                    .as_ref()
                    .and_then(|packed| packed.max_token_id())
            })
            .into_iter()
            .chain(
                self.special_token_terminals
                    .iter()
                    .filter(|special| {
                        !self.is_late_grammar_placeholder_terminal(special.terminal_id)
                    })
                    .map(|special| special.token_id),
            )
            .max()
    }

    /// Remove compiler-only late-grammar sentinel token IDs from the runtime
    /// model-token coordinate while retaining their terminal metadata for the
    /// linker. Returns whether the token coordinate changed.
    ///
    /// External-grammar placeholders are compiled initially as exact special
    /// tokens so the ordinary compiler can build the parent grammar. Once a
    /// terminal is recorded in `late_grammar_slots`, that backing token ID is
    /// no longer a public/model token. Leaving it in the original-token map
    /// makes persisted sparse mask fragments wider than `mask_len()`, which
    /// correctly excludes unresolved linker sentinels.
    pub(crate) fn sanitize_late_grammar_placeholder_token_domain(&mut self) -> bool {
        if self.late_grammar_slots.is_empty() {
            return false;
        }

        let placeholder_terminals = self
            .late_grammar_slots
            .iter()
            .map(|slot| slot.terminal_id)
            .collect::<BTreeSet<_>>();
        self.sanitize_placeholder_terminal_token_domain(&placeholder_terminals)
    }

    /// Remove non-vocabulary compiler sentinel token IDs associated with an
    /// explicit set of placeholder terminals.
    pub(crate) fn sanitize_placeholder_terminal_token_domain(
        &mut self,
        placeholder_terminals: &BTreeSet<TerminalID>,
    ) -> bool {
        if placeholder_terminals.is_empty() || !self.has_original_token_map() {
            return false;
        }

        let mut placeholder_tokens = self
            .special_token_terminals
            .iter()
            .filter(|special| placeholder_terminals.contains(&special.terminal_id))
            .filter(|special| self.token_bytes_for_id(special.token_id).is_none())
            .filter(|special| {
                !self.special_token_terminals.iter().any(|other| {
                    other.token_id == special.token_id
                        && !placeholder_terminals.contains(&other.terminal_id)
                })
            })
            .map(|special| special.token_id)
            .collect::<Vec<_>>();
        placeholder_tokens.sort_unstable();
        placeholder_tokens.dedup();
        if placeholder_tokens.is_empty()
            || !placeholder_tokens.iter().any(|&token| {
                self.original_token_internal_at(token)
                    .is_some_and(|internal| internal != u32::MAX)
            })
        {
            return false;
        }

        let group_count = self.internal_token_count();
        let mut original_to_internal = self.original_token_map().to_vec();
        for token in placeholder_tokens {
            if let Some(internal) = original_to_internal.get_mut(token as usize) {
                *internal = u32::MAX;
            }
        }
        let mut internal_to_tokens = (0..group_count)
            .map(|_| Vec::<u32>::new())
            .collect::<Vec<_>>();
        for (original, &internal) in original_to_internal.iter().enumerate() {
            if internal == u32::MAX {
                continue;
            }
            if let Some(group) = internal_to_tokens.get_mut(internal as usize) {
                group.push(original as u32);
            }
        }

        self.original_token_to_internal = original_to_internal;
        self.packed_original_token_to_internal = None;
        self.deferred_original_token_to_internal = OnceLock::new();
        self.internal_token_to_tokens = internal_to_tokens;
        self.deferred_internal_token_to_tokens = OnceLock::new();

        // Portable per-token fragments and every aggregate derived from them
        // were built before the slot became linker-only. Force one rebuild in
        // the reduced public token domain; subsequent save/load can reuse the
        // clean caches normally.
        self.internal_token_buf_masks.clear();
        self.internal_token_buf_flat = Box::new([]);
        self.backed_internal_token_buf_flat = None;
        self.internal_token_buf_offsets = Box::new([]);
        self.word_group_buf_masks.clear();
        self.pair_word_group_buf_masks = Default::default();
        self.quad_word_group_buf_masks = Default::default();
        self.super_word_group_buf_masks = Default::default();
        self.mega_word_group_buf_masks = Default::default();
        self.giga_word_group_buf_masks = Default::default();
        self.word_group_sparse_masks.clear();
        self.word_group_prefix_buf_masks = Default::default();
        self.word_group_sparse_prefix_entries.clear();
        self.quad_group_sparse_masks.clear();
        self.quad_group_dense_masks.clear();
        self.byte_group_sparse_masks.clear();
        self.byte_group_dense_masks.clear();
        self.word_group_sparse_total_entries = 0;
        self.word_group_sparse_max_entries = 0;
        self.all_tokens_buf_mask = Box::new([]);
        self.heavy_token_dense_masks.clear();
        self.heavy_token_indices.clear();
        self.internal_token_buf_op_costs.clear();
        self.word_group_buf_op_costs.clear();
        self.total_internal_buf_cost = 0;
        self.heavy_total_cost = 0;
        self.light_avg_cost_x256 = 0;
        self.parser_runtime_caches_prebuilt = false;
        self.serialized_artifact_cache = None;
        true
    }

    pub(crate) fn is_late_grammar_placeholder_terminal(&self, terminal_id: u32) -> bool {
        self.late_grammar_slots
            .iter()
            .any(|slot| slot.terminal_id == terminal_id)
    }

    pub(crate) fn has_special_token_id(&self, token_id: u32) -> bool {
        self.special_token_terminals
            .iter()
            .any(|special| {
                special.token_id == token_id
                    && !self.is_late_grammar_placeholder_terminal(special.terminal_id)
            })
    }
}


impl Constraint {
    /// Original zero-byte IDs, using the retained index in compiled/loaded data.
    /// The map fallback supports deliberately hand-built internal constraints.
    pub(crate) fn empty_byte_token_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.packed_token_bytes.iter().flat_map(|packed| packed.empty_token_ids().iter().copied())
            .chain(self.packed_token_bytes.is_none().then_some(&self.token_bytes)
                .into_iter().flat_map(|tokens| tokens.iter()
                    .filter_map(|(&id, bytes)| bytes.is_empty().then_some(id))))
    }
}
