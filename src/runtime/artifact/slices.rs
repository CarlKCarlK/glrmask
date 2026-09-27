//! Vocabulary slice admission and original-token walk materialization.

use super::{
    DYNAMIC_MASK_LLG_MASTER_CACHE_ID, DynamicMaskSliceTrie, DynamicMaskTrie, DynamicMaskVocab,
    dynamic_mask_llg_master_is_whitespace, dynamic_mask_llg_master_safe_chars,
};
use crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa;
use crate::ds::u8set::U8Set;
use std::sync::Arc;
use std::sync::OnceLock;
impl DynamicMaskVocab {
    /// Reuse only vocabulary-global proof languages and admitted-token masks.
    /// The master walk trie itself is deliberately excluded: a grammar quotient
    /// can merge tokens that differ in safe-string length/whitespace class, so
    /// sharing the full-vocabulary master trie would both lose the O2 runtime
    /// reduction and couple the wrong canonical coordinate to this constraint.
    pub(crate) fn inherit_llg_vocab_acceleration_without_master_from(&mut self, source: &Self) {
        self.llg_slice_leftovers = Arc::new(
            source
                .llg_slice_leftovers
                .iter()
                .filter(|slice| slice.cache_id() != DYNAMIC_MASK_LLG_MASTER_CACHE_ID)
                .cloned()
                .collect(),
        );
        self.llg_master_admitted_words = Arc::clone(&source.llg_master_admitted_words);
        self.llg_master_residual_ops = Arc::new(Vec::new());
        self.llg_master_max_safe_chars = source.llg_master_max_safe_chars;
    }

    pub(crate) fn set_llg_slice_leftovers(
        &mut self,
        slices: Vec<(
            u32,
            Arc<VocabPartitionDfa>,
            Arc<DynamicMaskTrie>,
            Arc<Vec<u32>>,
            U8Set,
            u32,
        )>,
    ) {
        let mut built = Vec::with_capacity(slices.len());
        for (
            cache_id,
            dfa,
            trie,
            slice_original_token_words,
            slice_token_bytes,
            slice_max_token_byte_len,
        ) in slices
        {
            let slice_node_token_markers = Self::build_node_token_markers(
                trie.as_ref(),
                &self.canonical_original_token_offsets,
                &self.canonical_original_tokens,
            );
            let full_walk_token_markers =
                Self::build_full_walk_token_markers(trie.as_ref(), &slice_node_token_markers);
            let (subtree_original_token_offsets, subtree_original_tokens) =
                Self::flatten_subtree_original_tokens(
                    trie.as_ref(),
                    &self.canonical_original_token_offsets,
                    &self.canonical_original_tokens,
                );
            built.push(Arc::new(DynamicMaskSliceTrie {
                cache_id,
                dfa,
                trie,
                full_walk_token_markers,
                subtree_original_token_offsets,
                subtree_original_tokens,
                slice_original_token_words,
                slice_token_bytes,
                slice_max_token_byte_len,
                first_bytes: OnceLock::new(),
            }));
        }
        self.llg_slice_leftovers = Arc::new(built);
        self.rebuild_llg_master_residual_ops_from_master_trie();
    }

    pub(crate) fn set_llg_master_admitted_words(
        &mut self,
        max_safe_chars: u16,
        admitted_words: Vec<Vec<u32>>,
    ) {
        debug_assert_eq!(admitted_words.len(), (usize::from(max_safe_chars) + 1) * 2);
        self.llg_master_max_safe_chars = max_safe_chars;
        self.llg_master_admitted_words = Arc::new(admitted_words);
    }

    #[inline(always)]
    pub(crate) fn llg_master_admitted_words(
        &self,
        safe_radius: u16,
        whitespace: bool,
    ) -> Option<&[u32]> {
        if self.llg_master_admitted_words.is_empty() {
            return None;
        }
        let radius = safe_radius.min(self.llg_master_max_safe_chars) as usize;
        self.llg_master_admitted_words
            .get(radius * 2 + usize::from(whitespace))
            .map(Vec::as_slice)
    }

    #[inline(always)]
    pub(crate) fn llg_master_residual_ops(
        &self,
        safe_radius: u16,
        whitespace: bool,
    ) -> Option<usize> {
        if self.llg_master_residual_ops.is_empty() {
            return None;
        }
        let radius = safe_radius.min(self.llg_master_max_safe_chars) as usize;
        self.llg_master_residual_ops
            .get(radius * 2 + usize::from(whitespace))
            .copied()
            .map(|ops| ops as usize)
    }

    #[inline(always)]
    pub(crate) fn llg_master_max_safe_chars(&self) -> u16 {
        self.llg_master_max_safe_chars
    }

    pub(super) fn rebuild_llg_master_residual_ops_from_master_trie(&mut self) {
        let Some(master) = self.llg_master_trie() else {
            self.llg_master_residual_ops = Arc::new(Vec::new());
            return;
        };
        let max_safe_chars = master
            .trie()
            .children(0)
            .iter()
            .enumerate()
            .filter_map(|(slot, _)| master.trie().root_layout_class(slot))
            .map(dynamic_mask_llg_master_safe_chars)
            .max()
            .unwrap_or(0);
        let mut rows = Vec::with_capacity((usize::from(max_safe_chars) + 1) * 2);
        for radius in 0..=max_safe_chars {
            for whitespace in [false, true] {
                let work = master
                    .trie()
                    .full_walk_ops_after_root_class_skips(|class| {
                        let safe_chars = dynamic_mask_llg_master_safe_chars(class);
                        (safe_chars != 0 && safe_chars <= radius)
                            || (whitespace && dynamic_mask_llg_master_is_whitespace(class))
                    })
                    .unwrap_or_else(|| master.trie().full_walk_ops().len());
                rows.push(u32::try_from(work).unwrap_or(u32::MAX));
            }
        }
        self.llg_master_residual_ops = Arc::new(rows);
    }

    /// Reconstruct cumulative admitted-token bitsets from the already-transferred
    /// master slice trie. Each root subtree has one exact `(safe_chars, whitespace)`
    /// language class, so this is linear in original-token membership plus the
    /// small number of cumulative rows and avoids serializing those dense rows.
    pub(super) fn rebuild_llg_master_admitted_words_from_master_trie(&mut self) {
        let Some(master) = self.llg_master_trie() else {
            self.llg_master_admitted_words = Arc::new(Vec::new());
            self.llg_master_residual_ops = Arc::new(Vec::new());
            self.llg_master_max_safe_chars = 0;
            return;
        };
        let root_children = master.trie().children(0);
        let max_safe_chars = root_children
            .iter()
            .enumerate()
            .filter_map(|(slot, _)| master.trie().root_layout_class(slot))
            .map(dynamic_mask_llg_master_safe_chars)
            .max()
            .unwrap_or(0);
        let word_len = self.all_original_token_words.len();
        let mut exact_safe_words = vec![vec![0u32; word_len]; usize::from(max_safe_chars) + 1];
        let mut whitespace_words = vec![0u32; word_len];
        for (slot, edge) in root_children.iter().enumerate() {
            let Some(class) = master.trie().root_layout_class(slot) else {
                continue;
            };
            let safe_chars = dynamic_mask_llg_master_safe_chars(class);
            let whitespace = dynamic_mask_llg_master_is_whitespace(class);
            if safe_chars == 0 && !whitespace {
                continue;
            }
            for &token in master.subtree_original_tokens(edge.child) {
                let word = token as usize / 32;
                if word >= word_len {
                    continue;
                }
                let bit = 1u32 << (token % 32);
                if safe_chars != 0 {
                    exact_safe_words[usize::from(safe_chars)][word] |= bit;
                }
                if whitespace {
                    whitespace_words[word] |= bit;
                }
            }
        }
        let mut admitted_words =
            Vec::<Vec<u32>>::with_capacity((usize::from(max_safe_chars) + 1) * 2);
        let mut safe_prefix = vec![0u32; word_len];
        for radius in 0..=usize::from(max_safe_chars) {
            if radius != 0 {
                for (target, &source) in safe_prefix.iter_mut().zip(&exact_safe_words[radius]) {
                    *target |= source;
                }
            }
            admitted_words.push(safe_prefix.clone());
            let mut with_whitespace = safe_prefix.clone();
            for (target, &source) in with_whitespace.iter_mut().zip(&whitespace_words) {
                *target |= source;
            }
            admitted_words.push(with_whitespace);
        }
        self.llg_master_max_safe_chars = max_safe_chars;
        self.llg_master_admitted_words = Arc::new(admitted_words);
        self.rebuild_llg_master_residual_ops_from_master_trie();
    }

    #[inline(always)]
    pub(crate) fn llg_master_trie(&self) -> Option<&DynamicMaskSliceTrie> {
        // O2 already walks a grammar quotient that is orders of magnitude
        // smaller than the model vocabulary. Switching to a separately
        // structured master trie loses that quotient and is unnecessary; the
        // exact ordinary quotient walk is both the semantic authority and the
        // cheaper coordinate here.
        if self.grammar_quotiented {
            return None;
        }
        self.llg_slice_by_cache_id(DYNAMIC_MASK_LLG_MASTER_CACHE_ID)
    }

    #[inline(always)]
    pub(crate) fn llg_slice_leftovers(&self) -> &[Arc<DynamicMaskSliceTrie>] {
        self.llg_slice_leftovers.as_ref()
    }

    #[inline(always)]
    pub(crate) fn has_llg_slice_leftovers(&self) -> bool {
        !self.llg_slice_leftovers.is_empty()
    }

    #[inline(always)]
    pub(crate) fn llg_slice_by_cache_id(&self, cache_id: u32) -> Option<&DynamicMaskSliceTrie> {
        self.llg_slice_leftovers
            .iter()
            .find(|slice| slice.cache_id() == cache_id)
            .map(Arc::as_ref)
    }

    #[inline(always)]
    pub(crate) fn residual_original_token_words_for(
        &self,
        trie: &DynamicMaskTrie,
    ) -> Option<&[u32]> {
        self.llg_slice_leftovers
            .iter()
            .find(|slice| slice.cache_id() & 0x100 != 0 && std::ptr::eq(trie, slice.trie()))
            .map(|slice| slice.slice_original_token_words())
    }

    #[inline(always)]
    pub(crate) fn full_walk_token_markers_for(&self, trie: &DynamicMaskTrie) -> &[u64] {
        if std::ptr::eq(trie, self.trie.as_ref()) {
            return self.full_walk_token_markers();
        }
        if let Some(slice) = self
            .llg_slice_leftovers
            .iter()
            .find(|slice| std::ptr::eq(trie, slice.trie()))
        {
            return slice.full_walk_token_markers();
        }
        debug_assert!(false, "unknown dynamic-mask walk trie");
        &[]
    }

    #[inline(always)]
    pub(crate) fn subtree_original_tokens_for(&self, trie: &DynamicMaskTrie, node: u32) -> &[u32] {
        if std::ptr::eq(trie, self.trie.as_ref()) {
            return self.subtree_original_tokens(node);
        }
        if let Some(slice) = self
            .llg_slice_leftovers
            .iter()
            .find(|slice| std::ptr::eq(trie, slice.trie()))
        {
            return slice.subtree_original_tokens(node);
        }
        debug_assert!(false, "unknown dynamic-mask walk trie");
        &[]
    }

    pub(super) fn flatten_subtree_original_tokens(
        trie: &DynamicMaskTrie,
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
    ) -> (Arc<Vec<u32>>, Arc<Vec<u32>>) {
        let subtree_canonical_tokens = trie.all_subtree_tokens();
        let mut offsets = Vec::with_capacity(subtree_canonical_tokens.len() + 1);
        let mut originals = Vec::new();
        offsets.push(0);
        for &canonical_token in subtree_canonical_tokens {
            let index = canonical_token as usize;
            let start = canonical_offsets[index] as usize;
            let end = canonical_offsets[index + 1] as usize;
            originals.extend_from_slice(&canonical_original_tokens[start..end]);
            offsets.push(originals.len() as u32);
        }
        (Arc::new(offsets), Arc::new(originals))
    }

    pub(super) fn build_node_token_markers(
        trie: &DynamicMaskTrie,
        canonical_offsets: &[u32],
        canonical_original_tokens: &[u32],
    ) -> Arc<Vec<u64>> {
        const FALLBACK_TAG: u64 = 1u64 << 63;
        let mut markers = Vec::with_capacity(trie.nodes.len());
        for node in &trie.nodes {
            let Some(canonical_token) = node.token_id else {
                markers.push(0);
                continue;
            };
            let index = canonical_token as usize;
            let start = canonical_offsets[index] as usize;
            let end = canonical_offsets[index + 1] as usize;
            let aliases = &canonical_original_tokens[start..end];
            let Some(&first_token) = aliases.first() else {
                markers.push(FALLBACK_TAG | (canonical_token as u64 + 1));
                continue;
            };
            let word = first_token / 32;
            let mut bits = 0u32;
            let mut one_word = true;
            for &token_id in aliases {
                if token_id / 32 != word {
                    one_word = false;
                    break;
                }
                bits |= 1u32 << (token_id % 32);
            }
            if one_word {
                debug_assert_ne!(bits, 0);
                debug_assert!(word < (1u32 << 31));
                markers.push((u64::from(word) << 32) | u64::from(bits));
            } else {
                markers.push(FALLBACK_TAG | (canonical_token as u64 + 1));
            }
        }
        Arc::new(markers)
    }

    pub(super) fn build_full_walk_token_markers(
        trie: &DynamicMaskTrie,
        node_token_markers: &[u64],
    ) -> Arc<Vec<u64>> {
        Arc::new(
            trie.full_walk_token_nodes()
                .iter()
                .map(|&node| {
                    debug_assert!((node as usize) < node_token_markers.len());
                    unsafe { *node_token_markers.get_unchecked(node as usize) }
                })
                .collect(),
        )
    }

    #[inline]
    pub(crate) fn subtree_original_tokens(&self, node: u32) -> &[u32] {
        let canonical_range = self.trie.subtree_token_index_range(node);
        let start = self.subtree_original_token_offsets[canonical_range.start] as usize;
        let end = self.subtree_original_token_offsets[canonical_range.end] as usize;
        &self.subtree_original_tokens[start..end]
    }
}
