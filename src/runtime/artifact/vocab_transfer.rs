//! Dynamic vocabulary transfer and validation against original token bytes.

use super::{
    DynamicMaskAliasStore, DynamicMaskSliceTrie, DynamicMaskTrie, DynamicMaskVocab,
    dynamic_mask_vocab_layout_class,
};
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::classify_vocab_char_type;
use rayon::prelude::ParallelSliceMut;
use rustc_hash::FxHashSet;
use std::collections::BTreeMap;
use std::sync::Arc;
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskVocabArtifact {
    pub(super) trie: DynamicMaskTrie,
    pub(super) token_aliases: DynamicMaskAliasStore,
    pub(super) llg_slice_leftovers: Vec<Arc<DynamicMaskSliceTrie>>,
    pub(super) mask_tokenizer: Option<Tokenizer>,
    pub(super) full_to_mask_state: Vec<u32>,
    pub(super) grammar_quotiented: bool,
    pub(super) source_vocab_digest: Option<[u8; 32]>,
}

impl DynamicMaskVocab {
    pub(super) fn to_artifact_impl(
        &self,
        include_mask_quotient: bool,
    ) -> Option<DynamicMaskVocabArtifact> {
        if !self.initialized || self.pending_source.is_some() {
            return None;
        }
        let mask_quotient = include_mask_quotient
            .then(|| self.mask_tokenizer_quotient_for_transfer())
            .flatten();
        Some(DynamicMaskVocabArtifact {
            trie: self.trie.as_ref().clone(),
            token_aliases: self.token_aliases.clone(),
            llg_slice_leftovers: self.llg_slice_leftovers.as_ref().clone(),
            mask_tokenizer: mask_quotient
                .as_ref()
                .map(|(tokenizer, _)| tokenizer.clone()),
            full_to_mask_state: mask_quotient
                .map(|(_, full_to_mask_state)| full_to_mask_state)
                .unwrap_or_default(),
            grammar_quotiented: self.grammar_quotiented,
            source_vocab_digest: self.source_vocab_digest,
        })
    }

    pub(crate) fn to_artifact(&self) -> Option<DynamicMaskVocabArtifact> {
        self.to_artifact_impl(true)
    }

    /// Serialize only vocabulary-derived runtime data. Constraint-specific
    /// mask-tokenizer quotients and projections are reconstructed after load.
    pub(crate) fn to_vocab_artifact(&self) -> Option<DynamicMaskVocabArtifact> {
        self.to_artifact_impl(false)
    }

    /// External-vocabulary transfer can omit model-vocabulary-only slice/master
    /// acceleration for grammar quotients. `load_with_vocab` reattaches the
    /// already-prepared full-vocabulary structure in the parent process.
    pub(crate) fn to_external_vocab_artifact(&self) -> Option<DynamicMaskVocabArtifact> {
        let mut artifact = self.to_artifact_impl(false)?;
        if artifact.grammar_quotiented {
            artifact.llg_slice_leftovers.clear();
        }
        Some(artifact)
    }

    pub(crate) fn from_artifact(artifact: DynamicMaskVocabArtifact) -> Result<Self, String> {
        Self::from_artifact_impl(artifact, None)
    }

    pub(crate) fn from_external_vocab_artifact(
        artifact: DynamicMaskVocabArtifact,
        full_vocab_template: &DynamicMaskVocab,
    ) -> Result<Self, String> {
        Self::from_artifact_impl(artifact, Some(full_vocab_template))
    }

    pub(super) fn from_artifact_impl(
        artifact: DynamicMaskVocabArtifact,
        full_vocab_template: Option<&DynamicMaskVocab>,
    ) -> Result<Self, String> {
        if artifact.trie.nodes.is_empty() {
            return Err("dynamic-mask vocabulary artifact has no trie root".to_owned());
        }
        let llg_slice_leftovers = artifact.llg_slice_leftovers;
        let external_acceleration = artifact.grammar_quotiented
            && llg_slice_leftovers.is_empty()
            && full_vocab_template.is_some();
        let mut result = if external_acceleration {
            DynamicMaskVocab::from_alias_store(Arc::new(artifact.trie), artifact.token_aliases)
        } else {
            DynamicMaskVocab::from_alias_store_with_slices(
                Arc::new(artifact.trie),
                artifact.token_aliases,
                Some(&llg_slice_leftovers),
            )
        };
        if external_acceleration {
            result.inherit_llg_vocab_acceleration_without_master_from(
                full_vocab_template.expect("external acceleration template checked above"),
            );
        } else {
            result.llg_slice_leftovers = Arc::new(llg_slice_leftovers);
            if result.llg_master_admitted_words.is_empty() {
                result.rebuild_llg_master_admitted_words_from_master_trie();
            } else {
                result.rebuild_llg_master_residual_ops_from_master_trie();
            }
        }
        result.grammar_quotiented = artifact.grammar_quotiented;
        result.source_vocab_digest = artifact.source_vocab_digest;
        match artifact.mask_tokenizer {
            Some(tokenizer) => {
                if artifact.full_to_mask_state.is_empty()
                    || artifact
                        .full_to_mask_state
                        .iter()
                        .any(|&state| state >= tokenizer.num_states())
                {
                    return Err(
                        "dynamic-mask vocabulary artifact has an invalid mask-tokenizer quotient"
                            .to_owned(),
                    );
                }
                result.set_mask_tokenizer_quotient(tokenizer, artifact.full_to_mask_state);
            }
            None if !artifact.full_to_mask_state.is_empty() => {
                return Err(
                    "dynamic-mask vocabulary artifact has a quotient map without a tokenizer"
                        .to_owned(),
                );
            }
            None => {}
        }
        Ok(result)
    }
    pub(crate) fn restore_root_layout_metadata_from_token_bytes(
        &mut self,
        token_bytes: &BTreeMap<u32, Vec<u8>>,
    ) {
        let mut canonical_meta = Vec::<Option<(u16, bool)>>::new();
        canonical_meta.reserve(self.canonical_token_count());
        for canonical in 0..self.canonical_token_count() as u32 {
            let Some(originals) = self.token_ids(canonical) else {
                return;
            };
            let Some(first_original) = originals.first() else {
                return;
            };
            let Some(bytes) = token_bytes.get(first_original).map(Vec::as_slice) else {
                return;
            };
            canonical_meta.push((!bytes.is_empty()).then(|| {
                (
                    dynamic_mask_vocab_layout_class(classify_vocab_char_type(bytes), bytes),
                    std::str::from_utf8(bytes).is_ok(),
                )
            }));
        }

        let mut root_layout_classes = Vec::with_capacity(self.trie.children(0).len());
        let mut root_layout_all_valid_utf8 = Vec::with_capacity(self.trie.children(0).len());
        for edge in self.trie.children(0) {
            let mut class = None::<u16>;
            let mut all_valid_utf8 = true;
            let mut saw_token = false;
            for &canonical in self.trie.subtree_tokens(edge.child) {
                let Some(Some((token_class, valid_utf8))) = canonical_meta.get(canonical as usize)
                else {
                    return;
                };
                if class.is_some_and(|existing| existing != *token_class) {
                    return;
                }
                class = Some(*token_class);
                all_valid_utf8 &= *valid_utf8;
                saw_token = true;
            }
            let Some(class) = class.filter(|_| saw_token) else {
                return;
            };
            root_layout_classes.push(class);
            root_layout_all_valid_utf8.push(all_valid_utf8);
        }

        let trie = Arc::make_mut(&mut self.trie);
        trie.root_layout_classes = root_layout_classes;
        trie.root_layout_all_valid_utf8 = root_layout_all_valid_utf8;
    }

    /// Verify that this vocabulary-only runtime index represents exactly the
    /// supplied original token-id -> byte mapping. This is used when loading a
    /// self-contained dynamic artifact: the persisted trie is an accelerator,
    /// never an independent source of vocabulary semantics.
    pub(crate) fn matches_token_bytes_exact(&self, token_bytes: &BTreeMap<u32, Vec<u8>>) -> bool {
        if self.grammar_quotiented {
            return self.matches_grammar_quotiented_token_bytes(token_bytes);
        }
        if self.canonical_original_tokens.len() != token_bytes.len() {
            return false;
        }

        // Ordinary dynamic vocabularies canonicalize only byte-identical model
        // tokens. Keep the historical strong validation for those artifacts.
        let mut sorted_tokens = token_bytes
            .iter()
            .map(|(&token_id, bytes)| (token_id, bytes.as_slice()))
            .collect::<Vec<_>>();
        let sort_tokens = |left: &(u32, &[u8]), right: &(u32, &[u8])| {
            left.1.cmp(right.1).then_with(|| left.0.cmp(&right.0))
        };
        if rayon::current_num_threads() == 1 {
            sorted_tokens.sort_unstable_by(sort_tokens);
        } else {
            sorted_tokens.par_sort_unstable_by(sort_tokens);
        }
        let mut canonical_bytes = Vec::<&[u8]>::with_capacity(self.canonical_token_count());
        let mut start = 0usize;
        while start < sorted_tokens.len() {
            let bytes = sorted_tokens[start].1;
            let mut end = start + 1;
            while end < sorted_tokens.len() && sorted_tokens[end].1 == bytes {
                end += 1;
            }
            let canonical = canonical_bytes.len() as u32;
            let Some(originals) = self.token_ids(canonical) else {
                return false;
            };
            if originals.len() != end - start
                || !originals.iter().copied().eq(sorted_tokens[start..end]
                    .iter()
                    .map(|(token_id, _)| *token_id))
            {
                return false;
            }
            canonical_bytes.push(bytes);
            start = end;
        }
        if canonical_bytes.len() != self.canonical_token_count() {
            return false;
        }

        struct Frame {
            node: u32,
            next_child: usize,
            prefix_len: usize,
        }

        let mut canonical_seen = vec![false; canonical_bytes.len()];
        let mut prefix = Vec::<u8>::new();
        let mut frames = vec![Frame {
            node: 0,
            next_child: 0,
            prefix_len: 0,
        }];
        while !frames.is_empty() {
            let frame_index = frames.len() - 1;
            let node_id = frames[frame_index].node;
            if frames[frame_index].next_child == 0 {
                if let Some(canonical) = self.trie.node(node_id).token_id {
                    let canonical = canonical as usize;
                    if canonical >= canonical_seen.len()
                        || canonical_seen[canonical]
                        || canonical_bytes[canonical] != prefix.as_slice()
                    {
                        return false;
                    }
                    canonical_seen[canonical] = true;
                }
            }

            let children = self.trie.children(node_id);
            if frames[frame_index].next_child < children.len() {
                let edge = children[frames[frame_index].next_child].clone();
                frames[frame_index].next_child += 1;
                let prefix_len = prefix.len();
                prefix.extend_from_slice(self.trie.edge_bytes(&edge));
                frames.push(Frame {
                    node: edge.child,
                    next_child: 0,
                    prefix_len,
                });
            } else {
                let prefix_len = frames[frame_index].prefix_len;
                frames.pop();
                prefix.truncate(prefix_len);
            }
        }

        canonical_seen.into_iter().all(|seen| seen)
    }

    pub(super) fn matches_grammar_quotiented_token_bytes(
        &self,
        token_bytes: &BTreeMap<u32, Vec<u8>>,
    ) -> bool {
        if self.canonical_original_tokens.len() != token_bytes.len() {
            return false;
        }
        let mut covered = FxHashSet::<u32>::default();
        for &token_id in self.canonical_original_tokens.iter() {
            if !token_bytes.contains_key(&token_id) || !covered.insert(token_id) {
                return false;
            }
        }
        if covered.len() != token_bytes.len() {
            return false;
        }

        struct Frame {
            node: u32,
            next_child: usize,
            prefix_len: usize,
        }
        let mut canonical_seen = vec![false; self.canonical_token_count()];
        let mut prefix = Vec::<u8>::new();
        let mut frames = vec![Frame {
            node: 0,
            next_child: 0,
            prefix_len: 0,
        }];
        while !frames.is_empty() {
            let frame_index = frames.len() - 1;
            let node_id = frames[frame_index].node;
            if frames[frame_index].next_child == 0 {
                if let Some(canonical) = self.trie.node(node_id).token_id {
                    let canonical = canonical as usize;
                    if canonical >= canonical_seen.len() || canonical_seen[canonical] {
                        return false;
                    }
                    let Some(originals) = self.token_ids(canonical as u32) else {
                        return false;
                    };
                    let Some(representative) = originals.first() else {
                        return false;
                    };
                    if token_bytes
                        .get(representative)
                        .is_none_or(|bytes| bytes.as_slice() != prefix.as_slice())
                    {
                        return false;
                    }
                    canonical_seen[canonical] = true;
                }
            }
            let children = self.trie.children(node_id);
            if frames[frame_index].next_child < children.len() {
                let edge = children[frames[frame_index].next_child].clone();
                frames[frame_index].next_child += 1;
                let prefix_len = prefix.len();
                prefix.extend_from_slice(self.trie.edge_bytes(&edge));
                frames.push(Frame {
                    node: edge.child,
                    next_child: 0,
                    prefix_len,
                });
            } else {
                let prefix_len = frames[frame_index].prefix_len;
                frames.pop();
                prefix.truncate(prefix_len);
            }
        }
        canonical_seen.into_iter().all(|seen| seen)
    }
}
