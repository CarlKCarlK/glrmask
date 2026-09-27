//! Dynamic vocabulary initialization, slicing, and prefix-shared trie construction.

use crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::classify_vocab_char_type;
use crate::ds::u8set::U8Set;
use crate::runtime::artifact::Constraint;
use crate::runtime::artifact::DynamicMaskTrie;
use crate::runtime::artifact::DynamicMaskTrieEdge;
use crate::runtime::artifact::DynamicMaskVocab;
use crate::runtime::artifact::PackedDynamicMaskTokenAliases;
use crate::runtime::artifact::dynamic_mask_vocab_layout_class;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::sync::Arc;

impl Constraint {


    /// Attach the derived lexer artifacts used only by exact dynamic masking.
    ///
    /// In particular, finite projections of symbolic residual lexers are a
    /// mask-execution optimization, not part of compilation semantics. They
    /// can cost several milliseconds to construct for a constraint that may
    /// never be asked for a mask, so callers should invoke this only for the
    /// runtime vocabulary selected by `dynamic_mask_vocab_for_runtime` (or for
    /// an already-serialized projection that merely needs its derived table
    /// rebuilt after load).
    #[inline]
    fn dynamic_runtime_max_token_byte_len(&self, vocab: &DynamicMaskVocab) -> usize {
        // A materialized DynamicMaskVocab is built from the exact runtime token
        // vocabulary and stores the maximum token length in its trie root. Use
        // that O(1) metadata instead of rescanning every model token through
        // Constraint::max_token_byte_len(). The fallback is needed only for
        // unmaterialized transfer/compiler values.
        if vocab.is_initialized() {
            return vocab.max_token_byte_len();
        }
        if let Some(bound_vocab) = self.late_bind_vocab.get() {
            return bound_vocab.max_token_byte_len();
        }
        self.max_token_byte_len()
    }

    pub(crate) fn prepare_dynamic_virtual_residual_mask_projection(
        &self,
        vocab: &mut DynamicMaskVocab,
    ) {
        let max_token_len = self.dynamic_runtime_max_token_byte_len(vocab);
        if max_token_len > 0
            && vocab.mask_projection_tokenizer().is_none()
            && self.tokenizer.has_any_virtual_runtime()
        {
            let virtual_residual_mask_projection_enabled = std::env::var(
                "GLRMASK_DYNAMIC_VIRTUAL_RESIDUAL_MASK_PROJECTION",
            )
            .ok()
            .is_none_or(|value| !matches!(value.trim(), "0" | "false" | "no" | "off"));

            // RESOURCE budget, not a corpus-tuned state threshold. The dense
            // estimate charges one u32 index per candidate coordinate before
            // the projection retains only reachable sparse states.
            const DEFAULT_VIRTUAL_RESIDUAL_PROJECTION_MAX_DENSE_STATES: usize = 1024 * 1024;
            let max_dense_states = std::env::var(
                "GLRMASK_DYNAMIC_VIRTUAL_RESIDUAL_MASK_PROJECTION_MAX_DENSE_STATES",
            )
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_VIRTUAL_RESIDUAL_PROJECTION_MAX_DENSE_STATES);
            let projection_work = self
                .tokenizer
                .virtual_residual_mask_projection_dense_state_work(max_token_len);
            let within_budget = projection_work.is_some_and(|work| work <= max_dense_states);
            if virtual_residual_mask_projection_enabled
                && within_budget
                && let Some((mask_tokenizer, projections)) = self
                    .tokenizer
                    .virtual_residuals_mask_tokenizer_with_vocab(
                        max_token_len,
                        self.late_bind_vocab.get(),
                    )
            {
                vocab.set_virtual_residuals_mask_projection(mask_tokenizer, projections);
            }
        }
    }

    pub(crate) fn prepare_dynamic_mask_runtime_artifacts(&self, vocab: &mut DynamicMaskVocab) {
        let profile_runtime_mask = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_some();
        let max_token_started = profile_runtime_mask.then(std::time::Instant::now);
        let max_token_len = self.dynamic_runtime_max_token_byte_len(vocab);
        if let Some(started) = max_token_started {
            eprintln!(
                "[glrmask/profile][dynamic_mask_runtime_prepare] max_token_len_ms={:.3} value={}",
                started.elapsed().as_secs_f64() * 1e3,
                max_token_len,
            );
        }
        let projection_started = profile_runtime_mask.then(std::time::Instant::now);
        self.prepare_dynamic_virtual_residual_mask_projection(vocab);
        if max_token_len > 0
            && vocab.mask_projection_tokenizer().is_none()
            && self.tokenizer.has_any_virtual_runtime()
        {
            if let Some((mask_tokenizer, projections)) = self
                .tokenizer
                .virtual_binary_repeat_intersections_mask_tokenizer(max_token_len)
            {
                vocab.set_virtual_repeat_intersections_mask_projection(mask_tokenizer, projections);
            } else if let Some((mask_tokenizer, projection)) =
                self.tokenizer.virtual_unit_repeat_mask_tokenizer(max_token_len)
            {
                vocab.set_virtual_unit_repeat_mask_projection(mask_tokenizer, projection);
            }
        }
        if let Some(started) = projection_started {
            eprintln!(
                "[glrmask/profile][dynamic_mask_runtime_prepare] virtual_residual_projection_ms={:.3}",
                started.elapsed().as_secs_f64() * 1e3,
            );
        }

        let mask_execution_source = vocab.mask_runtime_tokenizer().unwrap_or(&self.tokenizer);
        let eager_mask_execution = std::env::var("GLRMASK_DYNAMIC_EAGER_MASK_EXECUTION")
            .ok()
            .is_none_or(|value| !matches!(value.trim(), "0" | "false" | "no" | "off"));
        // A finite projection whose only epsilon structure is its reset
        // dispatcher executes directly over raw scalar component rows. The
        // 0x8000 bound is the Flat16 representation boundary.
        let scalar_started = profile_runtime_mask.then(std::time::Instant::now);
        let scalar_dispatch = mask_execution_source.has_scalar_deterministic_dispatch();
        if let Some(started) = scalar_started {
            eprintln!(
                "[glrmask/profile][dynamic_mask_runtime_prepare] scalar_dispatch_proof_ms={:.3} result={}",
                started.elapsed().as_secs_f64() * 1e3,
                scalar_dispatch,
            );
        }
        if eager_mask_execution
            && mask_execution_source.has_epsilon_transitions()
            && !mask_execution_source.has_any_virtual_runtime()
            && !scalar_dispatch
        {
            let _ = vocab.prepare_mask_execution(&self.tokenizer, max_token_len);
        }
        let full_walk_started = profile_runtime_mask.then(std::time::Instant::now);
        vocab.prepare_full_walk_fast_transitions(&self.tokenizer);
        if let Some(started) = full_walk_started {
            eprintln!(
                "[glrmask/profile][dynamic_mask_runtime_prepare] full_walk_fast_ms={:.3}",
                started.elapsed().as_secs_f64() * 1e3,
            );
        }
        let master_slice_started = profile_runtime_mask.then(std::time::Instant::now);
        let max_safe_chars = u32::from(vocab.llg_master_max_safe_chars());
        for &slice_id in &[0u32, 3u32] {
            if let Some(slice) = vocab.llg_slice_by_cache_id(slice_id) {
                self.tokenizer.prepare_virtual_residual_master_slice_artifacts(
                    slice.dfa().start_state(),
                    slice.dfa().class_count(),
                    slice.dfa().byte_to_class_map(),
                    slice.dfa().transition_table(),
                    slice.dfa().accepting_map(),
                    slice.dfa().can_reach_accepting_map(),
                    max_safe_chars,
                );
            }
        }
        if let Some(started) = master_slice_started {
            eprintln!(
                "[glrmask/profile][dynamic_mask_runtime_prepare] master_slice_artifacts_ms={:.3}",
                started.elapsed().as_secs_f64() * 1e3,
            );
        }

        // Projected-terminal containment quotients are proof accelerators, not
        // prerequisites for exact masking. Build them only once a master-slice
        // proof survives the cheap runtime eligibility gates.
    }

    /// Return the direct-dynamic vocabulary, materializing it only when a
    /// dynamic mask is actually requested. Static constraints with complete
    /// possible-matches tables never pay this cost; deferred-PM constraints
    /// pay it on their first exact fallback instead of during compile/load.
    /// Symbolic lexer finite projections are likewise deferred to first mask:
    /// they are execution accelerators and must not inflate dynamic compile
    /// latency for constraints that are never sampled.
    pub(crate) fn dynamic_mask_vocab_for_runtime(&self) -> &DynamicMaskVocab {
        let needs_runtime_projection = self.tokenizer.has_any_virtual_runtime()
            && self.dynamic_mask_vocab.mask_projection_tokenizer().is_none();
        if self.dynamic_mask_vocab.is_initialized() && !needs_runtime_projection {
            return &self.dynamic_mask_vocab;
        }
        self.lazy_dynamic_mask_vocab.get_or_init(|| {
            let profile_runtime_mask = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_some();
            let total_started = profile_runtime_mask.then(std::time::Instant::now);
            let mut vocab = self.dynamic_mask_vocab.clone();
            let materialize_started = profile_runtime_mask.then(std::time::Instant::now);
            let _ = vocab.materialize_pending_source();
            if !vocab.is_initialized() {
                vocab = self.build_dynamic_mask_vocab();
            }
            if let Some(started) = materialize_started {
                eprintln!(
                    "[glrmask/profile][dynamic_mask_first_use] vocab_materialize_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
            // A loaded/static fallback may have had only the compact vocab
            // source serialized. Rebuild vocabulary-only slice metadata before
            // any lazy lexer quotient tries to consume those proof languages.
            let slice_started = profile_runtime_mask.then(std::time::Instant::now);
            // Recursive masking uses scoped leaf lexers, not this transitional
            // outer tokenizer. Its source-only projection/slice proofs cannot
            // be used by the provider and must not be built on its first mask.
            let recursive_provider = self.uses_compact_segmented_parser_runtime();
            if !recursive_provider {
                self.prepare_llg_slice_leftovers(&mut vocab);
            }
            if let Some(started) = slice_started {
                eprintln!(
                    "[glrmask/profile][dynamic_mask_first_use] slice_prepare_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
            let runtime_started = profile_runtime_mask.then(std::time::Instant::now);
            if !recursive_provider {
                self.prepare_dynamic_mask_runtime_artifacts(&mut vocab);
            }
            if let Some(started) = runtime_started {
                eprintln!(
                    "[glrmask/profile][dynamic_mask_first_use] lexer_runtime_prepare_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
            if let Some(started) = total_started {
                eprintln!(
                    "[glrmask/profile][dynamic_mask_first_use] total_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
            vocab
        })
    }

    /// Return an already-materialized dynamic-mask vocabulary without causing
    /// first-use runtime work.  Callers that only want to consult an optional
    /// memoization cache must use this instead of `dynamic_mask_vocab_for_runtime`.
    #[inline]
    pub(super) fn initialized_dynamic_mask_vocab_for_runtime(&self) -> Option<&DynamicMaskVocab> {
        if let Some(vocab) = self.lazy_dynamic_mask_vocab.get() {
            return Some(vocab);
        }
        self.dynamic_mask_vocab.is_initialized().then_some(&self.dynamic_mask_vocab)
    }

    pub(super) fn build_dynamic_mask_vocab(&self) -> DynamicMaskVocab {
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let total_started_at = profile.then(std::time::Instant::now);
        let collect_started_at = profile.then(std::time::Instant::now);
        // `token_bytes` is id-sorted. Runtime traversal needs byte-sorted,
        // duplicate-collapsed leaves instead. Borrow the source byte slices
        // until trie construction is complete: this avoids cloning every token
        // once into a BTreeMap and again into VocabPrefixTree::build_owned.
        let mut sorted_tokens = self.token_bytes_iter().collect::<Vec<_>>();
        let collect_ms = collect_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let sort_started_at = profile.then(std::time::Instant::now);
        let sort_tokens = |left: &(u32, &[u8]), right: &(u32, &[u8])| {
            left.1.cmp(right.1).then_with(|| left.0.cmp(&right.0))
        };
        if rayon::current_num_threads() == 1 {
            sorted_tokens.sort_unstable_by(sort_tokens);
        } else {
            sorted_tokens.par_sort_unstable_by(sort_tokens);
        }

        let sort_ms = sort_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let aliases_started_at = profile.then(std::time::Instant::now);
        // Match the compiler's OrderedVocab coordinate system exactly: trie
        // token ids are dense byte-order ids, and each id maps back to one or
        // more original model token ids. Using an original token id as the trie
        // id is incorrect when lexical byte order differs from token-id order.
        let mut token_aliases = Vec::with_capacity(sorted_tokens.len());
        let mut trie_entries = Vec::with_capacity(sorted_tokens.len());

        let mut start = 0usize;
        while start < sorted_tokens.len() {
            let bytes = sorted_tokens[start].1;
            let mut end = start + 1;
            while end < sorted_tokens.len() && sorted_tokens[end].1 == bytes {
                end += 1;
            }

            let ordered_id = token_aliases.len() as u32;
            let aliases = if end == start + 1 {
                PackedDynamicMaskTokenAliases::Single(sorted_tokens[start].0)
            } else {
                PackedDynamicMaskTokenAliases::Many(
                    sorted_tokens[start..end]
                        .iter()
                        .map(|(token_id, _)| *token_id)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )
            };
            token_aliases.push(Some(aliases));
            trie_entries.push((ordered_id as usize, bytes));
            start = end;
        }

        let aliases_ms = aliases_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let trie_started_at = profile.then(std::time::Instant::now);
        // The runtime trie remains one flat radix-trie arena, but construction
        // inserts one zero-byte structural child per character-type vocabulary
        // partition. This deliberately separates lexically different token
        // families without introducing a second trie type or per-partition
        // dense masks. The ordinary walker treats these as no-op edges and can
        // certify an entire partition subtree in one bounded-continuation test.
        let trie = Self::build_dynamic_mask_trie_partitioned(&trie_entries);
        let trie_ms = trie_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if let Some(total_started_at) = total_started_at {
            let alias_groups = token_aliases.iter().flatten().count();
            let alias_many = token_aliases
                .iter()
                .flatten()
                .filter(|aliases| matches!(aliases, PackedDynamicMaskTokenAliases::Many(_)))
                .count();
            eprintln!(
                "[glrmask/profile][runtime_dynamic_vocab] tokens={} unique_bytes={} aliases={} alias_many={} collect_ms={:.3} sort_ms={:.3} aliases_ms={:.3} trie_ms={:.3} trie_nodes={} trie_edges={} trie_bytes={} total_ms={:.3}",
                self.token_bytes_count(),
                trie_entries.len(),
                alias_groups,
                alias_many,
                collect_ms,
                sort_ms,
                aliases_ms,
                trie_ms,
                trie.nodes.len(),
                trie.edges.len(),
                trie.edge_bytes_len(),
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        DynamicMaskVocab::from_packed(Arc::new(trie), Arc::new(token_aliases))
    }

    pub(super) fn prepare_llg_slice_leftovers(&self, vocab: &mut DynamicMaskVocab) {
        if vocab.has_llg_slice_leftovers() {
            return;
        }

        const SAFE_PLUS_CACHE_ID: u32 = 0;
        const WHITESPACE_CACHE_ID: u32 = 3;
        const ASCII_WORD_CACHE_ID: u32 = 0x40;
        let safe_plus = Arc::new(
            VocabPartitionDfa::compile_utf8_regex(
                "llg-safe+",
                r#"[^"\\\x00-\x1F\x7F]+"#,
            )
            .expect("safe-string slice regex must compile"),
        );
        let whitespace = Arc::new(
            VocabPartitionDfa::compile_utf8_regex("llg-whitespace", r"[\x20\x0A\x0D\x09]+")
                .expect("whitespace slice regex must compile"),
        );
        let max_token_byte_len = vocab.max_token_byte_len();
        let ascii_slice_max_token_byte_len = std::env::var(
            "GLRMASK_EXPERIMENT_DIRECT_RESIDUAL_ASCII_WORD_SLICE_MAX_BYTES",
        )
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value != 0)
        .map_or(max_token_byte_len, |value| value.min(max_token_byte_len));
        let ascii_word = (max_token_byte_len != 0
            && std::env::var_os("GLRMASK_EXPERIMENT_DIRECT_RESIDUAL_ASCII_WORD_SLICE").is_some())
        .then(|| {
            Arc::new(
                VocabPartitionDfa::compile_utf8_regex(
                    "dynamic-optional-space-nonspace-safe",
                    &format!(r#" ?[^\s"\\\x00-\x1F\x7F]{{1,{ascii_slice_max_token_byte_len}}}"#),
                )
                .expect("bounded optional-space nonspace-safe slice regex must compile"),
            )
        });
        let word_len = vocab.all_original_token_words().len();
        let mut safe_words = vec![0u32; word_len];
        let mut whitespace_words = vec![0u32; word_len];
        let mut ascii_word_words = vec![0u32; word_len];
        let mut safe_token_bytes = U8Set::empty();
        let mut whitespace_token_bytes = U8Set::empty();
        let mut ascii_word_token_bytes = U8Set::empty();
        let mut safe_max_token_byte_len = 0u32;
        let mut whitespace_max_token_byte_len = 0u32;
        let mut ascii_word_max_token_byte_len = 0u32;
        let mut ascii_word_residual_entries =
            Vec::<(u16, usize, &[u8])>::with_capacity(vocab.canonical_token_count());
        let mut entries = Vec::<(u16, usize, &[u8])>::with_capacity(vocab.canonical_token_count());
        let mut max_safe_chars = 0u16;

        for canonical in 0..vocab.canonical_token_count() as u32 {
            let Some(originals) = vocab.token_ids(canonical) else { continue; };
            let Some(&first) = originals.first() else { continue; };
            let Some(bytes) = self.token_bytes_for_id(first) else { continue; };
            let is_safe = safe_plus.is_match(bytes);
            let safe_chars = if is_safe {
                let chars = std::str::from_utf8(bytes)
                    .expect("safe-string UTF-8 regex matched invalid UTF-8")
                    .chars()
                    .count();
                u16::try_from(chars).expect("model token exceeds u16 Unicode-scalar count")
            } else {
                0
            };
            let is_whitespace = whitespace.is_match(bytes);
            let is_ascii_word = ascii_word.as_ref().is_some_and(|slice| slice.is_match(bytes));
            max_safe_chars = max_safe_chars.max(safe_chars);
            entries.push((
                crate::runtime::dynamic_mask_llg_master_layout_class(safe_chars, is_whitespace),
                canonical as usize,
                bytes,
            ));
            if is_safe {
                safe_max_token_byte_len = safe_max_token_byte_len.max(bytes.len() as u32);
                for &byte in bytes {
                    safe_token_bytes.insert(byte);
                }
            }
            if is_whitespace {
                whitespace_max_token_byte_len = whitespace_max_token_byte_len.max(bytes.len() as u32);
                for &byte in bytes {
                    whitespace_token_bytes.insert(byte);
                }
            }
            if is_ascii_word {
                ascii_word_max_token_byte_len = ascii_word_max_token_byte_len.max(bytes.len() as u32);
                for &byte in bytes {
                    ascii_word_token_bytes.insert(byte);
                }
            } else {
                ascii_word_residual_entries.push((0, canonical as usize, bytes));
            }
            for &token in originals {
                let word = token as usize / 32;
                if word >= word_len {
                    continue;
                }
                let bit = 1u32 << (token % 32);
                if is_safe {
                    safe_words[word] |= bit;
                }
                if is_whitespace {
                    whitespace_words[word] |= bit;
                }
                if is_ascii_word {
                    ascii_word_words[word] |= bit;
                }
            }
        }

        entries.sort_unstable_by(|left, right| {
            left.2
                .is_empty()
                .cmp(&right.2.is_empty())
                .reverse()
                .then_with(|| left.0.cmp(&right.0))
                .then_with(|| left.2.cmp(right.2))
                .then_with(|| left.1.cmp(&right.1))
        });
        let master_trie = DynamicMaskTrie::from_partitioned_token_refs(&entries);
        ascii_word_residual_entries.sort_unstable_by(|left, right| {
            left.2.cmp(right.2).then_with(|| left.1.cmp(&right.1))
        });
        let ascii_word_residual_trie =
            DynamicMaskTrie::from_partitioned_token_refs(&ascii_word_residual_entries);
        let mut exact_safe_words = vec![vec![0u32; word_len]; usize::from(max_safe_chars) + 1];
        for &(class, canonical, _) in &entries {
            let safe_chars = crate::runtime::dynamic_mask_llg_master_safe_chars(class);
            if safe_chars == 0 {
                continue;
            }
            let Some(originals) = vocab.token_ids(canonical as u32) else { continue; };
            for &token in originals {
                let word = token as usize / 32;
                if word < word_len {
                    exact_safe_words[usize::from(safe_chars)][word] |= 1u32 << (token % 32);
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
        vocab.set_llg_master_admitted_words(max_safe_chars, admitted_words);
        let mut slices = vec![
            (
                SAFE_PLUS_CACHE_ID,
                Arc::clone(&safe_plus),
                Arc::new(DynamicMaskTrie::new()),
                Arc::new(safe_words),
                safe_token_bytes,
                safe_max_token_byte_len,
            ),
            (
                WHITESPACE_CACHE_ID,
                Arc::clone(&whitespace),
                Arc::new(DynamicMaskTrie::new()),
                Arc::new(whitespace_words),
                whitespace_token_bytes,
                whitespace_max_token_byte_len,
            ),
            (
                crate::runtime::DYNAMIC_MASK_LLG_MASTER_CACHE_ID,
                safe_plus,
                Arc::new(master_trie),
                Arc::new(Vec::new()),
                U8Set::empty(),
                0,
            ),
        ];
        if let Some(ascii_word) = ascii_word {
            slices.push((
                ASCII_WORD_CACHE_ID,
                ascii_word,
                Arc::new(ascii_word_residual_trie),
                Arc::new(ascii_word_words),
                ascii_word_token_bytes,
                ascii_word_max_token_byte_len,
            ));
        }
        vocab.set_llg_slice_leftovers(slices);
    }

    #[inline]
    fn dynamic_mask_lcp_len(left: &[u8], right: &[u8], from: usize) -> usize {
        let max_len = left.len().min(right.len());
        let mut index = from;
        while index < max_len && left[index] == right[index] {
            index += 1;
        }
        index
    }

    fn build_dynamic_mask_trie_children(
        entries: &[(usize, &[u8])],
        parent_prefix_len: usize,
        parent_node_id: u32,
        trie: &mut DynamicMaskTrie,
    ) {
        let child_edges = Self::build_dynamic_mask_trie_child_edges(
            entries,
            parent_prefix_len,
            trie,
        );

        if !child_edges.is_empty() {
            let first_child = trie.edges.len() as u32;
            let child_len = child_edges.len() as u32;
            trie.edges.extend(child_edges);
            let parent = &mut trie.nodes[parent_node_id as usize];
            parent.first_child = first_child;
            parent.child_len = child_len;
        }
    }

    fn build_dynamic_mask_trie_child_edges(
        entries: &[(usize, &[u8])],
        parent_prefix_len: usize,
        trie: &mut DynamicMaskTrie,
    ) -> SmallVec<[DynamicMaskTrieEdge; 4]> {
        let mut child_edges = SmallVec::<[DynamicMaskTrieEdge; 4]>::new();
        let mut index = 0usize;
        while index < entries.len() {
            let group_start = index;
            let next_byte = entries[index].1[parent_prefix_len];
            index += 1;
            while index < entries.len() && entries[index].1[parent_prefix_len] == next_byte {
                index += 1;
            }
            let group = &entries[group_start..index];
            let (child, child_prefix_len) =
                Self::build_dynamic_mask_trie_node(group, parent_prefix_len, trie);
            let (byte_start, byte_len) =
                trie.push_edge_bytes(&group[0].1[parent_prefix_len..child_prefix_len]);
            child_edges.push(DynamicMaskTrieEdge {
                byte_start,
                byte_len,
                child,
            });
        }
        child_edges
    }

    fn build_dynamic_mask_trie_node(
        entries: &[(usize, &[u8])],
        parent_prefix_len: usize,
        trie: &mut DynamicMaskTrie,
    ) -> (u32, usize) {
        debug_assert!(!entries.is_empty());
        let prefix_len = Self::dynamic_mask_lcp_len(
            entries.first().expect("nonempty entries").1,
            entries.last().expect("nonempty entries").1,
            parent_prefix_len,
        );
        let has_token = entries[0].1.len() == prefix_len;
        let node_id = trie.nodes.len() as u32;
        trie.nodes.push(crate::runtime::artifact::DynamicMaskTrieNode {
            token_id: has_token.then_some(entries[0].0 as u32),
            first_child: 0,
            child_len: 0,
            subtree_token_start: 0,
            subtree_token_end: 0,
            subtree_bytes: [0; 4],
            subtree_first_bytes: [0; 4],
            prefix_byte_len: 0,
            subtree_max_byte_len: 0,
        });

        let child_entries = if has_token { &entries[1..] } else { entries };
        if !child_entries.is_empty() {
            Self::build_dynamic_mask_trie_children(child_entries, prefix_len, node_id, trie);
        }

        (node_id, prefix_len)
    }



    pub(crate) fn build_dynamic_mask_trie_partitioned(
        entries: &[(usize, &[u8])],
    ) -> DynamicMaskTrie {
        // Character-type classification is substantially more expensive than
        // comparing the resulting small integer. Do it once per vocabulary
        // entry rather than O(log N) times from the sort comparator.
        let mut classified_entries = entries
            .iter()
            .map(|&(token_id, bytes)| {
                (
                    dynamic_mask_vocab_layout_class(classify_vocab_char_type(bytes), bytes),
                    token_id,
                    bytes,
                )
            })
            .collect::<Vec<_>>();
        classified_entries.sort_unstable_by(|left, right| {
            right
                .2
                .is_empty()
                .cmp(&left.2.is_empty())
                .then_with(|| left.0.cmp(&right.0))
                .then_with(|| left.2.cmp(right.2))
                .then_with(|| left.1.cmp(&right.1))
        });
        let ordered_entries = classified_entries
            .into_iter()
            .map(|(_, token_id, bytes)| (token_id, bytes))
            .collect::<Vec<_>>();
        let entries = ordered_entries.as_slice();
        let mut trie = DynamicMaskTrie::new();
        if entries.is_empty() {
            return trie;
        }

        // Empty byte strings live directly on the global root. There can be at
        // most one canonical entry after byte-string alias collapsing.
        let mut start = 0usize;
        if entries[0].1.is_empty() {
            trie.nodes[0].token_id = Some(entries[0].0 as u32);
            start = 1;
        }

        let mut partition_edges = SmallVec::<[DynamicMaskTrieEdge; 16]>::new();
        let mut index = start;
        while index < entries.len() {
            let partition = dynamic_mask_vocab_layout_class(
                classify_vocab_char_type(entries[index].1),
                entries[index].1,
            );
            let partition_start = index;
            index += 1;
            while index < entries.len()
                && dynamic_mask_vocab_layout_class(
                    classify_vocab_char_type(entries[index].1),
                    entries[index].1,
                ) == partition
            {
                index += 1;
            }
            let partition_entries = &entries[partition_start..index];

            // Structural partition node. Its incoming zero-byte edge consumes
            // no vocabulary byte; below it the existing compressed radix-trie
            // builder is used unchanged.
            let partition_node = trie.nodes.len() as u32;
            trie.nodes.push(crate::runtime::artifact::DynamicMaskTrieNode {
                token_id: None,
                first_child: 0,
                child_len: 0,
                subtree_token_start: 0,
                subtree_token_end: 0,
                subtree_bytes: [0; 4],
                subtree_first_bytes: [0; 4],
                prefix_byte_len: 0,
                subtree_max_byte_len: 0,
            });
            Self::build_dynamic_mask_trie_children(
                partition_entries,
                0,
                partition_node,
                &mut trie,
            );
            let (byte_start, byte_len) = trie.push_edge_bytes(&[]);
            partition_edges.push(DynamicMaskTrieEdge {
                byte_start,
                byte_len,
                child: partition_node,
            });
        }

        if !partition_edges.is_empty() {
            let first_child = trie.edges.len() as u32;
            let child_len = partition_edges.len() as u32;
            trie.edges.extend(partition_edges);
            trie.nodes[0].first_child = first_child;
            trie.nodes[0].child_len = child_len;
        }

        trie.finalize_subtree_metadata();
        trie
    }
}
