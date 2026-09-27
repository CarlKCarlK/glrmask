//! Encode the current artifact without materializing already-backed sections.

use rayon::prelude::*;
use super::{CONSTRAINT_HEADER_LEN, CONSTRAINT_MAGIC, CONSTRAINT_VERSION, CURRENT_CORE_FLAG_OMIT_TSID_INVERSE, CURRENT_CORE_MAGIC, Constraint, ConstraintArtifactCurrentCoreBaseRef, ConstraintArtifactCurrentRuntimeRef, PackedInternalTokenBufMask, ROOT_POLICY_MAGIC, SECTION_HEADER_LEN, SECTION_MAGIC, constraint_serialized_weight_pool, encode_composition_metadata_for_save, encode_current_runtime_wire, encode_internal_token_buf_masks, encode_static_virtual_residual_mask_wire, encode_token_mask_cache, packed_constraint_serialized_weight_ids, segmented_runtime_artifact_ref};

/// Clone a large already-serialized artifact without forcing one core to move
/// the entire byte slab.  Current JS artifacts are ~20 MiB, so a serial
/// `Vec::clone` is itself multiple milliseconds on some hosts even though the
/// bytes need no transformation at all.
pub(super) fn clone_serialized_artifact(bytes: &[u8]) -> Vec<u8> {
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
                        let rules = self.retained_table_rules()
                            .expect("validated retained grammar rules must remain readable");
                        let bytes = crate::compiler::glr::table::artifact_serde::to_compact_bytes_with_rules(
                            &self.table, rules,
                        );
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
                                dynamic_mask_vocab: self
                                    .uses_dynamic_runtime()
                                    .then(|| self.dynamic_mask_vocab.to_vocab_artifact())
                                    .flatten(),
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
            + SECTION_HEADER_LEN
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
            + SECTION_HEADER_LEN
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
        let payload_len = SECTION_HEADER_LEN
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
            let header_len = CONSTRAINT_HEADER_LEN + SECTION_HEADER_LEN;
            let (header, mut body) = bytes.split_at_mut(header_len);
            let mut pos = 0usize;
            header[pos..pos + CONSTRAINT_MAGIC.len()].copy_from_slice(&CONSTRAINT_MAGIC);
            pos += CONSTRAINT_MAGIC.len();
            header[pos..pos + 2].copy_from_slice(&CONSTRAINT_VERSION.to_le_bytes());
            pos += 2;
            header[pos..pos + 8].copy_from_slice(&(payload_len as u64).to_le_bytes());
            pos += 8;
            header[pos..pos + SECTION_MAGIC.len()].copy_from_slice(&SECTION_MAGIC);
            pos += SECTION_MAGIC.len();
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
        bytes.extend_from_slice(&CONSTRAINT_VERSION.to_le_bytes());
        let payload_len_offset = bytes.len();
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&SECTION_MAGIC);
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
}
