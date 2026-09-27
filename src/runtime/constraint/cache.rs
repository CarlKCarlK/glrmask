//! Runtime preparation and cache installation; preserves lazy and parallel build ordering.

use crate::automata::lexer::Lexer;
use crate::runtime::artifact::Constraint;
use crate::runtime::artifact::DenseBufMaskRows;
use crate::runtime::artifact::DirectRegularTerminalSupport;
use crate::runtime::mask_mapping::FinalMaskMapping;
use crate::runtime::state::ConstraintState;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::sync::Mutex;
use super::mask_cache::FinalTokenSetPlan;

pub(super) const INITIAL_COMMIT_PRIME_MAX_TOKENS: usize = 16;


pub(super) fn initial_commit_prime_token_ids(mask: &[u32]) -> Option<Vec<u32>> {
    let mut token_ids = Vec::new();
    for (word_index, &word) in mask.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            if token_ids.len() == INITIAL_COMMIT_PRIME_MAX_TOKENS {
                return None;
            }
            let bit = remaining.trailing_zeros() as usize;
            token_ids.push((word_index * 32 + bit) as u32);
            remaining &= remaining - 1;
        }
    }
    Some(token_ids)
}


#[derive(Default)]
struct TokenMaskCacheBuildProfile {
    word_block_ms: f64,
    quad_block_ms: f64,
    byte_block_ms: f64,
    block_ms: f64,
    pair_ms: f64,
    quad_ms: f64,
    super_ms: f64,
    mega_ms: f64,
    giga_ms: f64,
    all_tokens_ms: f64,
    heavy_ms: f64,
    flat_ms: f64,
    costs_ms: f64,
    derived_ms: f64,
}

impl Constraint {


    #[cold]
    fn prime_initial_commit_hot_path(&self) {
        let mut state = ConstraintState {
            terminated: false,
            constraint: self,
            state: self.initial_state_map(),
            buffers: Default::default(),
            generation: 0,
            mask_cache: Mutex::new(None),
            mask_scratch: Arc::new(Mutex::new(crate::runtime::state::MaskScratch::for_constraint(self))),
        };
        state.prefill_mask_cache();
        state.reserve_linear_stack_hot_path();

        let token_ids = {
            let cache = state.mask_cache.lock().unwrap();
            let Some(mask) = cache.as_ref().map(|cache| cache.mask.as_slice()) else {
                return;
            };
            let Some(token_ids) = initial_commit_prime_token_ids(mask) else {
                return;
            };
            token_ids
        };

        let initial_state = &state.state;
        let buffers = &mut state.buffers;
        crate::runtime::commit::prime_initial_commits(self, initial_state, buffers, &token_ids);
    }

    pub(crate) fn rebuild_dynamic_runtime_caches(&mut self) {
        self.tokenizer_has_epsilon_transitions = self.tokenizer.has_epsilon_transitions();
        if self.table.unconditional_advance.len() != self.table.num_states as usize
            || self
                .table
                .unconditional_advance
                .iter()
                .any(|row| row.len() != self.table.num_terminals as usize)
        {
            self.table.rebuild_unconditional_advance_rows();
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let total_started_at = profile.then(std::time::Instant::now);
        // `terminal_live_states` is a legacy/static derived cache. Dynamic mask
        // and commit paths do not consume it, so rebuilding it here only adds
        // dynamic compile/load latency and allocations.
        self.terminal_live_states.clear();
        let started_at = profile.then(std::time::Instant::now);
        if self.table.guarded_shift_index.len() != self.table.num_states as usize {
            if self.table.has_guarded_stack_shifts() {
                self.table.rebuild_guarded_shift_index();
            } else {
                self.table.guarded_shift_index.clear();
            }
        }
        let guarded_shift_ms = started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let mut dynamic_mask_vocab = std::mem::take(&mut self.dynamic_mask_vocab);
        let small_ready_runtime = dynamic_mask_vocab.is_initialized()
            && self.tokenizer.num_states() <= 64
            && self.table.num_states <= 32
            && !self.uses_sparse_direct_regular_runtime();
        let build_vocab = || {
            let started_at = profile.then(std::time::Instant::now);
            if !dynamic_mask_vocab.is_initialized() {
                let _ = dynamic_mask_vocab.materialize_pending_source();
            }
            if !dynamic_mask_vocab.is_initialized() {
                let mut materialized = self.build_dynamic_mask_vocab();
                materialized.inherit_dynamic_lexer_metadata_from(&dynamic_mask_vocab);
                dynamic_mask_vocab = materialized;
            }
            let elapsed = started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            (dynamic_mask_vocab, elapsed)
        };
        let build_fast = || {
            let started_at = profile.then(std::time::Instant::now);
            let transitions = self.compute_tokenizer_fast_transitions();
            let elapsed = started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            (transitions, elapsed)
        };
        let build_support = || {
            let started_at = profile.then(std::time::Instant::now);
            let support = if self.uses_sparse_direct_regular_runtime() {
                self.direct_regular_automaton
                    .as_ref()
                    .map_or_else(DirectRegularTerminalSupport::default, |automaton| {
                        DirectRegularTerminalSupport::build(
                            automaton,
                            self.table.num_terminals as usize,
                        )
                    })
            } else {
                DirectRegularTerminalSupport::default()
            };
            let elapsed = started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            (support, elapsed)
        };
        let (
            ((mut dynamic_mask_vocab, dynamic_vocab_ms), (tokenizer_fast_transitions, tokenizer_fast_ms)),
            (direct_regular_terminal_support, support_ms),
        ) = if rayon::current_num_threads() == 1 || small_ready_runtime {
            ((build_vocab(), build_fast()), build_support())
        } else {
            rayon::join(|| rayon::join(build_vocab, build_fast), build_support)
        };
        dynamic_mask_vocab.set_direct_regular_terminal_support(
            direct_regular_terminal_support,
        );
        let slice_leftovers_started_at = profile.then(std::time::Instant::now);
        self.prepare_llg_slice_leftovers(&mut dynamic_mask_vocab);
        let slice_leftovers_ms = slice_leftovers_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        // Master-slice prover preparation remains available for O1 experiments
        // and for artifacts that actually carry a master trie. O2 deliberately
        // keeps its grammar quotient as the walk coordinate, so its master trie
        // accessor returns None and none of this eager prover work is charged to
        // the O2 build path.
        let o2_prepared_master = dynamic_mask_vocab.is_grammar_quotiented()
            && dynamic_mask_vocab.llg_master_trie().is_some()
            && std::env::var_os("GLRMASK_DISABLE_O2_PREPARED_MASTER_PROVERS").is_none();
        let restored_complete_master = dynamic_mask_vocab
            .has_complete_prepared_master_prover_rows(self.tokenizer.num_states() as usize);
        // Pay bounded scalar-proof setup before serving rather than during a
        // cold mask. This is a performance policy only; skipped or failed proof
        // candidates retain the exact vocabulary walk. O2 keeps its own policy.
        let ordinary_eager = self.uses_dynamic_runtime()
            && !dynamic_mask_vocab.is_grammar_quotiented()
            && std::env::var_os("GLRMASK_DISABLE_EAGER_CONTAINMENT_QUOTIENTS").is_none();
        let eager_containment_quotients = o2_prepared_master || ordinary_eager
            || std::env::var_os("GLRMASK_EXPERIMENT_EAGER_CONTAINMENT_QUOTIENTS").is_some();
        let eager_component_state_cap = ordinary_eager.then(|| {
            std::env::var("GLRMASK_EAGER_CONTAINMENT_MAX_STATES")
                .ok().and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(3_000)
        });
        if eager_containment_quotients {
            let started = std::time::Instant::now();
            let force_partition_provers =
                std::env::var_os("GLRMASK_EXPERIMENT_PREPARED_PARTITION_PROVERS").is_some();
            if force_partition_provers {
                let candidates = (0..self.tokenizer.num_terminals()).collect::<Vec<_>>();
                let quotients = self
                    .tokenizer
                    .build_terminal_projected_quotients_for_containment_candidates(&candidates);
                dynamic_mask_vocab.set_projected_terminal_quotients(quotients);
            } else if !restored_complete_master {
                let safe_plus = dynamic_mask_vocab
                    .llg_slice_by_cache_id(0)
                    .expect("safe+ slice prepared before containment quotient construction");
                dynamic_mask_vocab.prepare_runtime_projected_terminal_quotients(
                    &self.tokenizer,
                    &safe_plus.slice_token_bytes(),
                    eager_component_state_cap,
                );
            }
            if profile || std::env::var_os("GLRMASK_PROFILE_PREPARED_MASTER_PROVERS").is_some() {
                eprintln!(
                    "[glrmask/profile][eager_containment_quotients] prepared={} elapsed_ms={:.3}",
                    dynamic_mask_vocab.has_projected_terminal_quotients(),
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
            let prepare_master = o2_prepared_master
                || std::env::var_os("GLRMASK_EXPERIMENT_PREPARED_MASTER_PROVERS").is_some();
            if prepare_master {
                let proof_started = std::time::Instant::now();
                let force_master =
                    std::env::var_os("GLRMASK_EXPERIMENT_PREPARED_MASTER_PROVERS").is_some();
                let (entries, product_pairs) = if restored_complete_master && !force_master {
                    (0, 0)
                } else {
                    let include_safe_radii =
                        std::env::var_os("GLRMASK_EXPERIMENT_PREPARED_SAFE_RADII").is_some();
                    dynamic_mask_vocab.prepare_master_provers_all_sources(
                        &self.tokenizer,
                        self.tokenizer.num_states() as usize,
                        include_safe_radii,
                    )
                };
                if profile
                    || std::env::var_os("GLRMASK_PROFILE_PREPARED_MASTER_PROVERS").is_some()
                {
                    eprintln!(
                        "[glrmask/profile][prepared_master_provers] entries={} product_pairs={} restored={} elapsed_ms={:.3}",
                        entries,
                        product_pairs,
                        restored_complete_master && !force_master,
                        proof_started.elapsed().as_secs_f64() * 1e3,
                    );
                }
                let max_safe_chars = u32::from(dynamic_mask_vocab.llg_master_max_safe_chars());
                for &slice_id in &[0u32, 3u32] {
                    if let Some(slice) = dynamic_mask_vocab.llg_slice_by_cache_id(slice_id) {
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
            }
        }
        // Mask-runtime acceleration derived solely from the compiled
        // constraint belongs to build/finalization, not to the first token.
        // In particular, symbolic/virtual residual master-slice preparation can
        // cost milliseconds on pathological schemas. Deferring it would turn
        // build work into an unaccounted first-mask latency spike and make
        // warmup semantics affect measured TBM.
        self.prepare_dynamic_mask_runtime_artifacts(&mut dynamic_mask_vocab);
        let has_dense_mask_projection =
            dynamic_mask_vocab.has_dense_mask_tokenizer_projection();
        let terminal_observation_enabled =
            self.dynamic_terminal_observation_classes_enabled(&dynamic_mask_vocab);
        let terminal_observation_started_at = profile.then(std::time::Instant::now);
        let terminal_observation_classes = if !terminal_observation_enabled {
            Vec::new()
        } else if has_dense_mask_projection {
            dynamic_mask_vocab.terminal_observation_classes_cloned()
        } else if dynamic_mask_vocab.has_terminal_observation_classes() {
            dynamic_mask_vocab.terminal_observation_classes_cloned()
        } else {
            self.build_dynamic_terminal_observation_classes()
        };
        dynamic_mask_vocab.set_terminal_observation_classes(terminal_observation_classes);
        let terminal_observation_ms = terminal_observation_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let hot_frontier_started_at = profile.then(std::time::Instant::now);
        self.direct_regular_dynamic_hot_frontiers = self
            .compute_direct_regular_dynamic_hot_frontiers(
                dynamic_mask_vocab.direct_regular_terminal_support(),
            );
        let hot_frontier_ms = hot_frontier_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let hot_frontier_count = self.direct_regular_dynamic_hot_frontiers.len();
        self.dynamic_mask_vocab = dynamic_mask_vocab;
        self.tokenizer_fast_transitions = tokenizer_fast_transitions;
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][dynamic_runtime_finalize] guarded_shift_ms={:.3} dynamic_vocab_ms={:.3} tokenizer_fast_ms={:.3} direct_regular_support_ms={:.3} slice_leftovers_ms={:.3} terminal_observation_ms={:.3} hot_frontier_ms={:.3} hot_frontiers={} total_ms={:.3}",
                guarded_shift_ms,
                dynamic_vocab_ms,
                tokenizer_fast_ms,
                support_ms,
                slice_leftovers_ms,
                terminal_observation_ms,
                hot_frontier_ms,
                hot_frontier_count,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    pub(crate) fn prebuild_token_mask_caches(&mut self) {
        self.internal_token_buf_masks = self.compute_buf_masks();
        let _ = self.rebuild_token_mask_derived_caches(false);
    }

    fn rebuild_token_mask_derived_caches(
        &mut self,
        profile: bool,
    ) -> TokenMaskCacheBuildProfile {
        let skip_load_dense_group_caches = self.packed_parser_dwa.is_some()
            && std::env::var_os("GLRMASK_SKIP_LOAD_DENSE_GROUP_CACHES").is_some();
        let skip_load_sliding_dense_caches = self.packed_parser_dwa.is_some()
            && std::env::var_os("GLRMASK_SKIP_LOAD_SLIDING_DENSE_CACHES").is_some();
        self.word_group_buf_masks = Vec::new();
        let block_started_at = profile.then(std::time::Instant::now);
        let skip_small_group_caches = std::env::var("GLRMASK_SKIP_SMALL_GROUP_MASK_CACHES")
            .map(|value| {
                let value = value.trim();
                value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
        let expected_word_groups = self.internal_token_buf_mask_count().div_ceil(64);
        let prebuilt_word_blocks =
            (self.word_group_sparse_masks.len() == expected_word_groups).then(|| {
                let groups = std::mem::take(&mut self.word_group_sparse_masks);
                let total_entries = groups.iter().map(Vec::len).sum::<usize>();
                let max_entries = groups.iter().map(Vec::len).max().unwrap_or(0);
                (groups, total_entries, max_entries)
            });
        let build_word_blocks = || {
            let started = profile.then(std::time::Instant::now);
            let reused = prebuilt_word_blocks.is_some();
            let result = prebuilt_word_blocks
                .unwrap_or_else(|| self.compute_token_block_sparse_masks(64));
            let ms = if reused {
                0.0
            } else {
                started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
            };
            (result, ms)
        };
        let build_quad_blocks = || {
            let started = profile.then(std::time::Instant::now);
            let result = if skip_small_group_caches {
                (Vec::new(), 0, 0)
            } else {
                self.compute_token_block_sparse_masks(4)
            };
            let ms = started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            (result, ms)
        };
        let build_byte_blocks = || {
            let started = profile.then(std::time::Instant::now);
            let result = if skip_small_group_caches {
                (Vec::new(), 0, 0)
            } else {
                self.compute_token_block_sparse_masks(8)
            };
            let ms = started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            (result, ms)
        };
        let ((word_blocks, word_block_ms), (quad_blocks, quad_block_ms), (byte_blocks, byte_block_ms)) =
            if rayon::current_num_threads() == 1 {
                (build_word_blocks(), build_quad_blocks(), build_byte_blocks())
            } else {
                let (word, (quad, byte)) = rayon::join(
                    build_word_blocks,
                    || rayon::join(build_quad_blocks, build_byte_blocks),
                );
                (word, quad, byte)
            };
        let block_masks = (word_blocks, quad_blocks, byte_blocks);
        let (
            (word_group_sparse_masks, word_group_sparse_total_entries, word_group_sparse_max_entries),
            (quad_group_sparse_masks, _, _),
            (byte_group_sparse_masks, _, _),
        ) = block_masks;
        self.word_group_sparse_masks = word_group_sparse_masks;
        self.word_group_prefix_buf_masks = if skip_load_dense_group_caches {
            DenseBufMaskRows::default()
        } else {
            self.compute_word_group_prefix_buf_masks()
        };
        self.word_group_sparse_prefix_entries =
            Self::compute_sparse_entry_prefix(&self.word_group_sparse_masks);
        self.quad_group_sparse_masks = quad_group_sparse_masks;
        self.byte_group_sparse_masks = byte_group_sparse_masks;
        let mask_words = self.body_mask_len();
        let (quad_group_dense_masks, byte_group_dense_masks) =
            if rayon::current_num_threads() == 1 {
                (
                    Self::compute_heavy_group_dense_masks(
                        &self.quad_group_sparse_masks,
                        mask_words,
                    ),
                    Self::compute_heavy_group_dense_masks(
                        &self.byte_group_sparse_masks,
                        mask_words,
                    ),
                )
            } else {
                rayon::join(
                    || {
                        Self::compute_heavy_group_dense_masks(
                            &self.quad_group_sparse_masks,
                            mask_words,
                        )
                    },
                    || {
                        Self::compute_heavy_group_dense_masks(
                            &self.byte_group_sparse_masks,
                            mask_words,
                        )
                    },
                )
            };
        self.quad_group_dense_masks = quad_group_dense_masks;
        self.byte_group_dense_masks = byte_group_dense_masks;
        self.word_group_sparse_total_entries = word_group_sparse_total_entries;
        self.word_group_sparse_max_entries = word_group_sparse_max_entries;
        let block_ms = block_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_started_at = profile.then(std::time::Instant::now);
        let build_sliding = |len: usize| {
            let started = profile.then(std::time::Instant::now);
            let result = if skip_load_dense_group_caches || skip_load_sliding_dense_caches {
                DenseBufMaskRows::default()
            } else {
                self.compute_sliding_word_group_dense_masks(len)
            };
            let ms = started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            (result, ms)
        };
        let (
            ((pair_word_group_buf_masks, pair_ms), (quad_word_group_buf_masks, quad_ms)),
            (
                (super_word_group_buf_masks, super_ms),
                ((mega_word_group_buf_masks, mega_ms), (giga_word_group_buf_masks, giga_ms)),
            ),
        ) = if rayon::current_num_threads() == 1 {
            (
                (build_sliding(2), build_sliding(4)),
                (build_sliding(8), (build_sliding(16), build_sliding(32))),
            )
        } else {
            rayon::join(
                || rayon::join(|| build_sliding(2), || build_sliding(4)),
                || {
                    rayon::join(
                        || build_sliding(8),
                        || rayon::join(|| build_sliding(16), || build_sliding(32)),
                    )
                },
            )
        };
        self.pair_word_group_buf_masks = pair_word_group_buf_masks;
        self.quad_word_group_buf_masks = quad_word_group_buf_masks;
        self.super_word_group_buf_masks = super_word_group_buf_masks;
        self.mega_word_group_buf_masks = mega_word_group_buf_masks;
        self.giga_word_group_buf_masks = giga_word_group_buf_masks;
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.all_tokens_buf_mask = self.compute_all_tokens_buf_mask();
        let all_tokens_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.heavy_token_dense_masks = self.compute_heavy_token_dense_masks();
        let heavy_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        let flat_ready = self.internal_token_buf_offsets.len()
            == self.internal_token_count().saturating_add(1)
            && self
                .internal_token_buf_offsets
                .last()
                .is_some_and(|&end| end as usize == self.internal_token_buf_flat_len());
        if !flat_ready {
            let (flat, offsets) = Self::compute_flat_buf_masks(&self.internal_token_buf_masks);
            self.internal_token_buf_flat = flat;
            self.backed_internal_token_buf_flat = None;
            self.internal_token_buf_offsets = offsets;
        }
        let flat_ms = if flat_ready {
            0.0
        } else {
            derived_piece_started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
        };
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        self.total_internal_buf_cost = Self::compute_total_internal_buf_cost(
            &self.internal_token_buf_offsets,
            &self.heavy_token_dense_masks,
            self.body_mask_len(),
        );

        // Precompute heavy token stats for fast path decision in convert.
        let buf_len = self.body_mask_len();
        let n_internal = if self.internal_token_buf_offsets.len() > 1 {
            self.internal_token_buf_offsets.len() - 1
        } else {
            0
        };
        self.heavy_token_indices = self.heavy_token_dense_masks.iter().enumerate().filter_map(|(i, m)| if m.is_some() { Some(i) } else { None }).collect();
        self.heavy_total_cost = self.heavy_token_indices.len() * buf_len;
        self.internal_token_buf_op_costs = Self::compute_internal_token_buf_op_costs(
            &self.internal_token_buf_offsets,
            &self.heavy_token_dense_masks,
            buf_len,
        );
        self.word_group_buf_op_costs =
            Self::compute_word_group_buf_op_costs(&self.internal_token_buf_op_costs);
        let costs_ms = derived_piece_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let n_light = n_internal.saturating_sub(self.heavy_token_indices.len());
        let light_total = self.total_internal_buf_cost.saturating_sub(self.heavy_total_cost);
        self.light_avg_cost_x256 = if n_light > 0 { (light_total * 256) / n_light } else { 0 };
        let derived_ms = derived_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        TokenMaskCacheBuildProfile {
            word_block_ms,
            quad_block_ms,
            byte_block_ms,
            block_ms,
            pair_ms,
            quad_ms,
            super_ms,
            mega_ms,
            giga_ms,
            all_tokens_ms,
            heavy_ms,
            flat_ms,
            costs_ms,
            derived_ms,
        }
    }

    /// Build the expensive caches whose source of truth is the final parser
    /// DWA. Composition can do this at the final parser-union boundary; later
    /// generic finalization then reuses them instead of rescanning the same
    /// completed automaton.
    pub(crate) fn prebuild_parser_runtime_caches(&mut self) {
        debug_assert!(
            self.token_mask_caches_ready(),
            "parser runtime cache prebuild requires internal-token output masks",
        );
        self.final_mask_mapping = FinalMaskMapping::default();
        let (fast_transitions, (prebuilt_sparse, dense_words, dense_masks)) = rayon::join(
            || self.compute_fast_transitions(),
            || {
                let inventory = self.weight_token_set_inventory();
                let prebuilt_sparse = self.plan_final_token_sets(
                    &inventory.final_sets,
                );
                let (dense_words, dense_masks) =
                    self.compute_dense_token_masks_excluding_range_final(
                        &prebuilt_sparse.eligible,
                        inventory,
                    );
                (prebuilt_sparse, dense_words, dense_masks)
            },
        );
        self.internal_token_dense_words = dense_words;
        self.weight_token_dense_masks = dense_masks;
        self.packed_dwa_token_dense_masks = self.compute_packed_dwa_dense_token_masks();
        self.dwa_fast_transitions = fast_transitions;
        let (weight_token_buf_masks, weight_token_sparse_buf_masks, range_final_token_sets) =
            self.compute_final_output_mask_caches(prebuilt_sparse);
        self.weight_token_buf_masks = weight_token_buf_masks;
        self.weight_token_sparse_buf_masks = weight_token_sparse_buf_masks;
        self.range_final_token_sets = range_final_token_sets;
        self.parser_runtime_caches_prebuilt = true;
    }

    pub(crate) fn rebuild_runtime_caches_impl(
        &mut self,
        preserve_packed_dwa_dense_masks: bool,
    ) {
        let mut packed_weight_token_sets =
            crate::automata::weighted::dwa::take_packed_decode_token_set_inventory();
        self.tokenizer_has_epsilon_transitions = self.tokenizer.has_epsilon_transitions();
        self.table.rebuild_unconditional_advance_rows();
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let total_started_at = profile.then(std::time::Instant::now);
        let terminal_live_started_at = profile.then(std::time::Instant::now);
        if self.terminal_live_states.len() != self.tokenizer.num_terminals() as usize {
            self.terminal_live_states = self.compute_terminal_live_states();
        }
        let terminal_live_ms = terminal_live_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let prepare_composition_cache = std::env::var_os(
            "GLRMASK_EXPERIMENT_COMPONENT_ONLY_BUILD_DELTA_METADATA",
        )
        .is_some()
            || std::env::var_os("GLRMASK_PREPARE_COMPOSITION_CACHE").is_some();
        if prepare_composition_cache {
            let reset_started_at = profile.then(std::time::Instant::now);
            self.ensure_composition_reset_tokens_by_terminal();
            if profile {
                let token_pairs = self
                    .composition_reset_tokens_by_terminal
                    .iter()
                    .map(Vec::len)
                    .sum::<usize>();
                eprintln!(
                    "[glrmask/profile][composition_reset_tokens] terminals={} token_pairs={} ms={:.3}",
                    self.composition_reset_tokens_by_terminal
                        .iter()
                        .filter(|row| !row.is_empty())
                        .count(),
                    token_pairs,
                    reset_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                );
            }
        }
        let scoped_ignore_started_at = profile.then(std::time::Instant::now);
        self.rebuild_scoped_ignore_runtime_tokens();
        let scoped_ignore_ms = scoped_ignore_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile && !self.scoped_ignore_only_tokens.is_empty() {
            eprintln!(
                "[glrmask/profile][scoped_ignore_only_tokens] terminals={} tokens={} fusions={} ms={:.3}",
                self.scoped_ignore_only_tokens.len(),
                self.scoped_ignore_only_tokens
                    .iter()
                    .map(|(_, tokens)| tokens.len())
                    .sum::<usize>(),
                self.scoped_ignore_prefix_fusions
                    .iter()
                    .map(|(_, pairs)| pairs.len())
                    .sum::<usize>(),
                scoped_ignore_ms,
            );
        }
        if self.uses_sparse_direct_regular_runtime() {
            let support = self
                .direct_regular_automaton
                .as_ref()
                .map_or_else(DirectRegularTerminalSupport::default, |automaton| {
                    DirectRegularTerminalSupport::build(
                        automaton,
                        self.table.num_terminals as usize,
                    )
                });
            self.dynamic_mask_vocab
                .set_direct_regular_terminal_support(support);
        }

        let wide_frontier_started_at = profile.then(std::time::Instant::now);
        self.direct_regular_wide_frontier_acceptance =
            self.compute_direct_regular_wide_frontier_acceptance();
        self.direct_regular_parser_state_acceptance =
            self.compute_direct_regular_parser_state_acceptance();
        let wide_frontier_ms = wide_frontier_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile && !self.direct_regular_wide_frontier_acceptance.is_empty() {
            eprintln!(
                "[glrmask/profile][wide_frontier_acceptance] summaries={} max_states={} ms={:.3}",
                self.direct_regular_wide_frontier_acceptance.len(),
                self.direct_regular_wide_frontier_acceptance
                    .iter()
                    .map(|summary| summary.state_count)
                    .max()
                    .unwrap_or(0),
                wide_frontier_ms,
            );
        }

        if self.tokenizer.has_virtual_residual_runtime()
            && self.dynamic_mask_vocab.mask_projection_tokenizer().is_none()
        {
            let max_token_len = self.max_token_byte_len();
            let projection_started_at = profile.then(std::time::Instant::now);
            if let Some((mask_tokenizer, projections)) =
                self.tokenizer.virtual_residuals_mask_tokenizer(max_token_len)
            {
                if profile {
                    eprintln!(
                        "[glrmask/profile][static_runtime_finalize] mask_lexer=virtual_residual components={} mask_states={} horizon={} ms={:.3}",
                        projections.len(),
                        mask_tokenizer.num_states(),
                        max_token_len,
                        projection_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                    );
                }
                self.dynamic_mask_vocab
                    .set_virtual_residuals_mask_projection(mask_tokenizer, projections);
            }
        }
        // Ordinary static masking executes through the parser DWA, not the
        // dynamic/full-walk lexer coordinate. Do not speculatively determinize
        // every epsilon-NFA start here: the result is derived acceleration data,
        // is not serialized, and the exact epsilon-NFA path remains available
        // anywhere a dynamic full walk is explicitly requested. DynamicConstraint
        // prepares its own execution coordinate in rebuild_dynamic_runtime_caches().
        let guarded_shift_started_at = profile.then(std::time::Instant::now);
        if self.table.guarded_shift_index.len() != self.table.num_states as usize {
            if self.table.num_rules == 0 {
                self.table.guarded_shift_index =
                    vec![FxHashMap::default(); self.table.num_states as usize];
            } else {
                self.table.rebuild_guarded_shift_index();
            }
        }
        let guarded_index_ms = guarded_shift_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let state_relation_started_at = profile.then(std::time::Instant::now);
        let state_count = if self.tokenizer.has_virtual_residual_runtime() {
            self.dynamic_mask_vocab
                .mask_projection_tokenizer()
                .map_or(self.tokenizer.num_states(), |tokenizer| tokenizer.num_states())
        } else {
            self.tokenizer.num_states()
        } as usize;
        let singleton_state_relation_ready = self.state_internal_tsid_offsets.as_slice()
            == [u32::MAX]
            && self.state_internal_tsids.is_empty()
            && self.state_to_internal_tsid.len() == state_count;
        let deferred_singleton_state_relation_ready = self.runtime_source_state_offset.is_none()
            && self.state_internal_tsid_offsets.is_empty()
            && self.state_internal_tsids.is_empty()
            && self.state_to_internal_tsid.len() == state_count
            && self.state_to_internal_tsid.iter().all(|&tsid| tsid != u32::MAX);
        let state_relation_ready = singleton_state_relation_ready
            || deferred_singleton_state_relation_ready
            || (self.state_internal_tsid_offsets.len() == state_count + 1
                && self
                    .state_internal_tsid_offsets
                    .last()
                    .is_some_and(|&end| end as usize == self.state_internal_tsids.len()));
        if !state_relation_ready {
            self.rebuild_state_internal_tsid_relation();
        }
        self.rebuild_runtime_product_state_lookup();
        let state_relation_ms = state_relation_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let fast_template_started_at = profile.then(std::time::Instant::now);
        let fast_template_dfas_by_terminal = self.compute_fast_template_dfas();
        let fast_template_ms = fast_template_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let guarded_shift_ms = guarded_shift_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile {
            eprintln!(
                "[glrmask/profile][runtime_finalize_guarded_split] guarded_index_ms={guarded_index_ms:.3} state_relation_ms={state_relation_ms:.3} fast_template_ms={fast_template_ms:.3} guarded_cells={}",
                self.table
                    .guarded_shift_index
                    .iter()
                    .map(|row| row.len())
                    .sum::<usize>(),
            );
        }
        // This mapping is a derived cache. Reset it before scheduling the
        // independent cache builders so the direct sparse weight-cache branch
        // observes the same default mapping as the historical serial path.
        self.final_mask_mapping = FinalMaskMapping::default();
        // Static cache finalization never constructs the direct-dynamic
        // vocabulary. Deferred possible-match fallback materializes it lazily
        // on the first state that actually requires a dynamic mask.
        let dynamic_vocab_reused = false;
        let dynamic_vocab_ms = 0.0;
        let token_mask_caches_prebuilt = self.token_mask_caches_ready()
            || std::env::var_os("GLRMASK_SKIP_TOKEN_MASK_REBUILD_FOR_PROFILE").is_some();
        // v13+ cache artifacts can persist the portable per-internal-token
        // output fragments without persisting every derived aggregate cache.
        // Reuse those fragments independently; the remaining derived caches
        // are still rebuilt unless their own readiness invariant is satisfied.
        let internal_token_count = self.internal_token_count();
        let flat_internal_token_buf_masks_prebuilt =
            self.internal_token_buf_offsets.len() == internal_token_count.saturating_add(1)
                && self
                    .internal_token_buf_offsets
                    .last()
                    .is_some_and(|&end| end as usize == self.internal_token_buf_flat_len());
        let mut prebuilt_internal_token_buf_masks =
            (self.internal_token_buf_masks.len() == internal_token_count)
                .then(|| std::mem::take(&mut self.internal_token_buf_masks));
        let parser_runtime_caches_prebuilt = self.parser_runtime_caches_prebuilt;
        let mut prebuilt_parser_dense_masks = parser_runtime_caches_prebuilt.then(|| {
            (
                self.internal_token_dense_words,
                std::mem::take(&mut self.weight_token_dense_masks),
            )
        });
        let mut prebuilt_parser_fast_transitions = parser_runtime_caches_prebuilt
            .then(|| std::mem::take(&mut self.dwa_fast_transitions));
        let mut prebuilt_parser_weight_buf_caches = parser_runtime_caches_prebuilt.then(|| {
            (
                std::mem::take(&mut self.weight_token_buf_masks),
                std::mem::take(&mut self.weight_token_sparse_buf_masks),
                std::mem::take(&mut self.range_final_token_sets),
            )
        });
        let primary_started_at = profile.then(std::time::Instant::now);
        let mut prebuilt_tokenizer_fast_transitions =
            (self.tokenizer_fast_transitions.len() == self.tokenizer.num_states() as usize)
                .then(|| std::mem::take(&mut self.tokenizer_fast_transitions));
        let (
            internal_token_buf_masks,
            internal_token_buf_masks_ms,
            tokenizer_fast_transitions,
            tokenizer_fast_transitions_ms,
            (dense_mask_words, dense_masks),
            dense_token_masks_ms,
            fast_transitions,
            dwa_fast_transitions_ms,
            prebuilt_weight_caches,
            prebuilt_weight_sparse_ms,
        ) = if rayon::current_num_threads() == 1 {
            let started = profile.then(std::time::Instant::now);
            let reused_internal_token_buf_masks = prebuilt_internal_token_buf_masks.is_some()
                || flat_internal_token_buf_masks_prebuilt;
            let internal_token_buf_masks = prebuilt_internal_token_buf_masks.take().unwrap_or_else(|| {
                if flat_internal_token_buf_masks_prebuilt {
                    Vec::new()
                } else {
                    self.compute_buf_masks()
                }
            });
            let internal_token_buf_masks_ms = if reused_internal_token_buf_masks {
                0.0
            } else {
                started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
            };
            let started = profile.then(std::time::Instant::now);
            let parser_dense_prebuilt = prebuilt_parser_dense_masks.take();
            let (dense_masks, prebuilt_weight_caches, prebuilt_weight_sparse_ms, dense_token_masks_ms) =
                if let Some(dense_masks) = parser_dense_prebuilt {
                    (dense_masks, FinalTokenSetPlan::default(), 0.0, 0.0)
                } else {
                    let weight_token_sets = self
                        .weight_token_set_inventory_with_packed(packed_weight_token_sets.take());
                    let prebuilt_weight_caches = self.plan_final_token_sets(
                        &weight_token_sets.final_sets,
                    );
                    let prebuilt_weight_sparse_ms = started
                        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                    let dense_started = profile.then(std::time::Instant::now);
                    let dense_masks = self.compute_dense_token_masks_excluding_range_final(
                        &prebuilt_weight_caches.eligible,
                        weight_token_sets,
                    );
                    let dense_token_masks_ms = dense_started
                        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                    (
                        dense_masks,
                        prebuilt_weight_caches,
                        prebuilt_weight_sparse_ms,
                        dense_token_masks_ms,
                    )
                };
            let started = profile.then(std::time::Instant::now);
            let reused_tokenizer_fast_transitions = prebuilt_tokenizer_fast_transitions.is_some();
            let tokenizer_fast_transitions = prebuilt_tokenizer_fast_transitions
                .take()
                .unwrap_or_else(|| self.compute_tokenizer_fast_transitions());
            let tokenizer_fast_transitions_ms = if reused_tokenizer_fast_transitions {
                0.0
            } else {
                started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
            };
            let started = profile.then(std::time::Instant::now);
            let reused_parser_fast_transitions = prebuilt_parser_fast_transitions.is_some();
            let fast_transitions = prebuilt_parser_fast_transitions
                .take()
                .unwrap_or_else(|| self.compute_fast_transitions());
            let dwa_fast_transitions_ms = if reused_parser_fast_transitions {
                0.0
            } else {
                started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
            };
            (
                internal_token_buf_masks,
                internal_token_buf_masks_ms,
                tokenizer_fast_transitions,
                tokenizer_fast_transitions_ms,
                dense_masks,
                dense_token_masks_ms,
                fast_transitions,
                dwa_fast_transitions_ms,
                prebuilt_weight_caches,
                prebuilt_weight_sparse_ms,
            )
        } else {
            let (
                ((tokenizer_fast_transitions, tokenizer_fast_transitions_ms), (fast_transitions, dwa_fast_transitions_ms)),
                (((internal_token_buf_masks, internal_token_buf_masks_ms), ((dense_mask_words, dense_masks), dense_token_masks_ms)), (prebuilt_weight_caches, prebuilt_weight_sparse_ms)),
            ) = rayon::join(
                || {
                    let build_tokenizer_fast_transitions = || {
                        let started = profile.then(std::time::Instant::now);
                        if let Some(prebuilt) = prebuilt_tokenizer_fast_transitions.take() {
                            return (prebuilt, 0.0);
                        }
                        let result = self.compute_tokenizer_fast_transitions();
                        let ms = started
                            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                        (result, ms)
                    };
                    let build_dwa_fast_transitions = || {
                        let started = profile.then(std::time::Instant::now);
                        let reused = prebuilt_parser_fast_transitions.is_some();
                        let result = prebuilt_parser_fast_transitions
                            .take()
                            .unwrap_or_else(|| self.compute_fast_transitions());
                        let ms = if reused {
                            0.0
                        } else {
                            started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
                        };
                        (result, ms)
                    };
                    rayon::join(build_tokenizer_fast_transitions, build_dwa_fast_transitions)
                },
                || {
                    let started = profile.then(std::time::Instant::now);
                    let reused_internal_token_buf_masks = prebuilt_internal_token_buf_masks.is_some()
                        || flat_internal_token_buf_masks_prebuilt;
                    let internal_token_buf_masks = prebuilt_internal_token_buf_masks.take().unwrap_or_else(|| {
                        if flat_internal_token_buf_masks_prebuilt {
                            Vec::new()
                        } else {
                            self.compute_buf_masks()
                        }
                    });
                    let internal_token_buf_masks_ms = if reused_internal_token_buf_masks {
                        0.0
                    } else {
                        started.map_or(0.0, |started| {
                            started.elapsed().as_secs_f64() * 1000.0
                        })
                    };
                    let started = profile.then(std::time::Instant::now);
                    let (dense_masks, prebuilt_weight_caches, prebuilt_weight_sparse_ms, dense_token_masks_ms) =
                        if let Some(dense_masks) = prebuilt_parser_dense_masks.take() {
                            (dense_masks, FinalTokenSetPlan::default(), 0.0, 0.0)
                        } else {
                            let weight_token_sets = self
                                .weight_token_set_inventory_with_packed(packed_weight_token_sets.take());
                            let prebuilt_weight_caches = self.plan_final_token_sets(
                                &weight_token_sets.final_sets,
                            );
                            let prebuilt_weight_sparse_ms = started
                                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                            let dense_started = profile.then(std::time::Instant::now);
                            let dense_masks = self.compute_dense_token_masks_excluding_range_final(
                                &prebuilt_weight_caches.eligible,
                                weight_token_sets,
                            );
                            let dense_token_masks_ms = dense_started
                                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                            (
                                dense_masks,
                                prebuilt_weight_caches,
                                prebuilt_weight_sparse_ms,
                                dense_token_masks_ms,
                            )
                        };
                    (
                        ((internal_token_buf_masks, internal_token_buf_masks_ms), (dense_masks, dense_token_masks_ms)),
                        (prebuilt_weight_caches, prebuilt_weight_sparse_ms),
                    )
                },
            );
            (
                internal_token_buf_masks,
                internal_token_buf_masks_ms,
                tokenizer_fast_transitions,
                tokenizer_fast_transitions_ms,
                (dense_mask_words, dense_masks),
                dense_token_masks_ms,
                fast_transitions,
                dwa_fast_transitions_ms,
                prebuilt_weight_caches,
                prebuilt_weight_sparse_ms,
            )
        };
        let primary_ms = primary_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        self.internal_token_buf_masks = internal_token_buf_masks;
        let token_mask_profile = if token_mask_caches_prebuilt {
            TokenMaskCacheBuildProfile::default()
        } else {
            self.rebuild_token_mask_derived_caches(profile)
        };
        let TokenMaskCacheBuildProfile {
            word_block_ms,
            quad_block_ms,
            byte_block_ms,
            block_ms,
            pair_ms,
            quad_ms,
            super_ms,
            mega_ms,
            giga_ms,
            all_tokens_ms,
            heavy_ms,
            flat_ms,
            costs_ms,
            derived_ms,
        } = token_mask_profile;

        self.token_bytes_dense = Vec::new();
        self.internal_token_dense_words = dense_mask_words;
        self.weight_token_dense_masks = dense_masks;
        if !preserve_packed_dwa_dense_masks {
            self.packed_dwa_token_dense_masks = self.compute_packed_dwa_dense_token_masks();
        }
        let full_dense = Self::dense_words_from_internal_set_with_words(
            &self.internal_token_universe(),
            self.internal_token_dense_words,
        );
        let tsid_count = self.internal_tsid_count().max(1);
        let wide_parts = self
            .direct_regular_wide_frontier_acceptance
            .iter()
            .map(|summary| Arc::clone(&summary.acceptance_parts))
            .collect::<Vec<_>>();
        let parser_parts = self
            .direct_regular_parser_state_acceptance
            .iter()
            .map(|summary| Arc::clone(&summary.acceptance_parts))
            .collect::<Vec<_>>();
        let dense_cache = &self.weight_token_dense_masks;
        let direct_wide_dense_started_at = profile.then(std::time::Instant::now);
        let direct_parser_dense_started_at = profile.then(std::time::Instant::now);
        let build_wide = || {
            Self::materialize_direct_regular_acceptance_rows(
                &wide_parts,
                self.internal_token_dense_words,
                tsid_count,
                &full_dense,
                dense_cache,
            )
        };
        let build_parser = || {
            Self::materialize_direct_regular_acceptance_rows(
                &parser_parts,
                self.internal_token_dense_words,
                tsid_count,
                &full_dense,
                dense_cache,
            )
        };
        let (wide_dense, parser_dense) = if wide_parts.is_empty() && parser_parts.is_empty() {
            (Vec::new(), Vec::new())
        } else if rayon::current_num_threads() == 1 {
            (build_wide(), build_parser())
        } else {
            rayon::join(build_wide, build_parser)
        };
        let direct_wide_dense_ms = direct_wide_dense_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let direct_parser_dense_ms = direct_parser_dense_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        for (summary, dense) in self
            .direct_regular_wide_frontier_acceptance
            .iter_mut()
            .zip(wide_dense)
        {
            summary.dense_by_tsid = dense;
        }
        for (summary, dense) in self
            .direct_regular_parser_state_acceptance
            .iter_mut()
            .zip(parser_dense)
        {
            summary.dense_by_tsid = dense;
        }
        let derived_piece_started_at = profile.then(std::time::Instant::now);
        let (
            weight_token_buf_masks,
            weight_token_sparse_buf_masks,
            range_final_token_sets,
        ) = prebuilt_parser_weight_buf_caches
            .take()
            .unwrap_or_else(|| {
                self.compute_final_output_mask_caches(
                    prebuilt_weight_caches,
                )
            });
        self.weight_token_buf_masks = weight_token_buf_masks;
        let weight_buf_ms = if parser_runtime_caches_prebuilt {
            0.0
        } else {
            derived_piece_started_at
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
        };
        self.weight_token_sparse_buf_masks = weight_token_sparse_buf_masks;
        self.range_final_token_sets = range_final_token_sets;
        let weight_sparse_ms = 0.0;
        self.dwa_fast_transitions = fast_transitions;
        self.parser_runtime_caches_prebuilt = true;
        let indexed_dag_dense_started_at = profile.then(std::time::Instant::now);
        let (indexed_dag_dense_transitions, indexed_dag_dense_finals) =
            self.compute_indexed_dag_dense_tables();
        self.indexed_dag_dense_transitions = indexed_dag_dense_transitions;
        self.indexed_dag_dense_finals = indexed_dag_dense_finals;
        if let Some(started) = indexed_dag_dense_started_at {
            let transitions = self
                .parser_dwa
                .states()
                .iter()
                .map(|state| state.transitions.len())
                .sum::<usize>();
            eprintln!(
                "[glrmask/profile][indexed_dag_dense_transitions] ms={:.3} states={} transitions={} internal_tsids={}",
                started.elapsed().as_secs_f64() * 1000.0,
                self.parser_dwa.states().len(),
                transitions,
                self.internal_tsid_count(),
            );
        }
        self.fast_template_dfas_by_terminal = fast_template_dfas_by_terminal;
        self.tokenizer_fast_transitions = tokenizer_fast_transitions;
        let seed_started_at = profile.then(std::time::Instant::now);
        let seed_dense_prebuilt = self.seed_universe_dense.len() == self.internal_token_dense_words
            && self
                .seed_terminal_dense
                .values()
                .all(|mask| mask.len() == self.internal_token_dense_words);
        if !seed_dense_prebuilt {
            self.build_seed_dense_masks();
        }
        let seed_ms = if seed_dense_prebuilt {
            0.0
        } else {
            seed_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
        };
        // The bounded tokenizer scanner used by every allocation-free commit
        // reads this constraint-level cache. Materialize it during compile/load
        // finalization rather than charging its one-time allocations to the
        // first decoding commit.
        let tokenizer_closures_started_at = profile.then(std::time::Instant::now);
        if !self.tokenizer.has_packed_runtime_metadata() {
            let tokenizer_closures = self.tokenizer.all_singleton_epsilon_closures();
            if tokenizer_closures
                .get(self.tokenizer.initial_state() as usize)
                .is_some_and(|closure| closure.len() > 64)
            {
                let _ = self.tokenizer.initial_byte_frontiers();
            }
        }
        let tokenizer_closures_ms = tokenizer_closures_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let initial_commit_prime_started_at = profile.then(std::time::Instant::now);
        // Freshly compiled constraints may still choose to pay this one-time
        // warm-up before decoding starts. A current-format disk load already
        // has an explicit latency target, and warming an otherwise lazy cache
        // is not part of reconstructing its semantics. Do not charge it to
        // load; the first commit remains exact and will initialize lazily if
        // needed.
        if self.packed_parser_dwa.is_none() {
            self.prime_initial_commit_hot_path();
        }
        let initial_commit_prime_ms = initial_commit_prime_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][runtime_finalize_derived] pair_ms={:.3} quad_ms={:.3} super_ms={:.3} mega_ms={:.3} giga_ms={:.3} all_tokens_ms={:.3} heavy_ms={:.3} flat_ms={:.3} costs_ms={:.3} direct_wide_dense_ms={:.3} direct_parser_dense_ms={:.3} prebuilt_weight_sparse_ms={:.3} weight_buf_ms={:.3} weight_sparse_ms={:.3} final_weight_sets={} final_weight_sparse_sets={} direct_sparse_weight_sets={}",
                pair_ms,
                quad_ms,
                super_ms,
                mega_ms,
                giga_ms,
                all_tokens_ms,
                heavy_ms,
                flat_ms,
                costs_ms,
                direct_wide_dense_ms,
                direct_parser_dense_ms,
                prebuilt_weight_sparse_ms,
                weight_buf_ms,
                weight_sparse_ms,
                self.weight_token_buf_masks.len(),
                self.weight_token_sparse_buf_masks.len(),
                self.range_final_token_sets.len(),
            );
            eprintln!(
                "[glrmask/profile][runtime_finalize] terminal_live_ms={:.3} guarded_shift_ms={:.3} dynamic_mask_vocab_ms={:.3} dynamic_mask_vocab_reused={} internal_token_buf_masks_ms={:.3} tokenizer_fast_transitions_ms={:.3} dense_token_masks_ms={:.3} dwa_fast_transitions_ms={:.3} primary_ms={:.3} word_block_masks_ms={:.3} quad_word_block_masks_ms={:.3} byte_block_masks_ms={:.3} block_masks_ms={:.3} derived_masks_ms={:.3} seed_dense_ms={:.3} tokenizer_closures_ms={:.3} initial_commit_prime_ms={:.3} total_ms={:.3}",
                terminal_live_ms,
                guarded_shift_ms,
                dynamic_vocab_ms,
                dynamic_vocab_reused,
                internal_token_buf_masks_ms,
                tokenizer_fast_transitions_ms,
                dense_token_masks_ms,
                dwa_fast_transitions_ms,
                primary_ms,
                word_block_ms,
                quad_block_ms,
                byte_block_ms,
                block_ms,
                derived_ms,
                seed_ms,
                tokenizer_closures_ms,
                initial_commit_prime_ms,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
}
