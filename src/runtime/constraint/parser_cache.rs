//! Fast parser/tokenizer tables and weighted-token mask preparation.

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::runtime::artifact::Constraint;
use crate::runtime::artifact::DenseWeightMaskCache;
use crate::runtime::artifact::DenseWords;
use crate::runtime::artifact::DirectSparseWeightTokenSetCache;
use crate::runtime::artifact::FastCommitTemplateDfas;
use crate::runtime::artifact::FastDwaTransitionRow;
use crate::runtime::artifact::FastDwaTransitions;
use crate::runtime::artifact::FastTemplateDfasByTerminal;
use crate::runtime::artifact::FastTokenizerTransitions;
use crate::runtime::artifact::IndexedDagDenseMask;
use crate::runtime::artifact::IndexedDagDenseTransition;
use crate::runtime::artifact::IndexedDagDenseTransitionMasks;
use crate::runtime::artifact::IndexedDagDenseTransitionRow;
use crate::runtime::artifact::IndexedDagDenseTransitions;
use crate::runtime::artifact::PackedDwaDenseWeightMaskCache;
use crate::runtime::artifact::SeedTerminalDenseMasks;
use crate::runtime::artifact::empty_dense_words;
use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use super::RuntimeTokenSetRef;
use super::mask_cache::WeightTokenSetInventory;

impl Constraint {


    pub(super) fn compute_terminal_live_states(&self) -> Vec<Vec<u32>> {
        let terminal_count = self.tokenizer.num_terminals() as usize;
        if terminal_count == 0 {
            return Vec::new();
        }
        let closures = self.tokenizer.all_singleton_epsilon_closures();
        let state_count = self.tokenizer.num_states() as usize;
        let build_state = |mut rows: Vec<Vec<u32>>, state: usize| {
            // Rows are sorted and deduplicated once after the parallel transpose.
            // Stream observations directly instead of sorting/deduplicating a
            // temporary terminal list for every runtime tokenizer state.
            for &closure_state in closures[state].iter() {
                for terminal in self.tokenizer.matched_terminals_iter(closure_state) {
                    if let Some(row) = rows.get_mut(terminal as usize) {
                        row.push(state as u32);
                    }
                }
                for terminal in self.tokenizer.possible_future_terminals_iter(closure_state) {
                    if let Some(row) = rows.get_mut(terminal as usize) {
                        row.push(state as u32);
                    }
                }
            }
            rows
        };
        let empty = || (0..terminal_count).map(|_| Vec::<u32>::new()).collect::<Vec<_>>();
        let mut rows = if rayon::current_num_threads() == 1 || state_count < 4096 {
            (0..state_count).fold(empty(), build_state)
        } else {
            (0..state_count)
                .into_par_iter()
                .fold(empty, build_state)
                .reduce(empty, |mut left, mut right| {
                    for (left_row, right_row) in left.iter_mut().zip(&mut right) {
                        left_row.append(right_row);
                    }
                    left
                })
        };
        for row in &mut rows {
            row.sort_unstable();
            row.dedup();
        }
        rows
    }

    pub(super) fn compute_tokenizer_fast_transitions(&self) -> FastTokenizerTransitions {
        Self::compute_tokenizer_fast_transitions_for(&self.tokenizer)
    }

    pub(super) fn compute_tokenizer_fast_transitions_for(tokenizer: &Tokenizer) -> FastTokenizerTransitions {
        let num_states = tokenizer.num_states();
        // A tokenizer with lazy virtual runtimes can transition from one of
        // the physical rows below into a runtime state whose id lies outside
        // `0..num_states`. The compact physical-only slabs cannot represent
        // that target (and cannot service its subsequent row), so route this
        // family through the generic tokenizer transition API instead.
        if tokenizer.has_any_virtual_runtime() {
            return FastTokenizerTransitions::Fallback(num_states as usize);
        }
        // Current backed fast-wire loads already retain an allocation-light exact packed
        // transition table. Rebuilding a second state x 256 Flat16 slab here is
        // duplicate load-time work; ordinary commit/scan can call through to the
        // packed tokenizer directly. The strict dynamic mask walker owns a
        // separate mask-projection transition table, so this does not remove its
        // dense full-walk acceleration.
        if tokenizer.has_backed_runtime_transitions() {
            return FastTokenizerTransitions::Fallback(num_states as usize);
        }
        // Fresh compiler tokenizers do not yet own packed runtime rows. Give small
        // exact tokenizers one compact direct transition cell per consumed byte.
        if num_states <= 8_192
            && let Some(flat16) = FastTokenizerTransitions::flat16_transitions_only_for(tokenizer)
        {
            return flat16;
        }
        if tokenizer.has_packed_runtime_transitions() {
            return FastTokenizerTransitions::Fallback(num_states as usize);
        }
        let has_compressed =
            (0..num_states).any(|state| tokenizer.has_compressed_transition_state(state));
        if !has_compressed {
            let build = |state| tokenizer.transition_row(state);
            let rows = if rayon::current_num_threads() == 1 {
                (0..num_states).map(build).collect()
            } else {
                (0..num_states).into_par_iter().map(build).collect()
            };
            return FastTokenizerTransitions::Dense(rows);
        }

        let dense_states = (0..num_states)
            .filter(|&state| !tokenizer.has_compressed_transition_state(state))
            .collect::<Vec<_>>();
        let dense_rows = if rayon::current_num_threads() == 1 {
            dense_states
                .iter()
                .map(|&state| tokenizer.transition_row(state))
                .collect::<Vec<_>>()
        } else {
            dense_states
                .par_iter()
                .map(|&state| tokenizer.transition_row(state))
                .collect::<Vec<_>>()
        };
        let mut state_to_dense_row = vec![u32::MAX; num_states as usize];
        for (row, &state) in dense_states.iter().enumerate() {
            state_to_dense_row[state as usize] = row as u32;
        }
        FastTokenizerTransitions::Hybrid {
            state_to_dense_row,
            dense_rows,
        }
    }

    pub(super) fn compute_fast_template_dfas(&self) -> FastTemplateDfasByTerminal {
        self.template_dfas_by_terminal
            .iter()
            .map(|template| {
                template
                    .as_deref()
                    .map(FastCommitTemplateDfas::from_template)
                    .map(Arc::new)
            })
            .collect()
    }

    pub(super) fn compute_fast_transitions(&self) -> FastDwaTransitions {
        if self.packed_parser_dwa.is_some() {
            return FastDwaTransitions::default();
        }
        let build = |state: &crate::automata::weighted_u32::dwa::DWAState| {
            if state.transitions.is_packed() {
                return FastDwaTransitionRow::from_packed(state.transitions.clone());
            }
            let entries = state
                .transitions
                .entries()
                .map(|(label, target, weight)| (label, (target, weight.clone())));
            FastDwaTransitionRow::from_entries(entries)
        };
        if !self.parser_dwa.has_shared_transition_rows()
            || std::env::var_os("GLRMASK_EXPERIMENTAL_DISABLE_SHARED_FAST_DWA_ROWS").is_some()
        {
            let rows = if rayon::current_num_threads() == 1 {
                self.parser_dwa.states().iter().map(build).collect()
            } else {
                self.parser_dwa.states().par_iter().map(build).collect()
            };
            return FastDwaTransitions::direct(rows);
        }

        let mut row_by_ptr = FxHashMap::<usize, u32>::default();
        let mut representative_states = Vec::<usize>::new();
        let mut state_rows = Vec::<u32>::with_capacity(self.parser_dwa.states().len());
        for (state_index, state) in self.parser_dwa.states().iter().enumerate() {
            let key = state.transitions.ptr_key();
            let row = if let Some(&row) = row_by_ptr.get(&key) {
                row
            } else {
                let row = representative_states.len() as u32;
                row_by_ptr.insert(key, row);
                representative_states.push(state_index);
                row
            };
            state_rows.push(row);
        }
        let rows = if rayon::current_num_threads() == 1 {
            representative_states
                .iter()
                .map(|&state| build(&self.parser_dwa.states()[state]))
                .collect()
        } else {
            representative_states
                .par_iter()
                .map(|&state| build(&self.parser_dwa.states()[state]))
                .collect()
        };
        FastDwaTransitions::shared(rows, state_rows)
    }

    fn indexed_dag_dense_mask_for_tokens(
        &self,
        tokens: &Arc<RangeSetBlaze<u32>>,
    ) -> IndexedDagDenseMask {
        let token_key = Arc::as_ptr(tokens) as usize;
        let words = if let Some(dense) = self.weight_token_dense_masks.get(&token_key) {
            Arc::clone(dense)
        } else {
            let mut dense = vec![0u64; self.internal_token_dense_words];
            let max_token_exclusive = dense.len().saturating_mul(64);
            for range in tokens.ranges() {
                let lo = *range.start() as usize;
                if lo >= max_token_exclusive {
                    continue;
                }
                let hi = (*range.end() as usize).min(max_token_exclusive - 1);
                let word_lo = lo / 64;
                let word_hi = hi / 64;
                for word_index in word_lo..=word_hi {
                    let lo_bit = if word_index == word_lo { lo % 64 } else { 0 };
                    let hi_bit = if word_index == word_hi { hi % 64 } else { 63 };
                    let high_mask = if hi_bit == 63 {
                        !0u64
                    } else {
                        (1u64 << (hi_bit + 1)) - 1
                    };
                    let low_mask = if lo_bit == 0 {
                        0
                    } else {
                        (1u64 << lo_bit) - 1
                    };
                    dense[word_index] |= high_mask & !low_mask;
                }
            }
            dense.into()
        };
        let Some(start) = words.iter().position(|word| *word != 0) else {
            return IndexedDagDenseMask::Empty;
        };
        let end = words
            .iter()
            .rposition(|word| *word != 0)
            .expect("nonzero start implies nonzero end")
            + 1;
        IndexedDagDenseMask::Dense { words, start, end }
    }

    pub(super) fn compute_indexed_dag_dense_tables(
        &self,
    ) -> (
        IndexedDagDenseTransitions,
        Vec<IndexedDagDenseTransitionMasks>,
    ) {
        if self.packed_parser_dwa.is_some() {
            return (Vec::new(), Vec::new());
        }
        // Indexed-DAG masking is opt-in. Avoid duplicating the parser DWA into
        // dense runtime tables for every ordinary constraint. Unit tests keep
        // the tables available for forced exactness checks.
        if !cfg!(test) && !crate::runtime::mask::indexed_dag_mask_enabled() {
            return (Vec::new(), Vec::new());
        }
        // Narrow transition sets are intentionally absent from the general
        // dense-weight cache. Materialize every distinct token-set pointer at
        // most once here, then share the resulting Arc and span across all DWA
        // transitions and final weights that use it.
        let mut mask_by_token_set = FxHashMap::<usize, IndexedDagDenseMask>::default();
        for state in self.parser_dwa.states() {
            for (_, weight) in state.transitions.values() {
                if weight.is_full() {
                    continue;
                }
                for (_, tokens) in weight.raw_iter() {
                    let key = Arc::as_ptr(tokens) as usize;
                    if !mask_by_token_set.contains_key(&key) {
                        let mask = self.indexed_dag_dense_mask_for_tokens(tokens);
                        mask_by_token_set.insert(key, mask);
                    }
                }
            }
            if let Some(weight) = state.final_weight.as_ref()
                && !weight.is_full()
            {
                for (_, tokens) in weight.raw_iter() {
                    let key = Arc::as_ptr(tokens) as usize;
                    if !mask_by_token_set.contains_key(&key) {
                        let mask = self.indexed_dag_dense_mask_for_tokens(tokens);
                        mask_by_token_set.insert(key, mask);
                    }
                }
            }
        }
        let build = |state: &crate::automata::weighted_u32::dwa::DWAState| {
            IndexedDagDenseTransitionRow::from_entries(state.transitions.iter().map(
                |(&label, (target, weight))| {
                    let masks = if weight.is_full() {
                        IndexedDagDenseTransitionMasks::Full
                    } else {
                        IndexedDagDenseTransitionMasks::from_entries(
                            weight.raw_iter().map(|(tsid, tokens)| {
                                (
                                    tsid,
                                    mask_by_token_set[&(Arc::as_ptr(tokens) as usize)].clone(),
                                )
                            }),
                        )
                    };
                    (
                        label,
                        IndexedDagDenseTransition {
                            target: *target,
                            masks,
                        },
                    )
                },
            ))
        };
        let transitions = if rayon::current_num_threads() == 1 {
            self.parser_dwa.states().iter().map(build).collect()
        } else {
            self.parser_dwa.states().par_iter().map(build).collect()
        };
        let finals = self
            .parser_dwa
            .states()
            .iter()
            .map(|state| match state.final_weight.as_ref() {
                None => IndexedDagDenseTransitionMasks::from_entries(std::iter::empty()),
                Some(weight) if weight.is_full() => IndexedDagDenseTransitionMasks::Full,
                Some(weight) => IndexedDagDenseTransitionMasks::from_entries(
                    weight.raw_iter().map(|(tsid, tokens)| {
                        (
                            tsid,
                            mask_by_token_set[&(Arc::as_ptr(tokens) as usize)].clone(),
                        )
                    }),
                ),
            })
            .collect();
        (transitions, finals)
    }

    // For narrow token sets, RangeSetBlaze word-span intersection is less
    // work than scanning the full dense mask. Keep dense bitmaps only for wide
    // transition sets and for residual final sets that need contained-mask IO.
    const DENSE_WEIGHT_PRECOMPUTE_MIN_WORD_SPANS: usize = 16;
    // Packed token sets are decoded directly from the artifact, so eagerly
    // densifying every set that clears the compiler-side threshold would trade
    // a large amount of load work and memory for little runtime benefit. A
    // packed intersection scans one callback per covered word span, while its
    // dense equivalent scans `internal_token_dense_words` words. Require at
    // least four dense scans worth of packed span work before paying the load
    // cost. This is representation-level work accounting, independent of any
    // parser state or benchmark corpus.
    const PACKED_DWA_DENSE_MIN_SCAN_MULTIPLIER: usize = 4;

    fn token_set_dense_word_spans_at_least(
        tokens: &RangeSetBlaze<u32>,
        dense_word_count: usize,
        threshold: usize,
    ) -> bool {
        if dense_word_count == 0 || threshold == 0 {
            return threshold == 0;
        }
        let max_token = dense_word_count.saturating_mul(64).saturating_sub(1);
        let mut count = 0usize;
        for range in tokens.ranges() {
            let start = *range.start() as usize;
            if start > max_token {
                continue;
            }
            let end = (*range.end() as usize).min(max_token);
            count = count.saturating_add(end / 64 - start / 64 + 1);
            if count >= threshold {
                return true;
            }
        }
        false
    }

    fn compute_dense_token_masks(&self) -> (usize, DenseWeightMaskCache) {
        let inventory = self.weight_token_set_inventory();
        self.compute_dense_token_masks_excluding_direct_final(
            &DirectSparseWeightTokenSetCache::default(),
            inventory,
        )
    }

    pub(super) fn compute_dense_token_masks_excluding_direct_final(
        &self,
        direct_final_sets: &DirectSparseWeightTokenSetCache,
        inventory: WeightTokenSetInventory,
    ) -> (usize, DenseWeightMaskCache) {
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let internal_token_dense_words = self.internal_token_count().div_ceil(64);
        if internal_token_dense_words == 0 {
            return (0, DenseWeightMaskCache::default());
        }
        if inventory.final_sets.is_empty() && inventory.transition_sets.is_empty() {
            return (internal_token_dense_words, DenseWeightMaskCache::default());
        }

        let classify_started = profile.then(std::time::Instant::now);
        let WeightTokenSetInventory {
            final_sets,
            transition_sets: mut unique_sets,
            transition_word_spans,
        } = inventory;
        let mut residual_final_sets: DirectSparseWeightTokenSetCache = Default::default();
        for (key, token_set) in final_sets {
            if !direct_final_sets.contains(&key) {
                // These sets may need the contained-output cache, which
                // requires their dense form regardless of range width.
                residual_final_sets.insert(key);
                unique_sets.entry(key).or_insert(token_set);
            }
        }

        unique_sets.retain(|key, token_set| {
            residual_final_sets.contains(key)
                || transition_word_spans
                    .as_ref()
                    .and_then(|spans| spans.get(key))
                    .map_or_else(
                        || {
                            Self::token_set_dense_word_spans_at_least(
                                token_set,
                                internal_token_dense_words,
                                Self::DENSE_WEIGHT_PRECOMPUTE_MIN_WORD_SPANS,
                            )
                        },
                        |&spans| {
                            spans as usize >= Self::DENSE_WEIGHT_PRECOMPUTE_MIN_WORD_SPANS
                        },
                    )
        });
        let classify_ms = classify_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let build_started = profile.then(std::time::Instant::now);
        let dense_set_count = unique_sets.len();
        let build = |(key, token_set): (usize, Arc<RangeSetBlaze<u32>>)| {
            (
                key,
                Self::dense_words_from_internal_set_with_words(
                    token_set.as_ref(),
                    internal_token_dense_words,
                ),
            )
        };
        let dense_masks: DenseWeightMaskCache = if rayon::current_num_threads() == 1 {
            unique_sets.into_iter().map(build).collect()
        } else {
            unique_sets.into_par_iter().map(build).collect()
        };
        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][dense_weight_masks] sets={} words_per_set={} classify_ms={:.3} build_ms={:.3} total_ms={:.3}",
                dense_set_count,
                internal_token_dense_words,
                classify_ms,
                build_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }

        (internal_token_dense_words, dense_masks)
    }

    pub(super) fn compute_packed_dwa_dense_token_masks(&self) -> PackedDwaDenseWeightMaskCache {
        let Some(dwa) = self.packed_parser_dwa.as_ref() else {
            return PackedDwaDenseWeightMaskCache::default();
        };
        if self.internal_token_dense_words == 0 {
            return PackedDwaDenseWeightMaskCache::default();
        }

        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
        let started = profile.then(std::time::Instant::now);
        let min_word_spans = Self::DENSE_WEIGHT_PRECOMPUTE_MIN_WORD_SPANS.max(
            self.internal_token_dense_words
                .saturating_mul(Self::PACKED_DWA_DENSE_MIN_SCAN_MULTIPLIER),
        );
        let transition_ids = dwa.transition_token_set_ids();
        let transition_count = transition_ids.len();
        let build = |id: u32| {
            let token_set = dwa.token_set(id)?;
            if (token_set.word_spans() as usize) < min_word_spans {
                return None;
            }
            Some((
                id,
                self.dense_words_from_runtime_token_set(RuntimeTokenSetRef::PackedDwa(token_set)),
            ))
        };
        let rows: Vec<(u32, DenseWords)> = if rayon::current_num_threads() == 1 {
            transition_ids.into_iter().filter_map(build).collect()
        } else {
            transition_ids.into_par_iter().filter_map(build).collect()
        };
        let result = PackedDwaDenseWeightMaskCache::from_rows(
            dwa.token_set_count(),
            self.internal_token_dense_words,
            rows,
        )
        .expect("compiler-built packed DWA dense-mask cache must be internally consistent");
        if let Some(started) = started {
            eprintln!(
                "[glrmask/profile][packed_dwa_dense_weight_masks] token_sets={} transition_sets={} cached_sets={} min_word_spans={} words_per_set={} bytes={} total_ms={:.3}",
                dwa.token_set_count(),
                transition_count,
                result.len(),
                min_word_spans,
                self.internal_token_dense_words,
                result.len().saturating_mul(self.internal_token_dense_words).saturating_mul(8),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    /// Build fast transition lookup tables from the DWA's BTreeMap transitions.
    pub(crate) fn build_fast_transitions(&mut self) {
        self.dwa_fast_transitions = self.compute_fast_transitions();
        let (indexed_dag_dense_transitions, indexed_dag_dense_finals) =
            self.compute_indexed_dag_dense_tables();
        self.indexed_dag_dense_transitions = indexed_dag_dense_transitions;
        self.indexed_dag_dense_finals = indexed_dag_dense_finals;
    }

    pub(crate) fn build_dense_token_masks(&mut self) {
        let (internal_token_dense_words, dense_masks) = self.compute_dense_token_masks();
        self.internal_token_dense_words = internal_token_dense_words;
        self.weight_token_dense_masks = dense_masks;
    }

    /// Precompute dense bitmaps for the seed phase: one bitmap per (state, terminal)
    /// pair, plus the universe bitmap. This lets seed_weight_dense use bitwise ANDNOT
    /// instead of RangeSetBlaze subtraction.
    pub(crate) fn build_seed_dense_masks(&mut self) {
        self.seed_terminal_dense_fallback
            .lock()
            .expect("seed exclusion cache poisoned")
            .clear();
        let dw = self.internal_token_dense_words;
        if dw == 0 {
            self.seed_terminal_dense.clear();
            self.seed_universe_dense = empty_dense_words();
            return;
        }

        let universe = self.internal_token_universe();
        self.seed_universe_dense = self.dense_words_from_internal_set(&universe);

        self.seed_terminal_dense = self.build_seed_terminal_dense_masks();
    }



    pub(super) fn dense_words_from_internal_set_with_words(
        internal_tokens: &RangeSetBlaze<u32>,
        dense_word_count: usize,
    ) -> DenseWords {
        let mut words = vec![0u64; dense_word_count];
        let Some(max_token) = dense_word_count.checked_mul(64).and_then(|count| count.checked_sub(1)) else {
            return Arc::from(words.into_boxed_slice());
        };

        for token_range in internal_tokens.ranges() {
            let start = *token_range.start() as usize;
            if start > max_token {
                continue;
            }
            let end = (*token_range.end() as usize).min(max_token);
            let first_word = start / 64;
            let last_word = end / 64;
            let first_bit = start % 64;
            let last_bit = end % 64;

            if first_word == last_word {
                let high_mask = if last_bit == 63 {
                    u64::MAX
                } else {
                    (1u64 << (last_bit + 1)) - 1
                };
                words[first_word] |= (u64::MAX << first_bit) & high_mask;
                continue;
            }

            words[first_word] |= u64::MAX << first_bit;
            if first_word + 1 < last_word {
                words[first_word + 1..last_word].fill(u64::MAX);
            }
            let last_mask = if last_bit == 63 {
                u64::MAX
            } else {
                (1u64 << (last_bit + 1)) - 1
            };
            words[last_word] |= last_mask;
        }
        Arc::from(words.into_boxed_slice())
    }

    fn dense_words_from_internal_set(&self, internal_tokens: &RangeSetBlaze<u32>) -> DenseWords {
        Self::dense_words_from_internal_set_with_words(internal_tokens, self.internal_token_dense_words)
    }

    fn fill_dense_words_from_runtime_token_set(
        words: &mut [u64],
        internal_tokens: RuntimeTokenSetRef<'_>,
    ) {
        let Some(max_token) = words
            .len()
            .checked_mul(64)
            .and_then(|count| count.checked_sub(1))
        else {
            return;
        };
        internal_tokens.for_each_range(|start, end| {
            let start = start as usize;
            if start > max_token {
                return;
            }
            let end = (end as usize).min(max_token);
            let first_word = start / 64;
            let last_word = end / 64;
            let first_bit = start % 64;
            let last_bit = end % 64;
            if first_word == last_word {
                let high_mask = if last_bit == 63 {
                    u64::MAX
                } else {
                    (1u64 << (last_bit + 1)) - 1
                };
                words[first_word] |= (u64::MAX << first_bit) & high_mask;
                return;
            }
            words[first_word] |= u64::MAX << first_bit;
            if first_word + 1 < last_word {
                words[first_word + 1..last_word].fill(u64::MAX);
            }
            let last_mask = if last_bit == 63 {
                u64::MAX
            } else {
                (1u64 << (last_bit + 1)) - 1
            };
            words[last_word] |= last_mask;
        });
    }

    fn dense_words_from_runtime_token_set(
        &self,
        internal_tokens: RuntimeTokenSetRef<'_>,
    ) -> DenseWords {
        let mut words = vec![0u64; self.internal_token_dense_words];
        Self::fill_dense_words_from_runtime_token_set(&mut words, internal_tokens);
        Arc::from(words.into_boxed_slice())
    }

    #[inline]
    pub(crate) fn runtime_token_set_dense_mask(
        &self,
        token_set: RuntimeTokenSetRef<'_>,
    ) -> Option<&[u64]> {
        if let Some(key) = token_set.materialized_key() {
            return self.weight_token_dense_masks.get(&key).map(AsRef::as_ref);
        }
        token_set
            .packed_id()
            .and_then(|id| self.packed_dwa_token_dense_masks.get(id))
    }

    fn build_seed_terminal_dense_masks(&self) -> SeedTerminalDenseMasks {
        let mut result = SeedTerminalDenseMasks::default();
        let internal_tsid_to_states = self.internal_tsid_groups();
        for terminal_id in self.runtime_possible_match_terminals() {
            let Some(weight) = self.runtime_possible_match_weight(terminal_id) else {
                continue;
            };
            weight.for_each_entry(|start, end, token_set| {
                let dense = self.dense_words_from_runtime_token_set(token_set);
                for internal_tsid in start..=end {
                    if let Some(states) = internal_tsid_to_states.get(internal_tsid as usize) {
                        for &tokenizer_state in states {
                            let entry = result
                                .entry((tokenizer_state, terminal_id))
                                .or_insert_with(empty_dense_words);
                            let mut merged = entry.to_vec();
                            if merged.len() < dense.len() {
                                merged.resize(dense.len(), 0);
                            }
                            for (index, &word) in dense.iter().enumerate() {
                                merged[index] |= word;
                            }
                            *entry = merged.into();
                        }
                    } else {
                        result.insert((internal_tsid, terminal_id), dense.clone());
                    }
                }
            });
        }
        result
    }
}
