//! Vocabulary-relative terminal observations and projected runtime certificates.

use crate::automata::lexer::Lexer;
use crate::grammar::flat::TerminalID;
use crate::runtime::artifact::Constraint;
use crate::runtime::artifact::DynamicBoundedObservationSets;
use crate::runtime::artifact::DynamicMaskVocab;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

impl Constraint {




    pub(crate) fn build_dynamic_terminal_observation_classes(
        &self,
    ) -> Vec<(TerminalID, Arc<[u32]>)> {
        let classes = if std::env::var_os("GLRMASK_DISABLE_EARLY_OBSERVATION_FILTER").is_some() {
            self.build_dynamic_terminal_observation_classes_filtered::<false>()
        } else {
            self.build_dynamic_terminal_observation_classes_filtered::<true>()
        };
        if std::env::var_os("GLRMASK_ASSERT_EARLY_OBSERVATION_FILTER").is_some() {
            assert_eq!(
                classes,
                self.build_dynamic_terminal_observation_classes_filtered::<false>(),
                "early future filtering changed exact terminal-observation rows",
            );
        }
        classes
    }

    // Only reorder necessary, immutable selector predicates. The historical
    // selector remains available as an exact same-binary differential oracle.
    pub(super) fn build_dynamic_terminal_observation_classes_filtered<const EARLY_FUTURES: bool>(
        &self,
    ) -> Vec<(TerminalID, Arc<[u32]>)> {
        if std::env::var_os("GLRMASK_DISABLE_DYNAMIC_TERMINAL_OBSERVATION_CACHE").is_some() {
            return Vec::new();
        }
        if self.tokenizer.has_any_virtual_runtime() {
            // Virtual runtimes leave the finite physical DFA domain after
            // their physical proxy/root. An observation quotient over only raw DFA
            // states cannot certify those lazily-created virtual states.
            return Vec::new();
        }

        // Preserve the historical selector exactly on its successful path:
        // broad literal-self-loop residuals whose futures contain a terminal
        // admitted by at least one singleton LR row. This sidecar was already
        // paid for by production before parser-relative commit reuse consumed it.
        let mut singleton_rows = vec![0usize; self.table.num_terminals as usize];
        for row in &self.table.advance {
            let mut terminals = row.iter();
            let Some(terminal) = terminals.next() else {
                continue;
            };
            if terminals.next().is_none()
                && let Some(count) = singleton_rows.get_mut(terminal)
            {
                *count += 1;
            }
        }
        let mut best_by_futures = BTreeMap::<Vec<TerminalID>, (usize, u32)>::new();
        for state in 0..self.tokenizer.num_states() {
            // Singleton/empty and >8-terminal futures can never participate.
            // Check that cheap metadata before scanning potentially 256 byte
            // transitions twice. Nine entries are enough to reject >8 exactly;
            // a retained 2..=8 signature is complete, ordered and unchanged.
            let early_futures = if EARLY_FUTURES {
                let futures = self.tokenizer.possible_future_terminals_iter(state)
                    .take(9).collect::<SmallVec<[TerminalID; 8]>>();
                if !(2..=8).contains(&futures.len()) {
                    continue;
                }
                Some(futures)
            } else {
                None
            };
            if self.tokenizer.transitions_from(state).count() < 100 {
                continue;
            }
            let loop_len = self.tokenizer.self_loop_bytes(state).len();
            if loop_len < 64 {
                continue;
            }
            let futures = early_futures.map(SmallVec::into_vec).unwrap_or_else(|| {
                self.tokenizer.possible_future_terminals_iter(state).collect::<Vec<_>>()
            });
            if !(2..=8).contains(&futures.len()) {
                continue;
            }
            let entry = best_by_futures
                .entry(futures)
                .or_insert((loop_len, state));
            if (loop_len, std::cmp::Reverse(state))
                > (entry.0, std::cmp::Reverse(entry.1))
            {
                *entry = (loop_len, state);
            }
        }
        let mut ranked = best_by_futures
            .into_iter()
            .map(|(futures, (loop_len, state))| (loop_len, state, futures))
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| {
            (std::cmp::Reverse(left.0), left.1, &left.2)
                .cmp(&(std::cmp::Reverse(right.0), right.1, &right.2))
        });
        ranked.truncate(2);

        let mut candidates = BTreeSet::<TerminalID>::new();
        for (_, _, futures) in ranked {
            let best = futures
                .iter()
                .copied()
                .filter_map(|terminal| {
                    let count = singleton_rows
                        .get(terminal as usize)
                        .copied()
                        .unwrap_or(0);
                    (count != 0).then_some((count, std::cmp::Reverse(terminal)))
                })
                .max()
                .map(|(_, std::cmp::Reverse(terminal))| terminal);
            if let Some(terminal) = best {
                candidates.insert(terminal);
            }
        }

        // Only when the historical selector found nothing, look for the second
        // proven shape: a high-degree mixed residual signature repeated across
        // many raw states, with one of its terminals participating in a small
        // parser row. This captures bounded/string-prefix chains such as o9818
        // without adding any selection work to the existing sidecar population.
        let mut fallback_candidate = None;
        if candidates.is_empty() {
            const SMALL_PARSER_ROW_LIMIT: usize = 8;
            const REPEATED_MIXED_FAMILY_MIN_STATES: usize = 8;
            let mut small_rows = vec![0usize; self.table.num_terminals as usize];
            for row in &self.table.advance {
                let row_len = row.iter().count();
                if (1..=SMALL_PARSER_ROW_LIMIT).contains(&row_len) {
                    for terminal in row.iter() {
                        if let Some(count) = small_rows.get_mut(terminal) {
                            *count += 1;
                        }
                    }
                }
            }
            let mut repeated_mixed =
                BTreeMap::<Vec<TerminalID>, (usize, usize, u32)>::new();
            for state in 0..self.tokenizer.num_states() {
                let early_futures = if EARLY_FUTURES {
                    let futures = self.tokenizer.possible_future_terminals_iter(state)
                        .take(9).collect::<SmallVec<[TerminalID; 8]>>();
                    if !(2..=8).contains(&futures.len())
                        || !futures.iter().any(|&terminal| {
                            small_rows.get(terminal as usize).copied().unwrap_or(0) != 0
                        })
                    {
                        continue;
                    }
                    Some(futures)
                } else {
                    None
                };
                let transitions = self.tokenizer.transitions_from(state).count();
                if transitions < 100 {
                    continue;
                }
                let futures = early_futures.map(SmallVec::into_vec).unwrap_or_else(|| {
                    self.tokenizer.possible_future_terminals_iter(state).collect::<Vec<_>>()
                });
                if !(2..=8).contains(&futures.len())
                    || !futures.iter().any(|&terminal| {
                        small_rows.get(terminal as usize).copied().unwrap_or(0) != 0
                    })
                {
                    continue;
                }
                let entry = repeated_mixed
                    .entry(futures)
                    .or_insert((0, transitions, state));
                entry.0 += 1;
                if (transitions, std::cmp::Reverse(state))
                    > (entry.1, std::cmp::Reverse(entry.2))
                {
                    entry.1 = transitions;
                    entry.2 = state;
                }
            }
            let fallback = repeated_mixed
                .into_iter()
                .filter(|(_, (count, _, _))| *count >= REPEATED_MIXED_FAMILY_MIN_STATES)
                .max_by(|left, right| {
                    let (_, (left_count, left_transitions, left_state)) = left;
                    let (_, (right_count, right_transitions, right_state)) = right;
                    (*left_count, *left_transitions, std::cmp::Reverse(*left_state))
                        .cmp(&(*right_count, *right_transitions, std::cmp::Reverse(*right_state)))
                });
            if let Some((futures, _)) = fallback {
                fallback_candidate = futures
                    .iter()
                    .copied()
                    .filter_map(|terminal| {
                        let count = small_rows
                            .get(terminal as usize)
                            .copied()
                            .unwrap_or(0);
                        (count != 0).then_some((count, std::cmp::Reverse(terminal)))
                    })
                    .max()
                    .map(|(_, std::cmp::Reverse(terminal))| terminal);
                if let Some(terminal) = fallback_candidate {
                    candidates.insert(terminal);
                }
            }
        }

        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_DYNAMIC_TERMINAL_OBSERVATION_CACHE").is_some();
        candidates
            .into_par_iter()
            .filter_map(|terminal| {
                let started = std::time::Instant::now();
                let has_alias = |classes: &[u32]| {
                    let mut seen = BTreeSet::<u32>::new();
                    classes
                        .iter()
                        .copied()
                        .filter(|&class| class != 0)
                        .any(|class| !seen.insert(class))
                };
                let coordinate_classes = self
                    .tokenizer
                    .terminal_residual_coordinate_observation_partition(terminal)
                    .filter(|(_, _, useful_alias)| *useful_alias);
                let (classes, configs, rounds) = if let Some((classes, distinct, _)) = coordinate_classes {
                    (classes, distinct, 0)
                } else {
                    self.tokenizer
                        .exact_terminal_observation_partition(terminal, 100_000, 20_000_000)?
                };

                let useful = has_alias(&classes);
                if profile {
                    eprintln!(
                        "[glrmask/profile][dynamic_terminal_observation_cache_build] terminal={} singleton_rows={} fallback={} configs={} rounds={} useful={} elapsed_ms={:.3}",
                        terminal,
                        singleton_rows.get(terminal as usize).copied().unwrap_or(0),
                        fallback_candidate == Some(terminal),
                        configs,
                        rounds,
                        useful,
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                useful.then_some((terminal, Arc::from(classes)))
            })
            .collect()
    }

    pub(super) fn dynamic_terminal_observation_classes_enabled(
        &self,
        vocab: &DynamicMaskVocab,
    ) -> bool {
        match std::env::var("GLRMASK_DYNAMIC_TERMINAL_OBSERVATION_CLASSES") {
            Ok(value) => !matches!(value.trim(), "0" | "false" | "no" | "off"),
            // O2 already collapses the model vocabulary to the exact grammar
            // quotient. On the canonical runtime cohort this additional lexer
            // observation quotient costs build time without improving TBM, so
            // leave it off by default for grammar-quotiented constraints. An
            // explicit truthy env value remains an opt-in for experiments.
            Err(_) => !vocab.is_grammar_quotiented(),
        }
    }

    pub(crate) fn prepare_dynamic_terminal_observation_classes_for_artifact(&mut self) {
        if self.dynamic_mask_vocab.has_terminal_observation_classes() {
            return;
        }
        if !self.dynamic_terminal_observation_classes_enabled(&self.dynamic_mask_vocab) {
            return;
        }
        let classes = self.build_dynamic_terminal_observation_classes();
        self.dynamic_mask_vocab
            .set_terminal_observation_classes(classes);
    }

    pub(crate) fn prepare_dynamic_virtual_residual_mask_projections_for_artifact(&mut self) {
        if self
            .dynamic_mask_vocab
            .virtual_residual_mask_projection_parts()
            .is_some()
            || !self.tokenizer.has_any_virtual_runtime()
        {
            return;
        }
        if std::env::var("GLRMASK_DYNAMIC_TRANSFER_VIRTUAL_RESIDUAL_PROJECTIONS")
            .ok()
            .is_some_and(|value| matches!(value.trim(), "0" | "false" | "no" | "off"))
        {
            return;
        }
        let mut dynamic_mask_vocab = std::mem::take(&mut self.dynamic_mask_vocab);
        self.prepare_dynamic_virtual_residual_mask_projection(&mut dynamic_mask_vocab);
        self.dynamic_mask_vocab = dynamic_mask_vocab;
    }

    /// Preserve the exact projected-terminal proof coordinate for future mask
    /// acceleration work without making it part of ordinary runtime
    /// finalization. This used to be eagerly prepared for the retired general
    /// walker; callers should opt into the cost explicitly if/when the strict
    /// full walker learns to consume it again.
    pub(crate) fn prepare_dynamic_projected_terminal_quotients_for_artifact(&mut self) {
        if self
            .dynamic_mask_vocab
            .projected_terminal_quotients_prepared()
        {
            return;
        }
        if std::env::var("GLRMASK_DYNAMIC_PROJECTED_TERMINAL_QUOTIENTS")
            .ok()
            .is_some_and(|value| matches!(value.trim(), "0" | "false" | "no" | "off"))
        {
            self.dynamic_mask_vocab
                .set_projected_terminal_quotients(Vec::new());
            return;
        }
        let quotients = self
            .tokenizer
            .build_shared_component_terminal_projected_quotients(256);
        self.dynamic_mask_vocab
            .set_projected_terminal_quotients(quotients);
    }

    /// Materialize the exact H16/H64 observation-stability certificates that
    /// powered bounded subtree accelerators in the previous dynamic walker.
    /// Kept as dormant infrastructure for later strict-full-walk P50 work;
    /// ordinary runtime finalization deliberately does not call this today.
    pub(crate) fn prepare_dynamic_bounded_observation_sets(&mut self) {
        let observation_tokenizer = self
            .dynamic_mask_vocab
            .mask_projection_tokenizer()
            .unwrap_or(&self.tokenizer);
        let (bounded16, bounded64) =
            observation_tokenizer.precompute_bounded_observation_safe_byte_sets();
        self.dynamic_mask_vocab
            .set_bounded_observation_sets(DynamicBoundedObservationSets::from_raw(
                bounded16, bounded64,
            ));
    }
}
