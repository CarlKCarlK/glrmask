//! Generation-scoped mask retention and bounded equality certificates.
//!
//! Failure to prove equality invalidates the cached generation; it never
//! relaxes a grammar or admits a token. Scoped composition requires exact
//! state equality and cannot use ordinary lexer-coordinate certificates.

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::glr::{accumulator::TerminalsDisallowed, parser::ParserGSS};
use crate::runtime::artifact::DynamicMaskVocab;
use crate::runtime::state::{CommitBuffers, ConstraintState, ParserRelativeMaskEqCacheEntry, ParserStateMap};

use super::exact_admitted_terminals_for_candidates;

const DEFAULT_PROOF_STEPS: usize = 64;

#[cold]
#[inline(never)]
fn try_terminal_observation_quotient_commit_reuse(
    buffers: &mut CommitBuffers,
    generation: u64,
    vocab: &DynamicMaskVocab,
    tokenizer: &Tokenizer,
    left_source: u32,
    right_source: u32,
    admitted: &crate::ds::bitset::BitSet,
    cache_left: u32,
    cache_right: u32,
    visited_nodes: usize,
    byte_steps: usize,
    started: Option<std::time::Instant>,
) -> bool {
    let Some(quotient_relevant) = vocab.terminal_observation_equivalent_for_live_admitted(
        left_source,
        right_source,
        admitted,
        tokenizer.matched_terminal_bitset(left_source),
        tokenizer.possible_future_terminals(left_source),
        tokenizer.matched_terminal_bitset(right_source),
        tokenizer.possible_future_terminals(right_source),
    ) else {
        return false;
    };

    if let Some(started) = started {
        eprintln!(
            "[glrmask/profile][parser_relative_commit_reuse] generation={} result=terminal_observation_quotient left={} right={} admitted={} relevant={} nodes={} bytes={} elapsed_us={:.1}",
            generation,
            left_source,
            right_source,
            admitted.count_ones(),
            quotient_relevant,
            visited_nodes,
            byte_steps,
            started.elapsed().as_secs_f64() * 1e6,
        );
    }

    const CACHE_CAPACITY: usize = 8;
    if buffers.parser_relative_mask_eq_cache.len() == CACHE_CAPACITY {
        buffers.parser_relative_mask_eq_cache.remove(0);
    }
    buffers
        .parser_relative_mask_eq_cache
        .push(ParserRelativeMaskEqCacheEntry {
            left: cache_left,
            right: cache_right,
            admitted: admitted.clone(),
        });
    true
}

impl ConstraintState<'_> {
    /// Snapshot the exact runtime state only when a reusable mask is currently
    /// cached. `ParserGSS` is Arc-backed, so cloning this small map keeps the
    /// old GSS identities alive without copying parser stacks.
    #[inline]
    pub(super) fn snapshot_current_mask_state(&self) -> Option<ParserStateMap> {
        let cache = self.mask_cache.lock().unwrap();
        cache
            .as_ref()
            .is_some_and(|cache_data| cache_data.generation == self.generation)
            .then(|| self.state.clone())
    }

    /// Advance the commit generation while retaining a cached mask when commit
    /// left the complete semantic runtime state unchanged. `ParserStateMap`
    /// equality compares tokenizer keys and exact immutable GSS structure;
    /// `LeveledGSS` itself checks Arc identity first, so the usual unchanged
    /// object case remains a pointer comparison. `fill_mask` is a pure
    /// function of this state and the immutable constraint.
    #[inline]
    pub(super) fn finish_commit_generation(
        &mut self,
        previous_state: Option<ParserStateMap>,
        commit_succeeded: bool,
    ) {
        let previous_generation = self.generation;
        self.generation += 1;
        if !commit_succeeded {
            return;
        }
        let Some(previous_state) = previous_state else {
            return;
        };
        let mask_state_unchanged = previous_state == self.state
            || (self.constraint.uses_dynamic_runtime()
                // Ordinary tokenizer quotients cannot interpret scoped
                // recursive composition IDs. Only exact state equality above
                // is currently proved for retaining a composed result.
                && !self.constraint.uses_compact_segmented_parser_runtime()
                && (self.dynamic_mask_projection_state_eq(&previous_state, &self.state)
                    || (std::env::var_os(
                        "GLRMASK_DISABLE_DYNAMIC_MASK_PARSER_RELATIVE_COMMIT_REUSE",
                    )
                    .is_none()
                        && self.dynamic_parser_relative_vocab_mask_eq(&previous_state))));
        if !mask_state_unchanged {
            return;
        }

        let mut cache = self.mask_cache.lock().unwrap();
        if let Some(cache_data) = cache.as_mut()
            && cache_data.generation == previous_generation
        {
            cache_data.generation = self.generation;
        }
    }

    /// Exact equality in the lexer coordinate actually consumed by dynamic
    /// mask generation. A commit may advance the source/runtime lexer through
    /// states that are distinct only outside the model-vocabulary horizon; if
    /// every correlated parser GSS is unchanged and those source states map to
    /// the same mask-execution state, the next-token mask is identical and the
    /// already-cached mask can survive the generation bump.
    ///
    /// This is deliberately stricter than general semantic cache lookup: it
    /// does not merge parser alternatives or rely on observation classes. It
    /// only replaces the exact tokenizer-state coordinate with the exact
    /// finite mask coordinate selected by the constraint.
    pub(super) fn dynamic_mask_projection_state_eq(
        &self,
        previous: &ParserStateMap,
        current: &ParserStateMap,
    ) -> bool {
        if previous.len() != current.len() {
            return false;
        }
        let vocab = self.constraint.dynamic_mask_vocab_for_runtime();
        let exact_initial = self.constraint.tokenizer.initial_state();

        // The common case is one correlated lexer/parser branch. Keep that
        // path allocation-free and constant-time.
        if let ([(previous_state, previous_gss)], [(current_state, current_gss)]) =
            (previous.entries.as_slice(), current.entries.as_slice())
        {
            return previous_gss == current_gss
                && (*previous_state == exact_initial) == (*current_state == exact_initial)
                && vocab.mask_runtime_state(*previous_state)
                    == vocab.mask_runtime_state(*current_state);
        }

        // Flat GLR frontiers can contain duplicate exact tokenizer keys. Match
        // correlated branches as a multiset in mask-coordinate space rather
        // than assuming source-state ordering survives quotienting.
        let mut matched = vec![false; current.len()];
        'previous: for (previous_state, previous_gss) in &previous.entries {
            let projected = vocab.mask_runtime_state(*previous_state);
            let initial = *previous_state == exact_initial;
            for (index, (current_state, current_gss)) in current.entries.iter().enumerate() {
                if !matched[index]
                    && initial == (*current_state == exact_initial)
                    && projected == vocab.mask_runtime_state(*current_state)
                    && previous_gss == current_gss
                {
                    matched[index] = true;
                    continue 'previous;
                }
            }
            return false;
        }
        true
    }

    /// Prove equality of the next-token mask for the common singleton parser
    /// frontier even when the exact lexer residual changed.
    ///
    /// This proof is deliberately vocabulary- and parser-relative. Two lexer
    /// states may differ because one still carries futures for terminals the
    /// unchanged parser frontier cannot consume. Those differences do not
    /// affect the next model-token mask. We walk the model vocabulary trie in
    /// lockstep and compare only matched/future observations for terminals
    /// admitted by this parser GSS. Once the lexer states converge, or both
    /// no-finalization branches become parser-dead, the complete suffix
    /// subtree is equal and no further work is needed.
    ///
    /// Equal admitted finalizers also need no recursive parser simulation:
    /// both sides advance the same GSS on the same terminal and reset to the
    /// same lexer initial state, so every post-finalization suffix branch is
    /// identical by construction.
    fn dynamic_parser_relative_vocab_mask_eq(
        &mut self,
        previous: &ParserStateMap,
    ) -> bool {
        let current = &self.state;
        if self.constraint.static_dynamic_overlay.is_some()
            || self.constraint.uses_compact_segmented_parser_runtime()
            || self.constraint.tokenizer.has_any_virtual_runtime()
        {
            return false;
        }
        let (left_source, left_gss, right_source, right_gss) =
            if let ([(left_source, left_gss)], [(right_source, right_gss)]) =
                (previous.entries.as_slice(), current.entries.as_slice())
            {
                (left_source, left_gss, right_source, right_gss)
            } else {
                // Conservative multi-branch extension: first pair off every
                // branch whose exact correlated lexer/GSS state is unchanged.
                // Only if exactly one branch remains on each side do we apply
                // the parser-relative lexer proof below.  This covers the
                // common GLR shape where one guarded/reset branch is stable
                // and one ordinary lexer residual advances, without needing a
                // general assignment search over unrelated alternatives.
                if previous.len() != current.len() {
                    return false;
                }
                let mut matched = vec![false; current.len()];
                let mut unmatched_previous = None::<(&u32, &ParserGSS)>;
                for (previous_state, previous_gss) in &previous.entries {
                    let exact = current
                        .entries
                        .iter()
                        .enumerate()
                        .find(|(index, (current_state, current_gss))| {
                            !matched[*index]
                                && previous_state == current_state
                                && previous_gss == current_gss
                        })
                        .map(|(index, _)| index);
                    if let Some(index) = exact {
                        matched[index] = true;
                        continue;
                    }
                    if unmatched_previous
                        .replace((previous_state, previous_gss))
                        .is_some()
                    {
                        return false;
                    }
                }
                let Some((left_source, left_gss)) = unmatched_previous else {
                    return false;
                };
                let mut unmatched_current = current
                    .entries
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !matched[*index])
                    .map(|(_, pair)| pair);
                let Some((right_source, right_gss)) = unmatched_current.next() else {
                    return false;
                };
                if unmatched_current.next().is_some() {
                    return false;
                }
                (left_source, left_gss, right_source, right_gss)
            };
        if left_gss != right_gss
            || !left_gss.all_accs_satisfy(|acc: &TerminalsDisallowed| acc.is_empty())
        {
            return false;
        }

        let tokenizer = &self.constraint.tokenizer;
        let initial = tokenizer.initial_state();
        if (*left_source == initial) != (*right_source == initial)
            || tokenizer.state_has_epsilon_transitions(*left_source)
            || tokenizer.state_has_epsilon_transitions(*right_source)
        {
            return false;
        }

        let (cache_left, cache_right) = if left_source <= right_source {
            (*left_source, *right_source)
        } else {
            (*right_source, *left_source)
        };
        let candidates = crate::ds::bitset::BitSet::all(
            self.constraint.table.num_terminals as usize,
        );
        let mut admitted = exact_admitted_terminals_for_candidates(
            self.constraint,
            left_gss,
            &candidates,
        );
        // IGNORE is parser-transparent but still lexer-observable: a live
        // ignore future permits a token boundary, and an ignore match resets
        // the lexer while leaving the current GSS unchanged.  Treat it as an
        // always-observed terminal in this equivalence proof. Equal ignore
        // events therefore converge exactly just like equal parser-admitted
        // finalizers, except without a parser advance.
        if let Some(ignore) = self.constraint.ignore_terminal {
            admitted.set(ignore as usize);
        }
        if admitted.count_ones() == 0 {
            return true;
        }

        if self
            .buffers
            .parser_relative_mask_eq_cache
            .iter()
            .any(|entry| {
                entry.left == cache_left
                    && entry.right == cache_right
                    && entry.admitted == admitted
            })
        {
            if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK_COMMIT_REUSE").is_some() {
                eprintln!(
                    "[glrmask/profile][parser_relative_commit_reuse] generation={} result=cache_hit left={} right={} admitted={}",
                    self.generation,
                    left_source,
                    right_source,
                    admitted.count_ones(),
                );
            }
            return true;
        }
        // A new relation certificate can cost tens of microseconds. If the
        // destination state already has a materialized global dynamic-mask
        // cache entry, proving equality with the previous state cannot beat
        // simply letting the next fill copy that existing result. Keep this
        // lookup below all cheap structural/relation-cache gates so ordinary
        // commits never pay dynamic-mask key construction merely to decline.
        if crate::runtime::dynamic_mask::dynamic_mask_state_has_cached_result(self) {
            return false;
        }

        #[inline(always)]
        fn equal_under_mask(
            left: &crate::ds::bitset::BitSet,
            right: &crate::ds::bitset::BitSet,
            mask: &crate::ds::bitset::BitSet,
        ) -> bool {
            left.words()
                .iter()
                .zip(right.words())
                .zip(mask.words())
                .all(|((&left, &right), &mask)| ((left ^ right) & mask) == 0)
        }

        #[inline(always)]
        fn any_under_mask(
            value: &crate::ds::bitset::BitSet,
            mask: &crate::ds::bitset::BitSet,
        ) -> bool {
            value
                .words()
                .iter()
                .zip(mask.words())
                .any(|(&value, &mask)| value & mask != 0)
        }

        let compare = |left: u32, right: u32| -> Option<(bool, bool)> {
            const DEAD: u32 = u32::MAX;
            if left == DEAD || right == DEAD {
                let live = |state: u32| {
                    state != DEAD
                        && (state == initial
                            || any_under_mask(
                                tokenizer.matched_terminal_bitset(state),
                                &admitted,
                            )
                            || any_under_mask(
                                tokenizer.possible_future_terminals(state),
                                &admitted,
                            ))
                };
                let left_live = live(left);
                let right_live = live(right);
                return Some((left_live == right_live, left_live && right_live));
            }
            if tokenizer.state_has_epsilon_transitions(left)
                || tokenizer.state_has_epsilon_transitions(right)
                || (left == initial) != (right == initial)
            {
                return None;
            }
            let matched_equal = equal_under_mask(
                tokenizer.matched_terminal_bitset(left),
                tokenizer.matched_terminal_bitset(right),
                &admitted,
            );
            let futures_equal = equal_under_mask(
                tokenizer.possible_future_terminals(left),
                tokenizer.possible_future_terminals(right),
                &admitted,
            );
            if !matched_equal || !futures_equal {
                return Some((false, true));
            }
            let live = left == initial
                || any_under_mask(tokenizer.matched_terminal_bitset(left), &admitted)
                || any_under_mask(tokenizer.possible_future_terminals(left), &admitted);
            Some((true, live))
        };

        let Some((root_equal, root_live)) = compare(*left_source, *right_source) else {
            return false;
        };
        if !root_equal {
            return false;
        }
        if !root_live || left_source == right_source {
            return true;
        }

        let vocab = self.constraint.dynamic_mask_vocab_for_runtime();
        let started = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK_COMMIT_REUSE")
            .is_some()
            .then(std::time::Instant::now);
        if try_terminal_observation_quotient_commit_reuse(
            &mut self.buffers, self.generation, vocab, tokenizer,
            *left_source, *right_source, &admitted, cache_left, cache_right,
            0, 0, started,
        ) {
            return true;
        }

        // This is speculative work to avoid the next mask, not a second mask
        // computation. A failed short proof safely leaves normal masking in
        // charge. The explicit override remains available for diagnostics.
        let max_byte_steps = std::env::var(
            "GLRMASK_DYNAMIC_MASK_PARSER_RELATIVE_COMMIT_REUSE_MAX_STEPS",
        )
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_PROOF_STEPS);
        let trie = vocab.trie.as_ref();
        let mut stack = vec![(0u32, *left_source, *right_source)];
        let mut visited_nodes = 0usize;
        let mut byte_steps = 0usize;
        let mut converged_subtrees = 0usize;
        let mut dead_subtrees = 0usize;

        while let Some((node, left_state, right_state)) = stack.pop() {
            visited_nodes += 1;
            for edge in trie.children(node) {
                let mut left = left_state;
                let mut right = right_state;
                let mut subtree_done = false;
                for &byte in trie.edge_bytes(edge) {
                    if left == right {
                        converged_subtrees += 1;
                        subtree_done = true;
                        break;
                    }
                    byte_steps += 1;
                    if byte_steps > max_byte_steps
                        || tokenizer.state_has_epsilon_transitions(left)
                        || tokenizer.state_has_epsilon_transitions(right)
                    {
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][parser_relative_commit_reuse] generation={} result=decline reason={} nodes={} bytes={} converged={} dead={} elapsed_us={:.1}",
                                self.generation,
                                if byte_steps > max_byte_steps { "budget" } else { "epsilon" },
                                visited_nodes, byte_steps, converged_subtrees, dead_subtrees,
                                started.elapsed().as_secs_f64() * 1e6,
                            );
                        }
                        return false;
                    }
                    let left_next = tokenizer.get_transition(left, byte);
                    let right_next = tokenizer.get_transition(right, byte);
                    let Some((equal, live)) = compare(left_next, right_next) else {
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][parser_relative_commit_reuse] generation={} result=decline reason=epsilon_target nodes={} bytes={} converged={} dead={} elapsed_us={:.1}",
                                self.generation,
                                visited_nodes,
                                byte_steps,
                                converged_subtrees,
                                dead_subtrees,
                                started.elapsed().as_secs_f64() * 1e6,
                            );
                        }
                        return false;
                    };
                    if !equal {
                        if let Some(started) = started {
                            eprintln!(
                                "[glrmask/profile][parser_relative_commit_reuse] generation={} result=different nodes={} bytes={} converged={} dead={} elapsed_us={:.1}",
                                self.generation,
                                visited_nodes,
                                byte_steps,
                                converged_subtrees,
                                dead_subtrees,
                                started.elapsed().as_secs_f64() * 1e6,
                            );
                        }
                        return false;
                    }
                    if !live {
                        dead_subtrees += 1;
                        subtree_done = true;
                        break;
                    }
                    left = left_next;
                    right = right_next;
                }
                if subtree_done {
                    continue;
                }
                if left == right {
                    converged_subtrees += 1;
                    continue;
                }
                stack.push((edge.child, left, right));
            }
        }

        if let Some(started) = started {
            eprintln!(
                "[glrmask/profile][parser_relative_commit_reuse] generation={} result=equal left={} right={} admitted={} nodes={} bytes={} converged={} dead={} elapsed_us={:.1}",
                self.generation,
                left_source,
                right_source,
                admitted.count_ones(),
                visited_nodes,
                byte_steps,
                converged_subtrees,
                dead_subtrees,
                started.elapsed().as_secs_f64() * 1e6,
            );
        }
        const CACHE_CAPACITY: usize = 8;
        if self.buffers.parser_relative_mask_eq_cache.len() == CACHE_CAPACITY {
            self.buffers.parser_relative_mask_eq_cache.remove(0);
        }
        self.buffers
            .parser_relative_mask_eq_cache
            .push(ParserRelativeMaskEqCacheEntry {
                left: cache_left,
                right: cache_right,
                admitted,
            });
        true
    }

}

#[cfg(test)]
mod tests;
