//! Prepared master-slice certificates, exact coverage, and bounded radii.

use super::DynamicMaskVocab;
use crate::automata::lexer::DFA as LexerDfa;
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::TerminalProjectedQuotient;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::VecDeque;
use std::sync::Arc;
/// Compact parser-independent master-slice proof sidecar. All arrays are CSR
/// over `(exact tokenizer source, proof slot)` rows. `positive_*` stores true
/// proofs; `coverage_*` stores every terminal for which the build solved the
/// exact residual product, so covered-but-not-positive is an exact false.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedMasterProofArtifact {
    pub(crate) positive_row_ids: Vec<u32>,
    pub(crate) positive_offsets: Vec<u32>,
    pub(crate) positive_terminals: Vec<TerminalID>,
    pub(crate) coverage_row_ids: Vec<u32>,
    pub(crate) coverage_offsets: Vec<u32>,
    pub(crate) coverage_terminals: Vec<TerminalID>,
    /// Safe+ terminals for which the direct residual solver is complete over
    /// every exact source. For these terminals, absence from a source coverage
    /// row means that terminal has no residual coordinate at that source and is
    /// therefore an exact negative rather than an unknown requiring a quotient.
    pub(crate) safe_plus_complete_terminals: Vec<TerminalID>,
    pub(crate) safe_radius_row_ids: Vec<u32>,
    pub(crate) safe_radius_offsets: Vec<u32>,
    pub(crate) safe_radius_entries: Vec<(TerminalID, u16)>,
}

impl PreparedMasterProofArtifact {
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.positive_row_ids.is_empty()
            && self.coverage_row_ids.is_empty()
            && self.safe_plus_complete_terminals.is_empty()
            && self.safe_radius_row_ids.is_empty()
    }
}

#[inline]
pub(super) fn interned_terminal_row<'a>(
    row_ids: &[u32],
    offsets: &[u32],
    entries: &'a [TerminalID],
    row: usize,
) -> &'a [TerminalID] {
    let row_id = row_ids[row] as usize;
    &entries[offsets[row_id] as usize..offsets[row_id + 1] as usize]
}

pub(super) fn intern_terminal_rows(
    mut rows: Vec<SmallVec<[TerminalID; 4]>>,
) -> (Vec<u32>, Vec<u32>, Vec<TerminalID>) {
    let mut intern = FxHashMap::<Vec<TerminalID>, u32>::default();
    let mut unique_rows = Vec::<Vec<TerminalID>>::new();
    let mut row_ids = Vec::<u32>::with_capacity(rows.len());
    for row in &mut rows {
        row.sort_unstable();
        row.dedup();
        let slice = row.as_slice();
        let id = if let Some(&id) = intern.get(slice) {
            id
        } else {
            let owned = slice.to_vec();
            let id = unique_rows.len() as u32;
            intern.insert(owned.clone(), id);
            unique_rows.push(owned);
            id
        };
        row_ids.push(id);
    }
    let mut offsets = Vec::<u32>::with_capacity(unique_rows.len() + 1);
    let mut entries = Vec::<TerminalID>::new();
    offsets.push(0);
    for row in unique_rows {
        entries.extend(row);
        offsets.push(entries.len() as u32);
    }
    (row_ids, offsets, entries)
}

pub(super) fn intern_radius_rows(
    mut rows: Vec<SmallVec<[(TerminalID, u16); 4]>>,
) -> (Vec<u32>, Vec<u32>, Vec<(TerminalID, u16)>) {
    let mut intern = FxHashMap::<Vec<(TerminalID, u16)>, u32>::default();
    let mut unique_rows = Vec::<Vec<(TerminalID, u16)>>::new();
    let mut row_ids = Vec::<u32>::with_capacity(rows.len());
    for row in &mut rows {
        row.sort_unstable_by_key(|&(terminal, _)| terminal);
        row.dedup_by_key(|entry| entry.0);
        let slice = row.as_slice();
        let id = if let Some(&id) = intern.get(slice) {
            id
        } else {
            let owned = slice.to_vec();
            let id = unique_rows.len() as u32;
            intern.insert(owned.clone(), id);
            unique_rows.push(owned);
            id
        };
        row_ids.push(id);
    }
    let mut offsets = Vec::<u32>::with_capacity(unique_rows.len() + 1);
    let mut entries = Vec::<(TerminalID, u16)>::new();
    offsets.push(0);
    for row in unique_rows {
        entries.extend(row);
        offsets.push(entries.len() as u32);
    }
    (row_ids, offsets, entries)
}

impl DynamicMaskVocab {
    #[inline(always)]
    pub(crate) fn prepared_master_provers(&self, source: u32, slice_slot: usize) -> &[TerminalID] {
        if slice_slot >= Self::PREPARED_PROOF_SLOT_COUNT {
            return &[];
        }
        let row = source as usize * Self::PREPARED_PROOF_SLOT_COUNT + slice_slot;
        let Some(&row_id) = self.prepared_master_prover_row_ids.get(row) else {
            return &[];
        };
        let row_id = row_id as usize;
        let Some((&start, &end)) = self
            .prepared_master_prover_offsets
            .get(row_id)
            .zip(self.prepared_master_prover_offsets.get(row_id + 1))
        else {
            return &[];
        };
        self.prepared_master_prover_terminals
            .get(start as usize..end as usize)
            .unwrap_or(&[])
    }

    pub(crate) fn prepared_master_proof_artifact(&self) -> PreparedMasterProofArtifact {
        PreparedMasterProofArtifact {
            positive_row_ids: self.prepared_master_prover_row_ids.as_ref().to_vec(),
            positive_offsets: self.prepared_master_prover_offsets.as_ref().to_vec(),
            positive_terminals: self.prepared_master_prover_terminals.as_ref().to_vec(),
            coverage_row_ids: self.prepared_master_coverage_row_ids.as_ref().to_vec(),
            coverage_offsets: self.prepared_master_coverage_offsets.as_ref().to_vec(),
            coverage_terminals: self.prepared_master_coverage_terminals.as_ref().to_vec(),
            safe_plus_complete_terminals: self
                .prepared_safe_plus_complete_terminals
                .as_ref()
                .to_vec(),
            safe_radius_row_ids: self.prepared_safe_radius_row_ids.as_ref().to_vec(),
            safe_radius_offsets: self.prepared_safe_radius_offsets.as_ref().to_vec(),
            safe_radius_entries: self.prepared_safe_radius_entries.as_ref().to_vec(),
        }
    }

    pub(crate) fn restore_prepared_master_proof_artifact(
        &mut self,
        artifact: PreparedMasterProofArtifact,
        source_state_count: usize,
    ) -> Result<(), String> {
        let expected_rows = source_state_count
            .checked_mul(Self::PREPARED_PROOF_SLOT_COUNT)
            .ok_or_else(|| "prepared master proof row count overflow".to_owned())?;
        let validate_interned = |label: &str,
                                 row_ids: &[u32],
                                 offsets: &[u32],
                                 terminals: &[TerminalID],
                                 rows: usize| {
            if row_ids.is_empty() {
                return if offsets.is_empty() && terminals.is_empty() {
                    Ok(())
                } else {
                    Err(format!("{label} has table data without row ids"))
                };
            }
            if row_ids.len() != rows
                || offsets.is_empty()
                || offsets.first().copied() != Some(0)
                || offsets.windows(2).any(|pair| pair[0] > pair[1])
                || offsets.last().copied().map(|value| value as usize) != Some(terminals.len())
            {
                return Err(format!("{label} has invalid interned row framing"));
            }
            let unique_rows = offsets.len() - 1;
            if row_ids.iter().any(|&row| row as usize >= unique_rows) {
                return Err(format!("{label} has an out-of-range row id"));
            }
            for row in 0..unique_rows {
                let start = offsets[row] as usize;
                let end = offsets[row + 1] as usize;
                if terminals[start..end]
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                {
                    return Err(format!("{label} unique row {row} is not sorted and unique"));
                }
            }
            Ok(())
        };
        validate_interned(
            "prepared master positive rows",
            &artifact.positive_row_ids,
            &artifact.positive_offsets,
            &artifact.positive_terminals,
            expected_rows,
        )?;
        validate_interned(
            "prepared master coverage rows",
            &artifact.coverage_row_ids,
            &artifact.coverage_offsets,
            &artifact.coverage_terminals,
            expected_rows,
        )?;
        if !artifact.coverage_row_ids.is_empty() {
            if artifact.positive_row_ids.is_empty() {
                return Err(
                    "prepared master coverage exists without positive row framing".to_owned(),
                );
            }
            for row in 0..expected_rows {
                let coverage = interned_terminal_row(
                    &artifact.coverage_row_ids,
                    &artifact.coverage_offsets,
                    &artifact.coverage_terminals,
                    row,
                );
                if interned_terminal_row(
                    &artifact.positive_row_ids,
                    &artifact.positive_offsets,
                    &artifact.positive_terminals,
                    row,
                )
                .iter()
                .any(|terminal| coverage.binary_search(terminal).is_err())
                {
                    return Err(format!(
                        "prepared master positive row {row} is not a subset of coverage"
                    ));
                }
            }
        }
        if artifact
            .safe_plus_complete_terminals
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(
                "prepared safe+ complete terminal list is not sorted and unique".to_owned(),
            );
        }
        if !artifact.safe_radius_row_ids.is_empty() {
            if artifact.safe_radius_row_ids.len() != source_state_count
                || artifact.safe_radius_offsets.is_empty()
                || artifact.safe_radius_offsets.first().copied() != Some(0)
                || artifact
                    .safe_radius_offsets
                    .windows(2)
                    .any(|pair| pair[0] > pair[1])
                || artifact
                    .safe_radius_offsets
                    .last()
                    .copied()
                    .map(|value| value as usize)
                    != Some(artifact.safe_radius_entries.len())
            {
                return Err("prepared safe-radius rows have invalid interned framing".to_owned());
            }
            let unique_rows = artifact.safe_radius_offsets.len() - 1;
            if artifact
                .safe_radius_row_ids
                .iter()
                .any(|&row| row as usize >= unique_rows)
            {
                return Err("prepared safe-radius rows have an out-of-range row id".to_owned());
            }
            for row in 0..unique_rows {
                let start = artifact.safe_radius_offsets[row] as usize;
                let end = artifact.safe_radius_offsets[row + 1] as usize;
                if artifact.safe_radius_entries[start..end]
                    .windows(2)
                    .any(|pair| pair[0].0 >= pair[1].0)
                {
                    return Err(format!(
                        "prepared safe-radius unique row {row} is not sorted and unique"
                    ));
                }
            }
        } else if !artifact.safe_radius_offsets.is_empty()
            || !artifact.safe_radius_entries.is_empty()
        {
            return Err("prepared safe-radius table exists without row ids".to_owned());
        }
        self.prepared_master_prover_row_ids = Arc::from(artifact.positive_row_ids);
        self.prepared_master_prover_offsets = Arc::from(artifact.positive_offsets);
        self.prepared_master_prover_terminals = Arc::from(artifact.positive_terminals);
        self.prepared_master_coverage_row_ids = Arc::from(artifact.coverage_row_ids);
        self.prepared_master_coverage_offsets = Arc::from(artifact.coverage_offsets);
        self.prepared_master_coverage_terminals = Arc::from(artifact.coverage_terminals);
        self.prepared_safe_plus_complete_terminals =
            Arc::from(artifact.safe_plus_complete_terminals);
        self.prepared_safe_radius_row_ids = Arc::from(artifact.safe_radius_row_ids);
        self.prepared_safe_radius_offsets = Arc::from(artifact.safe_radius_offsets);
        self.prepared_safe_radius_entries = Arc::from(artifact.safe_radius_entries);
        Ok(())
    }

    /// Whether build/runtime preparation installed exact master-prover rows for
    /// this tokenizer source. An empty positive-terminal row is still a valid
    /// prepared row, so checking `prepared_master_provers()` itself is not
    /// sufficient.
    #[inline(always)]
    pub(crate) fn has_prepared_master_prover_row(&self, source: u32) -> bool {
        let row = source as usize * Self::PREPARED_PROOF_SLOT_COUNT;
        self.prepared_master_prover_row_ids
            .get(row + Self::PREPARED_PROOF_SLOT_COUNT - 1)
            .is_some()
    }

    /// Whether the compact prepared-master artifact contains row framing for
    /// every exact tokenizer source.  Transfer artifacts restore these rows
    /// verbatim; when they are complete, rebuilding the terminal projection
    /// quotients and solving the same source×slice products again on load is
    /// redundant.
    #[inline]
    pub(crate) fn has_complete_prepared_master_prover_rows(
        &self,
        source_state_count: usize,
    ) -> bool {
        source_state_count
            .checked_mul(Self::PREPARED_PROOF_SLOT_COUNT)
            .and_then(|rows| rows.checked_add(1))
            .is_some_and(|expected_offsets| {
                self.prepared_master_prover_offsets.len() == expected_offsets
            })
    }

    /// Exact build-time master-slice answer when this `(terminal, source)` is
    /// represented by an exact projected terminal quotient.
    ///
    /// `prepare_master_provers_all_sources` solves the complete
    /// quotientA-slice product for every retained quotient residual. Therefore
    /// once the compact row table is present, absence from a row is an exact
    /// negative result for a terminal whose quotient contains `source`.
    /// Missing rows or missing quotients remain `None` and must use the normal
    /// runtime proof path.
    #[inline]
    pub(crate) fn prepared_master_proof_result(
        &self,
        source: u32,
        slice_slot: usize,
        terminal: TerminalID,
    ) -> Option<bool> {
        if slice_slot >= Self::PREPARED_PROOF_SLOT_COUNT
            || self.prepared_master_prover_row_ids.is_empty()
        {
            return None;
        }
        let row = source as usize * Self::PREPARED_PROOF_SLOT_COUNT + slice_slot;
        self.prepared_master_prover_row_ids.get(row)?;
        if !self.prepared_master_coverage_row_ids.is_empty() {
            let coverage_id = *self.prepared_master_coverage_row_ids.get(row)? as usize;
            let (&start, &end) = self
                .prepared_master_coverage_offsets
                .get(coverage_id)
                .zip(self.prepared_master_coverage_offsets.get(coverage_id + 1))?;
            if self.prepared_master_coverage_terminals[start as usize..end as usize]
                .binary_search(&terminal)
                .is_err()
            {
                if slice_slot == Self::PREPARED_SAFE_PLUS_SLOT
                    && self
                        .prepared_safe_plus_complete_terminals
                        .binary_search(&terminal)
                        .is_ok()
                {
                    return Some(false);
                }
                return None;
            }
        } else if self.projected_terminal_quotient(terminal, source).is_none() {
            return None;
        }
        Some(
            self.prepared_master_provers(source, slice_slot)
                .binary_search(&terminal)
                .is_ok(),
        )
    }

    #[inline(always)]
    pub(crate) fn prepared_safe_radii(&self, source: u32) -> &[(TerminalID, u16)] {
        let Some(&row_id) = self.prepared_safe_radius_row_ids.get(source as usize) else {
            return &[];
        };
        let row_id = row_id as usize;
        let Some((&start, &end)) = self
            .prepared_safe_radius_offsets
            .get(row_id)
            .zip(self.prepared_safe_radius_offsets.get(row_id + 1))
        else {
            return &[];
        };
        self.prepared_safe_radius_entries
            .get(start as usize..end as usize)
            .unwrap_or(&[])
    }

    #[inline]
    pub(crate) fn prepared_safe_radius(&self, source: u32, terminal: TerminalID) -> Option<u16> {
        let row = self.prepared_safe_radii(source);
        if let Ok(index) = row.binary_search_by_key(&terminal, |&(candidate, _)| candidate) {
            return Some(row[index].1);
        }

        // Prepared radius rows intentionally omit zero radii to keep the
        // transfer compact. When the safe+ master-proof coverage row contains
        // this exact `(source, terminal)`, the radius solver was also run for
        // the same residual and an absent radius entry therefore means the
        // exact answer is zero, not "unprepared". Returning None here would
        // incorrectly launch the lazy projected-terminal quotient builder
        // during masking.
        self.prepared_master_proof_result(source, Self::PREPARED_SAFE_PLUS_SLOT, terminal)
            .map(|_| 0)
    }

    /// Exact all-start-state master-slice proof directly over an independently
    /// compiled terminal DFA retained by the partitioned lexer builder.
    ///
    /// The slice DFA already supplies an exact byte partition. For each
    /// terminal state we group live outgoing targets by slice class and require
    /// complete byte coverage for every class that can remain on an accepting
    /// slice prefix. This avoids materializing a dense global tokenizer-state
    /// quotient while preserving exact containment semantics.
    pub(super) fn terminal_dfa_partition_all_transparent_states(
        dfa: &LexerDfa,
        group: u32,
        partition: &VocabPartitionDfa,
    ) -> Option<(Vec<bool>, usize, usize)> {
        if dfa.has_epsilon_transitions() {
            return None;
        }
        let q_count = dfa.num_states();
        let p_count = partition.state_count();
        let class_count = partition.class_count();
        if q_count == 0 || p_count == 0 || class_count == 0 {
            return Some((vec![false; q_count], 0, 0));
        }

        let mut class_sizes = vec![0usize; class_count];
        let mut class_representatives = vec![u8::MAX; class_count];
        for raw in 0u16..=255 {
            let byte = raw as u8;
            let class = partition.byte_class(byte) as usize;
            class_sizes[class] += 1;
            if class_representatives[class] == u8::MAX {
                class_representatives[class] = byte;
            }
        }

        let relevant_classes = (0..p_count as u32)
            .map(|p| {
                if !partition.can_reach_accepting(p) {
                    return SmallVec::<[(usize, u32); 8]>::new();
                }
                class_representatives
                    .iter()
                    .enumerate()
                    .filter_map(|(class, &byte)| {
                        let target = partition.step(p, byte);
                        partition
                            .can_reach_accepting(target)
                            .then_some((class, target))
                    })
                    .collect::<SmallVec<[(usize, u32); 8]>>()
            })
            .collect::<Vec<_>>();

        let pair_count = p_count.saturating_mul(q_count);
        let pair_index = |p: u32, q: u32| p as usize * q_count + q as usize;
        let mut predecessors = vec![SmallVec::<[u32; 8]>::new(); pair_count];
        let mut bad = vec![false; pair_count];
        let mut queue = VecDeque::<u32>::new();
        let mut edge_count = 0usize;

        let state_live = |state: u32| {
            dfa.finalizers(state).contains(group as usize)
                || dfa
                    .possible_future_group_ids(state)
                    .contains(group as usize)
        };

        for q in 0..q_count as u32 {
            if !state_live(q) {
                for p in 0..p_count as u32 {
                    if !partition.can_reach_accepting(p) {
                        continue;
                    }
                    let current = pair_index(p, q);
                    bad[current] = true;
                    queue.push_back(current as u32);
                }
                continue;
            }

            let mut covered = vec![0usize; class_count];
            let mut targets = (0..class_count)
                .map(|_| SmallVec::<[u32; 4]>::new())
                .collect::<Vec<_>>();
            for (byte, target) in dfa.transitions(q) {
                if !state_live(target) {
                    continue;
                }
                let class = partition.byte_class(byte) as usize;
                covered[class] += 1;
                if !targets[class].contains(&target) {
                    targets[class].push(target);
                }
            }

            for p in 0..p_count as u32 {
                if !partition.can_reach_accepting(p) {
                    continue;
                }
                let current = pair_index(p, q);
                let relevant = &relevant_classes[p as usize];
                if relevant
                    .iter()
                    .any(|&(class, _)| covered[class] != class_sizes[class])
                {
                    bad[current] = true;
                    queue.push_back(current as u32);
                    continue;
                }
                for &(class, p_target) in relevant {
                    for &q_target in &targets[class] {
                        let target = pair_index(p_target, q_target);
                        predecessors[target].push(current as u32);
                        edge_count = edge_count.saturating_add(1);
                    }
                }
            }
        }

        while let Some(target) = queue.pop_front() {
            for &pred in &predecessors[target as usize] {
                if !bad[pred as usize] {
                    bad[pred as usize] = true;
                    queue.push_back(pred);
                }
            }
        }

        let start = partition.start_state();
        let mut transparent = vec![false; q_count];
        if partition.can_reach_accepting(start) {
            for q in 0..q_count as u32 {
                transparent[q as usize] = !bad[pair_index(start, q)];
            }
        } else {
            for q in 0..q_count as u32 {
                transparent[q as usize] = state_live(q);
            }
        }
        Some((transparent, pair_count, edge_count))
    }

    /// Exact safe-slice transparency and bounded radius for every residual state
    /// of one retained terminal DFA. Both answers are determined by the same
    /// shortest-counterexample product graph, so compute that graph once instead
    /// of separately solving containment and radius.
    pub(super) fn terminal_dfa_partition_all_transparency_and_repeat_radii(
        dfa: &LexerDfa,
        group: u32,
        slice: &VocabPartitionDfa,
        max_repetitions: u32,
    ) -> Option<(Vec<bool>, Vec<u32>, usize, usize)> {
        if dfa.has_epsilon_transitions() {
            return None;
        }
        let q_count = dfa.num_states();
        let p_count = slice.state_count();
        let class_count = slice.class_count();
        if q_count == 0 || p_count == 0 || class_count == 0 {
            return Some((vec![false; q_count], vec![0; q_count], 0, 0));
        }

        let live_states = (0..q_count as u32)
            .map(|state| {
                dfa.finalizers(state).contains(group as usize)
                    || dfa
                        .possible_future_group_ids(state)
                        .contains(group as usize)
            })
            .collect::<Vec<_>>();
        let state_live = |state: u32| live_states[state as usize];

        let mut class_sizes = vec![0usize; class_count];
        let mut class_representatives = vec![u8::MAX; class_count];
        for raw in 0u16..=255 {
            let byte = raw as u8;
            let class = slice.byte_class(byte) as usize;
            class_sizes[class] += 1;
            if class_representatives[class] == u8::MAX {
                class_representatives[class] = byte;
            }
        }

        let relevant_classes = (0..p_count as u32)
            .map(|p| {
                if !slice.can_reach_accepting(p) {
                    return SmallVec::<[(usize, u32, u8); 8]>::new();
                }
                class_representatives
                    .iter()
                    .enumerate()
                    .filter_map(|(class, &byte)| {
                        let target = slice.step(p, byte);
                        slice.can_reach_accepting(target).then_some((
                            class,
                            target,
                            u8::from(slice.is_accepting(target)),
                        ))
                    })
                    .collect::<SmallVec<[(usize, u32, u8); 8]>>()
            })
            .collect::<Vec<_>>();

        // Minimum completed-atom cost from each slice state to an accepting
        // state. Every byte in one slice class has the same target, so one
        // representative per class is exact here.
        let mut slice_reverse = vec![SmallVec::<[(u32, u8); 8]>::new(); p_count];
        for p in 0..p_count as u32 {
            let mut seen = SmallVec::<[u32; 16]>::new();
            for &byte in &class_representatives {
                let target = slice.step(p, byte);
                if target as usize >= p_count || seen.contains(&target) {
                    continue;
                }
                seen.push(target);
                slice_reverse[target as usize].push((p, u8::from(slice.is_accepting(target))));
            }
        }
        let mut min_to_accept = vec![u32::MAX; p_count];
        let mut zero_one = VecDeque::<u32>::new();
        for p in 0..p_count as u32 {
            if slice.is_accepting(p) {
                min_to_accept[p as usize] = 0;
                zero_one.push_back(p);
            }
        }
        while let Some(target) = zero_one.pop_front() {
            let base = min_to_accept[target as usize];
            for &(source, cost) in &slice_reverse[target as usize] {
                let candidate = base.saturating_add(u32::from(cost));
                if candidate < min_to_accept[source as usize] {
                    min_to_accept[source as usize] = candidate;
                    if cost == 0 {
                        zero_one.push_front(source);
                    } else {
                        zero_one.push_back(source);
                    }
                }
            }
        }

        let mut covered_by_q = Vec::<Vec<usize>>::with_capacity(q_count);
        let mut targets_by_q = Vec::<Vec<SmallVec<[u32; 4]>>>::with_capacity(q_count);
        for q in 0..q_count as u32 {
            let mut covered = vec![0usize; class_count];
            let mut targets = (0..class_count)
                .map(|_| SmallVec::<[u32; 4]>::new())
                .collect::<Vec<_>>();
            if state_live(q) {
                for (byte, target) in dfa.transitions(q) {
                    if !state_live(target) {
                        continue;
                    }
                    let class = slice.byte_class(byte) as usize;
                    covered[class] += 1;
                    if !targets[class].contains(&target) {
                        targets[class].push(target);
                    }
                }
            }
            covered_by_q.push(covered);
            targets_by_q.push(targets);
        }

        let pair_count = p_count.saturating_mul(q_count);
        let index = |p: u32, q: u32| p as usize * q_count + q as usize;
        let mut reverse = vec![SmallVec::<[(u32, u8); 8]>::new(); pair_count];
        let mut distance = vec![u32::MAX; pair_count];
        let mut max_seed_distance = 0u32;
        let mut edge_count = 0usize;

        for p in 0..p_count as u32 {
            if !slice.can_reach_accepting(p) {
                continue;
            }
            for q in 0..q_count as u32 {
                let current = index(p, q);
                if !state_live(q) {
                    distance[current] = 0;
                    continue;
                }
                for &(class, p_target, enter_cost) in &relevant_classes[p as usize] {
                    if covered_by_q[q as usize][class] != class_sizes[class] {
                        let completion = min_to_accept[p_target as usize];
                        if completion != u32::MAX {
                            let candidate = u32::from(enter_cost).saturating_add(completion);
                            if candidate < distance[current] {
                                distance[current] = candidate;
                                max_seed_distance = max_seed_distance.max(candidate);
                            }
                        }
                    }
                    for &q_target in &targets_by_q[q as usize][class] {
                        let target = index(p_target, q_target);
                        reverse[target].push((current as u32, enter_cost));
                        edge_count = edge_count.saturating_add(1);
                    }
                }
            }
        }

        // Product edges are 0/1-weighted. When the counterexample seeds are too,
        // 0-1 BFS avoids heap traffic; larger seed distances still use Dijkstra.
        if max_seed_distance <= 1 {
            let mut queue = VecDeque::<(u32, u32)>::new();
            for (state, &dist) in distance.iter().enumerate() {
                match dist {
                    0 => queue.push_front((state as u32, 0)),
                    1 => queue.push_back((state as u32, 1)),
                    _ => {}
                }
            }
            while let Some((target, dist)) = queue.pop_front() {
                if distance[target as usize] != dist {
                    continue;
                }
                for &(pred, cost) in &reverse[target as usize] {
                    let candidate = dist.saturating_add(u32::from(cost));
                    if candidate < distance[pred as usize] {
                        distance[pred as usize] = candidate;
                        if cost == 0 {
                            queue.push_front((pred, candidate));
                        } else {
                            queue.push_back((pred, candidate));
                        }
                    }
                }
            }
        } else {
            let mut heap = BinaryHeap::<(Reverse<u32>, u32)>::new();
            for (state, &dist) in distance.iter().enumerate() {
                if dist != u32::MAX {
                    heap.push((Reverse(dist), state as u32));
                }
            }
            while let Some((Reverse(dist), target)) = heap.pop() {
                if distance[target as usize] != dist {
                    continue;
                }
                for &(pred, cost) in &reverse[target as usize] {
                    let candidate = dist.saturating_add(u32::from(cost));
                    if candidate < distance[pred as usize] {
                        distance[pred as usize] = candidate;
                        heap.push((Reverse(candidate), pred));
                    }
                }
            }
        }

        let start = slice.start_state();
        let start_can_accept = slice.can_reach_accepting(start);
        let transparent = (0..q_count as u32)
            .map(|q| {
                if start_can_accept {
                    state_live(q) && distance[index(start, q)] == u32::MAX
                } else {
                    state_live(q)
                }
            })
            .collect::<Vec<_>>();
        let radii = (0..q_count as u32)
            .map(|q| {
                if !state_live(q) {
                    0
                } else {
                    match distance[index(start, q)] {
                        u32::MAX => max_repetitions,
                        first_counterexample => {
                            first_counterexample.saturating_sub(1).min(max_repetitions)
                        }
                    }
                }
            })
            .collect::<Vec<_>>();
        Some((transparent, radii, pair_count, edge_count))
    }

    /// Build exact compact master-slice and bounded-radius certificates from
    /// the partitioned lexer's retained terminal-residual coordinates. The
    /// result contains no terminal quotient transition matrix and is therefore
    /// cheap to transfer and restore.
    pub(crate) fn prepare_master_provers_from_residual_coordinates(
        &mut self,
        tokenizer: &Tokenizer,
        source_state_count: usize,
        safe_plus: &VocabPartitionDfa,
        whitespace: &VocabPartitionDfa,
        safe_slice_token_bytes: U8Set,
        max_safe_chars: u16,
    ) -> Option<(usize, usize, usize)> {
        let coordinates = tokenizer.terminal_residual_coordinates()?;
        if source_state_count == 0 || coordinates.len() == 0 {
            return Some((0, 0, 0));
        }
        let required_bytes = |dfa: &VocabPartitionDfa| {
            let mut bytes = U8Set::empty();
            for raw in 0u16..=255 {
                let byte = raw as u8;
                let used = (0..dfa.state_count() as u32).any(|state| {
                    dfa.can_reach_accepting(state) && dfa.can_reach_accepting(dfa.step(state, byte))
                });
                if used {
                    bytes.insert(byte);
                }
            }
            bytes
        };
        let safe_plus_required = required_bytes(safe_plus);
        let whitespace_required = required_bytes(whitespace);

        struct DirectPreparedPair {
            terminal: TerminalID,
            slice_slot: usize,
            transparent: Vec<bool>,
            product_pairs: usize,
            edges: usize,
        }

        struct DirectSafePreparedPair {
            terminal: TerminalID,
            transparent: Option<Vec<bool>>,
            radii: Option<Vec<u32>>,
            product_pairs: usize,
            edges: usize,
        }

        let candidates = (0..tokenizer.num_terminals())
            .filter(|&terminal| {
                tokenizer
                    .terminal_byte_support(terminal)
                    .is_some_and(|support| safe_slice_token_bytes.is_subset(&support))
                    && coordinates.terminal_dfa_and_group(terminal).is_some()
            })
            .collect::<Vec<_>>();
        let verify_combined =
            std::env::var_os("GLRMASK_VERIFY_DIRECT_RESIDUAL_MASTER_PROVERS").is_some();
        let build_whitespace = || {
            candidates
                .par_iter()
                .copied()
                .filter(|&terminal| {
                    tokenizer
                        .terminal_byte_support(terminal)
                        .is_some_and(|support| whitespace_required.is_subset(&support))
                })
                .filter_map(|terminal| {
                    let (dfa, group) = coordinates.terminal_dfa_and_group(terminal)?;
                    let (transparent, product_pairs, edges) =
                        Self::terminal_dfa_partition_all_transparent_states(
                            dfa, group, whitespace,
                        )?;
                    Some(DirectPreparedPair {
                        terminal,
                        slice_slot: Self::PREPARED_WHITESPACE_SLOT,
                        transparent,
                        product_pairs,
                        edges,
                    })
                })
                .collect::<Vec<_>>()
        };
        let build_safe = || {
            candidates
                .par_iter()
                .copied()
                .filter_map(|terminal| {
                    let (dfa, group) = coordinates.terminal_dfa_and_group(terminal)?;
                    let safe_supported = tokenizer
                        .terminal_byte_support(terminal)
                        .is_some_and(|support| safe_plus_required.is_subset(&support));
                    if max_safe_chars == 0 {
                        let (transparent, product_pairs, edges) =
                            Self::terminal_dfa_partition_all_transparent_states(
                                dfa, group, safe_plus,
                            )?;
                        return Some(DirectSafePreparedPair {
                            terminal,
                            transparent: safe_supported.then_some(transparent),
                            radii: None,
                            product_pairs,
                            edges,
                        });
                    }
                    let (transparent, radii, product_pairs, edges) =
                        Self::terminal_dfa_partition_all_transparency_and_repeat_radii(
                            dfa,
                            group,
                            safe_plus,
                            u32::from(max_safe_chars),
                        )?;
                    if verify_combined && safe_supported {
                        let (reference, _, _) = Self::terminal_dfa_partition_all_transparent_states(
                            dfa, group, safe_plus,
                        )?;
                        assert_eq!(
                            transparent, reference,
                            "combined direct residual safe+ solver disagrees for terminal {terminal}"
                        );
                    }
                    Some(DirectSafePreparedPair {
                        terminal,
                        transparent: safe_supported.then_some(transparent),
                        radii: Some(radii),
                        product_pairs,
                        edges,
                    })
                })
                .collect::<Vec<_>>()
        };
        let (proofs, safe_jobs) = rayon::join(build_whitespace, build_safe);

        let terminal_count = tokenizer.num_terminals();
        let mut by_terminal_slot = (0..terminal_count)
            .map(|_| [None::<Vec<bool>>, None::<Vec<bool>>])
            .collect::<Vec<_>>();
        let mut product_pairs = 0usize;
        let mut edges = 0usize;
        for proof in proofs {
            product_pairs = product_pairs.saturating_add(proof.product_pairs);
            edges = edges.saturating_add(proof.edges);
            by_terminal_slot[proof.terminal as usize][proof.slice_slot] = Some(proof.transparent);
        }
        let mut radius_by_terminal = (0..terminal_count)
            .map(|_| None::<Vec<u32>>)
            .collect::<Vec<_>>();
        for proof in safe_jobs {
            product_pairs = product_pairs.saturating_add(proof.product_pairs);
            edges = edges.saturating_add(proof.edges);
            if let Some(transparent) = proof.transparent {
                by_terminal_slot[proof.terminal as usize][Self::PREPARED_SAFE_PLUS_SLOT] =
                    Some(transparent);
            }
            if let Some(radii) = proof.radii {
                radius_by_terminal[proof.terminal as usize] = Some(radii);
            }
        }
        let mut safe_plus_complete_terminals = candidates
            .iter()
            .copied()
            .filter(|terminal| {
                by_terminal_slot[*terminal as usize][Self::PREPARED_SAFE_PLUS_SLOT].is_some()
                    && radius_by_terminal[*terminal as usize].is_some()
            })
            .collect::<Vec<_>>();
        safe_plus_complete_terminals.sort_unstable();
        safe_plus_complete_terminals.dedup();

        let row_count = source_state_count * Self::PREPARED_PROOF_SLOT_COUNT;
        let mut positive_rows = vec![SmallVec::<[TerminalID; 4]>::new(); row_count];
        let mut coverage_rows = vec![SmallVec::<[TerminalID; 4]>::new(); row_count];
        let mut radius_rows = vec![SmallVec::<[(TerminalID, u16); 4]>::new(); source_state_count];
        let source_limit = source_state_count.min(coordinates.len());
        positive_rows[..source_limit * Self::PREPARED_PROOF_SLOT_COUNT]
            .par_chunks_mut(Self::PREPARED_PROOF_SLOT_COUNT)
            .zip(
                coverage_rows[..source_limit * Self::PREPARED_PROOF_SLOT_COUNT]
                    .par_chunks_mut(Self::PREPARED_PROOF_SLOT_COUNT),
            )
            .zip(radius_rows[..source_limit].par_iter_mut())
            .enumerate()
            .for_each(|(source, ((positive_rows, coverage_rows), radius_row))| {
                let Some(entries) = coordinates.row(source as u32) else {
                    return;
                };
                for &(terminal, residual) in entries {
                    let proof_slots = &by_terminal_slot[terminal as usize];
                    for slice_slot in 0..Self::PREPARED_PROOF_SLOT_COUNT {
                        let Some(transparent) = proof_slots[slice_slot].as_ref() else {
                            continue;
                        };
                        coverage_rows[slice_slot].push(terminal);
                        if transparent.get(residual as usize).copied().unwrap_or(false) {
                            positive_rows[slice_slot].push(terminal);
                        }
                    }
                    if let Some(radii) = radius_by_terminal[terminal as usize].as_ref() {
                        let radius = radii
                            .get(residual as usize)
                            .copied()
                            .unwrap_or(0)
                            .min(u32::from(max_safe_chars))
                            as u16;
                        if radius != 0 {
                            radius_row.push((terminal, radius));
                        }
                    }
                }
            });

        let positive_entry_count = positive_rows.iter().map(SmallVec::len).sum::<usize>();
        let parallel_intern = candidates.len() >= 64 && rayon::current_num_threads() > 1;
        let ((positive, coverage), radius) = if parallel_intern {
            rayon::join(
                || {
                    rayon::join(
                        || intern_terminal_rows(positive_rows),
                        || intern_terminal_rows(coverage_rows),
                    )
                },
                || intern_radius_rows(radius_rows),
            )
        } else {
            (
                (
                    intern_terminal_rows(positive_rows),
                    intern_terminal_rows(coverage_rows),
                ),
                intern_radius_rows(radius_rows),
            )
        };
        let (positive_row_ids, positive_offsets, positive_terminals) = positive;
        let (coverage_row_ids, coverage_offsets, coverage_terminals) = coverage;
        let (radius_row_ids, radius_offsets, radius_entries) = radius;

        self.prepared_master_prover_row_ids = Arc::from(positive_row_ids);
        self.prepared_master_prover_offsets = Arc::from(positive_offsets);
        self.prepared_master_prover_terminals = Arc::from(positive_terminals);
        self.prepared_master_coverage_row_ids = Arc::from(coverage_row_ids);
        self.prepared_master_coverage_offsets = Arc::from(coverage_offsets);
        self.prepared_master_coverage_terminals = Arc::from(coverage_terminals);
        self.prepared_safe_plus_complete_terminals = Arc::from(safe_plus_complete_terminals);
        self.prepared_safe_radius_row_ids = Arc::from(radius_row_ids);
        self.prepared_safe_radius_offsets = Arc::from(radius_offsets);
        self.prepared_safe_radius_entries = Arc::from(radius_entries);
        Some((positive_entry_count, product_pairs, edges))
    }

    /// Build positive-only parser-independent master-slice certificates for
    /// every exact source TSID represented by the currently prepared terminal
    /// quotients. One `(terminal quotient, slice)` product graph is solved once
    /// for all quotient residual states; source TSIDs are then fanned out
    /// through the quotient's exact source->residual mapping.
    pub(crate) fn prepare_master_provers_all_sources(
        &mut self,
        tokenizer: &Tokenizer,
        source_state_count: usize,
        include_safe_radii: bool,
    ) -> (usize, usize) {
        let Some(safe_plus) = self
            .llg_slice_leftovers
            .iter()
            .find(|slice| slice.cache_id() == 0)
            .cloned()
        else {
            return (0, 0);
        };
        let Some(whitespace) = self
            .llg_slice_leftovers
            .iter()
            .find(|slice| slice.cache_id() == 3)
            .cloned()
        else {
            return (0, 0);
        };
        let mut proof_languages = Vec::<(usize, Arc<VocabPartitionDfa>, U8Set)>::new();
        let required_bytes = |dfa: &VocabPartitionDfa| {
            let mut bytes = U8Set::empty();
            for raw in 0u16..=255 {
                let byte = raw as u8;
                let used = (0..dfa.state_count() as u32).any(|state| {
                    dfa.can_reach_accepting(state) && dfa.can_reach_accepting(dfa.step(state, byte))
                });
                if used {
                    bytes.insert(byte);
                }
            }
            bytes
        };
        proof_languages.push((
            Self::PREPARED_SAFE_PLUS_SLOT,
            Arc::clone(&safe_plus.dfa),
            required_bytes(safe_plus.dfa()),
        ));
        proof_languages.push((
            Self::PREPARED_WHITESPACE_SLOT,
            Arc::clone(&whitespace.dfa),
            required_bytes(whitespace.dfa()),
        ));
        let quotients = self.active_projected_terminal_quotients();
        if quotients.is_empty() || source_state_count == 0 {
            self.prepared_master_prover_row_ids = Arc::from(Vec::<u32>::new());
            self.prepared_master_prover_offsets = Arc::from(Vec::<u32>::new());
            self.prepared_master_prover_terminals = Arc::from(Vec::<TerminalID>::new());
            self.prepared_master_coverage_row_ids = Arc::from(Vec::<u32>::new());
            self.prepared_master_coverage_offsets = Arc::from(Vec::<u32>::new());
            self.prepared_master_coverage_terminals = Arc::from(Vec::<TerminalID>::new());
            self.prepared_safe_plus_complete_terminals = Arc::from(Vec::<TerminalID>::new());
            self.prepared_safe_radius_row_ids = Arc::from(Vec::<u32>::new());
            self.prepared_safe_radius_offsets = Arc::from(Vec::<u32>::new());
            self.prepared_safe_radius_entries = Arc::from(Vec::<(TerminalID, u16)>::new());
            return (0, 0);
        }

        struct PreparedPair {
            terminal: TerminalID,
            slice_slot: usize,
            certified_sources: Vec<u32>,
            product_pairs: usize,
            elapsed_ns: u64,
            verified: usize,
            mismatches: usize,
            unknown: usize,
        }
        let verify = std::env::var_os("GLRMASK_VERIFY_PREPARED_MASTER_PROVERS").is_some();
        let jobs = quotients
            .par_iter()
            .flat_map_iter(|(terminal, quotient)| {
                proof_languages
                    .iter()
                    .filter_map(move |(slot, slice, required)| {
                        tokenizer
                            .terminal_byte_support(*terminal)
                            .is_some_and(|support| required.is_subset(&support))
                            .then_some((*terminal, Arc::clone(quotient), *slot, Arc::clone(slice)))
                    })
            })
            .map(|(terminal, quotient, slice_slot, slice)| {
                let started = std::time::Instant::now();
                let (transparent, product_pairs) = Self::terminal_partition_all_transparent_states(
                    quotient.as_ref(),
                    slice.as_ref(),
                );
                let mut verified = 0usize;
                let mut mismatches = 0usize;
                let mut unknown = 0usize;
                if verify {
                    for (&source, &projected) in quotient
                        .projected_source_states()
                        .iter()
                        .zip(quotient.projected_states_for_sources())
                    {
                        let batch = transparent
                            .get(projected as usize)
                            .copied()
                            .unwrap_or(false);
                        match Self::terminal_partition_product_is_transparent(
                            quotient.as_ref(),
                            slice.as_ref(),
                            source,
                            200_000,
                        ) {
                            Some(reference) => {
                                verified += 1;
                                mismatches += usize::from(reference != batch);
                            }
                            None => unknown += 1,
                        }
                    }
                }
                let certified_sources = quotient
                    .projected_source_states()
                    .iter()
                    .copied()
                    .zip(quotient.projected_states_for_sources().iter().copied())
                    .filter_map(|(source, projected)| {
                        ((source as usize) < source_state_count
                            && transparent
                                .get(projected as usize)
                                .copied()
                                .unwrap_or(false))
                        .then_some(source)
                    })
                    .collect::<Vec<_>>();
                PreparedPair {
                    terminal,
                    slice_slot,
                    certified_sources,
                    product_pairs,
                    elapsed_ns: started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
                    verified,
                    mismatches,
                    unknown,
                }
            })
            .collect::<Vec<_>>();

        let mut rows = vec![
            SmallVec::<[TerminalID; 4]>::new();
            source_state_count * Self::PREPARED_PROOF_SLOT_COUNT
        ];
        let mut product_pairs = 0usize;
        let mut pair_work_ns = 0u64;
        let mut max_pair_ns = 0u64;
        let mut verified = 0usize;
        let mut mismatches = 0usize;
        let mut unknown = 0usize;
        for job in jobs {
            product_pairs = product_pairs.saturating_add(job.product_pairs);
            pair_work_ns = pair_work_ns.saturating_add(job.elapsed_ns);
            max_pair_ns = max_pair_ns.max(job.elapsed_ns);
            verified += job.verified;
            mismatches += job.mismatches;
            unknown += job.unknown;
            for source in job.certified_sources {
                rows[source as usize * Self::PREPARED_PROOF_SLOT_COUNT + job.slice_slot]
                    .push(job.terminal);
            }
        }
        if std::env::var_os("GLRMASK_PROFILE_PREPARED_MASTER_PROVERS").is_some() {
            eprintln!(
                "[glrmask/profile][prepared_master_pair_work] jobs={} work_ms={:.3} max_pair_ms={:.3}",
                quotients.len() * 2,
                pair_work_ns as f64 / 1e6,
                max_pair_ns as f64 / 1e6,
            );
        }
        if verify {
            eprintln!(
                "[glrmask/profile][prepared_master_verify] verified={} mismatches={} unknown={}",
                verified, mismatches, unknown,
            );
            assert_eq!(
                mismatches, 0,
                "batched master proof disagrees with per-source exact proof"
            );
        }

        let entry_count = rows.iter().map(SmallVec::len).sum::<usize>();
        let (row_ids, offsets, terminals) = intern_terminal_rows(rows);
        if include_safe_radii {
            let max_radius = self.llg_master_max_safe_chars;
            let radius_jobs = quotients
                .par_iter()
                .map(|(terminal, quotient)| {
                    let radii = Self::terminal_partition_all_repeat_radii(
                        quotient.as_ref(),
                        safe_plus.dfa(),
                        u32::from(max_radius),
                    );
                    (*terminal, Arc::clone(quotient), radii)
                })
                .collect::<Vec<_>>();
            let mut radius_rows =
                vec![SmallVec::<[(TerminalID, u16); 4]>::new(); source_state_count];
            for (terminal, quotient, radii) in radius_jobs {
                for (&source, &projected) in quotient
                    .projected_source_states()
                    .iter()
                    .zip(quotient.projected_states_for_sources())
                {
                    if source as usize >= source_state_count {
                        continue;
                    }
                    let radius = radii
                        .get(projected as usize)
                        .copied()
                        .unwrap_or(0)
                        .min(u32::from(max_radius)) as u16;
                    if radius != 0 {
                        radius_rows[source as usize].push((terminal, radius));
                    }
                }
            }
            let (radius_row_ids, radius_offsets, radius_entries) = intern_radius_rows(radius_rows);
            self.prepared_safe_radius_row_ids = Arc::from(radius_row_ids);
            self.prepared_safe_radius_offsets = Arc::from(radius_offsets);
            self.prepared_safe_radius_entries = Arc::from(radius_entries);
        } else {
            self.prepared_safe_radius_row_ids = Arc::from(Vec::<u32>::new());
            self.prepared_safe_radius_offsets = Arc::from(Vec::<u32>::new());
            self.prepared_safe_radius_entries = Arc::from(Vec::<(TerminalID, u16)>::new());
        }
        // Keep the borrow of the active quotient sidecar alive through the
        // optional radius analysis above, then publish the compact prepared
        // rows only after all quotient reads are complete.
        self.prepared_master_prover_row_ids = Arc::from(row_ids);
        self.prepared_master_prover_offsets = Arc::from(offsets);
        self.prepared_master_prover_terminals = Arc::from(terminals);
        self.prepared_master_coverage_row_ids = Arc::from(Vec::<u32>::new());
        self.prepared_master_coverage_offsets = Arc::from(Vec::<u32>::new());
        self.prepared_master_coverage_terminals = Arc::from(Vec::<TerminalID>::new());
        self.prepared_safe_plus_complete_terminals = Arc::from(Vec::<TerminalID>::new());
        (entry_count, product_pairs)
    }

    /// Exact all-start-state analogue of `projected_terminal_slice_repeat_radius`.
    /// `distance[p,q]` is the minimum number of additional completed slice
    /// atoms in a completed slice word whose prefix first leaves the live
    /// terminal residual. Solving this reverse shortest-path problem once gives
    /// the bounded radius for every quotient residual state simultaneously.
    pub(super) fn terminal_partition_all_repeat_radii(
        quotient: &TerminalProjectedQuotient,
        slice: &VocabPartitionDfa,
        max_repetitions: u32,
    ) -> Vec<u32> {
        let q_count = quotient.projected_state_count();
        let p_count = slice.state_count();
        if q_count == 0 || p_count == 0 || max_repetitions == 0 {
            return vec![0; q_count];
        }

        let mut representatives = SmallVec::<[(u8, u8, u8); 32]>::new();
        for raw in 0u16..=255 {
            let byte = raw as u8;
            let p_class = slice.byte_class(byte);
            let q_class = quotient.projected_byte_class(byte);
            if !representatives
                .iter()
                .any(|&(p, q, _)| p == p_class && q == q_class)
            {
                representatives.push((p_class, q_class, byte));
            }
        }

        // Minimum completed-atom cost from each slice state to some accepting
        // state. Edge cost is one exactly when the target state completes an
        // atom; UTF-8 continuation transitions therefore cost zero.
        let mut slice_reverse = vec![SmallVec::<[(u32, u8); 8]>::new(); p_count];
        for p in 0..p_count as u32 {
            let mut seen = SmallVec::<[u32; 16]>::new();
            for &(_, _, byte) in &representatives {
                let target = slice.step(p, byte);
                if target as usize >= p_count || seen.contains(&target) {
                    continue;
                }
                seen.push(target);
                slice_reverse[target as usize].push((p, u8::from(slice.is_accepting(target))));
            }
        }
        let mut min_to_accept = vec![u32::MAX; p_count];
        let mut zero_one = VecDeque::<u32>::new();
        for p in 0..p_count as u32 {
            if slice.is_accepting(p) {
                min_to_accept[p as usize] = 0;
                zero_one.push_back(p);
            }
        }
        while let Some(target) = zero_one.pop_front() {
            let base = min_to_accept[target as usize];
            for &(source, cost) in &slice_reverse[target as usize] {
                let candidate = base.saturating_add(u32::from(cost));
                if candidate < min_to_accept[source as usize] {
                    min_to_accept[source as usize] = candidate;
                    if cost == 0 {
                        zero_one.push_front(source);
                    } else {
                        zero_one.push_back(source);
                    }
                }
            }
        }

        let pair_count = p_count * q_count;
        let index = |p: u32, q: u32| p as usize * q_count + q as usize;
        let mut reverse = vec![SmallVec::<[(u32, u8); 8]>::new(); pair_count];
        let mut distance = vec![u32::MAX; pair_count];
        let mut heap = BinaryHeap::<(Reverse<u32>, u32)>::new();

        for p in 0..p_count as u32 {
            if !slice.can_reach_accepting(p) {
                continue;
            }
            for q in 0..q_count as u32 {
                let current = index(p, q);
                let q_live = quotient.projected_state_is_accepting(q)
                    || quotient.projected_state_has_future(q);
                if !q_live {
                    distance[current] = 0;
                    heap.push((Reverse(0), current as u32));
                    continue;
                }
                for &(_, q_class, byte) in &representatives {
                    let p_target = slice.step(p, byte);
                    if !slice.can_reach_accepting(p_target) {
                        continue;
                    }
                    let enter_cost = u32::from(slice.is_accepting(p_target));
                    let q_target = quotient.projected_step_class(q, q_class);
                    let target_live = q_target.is_some_and(|target| {
                        quotient.projected_state_is_accepting(target)
                            || quotient.projected_state_has_future(target)
                    });
                    if !target_live {
                        let completion = min_to_accept[p_target as usize];
                        if completion != u32::MAX {
                            let candidate = enter_cost.saturating_add(completion);
                            if candidate < distance[current] {
                                distance[current] = candidate;
                                heap.push((Reverse(candidate), current as u32));
                            }
                        }
                        continue;
                    }
                    let target = index(p_target, q_target.expect("live target exists"));
                    reverse[target].push((current as u32, enter_cost as u8));
                }
            }
        }

        while let Some((Reverse(dist), target)) = heap.pop() {
            if distance[target as usize] != dist {
                continue;
            }
            for &(pred, cost) in &reverse[target as usize] {
                let candidate = dist.saturating_add(u32::from(cost));
                if candidate < distance[pred as usize] {
                    distance[pred as usize] = candidate;
                    heap.push((Reverse(candidate), pred));
                }
            }
        }

        let start = slice.start_state();
        (0..q_count as u32)
            .map(|q| {
                if !quotient.projected_state_is_accepting(q)
                    && !quotient.projected_state_has_future(q)
                {
                    0
                } else {
                    match distance[index(start, q)] {
                        u32::MAX => max_repetitions,
                        first_counterexample => {
                            first_counterexample.saturating_sub(1).min(max_repetitions)
                        }
                    }
                }
            })
            .collect()
    }

    /// Exact all-start-state version of `terminal_partition_product_is_transparent`.
    /// A product pair `(p,q)` is bad when some byte that keeps `P` on an
    /// accepting-prefix path either has no live `Q` successor or reaches a bad
    /// product pair. This computes the least backwards closure of those bad
    /// pairs once, yielding the answer for every quotient state simultaneously.
    pub(super) fn terminal_partition_all_transparent_states(
        quotient: &TerminalProjectedQuotient,
        partition: &VocabPartitionDfa,
    ) -> (Vec<bool>, usize) {
        let q_count = quotient.projected_state_count();
        let p_count = partition.state_count();
        if q_count == 0 || p_count == 0 {
            return (vec![false; q_count], 0);
        }

        // Refine the two exact byte partitions once. One representative is
        // sufficient for each `(P class, Q class)` pair because both automata
        // transition identically for every byte in that pair.
        let mut representatives = SmallVec::<[(u8, u8, u8); 32]>::new();
        for raw in 0u16..=255 {
            let byte = raw as u8;
            let p_class = partition.byte_class(byte);
            let q_class = quotient.projected_byte_class(byte);
            if !representatives
                .iter()
                .any(|&(p, q, _)| p == p_class && q == q_class)
            {
                representatives.push((p_class, q_class, byte));
            }
        }

        let pair_count = p_count.saturating_mul(q_count);
        let pair_index = |p: u32, q: u32| p as usize * q_count + q as usize;
        let mut predecessors = vec![SmallVec::<[u32; 8]>::new(); pair_count];
        let mut bad = vec![false; pair_count];
        let mut queue = VecDeque::<u32>::new();

        for p in 0..p_count as u32 {
            if !partition.can_reach_accepting(p) {
                continue;
            }
            for q in 0..q_count as u32 {
                let current = pair_index(p, q);
                if !quotient.projected_state_is_accepting(q)
                    && !quotient.projected_state_has_future(q)
                {
                    bad[current] = true;
                    queue.push_back(current as u32);
                    continue;
                }

                let mut immediate_bad = false;
                for &(_, q_class, byte) in &representatives {
                    let p_target = partition.step(p, byte);
                    if !partition.can_reach_accepting(p_target) {
                        continue;
                    }
                    let Some(q_target) = quotient.projected_step_class(q, q_class) else {
                        immediate_bad = true;
                        break;
                    };
                    if !quotient.projected_state_is_accepting(q_target)
                        && !quotient.projected_state_has_future(q_target)
                    {
                        immediate_bad = true;
                        break;
                    }
                    let target = pair_index(p_target, q_target);
                    predecessors[target].push(current as u32);
                }
                if immediate_bad {
                    bad[current] = true;
                    queue.push_back(current as u32);
                }
            }
        }

        while let Some(target) = queue.pop_front() {
            for &pred in &predecessors[target as usize] {
                if !bad[pred as usize] {
                    bad[pred as usize] = true;
                    queue.push_back(pred);
                }
            }
        }

        let start = partition.start_state();
        let mut transparent = vec![false; q_count];
        if partition.can_reach_accepting(start) {
            for q in 0..q_count as u32 {
                transparent[q as usize] = !bad[pair_index(start, q)];
            }
        } else {
            for q in 0..q_count as u32 {
                transparent[q as usize] = quotient.projected_state_is_accepting(q)
                    || quotient.projected_state_has_future(q);
            }
        }
        (transparent, pair_count)
    }
}
