//! Direct regular parser admission, cached frontiers, and exact advancement.

use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::labels::encode_positive_label;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::table::Action;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;
use crate::runtime::artifact::Constraint;
use crate::runtime::artifact::DenseAcceptanceRows;
use crate::runtime::artifact::DenseWeightMaskCache;
use crate::runtime::artifact::DenseWords;
use crate::runtime::artifact::DirectRegularDynamicFrontierCacheEntry;
use crate::runtime::artifact::DirectRegularDynamicHotFrontier;
use crate::runtime::artifact::DirectRegularParserStateAcceptance;
use crate::runtime::artifact::DirectRegularTerminalSupport;
use crate::runtime::artifact::DirectRegularWideFrontierAcceptance;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::sync::Arc;
use super::RuntimeWeightRef;

impl Constraint {


    #[inline]
    pub(crate) fn uses_sparse_direct_regular_runtime(&self) -> bool {
        self.direct_regular_automaton.is_some()
            && self.table.num_rules == 0
            && self.table.action.is_empty()
    }

    fn direct_regular_frontier_advances(
        &self,
        parser_states: &Arc<[u32]>,
    ) -> Arc<[(TerminalID, Arc<[u32]>)]> {
        let Some(automaton) = self.direct_regular_automaton.as_ref() else {
            return Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new());
        };
        let mut seen = vec![false; automaton.states.len()];
        let mut stack = Vec::with_capacity(parser_states.len());
        for &parser_state in parser_states.iter() {
            let Some(raw_state) = parser_state.checked_sub(1) else {
                return Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new());
            };
            if raw_state as usize >= automaton.states.len() {
                return Arc::from(Vec::<(TerminalID, Arc<[u32]>)>::new());
            }
            stack.push(raw_state);
        }

        let mut targets_by_terminal = BTreeMap::<TerminalID, Vec<u32>>::new();
        while let Some(raw_state) = stack.pop() {
            let index = raw_state as usize;
            if seen[index] {
                continue;
            }
            seen[index] = true;
            let state = &automaton.states[index];
            stack.extend(state.epsilons.iter().copied());
            for (&terminal, targets) in &state.transitions {
                let entry = targets_by_terminal.entry(terminal).or_default();
                entry.extend(targets.iter().map(|target| target + 1));
            }
        }

        targets_by_terminal
            .into_iter()
            .filter_map(|(terminal, mut targets)| {
                targets.sort_unstable();
                targets.dedup();
                (!targets.is_empty()).then(|| {
                    let targets: Arc<[u32]> = if targets.as_slice() == parser_states.as_ref() {
                        Arc::clone(parser_states)
                    } else {
                        targets.into()
                    };
                    (terminal, targets)
                })
            })
            .collect::<Vec<_>>()
            .into()
    }

    fn direct_regular_dynamic_hot_frontier_summary(
        &self,
        frontier_states: Arc<[u32]>,
        advance_by_terminal: Arc<[(TerminalID, Arc<[u32]>)]>,
    ) -> DirectRegularDynamicHotFrontier {
        let mut actionable_terminals =
            crate::ds::bitset::BitSet::new(self.table.num_terminals as usize);
        for &(terminal, _) in advance_by_terminal.iter() {
            actionable_terminals.set(terminal as usize);
        }
        let empty_acc_frontier = ParserGSS::from_sorted_unique_single_value_stacks(
            &frontier_states,
            TerminalsDisallowed::new(),
        );
        DirectRegularDynamicHotFrontier {
            frontier_states,
            empty_acc_frontier,
            actionable_terminals,
            advance_by_terminal,
        }
    }

    pub(super) fn compute_direct_regular_dynamic_hot_frontiers(
        &self,
        support: &DirectRegularTerminalSupport,
    ) -> Vec<DirectRegularDynamicHotFrontier> {
        const MIN_AUTOMATON_STATES: usize = 16_384;
        const MIN_PRIMARY_WORK: u64 = 64;
        const MIN_SECONDARY_FRONTIER_STATES: usize = 64;
        if !self.uses_sparse_direct_regular_runtime() {
            return Vec::new();
        }
        let Some(automaton) = self.direct_regular_automaton.as_ref() else {
            return Vec::new();
        };
        if automaton.states.len() < MIN_AUTOMATON_STATES {
            return Vec::new();
        }

        let mut parents = vec![Vec::<u32>::new(); automaton.states.len()];
        let mut remaining_children = Vec::<u32>::with_capacity(automaton.states.len());
        let mut queue = std::collections::VecDeque::<u32>::new();
        for (source, state) in automaton.states.iter().enumerate() {
            remaining_children.push(state.epsilons.len() as u32);
            if state.epsilons.is_empty() {
                queue.push_back(source as u32);
            }
            for &child in &state.epsilons {
                parents[child as usize].push(source as u32);
            }
        }
        let mut transition_work = vec![0u64; automaton.states.len()];
        let mut processed = 0usize;
        while let Some(raw) = queue.pop_front() {
            let state = &automaton.states[raw as usize];
            let own = state
                .transitions
                .values()
                .map(|targets| targets.len() as u64)
                .sum::<u64>();
            transition_work[raw as usize] = state.epsilons.iter().fold(own, |work, child| {
                work.saturating_add(transition_work[*child as usize])
            });
            processed += 1;
            for &parent in &parents[raw as usize] {
                let remaining = &mut remaining_children[parent as usize];
                *remaining -= 1;
                if *remaining == 0 {
                    queue.push_back(parent);
                }
            }
        }
        if processed != automaton.states.len() {
            return Vec::new();
        }
        let Some((primary_raw, primary_work)) = transition_work
            .iter()
            .copied()
            .enumerate()
            .max_by_key(|&(state, work)| {
                (
                    support.state_terminal_count(state as u32).unwrap_or(0),
                    work,
                    state,
                )
            })
        else {
            return Vec::new();
        };
        if primary_work < MIN_PRIMARY_WORK {
            return Vec::new();
        }
        let primary_states: Arc<[u32]> = Arc::from([primary_raw as u32 + 1]);
        let primary_advances = self.direct_regular_frontier_advances(&primary_states);
        let widest_frontier = primary_advances
            .iter()
            .map(|(_, targets)| Arc::clone(targets))
            .filter(|targets| targets.len() >= MIN_SECONDARY_FRONTIER_STATES)
            .max_by_key(|targets| targets.len());

        let mut summaries = Vec::with_capacity(2);
        summaries.push(self.direct_regular_dynamic_hot_frontier_summary(
            Arc::clone(&primary_states),
            primary_advances,
        ));
        if let Some(widest_frontier) = widest_frontier
            && widest_frontier.as_ref() != primary_states.as_ref()
        {
            let advances = self.direct_regular_frontier_advances(&widest_frontier);
            summaries.push(self.direct_regular_dynamic_hot_frontier_summary(
                widest_frontier,
                advances,
            ));
        }
        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][dynamic_hot_frontiers] primary_state={} primary_support={} primary_work={} summaries={} widths={:?}",
                primary_raw + 1,
                support.state_terminal_count(primary_raw as u32).unwrap_or(0),
                primary_work,
                summaries.len(),
                summaries
                    .iter()
                    .map(|summary| summary.frontier_states.len())
                    .collect::<Vec<_>>(),
            );
        }
        summaries
    }

    fn direct_regular_dynamic_hot_frontier_for_gss(
        &self,
        gss: &ParserGSS,
    ) -> Option<&DirectRegularDynamicHotFrontier> {
        if self.direct_regular_dynamic_hot_frontiers.is_empty() || gss.max_depth() != 1 {
            return None;
        }
        let top_count = gss.top_value_count();
        if top_count == 1 {
            let state = gss.single_top_value()?;
            return self
                .direct_regular_dynamic_hot_frontiers
                .iter()
                .find(|summary| summary.frontier_states.as_ref() == [state]);
        }
        if let Some(lower_id) = gss.single_interface_lower_id()
            && let Some(summary) = self
                .direct_regular_dynamic_hot_frontiers
                .iter()
                .find(|summary| {
                    summary.frontier_states.len() == top_count
                        && summary.empty_acc_frontier.single_interface_lower_id() == Some(lower_id)
                })
        {
            return Some(summary);
        }
        if !self
            .direct_regular_dynamic_hot_frontiers
            .iter()
            .any(|summary| summary.frontier_states.len() == top_count)
        {
            return None;
        }
        let mut top_values = gss.peek_values();
        top_values.sort_unstable();
        self.direct_regular_dynamic_hot_frontiers
            .iter()
            .find(|summary| summary.frontier_states.as_ref() == top_values.as_slice())
    }

    pub(super) fn compute_direct_regular_wide_frontier_acceptance(
        &self,
    ) -> Vec<DirectRegularWideFrontierAcceptance> {
        // Loaded current-format constraints can execute exact acceptance
        // directly from packed Weight ids. These summaries are only an
        // optimization over materialized Weight objects; rebuilding them would
        // defeat the packed-load path.
        if self.packed_non_dwa_weights.is_some() {
            return Vec::new();
        }
        const MIN_FRONTIER_STATES: usize = 64;
        if self.uses_dynamic_runtime() || self.table.num_rules != 0 {
            return Vec::new();
        }

        let mut seen_frontiers = FxHashMap::<Vec<u32>, usize>::default();
        let mut parts_cache = FxHashMap::<Vec<usize>, Arc<[Weight]>>::default();
        let mut summaries = Vec::<DirectRegularWideFrontierAcceptance>::new();
        for descriptor in &self.table.direct_regular_wide_frontiers {
            let action = self
                .table
                .action(descriptor.source_state, descriptor.terminal);
            let action_origin_and_states = action.and_then(|action| match action {
                Action::ReplaceShifts(targets) => {
                    Some((targets.as_ptr() as usize, targets.to_vec()))
                }
                Action::StackShifts(shifts)
                    if shifts
                        .iter()
                        .all(|shift| shift.pop == 1 && shift.pushes.len() == 1) =>
                {
                    Some((
                        shifts.as_ptr() as usize,
                        shifts.iter().map(|shift| shift.pushes[0]).collect(),
                    ))
                }
                _ => None,
            });
            if action.is_some() && action_origin_and_states.is_none() {
                continue;
            }
            let action_origin = action_origin_and_states.as_ref().map(|(origin, _)| *origin);

            let mut states = descriptor.target_states.clone();
            states.sort_unstable();
            states.dedup();
            if states.len() < MIN_FRONTIER_STATES {
                continue;
            }
            if let Some((_, mut action_states)) = action_origin_and_states {
                action_states.sort_unstable();
                action_states.dedup();
                debug_assert_eq!(
                    states,
                    action_states,
                    "direct-regular frontier descriptor drifted from the live table action",
                );
            } else if !self.uses_sparse_direct_regular_runtime() {
                continue;
            }

            if let Some(&summary_index) = seen_frontiers.get(&states) {
                if let Some(action_origin) = action_origin {
                    summaries[summary_index].action_origins.push(action_origin);
                }
                continue;
            }
            seen_frontiers.insert(states.clone(), summaries.len());

            let mut weights = Vec::<Weight>::new();
            for &state in &states {
                let label = encode_positive_label(state);
                if let Some(weight) = self
                    .parser_top_accept
                    .get(&label)
                    .or_else(|| self.parser_top_accept.get(&DEFAULT_LABEL))
                {
                    weights.push(weight.clone());
                }
                if let Some(parts) = self
                    .parser_top_accept_parts
                    .get(&label)
                    .or_else(|| self.parser_top_accept_parts.get(&DEFAULT_LABEL))
                {
                    weights.extend(parts.iter().cloned());
                }
            }
            let mut actionable_terminals =
                crate::ds::bitset::BitSet::new(self.table.num_terminals as usize + 1);
            for &state in &states {
                if let Some(row) = self.table.advance.get(state as usize) {
                    actionable_terminals.union_with(row);
                }
            }
            for terminal in actionable_terminals.iter_ones() {
                if let Some(weight) = self
                    .direct_regular_l1_complete_by_terminal
                    .get(&(terminal as TerminalID))
                {
                    weights.push(weight.clone());
                }
            }
            weights.sort_unstable_by_key(Weight::ptr_key);
            weights.dedup_by_key(|weight| weight.ptr_key());
            let parts_key = weights.iter().map(Weight::ptr_key).collect::<Vec<_>>();
            let acceptance_parts = if let Some(cached) = parts_cache.get(&parts_key) {
                Arc::clone(cached)
            } else {
                let parts: Arc<[Weight]> = weights.into();
                parts_cache.insert(parts_key, Arc::clone(&parts));
                parts
            };
            let state_count = states.len();
            let frontier_states: Arc<[u32]> = states.into();
            let empty_acc_frontier = ParserGSS::from_sorted_unique_single_value_stacks(
                &frontier_states,
                TerminalsDisallowed::new(),
            );
            let advance_by_terminal = self.direct_regular_frontier_advances(&frontier_states);
            summaries.push(DirectRegularWideFrontierAcceptance {
                action_origins: action_origin.into_iter().collect(),
                state_count,
                actionable_terminals,
                frontier_states,
                empty_acc_frontier,
                acceptance_parts,
                dense_by_tsid: Arc::new(DenseAcceptanceRows::default()),
                advance_by_terminal,
            });
        }
        summaries
    }

    pub(super) fn compute_direct_regular_parser_state_acceptance(
        &self,
    ) -> Vec<DirectRegularParserStateAcceptance> {
        if self.packed_non_dwa_weights.is_some() {
            return Vec::new();
        }
        const MIN_L1_TERMINALS: usize = 64;
        if self.uses_dynamic_runtime()
            || self.table.num_rules != 0
            || self.direct_regular_l1_complete_by_terminal.is_empty()
        {
            return Vec::new();
        }

        let mut l1_terminals =
            crate::ds::bitset::BitSet::new(self.table.num_terminals as usize + 1);
        for &terminal in self.direct_regular_l1_complete_by_terminal.keys() {
            l1_terminals.set(terminal as usize);
        }
        let l1_count = |row: &crate::ds::bitset::BitSet| {
            row.words()
                .iter()
                .zip(l1_terminals.words())
                .map(|(left, right)| (left & right).count_ones() as usize)
                .sum::<usize>()
        };
        let max_l1_terminals = self
            .table
            .advance
            .iter()
            .map(l1_count)
            .max()
            .unwrap_or(0);
        if max_l1_terminals < MIN_L1_TERMINALS {
            return Vec::new();
        }

        let mut parts_cache = FxHashMap::<Vec<usize>, Arc<[Weight]>>::default();
        let mut summaries = Vec::new();
        for (parser_state, row) in self.table.advance.iter().enumerate() {
            if l1_count(row) != max_l1_terminals {
                continue;
            }

            let label = encode_positive_label(parser_state as u32);
            let mut weights = Vec::<Weight>::with_capacity(max_l1_terminals + 4);
            if let Some(weight) = self
                .parser_top_accept
                .get(&label)
                .or_else(|| self.parser_top_accept.get(&DEFAULT_LABEL))
            {
                weights.push(weight.clone());
            }
            if let Some(parts) = self
                .parser_top_accept_parts
                .get(&label)
                .or_else(|| self.parser_top_accept_parts.get(&DEFAULT_LABEL))
            {
                weights.extend(parts.iter().cloned());
            }
            for terminal in row.iter_ones() {
                if let Some(weight) = self
                    .direct_regular_l1_complete_by_terminal
                    .get(&(terminal as TerminalID))
                {
                    weights.push(weight.clone());
                }
            }
            weights.sort_unstable_by_key(Weight::ptr_key);
            weights.dedup_by_key(|weight| weight.ptr_key());
            let parts_key = weights.iter().map(Weight::ptr_key).collect::<Vec<_>>();
            let acceptance_parts = if let Some(cached) = parts_cache.get(&parts_key) {
                Arc::clone(cached)
            } else {
                let parts: Arc<[Weight]> = weights.into();
                parts_cache.insert(parts_key, Arc::clone(&parts));
                parts
            };
            summaries.push(DirectRegularParserStateAcceptance {
                parser_state: parser_state as u32,
                acceptance_parts,
                dense_by_tsid: Arc::new(DenseAcceptanceRows::default()),
            });
        }
        summaries
    }

    fn materialize_acceptance_parts_dense(
        acceptance_parts: &[Weight],
        dense_word_count: usize,
        tsid_count: usize,
        full_dense: &DenseWords,
        dense_cache: &DenseWeightMaskCache,
    ) -> Arc<DenseAcceptanceRows> {
        let dense_cells = tsid_count
            .checked_mul(dense_word_count)
            .expect("acceptance dense matrix size must fit usize");
        // This used to allocate one Vec per tokenizer state. Large tokenizers
        // therefore performed tens of thousands of small allocations merely to
        // materialize one direct-regular acceptance summary. Keep the same
        // exact row-major matrix in one allocation and freeze only nonempty rows
        // into the sparse runtime representation below.
        let mut by_tsid = vec![0u64; dense_cells];
        let mut row_kinds = vec![0u8; tsid_count];
        for weight in acceptance_parts {
            if weight.is_full() {
                row_kinds.fill(2);
                continue;
            }
            if weight.is_empty() {
                continue;
            }
            for (tsid_range, token_set) in weight.raw_range_values() {
                let key = Arc::as_ptr(token_set) as usize;
                let dense = dense_cache.get(&key).cloned().unwrap_or_else(|| {
                    Self::dense_words_from_internal_set_with_words(
                        token_set.as_ref(),
                        dense_word_count,
                    )
                });
                for tsid in *tsid_range.start()..=*tsid_range.end() {
                    let tsid = tsid as usize;
                    if tsid >= tsid_count {
                        continue;
                    }
                    if row_kinds[tsid] == 2 {
                        continue;
                    }
                    let start = tsid * dense_word_count;
                    let dst = &mut by_tsid[start..start + dense_word_count];
                    for (dst_word, src_word) in dst.iter_mut().zip(dense.iter()) {
                        *dst_word |= *src_word;
                    }
                    row_kinds[tsid] = 1;
                }
            }
        }

        Arc::new(DenseAcceptanceRows::new(
            dense_word_count,
            by_tsid,
            row_kinds,
            Arc::clone(full_dense),
        ))
    }

    pub(super) fn materialize_direct_regular_acceptance_rows(
        acceptance_parts: &[Arc<[Weight]>],
        dense_word_count: usize,
        tsid_count: usize,
        full_dense: &DenseWords,
        dense_cache: &DenseWeightMaskCache,
    ) -> Vec<Arc<DenseAcceptanceRows>> {
        if dense_word_count == 0 {
            return (0..acceptance_parts.len())
                .map(|_| Arc::new(DenseAcceptanceRows::default()))
                .collect();
        }
        let mut materialized = FxHashMap::<usize, Arc<DenseAcceptanceRows>>::default();
        acceptance_parts
            .iter()
            .map(|parts| {
            let parts_key = Arc::as_ptr(parts) as *const Weight as usize;
            if let Some(cached) = materialized.get(&parts_key) {
                return Arc::clone(cached);
            }
            let entries = Self::materialize_acceptance_parts_dense(
                parts,
                dense_word_count,
                tsid_count,
                full_dense,
                dense_cache,
            );
            materialized.insert(parts_key, Arc::clone(&entries));
            entries
        })
        .collect()
    }


    #[inline]
    pub(crate) fn direct_regular_wide_frontier_index_for_gss(
        &self,
        gss: &ParserGSS,
    ) -> Option<usize> {
        // The cache below can only contain indices into this table. Ordinary
        // parser runtimes may have no direct-regular wide-frontier summaries;
        // avoid materializing deferred dynamic-vocab state merely to query a
        // cache that cannot contain a valid entry.
        if self.direct_regular_wide_frontier_acceptance.is_empty() {
            return None;
        }
        let lower_id = gss.single_interface_lower_id()?;
        if let Some(index) = self
            .direct_regular_wide_frontier_acceptance
            .iter()
            .position(|summary| {
                summary.empty_acc_frontier.single_interface_lower_id() == Some(lower_id)
            })
        {
            return Some(index);
        }
        if let Some(index) = self
            .initialized_dynamic_mask_vocab_for_runtime()
            .and_then(|vocab| vocab.cached_direct_regular_wide_frontier_index(lower_id))
        {
            return Some(index);
        }
        if gss.max_depth() != 1 {
            return None;
        }
        let mut top_values = gss.peek_values();
        top_values.sort_unstable();
        let index = self
            .direct_regular_wide_frontier_acceptance
            .iter()
            .position(|summary| {
                summary.state_count == top_values.len()
                    && summary.frontier_states.as_ref() == top_values.as_slice()
            })?;
        // This cache is only an optimization.  Do not materialize the entire
        // dynamic-mask vocabulary merely to memoize a wide-frontier lookup;
        // doing so can otherwise put tens of milliseconds onto the first
        // commit of an ordinary statically compiled constraint.
        if let Some(vocab) = self.initialized_dynamic_mask_vocab_for_runtime() {
            vocab.cache_direct_regular_wide_frontier_index(lower_id, index);
        }
        Some(index)
    }

    #[inline]
    pub(crate) fn direct_regular_wide_acceptance_for_parser_state(
        &self,
        parser_state: u32,
    ) -> Option<&DirectRegularParserStateAcceptance> {
        self.direct_regular_parser_state_acceptance
            .iter()
            .find(|summary| summary.parser_state == parser_state)
    }

    #[inline]
    pub(crate) fn direct_regular_wide_frontier_for_gss(
        &self,
        gss: &ParserGSS,
    ) -> Option<&DirectRegularWideFrontierAcceptance> {
        let index = self.direct_regular_wide_frontier_index_for_gss(gss)?;
        self.direct_regular_wide_frontier_acceptance.get(index)
    }

    pub(crate) fn for_each_direct_regular_l1_acceptance(
        &self,
        parser_state: u32,
        mut visit: impl FnMut(RuntimeWeightRef<'_>),
    ) -> bool {
        if self.runtime_direct_regular_l1_is_empty() {
            return false;
        }
        if let Some(row) = self.table.advance.get(parser_state as usize) {
            let mut found = false;
            for terminal in row.iter_ones() {
                if let Some(weight) =
                    self.runtime_direct_regular_l1_complete(terminal as TerminalID)
                {
                    found = true;
                    visit(weight);
                }
            }
            return found;
        }
        if !self.uses_sparse_direct_regular_runtime() {
            return false;
        }
        let Some(automaton) = self.direct_regular_automaton.as_ref() else {
            return false;
        };
        let mut stack = Vec::new();
        if parser_state == 0 {
            stack.extend(automaton.start_states.iter().copied());
        } else if let Some(raw) = parser_state.checked_sub(1) {
            stack.push(raw);
        }
        let mut seen = vec![false; automaton.states.len()];
        let mut terminals = crate::ds::bitset::BitSet::new(self.table.num_terminals as usize);
        while let Some(raw) = stack.pop() {
            let Some(state) = automaton.states.get(raw as usize) else {
                return false;
            };
            if std::mem::replace(&mut seen[raw as usize], true) {
                continue;
            }
            stack.extend(state.epsilons.iter().copied());
            for &terminal in state.transitions.keys() {
                terminals.set(terminal as usize);
            }
        }
        let mut found = false;
        for terminal in terminals.iter_ones() {
            if let Some(weight) =
                self.runtime_direct_regular_l1_complete(terminal as TerminalID)
            {
                found = true;
                visit(weight);
            }
        }
        found
    }

    pub(crate) fn sparse_direct_regular_gss_is_complete(&self, gss: &ParserGSS) -> Option<bool> {
        if !self.uses_sparse_direct_regular_runtime() || gss.max_depth() != 1 {
            return None;
        }
        let automaton = self.direct_regular_automaton.as_ref()?;
        let mut stack = Vec::new();
        gss.for_each_top_value(|state| {
            if state == 0 {
                stack.extend(automaton.start_states.iter().copied());
            } else if let Some(raw) = state.checked_sub(1) {
                stack.push(raw);
            }
        });
        let mut seen = vec![false; automaton.states.len()];
        while let Some(raw) = stack.pop() {
            let state = automaton.states.get(raw as usize)?;
            if std::mem::replace(&mut seen[raw as usize], true) {
                continue;
            }
            if state.is_accepting {
                return Some(true);
            }
            stack.extend(state.epsilons.iter().copied());
        }
        Some(false)
    }

    fn direct_regular_dynamic_frontier(
        &self,
        gss: &ParserGSS,
    ) -> Option<DirectRegularDynamicFrontierCacheEntry> {
        if !self.uses_sparse_direct_regular_runtime() || gss.max_depth() != 1 {
            return None;
        }
        let automaton = self.direct_regular_automaton.as_ref()?;
        let cache_key = gss.single_interface_lower_id();
        if let Some(key) = cache_key
            && let Some(cached) = self
                .initialized_dynamic_mask_vocab_for_runtime()
                .and_then(|vocab| vocab.cached_direct_regular_frontier(key))
        {
            return Some(cached);
        }

        let mut roots = Vec::new();
        gss.for_each_top_value(|state| {
            if state == 0 {
                roots.extend(automaton.start_states.iter().copied());
            } else if let Some(raw) = state.checked_sub(1) {
                roots.push(raw);
            }
        });
        let mut seen = vec![false; automaton.states.len()];
        let mut stack = roots;
        let mut targets_by_terminal = BTreeMap::<TerminalID, Vec<u32>>::new();
        while let Some(raw) = stack.pop() {
            let state = automaton.states.get(raw as usize)?;
            if std::mem::replace(&mut seen[raw as usize], true) {
                continue;
            }
            stack.extend(state.epsilons.iter().copied());
            for (&terminal, targets) in &state.transitions {
                if (terminal as usize) >= self.table.num_terminals as usize {
                    continue;
                }
                targets_by_terminal
                    .entry(terminal)
                    .or_default()
                    .extend(targets.iter().map(|target| target + 1));
            }
        }

        let mut actionable_terminals =
            crate::ds::bitset::BitSet::new(self.table.num_terminals as usize);
        let advance_by_terminal = targets_by_terminal
            .into_iter()
            .filter_map(|(terminal, mut targets)| {
                targets.sort_unstable();
                targets.dedup();
                if targets.is_empty() {
                    return None;
                }
                actionable_terminals.set(terminal as usize);
                Some((terminal, Arc::<[u32]>::from(targets)))
            })
            .collect::<Vec<_>>();
        let entry = DirectRegularDynamicFrontierCacheEntry {
            source: gss.clone(),
            actionable_terminals,
            advance_by_terminal: Arc::from(advance_by_terminal),
        };
        Some(cache_key.map_or(entry.clone(), |key| {
            self.initialized_dynamic_mask_vocab_for_runtime()
                .map_or(entry.clone(), |vocab| {
                    vocab.cache_direct_regular_frontier(key, entry)
                })
        }))
    }

    pub(crate) fn direct_regular_may_advance_on(
        &self,
        gss: &ParserGSS,
        terminal: TerminalID,
    ) -> Option<bool> {
        if !self.uses_sparse_direct_regular_runtime() || gss.max_depth() != 1 {
            return None;
        }
        let automaton = self.direct_regular_automaton.as_ref()?;
        let support = self.dynamic_mask_vocab.direct_regular_terminal_support();
        if support.is_initialized() {
            let mut found = false;
            gss.for_each_top_value(|state| {
                if found {
                    return;
                }
                if state == 0 {
                    found = automaton
                        .start_states
                        .iter()
                        .any(|&raw| support.contains(raw, terminal));
                } else if let Some(raw) = state.checked_sub(1) {
                    found = support.contains(raw, terminal);
                }
            });
            return Some(found);
        }

        let mut stack = Vec::new();
        gss.for_each_top_value(|state| {
            if state == 0 {
                stack.extend(automaton.start_states.iter().copied());
            } else if let Some(raw) = state.checked_sub(1) {
                stack.push(raw);
            }
        });
        let mut seen = vec![false; automaton.states.len()];
        while let Some(raw) = stack.pop() {
            let state = automaton.states.get(raw as usize)?;
            if std::mem::replace(&mut seen[raw as usize], true) {
                continue;
            }
            if state.transitions.contains_key(&terminal) {
                return Some(true);
            }
            stack.extend(state.epsilons.iter().copied());
        }
        Some(false)
    }

    pub(crate) fn direct_regular_may_advance_on_any(
        &self,
        gss: &ParserGSS,
        terminals: &crate::ds::bitset::BitSet,
    ) -> Option<bool> {
        if !self.uses_sparse_direct_regular_runtime() || gss.max_depth() != 1 {
            return None;
        }
        let automaton = self.direct_regular_automaton.as_ref()?;
        let support = self.dynamic_mask_vocab.direct_regular_terminal_support();
        let sparse_terminals = (terminals.count_ones() <= 8).then(|| {
            terminals
                .iter()
                .map(|terminal| terminal as TerminalID)
                .collect::<SmallVec<[TerminalID; 8]>>()
        });
        if support.is_initialized() {
            let mut found = false;
            gss.for_each_top_value(|state| {
                if found {
                    return;
                }
                let state_supports = |raw| {
                    sparse_terminals.as_ref().map_or_else(
                        || support.intersects(raw, terminals.words()),
                        |candidates| {
                            candidates
                                .iter()
                                .any(|&terminal| support.contains(raw, terminal))
                        },
                    )
                };
                if state == 0 {
                    found = automaton.start_states.iter().copied().any(state_supports);
                } else if let Some(raw) = state.checked_sub(1) {
                    found = state_supports(raw);
                }
            });
            return Some(found);
        }

        let sparse_terminals = sparse_terminals.map(|terminals| {
            terminals
                .into_iter()
                .map(|terminal| terminal as u32)
                .collect::<SmallVec<[u32; 8]>>()
        });
        let mut stack = Vec::new();
        gss.for_each_top_value(|state| {
            if state == 0 {
                stack.extend(automaton.start_states.iter().copied());
            } else if let Some(raw) = state.checked_sub(1) {
                stack.push(raw);
            }
        });
        let mut seen = vec![false; automaton.states.len()];
        while let Some(raw) = stack.pop() {
            let state = automaton.states.get(raw as usize)?;
            if std::mem::replace(&mut seen[raw as usize], true) {
                continue;
            }
            let found = sparse_terminals.as_ref().map_or_else(
                || {
                    state
                        .transitions
                        .keys()
                        .any(|terminal| terminals.contains(*terminal as usize))
                },
                |candidates| {
                    candidates
                        .iter()
                        .any(|terminal| state.transitions.contains_key(terminal))
                },
            );
            if found {
                return Some(true);
            }
            stack.extend(state.epsilons.iter().copied());
        }
        Some(false)
    }

    pub(crate) fn direct_regular_admissible_terminals(
        &self,
        gss: &ParserGSS,
    ) -> Option<crate::ds::bitset::BitSet> {
        if !self.uses_sparse_direct_regular_runtime() || gss.max_depth() != 1 {
            return None;
        }
        let automaton = self.direct_regular_automaton.as_ref()?;
        if let Some(summary) = self.direct_regular_dynamic_hot_frontier_for_gss(gss) {
            return Some(summary.actionable_terminals.clone());
        }
        if !self.direct_regular_wide_frontier_acceptance.is_empty()
            && let Some(summary) = self.direct_regular_wide_frontier_for_gss(gss)
        {
            return Some(summary.actionable_terminals.clone());
        }
        let support = self.dynamic_mask_vocab.direct_regular_terminal_support();
        if support.is_initialized() {
            let mut terminals =
                crate::ds::bitset::BitSet::new(self.table.num_terminals as usize);
            gss.for_each_top_value(|state| {
                let mut add_state = |raw| {
                    if !support.for_each_small_state_terminal(raw, |terminal| {
                        terminals.set(terminal as usize);
                    }) {
                        support.or_state_into(raw, terminals.words_mut());
                    }
                };
                if state == 0 {
                    for &raw in &automaton.start_states {
                        add_state(raw);
                    }
                } else if let Some(raw) = state.checked_sub(1) {
                    add_state(raw);
                }
            });
            return Some(terminals);
        }
        self.direct_regular_dynamic_frontier(gss)
            .map(|frontier| frontier.actionable_terminals)
    }

    pub(crate) fn direct_regular_cached_advance(
        &self,
        gss: &ParserGSS,
        terminal: TerminalID,
    ) -> Option<ParserGSS> {
        if let Some(summary) = self.direct_regular_dynamic_hot_frontier_for_gss(gss) {
            let acc = gss.uniform_accumulator()?;
            let Ok(index) = summary
                .advance_by_terminal
                .binary_search_by_key(&terminal, |(candidate, _)| *candidate)
            else {
                return Some(ParserGSS::empty());
            };
            let targets = &summary.advance_by_terminal[index].1;
            if Arc::ptr_eq(targets, &summary.frontier_states) {
                return summary.empty_acc_frontier.with_uniform_accumulator(acc);
            }
            if let Some(target_summary) = self
                .direct_regular_dynamic_hot_frontiers
                .iter()
                .find(|candidate| {
                    Arc::ptr_eq(&candidate.frontier_states, targets)
                        || candidate.frontier_states.as_ref() == targets.as_ref()
                })
            {
                return target_summary.empty_acc_frontier.with_uniform_accumulator(acc);
            }
            if let [target] = targets.as_ref() {
                return Some(ParserGSS::from_single_stack(vec![*target], acc));
            }
            return Some(ParserGSS::from_sorted_unique_single_value_stacks(
                targets,
                acc,
            ));
        }
        if let Some(summary) = self.direct_regular_wide_frontier_for_gss(gss) {
            let acc = gss.uniform_accumulator()?;
            let index = summary
                .advance_by_terminal
                .binary_search_by_key(&terminal, |(candidate, _)| *candidate)
                .ok()?;
            let targets = &summary.advance_by_terminal.get(index)?.1;
            if Arc::ptr_eq(targets, &summary.frontier_states) {
                return summary.empty_acc_frontier.with_uniform_accumulator(acc);
            }
            if let [target] = targets.as_ref() {
                return Some(ParserGSS::from_single_stack(vec![*target], acc));
            }
            return Some(ParserGSS::from_sorted_unique_single_value_stacks(
                targets,
                acc,
            ));
        }
        if self.uses_sparse_direct_regular_runtime() {
            if gss.max_depth() != 1 {
                return None;
            }
            let acc = gss.uniform_accumulator()?;
            let automaton = self.direct_regular_automaton.as_ref()?;
            let support = self.dynamic_mask_vocab.direct_regular_terminal_support();
            if support.is_initialized() {
                let mut stack = Vec::<u32>::new();
                gss.for_each_top_value(|state| {
                    if state == 0 {
                        stack.extend(
                            automaton
                                .start_states
                                .iter()
                                .copied()
                                .filter(|&raw| support.contains(raw, terminal)),
                        );
                    } else if let Some(raw) = state.checked_sub(1)
                        && support.contains(raw, terminal)
                    {
                        stack.push(raw);
                    }
                });
                let mut seen = crate::ds::bitset::BitSet::new(automaton.states.len());
                let mut target_bits =
                    crate::ds::bitset::BitSet::new(automaton.states.len() + 1);
                while let Some(raw) = stack.pop() {
                    let raw_index = raw as usize;
                    if seen.contains(raw_index) {
                        continue;
                    }
                    seen.set(raw_index);
                    let state = automaton.states.get(raw_index)?;
                    if let Some(state_targets) = state.transitions.get(&terminal) {
                        for &target in state_targets {
                            target_bits.set(target as usize + 1);
                        }
                    }
                    stack.extend(
                        state
                            .epsilons
                            .iter()
                            .copied()
                            .filter(|&child| support.contains(child, terminal)),
                    );
                }
                let target_count = target_bits.count_ones();
                if target_count == 0 {
                    return Some(ParserGSS::empty());
                }
                if target_count == 1 {
                    let target = target_bits
                        .iter_ones()
                        .next()
                        .expect("one target bit must be present") as u32;
                    return Some(ParserGSS::from_single_stack(vec![target], acc));
                }
                let targets = target_bits
                    .iter_ones()
                    .map(|target| target as u32)
                    .collect::<Vec<_>>();
                let source_is_target = target_count == gss.top_value_count() && {
                    let mut same = true;
                    gss.for_each_top_value(|state| {
                        same &= target_bits.contains(state as usize);
                    });
                    same
                };
                if source_is_target {
                    return Some(gss.clone());
                }
                if let Some(summary) = self
                    .direct_regular_wide_frontier_acceptance
                    .iter()
                    .find(|summary| summary.frontier_states.as_ref() == targets.as_slice())
                    && let Some(canonical) = summary
                        .empty_acc_frontier
                        .with_uniform_accumulator(acc.clone())
                {
                    return Some(canonical);
                }
                return Some(ParserGSS::from_sorted_unique_single_value_stacks(
                    &targets, acc,
                ));
            }

            let frontier = self.direct_regular_dynamic_frontier(gss)?;
            let Ok(index) = frontier
                .advance_by_terminal
                .binary_search_by_key(&terminal, |(candidate, _)| *candidate)
            else {
                return Some(ParserGSS::empty());
            };
            let targets = &frontier.advance_by_terminal[index].1;
            if let [target] = targets.as_ref() {
                return Some(ParserGSS::from_single_stack(vec![*target], acc));
            }
            return Some(ParserGSS::from_sorted_unique_single_value_stacks(
                targets,
                acc,
            ));
        }
        if gss.max_depth() != 1 {
            return None;
        }
        let acc = gss.uniform_accumulator()?;
        let state = gss.single_exclusive_top_value()?;
        let origin = match self.table.action(state, terminal)? {
            Action::ReplaceShifts(targets) => targets.as_ptr() as usize,
            Action::StackShifts(shifts)
                if shifts
                    .iter()
                    .all(|shift| shift.pop == 1 && shift.pushes.len() == 1) =>
            {
                shifts.as_ptr() as usize
            }
            _ => return None,
        };
        self.direct_regular_wide_frontier_acceptance
            .iter()
            .find(|summary| summary.action_origins.contains(&origin))?
            .empty_acc_frontier
            .with_uniform_accumulator(acc)
    }
}
