use crate::automata::lexer::Lexer;
use std::collections::BTreeMap;

use range_set_blaze::RangeSetBlaze;

use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{ParserGSS, close_control_stacks};
use crate::grammar::flat::TerminalID;

use super::artifact::Constraint;

impl Constraint {
	#[inline]
	pub(crate) fn has_original_token_map(&self) -> bool {
		!self.original_token_to_internal.is_empty()
			|| self
				.packed_original_token_to_internal
				.as_ref()
				.is_some_and(|packed| !packed.is_empty())
	}

	pub(crate) fn original_token_map(&self) -> &[u32] {
		if !self.original_token_to_internal.is_empty() {
			return &self.original_token_to_internal;
		}
		let Some(packed) = self.packed_original_token_to_internal.as_ref() else {
			return &[];
		};
		self.deferred_original_token_to_internal
			.get_or_init(|| packed.materialize())
			.as_slice()
	}

	#[inline]
	pub(crate) fn original_token_internal_at(&self, token_id: u32) -> Option<u32> {
		let index = token_id as usize;
		if let Some(&internal) = self.original_token_to_internal.get(index) {
			return Some(internal);
		}
		self.packed_original_token_to_internal
			.as_ref()
			.and_then(|packed| packed.get(index))
	}

	#[inline]
	pub(crate) fn internal_token_count(&self) -> usize {
		if !self.internal_token_to_tokens.is_empty() {
			return self.internal_token_to_tokens.len();
		}
		if let Some(groups) = self.deferred_internal_token_to_tokens.get() {
			return groups.len();
		}
		if self.internal_token_buf_offsets.len() > 1
			&& self.internal_token_buf_offsets.last().copied().map(|end| end as usize)
				== Some(self.internal_token_buf_flat_len())
		{
			return self.internal_token_buf_offsets.len() - 1;
		}
		if !self.internal_token_buf_masks.is_empty() {
			return self.internal_token_buf_masks.len();
		}
		self.original_token_map()
			.iter()
			.copied()
			.filter(|&internal| internal != u32::MAX)
			.max()
			.map_or_else(
				|| self.max_original_token_id().map_or(0, |id| id as usize + 1),
				|internal| internal as usize + 1,
			)
	}

	pub(crate) fn internal_token_groups(&self) -> Option<&[Vec<u32>]> {
		if !self.internal_token_to_tokens.is_empty() {
			return Some(&self.internal_token_to_tokens);
		}
		if !self.has_original_token_map() {
			return None;
		}
		let group_count = self.internal_token_count();
		let original_token_to_internal = self.original_token_map();
		Some(
			self.deferred_internal_token_to_tokens
				.get_or_init(|| {
					let mut counts = vec![0usize; group_count];
					for &internal in original_token_to_internal {
						if internal != u32::MAX {
							counts[internal as usize] += 1;
						}
					}
					let mut groups = counts
						.into_iter()
						.map(Vec::<u32>::with_capacity)
						.collect::<Vec<_>>();
					for (original, &internal) in original_token_to_internal.iter().enumerate() {
						if internal != u32::MAX {
							groups[internal as usize].push(original as u32);
						}
					}
					groups
				})
				.as_slice(),
		)
	}

	pub(crate) fn runtime_source_state_offset(&self) -> Option<u32> {
		self.runtime_source_state_offset
	}

	/// Scanner reset coordinate used while committing one model token.
	/// Hybrid runtime determinization is a boundary-only compression: commit
	/// itself must reset into the appended historical tokenizer, never into the
	/// provenance-free product start state.
	pub(crate) fn runtime_commit_initial_state(&self) -> u32 {
		let product_initial = self.tokenizer.initial_state();
		self.runtime_source_state_offset
			.and_then(|offset| {
				self.runtime_product_exact_source_state(product_initial)
					.map(|source| offset + source)
			})
			.unwrap_or(product_initial)
	}

	pub(crate) fn runtime_product_source_states(&self, product_state: u32) -> Option<&[u32]> {
		let source_offset = self.runtime_source_state_offset?;
		if product_state >= source_offset {
			return None;
		}
		let state = product_state as usize;
		let start = *self.runtime_product_source_offsets.get(state)? as usize;
		let end = *self.runtime_product_source_offsets.get(state + 1)? as usize;
		self.runtime_product_source_states.get(start..end)
	}

	pub(crate) fn runtime_product_exact_source_state(&self, product_state: u32) -> Option<u32> {
		let source_state = *self
			.runtime_product_exact_source_states
			.get(product_state as usize)?;
		(source_state != u32::MAX).then_some(source_state)
	}

	pub(crate) fn runtime_product_state_for_source_subset(&self, states: &[u32]) -> Option<u32> {
		self.runtime_product_state_by_source_subset.get(states).copied()
	}

	pub(crate) fn possible_matches_for_state(
		&self,
		tokenizer_state: u32,
	) -> BTreeMap<TerminalID, RangeSetBlaze<u32>> {
		self.runtime_possible_match_terminals()
			.filter_map(|terminal| {
				let weight = self.runtime_possible_match_weight(terminal)?;
				let mut tokens = RangeSetBlaze::new();
				for &internal_tsid in self.internal_tsids_for_state(tokenizer_state) {
					if let Some(token_set) = weight.token_set_for_tsid(internal_tsid) {
						tokens |= token_set.to_range_set();
					}
				}
				if tokens.is_empty() {
					None
				} else {
					Some((terminal, self.expand_internal_token_set(&tokens)))
				}
			})
			.collect()
	}

	#[inline]
	fn static_observation_state(&self, tokenizer_state: u32) -> u32 {
		if !self.uses_dynamic_runtime()
			&& self.tokenizer.has_virtual_residual_runtime()
			&& self.dynamic_mask_vocab.mask_projection_tokenizer().is_some()
		{
			self.dynamic_mask_vocab.mask_projection_state(tokenizer_state)
		} else {
			tokenizer_state
		}
	}

	pub(crate) fn internal_tsid_for_state(&self, tokenizer_state: u32) -> u32 {
		let tokenizer_state = self.static_observation_state(tokenizer_state);
		self.state_to_internal_tsid
			.get(tokenizer_state as usize)
			.copied()
			.unwrap_or(tokenizer_state)
	}

	pub(crate) fn internal_tsids_for_state(&self, tokenizer_state: u32) -> &[u32] {
		let state = self.static_observation_state(tokenizer_state) as usize;
		if let (Some(&start), Some(&end)) = (
			self.state_internal_tsid_offsets.get(state),
			self.state_internal_tsid_offsets.get(state + 1),
		) {
			if let Some(tsids) = self
				.state_internal_tsids
				.get(start as usize..end as usize)
			{
				return tsids;
			}
		}
		self.state_to_internal_tsid
			.get(state)
			.map(std::slice::from_ref)
			.unwrap_or(&[])
	}

	pub(crate) fn internal_token_for_original(&self, token_id: u32) -> u32 {
		self.original_token_internal_at(token_id)
			.filter(|internal_id| *internal_id != u32::MAX)
			.unwrap_or(token_id)
	}

	pub(crate) fn final_internal_token_for_original(&self, token_id: u32) -> Option<u32> {
		let internal = self.original_token_internal_at(token_id)?;

		if internal == u32::MAX {
			return None;
		}

		if internal as usize >= self.internal_token_count() {
			return None;
		}

		Some(internal)
	}

	/// Visit the original vocabulary tokens that can complete `terminal` from
	/// `tokenizer_state`. Returns `false` only for legacy artifacts that omitted
	/// possible matches.
	///
	/// IMPORTANT: callers must not treat this as an acceptable construction
	/// strategy. The dynamic possible-matches fallback is terrible, is planned
	/// for removal, and MUST NOT be used by new compiler paths. DO NOT REMOVE OR
	/// WEAKEN THIS COMMENT.
	pub(crate) fn visit_possible_match_original_tokens(
		&self,
		tokenizer_state: u32,
		terminal: TerminalID,
		mut visit: impl FnMut(u32),
	) -> bool {
		if !self.possible_matches_complete {
			return false;
		}
		let Some(weight) = self.runtime_possible_match_weight(terminal) else {
			return true;
		};
		let mut internal_tokens = RangeSetBlaze::new();
		for &internal_tsid in self.internal_tsids_for_state(tokenizer_state) {
			if let Some(token_set) = weight.token_set_for_tsid(internal_tsid) {
				internal_tokens |= token_set.to_range_set();
			}
		}
		let Some(groups) = self.internal_token_groups() else {
			for token in internal_tokens.iter() {
				visit(token);
			}
			return true;
		};
		for internal_token in internal_tokens.iter() {
			if let Some(originals) = groups.get(internal_token as usize) {
				for &original in originals {
					visit(original);
				}
			}
		}
		true
	}

	/// Return `Some(true)` when the exact possible-match relation proves that
	/// no public/original vocabulary token can complete `terminal` from
	/// `tokenizer_state`. This deliberately accounts for internal token classes
	/// whose original-token group is empty (for example linker-only placeholder
	/// classes), so callers can skip an otherwise expensive full expansion.
	/// Legacy artifacts without complete possible matches return `None`.
	pub(crate) fn possible_match_original_tokens_definitely_empty(
		&self,
		tokenizer_state: u32,
		terminal: TerminalID,
	) -> Option<bool> {
		// Dynamic transfer intentionally omits the heavyweight possible-match
		// table. Before falling back to a vocabulary scan, prove the stronger
		// vocabulary-independent fact that this terminal cannot match after any
		// byte string whose length fits in one model token. This exact BFS only
		// handles ordinary scalar lexer states; epsilon/virtual paths conservatively
		// decline and retain the existing fallback.
		let ordinary_impossible = self.ordinary_terminal_match_impossible_within(
			tokenizer_state,
			terminal,
			self.max_token_byte_len(),
		);
		if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_some() {
			eprintln!(
				"[glrmask/profile][pending_guard_bounded_reachability] state={} terminal={} impossible={:?} epsilon={} virtual={} max_token_len={}",
				tokenizer_state,
				terminal,
				ordinary_impossible,
				self.tokenizer.state_has_epsilon_transitions(tokenizer_state),
				self.tokenizer.state_is_virtual_runtime(tokenizer_state),
				self.max_token_byte_len(),
			);
		}
		if let Some(impossible) = ordinary_impossible {
			if impossible {
				return Some(true);
			}
		}
		if !self.possible_matches_complete {
			return None;
		}
		let Some(weight) = self.runtime_possible_match_weight(terminal) else {
			return Some(true);
		};
		let internal_count = self.internal_token_count();
		let flat_ready = self.internal_token_buf_offsets.len() == internal_count.saturating_add(1)
			&& self
				.internal_token_buf_offsets
				.last()
				.is_some_and(|&end| end as usize == self.internal_token_buf_flat_len());
		let materialized_ready = self.internal_token_buf_masks.len() == internal_count;
		if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_some() {
			eprintln!(
				"[glrmask/profile][possible_match_public_empty] internal_count={} offsets={} flat_len={} materialized={} flat_ready={} materialized_ready={}",
				internal_count,
				self.internal_token_buf_offsets.len(),
				self.internal_token_buf_flat_len(),
				self.internal_token_buf_masks.len(),
				flat_ready,
				materialized_ready,
			);
		}
		if !flat_ready && !materialized_ready {
			return None;
		}
		for &internal_tsid in self.internal_tsids_for_state(tokenizer_state) {
			let Some(tokens) = weight.token_set_for_tsid(internal_tsid) else {
				continue;
			};
			if tokens.is_empty() {
				continue;
			}
			let mut has_public = false;
			tokens.for_each_range(|start, end| {
				if has_public {
					return;
				}
				let start = start as usize;
				let end = (end as usize).min(internal_count.saturating_sub(1));
				if start > end {
					return;
				}
				if flat_ready {
					for internal in start..=end {
						if self.internal_token_buf_offsets[internal]
							!= self.internal_token_buf_offsets[internal + 1]
						{
							has_public = true;
							break;
						}
					}
				} else {
					for internal in start..=end {
						if !self.internal_token_buf_masks[internal].is_empty() {
							has_public = true;
							break;
						}
					}
				}
			});
			if has_public {
				return Some(false);
			}
		}
		Some(true)
	}

	fn ordinary_terminal_match_impossible_within(
		&self,
		start: u32,
		terminal: TerminalID,
		horizon: usize,
	) -> Option<bool> {
		let tokenizer = &self.tokenizer;
		if horizon == 0 || start >= tokenizer.num_states() {
			return Some(true);
		}
		if tokenizer.state_has_epsilon_transitions(start)
			|| tokenizer.state_is_virtual_runtime(start)
		{
			return None;
		}
		if !tokenizer
			.possible_future_terminals(start)
			.contains(terminal as usize)
		{
			return Some(true);
		}

		let mut distance = vec![usize::MAX; tokenizer.num_states() as usize];
		let mut queue = std::collections::VecDeque::new();
		distance[start as usize] = 0;
		queue.push_back(start);
		let mut ambiguous = false;
		while let Some(state) = queue.pop_front() {
			let depth = distance[state as usize];
			if depth >= horizon {
				continue;
			}
			for (_, target) in tokenizer.transitions_from(state) {
				if tokenizer
					.matched_terminal_bitset(target)
					.contains(terminal as usize)
				{
					return Some(false);
				}
				if !tokenizer
					.possible_future_terminals(target)
					.contains(terminal as usize)
				{
					continue;
				}
				if tokenizer.state_has_epsilon_transitions(target)
					|| tokenizer.state_is_virtual_runtime(target)
				{
					ambiguous = true;
					continue;
				}
				let next_depth = depth + 1;
				if next_depth < distance[target as usize] {
					distance[target as usize] = next_depth;
					queue.push_back(target);
				}
			}
		}
		if ambiguous { None } else { Some(true) }
	}

	pub(crate) fn internal_token_universe(&self) -> RangeSetBlaze<u32> {
		if !self.has_original_token_map() && self.internal_token_to_tokens.is_empty() {
			let Some(max_token_id) = self.max_original_token_id() else {
				return RangeSetBlaze::new();
			};
			return RangeSetBlaze::from_iter([0..=max_token_id]);
		}

		RangeSetBlaze::from_iter([0..=self.internal_token_count().saturating_sub(1) as u32])
	}

	pub(crate) fn expand_internal_token_set(
		&self,
		internal_tokens: &RangeSetBlaze<u32>,
	) -> RangeSetBlaze<u32> {
		let Some(groups) = self.internal_token_groups() else {
			return internal_tokens.clone();
		};

		let all_ids = Self::collect_original_token_ids(groups, internal_tokens);
		Self::range_set_from_sorted_ids(&all_ids)
	}

	pub(crate) fn initial_state_map(&self) -> crate::runtime::state::ParserStateMap {
		let parser_gss = ParserGSS::from_stacks(&[(vec![0u32], TerminalsDisallowed::new())]);
		let parser_gss = if let Some(closed) = self.close_compact_segmented_parser(&parser_gss) {
			closed
		} else if !self.parser_has_controls() {
			parser_gss
		} else {
			close_control_stacks(&self.table, &parser_gss)
		};
		if self.uses_compact_segmented_parser_runtime() {
			let partitions = self
				.partition_recursive_parser_gss_by_active_leaf(&parser_gss)
				.expect("recursive initial parser state must partition by active leaf");
			let mut state = crate::runtime::state::ParserStateMap::default();
			for (leaf_index, partition) in partitions {
				let reset = self
					.recursive_tokenizer_reset_state(leaf_index)
					.expect("recursive initial active leaf must expose a tokenizer reset");
				state.merge_insert(reset, partition);
			}
			return state;
		}
		crate::runtime::state::ParserStateMap::singleton(self.tokenizer.initial_state(), parser_gss)
	}

	fn collect_original_token_ids(
		groups: &[Vec<u32>],
		internal_tokens: &RangeSetBlaze<u32>,
	) -> Vec<u32> {
		let total_estimate: usize = internal_tokens
			.iter()
			.filter_map(|token| groups.get(token as usize))
			.map(Vec::len)
			.sum();
		let mut all_ids = Vec::with_capacity(total_estimate);
		for internal_token in internal_tokens.iter() {
			if let Some(originals) = groups.get(internal_token as usize) {
				all_ids.extend_from_slice(originals);
			}
		}
		all_ids.sort_unstable();
		all_ids.dedup();
		all_ids
	}

	fn range_set_from_sorted_ids(ids: &[u32]) -> RangeSetBlaze<u32> {
		let Some((&first, rest)) = ids.split_first() else {
			return RangeSetBlaze::new();
		};

		let mut ranges = Vec::new();
		let mut start = first;
		let mut end = first;
		for &id in rest {
			if id == end + 1 {
				end = id;
			} else {
				ranges.push(start..=end);
				start = id;
				end = id;
			}
		}
		ranges.push(start..=end);
		RangeSetBlaze::from_iter(ranges)
	}
}
