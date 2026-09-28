//! Bounded concrete and stack-language queues with reusable scratch.

use super::admission::ActionableTerminals;
use super::admission::cached_batched_end_state_admission;
use super::admission::cached_single_end_state_may_advance;
use super::admission::end_state_may_advance;
use super::admission::end_state_may_advance_from_cache_entry;
use super::admission::end_state_may_advance_from_row_words;
use super::admission::runtime_future_contains_ignore;
use super::admission::try_local_row_presence_admission_words;
use super::advance::advance_parser_stacks_if_possible;
use super::advance::apply_single_top_action_fast;
use super::advance::template_advance_enabled;
use super::assertions::format_token_bytes;
use super::lexical::NormalizedMatch;
use super::lexical::apply_future_terminal_disallow_for_states;
use super::lexical::collect_unique_actionable_matches;
use super::lexical::collect_unique_actionable_reusable_matches;
use super::lexical::is_actionable_terminal;
use super::lexical::is_ignored_terminal;
use super::lexical::prune_single_initial_state_for_parts;
use super::lexical::try_prune_single_initial_state_batched_accumulators;
use super::template_advance::TemplateAdvanceRuntime;
use super::tokenizer_scan;
use super::tokenizer_scan::execute_tokenizer_reusable;
use super::tokenizer_scan::execute_tokenizer_reusable_from_states;
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::TokenizerMatch;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::parser::advance_stacks_disjoint_top_terminals_bounded;
use crate::compiler::glr::table::Action;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::INLINE_PARSER_STATE_CAPACITY;
use crate::runtime::state::ParserAdmissionCacheEntry;
use crate::runtime::state::ParserStateMap;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::OnceLock;

pub(super) type SmallParserStates = SmallVec<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>;

pub(super) const SMALL_LANGUAGE_QUEUE_CAPACITY: usize = 32;

pub(super) const LANGUAGE_QUEUE_MAX_INPUT_STACK_DEPTH: u32 = 512;

pub(super) const LANGUAGE_QUEUE_MIN_TOP_VALUES: usize = 3;

pub(super) const LANGUAGE_QUEUE_MIN_PATHS: usize = 32;

pub(super) const LANGUAGE_QUEUE_MIN_NODES: usize = 48;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SmallLanguageParserState {
    pub(super) tokenizer_state: u32,
    pub(super) language: u32,
    pub(super) accumulator: TerminalsDisallowed,
}

pub(super) type SmallLanguageParserStates = SmallVec<[SmallLanguageParserState; 16]>;

#[derive(Debug)]
pub(crate) struct SmallCommitQueueScratch {
    pub(super) processing: [SmallParserStates; 17],
    pub(super) pending: SmallParserStates,
    pub(super) language_processing: [SmallLanguageParserStates; 17],
    pub(super) language_pending: SmallLanguageParserStates,
    pub(super) prune_union_starts: SmallVec<[u32; 8]>,
}

impl Default for SmallCommitQueueScratch {
    fn default() -> Self {
        Self {
            processing: std::array::from_fn(|_| SmallVec::new()),
            pending: SmallVec::new(),
            language_processing: std::array::from_fn(|_| SmallVec::new()),
            language_pending: SmallVec::new(),
            prune_union_starts: SmallVec::new(),
        }
    }
}

impl SmallCommitQueueScratch {
    pub(crate) fn clear(&mut self) {
        for bucket in &mut self.processing {
            bucket.clear();
        }
        self.pending.clear();
        for bucket in &mut self.language_processing {
            bucket.clear();
        }
        self.language_pending.clear();
        self.prune_union_starts.clear();
    }
}

pub(super) fn merge_small_parser_state(
    states: &mut SmallVec<[(u32, ParserGSS); INLINE_PARSER_STATE_CAPACITY]>,
    tokenizer_state: u32,
    gss: ParserGSS,
) {
    for (existing_state, existing_gss) in states.iter_mut() {
        if *existing_state == tokenizer_state {
            *existing_gss = existing_gss.merge(&gss);
            return;
        }
    }
    states.push((tokenizer_state, gss));
}

pub(super) fn merge_small_language_parser_state(
    runtime: &mut TemplateAdvanceRuntime,
    states: &mut SmallLanguageParserStates,
    tokenizer_state: u32,
    language: u32,
    accumulator: TerminalsDisallowed,
) -> bool {
    if language == 0 {
        return true;
    }
    for existing in states.iter_mut() {
        if existing.tokenizer_state == tokenizer_state && existing.accumulator == accumulator {
            existing.language = runtime.union_languages(existing.language, language);
            return !runtime.is_exhausted();
        }
    }
    if states.len() == SMALL_LANGUAGE_QUEUE_CAPACITY {
        return false;
    }
    states.push(SmallLanguageParserState { tokenizer_state, language, accumulator });
    true
}

pub(super) fn actionable_terminals_from_language(
    runtime: &TemplateAdvanceRuntime,
    language: u32,
) -> Option<ActionableTerminals> {
    let states = runtime.language_top_states(language);
    match states.as_slice() {
        [] => None,
        [state] => Some(ActionableTerminals::SingleState(*state)),
        _ => Some(ActionableTerminals::ManyStates(states)),
    }
}

pub(super) fn prune_uniform_accumulator_for_parts(
    constraint: &Constraint,
    actionable_terminals: Option<&ActionableTerminals>,
    accumulator: &TerminalsDisallowed,
    tokenizer_state: u32,
    end_states: &[u32],
    matches: &[TokenizerMatch],
) -> Option<TerminalsDisallowed> {
    if accumulator.is_empty() {
        return Some(TerminalsDisallowed::new());
    }

    let accepted_terminals = matches
        .iter()
        .filter(|matched| !is_ignored_terminal(constraint.ignore_terminal, matched.id))
        .filter(|matched| is_actionable_terminal(actionable_terminals, constraint, matched.id))
        .map(|matched| matched.id)
        .collect::<SmallVec<[u32; INLINE_PARSER_STATE_CAPACITY]>>();

    if let Some(disallowed) = accumulator.get(&tokenizer_state)
        && !accepted_terminals.is_empty()
        && accepted_terminals.iter().all(|terminal| disallowed.contains(terminal))
    {
        return None;
    }

    if let Some(remapped) = accumulator.try_remap_single_state_inline(tokenizer_state, end_states) {
        return Some(remapped);
    }

    let terminals = accumulator
        .get(&tokenizer_state)
        .map(|values| values.iter().copied().collect::<Vec<_>>())
        .unwrap_or_default();
    if terminals.is_empty() || end_states.is_empty() {
        return Some(TerminalsDisallowed::new());
    }
    let mut remapped = BTreeMap::<u32, BTreeSet<u32>>::new();
    for &end_state in end_states {
        remapped.entry(end_state).or_default().extend(terminals.iter().copied());
    }
    Some(TerminalsDisallowed::from_map(remapped))
}

pub(super) fn apply_future_terminal_disallow_to_accumulator(
    constraint: &Constraint,
    end_states: &[u32],
    terminal: u32,
    mut accumulator: TerminalsDisallowed,
) -> TerminalsDisallowed {
    for &end_state in end_states {
        if constraint.tokenizer.possible_future_terminals(end_state).contains(terminal as usize) {
            accumulator = accumulator.with_insert(end_state, terminal);
        }
    }
    accumulator
}

pub(super) fn language_end_state_may_advance(
    constraint: &Constraint,
    runtime: &mut TemplateAdvanceRuntime,
    language: u32,
    end_state: u32,
) -> Option<bool> {
    if end_state == constraint.runtime_commit_initial_state() {
        return Some(true);
    }
    if runtime_future_contains_ignore(
        constraint,
        constraint.tokenizer.possible_future_terminals(end_state),
    ) {
        return Some(true);
    }
    for terminal in constraint.tokenizer.possible_future_terminals(end_state).iter_ones() {
        let terminal = u32::try_from(terminal).ok()?;
        let advanced = runtime.advance_language(constraint, terminal, language)?;
        if advanced != 0 {
            return Some(true);
        }
    }
    Some(false)
}

pub(super) fn language_queue_top_value_count_at_most(
    state: &ParserStateMap,
    limit: usize,
) -> usize {
    let mut count = 0usize;
    for (_, gss) in state.iter() {
        count = count.saturating_add(gss.top_value_count()).min(limit);
        if count == limit {
            break;
        }
    }
    count
}

pub(super) fn language_queue_path_count_at_most(state: &ParserStateMap, limit: usize) -> usize {
    let mut count = 0usize;
    for (_, gss) in state.iter() {
        count = count.saturating_add(gss.path_count_at_most(limit)).min(limit);
        if count == limit {
            break;
        }
    }
    count
}

pub(super) fn language_queue_node_count_at_most(state: &ParserStateMap, limit: usize) -> usize {
    let mut count = 0usize;
    for (_, gss) in state.iter() {
        count = count.saturating_add(gss.node_count_at_most(limit)).min(limit);
        if count == limit {
            break;
        }
    }
    count
}

pub(super) fn language_small_queue_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let flag = |name: &str| {
            std::env::var(name).ok().map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
            })
        };
        if flag("GLRMASK_DISABLE_LANGUAGE_SMALL_QUEUE") == Some(true) {
            return false;
        }
        flag("GLRMASK_ENABLE_LANGUAGE_SMALL_QUEUE").unwrap_or(true)
    })
}

pub(super) fn language_queue_input_is_bounded(state: &ParserStateMap) -> bool {
    state.iter().all(|(_, gss)| gss.max_depth() <= LANGUAGE_QUEUE_MAX_INPUT_STACK_DEPTH)
}

/// Return whether one model token contains multiple parser-actionable,
/// non-ignored terminal completion boundaries at different byte offsets.
///
/// This is the structural case where the ordinary byte queue must materialize
/// and carry parser states at more than one offset inside the same token. The
/// language queue evaluates all such offset alternatives before reconstructing
/// a GSS once at token completion. Multiple terminals ending at the same byte
/// offset do not qualify: the ordinary queue already merges those without an
/// offset frontier split.
pub(super) fn has_multiple_actionable_terminal_boundaries(
    constraint: &Constraint,
    state: &ParserStateMap,
    bytes: &[u8],
    tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
) -> bool {
    for (&tokenizer_state, gss) in state.iter() {
        if !execute_tokenizer_reusable(constraint, bytes, tokenizer_state, tokenizer_scratch) {
            return false;
        }
        let actionable = ActionableTerminals::from_gss(constraint, gss);
        let matches = collect_unique_actionable_matches(
            constraint,
            actionable.as_ref(),
            constraint.ignore_terminal,
            &tokenizer_scratch.matches,
            None,
        );
        let mut widths = SmallVec::<[usize; 4]>::new();
        for matched in matches {
            if matched.ignored || widths.contains(&matched.width) {
                continue;
            }
            widths.push(matched.width);
            if widths.len() >= 2 {
                return true;
            }
        }
    }
    false
}

#[derive(Default)]
pub(super) struct LanguageCommitSimulationProfile {
    pub(super) canonicalize_ns: u64,
    pub(super) evaluate_ns: u64,
    pub(super) continuation_reconstruct_ns: u64,
    pub(super) continuation_check_ns: u64,
}

pub(super) fn simulate_language_commit(
    constraint: &Constraint,
    state: &ParserStateMap,
    bytes: &[u8],
    tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
    queue_scratch: &mut SmallCommitQueueScratch,
    template_runtime: &mut TemplateAdvanceRuntime,
    mut profile: Option<&mut LanguageCommitSimulationProfile>,
) -> Option<Result<SmallLanguageParserStates, String>> {
    if bytes.is_empty() || bytes.len() > 8 || state.is_empty() || state.len() > 8 {
        return None;
    }

    queue_scratch.clear();
    for (&tokenizer_state, gss) in state.iter() {
        let started = profile.is_some().then(std::time::Instant::now);
        let components = template_runtime.language_components_from_gss(gss)?;
        if let (Some(profile), Some(started)) = (profile.as_deref_mut(), started) {
            profile.canonicalize_ns += started.elapsed().as_nanos() as u64;
        }
        if template_runtime.is_exhausted() {
            return None;
        }
        for (language, accumulator) in components {
            if !merge_small_language_parser_state(
                template_runtime,
                &mut queue_scratch.language_processing[0],
                tokenizer_state,
                language,
                accumulator,
            ) {
                return None;
            }
        }
    }

    let initial_tokenizer_state = constraint.runtime_commit_initial_state();
    let mut offset = 0usize;
    while offset <= bytes.len() {
        if queue_scratch.language_processing[offset].is_empty() {
            offset += 1;
            continue;
        }

        let states_to_process = std::mem::take(&mut queue_scratch.language_processing[offset]);
        for mut entry in states_to_process {
            if !execute_tokenizer_reusable(
                constraint,
                &bytes[offset..],
                entry.tokenizer_state,
                tokenizer_scratch,
            ) {
                return None;
            }

            let actionable_terminals =
                actionable_terminals_from_language(template_runtime, entry.language);
            if offset == 0 && !entry.accumulator.is_empty() {
                entry.accumulator = prune_uniform_accumulator_for_parts(
                    constraint,
                    actionable_terminals.as_ref(),
                    &entry.accumulator,
                    entry.tokenizer_state,
                    &tokenizer_scratch.states,
                    &tokenizer_scratch.matches,
                )?;
            }

            let normalized_matches = collect_unique_actionable_matches(
                constraint,
                actionable_terminals.as_ref(),
                constraint.ignore_terminal,
                &tokenizer_scratch.matches,
                None,
            );
            let mut emitted = SmallVec::<[(usize, u32, TerminalsDisallowed); 4]>::new();

            for matched in normalized_matches {
                let new_offset = offset + matched.width;
                if new_offset > bytes.len() {
                    return None;
                }

                let (advanced_language, advanced_accumulator) = if matched.ignored {
                    (entry.language, entry.accumulator.clone())
                } else {
                    let started = profile.is_some().then(std::time::Instant::now);
                    let advanced = template_runtime.advance_language(
                        constraint,
                        matched.terminal_id,
                        entry.language,
                    )?;
                    if let (Some(profile), Some(started)) = (profile.as_deref_mut(), started) {
                        profile.evaluate_ns += started.elapsed().as_nanos() as u64;
                    }
                    if advanced == 0 {
                        continue;
                    }
                    (
                        advanced,
                        apply_future_terminal_disallow_to_accumulator(
                            constraint,
                            &tokenizer_scratch.states,
                            matched.terminal_id,
                            entry.accumulator.clone(),
                        ),
                    )
                };

                if emitted.iter().any(|(emitted_offset, language, accumulator)| {
                    *emitted_offset == new_offset
                        && *language == advanced_language
                        && accumulator == &advanced_accumulator
                }) {
                    continue;
                }
                emitted.push((new_offset, advanced_language, advanced_accumulator.clone()));

                let destination = if new_offset == bytes.len() {
                    &mut queue_scratch.language_pending
                } else {
                    &mut queue_scratch.language_processing[new_offset]
                };
                if !merge_small_language_parser_state(
                    template_runtime,
                    destination,
                    initial_tokenizer_state,
                    advanced_language,
                    advanced_accumulator,
                ) {
                    return None;
                }
            }

            if !tokenizer_scratch.states.is_empty() {
                let mut fallback_gss = None;
                for &end_state in &tokenizer_scratch.states {
                    let check_started = profile.is_some().then(std::time::Instant::now);
                    let language_viable = language_end_state_may_advance(
                        constraint,
                        template_runtime,
                        entry.language,
                        end_state,
                    );
                    if template_runtime.is_exhausted() {
                        return None;
                    }
                    let viable = match language_viable {
                        Some(viable) => viable,
                        None => {
                            let reconstruct_started =
                                profile.is_some().then(std::time::Instant::now);
                            let gss = fallback_gss.get_or_insert_with(|| {
                                template_runtime
                                    .gss_from_language(entry.language, entry.accumulator.clone())
                            });
                            if let (Some(profile), Some(started)) =
                                (profile.as_deref_mut(), reconstruct_started)
                            {
                                profile.continuation_reconstruct_ns +=
                                    started.elapsed().as_nanos() as u64;
                            }
                            end_state_may_advance(constraint, gss, end_state)
                        }
                    };
                    if let (Some(profile), Some(started)) = (profile.as_deref_mut(), check_started)
                    {
                        profile.continuation_check_ns += started.elapsed().as_nanos() as u64;
                    }
                    if viable
                        && !merge_small_language_parser_state(
                            template_runtime,
                            &mut queue_scratch.language_pending,
                            end_state,
                            entry.language,
                            entry.accumulator.clone(),
                        )
                    {
                        return None;
                    }
                }
            }
        }
        offset += 1;
    }

    if template_runtime.is_exhausted() {
        return None;
    }
    if queue_scratch.language_pending.is_empty() {
        return Some(Err("commit rejected: no valid parser states remain".to_string()));
    }
    Some(Ok(std::mem::take(&mut queue_scratch.language_pending)))
}

pub(super) fn commit_bytes_language_small_queue_fast_path(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
    queue_scratch: &mut SmallCommitQueueScratch,
    template_runtime: &mut TemplateAdvanceRuntime,
    profitability_prechecked: bool,
) -> Option<Result<(), String>> {
    if !language_small_queue_enabled() || !(2..=8).contains(&bytes.len()) || state.len() > 2 {
        return None;
    }

    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_LANGUAGE_SMALL_QUEUE").is_some();
    let top_values = language_queue_top_value_count_at_most(state, LANGUAGE_QUEUE_MIN_TOP_VALUES);
    if top_values < LANGUAGE_QUEUE_MIN_TOP_VALUES {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][language_small_queue_decline] reason=narrow_top_frontier bytes={} top_values={} min_top_values={}",
                format_token_bytes(bytes),
                top_values,
                LANGUAGE_QUEUE_MIN_TOP_VALUES,
            );
        }
        return None;
    }

    if !profitability_prechecked
        && !has_multiple_actionable_terminal_boundaries(constraint, state, bytes, tokenizer_scratch)
    {
        return None;
    }

    let parser_paths = language_queue_path_count_at_most(state, LANGUAGE_QUEUE_MIN_PATHS);
    if parser_paths < LANGUAGE_QUEUE_MIN_PATHS {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][language_small_queue_decline] reason=insufficient_stack_ambiguity bytes={} parser_paths={} min_paths={}",
                format_token_bytes(bytes),
                parser_paths,
                LANGUAGE_QUEUE_MIN_PATHS,
            );
        }
        return None;
    }
    let parser_nodes = language_queue_node_count_at_most(state, LANGUAGE_QUEUE_MIN_NODES);
    if parser_nodes < LANGUAGE_QUEUE_MIN_NODES {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][language_small_queue_decline] reason=insufficient_compact_gss_work bytes={} parser_paths_at_least={} parser_nodes={} min_nodes={}",
                format_token_bytes(bytes),
                LANGUAGE_QUEUE_MIN_PATHS,
                parser_nodes,
                LANGUAGE_QUEUE_MIN_NODES,
            );
        }
        return None;
    }
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][language_small_queue_dispatch] selected bytes={} state_entries={} parser_paths_at_least={} parser_nodes_at_least={}",
            format_token_bytes(bytes),
            state.len(),
            LANGUAGE_QUEUE_MIN_PATHS,
            LANGUAGE_QUEUE_MIN_NODES,
        );
    }
    if !language_queue_input_is_bounded(state) {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][language_small_queue_decline] reason=input_depth bytes={}",
                format_token_bytes(bytes),
            );
        }
        return None;
    }

    let total_started = profile_enabled.then(std::time::Instant::now);
    let mut profile = LanguageCommitSimulationProfile::default();
    template_runtime.begin_commit();
    let simulation = simulate_language_commit(
        constraint,
        state,
        bytes,
        tokenizer_scratch,
        queue_scratch,
        template_runtime,
        profile_enabled.then_some(&mut profile),
    );
    let Some(simulation) = simulation else {
        if profile_enabled {
            let reason = if template_runtime.is_exhausted() {
                "work_budget"
            } else {
                "simulation_bound_or_missing_template"
            };
            eprintln!(
                "[glrmask/profile][language_small_queue_decline] reason={} bytes={}",
                reason,
                format_token_bytes(bytes),
            );
        }
        return None;
    };
    let pending = match simulation {
        Ok(pending) => pending,
        // The language queue is an accelerator, not the authority for
        // rejection. Fall through to the established commit path so an
        // optimization bug cannot introduce a false negative.
        Err(_) => return None,
    };

    if template_runtime.is_exhausted() {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][language_small_queue_decline] reason=work_budget_after_simulation bytes={}",
                format_token_bytes(bytes),
            );
        }
        return None;
    }
    let mut final_reconstruct_ns = 0u64;
    let mut new_state = ParserStateMap::default();
    for entry in pending {
        let started = profile_enabled.then(std::time::Instant::now);
        let gss = template_runtime.gss_from_language(entry.language, entry.accumulator);
        if let Some(started) = started {
            final_reconstruct_ns += started.elapsed().as_nanos() as u64;
        }
        new_state.merge_insert(entry.tokenizer_state, gss);
    }
    for parser_state in new_state.values_mut() {
        *parser_state = parser_state.fuse(Some(1));
    }
    new_state.retain(|_, parser_state| !parser_state.is_empty());
    if new_state.is_empty() {
        return None;
    }
    if profile_enabled {
        let (
            template_calls,
            template_memo_hits,
            template_memo_entries,
            template_products_started,
            semantic_nodes,
            semantic_lower_keys,
            semantic_upper_keys,
            semantic_union_entries,
        ) = template_runtime.work_summary();
        eprintln!(
            "[glrmask/profile][language_small_queue] bytes={} total_ns={} canonicalize_ns={} evaluate_ns={} continuation_reconstruct_ns={} continuation_check_ns={} final_reconstruct_ns={} template_calls={} template_memo_hits={} template_memo_entries={} template_products_started={} semantic_nodes={} semantic_lower_keys={} semantic_upper_keys={} semantic_union_entries={} final_summaries={:?}",
            format_token_bytes(bytes),
            total_started.expect("language queue profile start exists").elapsed().as_nanos(),
            profile.canonicalize_ns,
            profile.evaluate_ns,
            profile.continuation_reconstruct_ns,
            profile.continuation_check_ns,
            final_reconstruct_ns,
            template_calls,
            template_memo_hits,
            template_memo_entries,
            template_products_started,
            semantic_nodes,
            semantic_lower_keys,
            semantic_upper_keys,
            semantic_union_entries,
            new_state.values().map(ParserGSS::summary).collect::<Vec<_>>(),
        );
    }
    *state = new_state;
    Some(Ok(()))
}

pub(super) fn try_advance_unique_actionable_top_fast(
    constraint: &Constraint,
    gss: &ParserGSS,
    terminal: u32,
) -> Option<ParserGSS> {
    if !constraint.table.control_terminals.is_empty() || template_advance_enabled() {
        return None;
    }
    // If every nonempty path has this top, isolate(Some(top)) is precisely
    // the original immutable GSS. Avoid enumerating/copying the top set and
    // retaining an extra Arc. The caller must not retry a declined action:
    // repeating it on this same GSS cannot make the shortcut more applicable.
    if let Some(top) = gss.single_exclusive_top_value() {
        let action = constraint.table.action(top, terminal)?;
        return apply_single_top_action_fast(constraint, gss, top, terminal, action);
    }
    let mut selected = None;
    for top in gss.peek_values() {
        let Some(action) = constraint.table.action(top, terminal) else {
            continue;
        };
        if selected.is_some() {
            return None;
        }
        selected = Some((top, action));
    }
    let (top, action) = selected?;
    let isolated = gss.isolate(Some(top));
    (!isolated.is_empty())
        .then(|| apply_single_top_action_fast(constraint, &isolated, top, terminal, action))
        .flatten()
}

pub(super) fn try_batch_same_width_disjoint_alias_actions(
    constraint: &Constraint,
    gss: &ParserGSS,
    matches: &[NormalizedMatch],
    width: usize,
    continuation_states: &[u32],
) -> Option<ParserGSS> {
    if !constraint.table.control_terminals.is_empty() || template_advance_enabled() {
        return None;
    }
    let group = matches
        .iter()
        .filter(|matched| matched.width == width && !matched.ignored)
        .collect::<SmallVec<[&NormalizedMatch; 16]>>();
    if group.len() < 2 {
        return None;
    }
    for matched in &group {
        if continuation_states.iter().any(|&state| {
            constraint
                .tokenizer
                .possible_future_terminals(state)
                .contains(matched.terminal_id as usize)
        }) {
            return None;
        }
    }

    let mut terminal_by_top = SmallVec::<[(u32, u32); 8]>::new();
    for top in gss.peek_values() {
        let mut selected = None;
        for matched in &group {
            if constraint.table.action(top, matched.terminal_id).is_none() {
                continue;
            }
            if selected.is_some() {
                return None;
            }
            selected = Some(matched.terminal_id);
        }
        if let Some(terminal) = selected {
            terminal_by_top.push((top, terminal));
        }
    }
    if terminal_by_top.len() < 2 {
        return None;
    }
    advance_stacks_disjoint_top_terminals_bounded(&constraint.table, gss, &terminal_by_top)
}

pub(super) fn try_batch_same_width_pure_matches(
    constraint: &Constraint,
    gss: &ParserGSS,
    matches: &[NormalizedMatch],
    width: usize,
    continuation_states: &[u32],
) -> Option<ParserGSS> {
    if !constraint.table.control_terminals.is_empty() {
        return None;
    }
    let group = matches
        .iter()
        .filter(|matched| matched.width == width && !matched.ignored)
        .collect::<SmallVec<[&NormalizedMatch; 16]>>();
    if group.len() < 2 {
        return None;
    }

    // The historical per-terminal path adds delayed longest-match exclusions
    // only when that same logical terminal remains possible after the model
    // token. Batch only when that transform is identity for every member.
    for matched in &group {
        if continuation_states.iter().any(|&state| {
            constraint
                .tokenizer
                .possible_future_terminals(state)
                .contains(matched.terminal_id as usize)
        }) {
            return None;
        }
    }

    let tops = gss.peek_values();
    let mut shifts = SmallVec::<[(u32, u32, bool); 32]>::new();
    for &top in &tops {
        for matched in &group {
            let Some(action) = constraint.table.action(top, matched.terminal_id) else {
                continue;
            };
            match action {
                Action::Shift(target, replace) => {
                    let edge = (top, *target, *replace);
                    if !shifts.contains(&edge) {
                        shifts.push(edge);
                    }
                }
                Action::ReplaceShifts(targets) => {
                    for &target in targets.iter() {
                        let edge = (top, target, true);
                        if !shifts.contains(&edge) {
                            shifts.push(edge);
                        }
                    }
                }
                Action::StackShifts(stack_shifts) => {
                    for shift in stack_shifts {
                        if shift.pushes.len() != 1 || shift.pop > 1 {
                            return None;
                        }
                        let edge = (top, shift.pushes[0], shift.pop == 1);
                        if !shifts.contains(&edge) {
                            shifts.push(edge);
                        }
                    }
                }
                _ => return None,
            }
        }
    }
    if shifts.is_empty() {
        return None;
    }

    let advanced = if tops.len() == 1
        && shifts.iter().all(|(top, _, _)| *top == tops[0])
        && shifts
            .first()
            .is_some_and(|(_, _, replace)| shifts.iter().all(|(_, _, other)| other == replace))
        && shifts.iter().enumerate().all(|(index, (_, target, _))| {
            !shifts[..index].iter().any(|(_, prior_target, _)| prior_target == target)
        })
        && let Some(stack) = gss.try_virtual_stack()
    {
        let replace = shifts[0].2;
        stack
            .into_gss_after_popping_and_pushing_unique_single_branches(
                usize::from(replace),
                shifts.iter().map(|(_, target, _)| target),
            )
            .unwrap_or_else(|| gss.apply_top_pure_shifts(shifts.clone()))
    } else {
        gss.apply_top_pure_shifts(shifts)
    };
    (!advanced.is_empty()).then_some(advanced)
}

pub(super) fn commit_bytes_small_queue_fast_path(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    bytes: &[u8],
    tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
    queue_scratch: &mut SmallCommitQueueScratch,
    admission_cache: &mut SmallVec<[ParserAdmissionCacheEntry; 8]>,
    prune_tokenizer_scratch: &mut tokenizer_scan::ReusableTokenizerExecScratch,
) -> Option<Result<(), String>> {
    let has_linker_controls = !constraint.table.control_terminals.is_empty()
        || constraint.uses_compact_segmented_parser_runtime();
    if bytes.len() > 16 || state.len() > 8 {
        return None;
    }
    queue_scratch.clear();
    for (&tokenizer_state, gss) in state.iter() {
        merge_small_parser_state(&mut queue_scratch.processing[0], tokenizer_state, gss.clone());
    }

    let initial_tokenizer_state = constraint.runtime_commit_initial_state();
    let mut offset = 0usize;
    while offset <= bytes.len() {
        if queue_scratch.processing[offset].is_empty() {
            offset += 1;
            continue;
        }

        let states_to_process = std::mem::take(&mut queue_scratch.processing[offset]);
        let mut groups = SmallVec::<[(ParserGSS, SmallVec<[u32; 8]>); 8]>::new();
        for (tokenizer_state, gss) in states_to_process {
            let can_group = gss.all_accs_satisfy(|acc: &TerminalsDisallowed| acc.is_empty());
            let mut grouped = false;
            if can_group {
                'groups: for (existing_gss, tokenizer_states) in &mut groups {
                    if !existing_gss.ptr_eq(&gss) {
                        continue;
                    }
                    let future = constraint.tokenizer.possible_future_terminals(tokenizer_state);
                    for &other_state in tokenizer_states.iter() {
                        if !future.is_disjoint(
                            constraint.tokenizer.possible_future_terminals(other_state),
                        ) {
                            continue 'groups;
                        }
                    }
                    tokenizer_states.push(tokenizer_state);
                    grouped = true;
                    break;
                }
            }
            if !grouped {
                groups.push((gss, smallvec::smallvec![tokenizer_state]));
            }
        }

        for (mut gss_at_offset, tokenizer_states) in groups {
            let tokenizer_state = tokenizer_states[0];
            let reused_prune_scan = offset == 0
                && !queue_scratch.prune_union_starts.is_empty()
                && queue_scratch.prune_union_starts.as_slice() == tokenizer_states.as_slice();
            if reused_prune_scan {
                tokenizer_scratch.states.clear();
                tokenizer_scratch.states.extend(prune_tokenizer_scratch.states.iter().copied());
                tokenizer_scratch.matches.clear();
                tokenizer_scratch.matches.extend(prune_tokenizer_scratch.matches.iter().cloned());
            } else if !execute_tokenizer_reusable_from_states(
                constraint,
                &bytes[offset..],
                &tokenizer_states,
                tokenizer_scratch,
            ) {
                return None;
            }

            if offset == 0
                && !gss_at_offset.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
            {
                gss_at_offset = try_prune_single_initial_state_batched_accumulators(
                    constraint,
                    &gss_at_offset,
                    bytes,
                    prune_tokenizer_scratch,
                    &mut queue_scratch.prune_union_starts,
                )
                .unwrap_or_else(|| {
                    prune_single_initial_state_for_parts(
                        constraint,
                        gss_at_offset,
                        tokenizer_state,
                        &tokenizer_scratch.states,
                        &tokenizer_scratch.matches,
                        bytes,
                    )
                });
                if gss_at_offset.is_empty() {
                    continue;
                }
            }

            let (actionable_terminals, normalized_matches) = if tokenizer_scratch.matches.is_empty()
            {
                (None, SmallVec::<[NormalizedMatch; 16]>::new())
            } else {
                let actionable_terminals = (!has_linker_controls)
                    .then(|| ActionableTerminals::from_gss(constraint, &gss_at_offset))
                    .flatten();
                let normalized_matches = collect_unique_actionable_reusable_matches(
                    constraint,
                    actionable_terminals.as_ref(),
                    constraint.ignore_terminal,
                    &tokenizer_scratch.matches,
                );
                (actionable_terminals, normalized_matches)
            };
            let mut emitted_terminal_outputs = SmallVec::<[(usize, ParserGSS); 4]>::new();
            let mut batched_widths = SmallVec::<[usize; 4]>::new();
            for matched in &normalized_matches {
                if matched.ignored || batched_widths.contains(&matched.width) {
                    continue;
                }
                let batched = try_batch_same_width_pure_matches(
                    constraint,
                    &gss_at_offset,
                    &normalized_matches,
                    matched.width,
                    &tokenizer_scratch.states,
                )
                .or_else(|| {
                    try_batch_same_width_disjoint_alias_actions(
                        constraint,
                        &gss_at_offset,
                        &normalized_matches,
                        matched.width,
                        &tokenizer_scratch.states,
                    )
                });
                if let Some(advanced) = batched {
                    let new_offset = offset + matched.width;
                    if new_offset > bytes.len() {
                        return None;
                    }
                    if emitted_terminal_outputs.iter().any(|(emitted_offset, emitted_gss)| {
                        *emitted_offset == new_offset && emitted_gss == &advanced
                    }) {
                        batched_widths.push(matched.width);
                        continue;
                    }
                    emitted_terminal_outputs.push((new_offset, advanced.clone()));
                    if new_offset == bytes.len() {
                        merge_small_parser_state(
                            &mut queue_scratch.pending,
                            initial_tokenizer_state,
                            advanced,
                        );
                    } else {
                        merge_small_parser_state(
                            &mut queue_scratch.processing[new_offset],
                            initial_tokenizer_state,
                            advanced,
                        );
                    }
                    batched_widths.push(matched.width);
                }
            }

            for matched in normalized_matches {
                let new_offset = offset + matched.width;
                if !matched.ignored && batched_widths.contains(&matched.width) {
                    continue;
                }
                if new_offset > bytes.len() {
                    return None;
                }

                if matched.ignored {
                    if new_offset == bytes.len() {
                        merge_small_parser_state(
                            &mut queue_scratch.pending,
                            initial_tokenizer_state,
                            gss_at_offset.clone(),
                        );
                    } else {
                        merge_small_parser_state(
                            &mut queue_scratch.processing[new_offset],
                            initial_tokenizer_state,
                            gss_at_offset.clone(),
                        );
                    }
                    continue;
                }

                let advanced = if !has_linker_controls
                    && !template_advance_enabled()
                    && let Some(advanced) = try_advance_unique_actionable_top_fast(
                        constraint,
                        &gss_at_offset,
                        matched.terminal_id,
                    ) {
                    advanced
                } else {
                    let Some(advanced) = advance_parser_stacks_if_possible(
                        constraint,
                        &gss_at_offset,
                        matched.terminal_id,
                    ) else {
                        continue;
                    };
                    advanced
                };
                let advanced = apply_future_terminal_disallow_for_states(
                    constraint,
                    &tokenizer_scratch.states,
                    matched.terminal_id,
                    advanced,
                );
                if advanced.is_empty() {
                    continue;
                }
                if emitted_terminal_outputs.iter().any(|(emitted_offset, emitted_gss)| {
                    *emitted_offset == new_offset && emitted_gss == &advanced
                }) {
                    continue;
                }
                emitted_terminal_outputs.push((new_offset, advanced.clone()));
                if new_offset == bytes.len() {
                    merge_small_parser_state(
                        &mut queue_scratch.pending,
                        initial_tokenizer_state,
                        advanced,
                    );
                } else {
                    merge_small_parser_state(
                        &mut queue_scratch.processing[new_offset],
                        initial_tokenizer_state,
                        advanced,
                    );
                }
            }

            let local_row_admission = try_local_row_presence_admission_words(
                constraint,
                &gss_at_offset,
                &tokenizer_scratch.states,
            );
            let admission_cache_index = if local_row_admission.is_none() {
                cached_batched_end_state_admission(
                    constraint,
                    &gss_at_offset,
                    &tokenizer_scratch.states,
                    admission_cache,
                )
            } else {
                None
            };
            for &end_state in &tokenizer_scratch.states {
                let may_advance = if let Some(words) = local_row_admission.as_ref() {
                    end_state_may_advance_from_row_words(constraint, end_state, words)
                } else if let Some(index) = admission_cache_index {
                    end_state_may_advance_from_cache_entry(
                        constraint,
                        end_state,
                        &admission_cache[index],
                    )
                } else {
                    cached_single_end_state_may_advance(
                        constraint,
                        &gss_at_offset,
                        end_state,
                        admission_cache,
                    )
                };
                if may_advance {
                    merge_small_parser_state(
                        &mut queue_scratch.pending,
                        end_state,
                        gss_at_offset.clone(),
                    );
                }
            }
        }
        offset += 1;
    }

    let mut new_state = ParserStateMap::default();
    let mut fused_by_source = SmallVec::<[(ParserGSS, ParserGSS); 8]>::new();
    for (tokenizer_state, parser_state) in queue_scratch.pending.drain(..) {
        let mut fused = fused_by_source
            .iter()
            .find(|(source, _)| source.ptr_eq(&parser_state))
            .map(|(_, fused)| fused.clone())
            .unwrap_or_else(|| {
                let source = parser_state.clone();
                let fused = parser_state.fuse(Some(1));
                fused_by_source.push((source, fused.clone()));
                fused
            });
        if let Some((_, canonical)) =
            fused_by_source.iter().find(|(_, candidate)| *candidate == fused)
        {
            fused = canonical.clone();
        }
        if !fused.is_empty() {
            if let Some(_uniform) = fused.uniform_accumulator() {
                new_state.insert_flat_alternative(tokenizer_state, fused);
            } else {
                let mut accumulators = SmallVec::<[TerminalsDisallowed; 4]>::new();
                let mut overflow = false;
                fused.for_each_acc(|acc| {
                    if overflow || accumulators.contains(acc) {
                        return;
                    }
                    if accumulators.len() == accumulators.capacity() {
                        overflow = true;
                        return;
                    }
                    accumulators.push(acc.clone());
                });
                if overflow || accumulators.len() <= 1 {
                    new_state.insert_flat_alternative(tokenizer_state, fused);
                } else {
                    for acc in accumulators {
                        let part = fused.apply_and_prune_no_promote(|candidate| {
                            (candidate == &acc).then_some(candidate.clone())
                        });
                        if !part.is_empty() {
                            new_state.insert_flat_alternative(tokenizer_state, part);
                        }
                    }
                }
            }
        }
    }
    if new_state.is_empty() {
        return Some(Err("commit rejected: no valid parser states remain".to_string()));
    }
    *state = new_state;
    Some(Ok(()))
}
