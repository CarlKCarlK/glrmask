//! Discover exact crossing-token paths and their terminal automata.

use crate::automata::lexer::Lexer;
use super::{
    Action, AnalyzedGrammar, Arc, AtomicU64, AtomicUsize, BTreeMap, BTreeSet, BitSet,
    BoundaryParserWork, BoundaryShardWork, CompositionGrammarSummary, ConcreteBoundaryDeltaPlan,
    Constraint, DWA, DWAState, Expr, FxHashMap, FxHashSet, Instant, InternalIdMap, ManyToOneIdMap,
    MappedArtifact, NWA, Ordering, RangeSetBlaze, SpecialTokenTerminal, Symbol, Templates,
    TerminalAutomaton, Tokenizer, U8Set, UnweightedDfa, VecDeque, Vocab, Weight,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count,
    characterize_finish_probe, characterize_selected_terminals_for_terminal_count,
    compile_terminal_expr_dfa, compose_profile_enabled, compute_ever_allowed_follows, determinize,
    find_difference, macro_parallelism_disabled, minimize_owned,
    normalize_weighted_parser_stack_nwa, report_macro_item_timings, resolve_negative_codes_in_nwa,
    reverse_hashcons_owned, tokenizer_tsid_relation_is_singleton,
    vocab_tokens_with_adjacent_pairs,
};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct BoundaryTokenNodeKey {
    pub(super) offset: usize,
    /// `u32::MAX` means no parser-visible terminal has committed yet.
    /// Globally erased trivia does not update this field.
    pub(super) last_terminal: u32,
    /// Whether the first non-globally-erased terminal of this model token is
    /// a FIRST/FOLLOW boundary seed. Later seed terminals do not establish a
    /// seed-only boundary witness; they require an actual interface crossing.
    pub(super) seeded: bool,
    /// Whether two concrete parser-visible terminals on this path witness an
    /// actual boundary interface pair. This is independently sufficient
    /// boundary evidence; the LR table later decides the full parser language.
    pub(super) interface_witnessed: bool,
    /// Whether this token path has consumed at least one globally erased
    /// terminal before the current point. This is the only multi-byte seed-only
    /// case not already represented by the reset one-terminal relation.
    pub(super) erased_seen: bool,
    /// False only for the arbitrary-residual first fragment. Once any
    /// terminal commits, subsequent fragments start from lexer reset.
    pub(super) started: bool,
}

#[derive(Debug, Clone)]
pub(super) struct BoundaryTokenEdge {
    pub(super) target: usize,
    pub(super) terminal: u32,
}

#[derive(Debug, Clone)]
pub(super) struct BoundaryTokenNode {
    pub(super) key: BoundaryTokenNodeKey,
    pub(super) outgoing: Vec<BoundaryTokenEdge>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct ResidualScanResult {
    /// Per-start-state longest matches. Different residual starts may produce
    /// different valid widths for the same terminal. These collections are
    /// tiny in practice; sorted vectors avoid one tree allocation per scan.
    pub(super) matches: Vec<(u32, usize)>,
    pub(super) future_terminals: Vec<u32>,
}

impl ResidualScanResult {
    pub(super) fn union_from(&mut self, other: &Self) {
        self.matches.extend_from_slice(&other.matches);
        self.future_terminals.extend_from_slice(&other.future_terminals);
    }

    pub(super) fn canonicalize(&mut self) {
        self.matches.sort_unstable();
        self.matches.dedup();
        self.future_terminals.sort_unstable();
        self.future_terminals.dedup();
    }
}

#[derive(Clone)]
pub(super) struct BoundaryTokenWitness {
    pub(super) token_id: u32,
    pub(super) start_states: Vec<u32>,
    pub(super) nodes: Vec<BoundaryTokenNode>,
    pub(super) good: Vec<bool>,
    pub(super) accepting: Vec<bool>,
}

#[derive(Clone)]
pub(super) struct BoundaryTokenDiscovery {
    pub(super) terminals: BitSet,
    pub(super) token_ids: Vec<u32>,
    pub(super) witnesses: Vec<BoundaryTokenWitness>,
}

pub(super) fn boundary_tokens_by_start_component(
    discovery: &BoundaryTokenDiscovery,
    terminal_offsets: &[u32],
    tokenizer_state_offsets: &[u32],
    component_count: usize,
) -> Vec<Vec<u32>> {
    let terminal_owner = |terminal: u32| -> Option<usize> {
        let component = terminal_offsets
            .partition_point(|&offset| offset <= terminal)
            .checked_sub(1)?;
        (component < component_count).then_some(component)
    };
    let tokenizer_state_owner = |state: u32| -> Option<usize> {
        if state == 0 {
            return None;
        }
        let component = tokenizer_state_offsets
            .partition_point(|&offset| offset <= state)
            .checked_sub(1)?;
        (component < component_count).then_some(component)
    };

    let mut by_component = vec![BTreeSet::<u32>::new(); component_count];
    for witness in &discovery.witnesses {
        let mut owners = BTreeSet::<usize>::new();
        for &state in &witness.start_states {
            if let Some(owner) = tokenizer_state_owner(state) {
                owners.insert(owner);
                continue;
            }

            // Raw state zero is the merged reset dispatcher rather than a
            // component-local tokenizer state.  At that coordinate the first
            // productive terminal in the exact witness chooses the component
            // in which token execution starts.  A witness may have more than
            // one such first owner; publish the token conservatively to each.
            if state == 0
                && let Some(root) = witness.nodes.first()
            {
                for edge in root
                    .outgoing
                    .iter()
                    .filter(|edge| witness.good.get(edge.target).copied().unwrap_or(false))
                {
                    if let Some(owner) = terminal_owner(edge.terminal) {
                        owners.insert(owner);
                    }
                }
            }
        }
        for owner in owners {
            by_component[owner].insert(witness.token_id);
        }
    }

    by_component
        .into_iter()
        .map(|tokens| tokens.into_iter().collect())
        .collect()
}

#[derive(Clone)]
pub(super) struct PreTableBoundaryBaseDiscovery {
    pub(super) terminal_offsets: Vec<u32>,
    pub(super) prefer_cross_interface_only: bool,
    pub(super) tokenizer_state_offsets: Vec<u32>,
    pub(super) summary: CompositionGrammarSummary,
    pub(super) base_interface_pairs: BTreeSet<(u32, u32)>,
    pub(super) discovery_interface_pairs: BTreeSet<(u32, u32)>,
    pub(super) discovery: BoundaryTokenDiscovery,
    pub(super) component_state_map: Option<ManyToOneIdMap>,
    pub(super) terminal_artifact: Option<MappedArtifact<TerminalAutomaton>>,
    pub(super) elapsed_ms: f64,
}

pub(super) fn replace_boundary_discovery_tokens(
    mut base: BoundaryTokenDiscovery,
    replacement: BoundaryTokenDiscovery,
) -> BoundaryTokenDiscovery {
    if replacement.token_ids.is_empty() {
        return base;
    }
    let replaced = replacement
        .token_ids
        .iter()
        .copied()
        .collect::<FxHashSet<_>>();
    base.token_ids.retain(|token| !replaced.contains(token));
    base.witnesses
        .retain(|witness| !replaced.contains(&witness.token_id));
    base.terminals.union_with(&replacement.terminals);
    base.token_ids.extend(replacement.token_ids);
    base.token_ids.sort_unstable();
    base.token_ids.dedup();
    base.witnesses.extend(replacement.witnesses);
    base
}


/// Build the exact reusable boundary-trigger Parser DWA for one component.
///
/// The trigger's weight coordinate is deliberately private and unquotiented:
/// raw local tokenizer-state IDs x original model-token IDs.  Ordinary
/// component TSID/token quotients are whole-token equivalences and need not
/// preserve proper-prefix boundary observations.
///
/// `candidate_tokens` may be any conservative superset. The current caller
/// feeds the cheaper Tokens trigger, which is built by scanning every raw lexer
/// state and therefore cannot remove a real proper-prefix event.
pub(crate) fn build_exact_component_boundary_trigger(
    constraint: &Constraint,
    candidate_tokens: &[u32],
) -> Result<Option<DWA>, String> {
    // Exact trigger semantics observe parser readiness *after* zero-width
    // linker closure. Compile that closure into a private table copy when this
    // reusable component is itself composed. Control elimination preserves LR
    // state IDs, so the trigger remains in the component-local parser-state
    // coordinate and the live component/table is never mutated.
    let controls_eliminated = !constraint.table.control_terminals.is_empty();
    let trigger_table_storage = if controls_eliminated {
        let mut table = constraint.table.clone();
        table.eliminate_control_terminals_exact()?;
        Some(table)
    } else {
        None
    };
    let trigger_table = trigger_table_storage.as_ref().unwrap_or(&constraint.table);

    let mut placeholders = constraint
        .unbound_grammar_placeholders
        .values()
        .copied()
        .chain(constraint.late_grammar_slots.iter().map(|slot| slot.terminal_id))
        .collect::<Vec<_>>();
    placeholders.sort_unstable();
    placeholders.dedup();
    placeholders.retain(|&terminal| terminal < trigger_table.num_terminals);

    let finish_terminal = trigger_table.num_terminals;
    let extended_terminal_count = finish_terminal
        .checked_add(1)
        .ok_or_else(|| "boundary trigger terminal-count overflow".to_owned())?;
    let mut event_terminals = placeholders.clone();
    event_terminals.push(finish_terminal);

    let mut terminal_nwa = NWA::new(
        constraint.tokenizer.num_states(),
        constraint.max_original_token_id().unwrap_or(0),
    );
    let global_start = terminal_nwa.add_state();
    let accept = terminal_nwa.add_state();
    terminal_nwa.set_start_states(vec![global_start]);
    terminal_nwa.set_final_weight(accept, Weight::all());

    let mut used_terminals = BTreeSet::<u32>::new();
    used_terminals.extend(event_terminals.iter().copied());
    let reset = constraint.runtime_commit_initial_state();
    let mut accepted_pairs = 0usize;

    for &token_id in candidate_tokens {
        let Some(bytes) = constraint.token_bytes_for_id(token_id) else {
            continue;
        };
        if bytes.len() < 2 {
            continue;
        }

        // Every raw tokenizer state is semantically distinct in this trigger
        // coordinate. Equal lexical DAGs may be hash-consed later; correctness
        // does not rely on the ordinary component TSID quotient.
        for raw_start in 0..constraint.tokenizer.num_states() {
            // Discover the tiny byte-offset DAG first. Do not allocate NWA
            // states for the overwhelmingly common `(raw state, token)` pairs
            // that have no proper internal lexeme boundary at all.
            let mut edges_by_offset = BTreeMap::<usize, Vec<(u32, usize)>>::new();
            let mut seen_offsets = BTreeSet::from([0usize]);
            let mut queue = VecDeque::from([0usize]);
            while let Some(offset) = queue.pop_front() {
                let lexer_start = if offset == 0 { raw_start } else { reset };
                let execution = constraint
                    .tokenizer
                    .execute_from_state(&bytes[offset..], lexer_start);
                for matched in execution.matches {
                    if matched.width == 0 {
                        continue;
                    }
                    let Some(next_offset) = offset.checked_add(matched.width) else {
                        continue;
                    };
                    // An event at model-token end is handled before the next
                    // mask call and must not appear in the internal trigger.
                    if next_offset >= bytes.len() {
                        continue;
                    }
                    edges_by_offset
                        .entry(offset)
                        .or_default()
                        .push((matched.id, next_offset));
                    if seen_offsets.insert(next_offset) {
                        queue.push_back(next_offset);
                    }
                }
            }
            if edges_by_offset.is_empty() {
                continue;
            }
            for edges in edges_by_offset.values_mut() {
                edges.sort_unstable();
                edges.dedup();
            }

            let mut state_by_offset = BTreeMap::<usize, u32>::new();
            for &offset in &seen_offsets {
                state_by_offset.insert(offset, terminal_nwa.add_state());
            }
            let root = state_by_offset[&0];
            let support = Weight::from_per_tsid_token_sets(std::iter::once((
                raw_start,
                std::iter::once(token_id).collect::<RangeSetBlaze<u32>>(),
            )));
            terminal_nwa.add_epsilon(global_start, root, support);
            for (&offset, &state) in &state_by_offset {
                if offset != 0 {
                    for &event in &event_terminals {
                        terminal_nwa.add_transition(state, event as i32, accept, Weight::all());
                    }
                }
            }
            for (offset, edges) in edges_by_offset {
                let source = state_by_offset[&offset];
                for (terminal, next_offset) in edges {
                    terminal_nwa.add_transition(
                        source,
                        terminal as i32,
                        state_by_offset[&next_offset],
                        Weight::all(),
                    );
                    used_terminals.insert(terminal);
                }
            }
            accepted_pairs += 1;
        }
    }

    if accepted_pairs == 0 {
        return Ok(Some(DWA::new(
            constraint.tokenizer.num_states(),
            constraint.max_original_token_id().unwrap_or(0),
        )));
    }

    // Reconstruct ordinary terminal stack relations. For a control-bearing
    // component cached templates describe the pre-closure table, so they are
    // deliberately ignored: characterize the trigger-only control-eliminated
    // table directly. For ordinary components retain the cheaper artifact
    // reuse/reconstruction path.
    let mut by_terminal = BTreeMap::<u32, UnweightedDfa>::new();
    if controls_eliminated {
        let mut selected = vec![false; trigger_table.num_terminals as usize];
        for &terminal in &used_terminals {
            if terminal != finish_terminal
                && let Some(slot) = selected.get_mut(terminal as usize)
            {
                *slot = true;
            }
        }
        let characterizations = characterize_selected_terminals_for_terminal_count(
            trigger_table,
            trigger_table.num_terminals,
            &selected,
        );
        if used_terminals
            .iter()
            .copied()
            .filter(|&terminal| terminal != finish_terminal)
            .any(|terminal| !characterizations.contains_key(&terminal))
        {
            return Ok(None);
        }
        by_terminal.extend(Templates::from_characterizations(&characterizations).by_terminal);
    } else {
        for &terminal in &used_terminals {
            if terminal == finish_terminal {
                continue;
            }
            if let Some(dfa) = constraint
                .composition_parser_templates_by_terminal
                .get(terminal as usize)
                .and_then(Option::as_ref)
            {
                by_terminal.insert(terminal, dfa.clone());
            }
        }

        if used_terminals
            .iter()
            .copied()
            .filter(|&terminal| terminal != finish_terminal)
            .any(|terminal| !by_terminal.contains_key(&terminal))
            && let Some(direct) =
                Templates::from_direct_regular_table(trigger_table, trigger_table.num_terminals)
        {
            for (terminal, dfa) in direct.by_terminal {
                if used_terminals.contains(&terminal) {
                    by_terminal.entry(terminal).or_insert(dfa);
                }
            }
        }

        let missing = used_terminals
            .iter()
            .copied()
            .filter(|&terminal| terminal != finish_terminal && !by_terminal.contains_key(&terminal))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            let mut characterizations = missing
                .iter()
                .filter_map(|&terminal| {
                    constraint
                        .composition_parser_characterizations_by_terminal
                        .get(terminal as usize)
                        .and_then(Option::as_ref)
                        .cloned()
                        .map(|characterization| (terminal, characterization))
                })
                .collect::<BTreeMap<_, _>>();
            if characterizations.len() != missing.len() {
                let mut selected = vec![false; trigger_table.num_terminals as usize];
                for &terminal in &missing {
                    if let Some(slot) = selected.get_mut(terminal as usize) {
                        *slot = true;
                    }
                }
                characterizations = characterize_selected_terminals_for_terminal_count(
                    trigger_table,
                    trigger_table.num_terminals,
                    &selected,
                );
            }
            if characterizations.len() != missing.len() {
                return Ok(None);
            }
            let rebuilt = Templates::from_characterizations(&characterizations);
            by_terminal.extend(rebuilt.by_terminal);
        }
    }

    if used_terminals
        .iter()
        .copied()
        .filter(|&terminal| terminal != finish_terminal)
        .any(|terminal| !by_terminal.contains_key(&terminal))
    {
        return Ok(None);
    }

    let finish_characterizations = BTreeMap::from([(
        finish_terminal,
        characterize_finish_probe(trigger_table),
    )]);
    let mut finish_templates = Templates::from_characterizations(&finish_characterizations);
    let Some(finish_dfa) = finish_templates.by_terminal.remove(&finish_terminal) else {
        return Ok(None);
    };
    by_terminal.insert(finish_terminal, finish_dfa);
    let templates = Templates::from_terminal_dfas(by_terminal);

    let terminal_automaton = TerminalAutomaton::EpsilonNwa(terminal_nwa);
    let Some(mut parser_nwa) =
        build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count(
            &terminal_automaton,
            extended_terminal_count,
            &templates,
            trigger_table,
        )
    else {
        return Ok(Some(DWA::new(
            constraint.tokenizer.num_states(),
            constraint.max_original_token_id().unwrap_or(0),
        )));
    };

    resolve_negative_codes_in_nwa(
        &mut parser_nwa,
        trigger_table.construction
            == crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
    );
    Ok(Some(normalize_weighted_parser_stack_nwa(
        trigger_table,
        &parser_nwa,
    )))
}

pub(super) fn scan_residual_starts(
    tokenizer: &Tokenizer,
    bytes: &[u8],
    starts: &[u32],
) -> ResidualScanResult {
    let mut result = ResidualScanResult::default();
    for &start in starts {
        let execution = tokenizer.execute_from_state(bytes, start);
        result.matches.extend(
            execution
                .matches
                .into_iter()
                .filter(|matched| matched.width > 0)
                .map(|matched| (matched.id, matched.width)),
        );
        for end_state in execution.end_state {
            result.future_terminals.extend(
                tokenizer
                    .possible_future_terminals_iter(end_state),
            );
        }
    }
    result.matches.sort_unstable();
    result.matches.dedup();
    result.future_terminals.sort_unstable();
    result.future_terminals.dedup();
    result
}


pub(super) fn component_tokenizer_state_layout(components: &[&Constraint]) -> (Vec<u32>, usize) {
    let mut next_state = 1u32; // Fresh merged epsilon dispatcher.
    let mut offsets = Vec::with_capacity(components.len());
    for component in components {
        offsets.push(next_state);
        next_state = next_state
            .checked_add(component.composition_tokenizer().num_states())
            .expect("composed tokenizer state count overflow");
    }
    (offsets, next_state as usize)
}

/// Reconstruct the merged terminal-expression sidecar when one or more loaded
/// components kept source expressions deferred outside their tokenizers.
/// Fresh components normally let `Tokenizer` merge these directly, so callers
/// only need this when the merged tokenizer otherwise has no expression list.
pub(crate) fn merged_retained_terminal_exprs(
    components: &[&Constraint],
    terminal_offsets: &[u32],
    total_terminals: u32,
) -> Option<Vec<crate::automata::regex::Expr>> {
    if components.len() != terminal_offsets.len() {
        return None;
    }
    let mut merged = vec![None; total_terminals as usize];
    for (component, &offset) in components.iter().zip(terminal_offsets) {
        let exprs = component.retained_terminal_exprs()?;
        if exprs.len() != component.tokenizer.num_terminals() as usize {
            return None;
        }
        for (local_terminal, expr) in exprs.iter().enumerate() {
            let global = offset as usize + local_terminal;
            let slot = merged.get_mut(global)?;
            if slot.is_some() {
                return None;
            }
            *slot = Some(expr.clone());
        }
    }
    merged.into_iter().collect()
}

pub(super) fn component_tokenizer_state_layout_owned_parent(
    components: &[&Constraint],
) -> (Vec<u32>, usize) {
    let mut next_state = 0u32;
    let mut offsets = Vec::with_capacity(components.len());
    for component in components {
        offsets.push(next_state);
        next_state = next_state
            .checked_add(component.composition_tokenizer().num_states())
            .expect("composed tokenizer state count overflow");
    }
    (offsets, next_state as usize)
}

pub(super) fn composite_reset_states(
    _components: &[&Constraint],
    _tokenizer_state_offsets: &[u32],
) -> Vec<u32> {
    // The disjoint-union tokenizer resets to its fresh dispatcher. Executing
    // from state zero epsilon-dispatches to every component start state.
    vec![0]
}

pub(super) fn expanded_component_reset_states(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
) -> Vec<u32> {
    let mut resets = components
        .iter()
        .zip(tokenizer_state_offsets)
        .map(|(component, offset)| offset + component.composition_tokenizer().start_state())
        .collect::<Vec<_>>();
    resets.sort_unstable();
    resets.dedup();
    resets
}

pub(super) fn component_reset_live_bytes(components: &[&Constraint]) -> Vec<U8Set> {
    components
        .iter()
        .map(|component| {
            let tokenizer = component.composition_tokenizer();
            let closures = tokenizer.all_singleton_epsilon_closures();
            let mut bytes = U8Set::empty();
            for &state in &closures[tokenizer.start_state() as usize] {
                for (byte, _) in tokenizer.transitions_from(state) {
                    bytes.insert(byte);
                }
            }
            bytes
        })
        .collect()
}

pub(super) fn scan_component_residual_starts(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    reset_live_bytes: &[U8Set],
    bytes: &[u8],
    starts: &[u32],
) -> ResidualScanResult {
    debug_assert_eq!(components.len(), tokenizer_state_offsets.len());
    debug_assert_eq!(components.len(), terminal_offsets.len());
    debug_assert_eq!(components.len(), reset_live_bytes.len());
    let mut result = ResidualScanResult::default();

    let mut scan_local = |component_index: usize, local_start: u32| {
        let component = components[component_index];
        let tokenizer = component.composition_tokenizer();
        let terminal_offset = terminal_offsets[component_index];
        let (end_states, matches) = tokenizer.execute_summary_from_state(bytes, local_start);
        result.matches.extend(
            matches
                .into_iter()
                .filter(|(_, width)| *width > 0)
                .map(|(terminal, width)| (terminal_offset + terminal, width)),
        );
        for end_state in end_states {
            result.future_terminals.extend(
                tokenizer.possible_future_terminals_iter(end_state)
                    .map(|terminal| terminal_offset + terminal),
            );
        }
    };

    for &global_start in starts {
        if global_start == 0 {
            for (component_index, component) in components.iter().enumerate() {
                if bytes.first().is_some_and(|byte| {
                    !reset_live_bytes[component_index].contains(*byte)
                }) {
                    continue;
                }
                scan_local(component_index, component.composition_tokenizer().start_state());
            }
            continue;
        }
        let component_index = tokenizer_state_offsets
            .partition_point(|&offset| offset <= global_start)
            .saturating_sub(1);
        let Some(component) = components.get(component_index) else {
            continue;
        };
        let offset = tokenizer_state_offsets[component_index];
        let local_start = global_start - offset;
        if local_start < component.composition_tokenizer().num_states() {
            scan_local(component_index, local_start);
        }
    }
    result.matches.sort_unstable();
    result.matches.dedup();
    result.future_terminals.sort_unstable();
    result.future_terminals.dedup();
    result
}

pub(super) fn scan_component_residual_start_groups(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    reset_live_bytes: &[U8Set],
    bytes: &[u8],
    candidate_groups: &[(u32, Vec<u32>)],
) -> FxHashMap<ResidualScanResult, Vec<u32>> {
    let validate =
        std::env::var_os("GLRMASK_VALIDATE_COMPOSE_TSID_REPRESENTATIVE_SCAN").is_some();
    let mut starts_by_scan = FxHashMap::<ResidualScanResult, Vec<u32>>::default();
    let mut by_component = vec![Vec::<(u32, &[u32])>::new(); components.len()];

    for (representative, support_states) in candidate_groups {
        if *representative == 0 {
            let scan = scan_component_residual_starts(
                components,
                tokenizer_state_offsets,
                terminal_offsets,
                reset_live_bytes,
                bytes,
                &[*representative],
            );
            starts_by_scan
                .entry(scan)
                .or_default()
                .extend(support_states);
            continue;
        }
        let component_index = tokenizer_state_offsets
            .partition_point(|&offset| offset <= *representative)
            .saturating_sub(1);
        let Some(component) = components.get(component_index) else {
            continue;
        };
        let offset = tokenizer_state_offsets[component_index];
        let local_start = *representative - offset;
        if local_start < component.composition_tokenizer().num_states() {
            by_component[component_index].push((local_start, support_states));
        }
    }

    for (component_index, starts) in by_component.into_iter().enumerate() {
        if starts.is_empty() {
            continue;
        }
        let component = components[component_index];
        let tokenizer = component.composition_tokenizer();
        let state_offset = tokenizer_state_offsets[component_index];
        let terminal_offset = terminal_offsets[component_index];
        let local_starts = starts.iter().map(|(start, _)| *start).collect::<Vec<_>>();
        let support_by_start = starts.into_iter().collect::<FxHashMap<_, _>>();

        for (end_states, matches, grouped_starts) in tokenizer.execute_summary_groups_from_states(bytes, &local_starts)
        {
            let mut scan = ResidualScanResult::default();
            scan.matches.extend(
                matches
                    .into_iter()
                    .filter(|(_, width)| *width > 0)
                    .map(|(terminal, width)| (terminal_offset + terminal, width)),
            );
            for end_state in end_states {
                scan.future_terminals.extend(
                    tokenizer.possible_future_terminals_iter(end_state)
                        .map(|terminal| terminal_offset + terminal),
                );
            }
            scan.matches.sort_unstable();
            scan.matches.dedup();
            scan.future_terminals.sort_unstable();
            scan.future_terminals.dedup();

            let output_support = starts_by_scan.entry(scan.clone()).or_default();
            for local_start in grouped_starts {
                let Some(support) = support_by_start.get(&local_start) else {
                    continue;
                };
                if validate {
                    for &global_state in *support {
                        let reference = scan_component_residual_starts(
                            components,
                            tokenizer_state_offsets,
                            terminal_offsets,
                            reset_live_bytes,
                            bytes,
                            &[global_state],
                        );
                        assert_eq!(
                            scan, reference,
                            "batched component residual scan differs for state {global_state} (component {component_index}, local start {local_start}, offset {state_offset})",
                        );
                    }
                }
                output_support.extend_from_slice(support);
            }
        }
    }

    for states in starts_by_scan.values_mut() {
        states.sort_unstable();
        states.dedup();
    }
    starts_by_scan
}


pub(super) fn visible_boundary_interface_pairs(
    analyzed: &AnalyzedGrammar,
    boundary_nonterminals: &BTreeSet<u32>,
    control_terminals: &BTreeSet<u32>,
) -> BTreeSet<(u32, u32)> {
    let set_len = analyzed.num_terminals as usize + 1;
    let mut last = vec![BitSet::new(set_len); analyzed.num_nonterminals as usize];
    loop {
        let mut changed = false;
        for rule in &analyzed.rules {
            let lhs = rule.lhs as usize;
            let mut additions = BitSet::new(set_len);
            for symbol in rule.rhs.iter().rev() {
                match symbol {
                    Symbol::Terminal(terminal) => {
                        if *terminal < analyzed.num_terminals {
                            additions.set(*terminal as usize);
                        }
                        break;
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        if let Some(row) = last.get(*nonterminal as usize) {
                            additions.union_with(row);
                        }
                        if !analyzed.nullable.contains(nonterminal) {
                            break;
                        }
                    }
                }
            }
            let before = last[lhs].count_ones();
            last[lhs].union_with(&additions);
            changed |= last[lhs].count_ones() != before;
        }
        if !changed {
            break;
        }
    }

    // PRECEDE is FOLLOW on the reversed grammar: the visible terminals that
    // may occur immediately before a nonterminal, propagating through nullable
    // prefixes and callers.
    let mut precede = vec![BitSet::new(set_len); analyzed.num_nonterminals as usize];
    loop {
        let mut changed = false;
        for rule in &analyzed.rules {
            for (position, symbol) in rule.rhs.iter().enumerate() {
                let Symbol::Nonterminal(nonterminal) = symbol else {
                    continue;
                };
                let mut additions = BitSet::new(set_len);
                let mut prefix_nullable = true;
                for prefix in rule.rhs[..position].iter().rev() {
                    match prefix {
                        Symbol::Terminal(terminal) => {
                            if *terminal < analyzed.num_terminals {
                                additions.set(*terminal as usize);
                            }
                            prefix_nullable = false;
                            break;
                        }
                        Symbol::Nonterminal(previous) => {
                            if let Some(row) = last.get(*previous as usize) {
                                additions.union_with(row);
                            }
                            if !analyzed.nullable.contains(previous) {
                                prefix_nullable = false;
                                break;
                            }
                        }
                    }
                }
                if prefix_nullable {
                    if let Some(lhs_precede) = precede.get(rule.lhs as usize) {
                        additions.union_with(lhs_precede);
                    }
                }
                let target = &mut precede[*nonterminal as usize];
                let before = target.count_ones();
                target.union_with(&additions);
                changed |= target.count_ones() != before;
            }
        }
        if !changed {
            break;
        }
    }

    let lexical = |terminal: usize| {
        terminal < analyzed.num_terminals as usize
            && !control_terminals.contains(&(terminal as u32))
    };
    let mut pairs = BTreeSet::new();
    for &nonterminal in boundary_nonterminals {
        let Some(before) = precede.get(nonterminal as usize) else {
            continue;
        };
        let Some(first) = analyzed.first.get(nonterminal as usize) else {
            continue;
        };
        for left in before.iter().filter(|&terminal| lexical(terminal)) {
            for right in first.iter().filter(|&terminal| lexical(terminal)) {
                pairs.insert((left as u32, right as u32));
            }
        }
        let Some(last_row) = last.get(nonterminal as usize) else {
            continue;
        };
        let Some(after) = analyzed.follow.get(nonterminal as usize) else {
            continue;
        };
        for left in last_row.iter().filter(|&terminal| lexical(terminal)) {
            for right in after.iter().filter(|&terminal| lexical(terminal)) {
                pairs.insert((left as u32, right as u32));
            }
        }
        if analyzed.nullable.contains(&nonterminal) {
            for left in before.iter().filter(|&terminal| lexical(terminal)) {
                for right in after.iter().filter(|&terminal| lexical(terminal)) {
                    pairs.insert((left as u32, right as u32));
                }
            }
        }
    }
    pairs
}

pub(super) fn component_grammar_adjacency_summary(
    component: &Constraint,
) -> Result<CompositionGrammarSummary, String> {
    if let Some(summary) = component.composition_grammar_summary.as_ref() {
        return Ok(summary.clone());
    }
    let augmented = component
        .table
        .rules
        .first()
        .ok_or_else(|| "component grammar has no augmented-start rule".to_string())?;
    let root = match augmented.rhs.as_slice() {
        [Symbol::Nonterminal(root)] => *root,
        rhs => {
            return Err(format!(
                "component augmented-start rule must contain one nonterminal, found {rhs:?}"
            ));
        }
    };
    let analyzed = AnalyzedGrammar::from_composed_rules(
        component.table.rules.clone(),
        component.table.num_terminals,
        component.terminal_display_names.clone(),
        component.table.nonterminal_display_names.clone(),
        augmented.lhs,
    );
    let num_terminals = analyzed.num_terminals as usize;
    let follows = compute_ever_allowed_follows(&analyzed)
        .into_iter()
        .map(|row| {
            let mut bits = BitSet::new(num_terminals);
            for terminal in row {
                if (terminal as usize) < num_terminals {
                    bits.set(terminal as usize);
                }
            }
            bits
        })
        .collect::<Vec<_>>();

    let mut last = vec![BitSet::new(num_terminals); analyzed.num_nonterminals as usize];
    loop {
        let mut changed = false;
        for rule in &analyzed.rules {
            let mut additions = BitSet::new(num_terminals);
            for symbol in rule.rhs.iter().rev() {
                match symbol {
                    Symbol::Terminal(terminal) => {
                        if (*terminal as usize) < num_terminals {
                            additions.set(*terminal as usize);
                        }
                        break;
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        if let Some(row) = last.get(*nonterminal as usize) {
                            additions.union_with(row);
                        }
                        if !analyzed.nullable.contains(nonterminal) {
                            break;
                        }
                    }
                }
            }
            let Some(target) = last.get_mut(rule.lhs as usize) else {
                continue;
            };
            let before = target.count_ones();
            target.union_with(&additions);
            changed |= before != target.count_ones();
        }
        if !changed {
            break;
        }
    }

    Ok(CompositionGrammarSummary {
        allowed_follows: follows,
        root_first: analyzed
            .first
            .get(root as usize)
            .cloned()
            .unwrap_or_else(|| BitSet::new(num_terminals)),
        root_last: last
            .get(root as usize)
            .cloned()
            .unwrap_or_else(|| BitSet::new(num_terminals)),
        root_nullable: analyzed.nullable.contains(&root),
    })
}

pub(super) fn compose_nonnullable_grammar_adjacency_summaries(
    components: &[&Constraint],
    terminal_offsets: &[u32],
    placeholder_terminals: &[u32],
    placeholder_component_indices: &[usize],
) -> Result<(CompositionGrammarSummary, BTreeSet<(u32, u32)>), String> {
    if components.len() != terminal_offsets.len()
        || placeholder_terminals.len() != placeholder_component_indices.len()
        || placeholder_component_indices
            .iter()
            .any(|&component| component == 0 || component >= components.len())
    {
        return Err("component grammar-summary shape mismatch".into());
    }
    let summaries = components
        .iter()
        .map(|component| component_grammar_adjacency_summary(component))
        .collect::<Result<Vec<_>, _>>()?;
    if summaries.iter().skip(1).any(|summary| summary.root_nullable) {
        return Err("nullable child requires the general composed-grammar analysis".into());
    }
    let total_terminals = components
        .iter()
        .zip(terminal_offsets)
        .map(|(component, offset)| offset + component.table.num_terminals)
        .max()
        .unwrap_or(0) as usize;
    let mut follows = vec![BitSet::new(total_terminals); total_terminals];
    let parent = &summaries[0];
    let placeholder_to_component = placeholder_terminals
        .iter()
        .copied()
        .zip(placeholder_component_indices.iter().copied())
        .collect::<BTreeMap<_, _>>();

    let expand_first = |terminal: u32| -> Vec<u32> {
        if let Some(&component_index) = placeholder_to_component.get(&terminal) {
            let offset = terminal_offsets[component_index];
            summaries[component_index]
                .root_first
                .iter()
                .map(|local| offset + local as u32)
                .collect()
        } else {
            vec![terminal]
        }
    };
    let expand_last = |terminal: u32| -> Vec<u32> {
        if let Some(&component_index) = placeholder_to_component.get(&terminal) {
            let offset = terminal_offsets[component_index];
            summaries[component_index]
                .root_last
                .iter()
                .map(|local| offset + local as u32)
                .collect()
        } else {
            vec![terminal]
        }
    };

    // Substitute every nonnullable placeholder into the parent adjacency
    // relation. This also handles adjacent placeholders: an edge x->y becomes
    // LAST(child_x) x FIRST(child_y).
    for (left, row) in parent.allowed_follows.iter().enumerate() {
        for right in row.iter() {
            let lefts = expand_last(left as u32);
            let rights = expand_first(right as u32);
            for expanded_left in lefts {
                for &expanded_right in &rights {
                    follows[expanded_left as usize].set(expanded_right as usize);
                }
            }
        }
    }
    // Child-internal adjacency is already exact for a precomposed child and
    // transports by a simple terminal offset.
    for component_index in 1..components.len() {
        let offset = terminal_offsets[component_index];
        for (left, row) in summaries[component_index]
            .allowed_follows
            .iter()
            .enumerate()
        {
            let global_left = offset + left as u32;
            for right in row.iter() {
                follows[global_left as usize].set((offset + right as u32) as usize);
            }
        }
    }

    let mut boundary_pairs = BTreeSet::new();
    for &placeholder in placeholder_terminals {
        let child_first = expand_first(placeholder);
        let child_last = expand_last(placeholder);
        for (left, row) in parent.allowed_follows.iter().enumerate() {
            if row.contains(placeholder as usize) {
                for expanded_left in expand_last(left as u32) {
                    for &right in &child_first {
                        boundary_pairs.insert((expanded_left, right));
                    }
                }
            }
        }
        if let Some(row) = parent.allowed_follows.get(placeholder as usize) {
            for right in row.iter() {
                for &left in &child_last {
                    for expanded_right in expand_first(right as u32) {
                        boundary_pairs.insert((left, expanded_right));
                    }
                }
            }
        }
    }
    let mut root_first = BitSet::new(total_terminals);
    for terminal in parent.root_first.iter() {
        for expanded in expand_first(terminal as u32) {
            root_first.set(expanded as usize);
        }
    }
    let mut root_last = BitSet::new(total_terminals);
    for terminal in parent.root_last.iter() {
        for expanded in expand_last(terminal as u32) {
            root_last.set(expanded as usize);
        }
    }
    Ok((
        CompositionGrammarSummary {
            allowed_follows: follows,
            root_first,
            root_last,
            root_nullable: parent.root_nullable,
        },
        boundary_pairs,
    ))
}

pub(super) fn disallowed_follows_from_allowed_rows(
    allowed_rows: &[BitSet],
    num_terminals: usize,
) -> BTreeMap<u32, BitSet> {
    let mut result = BTreeMap::new();
    for terminal in 0..num_terminals {
        let allowed = allowed_rows
            .get(terminal)
            .cloned()
            .unwrap_or_else(|| BitSet::new(num_terminals));
        let disallowed = allowed.complement();
        if !disallowed.is_zero() {
            result.insert(terminal as u32, disallowed);
        }
    }
    result
}


/// Extend grammar-derived boundary adjacency through parser-visible terminals
/// whose LR action preserves stack depth. These terminals need not occur in a
/// grammar RHS (scoped trivia is inserted directly into LR rows), so FIRST /
/// FOLLOW alone cannot see them.
///
/// For an actual interface pair `a -> b`, if a stack-neutral terminal `n` can
/// be consumed at a parser top from which `b` is admissible (or can replace the
/// top with a state from which `b` is admissible), then the concrete lexical
/// boundary may be `a -> n -> b`. Record exactly those two adjacencies.
/// `n` is deliberately *not* promoted to a global boundary seed: its relevance
/// is contextual to this interface, and arbitrary residual starts inside `n`
/// are recovered from the left-hand side of the resulting `(n, b)` pair.
pub(super) fn extend_boundary_interfaces_through_stack_neutral_lr_actions(
    table: &crate::compiler::glr::table::GLRTable,
    base_pairs: &BTreeSet<(u32, u32)>,
) -> BTreeSet<(u32, u32)> {
    let mut pairs = base_pairs.clone();
    if table.skip_terminals.is_empty() || base_pairs.is_empty() {
        return pairs;
    }

    let target_is_admissible = |state: u32, terminal: u32| {
        table
            .advance
            .get(state as usize)
            .is_some_and(|row| row.contains(terminal as usize))
            || table
                .action
                .get(state as usize)
                .and_then(|row| row.get(&terminal))
                .is_some()
    };

    for &(left, right) in base_pairs {
        for &neutral in &table.skip_terminals {
            let mut bridges = false;
            for (state, row) in table.action.iter().enumerate() {
                let state = state as u32;
                let Some(action) = row.get(&neutral) else {
                    continue;
                };
                match action {
                    Action::Skip => {
                        bridges |= target_is_admissible(state, right);
                    }
                    Action::Shift(target, true) => {
                        bridges |= target_is_admissible(*target, right);
                    }
                    Action::ReplaceShifts(targets) => {
                        bridges |= targets
                            .iter()
                            .copied()
                            .any(|target| target_is_admissible(target, right));
                    }
                    Action::StackShifts(shifts)
                        if shifts
                            .iter()
                            .all(|shift| shift.pop == 1 && shift.pushes.len() == 1) =>
                    {
                        bridges |= shifts.iter().any(|shift| {
                            target_is_admissible(shift.pushes[0], right)
                        });
                    }
                    Action::Split {
                        shift: Some((target, true)),
                        reduces,
                        accept: false,
                    } if reduces.is_empty() => {
                        bridges |= target_is_admissible(*target, right);
                    }
                    _ => {}
                }
                if bridges {
                    break;
                }
            }
            if bridges {
                pairs.insert((left, neutral));
                pairs.insert((neutral, right));
            }
        }
    }
    pairs
}

pub(super) fn transition_boundary_key(
    key: BoundaryTokenNodeKey,
    terminal: u32,
    next_offset: usize,
    seed_terminals: &[bool],
    globally_erased_terminals: &BitSet,
    interface_pairs: &BTreeSet<(u32, u32)>,
    disallowed_follows: Option<&BTreeMap<u32, BitSet>>,
    follow_transparent_terminals: &BitSet,
) -> Option<BoundaryTokenNodeKey> {
    if globally_erased_terminals.contains(terminal as usize) {
        return Some(BoundaryTokenNodeKey {
            offset: next_offset,
            started: true,
            ..key
        });
    }

    if key.last_terminal != u32::MAX
        && !follow_transparent_terminals.contains(key.last_terminal as usize)
        && !follow_transparent_terminals.contains(terminal as usize)
        && disallowed_follows
            .and_then(|rows| rows.get(&key.last_terminal))
            .is_some_and(|blocked| blocked.contains(terminal as usize))
    {
        return None;
    }
    let interface_witnessed = key.last_terminal != u32::MAX
        && interface_pairs.contains(&(key.last_terminal, terminal));
    Some(BoundaryTokenNodeKey {
        offset: next_offset,
        last_terminal: terminal,
        seeded: key.seeded
            || (key.last_terminal == u32::MAX
                && seed_terminals
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(false)),
        interface_witnessed: key.interface_witnessed || interface_witnessed,
        erased_seen: key.erased_seen || globally_erased_terminals.contains(terminal as usize),
        started: true,
    })
}

pub(super) fn boundary_token_graph_has_accepting_path(
    bytes: &[u8],
    arbitrary_scan: &ResidualScanResult,
    reset_scans: &[&ResidualScanResult],
    seed_terminals: &[bool],
    globally_erased_terminals: &BitSet,
    interface_pairs: &BTreeSet<(u32, u32)>,
    initial_interface_witnessed: bool,
    allow_seed_only: bool,
    disallowed_follows: Option<&BTreeMap<u32, BitSet>>,
    follow_transparent_terminals: &BitSet,
) -> bool {
    let accept_complete_cross_candidates =
        std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_COMPLETE_PATH_DISCOVERY").is_some();
    let start = BoundaryTokenNodeKey {
        offset: 0,
        last_terminal: u32::MAX,
        seeded: false,
        interface_witnessed: initial_interface_witnessed,
        erased_seen: false,
        started: false,
    };
    let mut seen = FxHashSet::<BoundaryTokenNodeKey>::default();
    let mut queue = VecDeque::<BoundaryTokenNodeKey>::new();
    seen.insert(start);
    queue.push_back(start);

    while let Some(key) = queue.pop_front() {
        if key.offset == bytes.len() {
            continue;
        }
        let scan = if key.started {
            reset_scans
                .get(key.offset - 1)
                .copied()
                .expect("reset scan must exist for every positive token offset")
        } else {
            arbitrary_scan
        };
        let accepting = |target: BoundaryTokenNodeKey| {
            target.offset == bytes.len()
                && (accept_complete_cross_candidates
                    || target.interface_witnessed
                    || (allow_seed_only && target.seeded))
        };

        for &(terminal, width) in &scan.matches {
            let next_offset = key.offset.saturating_add(width);
            if next_offset > bytes.len() {
                continue;
            }
            let Some(target) = transition_boundary_key(
                key,
                terminal,
                next_offset,
                seed_terminals,
                globally_erased_terminals,
                interface_pairs,
                disallowed_follows,
                follow_transparent_terminals,
            ) else {
                continue;
            };
            if accepting(target) {
                return true;
            }
            if seen.insert(target) {
                queue.push_back(target);
            }
        }
        for &terminal in &scan.future_terminals {
            let Some(target) = transition_boundary_key(
                key,
                terminal,
                bytes.len(),
                seed_terminals,
                globally_erased_terminals,
                interface_pairs,
                disallowed_follows,
                follow_transparent_terminals,
            ) else {
                continue;
            };
            if accepting(target) {
                return true;
            }
        }
    }
    false
}

pub(super) fn build_boundary_token_graph(
    bytes: &[u8],
    arbitrary_scan: &ResidualScanResult,
    reset_scans: &[&ResidualScanResult],
    seed_terminals: &[bool],
    globally_erased_terminals: &BitSet,
    interface_pairs: &BTreeSet<(u32, u32)>,
    initial_interface_witnessed: bool,
    allow_seed_only: bool,
    disallowed_follows: Option<&BTreeMap<u32, BitSet>>,
    follow_transparent_terminals: &BitSet,
) -> Option<(Vec<BoundaryTokenNode>, Vec<bool>, Vec<bool>)> {
    let accept_complete_cross_candidates =
        std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_COMPLETE_PATH_DISCOVERY").is_some();
    let mut nodes = Vec::<BoundaryTokenNode>::new();
    let mut node_ids = FxHashMap::<BoundaryTokenNodeKey, usize>::default();
    let mut queue = std::collections::VecDeque::<usize>::new();
    let start_key = BoundaryTokenNodeKey {
        offset: 0,
        last_terminal: u32::MAX,
        seeded: false,
        interface_witnessed: initial_interface_witnessed,
        erased_seen: false,
        started: false,
    };
    nodes.push(BoundaryTokenNode {
        key: start_key,
        outgoing: Vec::new(),
    });
    node_ids.insert(start_key, 0);
    queue.push_back(0);
    let mut accepting = vec![false];

    while let Some(node_id) = queue.pop_front() {
        let key = nodes[node_id].key;
        if key.offset == bytes.len() {
            continue;
        }
        let scan = if key.started {
            reset_scans
                .get(key.offset - 1)
                .copied()
                .expect("reset scan must exist for every positive token offset")
        } else {
            arbitrary_scan
        };

        for &(terminal, width) in &scan.matches {
            let next_offset = key.offset.saturating_add(width);
            if next_offset > bytes.len() {
                continue;
            }
            let Some(target_key) = transition_boundary_key(
                key,
                terminal,
                next_offset,
                seed_terminals,
                globally_erased_terminals,
                interface_pairs,
                disallowed_follows,
                follow_transparent_terminals,
            ) else {
                continue;
            };
            let target = if let Some(&target) = node_ids.get(&target_key) {
                target
            } else {
                let target = nodes.len();
                let is_accepting = target_key.offset == bytes.len()
                    && (accept_complete_cross_candidates
                        || target_key.interface_witnessed
                        || (allow_seed_only && target_key.seeded));
                nodes.push(BoundaryTokenNode {
                    key: target_key,
                    outgoing: Vec::new(),
                });
                node_ids.insert(target_key, target);
                queue.push_back(target);
                accepting.push(is_accepting);
                target
            };
            nodes[node_id]
                .outgoing
                .push(BoundaryTokenEdge { target, terminal });
        }

        // An unfinished final terminal is a real terminal-DWA label and may
        // itself be the boundary-begin seed at token end.
        for &terminal in &scan.future_terminals {
            let Some(target_key) = transition_boundary_key(
                key,
                terminal,
                bytes.len(),
                seed_terminals,
                globally_erased_terminals,
                interface_pairs,
                disallowed_follows,
                follow_transparent_terminals,
            ) else {
                continue;
            };
            let target = if let Some(&target) = node_ids.get(&target_key) {
                target
            } else {
                let target = nodes.len();
                let is_accepting = target_key.offset == bytes.len()
                    && (accept_complete_cross_candidates
                        || target_key.interface_witnessed
                        || (allow_seed_only && target_key.seeded));
                nodes.push(BoundaryTokenNode {
                    key: target_key,
                    outgoing: Vec::new(),
                });
                node_ids.insert(target_key, target);
                queue.push_back(target);
                accepting.push(is_accepting);
                target
            };
            nodes[node_id]
                .outgoing
                .push(BoundaryTokenEdge { target, terminal });
        }
    }

    if !accepting.iter().any(|&is_accepting| is_accepting) {
        return None;
    }
    let mut good = accepting.clone();
    let mut by_descending_offset = (0..nodes.len()).collect::<Vec<_>>();
    by_descending_offset.sort_unstable_by_key(|&node| std::cmp::Reverse(nodes[node].key.offset));
    for source in by_descending_offset {
        if !good[source]
            && nodes[source]
                .outgoing
                .iter()
                .any(|edge| good[edge.target])
        {
            good[source] = true;
        }
    }
    good[0].then_some((nodes, good, accepting))
}

pub(super) fn boundary_candidate_state_ranges_by_token(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    vocab: &Vocab,
    candidate_tokens: &BTreeSet<u32>,
) -> BTreeMap<u32, Vec<(usize, u32, u32)>> {
    debug_assert_eq!(components.len(), tokenizer_state_offsets.len());
    let mut by_token = BTreeMap::<u32, Vec<(usize, u32, u32)>>::new();
    for (component_index, constraint) in components.iter().enumerate() {
        if !constraint.possible_matches_complete {
            // Dynamic components deliberately do not publish the static
            // possible-matches table. Boundary discovery only uses these
            // ranges as a coarse starting-state prefilter; the subsequent
            // tokenizer graph expansion is exact. Conservatively admitting
            // every private TSID therefore cannot add a boundary token to the
            // final discovery unless the exact expansion witnesses it, and it
            // avoids materializing the component as a static parser DWA.
            let num_tsids = if constraint.state_to_internal_tsid.is_empty() {
                constraint.tokenizer.num_states()
            } else {
                constraint
                    .state_to_internal_tsid
                    .iter()
                    .copied()
                    .filter(|&tsid| tsid != u32::MAX)
                    .max()
                    .map_or(0, |tsid| tsid + 1)
            };
            if num_tsids != 0 {
                for &original in candidate_tokens {
                    if vocab
                        .entries_map()
                        .get(&original)
                        .is_some_and(|bytes| bytes.len() >= 2)
                    {
                        by_token.entry(original).or_default().push((
                            component_index,
                            0,
                            num_tsids - 1,
                        ));
                    }
                }
            }
            continue;
        }

        // Restrict the token dimension *before* walking possible-match weights.
        // The historical implementation iterated every internal token carried
        // by every weight and only then tested membership in `candidate_tokens`.
        // For large cached components those token sets are orders of magnitude
        // wider than the boundary prefilter. `original_token_to_internal` is
        // already the exact inverse relation we need: group the surviving
        // original candidates by component-private internal token, then merge
        // those sorted internal IDs against the RangeSet ranges in each weight.
        // This changes only iteration order; the emitted
        // `(component,start_tsid,end_tsid)` relation is identical.
        let mut candidates_by_internal = BTreeMap::<u32, Vec<u32>>::new();
        for &original in candidate_tokens {
            if !vocab
                .entries_map()
                .get(&original)
                .is_some_and(|bytes| bytes.len() >= 2)
            {
                continue;
            }
            let internal = if constraint.has_original_token_map() {
                constraint
                    .original_token_internal_at(original)
                    .unwrap_or(u32::MAX)
            } else {
                original
            };
            if internal == u32::MAX {
                continue;
            }
            candidates_by_internal
                .entry(internal)
                .or_default()
                .push(original);
        }
        if candidates_by_internal.is_empty() {
            continue;
        }
        let candidate_internal_ids = candidates_by_internal.keys().copied().collect::<Vec<_>>();

        // `possible_matches` may be retained in the compact packed weight pool
        // after loading a saved constraint. Use the runtime view here rather
        // than the materialized map so cached components stay directly
        // composable without unpacking their full weight inventory. Preserve
        // the old diagnostic union mode by materializing only when explicitly
        // requested.
        let mut visit_weight = |weight: crate::runtime::RuntimeWeightRef<'_>| {
            weight.for_each_entry(|start_tsid, end_tsid, internal_tokens| {
                internal_tokens.for_each_range(|start, end| {
                    let mut index =
                        candidate_internal_ids.partition_point(|&token| token < start);
                    while let Some(&internal_token) = candidate_internal_ids.get(index) {
                        if internal_token > end {
                            break;
                        }
                        if let Some(originals) = candidates_by_internal.get(&internal_token) {
                            for &original in originals {
                                by_token.entry(original).or_default().push((
                                    component_index,
                                    start_tsid,
                                    end_tsid,
                                ));
                            }
                        }
                        index += 1;
                    }
                });
            });
        };
        if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_UNION_POSSIBLE_MATCHES").is_some() {
            let mut materialized = (*constraint).clone();
            if materialized.materialize_non_dwa_weights_for_compilation().is_ok() {
                let union = Weight::union_all(materialized.possible_matches.values());
                visit_weight((&union).into());
            }
        } else {
            for terminal in constraint.runtime_possible_match_terminals() {
                if let Some(weight) = constraint.runtime_possible_match_weight(terminal) {
                    visit_weight(weight);
                }
            }
        }
    }
    for ranges in by_token.values_mut() {
        ranges.sort_unstable();
        ranges.dedup();
    }
    by_token
}

pub(super) fn boundary_candidate_state_ranges_by_token_reference(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    vocab: &Vocab,
    candidate_tokens: &BTreeSet<u32>,
) -> BTreeMap<u32, Vec<(usize, u32, u32)>> {
    debug_assert_eq!(components.len(), tokenizer_state_offsets.len());
    let union_possible_matches =
        std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_UNION_POSSIBLE_MATCHES").is_some();
    let mut by_token = BTreeMap::<u32, Vec<(usize, u32, u32)>>::new();
    for (component_index, constraint) in components.iter().enumerate() {
        if !constraint.possible_matches_complete {
            let num_tsids = if constraint.state_to_internal_tsid.is_empty() {
                constraint.tokenizer.num_states()
            } else {
                constraint
                    .state_to_internal_tsid
                    .iter()
                    .copied()
                    .filter(|&tsid| tsid != u32::MAX)
                    .max()
                    .map_or(0, |tsid| tsid + 1)
            };
            if num_tsids != 0 {
                for &original in candidate_tokens {
                    if vocab
                        .entries_map()
                        .get(&original)
                        .is_some_and(|bytes| bytes.len() >= 2)
                    {
                        by_token.entry(original).or_default().push((
                            component_index,
                            0,
                            num_tsids - 1,
                        ));
                    }
                }
            }
            continue;
        }
        let union;
        let weights: Box<dyn Iterator<Item = &Weight> + '_> = if union_possible_matches {
            union = Weight::union_all(constraint.possible_matches.values());
            Box::new(std::iter::once(&union))
        } else {
            Box::new(constraint.possible_matches.values())
        };
        let token_groups = constraint.internal_token_groups();
        for weight in weights {
            for (start_tsid, end_tsid, internal_tokens) in weight.range_entries() {
                for internal_token in internal_tokens.iter() {
                    let Some(token_groups) = token_groups else {
                        if candidate_tokens.contains(&internal_token)
                            && vocab
                                .entries_map()
                                .get(&internal_token)
                                .is_some_and(|bytes| bytes.len() >= 2)
                        {
                            by_token.entry(internal_token).or_default().push((
                                component_index,
                                start_tsid,
                                end_tsid,
                            ));
                        }
                        continue;
                    };
                    let Some(originals) = token_groups.get(internal_token as usize)
                    else {
                        continue;
                    };
                    for &original in originals {
                        if candidate_tokens.contains(&original)
                            && vocab
                                .entries_map()
                                .get(&original)
                                .is_some_and(|bytes| bytes.len() >= 2)
                        {
                            by_token.entry(original).or_default().push((
                                component_index,
                                start_tsid,
                                end_tsid,
                            ));
                        }
                    }
                }
            }
        }
    }
    for ranges in by_token.values_mut() {
        ranges.sort_unstable();
        ranges.dedup();
    }
    by_token
}

pub(super) fn candidate_start_state_groups_for_token(
    token_id: u32,
    candidate_ranges: &BTreeMap<u32, Vec<(usize, u32, u32)>>,
    extra_start_states_by_token: &BTreeMap<u32, Vec<u32>>,
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
) -> Vec<(u32, Vec<u32>)> {
    // Global state zero is the retained/fresh merged reset dispatcher. It is a
    // semantic state of its own and must not be conflated with an individual
    // component's local start state.
    let mut support_by_representative = FxHashMap::<u32, Vec<u32>>::default();
    support_by_representative.insert(0, vec![0]);
    if let Some(extra_states) = extra_start_states_by_token.get(&token_id) {
        // These states are an exact support subset, not necessarily a whole
        // existing TSID class, so preserve them as singleton representatives.
        // The residual scanner will still merge equal lexical scans afterward.
        for &state in extra_states {
            if state != 0 {
                support_by_representative.entry(state).or_default().push(state);
            }
        }
    }
    if let Some(ranges) = candidate_ranges.get(&token_id) {
        for &(component_index, start_tsid, end_tsid) in ranges {
            let constraint = components[component_index];
            let state_offset = tokenizer_state_offsets[component_index];
            for tsid in start_tsid..=end_tsid {
                let Some(states) = constraint.internal_tsid_groups().get(tsid as usize) else {
                    continue;
                };
                let mut representative = None;
                for &state in states {
                    let Some(global) = state_offset.checked_add(state) else {
                        continue;
                    };
                    if global == 0 {
                        continue;
                    }
                    representative.get_or_insert(global);
                    support_by_representative
                        .entry(*representative.as_ref().unwrap())
                        .or_default()
                        .push(global);
                }
            }
        }
    }
    let mut groups = support_by_representative.into_iter().collect::<Vec<_>>();
    for (_, support) in &mut groups {
        support.sort_unstable();
        support.dedup();
    }
    groups.sort_unstable_by_key(|(representative, _)| *representative);
    groups
}

#[derive(Clone, Copy)]
pub(super) struct ExprByteSummary {
    pub(super) nullable: bool,
    pub(super) first: U8Set,
    pub(super) last: U8Set,
    pub(super) reachable: U8Set,
}

pub(super) fn expr_byte_summary(expr: &Expr) -> ExprByteSummary {
    match expr {
        Expr::U8Seq(bytes) => ExprByteSummary {
            nullable: bytes.is_empty(),
            first: bytes.first().copied().map_or(U8Set::empty(), U8Set::single),
            last: bytes.last().copied().map_or(U8Set::empty(), U8Set::single),
            reachable: U8Set::from_bytes(bytes),
        },
        Expr::U8Class(bytes) => ExprByteSummary {
            nullable: false,
            first: *bytes,
            last: *bytes,
            reachable: *bytes,
        },
        // Opaque precompiled DFAs are uncommon in imported JSON schemas.  Use
        // a full-byte overapproximation so this filter can only retain extra
        // tokens, never discard a real boundary witness.
        Expr::Dfa(_) => ExprByteSummary {
            nullable: expr.is_nullable(),
            first: U8Set::all(),
            last: U8Set::all(),
            reachable: U8Set::all(),
        },
        Expr::Intersect { expr, intersect } => {
            let left = expr_byte_summary(expr);
            let right = expr_byte_summary(intersect);
            ExprByteSummary {
                nullable: left.nullable && right.nullable,
                first: left.first.intersection(&right.first),
                last: left.last.intersection(&right.last),
                reachable: left.reachable.intersection(&right.reachable),
            }
        }
        Expr::Seq(parts) => {
            let summaries = parts.iter().map(expr_byte_summary).collect::<Vec<_>>();
            let mut first = U8Set::empty();
            for summary in &summaries {
                first |= summary.first;
                if !summary.nullable {
                    break;
                }
            }
            let mut last = U8Set::empty();
            for summary in summaries.iter().rev() {
                last |= summary.last;
                if !summary.nullable {
                    break;
                }
            }
            ExprByteSummary {
                nullable: summaries.iter().all(|summary| summary.nullable),
                first,
                last,
                reachable: summaries
                    .iter()
                    .fold(U8Set::empty(), |bytes, summary| bytes | summary.reachable),
            }
        }
        Expr::Choice(options) => options.iter().map(expr_byte_summary).fold(
            ExprByteSummary {
                nullable: false,
                first: U8Set::empty(),
                last: U8Set::empty(),
                reachable: U8Set::empty(),
            },
            |mut combined, summary| {
                combined.nullable |= summary.nullable;
                combined.first |= summary.first;
                combined.last |= summary.last;
                combined.reachable |= summary.reachable;
                combined
            },
        ),
        Expr::Exclude { expr, .. } => expr_byte_summary(expr),
        Expr::Repeat { expr, min, max } => {
            if *max == Some(0) {
                return ExprByteSummary {
                    nullable: true,
                    first: U8Set::empty(),
                    last: U8Set::empty(),
                    reachable: U8Set::empty(),
                };
            }
            let child = expr_byte_summary(expr);
            ExprByteSummary {
                nullable: *min == 0 || child.nullable,
                ..child
            }
        }
        Expr::Shared(expr) => expr_byte_summary(expr),
        Expr::Epsilon => ExprByteSummary {
            nullable: true,
            first: U8Set::empty(),
            last: U8Set::empty(),
            reachable: U8Set::empty(),
        },
    }
}

pub(super) fn boundary_token_prefilter(
    vocab: &Vocab,
    components: &[&Constraint],
    terminal_offsets: &[u32],
    seed_terminals: &[bool],
    allow_suffix_seed: bool,
) -> BTreeSet<u32> {
    let num_terminals = terminal_offsets
        .iter()
        .copied()
        .zip(components)
        .map(|(offset, component)| offset + component.tokenizer.num_terminals())
        .max()
        .unwrap_or(0) as usize;
    let mut summaries = vec![None::<ExprByteSummary>; num_terminals];
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index] as usize;
        let token_groups = component.internal_token_groups();
        for local_terminal in 0..component.tokenizer.num_terminals() as usize {
            summaries[terminal_offset + local_terminal] = Some(
                component
                    .retained_terminal_expr(local_terminal as u32)
                    .map(expr_byte_summary)
                    .unwrap_or(ExprByteSummary {
                        nullable: false,
                        first: U8Set::all(),
                        last: U8Set::all(),
                        reachable: U8Set::all(),
                    }),
            );
        }
    }

    // Seed-only boundary evidence begins at model-token offset zero. When a
    // globally erased terminal exists, keep the older conservative suffix scan
    // because erased trivia may precede the first parser-visible seed inside the
    // same token. Otherwise later seed occurrences require an actual interface
    // witness and are supplied by the adjacent-pair candidate set.
    let mut endpoint_dfas = Vec::new();
    let mut endpoint_dfas_by_first = (0..256)
        .map(|_| Vec::<usize>::new())
        .collect::<Vec<_>>();
    let mut conservative_first = U8Set::empty();
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index] as usize;
        let token_groups = component.internal_token_groups();
        for local_terminal in 0..component.tokenizer.num_terminals() as usize {
            let global_terminal = terminal_offset + local_terminal;
            if !seed_terminals
                .get(global_terminal)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            let Some(expr) = component.retained_terminal_expr(local_terminal as u32) else {
                // Missing retained source is rare and cannot justify an unsafe
                // exclusion. Fall back to the conservative first-byte summary.
                conservative_first |= summaries[global_terminal]
                    .map(|summary| summary.first)
                    .unwrap_or(U8Set::all());
                continue;
            };
            let dfa_index = endpoint_dfas.len();
            let dfa = compile_terminal_expr_dfa(expr);
            for byte in 0u8..=u8::MAX {
                if dfa.step(0, byte).is_some() {
                    endpoint_dfas_by_first[byte as usize].push(dfa_index);
                }
            }
            endpoint_dfas.push(dfa);
        }
    }
    let mut candidates = BTreeSet::new();
    for (&token, bytes) in vocab.entries_map().iter() {
        if bytes.len() < 2 {
            continue;
        }
        let offset_count = if allow_suffix_seed { bytes.len() } else { 1 };
        'suffixes: for offset in 0..offset_count {
            let first = bytes[offset];
            if conservative_first.contains(first) {
                candidates.insert(token);
                break 'suffixes;
            }
            for &dfa_index in &endpoint_dfas_by_first[first as usize] {
                let dfa = &endpoint_dfas[dfa_index];
                let mut state = 0u32;
                let mut reached_token_end = true;
                for &byte in &bytes[offset..] {
                    let Some(next) = dfa.step(state, byte) else {
                        reached_token_end = false;
                        break;
                    };
                    state = next;
                    if dfa.finalizers(state).contains(0) {
                        candidates.insert(token);
                        break 'suffixes;
                    }
                }
                if reached_token_end && dfa.possible_future_group_ids(state).contains(0) {
                    candidates.insert(token);
                    break 'suffixes;
                }
            }
        }
    }
    let seed_dfa_candidates = candidates.len();

    // Legacy/global-erased-trivia fallback. `possible_matches` deliberately
    // includes terminals reached later inside a model token, so it is too broad
    // for the ordinary reset-origin seed rule. Retain it only when erased trivia
    // may legally precede the parser-visible seed.
    if allow_suffix_seed {
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index] as usize;
        let token_groups = component.internal_token_groups();
        for local_terminal in 0..component.tokenizer.num_terminals() as usize {
            if !seed_terminals
                .get(terminal_offset + local_terminal)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            let Some(weight) = component.possible_matches.get(&(local_terminal as u32)) else {
                continue;
            };
            for (_, _, internal_tokens) in weight.range_entries() {
                for internal_token in internal_tokens.iter() {
                    if let Some(token_groups) = token_groups {
                        if let Some(originals) = token_groups.get(internal_token as usize) {
                            candidates.extend(originals.iter().copied().filter(|token| {
                                vocab
                                    .entries_map()
                                    .get(token)
                                    .is_some_and(|bytes| bytes.len() >= 2)
                            }));
                        }
                    } else {
                        if vocab
                            .entries_map()
                            .get(&internal_token)
                            .is_some_and(|bytes| bytes.len() >= 2)
                        {
                            candidates.insert(internal_token);
                        }
                    }
                }
            }
        }
    }
    }
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_prefilter_sources] allow_suffix_seed={} after_seed_dfas={} after_seed_possible_matches={} seed_terminals={:?}",
            allow_suffix_seed,
            seed_dfa_candidates,
            candidates.len(),
            seed_terminals
                .iter()
                .enumerate()
                .filter_map(|(terminal, &seed)| seed.then_some(terminal))
                .collect::<Vec<_>>(),
        );
    }
    candidates
}


/// Extra arbitrary lexer starts needed by boundary discovery.
///
/// Static `possible_matches` is intentionally sparse: it indexes delayed-terminal
/// queries used by runtime masking, not every raw lexer residual from which a
/// model token may begin. Boundary composition has a stronger requirement: a
/// token can start in the middle of any parser-visible terminal that participates
/// in a boundary witness. Preserve those residual starts directly from the
/// component `terminal_live_states` inverse index.
///
/// This is deliberately terminal-generic. Scoped IGNORE participates only
/// because it is a visible terminal in `relevant_terminals`; globally erased
/// IGNORE is excluded by the caller and remains lexical epsilon.
pub(super) fn boundary_visible_residual_starts_by_first_byte(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    relevant_terminals: &BitSet,
) -> Vec<Vec<u32>> {
    debug_assert_eq!(components.len(), tokenizer_state_offsets.len());
    debug_assert_eq!(components.len(), terminal_offsets.len());

    // Necessary first-byte filter: if the epsilon closure of a residual raw
    // state has no transition on the model token's first byte, that token
    // cannot begin from that state. The inverse is useful both for selecting
    // candidate vocabulary tokens and for recovering their exact raw starts.
    let mut starts_by_first_byte = (0..256).map(|_| Vec::<u32>::new()).collect::<Vec<_>>();
    for (component_index, component) in components.iter().enumerate() {
        let tokenizer = component.composition_tokenizer();
        let terminal_offset = terminal_offsets[component_index];
        let state_offset = tokenizer_state_offsets[component_index];
        let closures = tokenizer.all_singleton_epsilon_closures();
        let mut relevant_states = Vec::<u32>::new();
        for local_terminal in 0..tokenizer.num_terminals() {
            let global_terminal = terminal_offset + local_terminal;
            if !relevant_terminals.contains(global_terminal as usize) {
                continue;
            }
            if std::ptr::eq(tokenizer, component.tokenizer.as_ref()) {
                let Some(live_states) = component.terminal_live_states.get(local_terminal as usize) else {
                    continue;
                };
                relevant_states.extend(
                    live_states
                        .iter()
                        .copied()
                        .filter(|&state| state != tokenizer.start_state()),
                );
            } else {
                // Virtual-Static constraints commit in an exact symbolic tokenizer,
                // while composition observes their finite Static tokenizer. The
                // persisted terminal_live_states sidecar belongs to the exact
                // runtime tokenizer, so derive the equivalent live-state inverse
                // directly in the finite observation coordinate instead.
                for local_state in 0..tokenizer.num_states() {
                    if local_state == tokenizer.start_state() {
                        continue;
                    }
                    let live = closures[local_state as usize].iter().any(|&closure_state| {
                        tokenizer
                            .matched_terminals_iter(closure_state)
                            .chain(tokenizer.possible_future_terminals_iter(closure_state))
                            .any(|terminal| terminal == local_terminal)
                    });
                    if live {
                        relevant_states.push(local_state);
                    }
                }
            }
        }
        relevant_states.sort_unstable();
        relevant_states.dedup();
        for local_state in relevant_states {
            let Some(global_state) = state_offset.checked_add(local_state) else {
                continue;
            };
            let Some(closure) = closures.get(local_state as usize) else {
                continue;
            };
            let mut bytes = U8Set::empty();
            for &closure_state in closure.iter() {
                for (byte, _) in tokenizer.transitions_from(closure_state) {
                    bytes.insert(byte);
                }
            }
            for byte in bytes.iter() {
                starts_by_first_byte[byte as usize].push(global_state);
            }
        }
    }
    for starts in &mut starts_by_first_byte {
        starts.sort_unstable();
        starts.dedup();
    }
    starts_by_first_byte
}

pub(super) fn boundary_visible_residual_starts_by_token(
    vocab: &Vocab,
    candidate_tokens: &BTreeSet<u32>,
    starts_by_first_byte: &[Vec<u32>],
) -> BTreeMap<u32, Vec<u32>> {
    candidate_tokens
        .iter()
        .filter_map(|&token| {
            let bytes = vocab.entries_map().get(&token)?;
            (bytes.len() >= 2)
                .then(|| starts_by_first_byte[bytes[0] as usize].clone())
                .filter(|starts| !starts.is_empty())
                .map(|starts| (token, starts))
        })
        .collect()
}


/// Necessary byte-level filter for a visible terminal interface realized inside
/// one model token. If terminal `a` is followed by terminal `b`, then at the
/// split where `a` finishes and `b` begins the token contains an adjacent byte
/// pair `(last(a), first(b))`. This says nothing about parser legality by itself;
/// it only supplies cheap candidate tokens for the exact graph below.
pub(super) fn boundary_interface_adjacent_pair_candidates(
    vocab: &Vocab,
    components: &[&Constraint],
    terminal_offsets: &[u32],
    interface_pairs: &BTreeSet<(u32, u32)>,
) -> Vec<u32> {
    let num_terminals = components
        .iter()
        .zip(terminal_offsets.iter().copied())
        .map(|(component, offset)| offset + component.tokenizer.num_terminals())
        .max()
        .unwrap_or(0) as usize;
    let mut summaries = vec![None::<ExprByteSummary>; num_terminals];
    let sparse_summaries = std::env::var_os(
        "GLRMASK_DISABLE_SPARSE_INTERFACE_BYTE_SUMMARIES",
    )
    .is_none();
    let needed_terminals = sparse_summaries.then(|| {
        interface_pairs
            .iter()
            .flat_map(|&(left, right)| [left, right])
            .collect::<BTreeSet<_>>()
    });
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index] as usize;
        for local_terminal in 0..component.tokenizer.num_terminals() as usize {
            let global_terminal = terminal_offset + local_terminal;
            if needed_terminals
                .as_ref()
                .is_some_and(|needed| !needed.contains(&(global_terminal as u32)))
            {
                continue;
            }
            summaries[global_terminal] = Some(
                component
                    .retained_terminal_expr(local_terminal as u32)
                    .map(expr_byte_summary)
                    .unwrap_or(ExprByteSummary {
                        nullable: false,
                        first: U8Set::all(),
                        last: U8Set::all(),
                        reachable: U8Set::all(),
                    }),
            );
        }
    }

    let mut allowed_pairs = [U8Set::empty(); 256];
    for &(left, right) in interface_pairs {
        let Some(left) = summaries.get(left as usize).and_then(|summary| *summary) else {
            continue;
        };
        let Some(right) = summaries.get(right as usize).and_then(|summary| *summary) else {
            continue;
        };
        for last in left.last.iter() {
            allowed_pairs[last as usize] |= right.first;
        }
    }
    vocab_tokens_with_adjacent_pairs(vocab, &allowed_pairs)
}



pub(super) fn boundary_terminal_residual_continuation_candidates(
    vocab: &Vocab,
    components: &[&Constraint],
    terminal_offsets: &[u32],
    terminals: &BitSet,
) -> Vec<u32> {
    let mut selected = BTreeSet::new();
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index];
        for local_terminal in 0..component.tokenizer.num_terminals() {
            let global_terminal = terminal_offset + local_terminal;
            if !terminals.contains(global_terminal as usize) {
                continue;
            }
            let Some(expr) = component.retained_terminal_expr(local_terminal) else {
                // Retained Expr metadata is expected for modern artifacts. If it
                // is absent, fall back conservatively to the exact component
                // possible-matches relation rather than dropping support.
                if let Some(weight) = component.possible_matches.get(&local_terminal) {
                    for (_, _, internal_tokens) in weight.range_entries() {
                        for internal_token in internal_tokens.iter() {
                            if component.internal_token_to_tokens.is_empty() {
                                if vocab
                                    .entries_map()
                                    .get(&internal_token)
                                    .is_some_and(|bytes| bytes.len() >= 2)
                                {
                                    selected.insert(internal_token);
                                }
                            } else if let Some(originals) =
                                component.internal_token_to_tokens.get(internal_token as usize)
                            {
                                selected.extend(originals.iter().copied().filter(|token| {
                                    vocab
                                        .entries_map()
                                        .get(token)
                                        .is_some_and(|bytes| bytes.len() >= 2)
                                }));
                            }
                        }
                    }
                }
                continue;
            };

            let dfa = compile_terminal_expr_dfa(expr);
            // States reachable after at least one byte are exactly the possible
            // within-terminal residual positions. Keep the start state too if a
            // nonempty cycle reaches it.
            let mut residual = vec![false; dfa.num_states()];
            let mut queue = VecDeque::new();
            for (_, target) in dfa.transitions(0) {
                if !residual[target as usize] {
                    residual[target as usize] = true;
                    queue.push_back(target);
                }
            }
            while let Some(state) = queue.pop_front() {
                for (_, target) in dfa.transitions(state) {
                    if !residual[target as usize] {
                        residual[target as usize] = true;
                        queue.push_back(target);
                    }
                }
            }
            let residual_states = residual
                .iter()
                .enumerate()
                .filter_map(|(state, &reachable)| reachable.then_some(state as u32))
                .collect::<Vec<_>>();
            if residual_states.is_empty() {
                continue;
            }

            for (&token, bytes) in vocab.entries_map() {
                if bytes.len() < 2 {
                    continue;
                }
                let mut current = residual_states.clone();
                for &byte in bytes {
                    let mut next = Vec::with_capacity(current.len());
                    for state in current {
                        if let Some(target) = dfa.step(state, byte) {
                            next.push(target);
                        }
                    }
                    if next.is_empty() {
                        current = next;
                        break;
                    }
                    next.sort_unstable();
                    next.dedup();
                    current = next;
                }
                if current.iter().copied().any(|state| {
                    dfa.finalizers(state).contains(0)
                        || dfa.possible_future_group_ids(state).contains(0)
                }) {
                    selected.insert(token);
                }
            }
        }
    }
    selected.into_iter().collect()
}


pub(super) fn boundary_context_residual_states(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    context_terminals: &BitSet,
) -> FxHashSet<u32> {
    let mut states = FxHashSet::default();
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index];
        let state_offset = tokenizer_state_offsets[component_index];
        for local_terminal in 0..component.tokenizer.num_terminals() {
            let global_terminal = terminal_offset + local_terminal;
            if !context_terminals.contains(global_terminal as usize) {
                continue;
            }
            let Some(live_states) = component.terminal_live_states.get(local_terminal as usize) else {
                continue;
            };
            for &local_state in live_states {
                if local_state == component.tokenizer.start_state() {
                    continue;
                }
                if let Some(global_state) = state_offset.checked_add(local_state) {
                    states.insert(global_state);
                }
            }
        }
    }
    states
}

pub(super) fn discover_boundary_token_paths(
    vocab: &Vocab,
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    seed_terminals: &[bool],
    ignore_terminals: &BitSet,
    interface_pairs: &BTreeSet<(u32, u32)>,
    context_terminals: &BitSet,
    follow_transparent_terminals: &BitSet,
    disallowed_follows: Option<&BTreeMap<u32, BitSet>>,
    prefer_cross_interface_only: bool,
    candidate_limit: Option<&BTreeSet<u32>>,
) -> BoundaryTokenDiscovery {
    let discovery_total_started_at = Instant::now();
    // Authoritative segmented B is the language whose model-token execution
    // actually crosses component ownership. LR skip closure can add both
    // halves of a concrete `left -> neutral -> right` bridge; one half may be
    // wholly inside the same component. Such a half-edge belongs to A, not B.
    // Filter it here as well as in the byte prefilter so the exact token graph
    // does not construct witnesses which CrossedFull discards later anyway.
    let cross_interface_pairs;
    let interface_pairs = if prefer_cross_interface_only {
        let owner = |terminal: u32| {
            terminal_offsets
                .partition_point(|&offset| offset <= terminal)
                .saturating_sub(1)
        };
        cross_interface_pairs = interface_pairs
            .iter()
            .copied()
            .filter(|&(left, right)| owner(left) != owner(right))
            .collect::<BTreeSet<_>>();
        &cross_interface_pairs
    } else {
        interface_pairs
    };
    let num_terminals = components
        .iter()
        .zip(terminal_offsets.iter().copied())
        .map(|(component, offset)| offset + component.tokenizer.num_terminals())
        .max()
        .unwrap_or(0) as usize;
    let reset_starts = composite_reset_states(components, tokenizer_state_offsets);
    let reset_state_set = reset_starts.iter().copied().collect::<FxHashSet<_>>();
    let reset_live_bytes = component_reset_live_bytes(components);
    let mut residual_start_terminals = BitSet::new(num_terminals);
    for (terminal, &seed) in seed_terminals.iter().enumerate() {
        if seed && !ignore_terminals.contains(terminal) {
            residual_start_terminals.set(terminal);
        }
    }
    for &(left, _) in interface_pairs {
        if !ignore_terminals.contains(left as usize) {
            residual_start_terminals.set(left as usize);
        }
    }
    // Only parser-visible stack-neutral terminals need an explicit carry bit
    // across model-token boundaries. Ordinary component terminals keep their
    // parser support through the transported component DWA; the neutral LR
    // terminals are newly state-dependent behavior in the composed table.
    for terminal in context_terminals.iter() {
        if !ignore_terminals.contains(terminal) {
            residual_start_terminals.set(terminal);
        }
    }
    // In the authoritative cross-component factor, the retained component A
    // whose LR state owns the composed stack top is also authoritative for its
    // scoped identity/ignore language.  Do not synthesize contextual B starts
    // merely because such a terminal was state-dependent in the composed
    // table: those are component-local paths, not boundary crossings.  Keep the
    // experiment flag as an opt-in for older/non-authoritative callers.
    let use_component_scoped_identity = prefer_cross_interface_only
        || std::env::var_os("GLRMASK_EXPERIMENT_COMPONENT_SCOPED_IGNORE_TOP_ACCEPT").is_some();
    let boundary_context_states = if use_component_scoped_identity {
        FxHashSet::default()
    } else {
        boundary_context_residual_states(
            components,
            tokenizer_state_offsets,
            terminal_offsets,
            context_terminals,
        )
    };
    let use_prefilter = std::env::var_os("GLRMASK_COMPOSE_DISABLE_BOUNDARY_PREFILTER").is_none();
    let all_multi_byte_entries = vocab
        .entries_map()
        .iter()
        .filter(|(_, bytes)| bytes.len() >= 2)
        .map(|(&token_id, bytes)| (token_id, bytes.as_slice()))
        .collect::<Vec<_>>();
    let residual_starts_by_first_byte = boundary_visible_residual_starts_by_first_byte(
        components,
        tokenizer_state_offsets,
        terminal_offsets,
        &residual_start_terminals,
    );
    // Cheap candidate selection precedes all suffix execution. In the ordinary
    // case a seed-only boundary starts at byte zero; later boundaries inside a
    // token are supplied by exact grammar/LR interface byte pairs. Only a
    // globally erased terminal requires the conservative old suffix-seed rule.
    let discovery_prelude_ms = discovery_total_started_at.elapsed().as_secs_f64() * 1000.0;
    let prefilter_started_at = Instant::now();
    let allow_suffix_seed = !ignore_terminals.is_empty();
    let cross_interface_only = use_prefilter
        && ignore_terminals.is_empty()
        && (prefer_cross_interface_only
            || std::env::var_os("GLRMASK_EXPERIMENT_CROSS_INTERFACE_PREFILTER_ONLY").is_some());
    // In the exact cross-interface factor, seed-only tokens are component-local
    // language and are deliberately excluded from boundary repair. Avoid the
    // full-vocabulary seed-DFA scan whose result would be discarded below.
    let mut prefilter = if cross_interface_only {
        BTreeSet::new()
    } else if use_prefilter {
        boundary_token_prefilter(
            vocab,
            components,
            terminal_offsets,
            seed_terminals,
            allow_suffix_seed,
        )
    } else {
        all_multi_byte_entries
            .iter()
            .map(|&(token_id, _)| token_id)
            .collect::<BTreeSet<_>>()
    };
    let interface_pairs_for_prefilter = interface_pairs.clone();
    let interface_pair_candidates = if use_prefilter {
        boundary_interface_adjacent_pair_candidates(
            vocab,
            components,
            terminal_offsets,
            &interface_pairs_for_prefilter,
        )
    } else {
        Vec::new()
    };
    if cross_interface_only {
        // Under the cross-component boundary factor, every surviving token
        // contains at least one adjacent pair of parser-visible terminals whose
        // owners differ. With no globally erased terminal between them, the
        // model-token bytes at that split necessarily contain
        // `(last(left), first(right))`. The byte-pair filter is therefore a
        // necessary condition, not a heuristic. Seed-only candidates are local
        // component language and are supplied by the retained component parser
        // segments rather than the cross-boundary repair.
        prefilter.extend(interface_pair_candidates.iter().copied());
    } else {
        prefilter.extend(interface_pair_candidates.iter().copied());
    }
    let context_residual_candidates = if use_prefilter
        && !cross_interface_only
        && !use_component_scoped_identity
    {
        boundary_terminal_residual_continuation_candidates(
            vocab,
            components,
            terminal_offsets,
            context_terminals,
        )
    } else {
        Vec::new()
    };
    prefilter.extend(context_residual_candidates.iter().copied());
    if let Some(limit) = candidate_limit {
        prefilter.retain(|token| limit.contains(token));
    }
    if let Some(path) = std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TOKEN_ALLOWLIST") {
        let allowed = std::fs::read_to_string(path)
            .expect("read experimental boundary token allowlist")
            .lines()
            .filter_map(|line| line.trim().parse::<u32>().ok())
            .collect::<BTreeSet<_>>();
        prefilter.retain(|token| allowed.contains(token));
        if compose_profile_enabled() {
            eprintln!("[glrmask/profile][constraint_boundary_oracle_allowlist] allowed={} surviving_prefilter={}", allowed.len(), prefilter.len());
        }
    }
    let extra_residual_starts = boundary_visible_residual_starts_by_token(
        vocab,
        &prefilter,
        &residual_starts_by_first_byte,
    );
    let prefilter_ms = prefilter_started_at.elapsed().as_secs_f64() * 1000.0;
    let multi_byte_entries = all_multi_byte_entries
        .iter()
        .copied()
        .filter(|(token_id, _)| prefilter.contains(token_id))
        .collect::<Vec<_>>();

    // Exact graph expansion needs reset scans only for suffixes of surviving
    // candidates. Building this cache before coarse selection previously scanned
    // ~271k unique Llama-3 suffixes even when only a few thousand tokens could
    // witness a boundary.
    let suffix_cache_started_at = Instant::now();
    let reset_suffix_cache = if std::env::var_os("GLRMASK_COMPOSE_DISABLE_SUFFIX_CACHE")
        .is_some()
    {
        None
    } else {
        let mut suffixes = FxHashSet::<&[u8]>::default();
        for &(_, bytes) in &multi_byte_entries {
            for offset in 1..bytes.len() {
                suffixes.insert(&bytes[offset..]);
            }
        }
        Some(
            suffixes
                .into_par_iter()
                .map(|suffix| {
                    let scan = scan_component_residual_starts(
                        components,
                        tokenizer_state_offsets,
                        terminal_offsets,
                        &reset_live_bytes,
                        suffix,
                        &reset_starts,
                    );
                    (suffix, scan)
                })
                .collect::<FxHashMap<_, _>>(),
        )
    };
    let suffix_cache_ms = suffix_cache_started_at.elapsed().as_secs_f64() * 1000.0;
    let candidate_ranges_started_at = Instant::now();
    let candidate_ranges = boundary_candidate_state_ranges_by_token(
        components,
        tokenizer_state_offsets,
        vocab,
        &prefilter,
    );
    if std::env::var_os("GLRMASK_VALIDATE_FAST_BOUNDARY_CANDIDATE_RANGES").is_some() {
        let reference = boundary_candidate_state_ranges_by_token_reference(
            components,
            tokenizer_state_offsets,
            vocab,
            &prefilter,
        );
        if candidate_ranges != reference {
            let keys = candidate_ranges
                .keys()
                .chain(reference.keys())
                .copied()
                .collect::<BTreeSet<_>>();
            for token in keys {
                let fast = candidate_ranges.get(&token);
                let slow = reference.get(&token);
                if fast != slow {
                    eprintln!(
                        "[glrmask/validate][fast_boundary_candidate_ranges_mismatch] token={token} fast={fast:?} reference={slow:?}"
                    );
                    for (component_index, constraint) in components.iter().enumerate() {
                        let inverse = constraint
                            .original_token_to_internal
                            .get(token as usize)
                            .copied()
                            .unwrap_or(u32::MAX);
                        let memberships = constraint
                            .internal_token_to_tokens
                            .iter()
                            .enumerate()
                            .filter_map(|(internal, originals)| {
                                originals.contains(&token).then_some(internal as u32)
                            })
                            .collect::<Vec<_>>();
                        eprintln!(
                            "[glrmask/validate][fast_boundary_candidate_token_inverse] component={component_index} token={token} inverse={inverse} memberships={memberships:?} internal_classes={} inverse_len={}",
                            constraint.internal_token_to_tokens.len(),
                            constraint.original_token_to_internal.len(),
                        );
                    }
                    panic!("fast boundary candidate ranges differ from reference");
                }
            }
        }
    }
    let candidate_ranges_ms = candidate_ranges_started_at.elapsed().as_secs_f64() * 1000.0;
    let candidate_range_rows = candidate_ranges.values().map(Vec::len).sum::<usize>();

    // Each model token is an independent acyclic same-token graph. Run those
    // scans in parallel, then merge in vocabulary order for deterministic
    // output and profiling.
    let candidate_start_visits = AtomicUsize::new(0);
    let distinct_scan_groups = AtomicUsize::new(0);
    let max_candidate_starts = AtomicUsize::new(0);
    let candidate_group_ns = AtomicU64::new(0);
    let residual_scan_ns = AtomicU64::new(0);
    let graph_ns = AtomicU64::new(0);
    let full_graph_groups = AtomicUsize::new(0);
    let token_union_survivors = AtomicUsize::new(0);
    let profile_discovery_detail = compose_profile_enabled();
    let boolean_graph_prefilter =
        std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_BOOLEAN_PREFILTER").is_some();
    let token_union_prefilter =
        std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TOKEN_UNION_PREFILTER").is_some();
    let exact_scan_started_at = Instant::now();
    let results = multi_byte_entries
        .par_iter()
        .filter_map(|&(token_id, bytes)| {
            let owned_reset_scans;
            let reset_scans = if let Some(cache) = reset_suffix_cache.as_ref() {
                (1..bytes.len())
                    .map(|offset| {
                        cache
                            .get(&bytes[offset..])
                            .expect("reset suffix cache must cover every vocabulary suffix")
                    })
                    .collect::<Vec<_>>()
            } else {
                owned_reset_scans = (1..bytes.len())
                    .map(|offset| {
                        scan_component_residual_starts(
                            components,
                            tokenizer_state_offsets,
                            terminal_offsets,
                            &reset_live_bytes,
                            &bytes[offset..],
                            &reset_starts,
                        )
                    })
                    .collect::<Vec<_>>();
                owned_reset_scans.iter().collect::<Vec<_>>()
            };
            let candidate_group_started_at = Instant::now();
            let candidate_groups = candidate_start_state_groups_for_token(
                token_id,
                &candidate_ranges,
                &extra_residual_starts,
                components,
                tokenizer_state_offsets,
            );
            candidate_start_visits.fetch_add(candidate_groups.len(), Ordering::Relaxed);
            max_candidate_starts.fetch_max(candidate_groups.len(), Ordering::Relaxed);
            if profile_discovery_detail {
                candidate_group_ns.fetch_add(
                    candidate_group_started_at.elapsed().as_nanos() as u64,
                    Ordering::Relaxed,
                );
            }
            let residual_scan_started_at = Instant::now();
            let starts_by_scan = scan_component_residual_start_groups(
                components,
                tokenizer_state_offsets,
                terminal_offsets,
                &reset_live_bytes,
                bytes,
                &candidate_groups,
            );
            distinct_scan_groups.fetch_add(starts_by_scan.len(), Ordering::Relaxed);
            if profile_discovery_detail {
                residual_scan_ns.fetch_add(
                    residual_scan_started_at.elapsed().as_nanos() as u64,
                    Ordering::Relaxed,
                );
            }
            let graph_started_at = Instant::now();
            let mut scan_groups = starts_by_scan.into_iter().collect::<Vec<_>>();
            scan_groups.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            if token_union_prefilter {
                let mut residual_union = ResidualScanResult::default();
                let mut reset_union = ResidualScanResult::default();
                let mut contextual_union = ResidualScanResult::default();
                let mut has_residual = false;
                let mut has_reset = false;
                let mut has_contextual = false;
                for (scan, start_states) in &scan_groups {
                    let mut group_residual = false;
                    let mut group_reset = false;
                    let mut group_contextual = false;
                    for &start_state in start_states {
                        if boundary_context_states.contains(&start_state) {
                            group_contextual = true;
                        } else if reset_state_set.contains(&start_state) {
                            group_reset = true;
                        } else {
                            group_residual = true;
                        }
                    }
                    if group_residual {
                        residual_union.union_from(scan);
                        has_residual = true;
                    }
                    if group_reset {
                        reset_union.union_from(scan);
                        has_reset = true;
                    }
                    if group_contextual {
                        contextual_union.union_from(scan);
                        has_contextual = true;
                    }
                }
                residual_union.canonicalize();
                reset_union.canonicalize();
                contextual_union.canonicalize();
                let residual_accepts = has_residual
                    && boundary_token_graph_has_accepting_path(
                        bytes,
                        &residual_union,
                        &reset_scans,
                        seed_terminals,
                        ignore_terminals,
                        interface_pairs,
                        false,
                        false,
                        disallowed_follows,
                        follow_transparent_terminals,
                    );
                let reset_accepts = has_reset
                    && boundary_token_graph_has_accepting_path(
                        bytes,
                        &reset_union,
                        &reset_scans,
                        seed_terminals,
                        ignore_terminals,
                        interface_pairs,
                        false,
                        !ignore_terminals.is_empty(),
                        disallowed_follows,
                        follow_transparent_terminals,
                    );
                let contextual_accepts = has_contextual
                    && boundary_token_graph_has_accepting_path(
                        bytes,
                        &contextual_union,
                        &reset_scans,
                        seed_terminals,
                        ignore_terminals,
                        interface_pairs,
                        true,
                        false,
                        disallowed_follows,
                        follow_transparent_terminals,
                    );
                if !(residual_accepts || reset_accepts || contextual_accepts) {
                    if profile_discovery_detail {
                        graph_ns.fetch_add(
                            graph_started_at.elapsed().as_nanos() as u64,
                            Ordering::Relaxed,
                        );
                    }
                    return None;
                }
                token_union_survivors.fetch_add(1, Ordering::Relaxed);
            }
            let mut local_terminals = FxHashSet::<u32>::default();
            let mut local_witnesses = Vec::new();
            for (arbitrary_scan, start_states) in scan_groups {
                let mut residual_starts = Vec::new();
                let mut reset_starts_for_scan = Vec::new();
                let mut contextual_starts = Vec::new();
                for start_state in start_states {
                    if boundary_context_states.contains(&start_state) {
                        contextual_starts.push(start_state);
                    } else if reset_state_set.contains(&start_state) {
                        reset_starts_for_scan.push(start_state);
                    } else {
                        residual_starts.push(start_state);
                    }
                }
                for (start_states, initial_interface_witnessed, allow_seed_only) in [
                    (residual_starts, false, false),
                    (reset_starts_for_scan, false, !ignore_terminals.is_empty()),
                    (contextual_starts, true, false),
                ] {
                    if start_states.is_empty() {
                        continue;
                    }
                    if boolean_graph_prefilter
                        && !boundary_token_graph_has_accepting_path(
                            bytes,
                            &arbitrary_scan,
                            &reset_scans,
                            seed_terminals,
                            ignore_terminals,
                            interface_pairs,
                            initial_interface_witnessed,
                            allow_seed_only,
                            disallowed_follows,
                            follow_transparent_terminals,
                        )
                    {
                        continue;
                    }
                    full_graph_groups.fetch_add(1, Ordering::Relaxed);
                    let Some((nodes, good, accepting)) = build_boundary_token_graph(
                        bytes,
                        &arbitrary_scan,
                        &reset_scans,
                        seed_terminals,
                        ignore_terminals,
                        interface_pairs,
                        initial_interface_witnessed,
                        allow_seed_only,
                        disallowed_follows,
                        follow_transparent_terminals,
                    ) else {
                        continue;
                    };
                    for (source, node) in nodes.iter().enumerate() {
                        if !good[source] {
                            continue;
                        }
                        for edge in &node.outgoing {
                            if good[edge.target] {
                                local_terminals.insert(edge.terminal);
                            }
                        }
                    }
                    local_witnesses.push(BoundaryTokenWitness {
                        token_id,
                        start_states,
                        nodes,
                        good,
                        accepting,
                    });
                }
            }
            if profile_discovery_detail {
                graph_ns.fetch_add(
                    graph_started_at.elapsed().as_nanos() as u64,
                    Ordering::Relaxed,
                );
            }
            (!local_witnesses.is_empty())
                .then_some((token_id, local_terminals, local_witnesses))
        })
        .collect::<Vec<_>>();
    let exact_scan_ms = exact_scan_started_at.elapsed().as_secs_f64() * 1000.0;

    let merge_started_at = Instant::now();
    let mut discovered = BitSet::new(num_terminals);
    let mut boundary_token_ids = Vec::with_capacity(results.len());
    let mut witnesses = Vec::new();
    for (token_id, terminals, mut token_witnesses) in results {
        boundary_token_ids.push(token_id);
        for terminal in terminals {
            discovered.set(terminal as usize);
        }
        witnesses.append(&mut token_witnesses);
    }
    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_discovery_phases] prelude_ms={discovery_prelude_ms:.3} prefilter_ms={prefilter_ms:.3} suffix_cache_ms={suffix_cache_ms:.3} candidate_ranges_ms={candidate_ranges_ms:.3} exact_scan_ms={exact_scan_ms:.3} merge_ms={merge_ms:.3} candidate_group_cpu_ms={:.3} residual_scan_cpu_ms={:.3} graph_cpu_ms={:.3} full_graph_groups={} boolean_prefilter={} token_union_prefilter={} token_union_survivors={} total_ms={:.3}",
            candidate_group_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            residual_scan_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            graph_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            full_graph_groups.load(Ordering::Relaxed),
            boolean_graph_prefilter,
            token_union_prefilter,
            token_union_survivors.load(Ordering::Relaxed),
            discovery_total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    if std::env::var_os("GLRMASK_DUMP_COMPOSE_BOUNDARY_TOKENS").is_some() {
        eprintln!(
            "[glrmask/dump][constraint_boundary_tokens] prefilter={:?} exact={:?}",
            prefilter,
            boundary_token_ids,
        );
    }
    if compose_profile_enabled() {
        let exact = boundary_token_ids.iter().copied().collect::<BTreeSet<_>>();
        let missing = exact.difference(&prefilter).copied().collect::<Vec<_>>();
        eprintln!(
            "[glrmask/profile][constraint_boundary_candidate_fanout] range_tokens={} range_rows={} ranges_ms={candidate_ranges_ms:.3} scanned_tokens={} raw_start_visits={} distinct_scan_groups={} max_starts={}",
            candidate_ranges.len(),
            candidate_range_rows,
            multi_byte_entries.len(),
            candidate_start_visits.load(Ordering::Relaxed),
            distinct_scan_groups.load(Ordering::Relaxed),
            max_candidate_starts.load(Ordering::Relaxed),
        );
        eprintln!(
            "[glrmask/profile][constraint_boundary_prefilter] enabled={} cross_interface_only={} interface_pairs={} interface_pair_candidates={} context_residual_candidates={} candidates={} scanned={} exact={} missing={} prefilter_ms={prefilter_ms:.3} missing_ids={:?}",
            use_prefilter,
            cross_interface_only,
            interface_pairs_for_prefilter.len(),
            interface_pair_candidates.len(),
            context_residual_candidates.len(),
            prefilter.len(),
            multi_byte_entries.len(),
            exact.len(),
            missing.len(),
            missing.iter().take(32).collect::<Vec<_>>(),
        );
        let suffix_occurrences = multi_byte_entries
            .iter()
            .map(|(_, bytes)| bytes.len() - 1)
            .sum::<usize>();
        eprintln!(
            "[glrmask/profile][constraint_boundary_suffix_cache] tokens={} suffix_occurrences={} unique_suffixes={} cache_ms={suffix_cache_ms:.3} enabled={}",
            multi_byte_entries.len(),
            suffix_occurrences,
            reset_suffix_cache.as_ref().map_or(0, FxHashMap::len),
            reset_suffix_cache.is_some(),
        );
    }
    BoundaryTokenDiscovery {
        terminals: discovered,
        token_ids: boundary_token_ids,
        witnesses,
    }
}


pub(super) fn collect_one_byte_seed_relations_serial(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    seed_terminals: &[bool],
    candidate_states: &[u32],
) -> BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>> {
    let mut relations = BTreeMap::<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>::new();
    let mut tokens_by_byte = vec![Vec::<u32>::new(); 256];
    for (&token_id, bytes) in vocab.entries_map().iter().filter(|(_, bytes)| bytes.len() == 1) {
        tokens_by_byte[bytes[0] as usize].push(token_id);
    }
    let closures = tokenizer.all_singleton_epsilon_closures();
    for &raw_state in candidate_states {
        let source_closure = &closures[raw_state as usize];
        let seed_reachable = source_closure.iter().copied().any(|state| {
            tokenizer
                .matched_terminals_iter(state)
                .chain(tokenizer.possible_future_terminals_iter(state))
                .any(|terminal| {
                    seed_terminals
                        .get(terminal as usize)
                        .copied()
                        .unwrap_or(false)
                })
        });
        if !seed_reachable {
            continue;
        }

        let mut targets_by_byte = BTreeMap::<u8, BTreeSet<u32>>::new();
        for &state in source_closure.iter() {
            for (byte, target) in tokenizer.transitions_from(state) {
                if !tokens_by_byte[byte as usize].is_empty() {
                    targets_by_byte.entry(byte).or_default().insert(target);
                }
            }
        }
        for (byte, targets) in targets_by_byte {
            let mut end_states = BTreeSet::<u32>::new();
            for target in targets {
                end_states.extend(closures[target as usize].iter().copied());
            }
            let mut terminals = BTreeSet::<u32>::new();
            for end_state in end_states {
                terminals.extend(
                    tokenizer
                        .matched_terminals_iter(end_state)
                        .chain(tokenizer.possible_future_terminals_iter(end_state))
                        .filter(|terminal| {
                            seed_terminals
                                .get(*terminal as usize)
                                .copied()
                                .unwrap_or(false)
                        }),
                );
            }
            for terminal in terminals {
                relations
                    .entry(vec![terminal])
                    .or_default()
                    .entry(raw_state)
                    .or_default()
                    .extend(tokens_by_byte[byte as usize].iter().copied());
            }
        }
    }
    relations
}

pub(super) fn collect_one_byte_seed_relations_parallel(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    seed_terminals: &[bool],
    candidate_states: &[u32],
) -> BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>> {
    let mut tokens_by_byte = vec![Vec::<u32>::new(); 256];
    for (&token_id, bytes) in vocab.entries_map().iter().filter(|(_, bytes)| bytes.len() == 1) {
        tokens_by_byte[bytes[0] as usize].push(token_id);
    }
    let seed_terminal_ids = seed_terminals
        .iter()
        .enumerate()
        .filter_map(|(terminal, &selected)| selected.then_some(terminal as u32))
        .collect::<Vec<_>>();
    let closures = tokenizer.all_singleton_epsilon_closures();
    // A relation entry only records existence: for one source state and byte,
    // a seed terminal is supported when any target epsilon-closure state either
    // finishes it or can continue it. Process each transition independently;
    // duplicate `(terminal, state, byte)` entries are harmless and collapse in
    // the final ordered relation. This avoids per-state maps and sets across the
    // million-state common case where every epsilon closure is a singleton.
    let entries = candidate_states
        .par_iter()
        .copied()
        .fold(Vec::<(u32, u32, u8)>::new, |mut output, raw_state| {
            let source_closure = &closures[raw_state as usize];
            let seed_reachable = source_closure.iter().copied().any(|state| {
                seed_terminal_ids.iter().copied().any(|terminal| {
                    tokenizer
                        .matched_terminal_bitset(state)
                        .contains(terminal as usize)
                        || tokenizer
                            .possible_future_terminals(state)
                            .contains(terminal as usize)
                })
            });
            if !seed_reachable {
                return output;
            }

            for &state in source_closure.iter() {
                for (byte, target) in tokenizer.transitions_from(state) {
                    if tokens_by_byte[byte as usize].is_empty() {
                        continue;
                    }
                    let target_closure = &closures[target as usize];
                    for &terminal in &seed_terminal_ids {
                        let supported = target_closure.iter().copied().any(|end_state| {
                            tokenizer
                                .matched_terminal_bitset(end_state)
                                .contains(terminal as usize)
                                || tokenizer
                                    .possible_future_terminals(end_state)
                                    .contains(terminal as usize)
                        });
                        if supported {
                            output.push((terminal, raw_state, byte));
                        }
                    }
                }
            }
            output
        })
        .reduce(Vec::new, |mut left, mut right| {
            left.append(&mut right);
            left
        });

    let mut relations = BTreeMap::<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>::new();
    for (terminal, raw_state, byte) in entries {
        relations
            .entry(vec![terminal])
            .or_default()
            .entry(raw_state)
            .or_default()
            .extend(tokens_by_byte[byte as usize].iter().copied());
    }
    relations
}

pub(super) fn collect_one_byte_seed_relations(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    seed_terminals: &[bool],
    candidate_states: &[u32],
    relations: &mut BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
) {
    let candidate = if rayon::current_num_threads() == 1
        || std::env::var_os("GLRMASK_COMPOSE_SERIAL_ONE_BYTE_REFERENCE").is_some()
    {
        collect_one_byte_seed_relations_serial(
            tokenizer,
            vocab,
            seed_terminals,
            candidate_states,
        )
    } else {
        collect_one_byte_seed_relations_parallel(
            tokenizer,
            vocab,
            seed_terminals,
            candidate_states,
        )
    };
    if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_ONE_BYTE_PARALLEL").is_some()
        && rayon::current_num_threads() > 1
    {
        let reference = collect_one_byte_seed_relations_serial(
            tokenizer,
            vocab,
            seed_terminals,
            candidate_states,
        );
        assert_eq!(candidate, reference, "parallel one-byte boundary relation differs from serial reference");
        eprintln!(
            "[glrmask/validate][compose_one_byte_parallel] relation_rows={} exact=true",
            candidate.len(),
        );
    }
    for (terminal_path, by_state) in candidate {
        let destination = relations.entry(terminal_path).or_default();
        for (state, tokens) in by_state {
            destination.entry(state).or_default().extend(tokens);
        }
    }
}

pub(super) fn collect_one_byte_seed_relations_components(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    vocab: &Vocab,
    seed_terminals: &[bool],
) -> BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>> {
    debug_assert_eq!(components.len(), tokenizer_state_offsets.len());
    debug_assert_eq!(components.len(), terminal_offsets.len());
    let mut relations = BTreeMap::<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>::new();

    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index];
        let state_offset = tokenizer_state_offsets[component_index];
        let mut local_seed_terminals =
            vec![false; component.tokenizer.num_terminals() as usize];
        for (local_terminal, selected) in local_seed_terminals.iter_mut().enumerate() {
            *selected = seed_terminals
                .get(terminal_offset as usize + local_terminal)
                .copied()
                .unwrap_or(false);
        }
        if !local_seed_terminals.iter().any(|&selected| selected) {
            continue;
        }

        let mut candidate_states = Vec::<u32>::new();
        if component.terminal_live_states.len() == local_seed_terminals.len() {
            for (terminal, &selected) in local_seed_terminals.iter().enumerate() {
                if selected {
                    candidate_states.extend_from_slice(&component.terminal_live_states[terminal]);
                }
            }
            candidate_states.sort_unstable();
            candidate_states.dedup();
        } else {
            candidate_states.extend(0..component.tokenizer.num_states());
        }
        let mut local_relations =
            BTreeMap::<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>::new();
        collect_one_byte_seed_relations(
            &component.tokenizer,
            vocab,
            &local_seed_terminals,
            &candidate_states,
            &mut local_relations,
        );
        if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_ONE_BYTE_STATE_INDEX").is_some() {
            let all_states = (0..component.tokenizer.num_states()).collect::<Vec<_>>();
            let reference = collect_one_byte_seed_relations_serial(
                &component.tokenizer,
                vocab,
                &local_seed_terminals,
                &all_states,
            );
            assert_eq!(
                local_relations, reference,
                "terminal-live-state index omitted an exact one-byte boundary relation",
            );
            eprintln!(
                "[glrmask/validate][compose_one_byte_state_index] component={} candidates={} states={} exact=true",
                component_index,
                candidate_states.len(),
                component.tokenizer.num_states(),
            );
        }
        let local_start = component.tokenizer.start_state();
        for (local_path, by_local_state) in local_relations {
            let global_path = local_path
                .into_iter()
                .map(|terminal| terminal_offset + terminal)
                .collect::<Vec<_>>();
            let destination = relations.entry(global_path).or_default();
            for (local_state, tokens) in by_local_state {
                destination
                    .entry(state_offset + local_state)
                    .or_default()
                    .extend(tokens.iter().copied());
                // The fresh merged state zero epsilon-dispatches to every
                // component start state. Its exact one-byte relation is the
                // union of those local start-state relations.
                if local_state == local_start {
                    destination.entry(0).or_default().extend(tokens);
                }
            }
        }
    }
    relations
}

pub(super) fn component_state_coordinate_map(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    merged_tokenizer_state_count: usize,
) -> Result<ManyToOneIdMap, String> {
    let mut state_to_global = vec![u32::MAX; merged_tokenizer_state_count];
    let mut global_to_states = vec![vec![0u32]];
    let mut representatives = vec![0u32];
    if let Some(reset) = state_to_global.first_mut() {
        *reset = 0;
    }
    for (component_index, component) in components.iter().enumerate() {
        let state_offset = tokenizer_state_offsets[component_index];
        let internal_tsid_to_states = component.internal_tsid_groups();
        let local_tsid_count = internal_tsid_to_states.len();
        if local_tsid_count == 0 {
            return Err(format!("component {component_index} has no internal TSIDs"));
        }
        if tokenizer_tsid_relation_is_singleton(component) {
            for local_states in internal_tsid_to_states {
                let mut merged_states = Vec::with_capacity(local_states.len());
                for &local_state in local_states {
                    let merged_state = state_offset
                        .checked_add(local_state)
                        .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
                    if merged_state != 0 {
                        merged_states.push(merged_state);
                    }
                }
                if merged_states.is_empty() {
                    continue;
                }
                let global_tsid = global_to_states.len() as u32;
                for &merged_state in &merged_states {
                    let Some(slot) = state_to_global.get_mut(merged_state as usize) else {
                        return Err(format!(
                            "component {component_index} tokenizer state {merged_state} lies outside merged tokenizer",
                        ));
                    };
                    if *slot != u32::MAX {
                        return Err(format!(
                            "merged tokenizer state {merged_state} belongs to multiple component TSIDs",
                        ));
                    }
                    *slot = global_tsid;
                }
                representatives.push(merged_states[0]);
                global_to_states.push(merged_states);
            }
            continue;
        }
        let mut states_by_signature = BTreeMap::<Vec<u32>, Vec<u32>>::new();
        for local_state in 0..component.tokenizer.num_states() {
            let mut signature = component.internal_tsids_for_state(local_state).to_vec();
            signature.sort_unstable();
            signature.dedup();
            if signature.is_empty() {
                return Err(format!(
                    "component {component_index} tokenizer state {local_state} has no internal TSID"
                ));
            }
            if let Some(&bad) = signature
                .iter()
                .find(|&&tsid| tsid as usize >= local_tsid_count)
            {
                return Err(format!(
                    "component {component_index} tokenizer state {local_state} references out-of-range internal TSID {bad}"
                ));
            }
            let merged_state = state_offset
                .checked_add(local_state)
                .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
            if merged_state as usize >= merged_tokenizer_state_count {
                return Err(format!(
                    "component {component_index} tokenizer state {local_state} maps outside merged tokenizer"
                ));
            }
            if merged_state != 0 {
                states_by_signature
                    .entry(signature)
                    .or_default()
                    .push(merged_state);
            }
        }
        for (_signature, mut merged_states) in states_by_signature {
            merged_states.sort_unstable();
            merged_states.dedup();
            let global_tsid = global_to_states.len() as u32;
            for &merged_state in &merged_states {
                let Some(slot) = state_to_global.get_mut(merged_state as usize) else {
                    return Err(format!(
                        "component {component_index} tokenizer state {merged_state} lies outside merged tokenizer",
                    ));
                };
                if *slot != u32::MAX {
                    return Err(format!(
                        "merged tokenizer state {merged_state} belongs to multiple exact membership classes",
                    ));
                }
                *slot = global_tsid;
            }
            representatives.push(merged_states[0]);
            global_to_states.push(merged_states);
        }
    }
    if state_to_global.iter().any(|&tsid| tsid == u32::MAX) {
        return Err("component TSID map does not cover merged tokenizer".into());
    }
    Ok(ManyToOneIdMap {
        original_to_internal: state_to_global,
        internal_to_originals: global_to_states,
        representative_original_ids: representatives,
    })
}

pub(super) fn boundary_id_map_for_selected_tokens(
    component_state_map: &ManyToOneIdMap,
    selected_original_tokens: &[u32],
) -> Result<InternalIdMap, String> {
    if selected_original_tokens.is_empty() {
        return Err("boundary witness construction selected no model tokens".into());
    }
    let max_original_token = selected_original_tokens.last().copied().unwrap_or(0);
    let mut original_to_internal = vec![u32::MAX; max_original_token as usize + 1];
    let mut internal_to_originals = Vec::with_capacity(selected_original_tokens.len());
    let mut token_representatives = Vec::with_capacity(selected_original_tokens.len());
    for (internal, &original) in selected_original_tokens.iter().enumerate() {
        original_to_internal[original as usize] = internal as u32;
        internal_to_originals.push(vec![original]);
        token_representatives.push(original);
    }
    Ok(InternalIdMap {
        tokenizer_states: ManyToOneIdMap {
            original_to_internal: Vec::new(),
            internal_to_originals: component_state_map
                .representative_original_ids
                .iter()
                .map(|&state| vec![state])
                .collect(),
            representative_original_ids: component_state_map
                .representative_original_ids
                .clone(),
        },
        vocab_tokens: ManyToOneIdMap {
            original_to_internal,
            internal_to_originals,
            representative_original_ids: token_representatives,
        },
        deferred_vocab_singleton_original_ids: None,
    })
}



pub(super) fn deterministic_weighted_prefix_dwa_from_small_acyclic_nwa(
    nwa: &NWA,
    max_visits: usize,
    max_words: usize,
    max_word_len: usize,
) -> Option<DWA> {
    if !nwa.is_acyclic() || nwa.start_states().is_empty() {
        return None;
    }

    let mut words = BTreeMap::<Vec<i32>, Weight>::new();
    let mut stack = nwa
        .start_states()
        .iter()
        .copied()
        .map(|state| (state, Vec::<i32>::new(), Weight::all()))
        .collect::<Vec<_>>();
    let mut visits = 0usize;
    while let Some((state_id, word, support)) = stack.pop() {
        visits += 1;
        if visits > max_visits || word.len() > max_word_len {
            return None;
        }
        let state = nwa.states().get(state_id as usize)?;
        if let Some(final_weight) = state.final_weight.as_ref() {
            let accepted = support.intersection(final_weight);
            if !accepted.is_empty() {
                match words.entry(word.clone()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(accepted);
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        let merged = entry.get().union(&accepted);
                        *entry.get_mut() = merged;
                    }
                }
                if words.len() > max_words {
                    return None;
                }
            }
        }
        for (target, edge_weight) in &state.epsilons {
            let next_support = support.intersection(edge_weight);
            if !next_support.is_empty() {
                stack.push((*target, word.clone(), next_support));
            }
        }
        for (&label, targets) in &state.transitions {
            for (target, edge_weight) in targets {
                let next_support = support.intersection(edge_weight);
                if next_support.is_empty() {
                    continue;
                }
                let mut next_word = word.clone();
                next_word.push(label);
                if next_word.len() > max_word_len {
                    return None;
                }
                stack.push((*target, next_word, next_support));
            }
        }
    }

    #[derive(Default)]
    struct PrefixNode {
        children: BTreeMap<i32, usize>,
        final_weight: Option<Weight>,
    }
    let mut nodes = vec![PrefixNode::default()];
    for (word, support) in words {
        let mut node = 0usize;
        for label in word {
            let next = if let Some(&next) = nodes[node].children.get(&label) {
                next
            } else {
                let next = nodes.len();
                nodes.push(PrefixNode::default());
                nodes[node].children.insert(label, next);
                next
            };
            node = next;
        }
        nodes[node].final_weight = Some(match nodes[node].final_weight.take() {
            Some(existing) => existing.union(&support),
            None => support,
        });
    }

    // Every child is allocated after its parent, so reverse numeric order is
    // a reverse topological order for this trie. `live[node]` is the exact
    // support of all accepted words in the subtree rooted at `node`.
    let mut live = vec![Weight::empty(); nodes.len()];
    for node_id in (0..nodes.len()).rev() {
        let mut support = nodes[node_id]
            .final_weight
            .clone()
            .unwrap_or_else(Weight::empty);
        for &child in nodes[node_id].children.values() {
            support = support.union(&live[child]);
        }
        live[node_id] = support;
    }

    let mut states = Vec::<DWAState>::with_capacity(nodes.len());
    for node in nodes {
        let transitions = node
            .children
            .into_iter()
            .filter_map(|(label, child)| {
                let weight = live[child].clone();
                (!weight.is_empty()).then_some((label, (child as u32, weight)))
            })
            .collect::<BTreeMap<_, _>>();
        states.push(DWAState {
            transitions: transitions.into(),
            final_weight: node.final_weight.filter(|weight| !weight.is_empty()),
        });
    }
    Some(DWA::from_parts(states, 0))
}

pub(super) fn direct_boundary_terminal_automaton(
    num_states: usize,
    component_state_map: Option<&ManyToOneIdMap>,
    vocab: &Vocab,
    coordinate_original_tokens: &[u32],
    seed_relations: BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
    one_byte_ms: f64,
    discovery: &BoundaryTokenDiscovery,
    globally_erasable_ignore_terminals: &BitSet,
    control_terminals: &BTreeSet<u32>,
    terminal_offsets: &[u32],
    tokenizer_state_offsets: &[u32],
    delta_plan: Option<&ConcreteBoundaryDeltaPlan>,
    start_component_filter: Option<usize>,
    preserve_symbolic_nwa: bool,
) -> Result<MappedArtifact<TerminalAutomaton>, String> {
    let total_started_at = Instant::now();
    if start_component_filter.is_some() && delta_plan.is_none() {
        return Err(
            "per-start-component boundary partition requires the concrete cross-boundary plan"
                .to_string(),
        );
    }

    // Keep the token coordinate published to the owned component-preparation
    // lane exactly, even when later semantic factoring proves some candidates
    // redundant.  The coordinate may contain unused token classes; changing it
    // after publication would invalidate the concurrently prepared remap.
    let selected_original_tokens = coordinate_original_tokens
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let max_original_token = vocab
        .entries_map()
        .keys()
        .next_back()
        .copied()
        .into_iter()
        .chain(selected_original_tokens.iter().copied())
        .max()
        .unwrap_or(0);
    let mut original_to_internal = vec![u32::MAX; max_original_token as usize + 1];
    let mut internal_to_originals = Vec::with_capacity(selected_original_tokens.len());
    let mut token_representatives = Vec::with_capacity(selected_original_tokens.len());
    for (internal, original) in selected_original_tokens.iter().copied().enumerate() {
        original_to_internal[original as usize] = internal as u32;
        internal_to_originals.push(vec![original]);
        token_representatives.push(original);
    }
    let vocab_tokens = ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids: token_representatives,
    };

    // Prefer the already-final component TSID coordinate. This avoids
    // constructing a second million-state quotient only to reconcile it back
    // immediately after boundary parser compilation. The finer coordinate is
    // exact: unsupported component TSIDs simply do not occur in boundary
    // weights.
    let quotient_started_at = Instant::now();
    let tokenizer_states = if let Some(component_state_map) = component_state_map {
        // Boundary weights use the compact component-TSID numbering, but the
        // independently prepared final component map may number equivalent
        // classes differently. Retain one raw representative per TSID so the
        // final reconciliation can translate exactly without cloning the full
        // million-state inverse map.
        ManyToOneIdMap {
            original_to_internal: Vec::new(),
            internal_to_originals: component_state_map
                .representative_original_ids
                .iter()
                .map(|&state| vec![state])
                .collect(),
            representative_original_ids: component_state_map
                .representative_original_ids
                .clone(),
        }
    } else {
        let mut state_signatures = vec![Vec::<(u32, Vec<u32>)>::new(); num_states];
        let mut weight_row = 0u32;
        for by_state in seed_relations.values() {
            for (&state, originals) in by_state {
                state_signatures[state as usize].push((
                    weight_row,
                    originals.iter().copied().collect::<Vec<_>>(),
                ));
            }
            weight_row += 1;
        }
        for witness in &discovery.witnesses {
            for &state in &witness.start_states {
                state_signatures[state as usize].push((weight_row, vec![witness.token_id]));
            }
            weight_row += 1;
        }
        let mut class_by_signature = BTreeMap::<Vec<(u32, Vec<u32>)>, u32>::new();
        let mut state_to_class = vec![u32::MAX; num_states];
        let mut state_representatives = Vec::<u32>::new();
        for (state, signature) in state_signatures.into_iter().enumerate() {
            let class = if let Some(&class) = class_by_signature.get(&signature) {
                class
            } else {
                let class = class_by_signature.len() as u32;
                class_by_signature.insert(signature, class);
                state_representatives.push(state as u32);
                class
            };
            state_to_class[state] = class;
        }
        ManyToOneIdMap::from_original_to_internal_with_representatives(
            state_to_class,
            state_representatives.len() as u32,
            state_representatives,
        )
    };
    let quotient_ms = quotient_started_at.elapsed().as_secs_f64() * 1000.0;
    let id_map = InternalIdMap {
        tokenizer_states,
        vocab_tokens,
        deferred_vocab_singleton_original_ids: None,
    };

    let state_to_tsid = |state: u32| {
        component_state_map
            .map(|map| map.original_to_internal[state as usize])
            .unwrap_or_else(|| id_map.tokenizer_states.original_to_internal[state as usize])
    };
    let relation_weight = |by_state: BTreeMap<u32, BTreeSet<u32>>| {
        let mut tokens_by_tsid = BTreeMap::<u32, BTreeSet<u32>>::new();
        for (state, originals) in by_state {
            let tsid = state_to_tsid(state);
            let tokens = tokens_by_tsid.entry(tsid).or_default();
            tokens.extend(
                originals
                    .into_iter()
                    .filter_map(|original| id_map.internal_token_for_original(original)),
            );
        }
        Weight::from_per_tsid_token_sets(tokens_by_tsid.into_iter().map(|(tsid, tokens)| {
            (
                tsid,
                tokens.into_iter().collect::<RangeSetBlaze<_>>(),
            )
        }))
    };

    let build_started_at = Instant::now();

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct CanonicalNodeKey {
        accepting: bool,
        transitions: Vec<(u32, usize)>,
        epsilons: Vec<usize>,
    }
    #[derive(Debug)]
    struct CanonicalNode {
        accepting: bool,
        transitions: Vec<(u32, usize)>,
        epsilons: Vec<usize>,
    }

    // Put each witness's `(start TSID, token)` support on the epsilon edge
    // entering its graph. The graph itself then denotes only an unweighted
    // terminal suffix language, so structurally equal suffixes can be shared
    // across all residual classes and model tokens without leaking support.
    let mut canonical_by_key = BTreeMap::<CanonicalNodeKey, usize>::new();
    let mut canonical_nodes = Vec::<CanonicalNode>::new();
    // Do not build and persistently union one singleton Weight per boundary
    // witness. Large composed children routinely have tens of thousands of
    // fused vocabulary witnesses that collapse onto the same canonical suffix
    // graph; repeated Weight::union then dominates composition. Accumulate the
    // exact support relation first and materialize one Weight per canonical
    // start only after all witnesses have been canonicalized.
    let mut start_tokens_by_canonical =
        BTreeMap::<usize, BTreeMap<u32, BTreeSet<u32>>>::new();
    let mut intern_canonical = |key: CanonicalNodeKey| -> usize {
        if let Some(&existing) = canonical_by_key.get(&key) {
            existing
        } else {
            let canonical = canonical_nodes.len();
            canonical_nodes.push(CanonicalNode {
                accepting: key.accepting,
                transitions: key.transitions.clone(),
                epsilons: key.epsilons.clone(),
            });
            canonical_by_key.insert(key, canonical);
            canonical
        }
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct ExpandedKey {
        local: usize,
        origin_component: Option<usize>,
        last_component: Option<usize>,
        crossed: bool,
        changed_count: u8,
        unsafe_path: bool,
    }
    #[derive(Debug, Clone)]
    struct ExpandedEdge {
        target: usize,
        terminal: u32,
    }
    #[derive(Debug, Clone)]
    struct ExpandedNode {
        key: ExpandedKey,
        outgoing: Vec<ExpandedEdge>,
    }
    #[derive(Debug, Clone, Copy)]
    enum DeltaLane {
        CrossedFull,
        LocalComplexFull,
        LocalDeltaNovelty,
    }

    let terminal_component = |terminal: u32| -> usize {
        terminal_offsets
            .partition_point(|&offset| offset <= terminal)
            .saturating_sub(1)
    };
    let tokenizer_state_component = |state: u32| -> Option<usize> {
        if state == 0 {
            None
        } else {
            Some(
                tokenizer_state_offsets
                    .partition_point(|&offset| offset <= state)
                    .saturating_sub(1),
            )
        }
    };
    let mut delta_cross_lane_starts = 0usize;
    let mut delta_cross_tokens = BTreeSet::<u32>::new();
    let mut delta_complex_lane_starts = 0usize;
    let mut delta_single_lane_starts = 0usize;
    let mut delta_start_groups = 0usize;
    let cross_lane_only =
        std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_CROSS_LANE_ONLY").is_some();

    for witness in &discovery.witnesses {
        let Some(internal_token) = id_map.internal_token_for_original(witness.token_id) else {
            continue;
        };

        let Some(delta_plan) = delta_plan else {
            let witness_tsids = witness
                .start_states
                .iter()
                .map(|&state| state_to_tsid(state))
                .collect::<BTreeSet<_>>();
            if witness_tsids.is_empty() {
                continue;
            }

            let mut local_to_canonical = vec![usize::MAX; witness.nodes.len()];
            let mut good_nodes = witness
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(local, node)| witness.good[local].then_some((local, node.key.offset)))
                .collect::<Vec<_>>();
            good_nodes.sort_unstable_by(|left, right| right.1.cmp(&left.1));
            for (local, _) in good_nodes {
                let mut transitions = Vec::new();
                let mut epsilons = Vec::new();
                for edge in witness.nodes[local]
                    .outgoing
                    .iter()
                    .filter(|edge| witness.good[edge.target])
                {
                    let target = local_to_canonical[edge.target];
                    debug_assert_ne!(target, usize::MAX);
                    if globally_erasable_ignore_terminals.contains(edge.terminal as usize) {
                        epsilons.push(target);
                    } else {
                        transitions.push((edge.terminal, target));
                    }
                }
                transitions.sort_unstable();
                transitions.dedup();
                epsilons.sort_unstable();
                epsilons.dedup();
                local_to_canonical[local] = intern_canonical(CanonicalNodeKey {
                    accepting: witness.accepting[local],
                    transitions,
                    epsilons,
                });
            }
            let start = local_to_canonical[0];
            let tokens_by_tsid = start_tokens_by_canonical.entry(start).or_default();
            for tsid in witness_tsids {
                tokens_by_tsid.entry(tsid).or_default().insert(internal_token);
            }
            continue;
        };

        // The same model token may be valid from tokenizer states owned by
        // different cached components. Keep those support classes separate:
        // "component-local" is a statement about a particular start component,
        // not about the token globally.
        let mut starts_by_component = BTreeMap::<Option<usize>, Vec<u32>>::new();
        for &state in &witness.start_states {
            starts_by_component
                .entry(tokenizer_state_component(state))
                .or_default()
                .push(state);
        }
        for (initial_component, start_states) in starts_by_component {
            if initial_component.is_some()
                && start_component_filter.is_some()
                && initial_component != start_component_filter
            {
                continue;
            }
            delta_start_groups += 1;
            let witness_tsids = start_states
                .iter()
                .map(|&state| state_to_tsid(state))
                .collect::<BTreeSet<_>>();
            if witness_tsids.is_empty() {
                continue;
            }

            let start_key = ExpandedKey {
                local: 0,
                origin_component: initial_component,
                last_component: initial_component,
                crossed: false,
                changed_count: 0,
                // Merged tokenizer state 0 epsilon-dispatches to every
                // component start, and component parser-DWA transport maps each
                // local start TSID onto this same global TSID 0. Therefore a
                // reset-origin path becomes component-local as soon as its
                // first committed terminal chooses a component; no extra
                // conservatism is required merely because the dispatcher was
                // the lexical start state.
                unsafe_path: false,
            };
            let mut expanded_by_key = FxHashMap::<ExpandedKey, usize>::default();
            let mut expanded_nodes = vec![ExpandedNode {
                key: start_key,
                outgoing: Vec::new(),
            }];
            expanded_by_key.insert(start_key, 0);
            let mut queue = VecDeque::from([0usize]);
            while let Some(source) = queue.pop_front() {
                let source_key = expanded_nodes[source].key;
                if !witness.good[source_key.local] {
                    continue;
                }
                let source_edges = witness.nodes[source_key.local]
                    .outgoing
                    .iter()
                    .filter(|edge| witness.good[edge.target])
                    .cloned()
                    .collect::<Vec<_>>();
                for edge in source_edges {
                    let next_component = terminal_component(edge.terminal);
                    let origin_component = source_key.origin_component.or(Some(next_component));
                    let parser_relevant = !globally_erasable_ignore_terminals
                        .contains(edge.terminal as usize);
                    let changed = parser_relevant
                        && delta_plan.by_global_terminal.contains_key(&edge.terminal);
                    let unsafe_terminal = parser_relevant
                        && delta_plan.unsafe_terminals.contains(&edge.terminal);
                    let switched = source_key
                        .last_component
                        .is_some_and(|component| component != next_component);
                    let next_key = ExpandedKey {
                        local: edge.target,
                        origin_component,
                        last_component: Some(next_component),
                        crossed: source_key.crossed || switched,
                        changed_count: source_key
                            .changed_count
                            .saturating_add(u8::from(changed))
                            .min(2),
                        unsafe_path: source_key.unsafe_path || unsafe_terminal,
                    };
                    let target = if let Some(&target) = expanded_by_key.get(&next_key) {
                        target
                    } else {
                        let target = expanded_nodes.len();
                        expanded_by_key.insert(next_key, target);
                        expanded_nodes.push(ExpandedNode {
                            key: next_key,
                            outgoing: Vec::new(),
                        });
                        queue.push_back(target);
                        target
                    };
                    expanded_nodes[source].outgoing.push(ExpandedEdge {
                        target,
                        terminal: edge.terminal,
                    });
                }
            }

            // Every edge consumes positive byte width in the witness DAG, so
            // descending byte offset is a reverse topological order even after
            // the finite metadata expansion above.
            let mut reverse_order = (0..expanded_nodes.len()).collect::<Vec<_>>();
            reverse_order.sort_unstable_by_key(|&node| {
                std::cmp::Reverse(witness.nodes[expanded_nodes[node].key.local].key.offset)
            });

            for lane in [
                DeltaLane::CrossedFull,
                DeltaLane::LocalComplexFull,
                DeltaLane::LocalDeltaNovelty,
            ] {
                if cross_lane_only && !matches!(lane, DeltaLane::CrossedFull) {
                    continue;
                }
                // For a safe component-local terminal word with one or more
                // ordinary changed terminals, do not rebuild the whole
                // composed word.  For every changed terminal t we have proved
                // Old_t âŠ† New_t and materialized the disjoint remainder
                // Delta_t = New_t \\ Old_t.  Therefore, for a word t1..tn,
                //
                //   New_1..New_n \\ Old_1..Old_n
                //
                // is the disjoint union obtained by choosing the *first*
                // changed occurrence that takes Delta: all earlier changed
                // occurrences take Old, that occurrence takes Delta, and all
                // later occurrences take New.  The cached component parser
                // artifact supplies the omitted all-Old branch.  The boolean
                // product below is exactly that first-delta decomposition and
                // works for arbitrarily many changed terminals without a full
                // local repair lane.
                if matches!(lane, DeltaLane::LocalDeltaNovelty) {
                    let lexical_accepts = |key: ExpandedKey| {
                        witness.accepting[key.local]
                            && start_component_filter
                                .is_none_or(|required| key.origin_component == Some(required))
                            && !key.crossed
                            && !key.unsafe_path
                            && key.changed_count != 0
                    };
                    let mut productive = vec![[false; 2]; expanded_nodes.len()];
                    let mut expanded_to_canonical =
                        vec![[usize::MAX; 2]; expanded_nodes.len()];

                    for &source in &reverse_order {
                        for novelty_seen in [true, false] {
                            let seen_index = usize::from(novelty_seen);
                            let accepting = novelty_seen && lexical_accepts(expanded_nodes[source].key);
                            let mut transitions = Vec::new();
                            let mut epsilons = Vec::new();

                            for edge in &expanded_nodes[source].outgoing {
                                if globally_erasable_ignore_terminals
                                    .contains(edge.terminal as usize)
                                {
                                    if productive[edge.target][seen_index] {
                                        let target =
                                            expanded_to_canonical[edge.target][seen_index];
                                        debug_assert_ne!(target, usize::MAX);
                                        epsilons.push(target);
                                    }
                                    continue;
                                }

                                if let Some(entry) = delta_plan.by_global_terminal.get(&edge.terminal) {
                                    if novelty_seen {
                                        if productive[edge.target][1] {
                                            let target = expanded_to_canonical[edge.target][1];
                                            debug_assert_ne!(target, usize::MAX);
                                            // Once novelty has occurred, later
                                            // changed terminals may use all of
                                            // New = Old âˆª Delta.
                                            transitions.push((edge.terminal, target));
                                        }
                                    } else {
                                        if productive[edge.target][0] {
                                            let target = expanded_to_canonical[edge.target][0];
                                            debug_assert_ne!(target, usize::MAX);
                                            transitions.push((entry.old_terminal, target));
                                        }
                                        if productive[edge.target][1] {
                                            let target = expanded_to_canonical[edge.target][1];
                                            debug_assert_ne!(target, usize::MAX);
                                            transitions.push((entry.delta_terminal, target));
                                        }
                                    }
                                } else if productive[edge.target][seen_index] {
                                    let target = expanded_to_canonical[edge.target][seen_index];
                                    debug_assert_ne!(target, usize::MAX);
                                    transitions.push((edge.terminal, target));
                                }
                            }

                            transitions.sort_unstable();
                            transitions.dedup();
                            epsilons.sort_unstable();
                            epsilons.dedup();
                            if !accepting && transitions.is_empty() && epsilons.is_empty() {
                                continue;
                            }
                            productive[source][seen_index] = true;
                            expanded_to_canonical[source][seen_index] =
                                intern_canonical(CanonicalNodeKey {
                                    accepting,
                                    transitions,
                                    epsilons,
                                });
                        }
                    }

                    if !productive[0][0] {
                        continue;
                    }
                    let start = expanded_to_canonical[0][0];
                    let tokens_by_tsid = start_tokens_by_canonical.entry(start).or_default();
                    for &tsid in &witness_tsids {
                        tokens_by_tsid.entry(tsid).or_default().insert(internal_token);
                    }
                    delta_single_lane_starts += 1;
                    continue;
                }

                let accepts = |key: ExpandedKey| {
                    if !witness.accepting[key.local] {
                        return false;
                    }
                    if start_component_filter
                        .is_some_and(|required| key.origin_component != Some(required))
                    {
                        return false;
                    }
                    match lane {
                        DeltaLane::CrossedFull => {
                            if std::env::var_os("GLRMASK_EXPERIMENT_STRICT_INTERFACE_CROSS_LANE").is_some() {
                                key.crossed && witness.nodes[key.local].key.interface_witnessed
                            } else {
                                key.crossed
                            }
                        }
                        DeltaLane::LocalComplexFull => !key.crossed && key.unsafe_path,
                        DeltaLane::LocalDeltaNovelty => unreachable!(),
                    }
                };
                let mut productive = expanded_nodes
                    .iter()
                    .map(|node| accepts(node.key))
                    .collect::<Vec<_>>();
                for &source in &reverse_order {
                    if !productive[source]
                        && expanded_nodes[source]
                            .outgoing
                            .iter()
                            .any(|edge| productive[edge.target])
                    {
                        productive[source] = true;
                    }
                }
                if !productive[0] {
                    continue;
                }

                let mut expanded_to_canonical = vec![usize::MAX; expanded_nodes.len()];
                for &source in &reverse_order {
                    if !productive[source] {
                        continue;
                    }
                    let mut transitions = Vec::new();
                    let mut epsilons = Vec::new();
                    for edge in expanded_nodes[source]
                        .outgoing
                        .iter()
                        .filter(|edge| productive[edge.target])
                    {
                        let target = expanded_to_canonical[edge.target];
                        debug_assert_ne!(target, usize::MAX);
                        if globally_erasable_ignore_terminals.contains(edge.terminal as usize) {
                            epsilons.push(target);
                        } else {
                            let terminal = match lane {
                                DeltaLane::LocalDeltaNovelty => unreachable!(),
                                DeltaLane::CrossedFull | DeltaLane::LocalComplexFull => edge.terminal,
                            };
                            transitions.push((terminal, target));
                        }
                    }
                    transitions.sort_unstable();
                    transitions.dedup();
                    epsilons.sort_unstable();
                    epsilons.dedup();
                    expanded_to_canonical[source] = intern_canonical(CanonicalNodeKey {
                        accepting: accepts(expanded_nodes[source].key),
                        transitions,
                        epsilons,
                    });
                }
                let start = expanded_to_canonical[0];
                let tokens_by_tsid = start_tokens_by_canonical.entry(start).or_default();
                for &tsid in &witness_tsids {
                    tokens_by_tsid.entry(tsid).or_default().insert(internal_token);
                }
                match lane {
                    DeltaLane::CrossedFull => {
                        delta_cross_lane_starts += 1;
                        delta_cross_tokens.insert(witness.token_id);
                    },
                    DeltaLane::LocalComplexFull => delta_complex_lane_starts += 1,
                    DeltaLane::LocalDeltaNovelty => unreachable!(),
                }
            }
            // Safe, non-crossing accepting paths with zero changed terminals
            // are intentionally absent: their parser behavior is already in
            // the transported cached component parser DWA.
        }
    }

    if let Some(plan) = delta_plan
        && plan.by_global_terminal.is_empty()
        && plan.unsafe_terminals.is_empty()
    {
        assert_eq!(
            delta_complex_lane_starts, 0,
            "empty boundary delta plan must not produce a productive local-complex repair lane",
        );
        assert_eq!(
            delta_single_lane_starts, 0,
            "empty boundary delta plan must not produce a productive local-delta repair lane",
        );
    }

    let start_weights_by_canonical = start_tokens_by_canonical
        .into_iter()
        .map(|(canonical, tokens_by_tsid)| {
            let weight = Weight::from_per_tsid_token_sets(
                tokens_by_tsid.into_iter().map(|(tsid, tokens)| {
                    (tsid, tokens.into_iter().collect::<RangeSetBlaze<_>>())
                }),
            );
            (canonical, weight)
        })
        .collect::<BTreeMap<_, _>>();

    let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let global_start = nwa.add_state();
    let seed_final = nwa.add_state();
    nwa.set_final_weight(seed_final, Weight::all());
    let canonical_state_offset = nwa.num_states();
    for _ in &canonical_nodes {
        nwa.add_state();
    }
    nwa.set_start_states(vec![global_start]);

    if !cross_lane_only && start_component_filter.is_none() {
        for (sequence, by_state) in seed_relations {
            debug_assert_eq!(sequence.len(), 1);
            let weight = relation_weight(by_state);
            if !weight.is_empty() {
                nwa.add_transition(global_start, sequence[0] as i32, seed_final, weight);
            }
        }
    }
    for (canonical, weight) in start_weights_by_canonical {
        nwa.add_epsilon(
            global_start,
            canonical_state_offset + canonical as u32,
            weight,
        );
    }
    for (canonical, node) in canonical_nodes.into_iter().enumerate() {
        let source = canonical_state_offset + canonical as u32;
        if node.accepting {
            nwa.set_final_weight(source, Weight::all());
        }
        for (terminal, target) in node.transitions {
            nwa.add_transition(
                source,
                terminal as i32,
                canonical_state_offset + target as u32,
                Weight::all(),
            );
        }
        for target in node.epsilons {
            nwa.add_epsilon(
                source,
                canonical_state_offset + target as u32,
                Weight::all(),
            );
        }
    }
    if compose_profile_enabled() {
        let mut support_tsids = BTreeSet::<u32>::new();
        let mut support_raw_states = BTreeSet::<u32>::new();
        let mut support_internal_tokens = BTreeSet::<u32>::new();
        let mut support_original_tokens = BTreeSet::<u32>::new();
        let start_node = &nwa.states()[global_start as usize];
        let start_weights = start_node
            .transitions
            .values()
            .flatten()
            .map(|(_, weight)| weight)
            .chain(start_node.epsilons.iter().map(|(_, weight)| weight));
        for weight in start_weights {
            for (range, tokens) in weight.raw_range_values() {
                for tsid in *range.start()..=*range.end() {
                    support_tsids.insert(tsid);
                    if let Some(raws) = id_map.tokenizer_states.internal_to_originals.get(tsid as usize) {
                        support_raw_states.extend(raws.iter().copied());
                    }
                }
                for token_range in tokens.ranges() {
                    for token in token_range {
                        support_internal_tokens.insert(token);
                        if let Some(originals) = id_map.vocab_tokens.internal_to_originals.get(token as usize) {
                            support_original_tokens.extend(originals.iter().copied());
                        }
                    }
                }
            }
        }
        eprintln!(
            "[glrmask/profile][constraint_boundary_start_support] tsids={} raw_states={} internal_tokens={} original_tokens={}",
            support_tsids.len(),
            support_raw_states.len(),
            support_internal_tokens.len(),
            support_original_tokens.len(),
        );
    }
    let raw_states = nwa.num_states();
    let raw_transitions = nwa.num_transitions();
    let canonical_state_count = raw_states.saturating_sub(canonical_state_offset);
    let build_ms = build_started_at.elapsed().as_secs_f64() * 1000.0;
    // Dynamic B consumes the compact acyclic terminal NWA directly at runtime.
    // Determinizing/minimizing it first only to convert the resulting DWA back
    // into an NWA during publication is pure work. Preserve the exact symbolic
    // source when possible. Control terminals require self-loops and therefore
    // make the source cyclic; those rare cases keep the established DWA path.
    let preserve_symbolic_nwa = preserve_symbolic_nwa && control_terminals.is_empty();
    let mut prefix_dwa_states = 0u32;
    let (terminal_automaton, determinize_ms, minimize_ms, final_states, final_transitions) =
        if preserve_symbolic_nwa {
            let final_states = nwa.num_states();
            let final_transitions = nwa.num_transitions();
            (
                TerminalAutomaton::EpsilonNwa(nwa),
                0.0,
                0.0,
                final_states,
                final_transitions,
            )
        } else {
            let started = Instant::now();
            let prefix_dwa_requested = std::env::var_os(
                "GLRMASK_EXPERIMENT_BOUNDARY_TERMINAL_PREFIX_DWA",
            )
            .is_some();
            let dwa = if prefix_dwa_requested {
                let candidate = deterministic_weighted_prefix_dwa_from_small_acyclic_nwa(
                    &nwa,
                    50_000,
                    4_096,
                    16,
                )
                .ok_or_else(|| {
                    "small boundary terminal prefix-DWA construction exceeded exact caps".to_string()
                })?;
                prefix_dwa_states = candidate.num_states();
                if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_TERMINAL_PREFIX_DWA").is_some() {
                    let reference = determinize(&nwa).map_err(|error| error.to_string())?;
                    let difference = find_difference(&candidate, &reference)
                        .map_err(|error| error.to_string())?;
                    assert!(
                        difference.is_none(),
                        "boundary terminal prefix DWA differs from generic determinization on {difference:?}",
                    );
                    eprintln!(
                        "[glrmask/validate][boundary_terminal_prefix_dwa] exact=true candidate_states={} reference_states={} candidate_transitions={} reference_transitions={}",
                        candidate.num_states(),
                        reference.num_states(),
                        candidate.num_transitions(),
                        reference.num_transitions(),
                    );
                }
                candidate
            } else {
                determinize(&nwa).map_err(|error| error.to_string())?
            };
            let determinize_ms = started.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            let mut dwa = if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TERMINAL_HASHCONS_ONLY")
                .is_some()
            {
                reverse_hashcons_owned(dwa)
            } else {
                minimize_owned(dwa)
            };
            let minimize_ms = started.elapsed().as_secs_f64() * 1000.0;
            for state in 0..dwa.num_states() {
                for &control in control_terminals {
                    dwa.add_transition(state, control as i32, state, Weight::all());
                }
            }
            let final_states = dwa.num_states();
            let final_transitions = dwa.num_transitions();
            (
                TerminalAutomaton::Dwa(dwa),
                determinize_ms,
                minimize_ms,
                final_states,
                final_transitions,
            )
        };

    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_direct_terminal] witnesses={} selected_tokens={} raw_lexer_states={} boundary_tsids={} canonical_states={} raw_states={} raw_transitions={} prefix_dwa_states={} final_states={} final_transitions={} controls={} delta_cross_lane_starts={} delta_cross_tokens={} delta_complex_lane_starts={} delta_single_lane_starts={} delta_start_groups={} one_byte_ms={one_byte_ms:.3} quotient_ms={quotient_ms:.3} build_ms={build_ms:.3} determinize_ms={determinize_ms:.3} minimize_ms={minimize_ms:.3} total_ms={:.3}",
            discovery.witnesses.len(),
            selected_original_tokens.len(),
            num_states,
            id_map.num_tsids(),
            canonical_state_count,
            raw_states,
            raw_transitions,
            prefix_dwa_states,
            final_states,
            final_transitions,
            control_terminals.len(),
            delta_cross_lane_starts,
            delta_cross_tokens.len(),
            delta_complex_lane_starts,
            delta_single_lane_starts,
            delta_start_groups,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(MappedArtifact::new(terminal_automaton, id_map))
}

pub(super) fn build_static_boundary_shard_work(
    merged_tokenizer_state_count: usize,
    component_state_map: &ManyToOneIdMap,
    vocab: &Vocab,
    discovery: &BoundaryTokenDiscovery,
    globally_erasable_ignore_terminals: &BitSet,
    control_terminals: &BTreeSet<u32>,
    terminal_offsets: &[u32],
    tokenizer_state_offsets: &[u32],
    plan: &ConcreteBoundaryDeltaPlan,
    candidate_tokens_by_component: &[Vec<u32>],
    static_boundary_components: Option<&BitSet>,
    parser_num_terminals: u32,
    templates: &Templates,
    parser_table_override: Option<Arc<crate::compiler::glr::table::GLRTable>>,
) -> Result<Vec<BoundaryShardWork>, String> {
    let build = |(component_index, tokens): (usize, &Vec<u32>)| {
            let artifact = direct_boundary_terminal_automaton(
                merged_tokenizer_state_count,
                Some(component_state_map),
                vocab,
                tokens,
                BTreeMap::new(),
                0.0,
                discovery,
                globally_erasable_ignore_terminals,
                control_terminals,
                terminal_offsets,
                tokenizer_state_offsets,
                Some(plan),
                Some(component_index),
                false,
            )?;
            let (terminal_automaton, id_map) = artifact.into_parts();
            Ok(BoundaryShardWork {
                start_component: component_index as u32,
                candidate_tokens: Arc::<[u32]>::from(tokens.clone()),
                parser: BoundaryParserWork::DeferredTerminalCount {
                    terminal_automaton,
                    id_map,
                    num_terminals: parser_num_terminals,
                    templates: templates.clone(),
                    prebuilt_bundle_cache: None,
                    parser_table_override: parser_table_override.clone(),
                },
            })
        };
    if macro_parallelism_disabled() {
        let mut timings = Vec::new();
        let result = candidate_tokens_by_component
            .iter()
            .enumerate()
            .filter(|(component_index, tokens)| {
                !tokens.is_empty()
                    && static_boundary_components
                        .is_none_or(|selected| selected.contains(*component_index))
            })
            .map(|item| {
                let started = Instant::now();
                let result = build(item);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect();
        report_macro_item_timings("compose_static_boundary_shards", &timings);
        result
    } else {
        candidate_tokens_by_component
            .par_iter()
            .enumerate()
            .filter(|(component_index, tokens)| {
                !tokens.is_empty()
                    && static_boundary_components
                        .is_none_or(|selected| selected.contains(*component_index))
            })
            .map(build)
            .collect()
    }
}

pub(super) fn add_control_loops_to_terminal_artifact(
    artifact: MappedArtifact<TerminalAutomaton>,
    control_terminals: &BTreeSet<u32>,
) -> MappedArtifact<TerminalAutomaton> {
    if control_terminals.is_empty() {
        return artifact;
    }
    let (automaton, id_map) = artifact.into_parts();
    let (mut nwa, had_epsilon) = match automaton {
        TerminalAutomaton::Dwa(dwa) => (dwa.to_nwa(), false),
        TerminalAutomaton::TokenDeterministicNwa(nwa) => (nwa, false),
        TerminalAutomaton::EpsilonNwa(nwa) => (nwa, true),
    };
    for state in 0..nwa.num_states() {
        for &control in control_terminals {
            nwa.add_transition(state, control as i32, state, Weight::all());
        }
    }
    MappedArtifact::new(
        if had_epsilon {
            TerminalAutomaton::EpsilonNwa(nwa)
        } else {
            TerminalAutomaton::TokenDeterministicNwa(nwa)
        },
        id_map,
    )
}

pub(super) fn add_boundary_special_token_paths(
    artifact: MappedArtifact<TerminalAutomaton>,
    special_token_terminals: &[SpecialTokenTerminal],
    raw_source_state: u32,
    fallback_state_map: Option<&ManyToOneIdMap>,
    control_terminals: &BTreeSet<u32>,
) -> Result<MappedArtifact<TerminalAutomaton>, String> {
    if special_token_terminals.is_empty() {
        return Ok(artifact);
    }

    let (automaton, mut id_map) = artifact.into_parts();
    let source_tsid = id_map
        .tokenizer_states
        .original_to_internal
        .get(raw_source_state as usize)
        .copied()
        .filter(|&tsid| tsid != u32::MAX)
        .or_else(|| {
            fallback_state_map
                .and_then(|map| map.original_to_internal.get(raw_source_state as usize))
                .copied()
                .filter(|&tsid| tsid != u32::MAX)
        })
        .ok_or_else(|| {
            format!(
                "boundary special-token source state {raw_source_state} has no tokenizer-state coordinate",
            )
        })?;

    let mut unique = special_token_terminals.to_vec();
    unique.sort_unstable_by_key(|special| (special.token_id, special.terminal_id));
    unique.dedup_by_key(|special| (special.token_id, special.terminal_id));

    let max_special_id = unique
        .iter()
        .map(|special| special.token_id)
        .max()
        .unwrap_or(0);
    if id_map.vocab_tokens.original_to_internal.len() <= max_special_id as usize {
        id_map
            .vocab_tokens
            .original_to_internal
            .resize(max_special_id as usize + 1, u32::MAX);
    }
    for special in &unique {
        let slot = &mut id_map.vocab_tokens.original_to_internal[special.token_id as usize];
        if *slot == u32::MAX {
            *slot = id_map.vocab_tokens.internal_to_originals.len() as u32;
            id_map
                .vocab_tokens
                .internal_to_originals
                .push(vec![special.token_id]);
            id_map
                .vocab_tokens
                .representative_original_ids
                .push(special.token_id);
        }
    }

    let mut nwa = match automaton {
        TerminalAutomaton::Dwa(dwa) => dwa.to_nwa(),
        TerminalAutomaton::TokenDeterministicNwa(nwa)
        | TerminalAutomaton::EpsilonNwa(nwa) => nwa,
    };
    let starts = nwa.start_states().to_vec();
    if starts.is_empty() {
        return Err("boundary terminal automaton has no start state".into());
    }
    let special_final = nwa.add_state();
    nwa.set_final_weight(special_final, Weight::all());
    for &control in control_terminals {
        nwa.add_transition(
            special_final,
            control as i32,
            special_final,
            Weight::all(),
        );
    }
    for special in unique {
        let internal_token = id_map
            .internal_token_for_original(special.token_id)
            .expect("special token was inserted into the boundary token coordinate");
        let weight = Weight::from_per_tsid_token_sets(std::iter::once((
            source_tsid,
            RangeSetBlaze::from_iter([internal_token]),
        )));
        for &start in &starts {
            nwa.add_transition(
                start,
                special.terminal_id as i32,
                special_final,
                weight.clone(),
            );
        }
    }

    // This may overlap a byte-token branch for an ID that intentionally has
    // both exact-special and byte semantics, so retain the general NWA form.
    Ok(MappedArtifact::new(
        TerminalAutomaton::EpsilonNwa(nwa),
        id_map,
    ))
}
