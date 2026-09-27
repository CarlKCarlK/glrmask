use super::*;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct RuntimeLexerProductReport {
    pub(super) attempted: bool,
    pub(super) selected: bool,
    pub(super) parser_overlap: bool,
    pub(super) source_states: usize,
    pub(super) product_states: usize,
    pub(super) source_transitions: usize,
    pub(super) product_transitions: usize,
    pub(super) multi_tsid_product_states: usize,
}

fn product_has_parser_relevant_multi_lane_overlap(
    constraint: &Constraint,
    candidate: &crate::automata::lexer::tokenizer::FullTokenizerDeterminization,
) -> bool {
    if constraint.table.advance.len() != constraint.table.num_states as usize {
        return false;
    }

    // A deterministic product is useful to the persistent mask frontier only
    // when at least two concrete source lexer lanes can be live under the same
    // parser-top admission row. Merely shrinking the static epsilon-NFA is not
    // enough: sequential subgrammars often have isomorphic lexer states but the
    // parser can enable only one copy at a time, in which case commit pays the
    // source-restoration cost for no frontier reduction.
    //
    // Ignore source states that themselves dispatch through epsilon edges;
    // those are structural closure nodes, not independent consuming lanes.
    for (product_state, subset) in candidate.source_subsets.iter().enumerate() {
        // The deterministic reset state represents one historical scanner lane
        // whose epsilon closure happens to contain every component reset. The
        // runtime product machinery deliberately maps it back to that one
        // exact source/reset lane on commit, so multi-component membership here
        // is not evidence of a persistent frontier split.
        if product_state == candidate.tokenizer.initial_state() as usize {
            continue;
        }
        let consuming_sources = subset
            .iter()
            .copied()
            .filter(|&source_state| {
                !constraint
                    .tokenizer
                    .state_has_epsilon_transitions(source_state)
            })
            .collect::<Vec<_>>();
        if consuming_sources.len() < 2 {
            continue;
        }

        for (parser_state, advance) in constraint.table.advance.iter().enumerate() {
            let mut live_lanes = Vec::<(u32, Vec<usize>)>::new();
            for &source_state in &consuming_sources {
                let terminals = constraint
                    .tokenizer
                    .possible_future_terminals(source_state)
                    .iter()
                    .filter(|&terminal| advance.contains(terminal))
                    .collect::<Vec<_>>();
                if !terminals.is_empty() {
                    live_lanes.push((source_state, terminals));
                }
            }
            if live_lanes.len() >= 2 {
                if std::env::var_os("GLRMASK_DEBUG_COMPOSE_RUNTIME_LEXER_PRODUCT").is_some() {
                    eprintln!(
                        "[glrmask/debug][compose_runtime_lexer_overlap] product_state={} parser_state={} subset={:?} live_lanes={:?}",
                        product_state,
                        parser_state,
                        subset,
                        live_lanes,
                    );
                }
                return true;
            }
        }
    }
    false
}

fn env_bool(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|value| {
        !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        )
    })
}

/// Install the existing exact runtime tokenizer product/fallback representation
/// on a composed constraint when it is clearly profitable.
///
/// This does not discard source provenance. Product states are an exact subset
/// construction over the composed epsilon-NFA; each product state retains its
/// source-state subset and the union of the already-compiled TSID classes for
/// those states. The historical source tokenizer is appended unchanged and is
/// used by commit whenever parser histories cease to be uniform across a
/// product subset. Thus this optimization changes only the *visible lexer
/// frontier* used by masking, not the recognized byte/token relation.
pub(super) fn maybe_install_runtime_lexer_product(
    constraint: &mut Constraint,
    terminal_aliases: usize,
    components_have_no_runtime_product: bool,
) -> RuntimeLexerProductReport {
    let source_states = constraint.tokenizer.num_states() as usize;
    let source_transitions = constraint.tokenizer.transition_count();
    let mut report = RuntimeLexerProductReport {
        source_states,
        source_transitions,
        ..RuntimeLexerProductReport::default()
    };

    if terminal_aliases == 0
        || !components_have_no_runtime_product
        || !constraint.tokenizer.has_epsilon_transitions()
    {
        return report;
    }

    let force = env_bool("GLRMASK_COMPOSE_RUNTIME_LEXER_PRODUCT");
    if force == Some(false) {
        return report;
    }
    let max_source_states = std::env::var("GLRMASK_COMPOSE_RUNTIME_LEXER_MAX_SOURCE_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(4_096);
    if force.is_none() && source_states > max_source_states {
        return report;
    }

    report.attempted = true;
    let state_limit = std::env::var("GLRMASK_COMPOSE_RUNTIME_LEXER_MAX_PRODUCT_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        // A product that is already larger than the source cannot reduce the
        // visible frontier, so abort the subset construction at that point.
        .unwrap_or(source_states.max(1));
    let transition_growth_percent = std::env::var(
        "GLRMASK_COMPOSE_RUNTIME_LEXER_MAX_TRANSITION_GROWTH_PERCENT",
    )
    .ok()
    .and_then(|value| value.trim().parse::<usize>().ok())
    .filter(|&value| value > 0)
    .unwrap_or(200);
    let transition_limit = source_transitions
        .saturating_mul(transition_growth_percent)
        / 100;
    let Some(candidate) = constraint
        .tokenizer
        .try_full_determinization(state_limit, transition_limit.max(1))
    else {
        return report;
    };

    report.product_states = candidate.source_subsets.len();
    report.product_transitions = candidate.tokenizer.transition_count();
    report.parser_overlap = product_has_parser_relevant_multi_lane_overlap(constraint, &candidate);
    let minimum_reduction_percent = std::env::var(
        "GLRMASK_COMPOSE_RUNTIME_LEXER_MIN_STATE_REDUCTION_PERCENT",
    )
    .ok()
    .and_then(|value| value.trim().parse::<usize>().ok())
    .filter(|&value| value <= 100)
    .unwrap_or(25);
    let reduction_is_large_enough = report
        .product_states
        .saturating_mul(100)
        <= source_states.saturating_mul(100usize.saturating_sub(minimum_reduction_percent));
    if force != Some(true) && !reduction_is_large_enough {
        return report;
    }
    if force != Some(true) && !report.parser_overlap {
        return report;
    }
    if report.product_states >= source_states && force != Some(true) {
        return report;
    }

    // Snapshot the exact TSID relation before moving the source tokenizer.
    // Composition normally has one TSID per raw state, but nested/runtime
    // artifacts may already expose a small relation; preserving the general
    // form makes the transform a true representation change rather than a
    // hidden singleton assumption.
    let source_state_tsids = (0..source_states)
        .map(|state| constraint.internal_tsids_for_state(state as u32).to_vec())
        .collect::<Vec<_>>();
    if source_state_tsids.iter().any(Vec::is_empty) {
        return report;
    }
    let internal_tsid_count = constraint.internal_tsid_count();
    if internal_tsid_count == 0 {
        return report;
    }
    if source_state_tsids
        .iter()
        .flatten()
        .any(|&tsid| tsid as usize >= internal_tsid_count)
    {
        return report;
    }

    let mut runtime_state_tsids = Vec::<Vec<u32>>::with_capacity(
        report.product_states.saturating_add(source_states),
    );
    for subset in &candidate.source_subsets {
        let mut tsids = subset
            .iter()
            .flat_map(|&source_state| {
                source_state_tsids
                    .get(source_state as usize)
                    .into_iter()
                    .flatten()
                    .copied()
            })
            .collect::<Vec<_>>();
        tsids.sort_unstable();
        tsids.dedup();
        if tsids.is_empty() {
            return report;
        }
        report.multi_tsid_product_states += usize::from(tsids.len() > 1);
        runtime_state_tsids.push(tsids);
    }
    runtime_state_tsids.extend(source_state_tsids.iter().cloned());

    let source_subsets = candidate.source_subsets.clone();
    let exact_source_states = candidate.exact_source_states.clone();
    let built = std::sync::Arc::make_mut(&mut constraint.tokenizer)
        .finish_full_determinization_with_source_fallback(candidate);
    debug_assert_eq!(built.source_state_offset as usize, report.product_states);
    debug_assert_eq!(built.tokenizer.num_states() as usize, runtime_state_tsids.len());

    let mut state_to_internal_tsid = Vec::with_capacity(runtime_state_tsids.len());
    let mut internal_tsid_to_states = vec![Vec::<u32>::new(); internal_tsid_count];
    for (runtime_state, tsids) in runtime_state_tsids.iter().enumerate() {
        let primary = tsids[0];
        state_to_internal_tsid.push(primary);
        for &tsid in tsids {
            // All TSIDs were range-checked before moving the source tokenizer;
            // no fallible return is permitted after that move.
            internal_tsid_to_states[tsid as usize].push(runtime_state as u32);
        }
    }

    let mut source_offsets = Vec::with_capacity(source_subsets.len() + 1);
    let mut source_states_flat = Vec::<u32>::new();
    source_offsets.push(0);
    for subset in &source_subsets {
        source_states_flat.extend_from_slice(subset);
        source_offsets.push(source_states_flat.len() as u32);
    }

    constraint.tokenizer = built.tokenizer.into();
    constraint.state_to_internal_tsid = state_to_internal_tsid;
    constraint.internal_tsid_to_states = internal_tsid_to_states;
    constraint.deferred_internal_tsid_to_states = Default::default();
    // Force finalization to rebuild the exact many-TSID relation for product
    // states rather than retaining the direct-union singleton sentinel.
    constraint.state_internal_tsid_offsets.clear();
    constraint.state_internal_tsids.clear();
    constraint.runtime_source_state_offset = Some(built.source_state_offset);
    constraint.runtime_product_source_offsets = source_offsets;
    constraint.runtime_product_source_states = source_states_flat;
    constraint.runtime_product_exact_source_states = exact_source_states;
    constraint.runtime_product_state_by_source_subset.clear();
    constraint.terminal_live_states.clear();
    constraint.tokenizer_fast_transitions = Default::default();
    report.selected = true;
    report
}
